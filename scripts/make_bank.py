"""Build the paired simulation bank the posterior is trained on.

    python scripts/make_bank.py --pairs 80000 --out data/bank

Each record is one virtual experiment: a network simulated twice, before and
after a shift in one or two mechanisms, with the wiring and the recording
nuisances held fixed across the pair.

Only about one parameter set in ten produces a living, network-driven culture.
In the rest, activity comes from membrane noise rather than from the network's
own synapses, so blocking a synaptic conductance changes nothing and no method
could say what a compound did. The bank is therefore built on parameter sets
that pass the criterion in `scripts/fit_regime.py`, proposed by the classifier
that script fits and then confirmed by simulation. The same criterion is
reported for the recorded data, so the twin's domain of validity is stated
rather than assumed.

Shards are written as they finish, so a run can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P, shift as SH, simulator as S
from fit_regime import CRITERION, ampa_dependent, is_living

N_ELEC = S.NELEC
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _features(args):
    events, duration = args
    return F.compute(events, N_ELEC, duration)


class Proposal:
    """Parameter sets likely to produce a living culture."""

    def __init__(self, path: pathlib.Path, quantile: float):
        import joblib
        self.model = joblib.load(path.with_suffix(".joblib"))
        stats = json.loads(path.read_text())
        key = f"{quantile:.2f}"
        if key not in stats["quantiles"]:
            key = sorted(stats["quantiles"], key=lambda k: abs(float(k) - quantile))[0]
        self.threshold = stats["quantiles"][key]["threshold"]
        self.stats = stats

    def draw(self, n: int, rng: np.random.Generator, pool: int = 24) -> np.ndarray:
        out = []
        while sum(len(o) for o in out) < n:
            cand = P.sample_prior(n * pool, rng)
            score = self.model.predict_proba(P.to_unit(cand))[:, 1]
            out.append(cand[score >= self.threshold])
        return np.concatenate(out)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=80000)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=192, help="pairs per GPU launch")
    ap.add_argument("--shifts-per-baseline", type=int, default=5)
    ap.add_argument("--probe-duration", type=float, default=30.0,
                    help="window for the screening simulation")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", default="data/bank")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--regime", default="models/regime.json")
    ap.add_argument("--regime-quantile", type=float, default=0.85)
    ap.add_argument("--no-regime", action="store_true",
                    help="sample straight from the prior, for an ablation")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proposal = None if args.no_regime else Proposal(ROOT / args.regime,
                                                    args.regime_quantile)
    meta = {"pairs": args.pairs, "duration_s": args.duration,
            "transient_s": args.transient, "seed": args.seed,
            "n_elec": N_ELEC, "param_keys": list(P.KEYS),
            "shift_keys": [P.KEYS[i] for i in P.SHIFT_IDX],
            "feature_names": list(F.NAMES),
            "inactive_scale": SH.INACTIVE_SCALE,
            "min_active_fold": float(np.exp(SH.MIN_LOG_FOLD)),
            "shifts_per_baseline": args.shifts_per_baseline,
            "regime_restricted": not args.no_regime,
            "regime_criterion": CRITERION}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    sim = S.Simulator()
    n_shards = (args.pairs + args.shard - 1) // args.shard
    pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    t_start = time.time()
    done_pairs = 0
    screened = accepted = 0

    for s in range(n_shards):
        path = out / f"shard_{s:04d}.npz"
        if path.exists():
            done_pairs += args.shard
            continue
        rng = np.random.default_rng(args.seed + 1000 * s)
        n_here = min(args.shard, args.pairs - s * args.shard)
        TC, DL, AC, XB, XT = [], [], [], [], []
        have = 0
        step = 0
        while have < n_here:
            # Propose baselines, then confirm by simulation that they live.
            want = max(int(np.ceil((n_here - have) / args.shifts_per_baseline)), 1)
            want = min(want, max(args.batch // args.shifts_per_baseline, 8))
            cand = (proposal.draw(want * 3, rng) if proposal
                    else P.sample_prior(want * 3, rng))
            seed = int(args.seed + 7919 * (s * 10000 + step))
            probe = sim.run(cand, duration_s=args.probe_duration,
                            transient_s=args.transient, seed=seed)
            stats = [F.regime_stats(probe.as_events(k), N_ELEC,
                                    args.probe_duration)
                     for k in range(cand.shape[0])]
            ratio = ampa_dependent(sim, cand, args.probe_duration,
                                   args.transient, seed + 500003,
                                   np.array([s_["mfr"] for s_ in stats]))
            live = [k for k in range(cand.shape[0])
                    if is_living(stats[k])
                    and ratio[k] < CRITERION["ampa_block_max_ratio"]]
            screened += cand.shape[0]
            accepted += len(live)
            if not live:
                step += 1
                continue
            base = cand[live[:want]]

            theta_c = np.repeat(base, args.shifts_per_baseline, axis=0)
            nb = theta_c.shape[0]
            delta, active = SH.sample_shift(nb, rng)
            theta_t, realised = SH.apply_shift(theta_c, delta)
            stacked = SH.interleave(theta_c, theta_t)
            res = sim.run(stacked, duration_s=args.duration,
                          transient_s=args.transient, seed=seed + 13, pair=True)
            jobs = [(res.as_events(i), args.duration) for i in range(2 * nb)]
            fx = np.stack(list(pool.map(_features, jobs, chunksize=8)))
            TC.append(P.to_unit(theta_c))
            DL.append(realised)
            AC.append(active)
            XB.append(fx[0::2])
            XT.append(fx[1::2])
            have += nb
            step += 1

        np.savez_compressed(
            path,
            theta_c=np.concatenate(TC)[:n_here].astype(np.float32),
            delta=np.concatenate(DL)[:n_here].astype(np.float32),
            active=np.concatenate(AC)[:n_here],
            x_base=np.concatenate(XB)[:n_here].astype(np.float32),
            x_treat=np.concatenate(XT)[:n_here].astype(np.float32))
        done_pairs += n_here
        rate = done_pairs / max(time.time() - t_start, 1e-9)
        print(f"shard {s + 1}/{n_shards}  pairs {done_pairs}  {rate:.1f} pairs/s  "
              f"acceptance {accepted / max(screened, 1):.2f}  "
              f"eta {(args.pairs - done_pairs) / max(rate, 1e-9) / 60:.0f} min",
              flush=True)
    pool.shutdown()
    print("bank complete", out)


if __name__ == "__main__":
    main()
