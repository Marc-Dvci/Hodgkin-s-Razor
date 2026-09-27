"""Build the paired simulation bank the posterior is trained on.

    python scripts/make_bank.py --pairs 240000 --out data/bank_v2

Each record is one virtual experiment: a network simulated twice, before and
after a shift in one to three mechanisms, with the wiring and the recording
nuisances held fixed across the pair. Every recording is read through each
recording system in `simulator.VIEWS`, so one bank serves every twin.

A culture is admitted when its simulated baseline sits in the domain of one
recording system, judged on the recording alone:

* grid16 (Tampere, Axion 48-well) and grid12 (Doorn et al., MCS 24-well):
  inside the box that system's own untreated baselines span, widened, and a
  living network-driven culture (`models/domain.json`, written by
  `scripts/fit_domain.py`).

Both also require the property that defines a cortical preparation
pharmacologically: blocking AMPA collapses its activity below 35 percent.
Candidate cultures come from the proposal `scripts/fit_domain.py` fitted for
each domain, and every one is confirmed by simulation before it is used. No recording enters this script
except through the domain boxes, which are built from untreated baselines.

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
from fit_regime import CRITERION
from fit_domain import GaussianProposal, violation

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWS = list(S.VIEWS)
DOMAINS = ("grid16", "grid12")


def sister_drift(theta_t: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """The second sister of a preparation: every parameter nudged, none labelled.

    Sister arrays plated from one dissociation are the same preparation, not
    the same network. Their wiring and electrode pickup differ (the simulator's
    unpaired mode draws both afresh), and so do their culture parameters, by a
    small amount. `scale` is that spread as a fraction of each parameter's
    baseline range, in transformed units; `scripts/calibrate_drift.py` sets it
    from sister pairs recorded before any drug. The drift is never labelled as
    a mechanism, so the twin learns to see through it.
    """
    t = P.to_unit(theta_t)
    t = t + rng.normal(0.0, scale, size=t.shape) * (P.TBHI - P.TBLO)
    return P.from_unit(np.clip(t, P.TLO, P.THI))


def _unpack(packed) -> np.ndarray:
    """Events sent to a worker as int32 steps and uint8 electrodes, not floats."""
    t, e, dt_ms = packed
    return np.stack([e.astype(np.float64), t * (dt_ms / 1000.0)], axis=1)


def _pack(res, i: int) -> tuple:
    return res.times[i], res.elecs[i], res.dt_ms


def _all_views(args):
    packed, duration = args
    events = _unpack(packed)
    out = []
    for v in VIEWS:
        ev, n = S.view_events(events, v)
        out.append(F.compute(ev, n, duration))
    return np.stack(out)


def _one_view(args):
    """Features through one system's view; the other views are left NaN.

    Each twin is trained on the cultures of its own recording system, so
    reading a culture through a view it will never be trained on only costs
    time.
    """
    packed, view, duration = args
    out = np.full((len(VIEWS), F.N_FEATURE), np.nan)
    ev, n = S.view_events(_unpack(packed), view)
    out[VIEWS.index(view)] = F.compute(ev, n, duration)
    return out


def _rate(args):
    packed, view, duration = args
    ev, n = S.view_events(_unpack(packed), view)
    return ev.shape[0] / duration / n


class RegimeProposal:
    """Version 1 proposal: a classifier over parameters for a living culture."""

    def __init__(self, path: pathlib.Path, quantile: float):
        import joblib
        self.model = joblib.load(path.with_suffix(".joblib"))
        stats = json.loads(path.read_text())
        key = f"{quantile:.2f}"
        if key not in stats["quantiles"]:
            key = sorted(stats["quantiles"], key=lambda k: abs(float(k) - quantile))[0]
        self.threshold = stats["quantiles"][key]["threshold"]

    def draw(self, n: int, rng: np.random.Generator, pool: int = 24) -> np.ndarray:
        out = []
        while sum(len(o) for o in out) < n:
            cand = P.sample_prior(n * pool, rng)
            score = self.model.predict_proba(P.to_unit(cand))[:, 1]
            out.append(cand[score >= self.threshold])
        return np.concatenate(out)[:n]


def in_domain(x_views: np.ndarray, domain: str, boxes: dict) -> bool:
    x = x_views[VIEWS.index(domain)]
    return bool(violation(x, boxes[domain])[0] == 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=240000)
    ap.add_argument("--shard", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=192, help="pairs per GPU launch")
    ap.add_argument("--shifts-per-baseline", type=int, default=6)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", default="data/bank_v2")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--regime", default="models/regime.json")
    ap.add_argument("--regime-quantile", type=float, default=0.85)
    ap.add_argument("--domain-file", default="models/domain.json")
    ap.add_argument("--domains", default="grid16,grid12")
    ap.add_argument("--design", choices=("paired", "sister"), default="paired",
                    help="paired: one well recorded twice; sister: two arrays "
                         "plated from one preparation")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="sister design: culture drift, fraction of the baseline range")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dom = json.loads((ROOT / args.domain_file).read_text())
    boxes = {v: d["box"] for v, d in dom["views"].items() if "box" in d}
    domains = tuple(args.domains.split(","))
    proposals = {v: GaussianProposal.load(dom["views"][v]["proposal"]) for v in domains}
    meta = {"version": 3 if args.design == "sister" else 2,
            "design": args.design, "drift": args.drift, "pairs": args.pairs, "duration_s": args.duration,
            "transient_s": args.transient, "seed": args.seed,
            "views": {v: {"n_elec": S.n_electrodes(v), "dead_ms": S.VIEWS[v]["dead_ms"]}
                      for v in VIEWS},
            "domains": list(domains), "param_keys": list(P.KEYS),
            "shift_keys": [P.KEYS[i] for i in P.SHIFT_IDX],
            "feature_names": list(F.NAMES),
            "inactive_scale": SH.INACTIVE_SCALE,
            "min_active_fold": float(np.exp(SH.MIN_LOG_FOLD)),
            "shifts_per_baseline": args.shifts_per_baseline,
            "criterion": CRITERION, "boxes": boxes}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    sim = S.Simulator()
    n_shards = (args.pairs + args.shard - 1) // args.shard
    # Two pools: the probe features gate the next GPU launch, so they must not
    # queue behind the pair features, which only need to be ready by the end
    # of the shard.
    pool = cf.ProcessPoolExecutor(max_workers=6)
    pair_pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    t_start = time.time()
    done_pairs = 0
    screened = {d: 0 for d in domains}
    accepted = {d: 0 for d in domains}

    for s in range(n_shards):
        path = out / f"shard_{s:04d}.npz"
        if path.exists():
            done_pairs += args.shard
            continue
        rng = np.random.default_rng(args.seed + 1000 * s)
        n_here = min(args.shard, args.pairs - s * args.shard)
        TC, TT, DL, AC, GR, DM, PENDING = [], [], [], [], [], [], []
        have = 0
        step = 0
        culture = 0
        while have < n_here:
            # Alternate the domains so every shard holds both in equal measure.
            domain = domains[step % len(domains)]
            want = max(int(np.ceil((n_here - have) / args.shifts_per_baseline)), 1)
            want = min(want, max(args.batch // args.shifts_per_baseline, 8))
            cand = proposals[domain].draw(want * 3, rng)
            seed = int(args.seed + 7919 * (s * 10000 + step))
            probe = sim.run(cand, duration_s=args.duration,
                            transient_s=args.transient, seed=seed)
            xv = list(pool.map(_one_view, [(_pack(probe, k), domain, args.duration)
                                            for k in range(cand.shape[0])], chunksize=8))
            first = [k for k in range(cand.shape[0])
                     if not probe.truncated[k] and in_domain(xv[k], domain, boxes)]
            live = []
            if first:
                sel = np.array(first)
                blocked = cand[sel].copy()
                blocked[:, P.index("g_ampa")] = P.LO[P.index("g_ampa")]
                rb = sim.run(blocked, duration_s=args.duration,
                             transient_s=args.transient, seed=seed + 500003)
                after = np.array(list(pool.map(_rate, [(_pack(rb, k), domain, args.duration)
                                                       for k in range(len(sel))])))
                before = np.array([xv[k][VIEWS.index(domain)][F.NAMES.index("mfr")]
                                   for k in sel])
                ratio = after / np.maximum(before, 1e-9)
                live = [int(sel[i]) for i in range(len(sel))
                        if ratio[i] < CRITERION["ampa_block_max_ratio"]]
            screened[domain] += cand.shape[0]
            accepted[domain] += len(live)
            step += 1
            if not live:
                continue
            base = cand[live[:want]]

            theta_c = np.repeat(base, args.shifts_per_baseline, axis=0)
            groups = np.repeat(np.arange(culture, culture + base.shape[0]),
                               args.shifts_per_baseline)
            culture += base.shape[0]
            nb = theta_c.shape[0]
            delta, active = SH.sample_shift(nb, rng, theta_c=theta_c)
            theta_t, realised, active = SH.apply_shift(theta_c, delta, active)
            sister = args.design == "sister"
            if sister:
                theta_t = sister_drift(theta_t, args.drift, rng)
            stacked = SH.interleave(theta_c, theta_t)
            res = sim.run(stacked, duration_s=args.duration,
                          transient_s=args.transient, seed=seed + 13, pair=not sister)
            ok = ~(res.truncated[0::2] | res.truncated[1::2])
            # Features are computed while the GPU runs the next batch.
            keep = np.flatnonzero(np.repeat(ok, 2))
            PENDING.append(pair_pool.map(_one_view, [(_pack(res, int(i)), domain, args.duration)
                                                 for i in keep], chunksize=8))
            TC.append(P.to_unit(theta_c)[ok])
            TT.append(P.to_unit(theta_t)[ok])
            DL.append(realised[ok])
            AC.append(active[ok])
            GR.append(groups[ok])
            DM.append(np.full(int(ok.sum()), VIEWS.index(domain), dtype=np.int8))
            have += int(ok.sum())

        fx = np.concatenate([np.stack(list(f)) for f in PENDING])
        XB, XT = [fx[0::2]], [fx[1::2]]
        np.savez_compressed(
            path,
            theta_c=np.concatenate(TC)[:n_here].astype(np.float32),
            # The second recording's own parameters: the shift plus, in the
            # sister design, the drift between sisters.
            theta_t=np.concatenate(TT)[:n_here].astype(np.float32),
            delta=np.concatenate(DL)[:n_here].astype(np.float32),
            active=np.concatenate(AC)[:n_here],
            x_base=np.concatenate(XB)[:n_here].astype(np.float32),
            x_treat=np.concatenate(XT)[:n_here].astype(np.float32),
            group=np.concatenate(GR)[:n_here].astype(np.int32),
            domain=np.concatenate(DM)[:n_here])
        done_pairs += n_here
        rate = done_pairs / max(time.time() - t_start, 1e-9)
        acc = {d: round(accepted[d] / max(screened[d], 1), 3) for d in domains}
        print(f"shard {s + 1}/{n_shards}  pairs {done_pairs}  {rate:.1f} pairs/s  "
              f"acceptance {acc}  "
              f"eta {(args.pairs - done_pairs) / max(rate, 1e-9) / 60:.0f} min",
              flush=True)
    pool.shutdown()
    pair_pool.shutdown()
    print("bank complete", out)


if __name__ == "__main__":
    main()
