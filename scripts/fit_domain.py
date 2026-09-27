"""Find where in parameter space the simulator produces each recorded domain.

    python scripts/fit_domain.py

A recording system's domain is the range its untreated recordings span, read
from the baseline recordings alone: no treated recording and no compound label
enters it. For each recording system this script

1. writes that range as a box over statistics that do not depend on the
   detection details of one amplifier (rate, bursting, synchrony), and
2. searches the baseline prior for parameter sets whose simulated recording,
   read through that system, falls inside the box, by a sequential
   population search: simulate, keep the closest fifth, refit a Gaussian
   proposal around them, repeat.

The result is a proposal the bank generator draws cultures from. Version 1
used hand-set caps for the Tampere domain (`fit_regime.BASELINE_BOX`); version
2 builds every domain by the same rule from the baselines of that system.
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

from hodgkins_razor import features as F, params as P, simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBUST = ("mfr", "nbr", "nbd", "psib", "mean_cc", "sttc_mean",
          "burst_participation", "tonic_rate")
# Statistics bounded in [0, 1] are widened additively; rates and durations
# multiplicatively; percentages in points.
BOUNDED = ("mean_cc", "sttc_mean", "burst_participation")
PERCENT = ("psib",)


def box_from_baselines(x: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> dict:
    """Range of the recorded baselines, widened so the box is not a fit."""
    box = {}
    for k in ROBUST:
        c = x[:, F.NAMES.index(k)]
        lo, hi = float(np.percentile(c, lo_q)), float(np.percentile(c, hi_q))
        if k in BOUNDED:
            lo, hi = max(lo - 0.1, -1.0), min(hi + 0.1, 1.0)
        elif k in PERCENT:
            lo, hi = max(lo - 15.0, 0.0), min(hi + 15.0, 100.0)
        elif k == "nbr":
            lo, hi = max(lo - 1.0, 0.0), hi + 3.0
        else:
            lo, hi = lo / 1.5, hi * 1.5
        box[k] = [lo, hi]
    return box


def violation(x: np.ndarray, box: dict) -> np.ndarray:
    """Summed distance outside the box, each statistic in units of its width."""
    x = np.atleast_2d(x)
    v = np.zeros(x.shape[0])
    for k, (lo, hi) in box.items():
        c = x[:, F.NAMES.index(k)]
        w = max(min(hi, 1e3) - lo, 1e-6)
        v += np.clip(lo - c, 0, None) / w + np.clip(c - hi, 0, None) / w
    v[~np.all(np.isfinite(x), axis=1)] = np.inf
    return v


def with_living(box: dict) -> dict:
    """Add the version 1 criterion for a living, network-driven culture.

    A culture whose activity is not driven by its own synapses cannot show
    what a synaptic compound did, whatever system records it.
    """
    from fit_regime import CRITERION
    out = {k: list(v) for k, v in box.items()}
    out["mfr"][0] = max(out["mfr"][0], CRITERION["mfr_min"])
    out["nbr"][0] = max(out["nbr"][0], CRITERION["nbr_min"])
    out["psib"][0] = max(out["psib"][0], CRITERION["psib_min"])
    out["active_frac"] = [CRITERION["active_frac_min"], 1.0]
    return out


class Closeness:
    """Distance from a simulated recording to the recorded baselines.

    The mean distance to the three nearest recorded baseline windows, over the
    detection-robust statistics, each scaled by its spread across the
    recordings. Searching on it makes the proposal follow where the recorded
    cultures actually sit inside the box, not only the box itself.
    """

    def __init__(self, real: np.ndarray):
        self.idx = [F.NAMES.index(k) for k in ROBUST]
        r = self._t(real)
        self.mu, self.sd = r.mean(0), r.std(0) + 0.05
        self.r = (r - self.mu) / self.sd

    def _t(self, x):
        x = np.atleast_2d(x)[:, self.idx]
        return np.sign(x) * np.log1p(np.abs(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (self._t(x) - self.mu) / self.sd
        z = np.nan_to_num(z, nan=1e3, posinf=1e3, neginf=-1e3)
        d = np.sqrt(((z[:, None, :] - self.r[None, :, :]) ** 2).sum(-1))
        return np.sort(d, axis=1)[:, :3].mean(1)

    def within(self) -> float:
        """Typical distance of a recorded window to its recorded neighbours."""
        d = np.sqrt(((self.r[:, None, :] - self.r[None, :, :]) ** 2).sum(-1))
        return float(np.median(np.sort(d, axis=1)[:, 1:4].mean(1)))


def recorded_baselines(view: str, species: str = "") -> np.ndarray:
    """Feature vectors of the untreated recordings a system produced.

    `species` restricts the Axion (grid16) baselines to one species, so human
    cultures, which the joint domain leaves outside the model, get their own.
    """
    if view == "grid16":
        from hodgkins_razor import tampere as T
        return np.array([F.compute(p.baseline, p.n_elec, p.duration)
                         for p in T.load_all() if not species or p.species == species])
    if view == "mcs60q":
        # The first array of every scored sister pair, which never carries a
        # drug, read as quadrants.
        from hodgkins_razor import charlesworth as C
        return np.array([F.compute(ev, C.N_ELEC, 60.0) for ev in C.baselines(n_windows=2)])
    from hodgkins_razor import doorn as D
    return np.array([F.compute(p.baseline, p.n_elec, p.duration)
                     for p in D.load()])


def _feat(args):
    events, view, duration = args
    ev, n = S.view_events(events, view)
    return F.compute(ev, n, duration)


def to_cube(theta: np.ndarray) -> np.ndarray:
    return (P.to_unit(theta) - P.TBLO) / (P.TBHI - P.TBLO)


def from_cube(u: np.ndarray) -> np.ndarray:
    return P.from_unit(P.TBLO + np.clip(u, 0.0, 1.0) * (P.TBHI - P.TBLO))


class GaussianProposal:
    """Mixture of Gaussians centred on elite parameter sets, in the unit cube."""

    def __init__(self, centres: np.ndarray, cov: np.ndarray, uniform: float = 0.1):
        self.centres = np.asarray(centres)
        self.cov = np.asarray(cov)
        self.uniform = uniform

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n_u = int(round(self.uniform * n))
        pick = self.centres[rng.integers(0, len(self.centres), n - n_u)]
        z = rng.multivariate_normal(np.zeros(P.N_PARAM), self.cov, size=n - n_u)
        u = np.concatenate([np.clip(pick + z, 0.0, 1.0),
                            rng.uniform(size=(n_u, P.N_PARAM))])
        return from_cube(rng.permutation(u))

    def state(self) -> dict:
        return {"centres": self.centres.tolist(), "cov": self.cov.tolist(),
                "uniform": self.uniform}

    @classmethod
    def load(cls, d: dict) -> "GaussianProposal":
        return cls(np.array(d["centres"]), np.array(d["cov"]), d["uniform"])


def search(sim, view: str, box: dict, rounds: int, pop: int, seed: int,
           pool, duration: float, transient: float,
           close: "Closeness | None" = None) -> tuple[GaussianProposal, dict]:
    rng = np.random.default_rng(seed)
    prop = None
    hist = []
    all_u, all_v = [], []
    for r in range(rounds):
        th = prop.draw(pop, rng) if prop else P.sample_prior(pop, rng)
        res = sim.run(th, duration_s=duration, transient_s=transient, seed=seed + 1000 * r)
        x = np.stack(list(pool.map(_feat, [(res.raw_events(k), view, duration)
                                           for k in range(pop)], chunksize=8)))
        v = violation(x, box)
        v[res.truncated] = np.inf
        if close is not None:
            # Inside the box, rank by closeness to the recorded baselines.
            v = np.where(v == 0, 0.0, v + 10.0) + 0.1 * close(x)
        u = to_cube(th)
        all_u.append(u)
        all_v.append(v)
        U, V = np.concatenate(all_u), np.concatenate(all_v)
        elite = U[np.argsort(V)[:max(pop // 5, 32)]]
        cov = np.cov(elite.T) * 1.5 + np.eye(P.N_PARAM) * 1e-4
        prop = GaussianProposal(elite, cov * 0.25)
        inside = float(np.mean(violation(x, box) == 0))
        hist.append({"round": r, "inside": inside,
                     "median_violation": float(np.median(v[np.isfinite(v)]))})
        if close is not None:
            hist[-1]["median_closeness"] = float(np.median(close(x)))
        print(f"  {view} round {r}: inside {inside:.3f}, "
              f"median violation {hist[-1]['median_violation']:.2f}", flush=True)
    U, V = np.concatenate(all_u), np.concatenate(all_v)
    good = U[V < 10.0] if close is not None else U[V == 0]
    if close is not None and len(good) > 64:
        # Keep the inside-box cultures closest to the recorded ones.
        order = np.argsort(V[V < 10.0])
        good = good[order[:max(len(good) // 2, 64)]]
    centres = good if len(good) >= 32 else U[np.argsort(V)[:64]]
    cov = np.cov(centres.T) * 0.25 + np.eye(P.N_PARAM) * 1e-4
    return GaussianProposal(centres, cov, uniform=0.05), {"history": hist,
                                                          "n_inside": int(len(good))}


def refine_by_neighbours(sim, view: str, box: dict, prop: "GaussianProposal",
                         real: np.ndarray, n: int, k: int, seed: int, pool,
                         duration: float, transient: float) -> tuple:
    """Centre the proposal on admissible cultures that resemble recorded ones.

    Screens `n` cultures from a mixture of the searched proposal and the plain
    prior, keeps those a bank would admit (inside the box and AMPA-dependent),
    and takes, for every recorded baseline window, its `k` nearest admissible
    cultures over the detection-robust statistics. The union becomes the
    proposal's centres, so the bank follows the recorded distribution rather
    than whatever corner of the box a search happened to reach.
    """
    from fit_regime import CRITERION
    rng = np.random.default_rng(seed)
    mix = np.concatenate([prop.draw(n // 2, rng), P.sample_prior(n - n // 2, rng)])
    xs, ok = [], []
    for a in range(0, n, 768):
        th = mix[a:a + 768]
        res = sim.run(th, duration_s=duration, transient_s=transient, seed=seed + a)
        x = np.stack(list(pool.map(_feat, [(res.raw_events(j), view, duration)
                                           for j in range(th.shape[0])], chunksize=8)))
        inside = (violation(x, box) == 0) & ~res.truncated
        blocked = th.copy()
        blocked[:, P.index("g_ampa")] = P.LO[P.index("g_ampa")]
        rb = sim.run(blocked, duration_s=duration, transient_s=transient,
                     seed=seed + a + 99991)
        after = np.array([S.view_events(rb.raw_events(j), view)[0].shape[0]
                          for j in range(th.shape[0])]) / duration / S.n_electrodes(view)
        dep = after / np.maximum(x[:, F.NAMES.index("mfr")], 1e-9) < CRITERION["ampa_block_max_ratio"]
        xs.append(x)
        ok.append(inside & dep)
    X, OK = np.concatenate(xs), np.concatenate(ok)
    adm_x, adm_t = X[OK], mix[OK]
    close = Closeness(real)
    z_adm = (close._t(adm_x) - close.mu) / close.sd
    pick = set()
    for r in close.r:
        d = np.sqrt(((z_adm - r[None, :]) ** 2).sum(1))
        pick.update(np.argsort(d)[:k].tolist())
    centres = to_cube(adm_t[sorted(pick)])
    cov = np.cov(centres.T) * 0.05 + np.eye(P.N_PARAM) * 1e-4
    info = {"screened": int(n), "admissible": int(OK.sum()),
            "centres": int(len(centres)),
            "admissible_rate": float(OK.mean())}
    return GaussianProposal(centres, cov, uniform=0.0), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pop", type=int, default=1536)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=2718)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--views", default="grid12")
    ap.add_argument("--neighbours", type=int, default=0,
                    help="centre the proposal on the k admissible cultures "
                         "nearest each recorded baseline window")
    ap.add_argument("--screen", type=int, default=12288)
    ap.add_argument("--match", action="store_true",
                    help="inside the box, follow where the recorded baselines sit")
    ap.add_argument("--species", default="",
                    help="grid16 only: fit on one species' baselines (rat or hPSC)")
    ap.add_argument("--out", default="models/domain.json")
    args = ap.parse_args()

    path = ROOT / args.out
    out: dict = (json.loads(path.read_text()) if path.exists()
                 else {"robust": list(ROBUST), "views": {}})
    sim = S.Simulator()
    pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    t0 = time.time()
    for view in args.views.split(","):
        x = recorded_baselines(view, args.species)
        # One rule for every recording system: the range its own untreated
        # baselines span, widened, and a living network-driven culture.
        box = with_living(box_from_baselines(x))
        inside_real = float(np.mean(violation(x, box) == 0))
        print(f"{view}: box from {len(x)} baseline windows, "
              f"{inside_real:.2f} of them inside", flush=True)
        close = Closeness(x) if args.match else None
        prop, info = search(sim, view, box, args.rounds, args.pop, args.seed,
                            pool, args.duration, args.transient, close)
        if close is not None:
            info["recorded_within"] = close.within()
        if args.neighbours:
            prop, ninfo = refine_by_neighbours(sim, view, box, prop, x, args.screen,
                                               args.neighbours, args.seed + 7,
                                               pool, args.duration, args.transient)
            info["neighbours"] = ninfo
            print(f"  {view}: {ninfo}", flush=True)
        out["views"][view] = {"box": box, "n_baseline_windows": int(len(x)),
                              "baselines_inside": inside_real,
                              "proposal": prop.state(), **info}
        print(f"  {view}: {info['n_inside']} simulated cultures inside "
              f"({time.time() - t0:.0f}s)", flush=True)
    pool.shutdown()
    (ROOT / args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
