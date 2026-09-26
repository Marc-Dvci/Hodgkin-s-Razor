"""Screen the prior and ask which recorded baselines the simulator can reach.

    python scripts/screen_domain.py --n 6000

Simulates untreated networks drawn from the baseline prior, reads each one
through every electrode layout in `simulator.VIEWS`, and stores the parameters
and features. `scripts/fit_domain.py` uses the result to fit the proposal for
the bank. Only simulations are produced here; no recording is read.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P, simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _views(args):
    events, duration = args
    out = []
    for v in S.VIEWS:
        ev, n = S.view_events(events, v)  # events are raw, at kernel resolution
        out.append(F.compute(ev, n, duration))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--out", default="data/screen_domain.npz")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    TH, X, AR = [], [], []
    t0 = time.time()
    for a in range(0, args.n, args.batch):
        nb = min(args.batch, args.n - a)
        th = P.sample_prior(nb, rng)
        r = sim.run(th, duration_s=args.duration, transient_s=args.transient,
                    seed=args.seed + a)
        x = np.stack(list(pool.map(_views, [(r.raw_events(k), args.duration)
                                            for k in range(nb)], chunksize=8)))
        # The AMPA-block response, which defines a cortical preparation.
        blocked = th.copy()
        blocked[:, P.index("g_ampa")] = P.LO[P.index("g_ampa")]
        rb = sim.run(blocked, duration_s=args.duration,
                     transient_s=args.transient, seed=args.seed + a + 777777)
        after = np.array([rb.as_events(k).shape[0] for k in range(nb)])
        before = np.array([r.as_events(k).shape[0] for k in range(nb)])
        TH.append(th)
        X.append(x)
        AR.append(after / np.maximum(before, 1))
        print(f"  {a + nb}/{args.n}  {time.time() - t0:.0f}s", flush=True)
    pool.shutdown()
    out = ROOT / args.out
    np.savez_compressed(out, theta=np.concatenate(TH), x=np.concatenate(X),
                        ampa_ratio=np.concatenate(AR),
                        views=np.array(list(S.VIEWS)),
                        duration=args.duration)
    print("wrote", out)


if __name__ == "__main__":
    main()
