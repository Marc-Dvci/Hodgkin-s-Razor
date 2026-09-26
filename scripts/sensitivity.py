"""Measure what each parameter actually does to the recording.

    python scripts/sensitivity.py --out results/sensitivity.json

A parameter whose prior range sits in a flat region of the output cannot be
recovered and cannot represent a compound that acts on it. This sweep is how
the prior ranges were set, and it is re-run whenever the model changes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P, simulator as S

WATCH = ("mfr", "nbr", "nbd", "psib", "mean_cc", "isi_cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=9)
    ap.add_argument("--backgrounds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/sensitivity.json")
    args = ap.parse_args()

    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    backgrounds = P.sample_prior(args.backgrounds, rng)

    out: dict = {"levels": args.levels, "backgrounds": args.backgrounds,
                 "duration_s": args.duration, "parameters": {}}
    for j, p in enumerate(P.PARAMS):
        if p.log:
            grid = np.exp(np.linspace(np.log(p.lo), np.log(p.hi), args.levels))
        else:
            grid = np.linspace(p.lo, p.hi, args.levels)
        theta = np.repeat(backgrounds, args.levels, axis=0)
        theta[:, j] = np.tile(grid, args.backgrounds)
        res = sim.run(theta, duration_s=args.duration,
                      transient_s=args.transient, seed=int(100 + j))
        feats = np.stack([F.compute(res.as_events(i), S.NELEC, args.duration)
                          for i in range(theta.shape[0])])
        feats = feats.reshape(args.backgrounds, args.levels, F.N_FEATURE)

        rec: dict = {"grid": grid.tolist(), "log": bool(p.log), "watch": {}}
        for name in WATCH:
            k = F.NAMES.index(name)
            curves = feats[:, :, k]
            # Monotone trend across the grid, averaged over backgrounds, and the
            # fraction of the range where the output actually moves.
            med = np.median(curves, axis=0)
            spread = float(np.ptp(med))
            scale = float(np.median(np.abs(med)) + 1e-9)
            steps = np.abs(np.diff(med))
            live = float(np.mean(steps > 0.05 * max(spread, 1e-9)))
            rho = _spearman(np.arange(args.levels), med)
            rec["watch"][name] = {"median_curve": [round(float(v), 4) for v in med],
                                  "range": spread, "relative_range": spread / scale,
                                  "live_fraction": live, "spearman": rho}
        rec["verdict"] = _verdict(rec)
        out["parameters"][p.key] = rec
        print(f"{p.key:10s} mfr range {rec['watch']['mfr']['range']:8.2f} "
              f"live {rec['watch']['mfr']['live_fraction']:.2f} "
              f"rho {rec['watch']['mfr']['spearman']:+.2f}   {rec['verdict']}",
              flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _verdict(rec: dict) -> str:
    best = max(abs(v["spearman"]) for v in rec["watch"].values())
    live = max(v["live_fraction"] for v in rec["watch"].values())
    if best < 0.4 and live < 0.4:
        return "FLAT: prior range does not move the recording"
    if live < 0.5:
        return "partly flat"
    return "responsive"


if __name__ == "__main__":
    main()
