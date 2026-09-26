"""Does the twin reproduce pharmacology every MEA laboratory knows?

    python scripts/pharmacology_check.py

Applies a saturating block or agonist at each mechanism to living simulated
cultures and compares the result against the published direction and rough
magnitude for the matching compound. These expectations come from the
neuropharmacology literature, not from any dataset scored in this project, so
the check is independent of the evaluation.

A model that fails here cannot be trusted to attribute a compound to the
mechanism that failed, whatever its accuracy on a scored set.
"""
from __future__ import annotations

import argparse
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
from fit_regime import is_living

ROOT = pathlib.Path(__file__).resolve().parents[1]

# fold: what a saturating concentration does to the parameter.
# rate_lo / rate_hi: the firing rate after, as a fraction of before.
EXPECTATIONS = (
    {"drug": "TTX", "key": "g_na", "fold": 0.10,
     "rate_lo": 0.0, "rate_hi": 0.10, "bursts": "abolished",
     "note": "a sodium channel block silences the culture"},
    {"drug": "CNQX or NBQX", "key": "g_ampa", "fold": 0.08,
     "rate_lo": 0.0, "rate_hi": 0.35, "bursts": "abolished",
     "note": "AMPA carries fast excitatory transmission; blocking it "
             "abolishes network bursts in cortical culture"},
    {"drug": "D-AP5", "key": "g_nmda", "fold": 0.08,
     "rate_lo": 0.15, "rate_hi": 0.90, "bursts": "reduced",
     "note": "an NMDA block shortens bursts and reduces but does not "
             "abolish activity"},
    {"drug": "gabazine or picrotoxin", "key": "g_gaba", "fold": 0.08,
     "rate_lo": 1.05, "rate_hi": 100.0, "bursts": "increased",
     "note": "removing inhibition raises firing and synchrony"},
    {"drug": "GABA or muscimol", "key": "g_gaba", "fold": 8.0,
     "rate_lo": 0.0, "rate_hi": 0.60, "bursts": "reduced",
     "note": "raising inhibitory conductance suppresses firing"},
    {"drug": "4-aminopyridine", "key": "g_kdr", "fold": 0.25,
     "rate_lo": 1.0, "rate_hi": 100.0, "bursts": "increased",
     "note": "a potassium channel block raises excitability"},
)


def living_baselines(sim, n: int, rng, duration: float, transient: float,
                     batch: int = 384) -> np.ndarray:
    keep: list[np.ndarray] = []
    tries = 0
    while len(keep) < n and tries < 40:
        th = P.sample_prior(batch, rng)
        r = sim.run(th, duration_s=duration, transient_s=transient,
                    seed=5000 + tries)
        for k in range(batch):
            if is_living(F.regime_stats(r.as_events(k), S.NELEC, duration)):
                keep.append(th[k])
        tries += 1
    return np.array(keep[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", type=int, default=120)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="results/pharmacology.json")
    args = ap.parse_args()

    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    base = living_baselines(sim, args.baselines, rng, args.duration, args.transient)
    print(f"{len(base)} living baselines in {time.time() - t0:.0f}s")

    rows, pairs = [], []
    for exp in EXPECTATIONS:
        j = P.index(exp["key"])
        for th in base:
            t = th.copy()
            t[j] = float(np.clip(th[j] * exp["fold"], P.LO[j], P.HI[j]))
            pairs.append(th)
            pairs.append(t)
        rows.extend([exp] * len(base))

    feats = []
    arr = np.array(pairs)
    for a in range(0, arr.shape[0], 384):
        chunk = arr[a:a + 384]
        if chunk.shape[0] % 2:
            chunk = arr[a:a + 383]
        res = sim.run(chunk, duration_s=args.duration,
                      transient_s=args.transient, seed=9000 + a, pair=True)
        feats.extend(F.compute(res.as_events(i), S.NELEC, args.duration)
                     for i in range(chunk.shape[0]))
    feats = np.array(feats)
    fb, ft = feats[0::2], feats[1::2]
    mfr, nbr = F.NAMES.index("mfr"), F.NAMES.index("nbr")

    out = {"baselines": int(len(base)), "duration_s": args.duration,
           "checks": []}
    print()
    print(f"{'drug':24s} {'parameter':9s} {'rate ratio':>22s} {'expected':>14s}  verdict")
    n = len(base)
    for i, exp in enumerate(EXPECTATIONS):
        s = slice(i * n, (i + 1) * n)
        ratio = ft[s, mfr] / np.maximum(fb[s, mfr], 1e-9)
        burst = ft[s, nbr] / np.maximum(fb[s, nbr], 1e-9)
        med = float(np.median(ratio))
        ok = exp["rate_lo"] <= med <= exp["rate_hi"]
        out["checks"].append({
            "drug": exp["drug"], "key": exp["key"], "fold": exp["fold"],
            "rate_ratio_p25": float(np.percentile(ratio, 25)),
            "rate_ratio_median": med,
            "rate_ratio_p75": float(np.percentile(ratio, 75)),
            "burst_ratio_median": float(np.median(burst)),
            "expected_rate_lo": exp["rate_lo"], "expected_rate_hi": exp["rate_hi"],
            "expected_bursts": exp["bursts"], "passes": bool(ok),
            "note": exp["note"]})
        band = f"{exp['rate_lo']:.2f}-{exp['rate_hi']:.2f}"
        print(f"{exp['drug']:24s} {exp['key']:9s} "
              f"{np.percentile(ratio, 25):6.3f} {med:6.3f} {np.percentile(ratio, 75):6.3f}  "
              f"{band:>14s}  {'pass' if ok else 'FAIL'}")

    n_fail = sum(1 for c in out["checks"] if not c["passes"])
    out["n_failed"] = n_fail
    pathlib.Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(ROOT / args.out).write_text(json.dumps(out, indent=1))
    print(f"\n{len(out['checks']) - n_fail} of {len(out['checks'])} checks pass")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
