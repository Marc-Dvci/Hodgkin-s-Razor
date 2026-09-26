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

ROOT = pathlib.Path(__file__).resolve().parents[1]

# fold: what a saturating concentration does to the parameter.
# rate_lo / rate_hi: the firing rate after, as a fraction of before.
EXPECTATIONS = (
    {"drug": "TTX", "key": "g_na", "set": 0.09,
     "rate_lo": 0.0, "rate_hi": 0.10, "bursts": "abolished",
     "note": "a sodium channel block silences the culture"},
    {"drug": "CNQX or NBQX", "key": "g_ampa", "set": 0.005,
     "rate_lo": 0.0, "rate_hi": 0.35, "bursts": "abolished",
     "note": "AMPA carries fast excitatory transmission; blocking it "
             "abolishes network bursts in cortical culture"},
    {"drug": "D-AP5", "key": "g_nmda", "set": 0.0005,
     "rate_lo": 0.15, "rate_hi": 0.90, "bursts": "reduced",
     "note": "an NMDA block shortens bursts and reduces but does not "
             "abolish activity"},
    {"drug": "gabazine or picrotoxin", "key": "g_gaba", "set": 0.005,
     "rate_lo": 1.02, "rate_hi": 100.0, "bursts": "increased",
     "note": "removing inhibition raises firing and synchrony"},
    {"drug": "GABA or muscimol", "key": "g_tonic_inh", "set": 0.25,
     "rate_lo": 0.0, "rate_hi": 0.60, "bursts": "reduced",
     "note": "a bath agonist opens extrasynaptic receptors on every cell"},
    {"drug": "4-aminopyridine", "key": "g_kdr", "fold": 0.25,
     "rate_lo": 1.0, "rate_hi": 100.0, "bursts": "increased",
     "note": "a potassium channel block raises excitability"},
)


def bank_cultures(bank: pathlib.Path, view: str, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Untreated cultures the version 2 bank admitted to one recording system.

    These are the preparations the twin is trained on: inside the range that
    system's recorded baselines span, living, and AMPA-dependent.
    """
    thetas = []
    for f in sorted(bank.glob("shard_*.npz")):
        d = np.load(f)
        keep = d["domain"] == list(S.VIEWS).index(view)
        tc, g = d["theta_c"][keep], d["group"][keep]
        _, first = np.unique(g, return_index=True)
        thetas.append(tc[first])
    t = np.concatenate(thetas)
    pick = rng.choice(len(t), size=min(n, len(t)), replace=False)
    return P.from_unit(t[pick].astype(np.float64))


GRADED_GABA = (0.005, 0.01, 0.02, 0.04, 0.08)


def _feats(args):
    events, view, duration = args
    ev, n = S.view_events(events, view)
    return F.compute(ev, n, duration)


def main() -> None:
    import concurrent.futures as cf
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/bank_v2")
    ap.add_argument("--views", default="grid16,grid12")
    ap.add_argument("--baselines", type=int, default=120)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="results/v2/pharmacology.json")
    args = ap.parse_args()

    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    pool = cf.ProcessPoolExecutor(max_workers=8)
    out = {"duration_s": args.duration, "views": {}}
    mfr, nbr = F.NAMES.index("mfr"), F.NAMES.index("nbr")
    for view in args.views.split(","):
        base = bank_cultures(ROOT / args.bank, view, args.baselines, rng)
        n = len(base)
        conds = [dict(e) for e in EXPECTATIONS]
        conds += [{"drug": f"bath GABA, tonic +{g}", "key": "g_tonic_inh", "add": g,
                   "graded": True} for g in GRADED_GABA]
        pairs = []
        for exp in conds:
            j = P.index(exp["key"])
            for th in base:
                t = th.copy()
                if "set" in exp:
                    t[j] = exp["set"]
                elif "fold" in exp:
                    t[j] = th[j] * exp["fold"]
                else:
                    t[j] = th[j] + exp["add"]
                t[j] = float(np.clip(t[j], P.LO[j], P.HI[j]))
                pairs.extend([th, t])
        arr = np.array(pairs)
        feats = []
        for a in range(0, arr.shape[0], 384):
            chunk = arr[a:a + 384]
            res = sim.run(chunk, duration_s=args.duration, transient_s=args.transient,
                          seed=9000 + a, pair=True)
            feats.extend(pool.map(_feats, [(res.raw_events(i), view, args.duration)
                                           for i in range(chunk.shape[0])], chunksize=8))
        feats = np.array(feats)
        fb, ft = feats[0::2], feats[1::2]
        rows = []
        print(f"{view}: {n} admitted cultures")
        for i, exp in enumerate(conds):
            s = slice(i * n, (i + 1) * n)
            ratio = ft[s, mfr] / np.maximum(fb[s, mfr], 1e-9)
            burst = ft[s, nbr] / np.maximum(fb[s, nbr], 1e-9)
            med = float(np.median(ratio))
            row = {"drug": exp["drug"], "key": exp["key"],
                   "rate_ratio_p25": float(np.percentile(ratio, 25)),
                   "rate_ratio_median": med,
                   "rate_ratio_p75": float(np.percentile(ratio, 75)),
                   "burst_ratio_median": float(np.median(burst))}
            if not exp.get("graded"):
                row.update({"intervention": (f"set to {exp['set']}" if "set" in exp
                                             else f"x{exp['fold']}"),
                            "expected_rate_lo": exp["rate_lo"],
                            "expected_rate_hi": exp["rate_hi"],
                            "expected_bursts": exp["bursts"],
                            "passes": bool(exp["rate_lo"] <= med <= exp["rate_hi"]),
                            "note": exp["note"]})
            rows.append(row)
            verdict = "" if exp.get("graded") else ("pass" if row["passes"] else "FAIL")
            print(f"  {exp['drug']:26s} {exp['key']:11s} {row['rate_ratio_p25']:6.3f} "
                  f"{med:6.3f} {row['rate_ratio_p75']:6.3f}  {verdict}")
        graded = [r["rate_ratio_median"] for r in rows if r["drug"].startswith("bath GABA")]
        out["views"][view] = {"cultures": n, "checks": rows,
                              "n_failed": sum(1 for r in rows if r.get("passes") is False),
                              "graded_gaba_monotone": bool(np.all(np.diff(graded) <= 1e-9))}
    pool.shutdown()
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
