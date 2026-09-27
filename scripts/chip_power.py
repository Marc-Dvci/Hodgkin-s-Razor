"""How many chips does a laboratory need to show that a channel is directional?

    python scripts/chip_power.py

The recorded chip test of version 2 (Mateus et al. 2024, 17 chips) gave a
channel-statistic AUROC of 0.62 with a 95% interval of 0.28 to 0.88: it could
not have told 0.75 from 0.5. This script turns that into a design output. From
the simulated chip bank (`data/chip_bank_v2.npz`, chips whose two chambers are
both self-active, as in Mateus et al.), it draws n strong-diode and n
symmetric-channel chips, computes the AUROC of each readout statistic with a
Hanley-McNeil 95% interval, and repeats. Power is the share of draws whose
interval clears 0.5.

Two assumptions are reported side by side:

* **twin**: the separation the simulated chips show;
* **observed**: a binormal separation equal to the recorded AUROC (0.62), in
  case the simulations are optimistic.

Reads simulations only; no recording enters it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import features as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMMETRIC_SEEDING = 0.7          # as in chip_study.prediction_for_recorded_chips
RECORDED = {"auroc": 0.62, "ci95": [0.28, 0.88], "chips": 17,
            "source": "results/v2/mateus.json (Mateus et al. 2024)"}


def auroc(s1: np.ndarray, s0: np.ndarray) -> float:
    r = np.argsort(np.argsort(np.r_[s1, s0])) + 1.0
    return float((r[:s1.size].sum() - s1.size * (s1.size + 1) / 2) / (s1.size * s0.size))


def hanley_mcneil_lower(a: float, n1: int, n0: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return a - 1.96 * np.sqrt(max(var, 0.0))


def power_curve(s1: np.ndarray, s0: np.ndarray, sizes: list[int], reps: int,
                rng: np.random.Generator) -> list[dict]:
    out = []
    for n in sizes:
        a = np.empty(reps)
        clear = np.empty(reps, bool)
        for r in range(reps):
            x1 = rng.choice(s1, n, replace=True)
            x0 = rng.choice(s0, n, replace=True)
            a[r] = auroc(x1, x0)
            clear[r] = hanley_mcneil_lower(a[r], n, n) > 0.5
        out.append({"chips_per_design": n, "power_ci_clears_0.5": float(clear.mean()),
                    "median_auroc": float(np.median(a)),
                    "auroc_90pct_range": [float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95))]})
    return out


def needed(curve: list[dict], target: float) -> int | None:
    for row in curve:
        if row["power_ci_clears_0.5"] >= target:
            return row["chips_per_design"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/chip_bank_v2.npz")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="results/chip_power.json")
    args = ap.parse_args()
    z = np.load(ROOT / args.bank)
    keep = z["chip"][:, 3] >= SYMMETRIC_SEEDING
    d = z["chip"][keep, 1]
    strong, sym = d > 0.85, d < 0.6
    stats = {
        "channel_dominant_share": np.maximum(z["channel"][keep, 0], 1 - z["channel"][keep, 0]),
        "chamber_asymmetry": np.abs(z["compartment"][keep, 2 * F.N_FEATURE + 2]),
    }
    sizes = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    rng = np.random.default_rng(args.seed)
    res = {"recorded": RECORDED, "power_target": args.power, "reps": args.reps,
           "n_sim_strong": int(strong.sum()), "n_sim_symmetric": int(sym.sum()),
           "readouts": {}}
    for name, s in stats.items():
        ok = np.isfinite(s)
        s1, s0 = s[strong & ok], s[sym & ok]
        a_sim = auroc(s1, s0)
        twin = power_curve(s1, s0, sizes, args.reps, rng)
        # Binormal stand-in with the recorded separation: AUROC = Phi(d / sqrt 2).
        # Only the channel statistic was predicted to separate; the recorded
        # chamber statistic (0.41) has no separation to plan for.
        if name == "channel_dominant_share":
            dprime = np.sqrt(2) * norm.ppf(RECORDED["auroc"])
            g1 = rng.normal(dprime, 1.0, 200000)
            g0 = rng.normal(0.0, 1.0, 200000)
            obs = power_curve(g1, g0, sizes + [120, 150, 200], args.reps, rng)
        else:
            obs = []
        res["readouts"][name] = {
            "simulated_auroc": a_sim,
            "twin": {"curve": twin, "chips_needed": needed(twin, args.power)},
            "observed": {"curve": obs, "chips_needed": needed(obs, args.power) if obs else None},
            "power_at_recorded_size": {
                "twin": next(r["power_ci_clears_0.5"] for r in power_curve(
                    s1, s0, [RECORDED["chips"] // 2], args.reps, rng)),
            },
        }
        print(f"{name}: simulated AUROC {a_sim:.2f}; chips per design for "
              f"{args.power:.0%} power: twin {needed(twin, args.power)}, "
              f"observed-separation {needed(obs, args.power) if obs else 'n/a'}; power at "
              f"{RECORDED['chips'] // 2} per design (twin) "
              f"{res['readouts'][name]['power_at_recorded_size']['twin']:.2f}", flush=True)
    (ROOT / args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
