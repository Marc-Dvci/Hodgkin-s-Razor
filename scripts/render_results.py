"""Render results/results.json into results/RESULTS.md.

    python scripts/render_results.py

Every pre-registered outcome is printed, including the ones that failed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]
TARGETS = {"CNQX": "g_ampa", "D-AP5": "g_nmda", "GABA": "g_gaba",
           "Gabazine": "g_gaba", "Kainic acid": "g_ampa", "TTX": "g_na"}


def verdict(ok: bool) -> str:
    return "met" if ok else "**not met**"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="results/RESULTS.md")
    args = ap.parse_args()
    r = json.loads((ROOT / args.results).read_text())
    m = r["mechanism"]
    L: list[str] = []

    L += ["# Results", "",
          f"Produced by `scripts/evaluate.py`. Pre-registration hash "
          f"`{r['prereg_sha256'][:16]}`, verified before the run.", "",
          f"Bank: {r['twin_meta']['pairs']} simulated pairs, "
          f"{r['twin_meta']['n_val']} held out for validation and "
          f"{r['twin_meta']['n_cal']} for calibration. "
          f"Windows of {r['duration_s']:.0f} s.", ""]

    # ---------------------------------------------------------------- primary
    L += ["## Primary metric", "",
          "| Quantity | Value | Pre-registered | Outcome |", "|---|---|---|---|"]
    L += [f"| Top-1 mechanism accuracy | **{m['top1_accuracy']:.3f}** "
          f"({m['n_wells_scored']} wells) | at least 0.50 | "
          f"{verdict(m['top1_accuracy'] >= 0.50)} |"]
    L += [f"| Chance | {m['chance']:.3f} | | |"]
    L += [f"| Control false-mechanism rate | **{m['control_false_mechanism_rate']:.3f}** "
          f"({m['n_controls']} wells) | at most 0.20 | "
          f"{verdict(m['control_false_mechanism_rate'] <= 0.20)} |"]
    L += [f"| Top-2 accuracy | {m['top2_accuracy']:.3f} | | |",
          f"| Direction agreement | {m['direction_agreement']:.3f} "
          f"({m['direction_n']} wells) | | |", ""]

    # ------------------------------------------------------------- comparison
    L += ["## Against the baselines", "",
          "| Method | Sees compound labels | Top-1 accuracy |", "|---|---|---|",
          f"| Hodgkin's Razor | no | **{m['top1_accuracy']:.3f}** |"]
    if "unpaired_baseline" in r:
        u = r["unpaired_baseline"]
        L += [f"| Same twin, paired design removed | no | {u['top1_accuracy']:.3f} |"]
    s = r.get("supervised_baseline", {})
    if s and np.isfinite(s.get("top1_accuracy", float("nan"))):
        L += [f"| Supervised nearest centroid, leave-one-well-out | "
              f"yes | {s['top1_accuracy']:.3f} |"]
    L += [""]

    # ----------------------------------------------------------- per compound
    L += ["## Per compound", "",
          "| Compound | Target | Wells | Top-1 | Top-2 | Rat | Human |",
          "|---|---|---|---|---|---|---|"]
    for name in sorted(m["per_compound"]):
        d = m["per_compound"][name]
        sp = d["species"]
        rat = sp.get("rat", {})
        hum = sp.get("hPSC", {})
        f = lambda x: (f"{x['top1']}/{x['n']}" if x else "-")
        L += [f"| {name} | `{TARGETS.get(name, '-')}` | {d['n']} | "
              f"{d['top1']}/{d['n']} | {d['top2']}/{d['n']} | {f(rat)} | {f(hum)} |"]
    L += [""]

    L += ["## Per species", "", "| Species | Wells | Top-1 accuracy |", "|---|---|---|"]
    for sp, d in sorted(m["per_species"].items()):
        L += [f"| {sp} | {d['n']} | {d['top1'] / max(d['n'], 1):.3f} |"]
    L += ["", f"Windows scored: {r['real']['n_windows']}. "
              f"Excluded as unreadable before any model ran: {r['real']['n_excluded']}.", ""]

    # ------------------------------------------------------------- confusion
    conf = np.array(m["confusion"])
    rows = [i for i in range(conf.shape[0]) if conf[i].sum()]
    if rows:
        L += ["## Confusion", "",
              "Rows are the mechanism the compound acts on, columns the one the "
              "twin named.", "",
              "| acts on \\ named | " + " | ".join(SHIFT_KEYS) + " |",
              "|" + "---|" * (len(SHIFT_KEYS) + 1)]
        for i in rows:
            L += [f"| `{SHIFT_KEYS[i]}` | " +
                  " | ".join(str(int(v)) if v else "." for v in conf[i]) + " |"]
        L += [""]

    # ----------------------------------------------------------- calibration
    cal = r["calibration"]
    L += ["## Calibration on held-out simulations", "",
          f"{cal['n_records']} records the model never saw.", "",
          "| Parameter | 50% | 80% | 90% | rank KS | recovery r (active) |",
          "|---|---|---|---|---|---|"]
    for k in SHIFT_KEYS:
        c = cal["coverage"][k]
        rec = cal["recovery"][k]
        rr = "-" if not np.isfinite(rec["pearson_r"]) else f"{rec['pearson_r']:.3f}"
        L += [f"| `{k}` | {c['50']:.2f} | {c['80']:.2f} | {c['90']:.2f} | "
              f"{cal['rank_ks'][k]:.3f} | {rr} ({rec['n_active']}) |"]
    L += ["", "Nominal coverage is 0.50, 0.80 and 0.90. The rank statistic is the "
              "Kolmogorov-Smirnov distance from uniform; 0 is exact.", ""]

    pres = r["presence"]
    L += ["## Presence probability", "",
          "| Mechanism | AUROC | Expected calibration error |", "|---|---|---|"]
    for k in SHIFT_KEYS:
        d = pres[k]
        L += [f"| `{k}` | {d['auroc']:.3f} | {d['ece']:.3f} |"]
    L += [""]

    # ----------------------------------------------------------------- guard
    if "guard" in r:
        g = r["guard"]
        gc = r["guard_calibration"]
        L += ["## The guard", "",
              f"Threshold {gc['threshold']:.2f}, the {gc['quantile']:.0%} "
              f"percentile of the discrepancy over {gc['n_records']} held-out "
              f"records, fixed before any recording was scored.", "",
              "| Case | Must | n | Fired | Median discrepancy | Outcome |",
              "|---|---|---|---|---|---|"]
        spec = [("bank_holdout", "pass", "at most 0.10", lambda v: v <= 0.10),
                ("variant_kinetics", "fire", "at least 0.80", lambda v: v >= 0.80),
                ("shuffled_real", "fire", "at least 0.80", lambda v: v >= 0.80),
                ("real_pairs", "reported", "-", None)]
        names = {"bank_holdout": "held-out simulations",
                 "variant_kinetics": "changed receptor kinetics",
                 "shuffled_real": "real recordings, structure destroyed",
                 "real_pairs": "real recordings"}
        for k, must, target, test in spec:
            d = g[k]
            out = "-" if test is None else verdict(test(d["fire_rate"]))
            L += [f"| {names[k]} | {must} ({target}) | {d['n']} | "
                  f"{d['fire_rate']:.2f} | {d['median_discrepancy']:.1f} | {out} |"]
        L += [""]

    sr = r.get("simulated_recovery")
    if sr:
        L += ["## The same metric on simulations", "",
              "Held-out simulated experiments, where the answer is known. The "
              "second row is the regime a saturating concentration produces, "
              "which is where the recorded compounds sit.", "",
              "| Case | n | Top-1 | Top-2 | Class |", "|---|---|---|---|---|"]
        names = {"all_single_mechanism": "every single-mechanism pair",
                 "saturating": "strong effect and a large observable change"}
        for key, label in names.items():
            d = sr.get(key, {})
            if "top1" not in d:
                continue
            L += [f"| {label} | {d['n']} | {d['top1']:.3f} | {d['top2']:.3f} | "
                  f"{d['class_top1']:.3f} |"]
        L += [f"| chance | | {sr['chance']:.3f} | | {sr['class_chance']:.3f} |", ""]

    cm = r.get("mechanism_class")
    if cm and cm.get("n"):
        L += ["## By mechanism class", "",
              "Secondary and not pre-registered. Classes are excitatory "
              "transmission, inhibitory transmission, intrinsic excitability, "
              "and adaptation with short-term plasticity.", "",
              "| Quantity | Value |", "|---|---|",
              f"| Class accuracy | **{cm['top1_accuracy']:.3f}** ({cm['n']} wells) |",
              f"| Chance | {cm['chance']:.3f} |", ""]
        L += ["| Compound | Class correct |", "|---|---|"]
        for name in sorted(cm["per_compound"]):
            d = cm["per_compound"][name]
            L += [f"| {name} | {d['hit']}/{d['n']} |"]
        L += [""]

    if "mechanism_inside_model" in r:
        im = r["mechanism_inside_model"]
        L += ["## Restricted to wells the twin reproduces", "",
              "Secondary and not pre-registered. It is here so that a refusal "
              "cannot be read as an error.", "",
              "| Quantity | Value |", "|---|---|",
              f"| Wells passing the predictive check | {im['n_wells_inside']} "
              f"of {im['n_wells_total']} ({im['inside_fraction']:.0%}) |",
              f"| Top-1 accuracy on those wells | {im['top1_accuracy']:.3f} "
              f"({im['n_wells_scored']} scored) |", ""]

    L += ["## Figures", "",
          "`results/figures/` holds the confusion matrix, the per-compound "
          "accuracy, interval coverage, presence reliability, the guard, the "
          "recovered effect sizes, the parameter sweep and a measured raster "
          "beside the twin's.", ""]

    (ROOT / args.out).write_text("\n".join(L))
    print("wrote", args.out)
    print(f"top-1 {m['top1_accuracy']:.3f}  controls {m['control_false_mechanism_rate']:.3f}")


if __name__ == "__main__":
    main()
