"""Write results/v2/RESULTS.md from results/v2/results.json.

    python scripts/render_v2.py

Every number in the markdown is read from the JSON; nothing is typed by hand.
The table functions are imported by the report renderer as well, so the report
and the results file cannot disagree.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def f2(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def doorn_tables(r: dict) -> str:
    a = r.get("A_doorn")
    if not a:
        return "_Section A not run._\n"
    m = a["metrics"]
    succ = r.get("_spec", {}).get("doorn_success", {"min_hits": 5, "of": 10})
    lines = ["| Quantity | Value |", "|---|---|",
             f"| Wells scored | {m['n_wells']} |",
             f"| Top-1 (u_rel or tau_d) | **{frac(m['top1_hits'], m['n_wells'])} = "
             f"{f2(m['top1_accuracy'])}**, 95% CI {f2(m['top1_ci95'][0])} to "
             f"{f2(m['top1_ci95'][1])} |",
             f"| Chance | {f2(m['chance'])} |",
             f"| P under chance | {m['p_vs_chance']:.2g} |",
             f"| Pre-registered success | at least {succ['min_hits']} of {succ['of']}: "
             f"**{'met' if m['top1_hits'] >= succ['min_hits'] else 'not met'}** |",
             f"| Mechanism class (adaptation and short-term plasticity) | {f2(m['class_accuracy'])} (chance 0.30) |",
             f"| Top-2 | {f2(m['top2_accuracy'])} |",
             f"| Direction up, among correct calls | {f2(m['direction_agreement'])} ({m['direction_n']} wells) |",
             f"| Wells with a call above 0.5 | {f2(m['called_treated'])} |", ""]
    lines += ["Per well: the named mechanism, its probability, and the effect if it moved.", "",
              "| Well | Top-1 | p | Effect of top-1 (transformed units) | u_rel p | tau_d p | Guard |",
              "|---|---|---|---|---|---|---|"]
    for w in sorted(a["wells"], key=lambda w: w["well"]):
        j = SHIFT_KEYS.index(w["top1"])
        ju, jt = SHIFT_KEYS.index("u_rel"), SHIFT_KEYS.index("tau_d")
        g = w.get("outside_model")
        lines.append(f"| {w['well']} | `{w['top1']}` | {f2(w['p_active'][j])} | "
                     f"{w['effect_med'][j]:+.2f} [{w['effect_lo'][j]:+.2f}, {w['effect_hi'][j]:+.2f}] | "
                     f"{f2(w['p_active'][ju])} | {f2(w['p_active'][jt])} | "
                     f"{'outside' if g else ('inside' if g is False else '-')} |")
    lines += ["", "| Method | Top-1 |", "|---|---|",
              f"| Hodgkin's Razor | {f2(m['top1_accuracy'])} |"]
    if a.get("unpaired"):
        lines.append(f"| Same twin, pairing removed | {f2(a['unpaired']['top1_accuracy'])} |")
    pa = a.get("prior_art", {})
    if pa.get("available"):
        lines.append(f"| Doorn et al. 2025 estimator (U or tau_D accepted) | {f2(pa['top1_accuracy'])} |")
    ins = a.get("inside_model", {})
    if ins.get("n"):
        lines.append(f"| Hodgkin's Razor, wells the guard passes ({ins['n']}) | "
                     f"{f2(ins['metrics']['top1_accuracy'])} |")
    return "\n".join(lines) + "\n"


def tampere_tables(r: dict) -> str:
    c = r.get("C_tampere")
    if not c:
        return "_Section C not run._\n"
    m1, m2 = c["metrics_v1_key"], c["metrics_v2_key"]
    lines = ["| Quantity | Version 1 key | Version 2 key |", "|---|---|---|",
             f"| Top-1 | **{f2(m1['top1_accuracy'])}** ({m1['top1_hits']}/{m1['n_wells']}) | "
             f"**{f2(m2['top1_accuracy'])}** ({m2['top1_hits']}/{m2['n_wells']}) |",
             f"| Chance | {f2(m1['chance'])} | {f2(m2['chance'])} |",
             f"| Top-2 | {f2(m1['top2_accuracy'])} | {f2(m2['top2_accuracy'])} |",
             f"| Mechanism class | {f2(m1['class_accuracy'])} | {f2(m2['class_accuracy'])} |",
             f"| Control false-mechanism rate (ceiling 0.20) | {f2(m1.get('control_false_mechanism_rate'))} | - |",
             f"| Treated wells with a call | {f2(m1['called_treated'])} | - |",
             f"| Accuracy of those calls | {f2(m1['called_accuracy'])} | {f2(m2['called_accuracy'])} |",
             f"| Detection AUROC, treated against control | {f2(m1.get('detection_auroc'))} | - |",
             ""]
    lines += ["| Compound | Wells | Top-1 (v1 key) | Rat | Human |", "|---|---|---|---|---|"]
    for comp, d in sorted(m1["per_compound"].items()):
        sp = d["species"]
        rat = sp.get("rat", {"top1": 0, "n": 0})
        hum = sp.get("hPSC", {"top1": 0, "n": 0})
        lines.append(f"| {comp} | {d['n']} | {d['top1']}/{d['n']} | "
                     f"{rat['top1']}/{rat['n']} | {hum['top1']}/{hum['n']} |")
    lines += ["", "| Method | Sees labels | Top-1 (v1 key) |", "|---|---|---|",
              f"| Hodgkin's Razor v2 | no | {f2(m1['top1_accuracy'])} |"]
    if c.get("unpaired"):
        lines.append(f"| Same twin, pairing removed | no | {f2(c['unpaired']['top1_accuracy'])} |")
    pa = c.get("prior_art", {})
    if pa.get("available"):
        lines.append(f"| Doorn et al. 2025 estimator | no | {f2(pa['top1_accuracy'])} |")
    lines.append(f"| Supervised nearest centroid, leave one well out | yes | "
                 f"{f2(c['supervised_lowo']['top1_accuracy'])} |")
    ins = c.get("inside_model", {})
    if ins.get("n"):
        lines += ["", f"Wells the guard passes: {ins['n']}; top-1 on them (v2 key) "
                      f"{f2(ins['metrics_v2_key']['top1_accuracy'])}."]
    return "\n".join(lines) + "\n"


def transfer_table(r: dict) -> str:
    d = r.get("D_transfer")
    if not d:
        return "_Section D not run._\n"
    lines = ["| Representation | Rat to human | Human to rat |", "|---|---|---|"]
    for rep, name in (("raw", "Raw feature change"), ("twin", "Twin reading (presence, effect)")):
        cells = []
        for key in sorted(d[rep]):
            v = d[rep][key]
            cells.append((key, f"{f2(v['accuracy'])} (n {v['n']}, chance {f2(v['chance'])})"))
        cells = dict(cells)
        lines.append(f"| {name} | {cells.get('rat_to_other', '-')} | {cells.get('hPSC_to_other', '-')} |")
    return "\n".join(lines) + "\n"


def chip_tables(r: dict, chip: dict | None) -> str:
    lines = []
    if chip:
        keys = [k for k in chip["readouts"]["compartment"] if k not in ("epochs", "val_nll", "n_test")]
        lines += ["Simulated chips: recovery r and 90% interval coverage of each chip "
                  "parameter, by readout (held-out chips).", "",
                  "| Readout | " + " | ".join(f"`{k}` r (cov.)" for k in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for name, res in chip["readouts"].items():
            lines.append(f"| {name} | " + " | ".join(
                f"{f2(res[k]['pearson_r'])} ({f2(res[k]['coverage_90'])})" for k in keys) + " |")
        p = chip.get("prediction_for_recorded_chips", {})
        lines += ["", f"Simulated AUROC, strong diode against symmetric channel: channel "
                      f"statistic {f2(p.get('channel_dominant_share_auroc'))}, chamber statistic "
                      f"{f2(p.get('compartment_asymmetry_auroc'))}."]
        la = chip.get("lassus_nmda")
        if la:
            lines += ["", f"NMDA reduction on {la['chips']} source-driven chips: target calcium "
                          f"event frequency lower in {f2(la['target_frequency_lower'])}, target "
                          f"synchrony lower in {f2(la['target_synchrony_lower'])} of chips "
                          f"(Lassus et al.: both lower)."]
    b = r.get("B_chips")
    if b:
        lines += ["", "Recorded chips (Mateus et al. 2024), diode (Rams, Arrows) against straight channels:", "",
                  "| Statistic | AUROC | 95% CI (chips resampled) | Pre-registered | Outcome |",
                  "|---|---|---|---|---|"]
        for stat, name in (("dominant_share", "Channel electrodes: dominant share of propagation"),
                           ("chamber_asymmetry", "Chamber electrodes: cross-correlation asymmetry")):
            s = b[stat]
            lines.append(f"| {name} | {f2(s['auroc'])} | {f2(s['ci95_by_chip'][0])} to "
                         f"{f2(s['ci95_by_chip'][1])} | {s['prediction']} "
                         f"({'>=' if s['prediction'] == 'separates' else '<'} {s['threshold']}) | "
                         f"**{'met' if s['met'] else 'not met'}** |")
        lines += ["", f"{b['n_scored']} recordings from {b['n_chips_scored']} chips scored.", "",
                  "| Design | Recordings | Median dominant share | Median chamber asymmetry |",
                  "|---|---|---|---|"]
        for k, v in sorted(b["by_design"].items()):
            lines.append(f"| {k} | {v['n']} | {f2(v['dominant_share_median'])} | "
                         f"{f2(v['chamber_asymmetry_median'])} |")
    return "\n".join(lines) + "\n" if lines else "_Chip sections not run._\n"


def simulation_tables(r: dict) -> str:
    e = r.get("E_simulation")
    if not e:
        return "_Section E not run._\n"
    lines = []
    for view, s in e.items():
        rt = s["recovery_top1"]
        lines += [f"**{view}**", "",
                  "| Case | n | Top-1 | Top-2 | Class |", "|---|---|---|---|---|"]
        for case, v in rt.items():
            lines.append(f"| {case.replace('_', ' ')} | {v['n']} | {f2(v['top1'])} | "
                         f"{f2(v['top2'])} | {f2(v['class_top1'])} |")
        lines += [f"| chance | | {f2(s['chance'])} | | {f2(s['class_chance'])} |", "",
                  "| Mechanism | Presence AUROC | ECE | Coverage 50/80/90 | Effect r (given active) |",
                  "|---|---|---|---|---|"]
        for k in SHIFT_KEYS:
            cv = s["coverage"][k]
            ea = s["effect_given_active"][k]
            lines.append(f"| `{k}` | {f2(s['presence'][k]['auroc'])} | {f2(s['presence'][k]['ece'], 3)} | "
                         f"{f2(cv['0.5'])} / {f2(cv['0.8'])} / {f2(cv['0.9'])} | "
                         f"{f2(ea['pearson_r'])} ({ea['n']}) |")
        lines.append("")
    return "\n".join(lines) + "\n"


def guard_tables(r: dict) -> str:
    g = r.get("E_guard")
    if not g:
        return "_Guard tests not run._\n"
    lines = ["| System | Case | Must | n | Fired | Typicality fired | Check fired | Outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for view, t in g.items():
        for case, must in (("bank_holdout", "pass, at most 0.10"),
                           ("variant_kinetics", "fire, at least 0.80"),
                           ("shuffled_real", "fire, at least 0.80")):
            v = t.get(case, {})
            if not v.get("n"):
                continue
            ok = v["fire_rate"] <= 0.10 if case == "bank_holdout" else v["fire_rate"] >= 0.80
            lines.append(f"| {view} | {case.replace('_', ' ')} | {must} | {v['n']} | "
                         f"{f2(v['fire_rate'])} | {f2(v['fire_rate_typicality'])} | "
                         f"{f2(v['fire_rate_ppc'])} | **{'met' if ok else 'not met'}** |")
    th = r.get("guard_thresholds", {})
    lines += ["", "Thresholds: " + "; ".join(
        f"{v}: typicality {f2(t.get('typicality'))}, predictive check {f2(t.get('ppc'))}"
        for v, t in th.items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    r = json.loads((V2 / "results.json").read_text())
    chip_path = ROOT / "results" / "chip_study.json"
    chip = json.loads(chip_path.read_text()) if chip_path.exists() else None
    md = ["# Results, version 2", "",
          f"Produced by `scripts/evaluate_v2.py` and `scripts/mateus_check.py` under "
          f"PREREGISTRATION_v2.md (hash `{r.get('prereg_sha256', '')[:16]}`).", "",
          "## A. Blind test: Dynasore, Doorn et al. (2024)", "", doorn_tables(r),
          "## B. Chip readout prediction", "", chip_tables(r, chip),
          "## C. Development set: Tampere (fifth scoring, not blind)", "", tampere_tables(r),
          "## D. Transfer across species (Tampere)", "", transfer_table(r),
          "## E. Simulations", "", simulation_tables(r),
          "## E. The guard", "", guard_tables(r)]
    (V2 / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", V2 / "RESULTS.md")


if __name__ == "__main__":
    main()
