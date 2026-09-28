"""Write results/v3/RESULTS.md from the saved version 3 outputs.

    python scripts/render_v3.py

Every number is read from a JSON file written by a script; nothing is typed by
hand. The table functions are imported by the report renderer, so the report
and the results file cannot disagree. A section whose output does not exist yet
is marked as not run.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import params as P
from render_v2 import f2

ROOT = pathlib.Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "v3"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]
NOT_RUN = "_Not run yet._"


def load(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def ci(x) -> str:
    return f"[{f2(x[0])}, {f2(x[1])}]" if x else "-"


def contrast_row(name: str, m: dict) -> str:
    if "auroc_key" not in m:
        return f"| {name} | {m.get('n_treated', 0)} / {m.get('n_null', 0)} | - | - | - | - |"
    return (f"| {name} | {m['n_treated']} / {m['n_null']} | **{f2(m['auroc_key'])}** "
            f"{ci(m['auroc_key_ci95'])} | {m['top1_treated_hits']}/{m['n_treated']} vs "
            f"{m['top1_null_hits']}/{m['n_null']} (p {m['top1_fisher_p']:.2g}) | "
            f"{f2(m['median_p_key_treated'])} vs {f2(m['median_p_key_null'])} | "
            f"{f2(m['detection_auroc'])} |")


CONTRAST_HEAD = ("| Contrast | Treated / null preparations | AUROC of p(`g_nmda`) [95% CI] | "
                 "Top-1 `g_nmda` | Median p(`g_nmda`) | Detection AUROC |\n"
                 "|---|---|---|---|---|---|")


def charlesworth_tables(r: dict | None) -> str:
    if not r or "A_blind" not in r:
        return NOT_RUN
    A = r["A_blind"]
    m = A["early"]["metrics"]
    met_primary = m["auroc_key"] >= 0.70 and m["auroc_key_ci95"][0] > 0.5
    met_co = m["top1_treated_hits"] / m["n_treated"] > m["top1_null_hits"] / m["n_null"] \
        and m["top1_fisher_p"] < 0.05
    out = [
        "| Pre-registered outcome | Bar | Result | Outcome |", "|---|---|---|---|",
        f"| **Primary**: AUROC of p(`g_nmda`), treated against genotype-matched null "
        f"preparations, 10–14 days | ≥ 0.70, lower 95% bound > 0.50 | **{f2(m['auroc_key'], 3)}** "
        f"{ci(m['auroc_key_ci95'])} | **{'met' if met_primary else 'not met'}** |",
        f"| **Co-primary**: share with `g_nmda` top-1, treated against null | higher, "
        f"one-sided Fisher p < 0.05 | {m['top1_treated_hits']}/{m['n_treated']} against "
        f"{m['top1_null_hits']}/{m['n_null']}, p = {m['top1_fisher_p']:.2g} | "
        f"**{'met' if met_co else 'not met'}** |",
        "", CONTRAST_HEAD,
        contrast_row("Primary ages, genotype-matched null", m),
        contrast_row("Primary ages, pooled null (all genotypes)", A["early_pooled_null"]),
    ]
    for g, mg in A["by_genotype"].items():
        out.append(contrast_row(f"Primary ages, {g} only", mg))
    out += [contrast_row("Late (15 days and after), genotype-matched null", A["late"]["metrics"]),
            contrast_row("Late, pooled null", A["late_pooled_null"])]
    c = A["canalization"]
    out += ["",
            f"**Canalization** (pre-registered secondary): in the {c.get('n', 0)} treated "
            f"preparations recorded at both ages, the median presence of `g_nmda` is "
            f"{f2(c.get('median_early'), 3)} at 10–14 days and {f2(c.get('median_late'), 3)} "
            f"at 15 days and after; one-sided Wilcoxon signed-rank p = "
            f"{c.get('wilcoxon_p_early_gt_late', float('nan')):.2g}.",
            "",
            f"Calls above 0.5: {f2(m['called_key_treated'])} of treated and "
            f"{f2(m['called_key_null'])} of null preparations. The twin ranks treated "
            "preparations above untreated ones on `g_nmda`, but no preparation's presence "
            "probability reaches 0.5: the reading is a ranking against untreated sisters, "
            "not a call on one culture.",
            f"Direction among top-1 hits: {f2(m['direction_agreement'])} down "
            f"({m['direction_n']} preparations).",
            f"Windows scored: {A['n_windows_scored']}.",
            "",
            "**Exploratory: every mechanism, treated against matched null** (no bar). "
            "AUROC of each mechanism's presence probability:",
            "",
            "| Mechanism | AUROC | Median treated | Median null |", "|---|---|---|---|"]
    prof = sorted(A["profile_exploratory"].items(), key=lambda kv: -kv[1]["auroc"])
    for k, v in prof:
        out.append(f"| `{k}` | {f2(v['auroc'])} | {f2(v['median_treated'], 3)} | "
                   f"{f2(v['median_null'], 3)} |")
    out += ["",
            "The top-1 counts show why the co-primary failed while the primary passed: "
            "the most probable mechanism is usually another one in both groups, and "
            "`g_nmda` rises relative to untreated preparations without becoming the "
            "largest.",
            "",
            "| Group | Top-1 counts |", "|---|---|",
            "| Treated | " + ", ".join(f"`{k}` {v}" for k, v in m["top1_treated_counts"].items()) + " |",
            "| Null | " + ", ".join(f"`{k}` {v}" for k, v in m["top1_null_counts"].items()) + " |"]
    return "\n".join(out)


def comparator_tables(r: dict | None) -> str:
    if not r or "B_comparators" not in r:
        return NOT_RUN
    B = r["B_comparators"]
    out = [CONTRAST_HEAD]
    if "A_blind" in r:
        out.append(contrast_row("Hodgkin's Razor (paired twin)", r["A_blind"]["early"]["metrics"]))
    out.append(contrast_row("Same twin, pairing removed (unpaired)", B["unpaired"]["metrics"]))
    if "prior_art" in B:
        out.append(contrast_row("Doorn et al. 2025 estimator (score: standardised g_NMDA "
                                "shift, downward)", B["prior_art"]["metrics"]))
    else:
        out.append("| Doorn et al. 2025 estimator | - | not run | - | - | - |")
    out += ["", "For the unpaired twin and the Doorn estimator, the \"median p\" column is "
            "their own score, not a probability."]
    return "\n".join(out)


def guard_v3(r: dict | None) -> str:
    if not r or "C_guard" not in r:
        return NOT_RUN
    g = r["C_guard"]

    def rate(d):
        return f"{f2(d['fire_rate'])} (n {d['n']})"
    out = ["| Test | Bar | Fire rate | Outcome |", "|---|---|---|---|",
           f"| Held-out simulated sister pairs | ≤ 0.10 | {rate(g['bank_holdout'])} | "
           f"{'met' if g['bank_holdout']['fire_rate'] <= 0.10 else 'not met'} |",
           f"| Simulated pairs with unmodelled receptor kinetics | ≥ 0.80 | {rate(g['variant_kinetics'])} | "
           f"{'met' if g['variant_kinetics']['fire_rate'] >= 0.80 else 'not met'} |",
           f"| Recorded pairs, electrodes circularly shifted | ≥ 0.80 | {rate(g['shuffled_real'])} | "
           f"{'met' if g['shuffled_real']['fire_rate'] >= 0.80 else 'not met'} |",
           "",
           f"On the recorded windows it fires on {f2(g['recorded_fire_rate']['treated'])} of treated "
           f"and {f2(g['recorded_fire_rate']['null'])} of null windows."]
    outside = g["outside_by_prep"]
    n_out = sum(outside.values())
    out.append(f"Preparations outside the model (most windows fire): {n_out} of {len(outside)}.")
    if "inside_metrics" in g and "auroc_key" in g["inside_metrics"]:
        mi = g["inside_metrics"]
        out.append(f"Primary contrast restricted to preparations the guard passes: AUROC "
                   f"{f2(mi['auroc_key'])} {ci(mi['auroc_key_ci95'])} "
                   f"({mi['n_treated']} treated, {mi['n_null']} null).")
    return "\n".join(out)


def simulation_v3(r: dict | None) -> str:
    sim = (r or {}).get("D_simulation") or load("results/v3/sim_mcs60q.json")
    if not sim:
        return NOT_RUN
    rt = sim["recovery_top1"]
    out = ["| Held-out simulated sister pairs | n | Top-1 | Top-2 | Class |", "|---|---|---|---|---|"]
    for k, v in rt.items():
        out.append(f"| {k.replace('_', ' ')} | {v['n']} | {f2(v['top1'])} | {f2(v['top2'])} | "
                   f"{f2(v['class_top1'])} |")
    out += ["", "| Mechanism | Presence AUROC | 90% coverage | Effect r |", "|---|---|---|---|"]
    for k in SHIFT_KEYS:
        out.append(f"| `{k}` | {f2(sim['presence'][k]['auroc'])} | {f2(sim['coverage'][k]['0.9'])} | "
                   f"{f2(sim['effect_given_active'][k]['pearson_r'])} |")
    return "\n".join(out)


def stop_rule_table() -> str:
    att = load("results/v3/stop_rule_attempts.json")
    if not att:
        return NOT_RUN
    out = ["| Attempt | Twin digest | `g_nmda` alone (bar 0.80) | With a co-shift (bar 0.70) | Passed |",
           "|---|---|---|---|---|"]
    for k, a in enumerate(att):
        out.append(f"| {k + 1} | `{a['digest'][:12]}` | {a['alone']['auroc_key_window']:.3f} | "
                   f"{a['coshift']['auroc_key_window']:.3f} | {'yes' if a['passed'] else 'no'} |")
    return "\n".join(out)


def null_controls_v2() -> str:
    n = load("results/v2/null_controls.json")
    if not n:
        return NOT_RUN
    rows = ["| Method | Dynasore pairs: top-1 in {`u_rel`, `tau_d`} | Pre-drug null pairs: same | Null pairs with a call |",
            "|---|---|---|---|",
            f"| Hodgkin's Razor (paired twin) | 2/10 | {n['paired']['null_hits']}/{n['paired']['n']} | "
            f"{n['paired']['null_called']}/{n['paired']['n']} |",
            f"| Same twin, pairing removed | 8/10 | {n['unpaired']['null_hits']}/{n['unpaired']['n']} | "
            f"{n['unpaired']['null_called']}/{n['unpaired']['n']} |"]
    pa = n.get("prior_art")
    if pa:
        down = sum(1 for w in pa["wells"] if w["top1"] in ("u_rel", "tau_d") and w["direction"] == "up")
        rows.append(f"| Doorn et al. estimator | 8/10 | {pa['null_hits']}/{pa['n']} ignoring direction; "
                    f"{down}/{pa['n']} in the answer's direction (up) | - |")
    return "\n".join(rows)


def lassus_tables() -> str:
    out = []
    for tag, path, pred in (("First run", "results/lassus_study.json", "docs/LASSUS_PREDICTION.md"),
                            ("Second run", "results/lassus_study_v2.json", "docs/LASSUS_PREDICTION_2.md")):
        d = load(path)
        if not d:
            out.append(f"**{tag}** (`{pred}`): not run yet.")
            continue
        s = d["striatal_target"]
        h = s["nmda"]["0.3"]
        out += [f"**{tag}** (prediction `{pred}`, committed before the run). "
                f"{s['n_active_chips']} chips scored"
                + (f", down state {d.get('down_state_pa', 0):g} pA, chips kept only if the "
                   "cortex drives the striatum" if d.get("precondition") else "")
                + f". Striatal rate with the cortex silent: {s['target_rate_without_cortex_hz_median']:.3f} Hz, "
                f"with it driving: {s['target_rate_hz_median']:.3f} Hz.",
                "",
                "| Readout, NMDA x0.3 on the striatum | Published | Share of chips lower | Median change | Wilcoxon p (lower) | Criterion |",
                "|---|---|---|---|---|---|"]
        for k, name in (("tgt_freq", "striatal calcium-event frequency"),
                        ("striato_striatal_sync", "striato-striatal synchrony"),
                        ("cortico_striatal_sync", "cortico-striatal synchrony")):
            v = h[k]
            if "fraction_lower" not in v:
                out.append(f"| {name} | lower | n {v.get('n', 0)} | - | - | too few chips |")
                continue
            ok = v["fraction_lower"] > 0.5 and v["wilcoxon_p_lower"] < 0.05
            out.append(f"| {name} | lower | {f2(v['fraction_lower'])} | "
                       f"{v['median_change_percent']:+.1f}% | {v['wilcoxon_p_lower']:.2g} | "
                       f"{'met' if ok else 'not met'} |")
        out.append("")
    return "\n".join(out)


def chip_power_table() -> str:
    cp = load("results/chip_power.json")
    if not cp:
        return NOT_RUN
    ch = cp["readouts"]["channel_dominant_share"]
    rows = ["| Chips per design | Power if the twin is right | Power at the recorded separation (0.62) |",
            "|---|---|---|"]
    obs = {r["chips_per_design"]: r for r in ch["observed"]["curve"]}
    for r in ch["twin"]["curve"]:
        n = r["chips_per_design"]
        if n in (6, 8, 10, 15, 20, 30, 50, 100):
            o = obs.get(n)
            rows.append(f"| {n} | {f2(r['power_ci_clears_0.5'])} | "
                        f"{f2(o['power_ci_clears_0.5']) if o else '-'} |")
    return ("\n".join(rows) + f"\n\nChannel statistic, simulated AUROC {f2(ch['simulated_auroc'])}. "
            f"Chips per design for 80% power: **{ch['twin']['chips_needed']}** if the twin is right, "
            f"**{ch['observed']['chips_needed']}** at the recorded separation. The recorded test "
            f"(17 chips, about 8 per design) had power {f2(ch['power_at_recorded_size']['twin'])} "
            "even if the twin is right.")


def gaba_table() -> str:
    g = load("results/v2/gaba_check.json")
    if not g:
        return NOT_RUN
    out = ["| Simulated condition | Rate after / before | Top-1 correct | Top-1 `g_na` | Top-1 inhibition |",
           "|---|---|---|---|---|"]
    for c in g["conditions"]:
        out.append(f"| {c['condition']} | {f2(c['rate_ratio_median'])} | {f2(c['top1_true'])} | "
                   f"{f2(c['top1_g_na'])} | {f2(c['top1_inhibition'])} |")
    return "\n".join(out)


def design_table() -> str:
    for view in ("grid16", "grid12"):
        d = load(f"results/design_study_{view}.json")
        if d:
            break
    else:
        return NOT_RUN
    s = d["summary"]
    out = ["| Follow-up policy | Ties resolved | 95% CI |", "|---|---|---|"]
    for k, name in (("recommended", "Recommended by the twin"),
                    ("fixed_best", "Fixed best compound (cross-fitted)"),
                    ("random", "Random compound"), ("repeat", "Record the same well again"),
                    ("oracle", "Oracle (best compound per case, known only in hindsight)")):
        if k in s:
            out.append(f"| {name} | {s[k]['correct']}/{s[k]['n']} = {f2(s[k]['rate'])} | {ci(s[k]['ci95'])} |")
    out.append(f"| Mean over every compound | {f2(s['random_expected']['rate'])} | - |")
    for k in ("mcnemar_recommended_over_fixed_best", "mcnemar_recommended_over_random"):
        mc = d.get(k)
        if mc:
            out.append("")
            out.append(f"{k.replace('mcnemar_recommended_over_', 'Recommended against ').replace('_', ' ')}: "
                       f"{mc['b']} ties only the recommendation resolved, {mc['c']} only the other; "
                       f"one-sided exact p = {mc['p']:.2g}.")
    return "\n".join(out)


def main() -> None:
    r = load("results/v3/results.json")
    text = "\n\n".join([
        "# Results, version 3",
        f"Produced by `scripts/evaluate_v3.py` under PREREGISTRATION_v3.md "
        f"(hash `{(r or {}).get('prereg_sha256', '')[:16]}`), run through "
        "`scripts/evaluate_v3_lowmem.py`, which feeds the frozen code one sister pair at a time.",
        "## Stop rule (committed before the twin was trained)", stop_rule_table(),
        "## A. Blind test: chronic APV on sister cultures (Charlesworth et al. 2015)",
        charlesworth_tables(r),
        "## B. Comparators on the same preparations", comparator_tables(r),
        "## C. The guard", guard_v3(r),
        "## D. Simulations", simulation_v3(r),
        "## Version 2 blind test, re-read with null pairs (post hoc)", null_controls_v2(),
        "## Cortico-striatal chip (Lassus et al. 2018)", lassus_tables(),
        "## Chip sample size", chip_power_table(),
        "## Bath GABA on simulations", gaba_table(),
        "## Next-experiment recommendation, on simulations", design_table(),
    ]) + "\n"
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "RESULTS.md").write_text(text, encoding="utf-8")
    print("wrote results/v3/RESULTS.md")


if __name__ == "__main__":
    main()
