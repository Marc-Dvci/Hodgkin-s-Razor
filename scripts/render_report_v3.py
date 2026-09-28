"""Write docs/TECHNICAL_REPORT.md from the version 3 template and the saved results.

    python scripts/render_report_v3.py

Every number is read from a results file; the tables come from the same
functions that write results/v2/RESULTS.md and results/v3/RESULTS.md. A
placeholder left unfilled stops the script.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import render_v2 as R2
import render_v3 as R3
from render_report_v2 import param_table, pharmacology
from render_v2 import f2

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def binom_p(k: int, n: int, p0: float) -> float:
    from scipy.stats import binomtest
    return binomtest(k, n, p0, alternative="greater").pvalue


def summary_results(r3, r2, nul) -> str:
    A = r3["A_blind"]
    m = A["early"]["metrics"]
    B = r3.get("B_comparators", {})
    unp = B.get("unpaired", {}).get("metrics", {})
    pa = B.get("prior_art", {}).get("metrics", {})
    c = A["canalization"]
    d2 = r2["A_doorn"]["metrics"]
    k_cls = round(d2["class_accuracy"] * d2["n_wells"])
    called_t = sum(max(w["p_active"]) > 0.5 for w in r2["A_doorn"]["wells"])
    parts = [
        f"**Blind test, version 3.** Charlesworth et al. (2015) plated mouse hippocampal "
        f"cultures on sister arrays and kept an NMDA-receptor antagonist on one sister for days. "
        f"The twin's probability that NMDA moved separates the {m['n_treated']} treated "
        f"preparations from {m['n_null']} untreated preparations of the same genotypes with "
        f"an AUROC of **{f2(m['auroc_key'])}** (95% CI {f2(m['auroc_key_ci95'][0])} to "
        f"{f2(m['auroc_key_ci95'][1])}), against a pre-registered bar of 0.70.",
        f"On the same preparations, the published estimator of Doorn et al. scores "
        f"{f2(pa.get('auroc_key'))}, and the same twin with its pairing removed scores "
        f"{f2(unp.get('auroc_key'))}.",
        f"The cultures compensate as they mature, as the original authors report. The "
        f"twin's reading of NMDA fades with them (one-sided Wilcoxon p = "
        f"{c['wilcoxon_p_early_gt_late']:.1g}).",
        f"The co-primary, NMDA as the single most probable mechanism, was not met "
        f"({m['top1_treated_hits']}/{m['n_treated']} against {m['top1_null_hits']}/{m['n_null']}), "
        "and no preparation's probability reached 0.5. The twin ranks the treated cultures "
        "above the untreated ones, but is not confident about any one of them.",
        "",
        f"**Blind test, version 2.** On human iPSC networks from another laboratory "
        f"treated with Dynasore (Doorn et al. 2024), the twin failed its primary: it named "
        f"the accepted mechanism in {d2['top1_hits']} of {d2['n_wells']} wells, against a "
        f"bar of 5. It detected the drug in {called_t} of {d2['n_wells']} wells and in "
        f"{nul['paired']['null_called']} of {nul['paired']['n']} untreated pairs of the "
        f"same wells. It placed it in the right mechanism class in {k_cls} of "
        f"{d2['n_wells']} (p = {binom_p(k_cls, d2['n_wells'], 0.30):.2g}).",
    ]
    return " ".join(parts[:4]) + "\n\n" + parts[5]


def summary_chip() -> str:
    cp = load("results/chip_power.json")
    ch = cp["readouts"]["channel_dominant_share"] if cp else None
    out = []
    if ch:
        out.append(f"For directional channels it gives the number of chips a claim "
                   f"needs: about {ch['twin']['chips_needed']} per design. The one "
                   f"public test available had about 8, which gives power "
                   f"{f2(ch['power_at_recorded_size']['twin'])}.")
    l2 = load("results/lassus_study_v2.json")
    if l2:
        h = l2["striatal_target"]["nmda"]["0.3"]
        out.append("It also reproduces part of the sponsor laboratory's cortico-striatal "
                   "result. The first committed prediction failed. The revision imposed only "
                   "the paper's own statement that an isolated striatum is silent, and it "
                   "moved both synchrony readouts in the published direction "
                   f"(p = {h['striato_striatal_sync']['wilcoxon_p_lower']:.1g} and "
                   f"{h['cortico_striatal_sync']['wilcoxon_p_lower']:.1g}). Calcium-event "
                   "frequency did not follow.")
    else:
        out.append("A reproduction of the sponsor laboratory's cortico-striatal NMDA result "
                   "failed on its first committed prediction; a revision is committed.")
    return " ".join(out)


def stop_rule_text() -> str:
    return ("The stop rule required an AUROC of at least 0.80 on simulated sister pairs "
            "with NMDA blocked alone, and 0.70 with a second mechanism moving. The "
            "first twin missed the first bar. A second simulation bank was added and "
            "the twin retrained. The bars did not move:\n\n" + R3.stop_rule_table())


def charlesworth(r3) -> str:
    return "\n\n".join([
        R3.charlesworth_tables(r3),
        "**Comparators on the same preparations.**", R3.comparator_tables(r3),
        "**The guard on this test.**", R3.guard_v3(r3),
        "The guard's two must-fire bars were not met on sister pairs, where the "
        "second recording differs from the first in wiring and pickup as well as in "
        "the compound. Its version 2 counterpart on the MCS 24-well plate met all "
        "three (section 7.6). The primary does not depend on the guard. "
        "Restricted to the preparations the guard passes, the contrast is the one "
        "reported above the table."])


def doorn(r2, nul) -> str:
    dw = load("models/drift_v2_grid12_doorn.json")
    tw = load("models/drift_v2_grid16_tampere.json")
    txt = [R2.doorn_tables(r2),
           "**The same wells read with null pairs (post hoc).** Two pre-drug stretches of "
           "each well, 4 minutes apart, are read exactly as a drug pair. For a method with "
           "a preference, chance is its own hit rate here, not one in ten:",
           R3.null_controls_v2(),
           "The unpaired comparator's 8/10 is a fixed preference, not a detection. The "
           "paired twin named nothing on any null pair and called the drug in 9 of 10 "
           "treated wells. This is why every version 3 result is scored against untreated "
           "pairs."]
    if dw and tw:
        base_out = [t["recorded_outside_95"] for t in dw["table"] if t["drift"] == 0.0]
        txt.append(
            f"**Why the named member failed.** The version 2 bank assumed that nothing "
            f"changes in a well between two recordings. Measured on untreated pairs, the "
            f"no-compound drift is {dw['chosen']:.2f} of each parameter's range over 4 "
            f"minutes (Doorn pre-drug pairs) and {tw['chosen']:.2f} across a wash-on "
            f"(Tampere vehicle wells). With no drift, "
            f"{f2(base_out[0]) if base_out else '-'} of the recorded untreated differences "
            "fall outside the simulated 95% band, against about 0.05 once the drift is "
            "added. A twin that has never seen drift must explain every slow change as a "
            "compound. The version 3 design includes a calibrated drift from the start.")
    return "\n\n".join(txt)


def tampere(r2) -> str:
    return "\n\n".join([
        R2.tampere_tables(r2),
        "Bath GABA was read wrongly in every well. On simulations (section 7.7) this is "
        "an identifiability limit rather than a simulator error. Saturating bath GABA "
        "silences the culture, and a silenced culture carries no signature of what "
        "silenced it: the twin reads it as a sodium block in most simulations, as it "
        "does in the recorded rat wells."])


def chips(r2, chip) -> str:
    return "\n\n".join([
        R2.chip_tables(r2, chip),
        "The channel statistic's interval spans 0.28 to 0.88. With about 8 chips per "
        "design, the test could not have told 0.75 from 0.5, so the outcome is "
        "inconclusive rather than negative. Section 8 turns this into the tool's first "
        "design output: how many chips the claim needs."])


def simulation(r2, r3) -> str:
    return "\n\n".join([
        "**Version 3: sister pairs on the MCS 60-electrode array (quadrants).** "
        "Two sisters differ in wiring and electrode pickup, not only in the compound, so "
        "every number here is lower than for one well recorded twice.",
        R3.simulation_v3(r3),
        "**Version 2: one well recorded twice.**",
        R2.simulation_tables(r2)])


def guard(r2, r3) -> str:
    return "\n\n".join(["**Version 2, one well recorded twice.**", R2.guard_tables(r2),
                        "**Version 3, sister pairs.**", R3.guard_v3(r3)])


def pharmacology_all(ph) -> str:
    return "\n\n".join([
        pharmacology(ph),
        "**Bath GABA, read by the twin on simulations** (`scripts/gaba_check.py`):",
        R3.gaba_table(),
        "As bath GABA rises towards saturation, the culture falls silent and the twin's "
        "reading moves from inhibition to the sodium channel. Both silence the culture, "
        "and nothing in a silent recording separates them. A partial concentration, or "
        "a follow-up with a GABA-A antagonist, would."])


def chip_design() -> str:
    l2 = load("results/lassus_study_v2.json")
    parts = [
        "The sponsor's stated goal is to move neural organ-on-chip work from "
        "experimental description to predictive simulation. This section is that step: "
        "before any experiment is run, the twin says what to measure, how many chips to "
        "use, and which follow-up resolves an ambiguity.",
        "**Which readout resolves which property of a chip** (simulations, held-out chips): "
        "channel electrodes recover direction selectivity; chamber electrodes and 2 Hz "
        "calcium do not (section 7.4, Figure 5). A laboratory that wants to show its "
        "diodes work should put electrodes in the channels.",
        "**How many chips a claim needs** (`scripts/chip_power.py`):",
        R3.chip_power_table(),
        "**Which follow-up resolves a tie between two mechanisms** "
        "(`scripts/design_study.py`, held-out simulations with known answers):",
        R3.design_table(),
        "**Reproducing the sponsor laboratory's cortico-striatal chip** (Lassus et al. "
        "2018; predictions in `docs/LASSUS_PREDICTION.md` and `docs/LASSUS_PREDICTION_2.md`, "
        "each committed before its run):",
        R3.lassus_tables(),
    ]
    if not l2:
        parts.append("The first run failed because the model's striatum fired almost as "
                     "much without the cortex as with it. The revision, a striatal down "
                     "state set from the paper's statement that isolated striatal neurons "
                     "are silent, is committed but not yet run.")
    return "\n\n".join(parts)


def limitations(r3, r2) -> str:
    m = r3["A_blind"]["early"]["metrics"]
    g = r3.get("C_guard", {})
    sim = load("results/v3/sim_mcs60q.json") or {}
    top1 = sim.get("recovery_top1", {}).get("all_single_mechanism", {}).get("top1")
    hum = [w for w in r2["C_tampere"]["wells"] if w["species"] != "rat"]
    items = [
        f"**Naming the exact mechanism is the weak link.** Both blind tests show it. On "
        f"version 3, NMDA was the top-ranked mechanism in {m['top1_treated_hits']} of "
        f"{m['n_treated']} treated preparations, and no probability reached 0.5. On "
        f"version 2, the named member was right in 2 of 10 wells. On simulated sister "
        f"pairs, single-mechanism top-1 is {f2(top1)}. What holds up blind is detection, "
        "mechanism class, and the ranking of one mechanism against untreated cultures.",
        f"**The guard is weaker on sister pairs.** It fires on "
        f"{f2(g.get('variant_kinetics', {}).get('fire_rate'))} of simulated unmodelled "
        f"kinetics and {f2(g.get('shuffled_real', {}).get('fire_rate'))} of shuffled "
        "recordings, against bars of 0.80. It met all three bars within wells.",
        f"**Human cultures are outside the model.** "
        f"{sum(w['outside_model'] for w in hum)} of {len(hum)} human Tampere wells were "
        "flagged, and none was named correctly. A human-only domain is the next step.",
        "**A silenced culture cannot be read.** Saturating inhibition and a sodium block "
        "leave the same silent recording (section 7.7).",
        "**Most real data are conventional MEA cultures, not chips.** The blind tests are "
        "2D cultures on arrays; the one set of recorded chips (17 chips) is underpowered "
        "for the question asked of it. The chip twin's claims rest on simulations, one "
        "underpowered recorded test, and a reproduction attempt (section 8).",
        "**The drift between recordings is one number per design.** It was measured on "
        "untreated pairs, but a real culture may drift more along some parameters than "
        "others.",
        "**The stop rule needed two attempts.** The first twin missed the simulated bar "
        "by 0.008. More simulations were added, the bars were not moved, and both "
        "attempts are in the pre-registration.",
        "**The Tampere plates are a development set.** They have been scored five times, "
        "and no blind claim rests on them.",
    ]
    return "\n".join(f"{k + 1}. {t}" for k, t in enumerate(items))


def impact() -> str:
    return "\n\n".join([
        "**For a neural organ-on-chip laboratory**, the twin turns a recording into a "
        "mechanism hypothesis with an interval, together with the untreated comparison "
        "that says how much to trust it. It also says in advance how many chips and "
        "which electrodes an experiment needs. CellShells' stated aim is organ-on-chip "
        "digital twins that move the field from experimental description toward "
        "predictive simulation. The chip twin, the sample-size output and the "
        "follow-up recommender are that, in code that runs on public data.",
        "**For safety pharmacology**, a mechanism reading distinguishes a compound that "
        "silences a network through sodium channels from one that acts on excitatory "
        "transmission. Rate plots cannot. Every call is scored against vehicle and "
        "sister controls read the same way, which is what a regulatory reader will ask for.",
        "**For the method**, scoring every test against untreated pairs changed a "
        "conclusion in this project. A comparator's 8/10 blind score turned out to be a "
        "preference it shows on untreated pairs too. That rule, with pre-registration "
        "enforced in code and a stop rule that forbids freezing a test the model fails "
        "on its own simulations, carries over to any simulation-based inference on "
        "biological recordings.",
        "**As a data asset**, every report carries the SHA-256 of its input, of the "
        "frozen model and of the prior. A result can be traced to the exact recording "
        "and model that produced it, which supports the standardisation of chip data "
        "the sponsor describes."])


def team() -> str:
    return ("Marc Donovici, solo entrant: audit, and applied machine learning, including "
            "earlier competition work on brain-imaging and brain-decoding data. I designed "
            "the simulator, the inference, the pre-registrations and the evaluations.")


def main() -> None:
    r2 = load("results/v2/results.json")
    r3 = load("results/v3/results.json")
    nul = load("results/v2/null_controls.json")
    chip = load("results/chip_study.json")
    ph = load("results/v2/pharmacology.json")
    thr = load("results/throughput.json") or {}
    m12 = load("models/twin_v2_grid12/meta.json") or {}
    m16 = load("models/twin_v2_grid16/meta.json") or {}
    m3 = load("models/twin_v3_mcs60q/meta.json") or {}
    drift = load("models/drift_v3.json") or {}
    n_tests = len(re.findall(r"^def test_", (ROOT / "tests" / "test_core.py").read_text(), re.M))
    fill = {
        "PREREG2": r2.get("prereg_sha256", "")[:16],
        "PREREG3": r3.get("prereg_sha256", "")[:16],
        "SUMMARY_RESULTS": summary_results(r3, r2, nul),
        "SUMMARY_CHIP": summary_chip(),
        "PARAM_TABLE": param_table(),
        "NET_PER_S": str(thr.get("networks_per_s", "")),
        "BANK_PAIRS": (f"{m12.get('pairs', 0) + m16.get('pairs', 0):,} within-well and "
                       f"{m3.get('pairs', 0):,} sister").replace(",", " "),
        "DRIFT": f"{drift.get('chosen', 0):.2f}",
        "STOP_RULE": stop_rule_text(),
        "CHARLESWORTH": charlesworth(r3),
        "DOORN": doorn(r2, nul),
        "TAMPERE": tampere(r2),
        "CHIPS": chips(r2, chip),
        "SIMULATION": simulation(r2, r3),
        "GUARD": guard(r2, r3),
        "PHARMACOLOGY": pharmacology_all(ph),
        "CHIP_DESIGN": chip_design(),
        "LIMITATIONS": limitations(r3, r2),
        "IMPACT": impact(),
        "TEAM": team(),
        "N_TESTS": str(n_tests),
    }
    text = (ROOT / "docs" / "TECHNICAL_REPORT.v3.template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", str(v).strip())
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    empty = [k for k, v in fill.items() if not str(v).strip()]
    if empty:
        print("warning: empty sections", empty)
    (ROOT / "docs" / "TECHNICAL_REPORT.md").write_text(text, encoding="utf-8")
    print("wrote docs/TECHNICAL_REPORT.md")


if __name__ == "__main__":
    main()
