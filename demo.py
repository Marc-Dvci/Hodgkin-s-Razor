"""One-command demonstration.

    python demo.py                 # the pre-registered evidence, well by well (no GPU)
    python demo.py --windows       # single 60 s windows of the bundled recordings (cached)
    python demo.py --live          # re-run the twin on the bundled windows (CUDA)
    python demo.py --serve         # open the web application instead

The default view reads the frozen evaluation outputs in `results/` and prints
them at the level they were scored: whole wells and whole preparations,
treated recordings beside untreated ones read the same way. Nothing is
recomputed and nothing is selected; every number comes from a file a
reviewer can regenerate with the scripts named beside it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, nde, params as P

EXAMPLES = ROOT / "data" / "examples"
KEY = {"CNQX": {"g_ampa"}, "D-AP5": {"g_nmda"}, "GABA": {"g_gaba", "g_tonic_inh"},
       "gabazine": {"g_gaba"}, "kainic acid": {"g_ampa"}, "TTX": {"g_na"},
       "Dynasore": {"u_rel", "tau_d"}, "vehicle control": None}
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]
CLASS = P.CLASS_OF
SHORT = {"excitatory transmission": "excitatory", "inhibitory transmission": "inhibitory",
         "intrinsic excitability": "excitability",
         "adaptation and short-term plasticity": "adaptation/STP"}
CLASS_CHANCE = 0.30      # pre-registered in PREREGISTRATION_v2.md


def line(char: str = "-", n: int = 92) -> None:
    print(char * n)


def _load(path: str) -> dict | None:
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def evidence() -> int:
    """The four claims, each with the untreated comparison beside it."""
    v2 = _load("results/v2/results.json")
    nulls = _load("results/v2/null_controls.json")
    if v2 is None:
        print("results/v2/results.json not found. Run: python scripts/evaluate_v2.py")
        return 1
    print()
    print("Hodgkin's Razor: what a compound did to a neural culture, and when to say nothing")
    line("=")

    v3 = _load("results/v3/results.json")
    if v3 and "A_blind" in v3:
        A3 = v3["A_blind"]
        m3 = A3["early"]["metrics"]
        print("0. Blind test (pre-registration v3): chronic NMDA blockade on sister cultures,")
        print("   Charlesworth et al. 2015. One sister of each preparation kept on APV; the")
        print("   twin reads the untreated sister as baseline and the other as treated, and")
        print("   reads untreated sister pairs of the same genotypes the same way.")
        line()
        print(f"   p(g_nmda), {m3['n_treated']} treated against {m3['n_null']} untreated "
              f"preparations: AUROC {m3['auroc_key']:.2f} "
              f"[{m3['auroc_key_ci95'][0]:.2f}, {m3['auroc_key_ci95'][1]:.2f}]  (bar 0.70): "
              + ("MET" if m3["auroc_key"] >= 0.70 and m3["auroc_key_ci95"][0] > 0.5 else "NOT MET"))
        B3 = v3.get("B_comparators", {})
        for name, key in (("same twin, pairing removed", "unpaired"),
                          ("Doorn et al. 2025 estimator", "prior_art")):
            if key in B3:
                mb = B3[key]["metrics"]
                print(f"   {name:31s} AUROC {mb['auroc_key']:.2f} "
                      f"[{mb['auroc_key_ci95'][0]:.2f}, {mb['auroc_key_ci95'][1]:.2f}]")
        c3 = A3["canalization"]
        print(f"   The reading fades as the cultures compensate: early > late, "
              f"p = {c3['wilcoxon_p_early_gt_late']:.1g}.")
        print(f"   NMDA as the single top mechanism: {m3['top1_treated_hits']}/{m3['n_treated']} "
              f"treated, {m3['top1_null_hits']}/{m3['n_null']} untreated (p {m3['top1_fisher_p']:.2g}): "
              + ("MET." if m3["top1_fisher_p"] < 0.05 else "NOT MET."))
        print()
    # 1. Detection and class on the blind Dynasore wells, beside pre-drug null pairs.
    A = v2["A_doorn"]
    wells = A["wells"]
    null = {w["well"]: w for w in (nulls or {}).get("paired", {}).get("wells", [])}
    print("1. Blind test (pre-registration v2): Dynasore, Doorn et al. 2024, 10 wells.")
    print("   Each well is read twice: before against after the drug, and two pre-drug")
    print("   stretches of the same well against each other (the null pair).")
    line()
    print(f"   {'well':9s} {'drug pair: top-1':17s} {'p':>5s} {'called':7s} {'class':17s} "
          f"{'null pair: top-1':17s} {'p':>5s} {'called'}")
    for w in wells:
        pa = np.array(w["p_active"])
        n = null.get(w["well"])
        cls = CLASS.get(w["top1"], "?")
        mark = "+" if cls == "adaptation and short-term plasticity" else "."
        print(f"   {w['well']:9s} {w['top1']:17s} {pa.max():5.2f} {str(bool(pa.max() > 0.5)):7s} "
              f"{SHORT[cls] + ' ' + mark:17s} "
              + (f"{n['top1']:17s} {n['p_top1']:5.2f} {n['called']}" if n else "(not run)"))
    m = A["metrics"]
    called_t = sum(np.max(w["p_active"]) > 0.5 for w in wells)
    called_n = sum(n["called"] for n in null.values()) if null else None
    line()
    print(f"   Detection: a mechanism called in {called_t}/{len(wells)} drug pairs"
          + (f", in {called_n}/{len(null)} null pairs." if null else "."))
    from scipy.stats import binomtest
    k_cls = round(m["class_accuracy"] * m["n_wells"])
    p_cls = binomtest(k_cls, m["n_wells"], CLASS_CHANCE, alternative="greater").pvalue
    print(f"   Class (adaptation/STP, pre-registered secondary): {k_cls}/{m['n_wells']}, "
          f"chance {CLASS_CHANCE:.2f}, p = {p_cls:.2g}.")
    print(f"   Named member (u_rel or tau_d, the primary): {m['top1_hits']}/{m['n_wells']} "
          "against a bar of 5: NOT MET. Reported as failed.")
    if nulls:
        u = nulls["unpaired"]
        print(f"   The unpaired comparator scored 8/10, but names the same mechanisms in "
              f"{u['null_hits']}/{u['n']} null pairs:")
        print("   its 8/10 is a fixed preference, not a detection. Chance for a method is its")
        print("   own hit rate when nothing was applied.")
    print()

    # 2. Vehicle controls and detection on the development set.
    C = v2["C_tampere"]
    mk = C["metrics_v2_key"]
    ctrl = [w for w in C["wells"] if w["compound"] == "Control"]
    print("2. Development set: Tampere comparative MEA (66 compound wells, 11 vehicle wells).")
    line()
    print(f"   Vehicle wells with a mechanism called: {sum(w['called'] for w in ctrl)}/{len(ctrl)}. "
          f"Treated against vehicle, detection AUROC {mk['detection_auroc']:.2f}.")
    sp = {}
    for c in mk["per_compound"].values():
        for k, v in c.get("species", {}).items():
            sp.setdefault(k, [0, 0])
            sp[k][0] += v["top1"]
            sp[k][1] += v["n"]
    print("   Named mechanism: " + "; ".join(f"{k} {h}/{n}" for k, (h, n) in sp.items())
          + f" (chance {mk['chance']:.2f}; human wells are outside the model, see 3).")
    print()

    # 3. Refusal.
    W = C["wells"]
    hum = [w for w in W if w["species"] != "rat"]
    rat = [w for w in W if w["species"] == "rat"]
    g12 = v2["E_guard"]["grid12"]
    print("3. The guard: the twin says when it cannot reproduce a recording.")
    line()
    print(f"   Human wells outside the model: {sum(w['outside_model'] for w in hum)}/{len(hum)}; "
          f"rat wells: {sum(w['outside_model'] for w in rat)}/{len(rat)}.")
    print(f"   On the MCS system it fires on {g12['bank_holdout']['fire_rate']:.2f} of held-out "
          f"simulations (bar <= 0.10), {g12['variant_kinetics']['fire_rate']:.2f} of unmodelled "
          "kinetics (bar >= 0.80)")
    print(f"   and {g12['shuffled_real']['fire_rate']:.2f} of shuffled recordings (bar >= 0.80).")
    print()

    # 4. Design outputs.
    print("4. What to run next.")
    line()
    cp = _load("results/chip_power.json")
    if cp:
        ch = cp["readouts"]["channel_dominant_share"]
        print(f"   Chip directionality (channel statistic): {ch['twin']['chips_needed']} chips per "
              f"design for 80% power if the twin is right, {ch['observed']['chips_needed']} at the")
        print(f"   recorded separation. The recorded test (Mateus et al., 17 chips) had power "
              f"{ch['power_at_recorded_size']['twin']:.2f}.")
    ds = _load("results/design_study_grid16.json")
    if ds:
        s = ds["summary"]
        print(f"   Tie-breaking follow-up on simulations ({s['recommended']['n']} ties): recommended "
              f"{s['recommended']['rate']:.2f}, fixed best {s['fixed_best']['rate']:.2f}, "
              f"random {s['random_expected']['rate']:.2f}, repeat {s['repeat']['rate']:.2f}.")
    print()

    print("Every outcome, including what failed: results/v2/RESULTS.md"
          + (" and results/v3/RESULTS.md" if v3 else ""))
    print("Single windows of the bundled recordings: python demo.py --windows")
    print()
    return 0

def windows(live: bool) -> int:
    """Single 60 s windows of the bundled recordings, as the web application shows them."""
    files = sorted(EXAMPLES.glob("*.json"))
    if not files:
        print("No examples found. Run: python scripts/make_examples.py")
        return 1

    twin = sim = None
    if live:
        from hodgkins_razor import simulator as S
        if not S.available():
            print("No CUDA device found; falling back to cached analyses.")
            live = False
        else:
            twin = {v: nde.Twin.load(ROOT / "models" / f"twin_v2_{v}", device="cuda")
                    for v in ("grid16", "grid12")}
            sim = S.Simulator()

    print()
    print("Hodgkin's Razor - which mechanism did the compound move?")
    print(f"{len(files)} recording pairs: Tampere comparative MEA dataset (CC BY 4.0)"
          + (" and Doorn et al. 2024 (Apache-2.0)" if any("dynasore" in f.name for f in files) else ""))
    line("=")
    print(f"{'recording':50s} {'top-1':12s} {'prob':>5s} {'verdict':9s} {'expected':20s}")
    line()

    hits = total = 0
    for path in files:
        d = json.loads(path.read_text())
        label = d["label"]
        want = KEY.get(d["compound"])
        if live:
            base = np.array(d["baseline"]).reshape(-1, 2)
            treat = np.array(d["treated"]).reshape(-1, 2)
            dur = float(d["duration"])
            n = int(d.get("n_elec", 16))
            view = "grid12" if n == 12 else "grid16"
            xb = F.compute(base, n, dur)
            xt = F.compute(treat, n, dur)
            rep = _live_report(twin[view], sim, xb, xt, dur, view)
        elif "analysis" in d:
            rep = d["analysis"]["report"]
        else:
            print(f"{label:50s} no cached analysis; run with --live")
            continue

        top = rep["mechanisms"][0]
        verdict = {"outside_model": "outside", "no_mechanism_called": "no call",
                   "mechanism_called": "called"}[rep["verdict"]]
        if want is None:
            expected = "nothing"
            mark = "+" if verdict != "called" else "."
        else:
            total += 1
            good = top["key"] in want
            hits += good
            expected = "/".join(sorted(want))
            mark = "+" if good else "."
        print(f"{label:50s} {top['key']:12s} {top['p_active']:5.2f} {verdict:9s} {expected:20s} {mark}")

    line()
    if total:
        print(f"top-1 on these single 60 s windows: {hits}/{total}   (chance about "
              f"{1 / len(SHIFT_KEYS):.2f}). 'outside': the guard says the twin cannot")
        print("reproduce the pair, so no mechanism is named. One window is noisy: the rat")
        print("vehicle window above is called, while the same well scored on all its windows")
        print("is not (0/11 vehicle wells called). The pre-registered scores average every")
        print("window of every well: python demo.py, and results/v2/RESULTS.md.")
    print("Full pre-registered evaluation: python scripts/evaluate_v2.py")
    print()
    return 0


def _live_report(twin, sim, xb, xt, dur, view):
    from hodgkins_razor import ppc, report
    post = twin.posterior(xb, xt)
    thr = float("inf")
    res = ROOT / "results" / "v2" / "results.json"
    if res.exists():
        thr = json.loads(res.read_text()).get("guard_thresholds", {}).get(view, {}).get("ppc", thr)
    g = ppc.check(twin, sim, xb, xt, thr, dur, 5.0, n_draws=24, post=post)
    typ = ppc.Typicality.for_twin(twin)
    guard = {"discrepancy": g["discrepancy"], "threshold": thr,
             "inside_model": g["inside_model"]}
    if typ is not None:
        t = typ.of_pair(twin, xb, xt)
        guard.update({"typicality": t, "typicality_threshold": typ.threshold})
        guard["inside_model"] = guard["inside_model"] and t <= typ.threshold
    return report.build(post, guard, xb, xt, meta={"duration_s": dur, "recording_system": view})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", action="store_true",
                    help="single windows of the bundled recordings (cached analyses)")
    ap.add_argument("--live", action="store_true",
                    help="re-run the twin on the bundled windows (needs CUDA)")
    ap.add_argument("--serve", action="store_true", help="start the web application")
    args = ap.parse_args()
    if args.serve:
        cmd = [sys.executable, str(ROOT / "app" / "server.py")]
        if not args.live:
            cmd.append("--no-gpu")
        print("http://127.0.0.1:8000")
        raise SystemExit(subprocess.call(cmd))
    if args.windows or args.live:
        raise SystemExit(windows(args.live))
    raise SystemExit(evidence())


if __name__ == "__main__":
    main()
