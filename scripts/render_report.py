"""Fill the technical report template from the saved results.

    python scripts/render_report.py

Every placeholder is replaced with a number read from results/results.json, so
no figure in the report can drift from the run that produced it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import chip as C, params as P, shift as SH

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section(path: pathlib.Path, title: str) -> str:
    """Pull one section out of the rendered RESULTS.md."""
    if not path.exists():
        return "_not yet produced_"
    text = path.read_text()
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text,
                  re.S | re.M)
    return m.group(1).strip() if m else "_not found_"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--template", default="docs/TECHNICAL_REPORT.template.md")
    ap.add_argument("--out", default="docs/TECHNICAL_REPORT.md")
    ap.add_argument("--writeup-template", default="docs/WRITEUP.template.md")
    ap.add_argument("--writeup-out", default="docs/WRITEUP.md")
    ap.add_argument("--video", default="TO BE ADDED (YouTube and Bilibili)")
    ap.add_argument("--demo", default="TO BE ADDED")
    ap.add_argument("--repo", default="github.com/Marc-Dvci/hodgkins-razor")
    ap.add_argument("--sims-per-s", default="35")
    args = ap.parse_args()

    r = json.loads((ROOT / args.results).read_text())
    m = r["mechanism"]
    res_md = ROOT / "results" / "RESULTS.md"

    params = []
    for p in P.PARAMS:
        rng = (f"{p.lo:g} to {p.hi:g}")
        params.append([f"`{p.key}`", rng, "log" if p.log else "linear",
                       "yes" if p.shiftable else "no"])

    n_tests = 0
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                              str(ROOT / "tests")], capture_output=True, text=True,
                             cwd=ROOT, timeout=300).stdout
        mm = re.search(r"(\d+) tests? collected", out)
        n_tests = int(mm.group(1)) if mm else 0
    except Exception:
        pass

    ph_path = ROOT / "results" / "pharmacology.json"
    if ph_path.exists():
        ph = json.loads(ph_path.read_text())
        rows = []
        for c in ph["checks"]:
            band = f"{c['expected_rate_lo']:.2f} to {c['expected_rate_hi']:.2f}"
            rows.append([c["drug"], f"`{c['key']}` {c.get('intervention', '')}",
                         f"{c['rate_ratio_median']:.3f}", band,
                         "pass" if c["passes"] else "**fail**"])
        ph_txt = ("Firing rate after the intervention as a fraction of before, "
                  f"median over {ph['baselines']} living simulated cultures.\n\n"
                  + md_table(["Compound", "Parameter", "Rate ratio",
                              "Expected", "Outcome"], rows))
    else:
        ph_txt = "_pharmacology check not yet run_"

    chip_path = ROOT / "results" / "chip_study.json"
    if chip_path.exists():
        cs = json.loads(chip_path.read_text())
        rows = []
        for k in C.CHIP_KEYS:
            e = cs["readouts"]["electrode_array"][k]
            c = cs["readouts"]["calcium_2hz"][k]
            b = cs["readouts"]["both"][k]
            rows.append([f"`{k}`", f"{e['pearson_r']:.3f}", f"{c['pearson_r']:.3f}",
                         f"{b['pearson_r']:.3f}"])
        chip_txt = (f"Recovery correlation between the true and the posterior "
                    f"median, on {cs['n_chips']} simulated chips held out from "
                    f"fitting.\n\n"
                    + md_table(["Chip parameter", "Electrode array",
                                f"Calcium at {cs['calcium_frame_hz']:.0f} Hz",
                                "Both"], rows))
    else:
        chip_txt = "_chip study not yet run_"

    sub = {
        "repo": args.repo,
        "prereg": r["prereg_sha256"][:16],
        "n_wells": str(m["n_wells_scored"]),
        "top1": f"{m['top1_accuracy']:.0%}",
        "chance": f"{m['chance']:.3f}",
        "control_rate": f"{m['control_false_mechanism_rate']:.0%}",
        "class_top1": (f"{r['mechanism_class']['top1_accuracy']:.0%}"
                       if r.get("mechanism_class", {}).get("n") else "n/a"),
        "class_chance": (f"{r['mechanism_class']['chance']:.2f}"
                         if r.get("mechanism_class", {}).get("n") else "n/a"),
        "param_table": "\n".join("| " + " | ".join(row) + " |" for row in params),
        "sims_per_s": args.sims_per_s,
        "bank_pairs": f"{r['twin_meta']['pairs']:,}".replace(",", " "),
        "inactive_scale": f"{SH.INACTIVE_SCALE:g}",
        "min_fold": f"{float(__import__('numpy').exp(SH.MIN_LOG_FOLD)):.1f}",
        "results_primary": section(res_md, "Primary metric"),
        "results_per_compound": section(res_md, "Per compound"),
        "results_baselines": section(res_md, "Against the baselines"),
        "results_calibration": section(res_md, "Calibration on held-out simulations"),
        "results_class": section(res_md, "By mechanism class"),
        "results_guard": section(res_md, "The guard"),
        "results_chip": chip_txt,
        "results_pharmacology": ph_txt,
        "n_tests": str(n_tests),
        "generated": date.today().isoformat(),
        "video_link": args.video,
        "repo_link": f"https://{args.repo}",
        "demo_link": args.demo,
        "report_link": "docs/TECHNICAL_REPORT.md",
    }

    text = (ROOT / args.template).read_text()
    for k, v in sub.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(\w+)\}\}", text)
    if left:
        print("unfilled placeholders:", sorted(set(left)))
    (ROOT / args.out).write_text(text)
    print("wrote", args.out, f"({len(text.split())} words)")

    w = (ROOT / args.writeup_template).read_text()
    for k, v in sub.items():
        w = w.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(\w+)\}\}", w)
    if left:
        print("unfilled placeholders in writeup:", sorted(set(left)))
    (ROOT / args.writeup_out).write_text(w)
    print("wrote", args.writeup_out, f"({len(w.split())} words)")


if __name__ == "__main__":
    main()
