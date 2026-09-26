"""Write docs/TECHNICAL_REPORT.md from the template and the saved results.

    python scripts/render_report_v2.py

Every number is read from results/v2/results.json, results/chip_study.json,
results/v2/pharmacology.json and the model metadata; the tables come from the
same functions that write results/v2/RESULTS.md. A placeholder left unfilled
stops the script.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import params as P
import render_v2 as R

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def param_table() -> str:
    lines = ["| Parameter | Range | Scale | Compound may move it |", "|---|---|---|---|"]
    for p in P.PARAMS:
        lines.append(f"| `{p.key}` ({p.label}) | {p.lo:g} to {p.hi:g} | "
                     f"{'log' if p.log else 'linear'} | {'yes' if p.shiftable else 'no, held fixed across a pair'} |")
    return "\n".join(lines)


def pharmacology(ph: dict | None) -> str:
    if not ph:
        return "_Not run._"
    lines = ["| System | Compound | Parameter | Rate after / before (median, IQR) | Published range | Outcome |",
             "|---|---|---|---|---|---|"]
    for view, v in ph["views"].items():
        for c in v["checks"]:
            if "expected_rate_lo" in c:
                lines.append(f"| {view} | {c['drug']} | `{c['key']}` | {c['rate_ratio_median']:.2f} "
                             f"({c['rate_ratio_p25']:.2f} to {c['rate_ratio_p75']:.2f}) | "
                             f"{c['expected_rate_lo']:g} to {c['expected_rate_hi']:g} | "
                             f"{'pass' if c['passes'] else '**fail**'} |")
        graded = [c for c in v["checks"] if c["drug"].startswith("bath GABA")]
        if graded:
            lines.append(f"| {view} | bath GABA, graded | `g_tonic_inh` | " + ", ".join(
                f"{c['rate_ratio_median']:.2f}" for c in graded) + " | falls with dose | "
                + ("monotone" if v.get("graded_gaba_monotone") else "**not monotone**") + " |")
    return "\n".join(lines)


def main() -> None:
    r = load(ROOT / "results" / "v2" / "results.json") or {}
    chip = load(ROOT / "results" / "chip_study.json")
    ph = load(ROOT / "results" / "v2" / "pharmacology.json")
    narrative = load(ROOT / "docs" / "report_narrative.json") or {}
    meta16 = load(ROOT / "models" / "twin_v2_grid16" / "meta.json") or {}
    meta12 = load(ROOT / "models" / "twin_v2_grid12" / "meta.json") or {}
    n_tests = narrative.get("n_tests", "")
    fill = {
        "PREREG2": r.get("prereg_sha256", "")[:16],
        "PARAM_TABLE": param_table(),
        "NET_PER_S": str(narrative.get("net_per_s", "")),
        "BANK_PAIRS": f"{meta16.get('pairs', 0) + meta12.get('pairs', 0):,}".replace(",", " "),
        "DOORN": R.doorn_tables(r),
        "CHIPS": R.chip_tables(r, chip),
        "TAMPERE": R.tampere_tables(r),
        "TRANSFER": R.transfer_table(r),
        "SIMULATION": R.simulation_tables(r),
        "GUARD": R.guard_tables(r),
        "PHARMACOLOGY": pharmacology(ph),
        "N_TESTS": str(n_tests),
    }
    for k in ("SUMMARY_RESULTS", "SUMMARY_CHIP", "CHIP_DESIGN", "LIMITATIONS", "IMPACT"):
        fill[k] = narrative.get(k.lower(), "")
    text = (ROOT / "docs" / "TECHNICAL_REPORT.template.md").read_text(encoding="utf-8")
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
