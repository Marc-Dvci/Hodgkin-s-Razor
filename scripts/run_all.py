"""The whole pipeline, versions 2 and 3, in order.

    python scripts/run_all.py

Each stage is skipped when its output already exists, unless --force is given,
so an interrupted run resumes. The pipeline stops before scoring anything if
PREREGISTRATION_v2.md or _v3.md is missing, because the blind data may only be scored
after the pre-registration is written (`scripts/freeze_v2.py`, `scripts/freeze_v3.py`).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
DOORN_PY = ROOT / ".venv-doorn" / ("Scripts" if sys.platform == "win32" else "bin") / "python"


def run(name: str, cmd: list[str], produces: pathlib.Path | None, force: bool,
        python: str | pathlib.Path = PY) -> None:
    if produces is not None and produces.exists() and not force:
        print(f"== {name}: already done ({produces.relative_to(ROOT)})")
        return
    print(f"== {name}")
    t0 = time.time()
    code = subprocess.call([str(python)] + cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}")
    print(f"   {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=240000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    f = args.force
    M = ROOT / "models"

    run("domains of each recording system",
        ["scripts/fit_domain.py", "--views", "grid16,grid12", "--rounds", "8",
         "--neighbours", "12"], M / "domain.json", f)
    run("simulation bank", ["scripts/make_bank.py", "--pairs", str(args.pairs),
                            "--out", "data/bank_v2"], None, f)
    for v in ("grid16", "grid12"):
        run(f"train the {v} twin",
            ["scripts/train.py", "--bank", "data/bank_v2", "--view", v, "--domain", v,
             "--conditional", "--ensemble", "5", "--epochs", "80", "--patience", "10",
             "--out", f"models/twin_v2_{v}"], M / f"twin_v2_{v}" / "flow.pt", f)
        run(f"train the {v} unpaired baseline",
            ["scripts/train.py", "--bank", "data/bank_v2", "--view", v, "--domain", v,
             "--unpaired", "--epochs", "80", "--patience", "10",
             "--out", f"models/twin_v2_unpaired_{v}"],
            M / f"twin_v2_unpaired_{v}" / "flow.pt", f)
    run("pharmacology check", ["scripts/pharmacology_check.py"],
        ROOT / "results" / "v2" / "pharmacology.json", f)
    run("chip readout study", ["scripts/chip_study.py"],
        ROOT / "results" / "chip_study.json", f)

    if not (ROOT / "PREREGISTRATION_v2.md").exists():
        print("\nThe models are built. Write the pre-registration before scoring:\n"
              "    python scripts/freeze_v2.py\nthen run this script again.")
        return

    if DOORN_PY.exists():
        for ds in ("tampere", "doorn"):
            run(f"prior-art estimator on {ds}", ["scripts/prior_art_doorn.py", "--dataset", ds],
                ROOT / "results" / f"prior_art_doorn_{ds}.json", f, python=DOORN_PY)
    else:
        print("== prior-art estimator: skipped (create .venv-doorn from requirements-doorn.txt)")
    run("chip readout prediction on recorded chips", ["scripts/mateus_check.py"],
        ROOT / "results" / "v2" / "mateus.json", f)
    run("pre-registered evaluation", ["scripts/evaluate_v2.py"],
        ROOT / "results" / "v2" / "results.json", f)
    run("render results", ["scripts/render_v2.py"], None, True)
    run("bundle examples", ["scripts/make_examples.py", "--cache"], None, True)
    run("figures", ["scripts/figures_v2.py"], None, True)
    run("null controls, version 2", ["scripts/null_controls_v2.py"],
        ROOT / "results" / "v2" / "null_controls.json", f)

    # Version 3: sister cultures on the MCS 60-electrode array.
    run("domain of the MCS 60-electrode array",
        ["scripts/fit_domain.py", "--views", "mcs60q", "--neighbours", "12", "--match",
         "--out", "models/domain_v3.json"], M / "domain_v3.json", f)
    run("drift between sisters, 6 and 7 days", ["scripts/calibrate_drift.py"],
        M / "drift_v3.json", f)
    run("drift between sisters, A-C pairs at 10 to 14 days",
        ["scripts/calibrate_drift.py", "--kind", "a-c", "--min-div", "10", "--max-div", "14",
         "--drifts", "0.06,0.08,0.10,0.12,0.14,0.16,0.20,0.25,0.30",
         "--out", "models/drift_v3_ac.json"], M / "drift_v3_ac.json", f)
    for bank, seed in (("data/bank_v3", "20261003"), ("data/bank_v3b", "20261004")):
        run(f"sister bank {bank}",
            ["scripts/make_bank.py", "--design", "sister", "--drift", "0.12",
             "--domains", "mcs60q", "--domain-file", "models/domain_v3.json",
             "--seed", seed, "--pairs", "144000", "--workers", "5", "--out", bank], None, f)
    banks = "data/bank_v3,data/bank_v3b"
    run("train the v3 twin",
        ["scripts/train.py", "--bank", banks, "--view", "mcs60q", "--domain", "mcs60q",
         "--conditional", "--ensemble", "5", "--epochs", "150", "--patience", "12",
         "--out", "models/twin_v3_mcs60q"], M / "twin_v3_mcs60q" / "flow.pt", f)
    run("train the v3 unpaired baseline",
        ["scripts/train.py", "--bank", banks, "--view", "mcs60q", "--domain", "mcs60q",
         "--unpaired", "--epochs", "150", "--patience", "12",
         "--out", "models/twin_v3_unpaired_mcs60q"],
        M / "twin_v3_unpaired_mcs60q" / "flow.pt", f)
    run("v3 simulations", ["scripts/sim_eval.py", "models/twin_v3_mcs60q", "--bank", banks,
                           "--out", "results/v3/sim_mcs60q.json"],
        ROOT / "results" / "v3" / "sim_mcs60q.json", f)
    if not (ROOT / "PREREGISTRATION_v3.md").exists():
        print("\nThe v3 twin is built. Check the stop rule and write the pre-registration:\n"
              "    python scripts/freeze_v3.py\nthen run this script again.")
        return
    if DOORN_PY.exists():
        run("prior-art estimator on charlesworth",
            ["scripts/prior_art_doorn.py", "--dataset", "charlesworth"],
            ROOT / "results" / "prior_art_doorn_charlesworth.json", f, python=DOORN_PY)
    run("pre-registered evaluation, version 3", ["scripts/evaluate_v3_lowmem.py"],
        ROOT / "results" / "v3" / "results.json", f)

    # Studies on simulations, and the chip twin.
    run("chip sample size", ["scripts/chip_power.py"], ROOT / "results" / "chip_power.json", f)
    run("bath GABA on simulations", ["scripts/gaba_check.py"],
        ROOT / "results" / "v2" / "gaba_check.json", f)
    run("Lassus study, first prediction", ["scripts/lassus_study.py"],
        ROOT / "results" / "lassus_study.json", f)
    run("Lassus study, down-state calibration",
        ["scripts/lassus_study.py", "--chips", "576", "--duration", "120",
         "--calibrate=0,-2,-4,-6,-8"], ROOT / "results" / "lassus_calibration.json", f)
    run("Lassus study, second prediction",
        ["scripts/lassus_study.py", "--chips", "1800", "--down-state", "-6", "--precondition",
         "--out", "results/lassus_study_v2.json"], ROOT / "results" / "lassus_study_v2.json", f)
    run("next-experiment study",
        ["scripts/design_study.py", "--twin", "models/twin_v2_grid16", "--bank", "data/bank_v2"],
        ROOT / "results" / "design_study_grid16.json", f)

    run("render results, version 3", ["scripts/render_v3.py"], None, True)
    run("figures, version 3", ["scripts/figures_v3.py"], None, True)
    run("render report", ["scripts/render_report_v3.py"], None, True)
    run("static site", ["scripts/export_static.py"], None, True)
    run("verify", ["scripts/verify.py"], None, True)
    print("\nall stages complete")


if __name__ == "__main__":
    main()
