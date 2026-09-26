"""The whole version 2 pipeline, in order.

    python scripts/run_all.py

Each stage is skipped when its output already exists, unless --force is given,
so an interrupted run resumes. The pipeline stops before scoring anything if
PREREGISTRATION_v2.md is missing, because the blind data may only be scored
after the pre-registration is written (`scripts/freeze_v2.py`).
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
    run("render report", ["scripts/render_report_v2.py"], None, True)
    run("static site", ["scripts/export_static.py"], None, True)
    run("verify", ["scripts/verify.py"], None, True)
    print("\nall stages complete")


if __name__ == "__main__":
    main()
