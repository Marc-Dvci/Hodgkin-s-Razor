"""Everything downstream of the simulation bank, in order.

    python scripts/run_all.py                 # assumes data/bank exists
    python scripts/run_all.py --with-bank     # build the bank first

Each stage is skipped when its output already exists, unless --force is given,
so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name: str, cmd: list[str], produces: pathlib.Path | None,
        force: bool) -> None:
    if produces is not None and produces.exists() and not force:
        print(f"== {name}: already done ({produces})")
        return
    print(f"== {name}")
    t0 = time.time()
    code = subprocess.call([PY] + cmd, cwd=ROOT)
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}")
    print(f"   {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bank", action="store_true")
    ap.add_argument("--pairs", type=int, default=110000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-chip", action="store_true")
    args = ap.parse_args()
    f = args.force

    if args.with_bank:
        run("regime proposal", ["scripts/fit_regime.py"],
            ROOT / "models" / "regime.joblib", f)
        run("simulation bank", ["scripts/make_bank.py", "--pairs", str(args.pairs)],
            None, f)

    run("train the twin", ["scripts/train.py", "--bank", "data/bank",
                           "--out", "models/twin"],
        ROOT / "models" / "twin" / "flow.pt", f)
    run("train the unpaired baseline",
        ["scripts/train.py", "--bank", "data/bank",
         "--out", "models/twin_unpaired", "--unpaired"],
        ROOT / "models" / "twin_unpaired" / "flow.pt", f)
    run("evaluate", ["scripts/evaluate.py"], ROOT / "results" / "results.json", f)
    run("render results", ["scripts/render_results.py"],
        ROOT / "results" / "RESULTS.md", f)
    if not args.skip_chip:
        run("chip readout study", ["scripts/chip_study.py"],
            ROOT / "results" / "chip_study.json", f)
    run("bundle examples", ["scripts/make_examples.py", "--cache"], None, True)
    run("figures", ["scripts/figures.py"], None, True)
    run("render report and writeup", ["scripts/render_report.py"],
        ROOT / "docs" / "TECHNICAL_REPORT.md", f)
    run("static site", ["scripts/export_static.py"], None, True)
    run("verify", ["scripts/verify.py"], None, True)
    print("\nall stages complete")


if __name__ == "__main__":
    main()
