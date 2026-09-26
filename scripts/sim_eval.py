"""Simulation-side scores of a trained twin, for choosing between versions.

    python scripts/sim_eval.py models/twin_v2_grid12 --bank data/bank_v2

Reads only simulations: held-out cultures of the twin's own domain. This is
the evidence version 2 was selected on; no recording enters it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import nde, simulator as S
from train import load_bank
from evaluate_v2 import simulation_side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("--bank", default="data/bank_v2")
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    twin = nde.Twin.load(args.twin, device="cuda")
    view = twin.meta.get("view", "grid16")
    bank = load_bank(pathlib.Path(args.bank), view)
    keep = bank["domain"] == list(S.VIEWS).index(twin.meta.get("domain", view))
    bank = {k: v[keep] for k, v in bank.items()}
    val = np.load(pathlib.Path(args.twin) / "val_index.npy")
    res = simulation_side(twin, bank, val, n_cal=args.n_cal)
    rt = res["recovery_top1"]
    print(json.dumps({k: v for k, v in rt.items()}, indent=1))
    print("presence AUROC", {k: round(v["auroc"], 3) for k, v in res["presence"].items()})
    print("coverage90", {k: round(v["0.9"], 2) for k, v in res["coverage"].items()})
    print("effect r", {k: round(v["pearson_r"], 2) for k, v in res["effect_given_active"].items()})
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
