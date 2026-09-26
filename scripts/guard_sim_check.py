"""The guard on simulations alone: false alarms and unmodelled kinetics.

    python scripts/guard_sim_check.py models/twin_v2_grid12 --bank data/bank_v2

The part of the guard test that needs no recording, used while the guard was
designed. The pre-registered guard test, which adds recorded pairs with their
structure destroyed, is in `scripts/evaluate_v2.py`.
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
import evaluate_v2 as E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("--bank", default="data/bank_v2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-cal", type=int, default=100)
    args = ap.parse_args()
    twin = nde.Twin.load(args.twin, device="cuda")
    v = twin.meta.get("view", "grid16")
    bank = load_bank(pathlib.Path(args.bank), v)
    keep = bank["domain"] == list(S.VIEWS).index(v)
    bank = {k: a[keep] for k, a in bank.items()}
    val = np.load(pathlib.Path(args.twin) / "val_index.npy")
    sim = S.Simulator()
    cal = E.calibrate_ppc(twin, sim, bank, val, 60.0, 5.0, args.n_cal, 48, 0.975, v)
    g = E.Guard(twin, sim, cal["threshold"], 60.0, 5.0, 48)
    rng = np.random.default_rng(3)
    pick = rng.choice(val, args.n, replace=False)
    hold = E._rate([g(bank["x_base"][i], bank["x_treat"][i], seed=int(k))
                    for k, i in enumerate(pick)])
    var = E.variant_pairs(v, args.n, 60.0, 5.0, 11)
    varr = E._rate([g(a, b, seed=100 + k) for k, (a, b) in enumerate(var)])
    print(json.dumps({"ppc_threshold": cal["threshold"],
                      "typicality_threshold": twin.meta.get("typicality_threshold"),
                      "holdout": hold, "variant_kinetics": varr}, indent=1))


if __name__ == "__main__":
    main()
