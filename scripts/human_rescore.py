"""Post hoc, development evidence: human cultures read by a human-only twin.

    python scripts/human_rescore.py --twin models/twin_v2h_grid16 --bank data/bank_v2h

The version 2 Axion twin was trained on a domain fitted to rat and human
baselines together, and its guard put 27 of 28 human Tampere wells outside the
model. This twin is trained on a domain fitted to the human baselines alone
(`scripts/fit_domain.py --species hPSC`), with the within-well drift measured on
the Tampere vehicle wells (0.04). The Tampere plates are the development set,
scored five times already, so nothing here is blind.

Scored exactly as version 2 scored Tampere: well-level presence averaged over
windows, the version 2 answer key, and the guard (typicality and predictive
check, thresholds calibrated on this twin's own held-out simulations at the
version 2 quantiles). The frozen version 2 twin's human wells are taken from
results/v2/results.json. Writes results/v2/human_rescore.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import evaluate_v2 as E
from hodgkins_razor import nde, simulator as S, tampere as T
from train import load_bank

ROOT = pathlib.Path(__file__).resolve().parents[1]


def spec_v2() -> dict:
    md = (ROOT / "PREREGISTRATION_v2.md").read_text(encoding="utf-8")
    return json.loads(re.search(r"```json\n(.*?)\n```", md, re.S).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin_v2h_grid16")
    ap.add_argument("--bank", default="data/bank_v2h")
    args = ap.parse_args()
    spec = spec_v2()
    ops = spec["operating_points"]
    key = spec["answer_keys"]["tampere_v2"]
    pairs = [p for p in T.load_all() + [q for n in T.PLATES for q in T.load_ttx(n)]
             if p.species != "rat"]
    for p in pairs:
        p.compound = {"DAP5": "D-AP5", "KainicAcid": "Kainic acid"}.get(p.compound, p.compound)
    twin = nde.Twin.load(ROOT / args.twin, device="cuda")
    sim = S.Simulator()
    bank = load_bank(ROOT / args.bank, "grid16")
    keep = bank["domain"] == list(S.VIEWS).index("grid16")
    bank = {k: a[keep] for k, a in bank.items()}
    val = np.load(ROOT / args.twin / "val_index.npy")
    cal = E.calibrate_ppc(twin, sim, bank, val, ops["window_s"], 5.0,
                          ops["ppc_calibration_records"], ops["predictive_draws"],
                          ops["ppc_quantile"], "grid16")
    guard = E.Guard(twin, sim, cal["threshold"], ops["window_s"], 5.0, ops["predictive_draws"])
    wells = E.by_well(E.score_pairs(twin, pairs, ops.get("posterior_samples", 4000)))
    gw = E.guard_on_wells(guard, pairs, seed=60000)
    for w in wells:
        w["outside_model"] = gw.get((w["plate"], w["well"], w["compound"]), None)
    r2 = json.loads((ROOT / "results" / "v2" / "results.json").read_text())
    old = [w for w in r2["C_tampere"]["wells"] if w["species"] != "rat"]

    def summary(ws):
        treated = [w for w in ws if w["compound"] != "Control"]
        m = E.mechanism_metrics(ws, key)
        return {"n_wells": len(ws), "outside": int(sum(bool(w["outside_model"]) for w in ws)),
                "top1_hits": m.get("top1_hits"), "n_treated": len(treated),
                "class_accuracy": m.get("class_accuracy"),
                "control_false_mechanism_rate": m.get("control_false_mechanism_rate"),
                "detection_auroc": m.get("detection_auroc")}
    out = {"note": "Post hoc development evidence on the Tampere human plate.",
           "twin": args.twin, "ppc_threshold": cal["threshold"],
           "frozen_v2": summary(old), "human_domain": summary(wells), "wells": wells}
    (ROOT / "results" / "v2" / "human_rescore.json").write_text(json.dumps(out, indent=1))
    for k in ("frozen_v2", "human_domain"):
        print(k, {kk: vv for kk, vv in out[k].items()})


if __name__ == "__main__":
    main()
