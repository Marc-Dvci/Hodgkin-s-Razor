"""Freeze the version 3 models and write the third pre-registration.

    python scripts/freeze_v3.py

Fills docs/PREREGISTRATION_v3.draft.md with a digest of every model directory
and of the files that define the scored set, the chosen drift, and the
simulation evidence the bars were set against. Writes PREREGISTRATION_v3.md and
its SHA-256, and stops if either already exists: a pre-registration is written
once. Reads simulations only.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_v2 import auroc, model_digest
from hodgkins_razor import nde, params as P, simulator as S
from train import load_bank

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEW = "mcs60q"
TWIN = "models/twin_v3_mcs60q"
UNPAIRED = "models/twin_v3_unpaired_mcs60q"
BANK = "data/bank_v3"
DOMAIN = "models/domain_v3.json"
DRIFT = "models/drift_v3.json"
KEY = "g_nmda"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def simulated_test(twin, bank, idx_val) -> dict:
    """The blind test's contrast on held-out simulated sister pairs.

    Treated: pairs whose only mechanism is `g_nmda` down by at least
    five-fold, the regime of a saturating antagonist. Null: pairs with no
    mechanism, only the drift between sisters. One 60 s quadrant window per
    pair, where the recorded test averages many per preparation.
    """
    j = SHIFT_KEYS.index(KEY)
    act = bank["active"][idx_val]
    d = bank["delta"][idx_val][:, P.SHIFT_IDX]
    t = idx_val[(act.sum(1) == 1) & act[:, j] & (d[:, j] < -np.log(5.0))]
    z = idx_val[act.sum(1) == 0]
    sel = np.r_[t, z]
    c = twin.scaler(nde.context(bank["x_base"][sel], bank["x_treat"][sel]))
    with torch.no_grad():
        logits = twin.presence(torch.as_tensor(c, dtype=torch.float32,
                                               device=twin.device)).cpu().numpy()
    p = twin.calibrate(1.0 / (1.0 + np.exp(-logits)))
    y = np.r_[np.ones(len(t)), np.zeros(len(z))].astype(bool)
    top = p.argmax(1)
    return {"n_treated": int(len(t)), "n_null": int(len(z)),
            "auroc_key_window": auroc(p[:, j], y),
            "top1_treated": float(np.mean(top[y] == j)),
            "top1_null": float(np.mean(top[~y] == j)),
            "called_key_treated": float(np.mean(p[y, j] > 0.5)),
            "called_key_null": float(np.mean(p[~y, j] > 0.5))}


def main() -> None:
    md, rec = ROOT / "PREREGISTRATION_v3.md", ROOT / "PREREGISTRATION_v3.sha256"
    if md.exists() or rec.exists():
        raise SystemExit("PREREGISTRATION_v3 already exists; it is written once")
    drift = json.loads((ROOT / DRIFT).read_text())
    sim_path = ROOT / "results" / "v3" / "sim_mcs60q.json"
    sim = json.loads(sim_path.read_text())
    twin = nde.Twin.load(ROOT / TWIN, device="cuda")
    bank = load_bank(ROOT / BANK, VIEW)
    keep = bank["domain"] == list(S.VIEWS).index(VIEW)
    bank = {k: v[keep] for k, v in bank.items()}
    idx_val = np.load(ROOT / TWIN / "val_index.npy")
    test = simulated_test(twin, bank, idx_val)

    def sha(p: str) -> str:
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

    spec = {
        "view": VIEW, "twin": TWIN, "unpaired": UNPAIRED, "bank": BANK,
        "domain_file": DOMAIN, "drift": drift["chosen"],
        "frozen_models": {TWIN: model_digest(ROOT / TWIN),
                          UNPAIRED: model_digest(ROOT / UNPAIRED)},
        "frozen_files": {p: sha(p) for p in (DOMAIN, DRIFT, "hodgkins_razor/charlesworth.py",
                                             "data/raw/charlesworth2015/g2c-1/00g2cdata.csv")},
        "operating_points": {"posterior_samples": 1000, "window_s": 60.0,
                             "windows_per_recording": 6, "presence_call": 0.5,
                             "predictive_draws": 48, "ppc_quantile": 0.975,
                             "ppc_calibration_records": 200,
                             "min_events": 50, "min_electrodes": 3},
        "blind_test": {"dataset": "Charlesworth et al. 2015, Zenodo 31085",
                       "compound": "APV 50 uM, chronic from 7 days in vitro",
                       "accept": KEY, "direction": "down", "early_divs": [9.0, 15.0],
                       "unit": "preparation",
                       "primary": {"metric": "AUROC of p(g_nmda), treated against null preparations",
                                   "min_auroc": 0.70, "ci_lower_above": 0.50,
                                   "bootstrap": 4000},
                       "co_primary": {"metric": "top-1 g_nmda share, treated against null",
                                      "test": "one-sided Fisher exact", "alpha": 0.05}},
        "guard": {"must_pass_max": 0.10, "must_fire_min": 0.80, "n_cases": 60,
                  "windows_per_quadrant": 2},
        "simulated_test": test,
    }
    rt = sim["recovery_top1"]
    lines = [
        f"Held-out simulated sister pairs of this recording system (`{sim_path.relative_to(ROOT)}`):",
        "",
        "| Case | n | Top-1 | Top-2 | Class |", "|---|---|---|---|---|"]
    for name, r in rt.items():
        lines.append(f"| {name.replace('_', ' ')} | {r['n']} | {r['top1']:.2f} | "
                     f"{r['top2']:.2f} | {r['class_top1']:.2f} |")
    lines += ["", "Presence AUROC per mechanism: " + ", ".join(
        f"`{k}` {v['auroc']:.2f}" for k, v in sim["presence"].items()) + ".", "",
        "The test's contrast on simulations, one window per pair: "
        f"`g_nmda` blocked at least five-fold alone ({test['n_treated']} pairs) against "
        f"no mechanism ({test['n_null']} pairs): AUROC {test['auroc_key_window']:.2f}; "
        f"top-1 `g_nmda` {test['top1_treated']:.2f} against {test['top1_null']:.2f}; "
        f"called above 0.5 {test['called_key_treated']:.2f} against "
        f"{test['called_key_null']:.2f}.",
        "",
        "The recorded test has more between-preparation variation than the "
        "simulations, the drug acted for days rather than minutes, and a "
        "preparation may compensate, so the recorded AUROC is expected below the "
        "simulated one; the bar of 0.70 sits under it for that reason. With about "
        "29 treated and 100 null preparations, an AUROC of 0.70 carries a 95 "
        "percent interval of roughly 0.60 to 0.80, so the bar is testable with "
        "this sample."]
    draft = (ROOT / "docs" / "PREREGISTRATION_v3.draft.md").read_text(encoding="utf-8")
    text = (draft.replace("{DATE}", datetime.date.today().strftime("%d %B %Y"))
            .replace("{DRIFT}", f"{drift['chosen']:.2f}")
            .replace("{SIMULATION}", "\n".join(lines))
            .replace("{SPEC}", json.dumps(spec, indent=1)))
    md.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(md.read_bytes()).hexdigest()
    rec.write_text(f"{digest}  PREREGISTRATION_v3.md\n")
    print("wrote PREREGISTRATION_v3.md", digest)
    print(json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
