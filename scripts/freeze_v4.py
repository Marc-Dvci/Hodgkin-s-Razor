"""Write the fourth pre-registration from the twin's simulated prediction.

    python scripts/freeze_v4.py

Fills scripts/templates/PREREGISTRATION_v4.draft.md with the predictions in
results/v4/brewer_prediction.json, the digest of that file and of every data
file, and the inclusion rules. Writes PREREGISTRATION_v4.md and its SHA-256,
and stops if either already exists: a pre-registration is written once. Reads
no spike train.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hodgkins_razor import brewer as B

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRED = "results/v4/brewer_prediction.json"
RULES = {"min_well_spikes": 30, "min_axons": 2, "min_axon_spikes": 50}
N_BND, N_HOME = 36, 72


def fmt(pi: list[float]) -> str:
    return f"[{pi[0]:.2f}, {pi[1]:.2f}]"


def main() -> None:
    out_md, out_sha = ROOT / "PREREGISTRATION_v4.md", ROOT / "PREREGISTRATION_v4.sha256"
    if out_md.exists() or out_sha.exists():
        raise SystemExit("PREREGISTRATION_v4 already exists; it is written once")
    pred_path = ROOT / PRED
    pred = json.loads(pred_path.read_text())
    pred_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if pred["settings"]["min_axon_spikes"] != RULES["min_axon_spikes"]:
        raise SystemExit("the prediction used a different axon-spike rule")
    at = pred["predictive_by_n"][str(N_BND)]
    home = pred["home"]["predictive_by_n"][str(N_HOME)]
    spec = {"data": "Lassers et al. 2023, Zenodo 10257483 (Dryad 10.5061/dryad.7h44j1013), CC0 1.0",
            "data_sha256": B.digest(),
            "prediction_file": PRED, "prediction_sha256": pred_sha,
            "primary_condition": "nostim", "secondary_conditions": ["hfs5", "hfs40"],
            "inclusion": RULES,
            "primary": {
                "P1": "share of home units with home > 0 >= lower bound of the twin's 90% "
                      "predictive interval at the scored n (home.predictive_by_n)",
                "P2": "Spearman(asym, traffic_ff) over boundary units inside the twin's 90% "
                      "predictive interval at the scored n (predictive_by_n, asym_vs_traffic_ff)"},
            "secondary": ["Spearman(asym, axons_ff) inside its 90% interval",
                          "P1, P2 and the above on hfs5 and hfs40, same intervals"],
            "n_clamp": {"boundary": [12, 36], "home": [12, 72]}}
    w = pred["whole_bank_spearman"]
    fill = {"DATE": datetime.date.today().strftime("%d %B %Y").lstrip("0"),
            "MIN_WELL": str(RULES["min_well_spikes"]), "MIN_AXONS": str(RULES["min_axons"]),
            "MIN_AXON_SPIKES": str(RULES["min_axon_spikes"]),
            "N_SIM": f"{pred['simulated_boundaries']:,}",
            "PRED_SHA_SHORT": pred_sha[:16], "N_BND": str(N_BND), "N_HOME": str(N_HOME),
            "RHO_TRAFFIC": f"{w['asym_vs_traffic_ff']:.2f}",
            "RHO_AXONS": f"{w['asym_vs_axons_ff']:.2f}",
            "PI_TRAFFIC": fmt(at["asym_vs_traffic_ff"]["pi90"]),
            "PI_AXONS": fmt(at["asym_vs_axons_ff"]["pi90"]),
            "HOME_SHARE": f"{pred['home']['share_positive']:.2f}",
            "PI_HOME": fmt(home["pi90"]),
            "HOME_PCT": f"{round(100 * pred['home']['share_positive'])}%",
            "SPEC": json.dumps(spec, indent=1)}
    text = (ROOT / "scripts" / "templates" / "PREREGISTRATION_v4.draft.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        raise SystemExit("unfilled placeholder in the draft")
    out_md.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(out_md.read_bytes()).hexdigest()
    out_sha.write_text(f"{digest}  PREREGISTRATION_v4.md\n", newline="\n")
    print(f"PREREGISTRATION_v4.md sha256 {digest}")


if __name__ == "__main__":
    main()
