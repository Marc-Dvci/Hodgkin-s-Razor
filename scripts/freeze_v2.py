"""Freeze the version 2 models and write the second pre-registration.

    python scripts/freeze_v2.py

Fills the draft in docs/PREREGISTRATION_v2.draft.md with a digest of every
model directory, the operating points, the answer keys and the chip twin's
simulated prediction, writes PREREGISTRATION_v2.md and its SHA-256, and stops
if either already exists: a pre-registration is written once.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_v2 import model_digest

ROOT = pathlib.Path(__file__).resolve().parents[1]

TWINS = {"grid16": "models/twin_v2_grid16", "grid12": "models/twin_v2_grid12"}
UNPAIRED = {"grid16": "models/twin_v2_unpaired_grid16",
            "grid12": "models/twin_v2_unpaired_grid12"}

KEY_V1 = {
    "CNQX": {"accept": ["g_ampa"], "direction": "down"},
    "D-AP5": {"accept": ["g_nmda"], "direction": "down"},
    "GABA": {"accept": ["g_gaba"], "direction": "up"},
    "Gabazine": {"accept": ["g_gaba"], "direction": "down"},
    "Kainic acid": {"accept": ["g_ampa"], "direction": "any"},
    "TTX": {"accept": ["g_na"], "direction": "down"},
    "Control": {"control": True},
}


def main() -> None:
    md, rec = ROOT / "PREREGISTRATION_v2.md", ROOT / "PREREGISTRATION_v2.sha256"
    if md.exists() or rec.exists():
        raise SystemExit("PREREGISTRATION_v2 already exists; it is written once")
    chip = json.loads((ROOT / "results" / "chip_study.json").read_text())
    pred = chip["prediction_for_recorded_chips"]

    def call(a: float) -> str:
        return "separates" if a >= 0.70 else "does not separate"

    key_v2 = json.loads(json.dumps(KEY_V1))
    key_v2["GABA"]["accept"] = ["g_gaba", "g_tonic_inh"]
    frozen = {p: model_digest(ROOT / p) for p in list(TWINS.values()) + list(UNPAIRED.values())}
    frozen["models/domain.json"] = hashlib.sha256(
        (ROOT / "models" / "domain.json").read_bytes()).hexdigest()
    spec = {
        "twins": TWINS, "unpaired": UNPAIRED, "bank": "data/bank_v2",
        "frozen_models": {k: v for k, v in frozen.items() if k.startswith("models/twin")},
        "domain_sha256": frozen["models/domain.json"],
        "operating_points": {"posterior_samples": 4000, "predictive_draws": 48,
                             "window_s": 60.0, "presence_call": 0.5,
                             "credible": 0.90, "ppc_quantile": 0.975,
                             "ppc_calibration_records": 200,
                             "min_events": 50, "min_electrodes": 3},
        "answer_keys": {
            "doorn": {"Dynasore": {"accept": ["u_rel", "tau_d"], "direction": "up"}},
            "tampere_v1": KEY_V1, "tampere_v2": key_v2},
        "doorn_success": {"min_hits": 5, "of": 10, "chance": 0.2},
        "chips": {
            "diode_designs": ["rams", "arrows"], "straight_designs": ["control"],
            "reported_only": ["tesla", "tesla_v2"],
            "min_propagation_events": 20,
            "simulated_auroc": {"dominant_share": pred["channel_dominant_share_auroc"],
                                "chamber_asymmetry": pred["compartment_asymmetry_auroc"]},
            "predictions": {"dominant_share": call(pred["channel_dominant_share_auroc"]),
                            "chamber_asymmetry": call(pred["compartment_asymmetry_auroc"])},
            "simulated_condition": chip.get("prediction_condition"),
            "thresholds": {
                "dominant_share": 0.75 if call(pred["channel_dominant_share_auroc"]) == "separates" else 0.70,
                "chamber_asymmetry": 0.75 if call(pred["compartment_asymmetry_auroc"]) == "separates" else 0.70}},
        "guard": {"must_pass_max": 0.10, "must_fire_min": 0.80},
    }
    draft = (ROOT / "docs" / "PREREGISTRATION_v2.draft.md").read_text(encoding="utf-8")
    text = (draft.replace("{DATE}", datetime.date.today().strftime("%d %B %Y"))
            .replace("{SPEC}", json.dumps(spec, indent=1)))
    md.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(md.read_bytes()).hexdigest()
    rec.write_text(f"{digest}  PREREGISTRATION_v2.md\n")
    print("wrote PREREGISTRATION_v2.md", digest)


if __name__ == "__main__":
    main()
