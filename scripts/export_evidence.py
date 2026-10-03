"""Write app/static/evidence.json: the scored results the page's evidence views draw.

    python scripts/export_evidence.py

Everything comes from the results files written by the frozen evaluations
(results/v3/results.json, results/v2/results.json, results/v2/null_controls.json,
results/chip_study.json, results/chip_power.json). Nothing is recomputed here,
so the page shows exactly what was scored. The page recomputes each AUROC from
the per-preparation values, and this script checks that its own recomputation
matches the written primary before it writes anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hodgkins_razor import params as P  # noqa: E402

OUT = ROOT / "app" / "static" / "evidence.json"
SHIFT = [P.KEYS[i] for i in P.SHIFT_IDX]
MATCHED = ("WT", "GluR1")
CHIP_LABELS = {"p_cross": "axons in the channels", "direction_sel": "channel directionality",
               "g_cross": "cross-chamber synapse strength", "tgt_autonomy": "target self-drive"}
READOUT_LABELS = {"compartment": "chamber electrodes", "calcium_2hz": "calcium imaging, 2 Hz",
                  "channel": "channel electrodes", "perfusion": "chamber perfusion",
                  "compartment+channel": "chamber + channel electrodes"}


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def r4(x) -> float:
    return float(round(float(x), 4))


def v3() -> dict:
    d = json.loads((ROOT / "results/v3/results.json").read_text())
    a = d["A_blind"]
    early = a["early"]["preps"]
    late = {p["prep"]: p["p_key"] for p in a["late"]["preps"] if p["kind"] == "treated"}
    unp = {(p["prep"], p["kind"]): p["p_key"] for p in d["B_comparators"]["unpaired"]["preps"]}
    pa = {(p["prep"], p["kind"]): p["p_key"] for p in d["B_comparators"]["prior_art"]["preps"]}
    preps = []
    for p in early:
        k = (p["prep"], p["kind"])
        preps.append({"id": p["prep"], "kind": p["kind"], "genotype": p["genotype"],
                      "matched": p["genotype"] in MATCHED,
                      "p": [r4(v) for v in p["p_active"]],
                      "top1": p["top1"], "late": r4(late[p["prep"]]) if p["prep"] in late else None,
                      "unpaired": r4(unp[k]) if k in unp else None,
                      "prior_art": r4(pa[k]) if k in pa else None})
    j = SHIFT.index("g_nmda")
    t = [x["p"][j] for x in preps if x["kind"] == "treated"]
    n = [x["p"][j] for x in preps if x["kind"] == "null" and x["matched"]]
    mine = auroc(t, n)
    written = a["early"]["metrics"]["auroc_key"]
    if abs(mine - written) > 2e-3:
        raise SystemExit(f"recomputed primary {mine:.4f} differs from the written {written:.4f}")
    m = a["early"]["metrics"]
    return {"prereg_sha256": d["prereg_sha256"], "freeze_commit": "7251b7b",
            "key": "g_nmda", "bar": 0.70, "ages": "10 to 14 days in vitro",
            "primary": {"auroc": r4(m["auroc_key"]), "ci95": [r4(v) for v in m["auroc_key_ci95"]],
                        "n_treated": m["n_treated"], "n_null": m["n_null"]},
            "pooled": {"auroc": r4(a["early_pooled_null"]["auroc_key"]),
                       "n_null": a["early_pooled_null"]["n_null"]},
            "comparators": {"unpaired": r4(d["B_comparators"]["unpaired"]["metrics"]["auroc_key"]),
                            "prior_art": r4(d["B_comparators"]["prior_art"]["metrics"]["auroc_key"])},
            "top1": {"treated": f'{m["top1_treated_hits"]}/{m["n_treated"]}',
                     "null": f'{m["top1_null_hits"]}/{m["n_null"]}'},
            "canalization": {"p": a["canalization"]["wilcoxon_p_early_gt_late"],
                             "late_auroc": r4(a["late"]["metrics"]["auroc_key"])},
            "preps": preps}


def doorn() -> dict:
    d = json.loads((ROOT / "results/v2/results.json").read_text())["A_doorn"]
    null = {w["well"]: w for w in
            json.loads((ROOT / "results/v2/null_controls.json").read_text())["paired"]["wells"]}
    wells = []
    for w in d["wells"]:
        pa = np.asarray(w["p_active"])
        cls = P.CLASS_OF[w["top1"]]
        wells.append({"well": w["well"], "treated_max": r4(pa.max()), "treated_called": w["called"],
                      "treated_top1": w["top1"], "treated_class": cls,
                      "class_hit": cls == P.CLASS_OF["tau_d"],
                      "null_max": r4(null[w["well"]]["p_top1"]),
                      "null_called": null[w["well"]]["called"],
                      "null_top1": null[w["well"]]["top1"]})
    return {"accept": ["u_rel", "tau_d"], "class": P.CLASS_OF["tau_d"], "wells": wells,
            "top1": f'{d["metrics"]["top1_hits"]}/{d["metrics"]["n_wells"]}'}


def chips() -> dict:
    st = json.loads((ROOT / "results/chip_study.json").read_text())["readouts"]
    pw = json.loads((ROOT / "results/chip_power.json").read_text())
    matrix = []
    for ro, v in st.items():
        matrix.append({"readout": READOUT_LABELS.get(ro, ro),
                       "r": {k: r4(v[k]["pearson_r"]) for k in CHIP_LABELS if k in v}})
    ch = pw["readouts"]["channel_dominant_share"]
    return {"properties": CHIP_LABELS, "matrix": matrix,
            "power": {"target": pw["power_target"],
                      "twin": [{"n": c["chips_per_design"], "power": r4(c["power_ci_clears_0.5"]),
                                "lo": r4(c["auroc_90pct_range"][0]), "hi": r4(c["auroc_90pct_range"][1]),
                                "median": r4(c["median_auroc"])} for c in ch["twin"]["curve"]],
                      "needed_twin": ch["twin"]["chips_needed"],
                      "needed_observed": ch["observed"]["chips_needed"],
                      "recorded": pw["recorded"],
                      "power_at_recorded": r4(ch["power_at_recorded_size"]["twin"])}}


def main() -> None:
    mech = [{"key": k, "label": P.PARAMS[P.KEYS.index(k)].label,
             "target": P.PARAMS[P.KEYS.index(k)].target, "class": P.CLASS_OF[k]} for k in SHIFT]
    ev = {"mechanisms": mech, "v3": v3(), "doorn": doorn(), "chips": chips()}
    OUT.write_text(json.dumps(ev, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB); "
          f"v3 primary {ev['v3']['primary']['auroc']}, {len(ev['v3']['preps'])} preparations")


if __name__ == "__main__":
    main()
