"""Post hoc, development evidence: does within-well drift fix the Dynasore reading?

    python scripts/drift_rescore_doorn.py --twin models/twin_v2d_grid12

The version 2 grid12 twin was trained on a bank in which nothing changes in a
well between two recordings. Untreated pairs show a drift of 0.03 to 0.04 of
each parameter's range (`models/drift_v2_grid12_doorn.json`,
`models/drift_v2_grid16_tampere.json`), and the audit of version 2 suggested
that the missing drift is why the twin named `g_ahp` rather than short-term
depression on Dynasore. The twin given here was trained on a grid12 bank built
with the Tampere-vehicle drift (0.04), calibrated without any Dynasore label.

The Dynasore wells were unblinded by the version 2 scoring, so nothing here is
a blind result. The same metrics as version 2 are computed on the treated pairs
and on the pre-drug null pairs, for the frozen version 2 twin and for the drift
twin, side by side. Writes results/v2/drift_rescore.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import evaluate_v2 as E
from hodgkins_razor import doorn as D, nde, params as P
from null_controls_v2 import null_pairs

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACCEPT = {"u_rel", "tau_d"}
CLS = P.CLASS_OF["u_rel"]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def summarise(wells: list[dict]) -> dict:
    top = [w["top1"] for w in wells]
    return {"n": len(wells),
            "top1_hits": int(sum(t in ACCEPT for t in top)),
            "class_hits": int(sum(E.class_of_well(w) == CLS for w in wells)),
            "called": int(sum(w["called"] for w in wells)),
            "top1": dict(zip([w["well"] for w in wells], top)),
            "p_accept_max": {w["well"]: float(max(w["p_active"][SHIFT_KEYS.index(k)]
                                                  for k in ACCEPT)) for w in wells}}


def score(twin_dir: str, treated, null) -> dict:
    twin = nde.Twin.load(ROOT / twin_dir, device="cuda")
    t = summarise(E.by_well(E.score_pairs(twin, treated, 4000)))
    z = summarise(E.by_well(E.score_pairs(twin, null, 4000)))
    # The version 3 primary, applied post hoc: the accepted mechanisms'
    # presence, treated wells against their own pre-drug null pairs.
    a, b = np.array(list(t["p_accept_max"].values())), np.array(list(z["p_accept_max"].values()))
    auc = E.auroc(np.r_[a, b], np.r_[np.ones(a.size), np.zeros(b.size)])
    return {"treated": t, "null": z, "auroc_accept_treated_vs_null": auc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin_v2d_grid12")
    args = ap.parse_args()
    treated = D.load(treated=True)
    null = null_pairs()
    out = {"note": "Post hoc development evidence; the Dynasore wells are not blind any more.",
           "frozen_v2": score("models/twin_v2_grid12", treated, null),
           "drift": score(args.twin, treated, null), "drift_twin": args.twin}
    (ROOT / "results" / "v2" / "drift_rescore.json").write_text(json.dumps(out, indent=1))
    for name in ("frozen_v2", "drift"):
        t, z = out[name]["treated"], out[name]["null"]
        print(f"{name:10s} treated: top-1 {t['top1_hits']}/{t['n']}, class {t['class_hits']}/{t['n']}, "
              f"called {t['called']}/{t['n']}  |  null: top-1 {z['top1_hits']}/{z['n']}, "
              f"called {z['called']}/{z['n']}  |  AUROC {out[name]['auroc_accept_treated_vs_null']:.2f}")


if __name__ == "__main__":
    main()
