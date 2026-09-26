"""Exploratory analyses after the second pre-registration was scored.

    python scripts/exploratory_v2.py

Not pre-registered, and written after the blind results were known. It asks
one question the results raise: the paired twin won on the Tampere plates and
lost on the Dynasore wells, where the guard flagged most wells as outside the
model, while the unpaired reading did the opposite. Would a reading routed by
the guard (paired when the guard passes, unpaired when it fires) have done
better on both? The answer is a hypothesis for a third pre-registration, not a
result, and is reported as such.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import nde
import evaluate_v2 as E

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    r = json.loads((ROOT / "results" / "v2" / "results.json").read_text())
    spec = E.read_prereg()
    out = {"note": "exploratory, after unblinding; not pre-registered"}
    from hodgkins_razor import doorn as D, tampere as T
    sets = {
        "doorn": ("grid12", D.load(treated=True), r["A_doorn"]["wells"], spec["answer_keys"]["doorn"]),
        "tampere": ("grid16", T.load_all() + [p for n in T.PLATES for p in T.load_ttx(n)],
                    r["C_tampere"]["wells"], spec["answer_keys"]["tampere_v1"]),
    }
    for name, (view, pairs, paired_wells, key) in sets.items():
        for p in pairs:
            p.compound = {"DAP5": "D-AP5", "KainicAcid": "Kainic acid"}.get(p.compound, p.compound)
        up = nde.UnpairedTwin.load(ROOT / spec["unpaired"][view], device="cuda")
        unpaired = {(w["plate"], w["well"], w["compound"]): w
                    for w in E.unpaired_wells(up, pairs, 4000)}
        routed, rows = [], []
        for w in paired_wells:
            k = (w["plate"], w["well"], w["compound"])
            use = unpaired.get(k) if w.get("outside_model") else w
            if use is None:
                continue
            routed.append({**w, "top1": use["top1"], "top2": use["top2"]})
            rows.append({"well": w["well"], "compound": w["compound"],
                         "outside_model": w.get("outside_model"),
                         "paired": w["top1"], "unpaired": unpaired.get(k, {}).get("top1")})
        m = E.mechanism_metrics(routed, key)
        inside = [x for x in rows if x["outside_model"] is False]
        outside = [x for x in rows if x["outside_model"]]

        def acc(xs, field):
            s = [x for x in xs if key.get(x["compound"], {}).get("accept")]
            return (sum(x[field] in key[x["compound"]]["accept"] for x in s), len(s))
        out[name] = {"routed_top1": m["top1_accuracy"], "routed_hits": m["top1_hits"],
                     "n": m["n_wells"],
                     "inside": {"paired": acc(inside, "paired"), "unpaired": acc(inside, "unpaired")},
                     "outside": {"paired": acc(outside, "paired"), "unpaired": acc(outside, "unpaired")},
                     "rows": rows}
        print(name, {k: out[name][k] for k in ("routed_top1", "routed_hits", "n", "inside", "outside")})
    (ROOT / "results" / "v2" / "exploratory.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
