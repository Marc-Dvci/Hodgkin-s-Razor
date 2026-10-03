"""Read each mechanism against the untreated controls of its own experiment. Post hoc.

    python scripts/control_referenced.py            # needs CUDA for the Dynasore null pairs
    python scripts/control_referenced.py --cached   # reuse results/control_referenced_doorn_null.json

Run after all three pre-registrations were scored, so none of this is
pre-registered and every set below has been seen.

Why. The presence heads give each mechanism a different probability when
nothing was applied: on the untreated v3 sisters, g_kdr has a median of 0.15
and g_ampa 0.04. Taking the largest raw probability therefore names the
mechanism with the highest base rate, whatever the compound did. Every
laboratory plate carries untreated or vehicle wells, so each mechanism can be
read against its own untreated distribution instead.

Four rules are compared, all on the same data, and all four are reported:
  raw       argmax p_m                        (the frozen rule)
  logratio  argmax log p_m - log median_ctrl(p_m)
  z         argmax (log p_m - mean_ctrl log p_m) / sd_ctrl log p_m
  pct       argmax mid-rank percentile of p_m among controls (ties: raw p)

A control is never its own reference (leave one out). The three sets:
  v3 Charlesworth (blind when scored): 29 treated, 23 genotype-matched untreated
     preparations at 10-14 DIV; answer g_nmda.
  v2 Doorn Dynasore (blind when scored): 10 treated wells; controls are the
     pre-drug pairs of the same 10 wells; answer u_rel or tau_d.
  Tampere (development set): rat 42 treated / 7 control wells, human 24 / 4;
     answer key tampere_v2.
Results: results/control_referenced.json.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "control_referenced.json"
DOORN_NULL = ROOT / "results" / "control_referenced_doorn_null.json"
KEYS = ["g_na", "g_kdr", "g_ahp", "g_ampa", "g_nmda", "g_gaba", "g_tonic_inh", "tau_d",
        "u_rel", "i_drive"]


def rule_raw(x, ref):
    return x


def rule_logratio(x, ref):
    return np.log(x) - np.log(np.median(ref, axis=0))


def rule_z(x, ref):
    lr = np.log(ref)
    return (np.log(x) - lr.mean(0)) / (lr.std(0) + 1e-9)


def rule_pct(x, ref):
    below = (ref < x).sum(0) + 0.5 * (ref == x).sum(0)
    return below / len(ref) + 1e-6 * x


RULES = {"raw": rule_raw, "logratio": rule_logratio, "z": rule_z, "pct": rule_pct}


def top1(rule, x, ref) -> str:
    return KEYS[int(np.argmax(rule(np.asarray(x, float), np.asarray(ref, float))))]


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p that the treated hit rate exceeds the control rate."""
    k, n = a + b, n1 + n2
    def h(i):
        return math.comb(n1, i) * math.comb(n2, k - i) / math.comb(n, k)
    return float(sum(h(i) for i in range(a, min(n1, k) + 1)))


def score(treated, controls, accept, ref_for_treated=None) -> dict:
    """treated/controls: lists of p vectors; accept: set of answer keys."""
    C = np.asarray(controls, float)
    R = C if ref_for_treated is None else np.asarray(ref_for_treated, float)
    out = {}
    for name, rule in RULES.items():
        t = [top1(rule, x, R) for x in treated]
        c = [top1(rule, C[i], np.delete(C, i, 0)) for i in range(len(C))]
        th, ch = sum(k in accept for k in t), sum(k in accept for k in c)
        out[name] = {"treated_hits": th, "n_treated": len(t), "control_hits": ch,
                     "n_control": len(c), "fisher_p": fisher_greater(th, len(t), ch, len(c)),
                     "treated_top1": t, "control_top1": c}
    return out


def v3_set() -> dict:
    d = json.loads((ROOT / "results/v3/results.json").read_text())
    preps = d["A_blind"]["early"]["preps"]
    T = [p["p_active"] for p in preps if p["kind"] == "treated"]
    C = [p["p_active"] for p in preps if p["kind"] == "null" and p["genotype"] in ("WT", "GluR1")]
    return score(T, C, {"g_nmda"})


def tampere_sets() -> dict:
    txt = (ROOT / "PREREGISTRATION_v2.md").read_text(encoding="utf-8")
    spec = json.loads(re.search(r"```json\n(.*?)\n```", txt, re.S).group(1))
    key = {k: set(v["accept"]) for k, v in spec["answer_keys"]["tampere_v2"].items()
           if isinstance(v, dict) and "accept" in v}
    wells = json.loads((ROOT / "results/v2/results.json").read_text())["C_tampere"]["wells"]
    out = {}
    for sp, label in (("rat", "tampere_rat"), ("hPSC", "tampere_human")):
        C = np.array([w["p_active"] for w in wells if w["species"] == sp and w["compound"] == "Control"])
        tw = [w for w in wells if w["species"] == sp and w["compound"] in key]
        res = {}
        for name, rule in RULES.items():
            t = [top1(rule, w["p_active"], C) for w in tw]
            hits = sum(k in key[w["compound"]] for k, w in zip(t, tw))
            per = {}
            for k, w in zip(t, tw):
                per.setdefault(w["compound"], [0, 0])
                per[w["compound"]][0] += k in key[w["compound"]]
                per[w["compound"]][1] += 1
            res[name] = {"treated_hits": hits, "n_treated": len(tw), "per_compound": per,
                         "n_control": len(C)}
        out[label] = res
    return out


def doorn_null_vectors() -> list[dict]:
    import evaluate_v2 as E
    import null_controls_v2 as N
    from hodgkins_razor import nde
    twin = nde.Twin.load(ROOT / "models/twin_v2_grid12", device="cuda")
    wells = E.by_well(E.score_pairs(twin, N.null_pairs(), 4000))
    return [{"well": w["well"], "p_active": w["p_active"]} for w in wells]


def doorn_set(cached: bool) -> dict:
    if cached and DOORN_NULL.exists():
        null = json.loads(DOORN_NULL.read_text())
    else:
        null = doorn_null_vectors()
        DOORN_NULL.write_text(json.dumps(null, indent=1))
    treated = json.loads((ROOT / "results/v2/results.json").read_text())["A_doorn"]["wells"]
    return score([w["p_active"] for w in treated], [w["p_active"] for w in null],
                 {"u_rel", "tau_d"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    res = {"note": ("Post hoc. Four rules compared on data already seen; all four reported. "
                    "Controls are left out of their own reference."),
           "v3_charlesworth": v3_set(), "v2_doorn": doorn_set(args.cached)}
    res.update(tampere_sets())
    OUT.write_text(json.dumps(res, indent=1))
    for s, r in res.items():
        if s == "note":
            continue
        line = "  ".join(f"{k} {v['treated_hits']}/{v['n_treated']}"
                         + (f" (ctrl {v['control_hits']}/{v['n_control']}, p {v['fisher_p']:.3f})"
                            if "control_hits" in v else "") for k, v in r.items())
        print(f"{s:15s} {line}")


if __name__ == "__main__":
    main()
