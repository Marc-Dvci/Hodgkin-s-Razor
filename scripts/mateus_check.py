"""The chip twin's readout prediction, scored on recorded microchannel chips.

    python scripts/mateus_check.py

The twin predicts, from simulation alone (`scripts/chip_study.py`), that
electrodes inside the microchannels resolve a chip's directionality and
electrodes under the chambers do not. PREREGISTRATION_v2.md fixes the two
statistics, the groups and the thresholds before this script reads a single
recorded chip. This script computes, for every recording of Mateus et al.
(2024):

* the channel statistic: the share of propagating events that run the
  channel's dominant direction (`mateus.propagation`);
* the chamber statistic: the unsigned asymmetry of the two chambers'
  population-rate cross-correlation (`mateus.chamber_asymmetry`),

and scores each by how well it separates the diode designs (Rams, Arrows) from
straight channels. Tesla designs are reported alongside.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import mateus as M

ROOT = pathlib.Path(__file__).resolve().parents[1]


def auroc(score, label) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_bootstrap(score, label, cluster, n: int = 4000, seed: int = 0) -> list[float]:
    """95 percent interval of the AUROC, resampling chips rather than recordings."""
    rng = np.random.default_rng(seed)
    score, label, cluster = map(np.asarray, (score, label, cluster))
    ids = np.unique(cluster)
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, size=ids.size, replace=True)
        idx = np.concatenate([np.flatnonzero(cluster == c) for c in pick])
        a = auroc(score[idx], label[idx])
        if np.isfinite(a):
            vals.append(a)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main() -> None:
    md = ROOT / "PREREGISTRATION_v2.md"
    rec = (ROOT / "PREREGISTRATION_v2.sha256").read_text().split()[0]
    if hashlib.sha256(md.read_bytes()).hexdigest() != rec:
        raise SystemExit("PREREGISTRATION_v2.md does not match its recorded hash")
    spec = json.loads(re.search(r"```json\n(.*?)\n```", md.read_text(encoding="utf-8"),
                                re.S).group(1))["chips"]

    rows = []
    for path in M.recordings():
        r = M.load(path)
        prop = M.propagation(r)
        asym = M.chamber_asymmetry(r)
        rows.append({"file": path.name, "experiment": r.experiment, "design": r.design,
                     "chip": f"{r.experiment}/{r.design}/{r.chip}", "div": r.div,
                     "events": prop["events"], "dominant_share": prop["dominant_share"],
                     "chamber_asymmetry": asym["asymmetry"],
                     "rate_upper": asym["rate_upper"], "rate_lower": asym["rate_lower"],
                     "unreadable_electrodes": len(r.unreadable or [])})
        print(f"  {r.design:9s} {path.name[:44]:44s} events {prop['events']:6d} "
              f"share {prop['dominant_share']:.3f}  chamber asym {asym['asymmetry']:.3f}",
              flush=True)

    diode = set(spec["diode_designs"])
    straight = set(spec["straight_designs"])
    min_events = spec["min_propagation_events"]
    use = [r for r in rows if r["design"] in diode | straight and r["events"] >= min_events]
    y = [r["design"] in diode for r in use]
    cl = [r["chip"] for r in use]
    out = {"n_recordings": len(rows), "n_scored": len(use),
           "n_chips_scored": len(set(cl)),
           "excluded_low_events": [r["file"] for r in rows
                                   if r["design"] in diode | straight
                                   and r["events"] < min_events]}
    for stat in ("dominant_share", "chamber_asymmetry"):
        s = [r[stat] for r in use]
        a = auroc(s, y)
        out[stat] = {"auroc": a, "ci95_by_chip": cluster_bootstrap(s, y, cl),
                     "threshold": spec["thresholds"][stat],
                     "prediction": spec["predictions"][stat],
                     "met": (a >= spec["thresholds"][stat]) if spec["predictions"][stat] == "separates"
                     else (a < spec["thresholds"][stat])}
    by_design = {}
    for r in rows:
        d = by_design.setdefault(r["design"], {"n": 0, "share": [], "asym": []})
        d["n"] += 1
        if r["events"] >= min_events:
            d["share"].append(r["dominant_share"])
        d["asym"].append(r["chamber_asymmetry"])
    out["by_design"] = {k: {"n": v["n"],
                            "dominant_share_median": float(np.nanmedian(v["share"])) if v["share"] else float("nan"),
                            "chamber_asymmetry_median": float(np.nanmedian(v["asym"]))}
                        for k, v in by_design.items()}
    out["rows"] = rows
    dest = ROOT / "results" / "v2" / "mateus.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_scored", "n_chips_scored", "dominant_share",
                                          "chamber_asymmetry", "by_design")}, indent=1))


if __name__ == "__main__":
    main()
