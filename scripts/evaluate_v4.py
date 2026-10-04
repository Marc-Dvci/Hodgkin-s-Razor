"""Score the fourth pre-registration: the chip twin on a four-compartment chip.

    python scripts/evaluate_v4.py

Reads PREREGISTRATION_v4.md, checks it against its recorded hash and the
input files against their recorded digests, then computes on the recordings of
Lassers et al. (2023) exactly the statistics `scripts/brewer_prediction.py`
computed on simulated chips, and scores them against the twin's predictive
intervals at the number of units that pass the inclusion rules.

Writes results/v4/results.json and results/v4/RESULTS.md.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import brewer as B
from hodgkins_razor import chip as C

ROOT = pathlib.Path(__file__).resolve().parents[1]


def spec() -> dict:
    md = ROOT / "PREREGISTRATION_v4.md"
    rec = (ROOT / "PREREGISTRATION_v4.sha256").read_text().split()[0]
    if hashlib.sha256(md.read_bytes()).hexdigest() != rec:
        raise SystemExit("PREREGISTRATION_v4.md does not match its recorded hash")
    return json.loads(re.search(r"```json\n(.*?)\n```", md.read_text(encoding="utf-8"),
                                re.S).group(1))


def units(cultures: list[dict], rules: dict) -> tuple[list[dict], list[dict]]:
    """Boundary units and (boundary, direction) units of one condition."""
    bnd, home = [], []
    for c in cultures:
        for b, (up, down) in B.BOUNDARIES.items():
            wu, wd = c["well"].get(up, np.zeros(0)), c["well"].get(down, np.zeros(0))
            axons = [a for a in c["axons"] if a["boundary"] == b
                     and a.get("up", np.zeros(0)).size > 0]
            pools = {d: np.sort(np.concatenate([a["up"] for a in axons if a["direction"] == d]
                                               or [np.zeros(0)]))
                     for d in ("ff", "fb")}
            n_ff = sum(a["direction"] == "ff" for a in axons)
            spikes = pools["ff"].size + pools["fb"].size
            row = {"culture": c["culture"], "boundary": b, "axons": len(axons),
                   "axons_ff_n": n_ff, "spikes_ff": int(pools["ff"].size),
                   "spikes_fb": int(pools["fb"].size),
                   "well_spikes_up": int(wu.size), "well_spikes_down": int(wd.size)}
            wells_ok = wu.size >= rules["min_well_spikes"] and wd.size >= rules["min_well_spikes"]
            row["included"] = bool(wells_ok and len(axons) >= rules["min_axons"]
                                   and spikes >= rules["min_axon_spikes"])
            if row["included"]:
                row["asym"] = float(C.rate_xcorr(wu, wd, B.DURATION)[2])
                row["traffic_ff"] = pools["ff"].size / spikes
                row["axons_ff"] = n_ff / len(axons)
            bnd.append(row)
            if not wells_ok:
                continue
            for d, own, other in (("ff", wu, wd), ("fb", wd, wu)):
                ax = pools[d]
                if ax.size >= rules["min_axon_spikes"]:
                    h = (C.rate_xcorr(own, ax, B.DURATION)[0]
                         - C.rate_xcorr(other, ax, B.DURATION)[0])
                    home.append({"culture": c["culture"], "boundary": b, "direction": d,
                                 "spikes": int(ax.size), "home": float(h)})
    return bnd, home


def cluster_ci(values_by_culture: dict, fn, n: int = 4000, seed: int = 0) -> list[float]:
    """95 percent interval of fn over cultures resampled with replacement."""
    rng = np.random.default_rng(seed)
    ids = list(values_by_culture)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        rows = [r for i in pick for r in values_by_culture[ids[i]]]
        v = fn(rows)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def by_culture(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["culture"], []).append(r)
    return out


def rho(rows: list[dict], y: str) -> float:
    if len(rows) < 4:
        return float("nan")
    return float(spearmanr([r["asym"] for r in rows], [r[y] for r in rows])[0])


def share(rows: list[dict]) -> float:
    return float(np.mean([r["home"] > 0 for r in rows])) if rows else float("nan")


def permutation_p(rows: list[dict], y: str, n: int = 20000, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    x = np.array([r["asym"] for r in rows])
    t = np.array([r[y] for r in rows])
    obs = abs(spearmanr(x, t)[0])
    hits = sum(abs(spearmanr(x, rng.permutation(t))[0]) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def interval_at(pred: dict, n: int, lo: int, hi: int) -> dict:
    return pred[str(min(max(n, lo), hi))]


def score(condition: str, sp: dict, pred: dict) -> dict:
    cultures = B.load(condition)
    bnd, home = units(cultures, sp["inclusion"])
    use = [r for r in bnd if r["included"]]
    out = {"condition": condition, "cultures": len(cultures),
           "boundary_units": len(bnd), "boundary_units_scored": len(use),
           "home_units_scored": len(home)}
    for y in ("traffic_ff", "axons_ff"):
        r = rho(use, y)
        pi = interval_at(pred["predictive_by_n"], len(use), 12, 36)[f"asym_vs_{y}"]["pi90"]
        out[f"asym_vs_{y}"] = {"spearman": r, "ci95_by_culture":
                               cluster_ci(by_culture(use), lambda rows, y=y: rho(rows, y)),
                               "permutation_p": permutation_p(use, y) if len(use) >= 4 else None,
                               "twin_pi90": pi,
                               "inside": bool(pi[0] <= r <= pi[1]) if np.isfinite(r) else None}
    s = share(home)
    pi = interval_at(pred["home"]["predictive_by_n"], len(home), 12, 72)["pi90"]
    k = sum(r["home"] > 0 for r in home)
    out["home"] = {"share_positive": s, "k": int(k), "n": len(home),
                   "median_home_index": float(np.median([r["home"] for r in home])) if home else None,
                   "ci95_by_culture": cluster_ci(by_culture(home), share),
                   "sign_test_p_vs_half": float(binomtest(k, len(home), 0.5,
                                                          alternative="greater").pvalue)
                   if home else None,
                   "twin_pi90": pi, "at_or_above_lower_bound": bool(s >= pi[0]) if home else None}
    out["units"] = bnd
    out["home_units"] = home
    return out


def main() -> None:
    sp = spec()
    pinned = sp["data_sha256"]
    now = B.digest()
    for name, h in pinned.items():
        if now.get(name) != h:
            raise SystemExit(f"{name} differs from the file digested in the pre-registration")
    pred = json.loads((ROOT / sp["prediction_file"]).read_text())
    if hashlib.sha256((ROOT / sp["prediction_file"]).read_bytes()).hexdigest() != sp["prediction_sha256"]:
        raise SystemExit("the prediction file changed after the pre-registration")
    res = {"preregistration": (ROOT / "PREREGISTRATION_v4.sha256").read_text().split()[0]}
    res["primary"] = score("nostim", sp, pred)
    p = res["primary"]
    res["verdicts"] = {
        "P1_home_share": {"observed": p["home"]["share_positive"],
                          "bar": f">= {p['home']['twin_pi90'][0]:.3f} (twin 90% lower bound)",
                          "met": p["home"]["at_or_above_lower_bound"]},
        "P2_asym_vs_traffic": {"observed": p["asym_vs_traffic_ff"]["spearman"],
                               "bar": "inside the twin's 90% interval "
                                      f"{[round(v, 3) for v in p['asym_vs_traffic_ff']['twin_pi90']]}",
                               "met": p["asym_vs_traffic_ff"]["inside"]},
    }
    res["secondary"] = {c: score(c, sp, pred) for c in sp["secondary_conditions"]}
    dest = ROOT / "results" / "v4"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdicts"], indent=1))
    for c in ["nostim"] + sp["secondary_conditions"]:
        r = res["primary"] if c == "nostim" else res["secondary"][c]
        print(c, {k: r[k] for k in ("boundary_units_scored", "home_units_scored")},
              "home", round(r["home"]["share_positive"], 3) if r["home"]["n"] else None,
              "rho traffic", round(r["asym_vs_traffic_ff"]["spearman"], 3),
              "rho axons", round(r["asym_vs_axons_ff"]["spearman"], 3))


if __name__ == "__main__":
    main()
