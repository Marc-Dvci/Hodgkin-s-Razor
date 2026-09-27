"""Every number of the third pre-registration, from one command.

    python scripts/evaluate_v3.py

Reads the answer key, the metrics and the operating points from the JSON block
in PREREGISTRATION_v3.md, checks that file against its recorded hash and the
frozen models against the digests it lists, and refuses to run if either
changed. Writes results/v3/results.json; `scripts/render_v3.py` turns it into
results/v3/RESULTS.md.

A. The blind test: chronic APV on sister arrays (Charlesworth et al. 2015),
   every treated preparation scored against untreated sister preparations
   recorded at the same ages, so chance is what the twin does when nothing was
   applied.
B. The same comparison for the unpaired twin, whose ranking is known to carry
   a fixed preference (version 2 audit), and for the published estimator of
   Doorn et al.
C. The guard on the recorded pairs, and its three pre-registered tests.
D. Simulation-side calibration of the frozen twin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import charlesworth as C, features as F, nde, params as P
from hodgkins_razor import shift as SH, simulator as S
from evaluate_v2 import (Guard, _rate, auroc, calibrate_ppc, model_digest, readable,
                         sha256, shuffle_events, simulation_side, wilson)
from make_bank import sister_drift
from train import load_bank

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def read_prereg() -> dict:
    md = ROOT / "PREREGISTRATION_v3.md"
    rec = (ROOT / "PREREGISTRATION_v3.sha256").read_text().split()[0]
    if sha256(md) != rec:
        raise SystemExit("PREREGISTRATION_v3.md does not match its recorded hash")
    block = re.search(r"```json\n(.*?)\n```", md.read_text(encoding="utf-8"), re.S)
    spec = json.loads(block.group(1))
    for name, digest in spec["frozen_models"].items():
        got = model_digest(ROOT / name)
        if got != digest:
            raise SystemExit(f"{name} changed since the pre-registration "
                             f"({got[:12]} != {digest[:12]})")
    for name, digest in spec["frozen_files"].items():
        if sha256(ROOT / name) != digest:
            raise SystemExit(f"{name} changed since the pre-registration")
    spec["digest"] = rec
    return spec


# ------------------------------------------------------------------ scoring
def score_windows(twin, pairs, n_samples: int, key: str, min_events: int,
                  min_elec: int) -> list[dict]:
    """Presence per window, and the effect of the answer-key mechanism if it moved."""
    j = SHIFT_KEYS.index(key)
    rows = []
    for p in pairs:
        if not readable(p.baseline, min_events, min_elec):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        post = twin.posterior(xb, xt, n_samples=n_samples)
        eff = post.get("effect_given_active", post["delta"])
        rows.append({"prep": p.prep, "div": p.div, "kind": p.kind,
                     "genotype": p.genotype, "quadrant": p.quadrant, "window": p.window,
                     "p_active": post["p_active"].tolist(),
                     "effect_key": float(np.median(eff[:, j]))})
    return rows


def unpaired_windows(model, pairs, n_samples: int, min_events: int,
                     min_elec: int) -> list[dict]:
    rows = []
    for p in pairs:
        if not readable(p.baseline, min_events, min_elec):
            continue
        post = model.posterior(F.compute(p.baseline, p.n_elec, p.duration),
                               F.compute(p.treated, p.n_elec, p.duration),
                               n_samples=n_samples)
        med = np.median(post["delta"], axis=0)
        rows.append({"prep": p.prep, "div": p.div, "kind": p.kind,
                     "genotype": p.genotype, "quadrant": p.quadrant, "window": p.window,
                     "p_active": post["p_active"].tolist(), "effect": med.tolist()})
    return rows


def by_prep(rows: list[dict], lo: float, hi: float, key: str) -> list[dict]:
    """One record per preparation: presence averaged over its pairs in the age range."""
    j = SHIFT_KEYS.index(key)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if lo <= r["div"] <= hi:
            groups.setdefault((r["prep"], r["kind"]), []).append(r)
    out = []
    for (prep, kind), rs in groups.items():
        pa = np.mean([r["p_active"] for r in rs], axis=0)
        eff = [r["effect_key"] if "effect_key" in r else r["effect"][j] for r in rs]
        out.append({"prep": prep, "kind": kind, "genotype": rs[0]["genotype"],
                    "n_windows": len(rs), "divs": sorted({r["div"] for r in rs}),
                    "p_active": pa.tolist(), "p_key": float(pa[j]),
                    "top1": SHIFT_KEYS[int(np.argmax(pa))],
                    "called_key": bool(pa[j] > 0.5),
                    "effect_key": float(np.median(eff))})
    return out


def bootstrap_auroc(s1: np.ndarray, s0: np.ndarray, n: int = 4000,
                    seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        a = rng.choice(s1, size=s1.size, replace=True)
        b = rng.choice(s0, size=s0.size, replace=True)
        vals.append(auroc(np.r_[a, b], np.r_[np.ones(a.size), np.zeros(b.size)]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def fisher_greater(k1: int, n1: int, k0: int, n0: int) -> float:
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1])


def contrast(preps: list[dict], key: str, direction: str) -> dict:
    """Treated preparations against null preparations, on one answer-key mechanism."""
    t = [p for p in preps if p["kind"] == "treated"]
    z = [p for p in preps if p["kind"] == "null"]
    if not t or not z:
        return {"n_treated": len(t), "n_null": len(z)}
    s1 = np.array([p["p_key"] for p in t])
    s0 = np.array([p["p_key"] for p in z])
    k1 = sum(p["top1"] == key for p in t)
    k0 = sum(p["top1"] == key for p in z)
    hits = [p for p in t if p["top1"] == key]
    sign = [(p["effect_key"] < 0) if direction == "down" else (p["effect_key"] > 0)
            for p in hits]
    det1 = np.array([max(p["p_active"]) for p in t])
    det0 = np.array([max(p["p_active"]) for p in z])
    return {"n_treated": len(t), "n_null": len(z),
            "auroc_key": auroc(np.r_[s1, s0], np.r_[np.ones(s1.size), np.zeros(s0.size)]),
            "auroc_key_ci95": bootstrap_auroc(s1, s0),
            "median_p_key_treated": float(np.median(s1)),
            "median_p_key_null": float(np.median(s0)),
            "top1_treated": k1 / len(t), "top1_treated_hits": k1,
            "top1_treated_ci95": wilson(k1, len(t)),
            "top1_null": k0 / len(z), "top1_null_hits": k0,
            "top1_null_ci95": wilson(k0, len(z)),
            "top1_fisher_p": fisher_greater(k1, len(t), k0, len(z)),
            "called_key_treated": float(np.mean([p["called_key"] for p in t])),
            "called_key_null": float(np.mean([p["called_key"] for p in z])),
            "direction_agreement": float(np.mean(sign)) if sign else float("nan"),
            "direction_n": len(sign),
            "detection_auroc": auroc(np.r_[det1, det0],
                                     np.r_[np.ones(det1.size), np.zeros(det0.size)]),
            "top1_treated_counts": _counts([p["top1"] for p in t]),
            "top1_null_counts": _counts([p["top1"] for p in z])}


def matched(preps: list[dict]) -> list[dict]:
    """Treated preparations, and the null preparations of the treated genotypes.

    The null group holds knockouts the treated group lacks (PSD-95, PSD-93 and
    SAP102 among them, all NMDA-receptor scaffolds), which could shift where a
    culture sits and how readable `g_nmda` is. The primary compares like with
    like; the pooled contrast is secondary.
    """
    genes = {C._gene(p["genotype"]) for p in preps if p["kind"] == "treated"}
    return [p for p in preps if p["kind"] == "treated" or C._gene(p["genotype"]) in genes]


def by_genotype(preps: list[dict], key: str, direction: str) -> dict:
    """Treated preparations of one genotype against null preparations of the same one."""
    genes = sorted({C._gene(p["genotype"]) for p in preps if p["kind"] == "treated"})
    return {g: contrast([p for p in preps if C._gene(p["genotype"]) == g], key, direction)
            for g in genes}


def profile(preps: list[dict]) -> dict:
    """Exploratory: every mechanism's presence, treated against matched null.

    Days of NMDA blockade can recruit compensation, so `g_nmda` may move with a
    second mechanism. Reported for all mechanisms, with no bar.
    """
    t = [p for p in preps if p["kind"] == "treated"]
    z = [p for p in preps if p["kind"] == "null"]
    if not t or not z:
        return {}
    y = np.r_[np.ones(len(t)), np.zeros(len(z))]
    out = {}
    for j, k in enumerate(SHIFT_KEYS):
        s = np.r_[[p["p_active"][j] for p in t], [p["p_active"][j] for p in z]]
        out[k] = {"auroc": auroc(s, y),
                  "median_treated": float(np.median(s[:len(t)])),
                  "median_null": float(np.median(s[len(t):]))}
    return out


def _counts(xs: list[str]) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def canalization(rows: list[dict], key: str, lo: float, hi: float) -> dict:
    """Does the reading of the key mechanism fade with age in treated preparations?

    Per preparation, the mean presence in the early and in the late range;
    the paired difference is tested with a one-sided Wilcoxon signed-rank test.
    """
    from scipy.stats import wilcoxon
    early = {p["prep"]: p["p_key"] for p in by_prep(rows, lo, hi, key) if p["kind"] == "treated"}
    late = {p["prep"]: p["p_key"] for p in by_prep(rows, hi + 0.5, 99, key) if p["kind"] == "treated"}
    both = sorted(set(early) & set(late))
    if len(both) < 5:
        return {"n": len(both)}
    d = np.array([early[k] - late[k] for k in both])
    return {"n": len(both), "median_early": float(np.median([early[k] for k in both])),
            "median_late": float(np.median([late[k] for k in both])),
            "wilcoxon_p_early_gt_late": float(wilcoxon(d, alternative="greater")[1])}


# ------------------------------------------------------------------- guard
def variant_pairs_sister(view: str, domain_file: str, drift: float, n: int,
                         duration: float, transient: float, seed: int) -> list[tuple]:
    """Sister pairs from receptor kinetics the twin has no parameter for."""
    from fit_domain import GaussianProposal
    dom = json.loads((ROOT / domain_file).read_text())
    rng = np.random.default_rng(seed)
    variant = S.Simulator(kinetics={"TAU_AMPA": 20.0, "TAU_GABA": 40.0})
    theta_c = GaussianProposal.load(dom["views"][view]["proposal"]).draw(n, rng)
    delta, active = SH.sample_shift(n, rng, theta_c=theta_c)
    theta_t, _, _ = SH.apply_shift(theta_c, delta, active)
    theta_t = sister_drift(theta_t, drift, rng)
    res = variant.run(SH.interleave(theta_c, theta_t), duration_s=duration,
                      transient_s=transient, seed=seed, pair=False)
    out = []
    for k in range(n):
        eb, nb = S.view_events(res.raw_events(2 * k), view)
        et, _ = S.view_events(res.raw_events(2 * k + 1), view)
        if eb.shape[0] < 50:
            continue
        out.append((F.compute(eb, nb, duration), F.compute(et, nb, duration)))
    return out


def guard_pairs(guard: Guard, pairs, seed: int, min_events: int, min_elec: int) -> list[dict]:
    rows = []
    for k, p in enumerate(pairs):
        if not readable(p.baseline, min_events, min_elec):
            continue
        g = guard(F.compute(p.baseline, p.n_elec, p.duration),
                  F.compute(p.treated, p.n_elec, p.duration), seed=int(seed + k))
        rows.append({"prep": p.prep, "div": p.div, "kind": p.kind, "fires": g["fires"],
                     "fires_typicality": g["fires_typicality"], "fires_ppc": g["fires_ppc"]})
    return rows


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="A,B,C,D")
    args = ap.parse_args()
    spec = read_prereg()
    ops = spec["operating_points"]
    test = spec["blind_test"]
    key, direction = test["accept"], test["direction"]
    lo, hi = test["early_divs"]
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["prereg_sha256"] = spec["digest"]
    t0 = time.time()
    view = spec["view"]
    twin = nde.Twin.load(ROOT / spec["twin"], device="cuda")
    kw = {"min_events": ops["min_events"], "min_elec": ops["min_electrodes"]}
    sections = set(args.sections.split(","))

    pairs = C.load(kinds=("treated", "null"), min_div=lo, max_div=99.0,
                   window_s=ops["window_s"], n_windows=ops["windows_per_recording"])
    print(f"{len(pairs)} quadrant windows from "
          f"{len({(p.prep, p.div, p.kind) for p in pairs})} sister pairs", flush=True)

    if "A" in sections:
        rows = score_windows(twin, pairs, ops["posterior_samples"], key, **kw)
        early, late = by_prep(rows, lo, hi, key), by_prep(rows, hi + 0.5, 99.0, key)
        res["A_blind"] = {
            # Primary and co-primary: genotype-matched nulls.
            "early": {"preps": early, "metrics": contrast(matched(early), key, direction)},
            "early_pooled_null": contrast(early, key, direction),
            "late": {"preps": late, "metrics": contrast(matched(late), key, direction)},
            "late_pooled_null": contrast(late, key, direction),
            "by_genotype": by_genotype(early, key, direction),
            "profile_exploratory": profile(matched(early)),
            "canalization": canalization(rows, key, lo, hi),
            "n_windows_scored": len(rows)}
        m = res["A_blind"]["early"]["metrics"]
        print(f"A done ({time.time() - t0:.0f}s): AUROC {m['auroc_key']:.3f} "
              f"{m['auroc_key_ci95']}, top-1 {m['top1_treated']:.2f} vs null "
              f"{m['top1_null']:.2f} (p {m['top1_fisher_p']:.3g})", flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    if "B" in sections:
        unp = nde.UnpairedTwin.load(ROOT / spec["unpaired"], device="cuda")
        urows = unpaired_windows(unp, pairs, ops["posterior_samples"], **kw)
        ue = by_prep(urows, lo, hi, key)
        res["B_comparators"] = {"unpaired": {"preps": ue, "metrics": contrast(matched(ue), key, direction),
                                             "pooled_null": contrast(ue, key, direction)}}
        pa = ROOT / "results" / "prior_art_doorn_charlesworth.json"
        if pa.exists():
            res["B_comparators"]["prior_art"] = prior_art(json.loads(pa.read_text()),
                                                          key, direction, lo, hi)
        print(f"B done ({time.time() - t0:.0f}s)", flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    if "C" in sections:
        sim = S.Simulator()
        bank = load_bank(ROOT / spec["bank"], view)
        keep = bank["domain"] == list(S.VIEWS).index(view)
        bank = {k: a[keep] for k, a in bank.items()}
        idx_val = np.load(ROOT / spec["twin"] / "val_index.npy")
        cal = calibrate_ppc(twin, sim, bank, idx_val, ops["window_s"], 5.0,
                            ops["ppc_calibration_records"], ops["predictive_draws"],
                            ops["ppc_quantile"], view)
        guard = Guard(twin, sim, cal["threshold"], ops["window_s"], 5.0,
                      ops["predictive_draws"])
        g = spec["guard"]
        sel = [p for p in pairs if lo <= p.div <= hi and p.window < g["windows_per_quadrant"]]
        grows = guard_pairs(guard, sel, 50000, **kw)
        rng = np.random.default_rng(3)
        pick = rng.choice(idx_val, size=min(g["n_cases"], len(idx_val)), replace=False)
        hold = _rate([guard(bank["x_base"][i], bank["x_treat"][i], seed=int(1e4 + k))
                      for k, i in enumerate(pick)])
        var = variant_pairs_sister(view, spec["domain_file"], spec["drift"], g["n_cases"],
                                   ops["window_s"], 5.0, 11)
        vr = _rate([guard(xb, xt, seed=int(2e4 + k)) for k, (xb, xt) in enumerate(var)])
        usable = [p for p in sel if readable(p.baseline, **{"min_events": kw["min_events"],
                                                           "min_elec": kw["min_elec"]})]
        rng.shuffle(usable)
        shuf = []
        for k, p in enumerate(usable[:g["n_cases"]]):
            xb = F.compute(shuffle_events(p.baseline, p.duration, rng), p.n_elec, p.duration)
            xt = F.compute(shuffle_events(p.treated, p.duration, rng), p.n_elec, p.duration)
            shuf.append(guard(xb, xt, seed=int(3e4 + k)))
        per_prep: dict[tuple, list[bool]] = {}
        for r in grows:
            per_prep.setdefault((r["prep"], r["kind"]), []).append(r["fires"])
        outside = {f"{k[0]}|{k[1]}": bool(np.mean(v) > 0.5) for k, v in per_prep.items()}
        res["C_guard"] = {"threshold_ppc": cal["threshold"],
                          "threshold_typicality": twin.meta.get("typicality_threshold"),
                          "bank_holdout": hold, "variant_kinetics": vr,
                          "shuffled_real": _rate(shuf),
                          "recorded_fire_rate": {kind: float(np.mean([r["fires"] for r in grows
                                                                      if r["kind"] == kind]))
                                                 for kind in ("treated", "null")},
                          "outside_by_prep": outside}
        if "A_blind" in res:
            early = res["A_blind"]["early"]["preps"]
            inside = [p for p in early if not outside.get(f"{p['prep']}|{p['kind']}", False)]
            res["C_guard"]["inside_metrics"] = contrast(matched(inside), key, direction)
        print(f"C done ({time.time() - t0:.0f}s)", flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    if "D" in sections:
        bank = load_bank(ROOT / spec["bank"], view)
        keep = bank["domain"] == list(S.VIEWS).index(view)
        bank = {k: a[keep] for k, a in bank.items()}
        idx_val = np.load(ROOT / spec["twin"] / "val_index.npy")
        res["D_simulation"] = simulation_side(twin, bank, idx_val)
        print(f"D done ({time.time() - t0:.0f}s)", flush=True)
        res_path.write_text(json.dumps(res, indent=1))
    print("wrote", res_path)


def prior_art(d: dict, key: str, direction: str, lo: float, hi: float) -> dict:
    """Doorn et al.'s estimator: z of each parameter's shift between sisters."""
    names = d["params"]
    cand = {"g_Na": "g_na", "g_K": "g_kdr", "g_AHP": "g_ahp", "g_AMPA": "g_ampa",
            "g_NMDA": "g_nmda", "tau_D": "tau_d", "U_STD": "u_rel", "U_asyn": "u_asyn"}
    idx = [names.index(k) for k in cand]
    groups: dict[tuple, list] = {}
    genotype: dict[tuple, str] = {}
    for r in d["rows"]:
        if not (lo <= r["div"] <= hi) or not r.get("baseline") or not r.get("treated"):
            continue
        genotype[(r["prep"], r["kind"])] = r["genotype"]
        mb, mt = np.array(r["baseline"]["median"]), np.array(r["treated"]["median"])
        sb, st = np.array(r["baseline"]["sd"]), np.array(r["treated"]["sd"])
        groups.setdefault((r["prep"], r["kind"]), []).append((mt - mb) / np.sqrt(sb ** 2 + st ** 2 + 1e-12))
    j = names.index("g_NMDA")
    preps = []
    for (prep, kind), zs in groups.items():
        z = np.mean(zs, axis=0)
        top = max(idx, key=lambda i: abs(z[i]))
        # Their design has no presence probability; the score for the key
        # mechanism is the size of its standardised shift in the answer's
        # direction.
        s = -z[j] if direction == "down" else z[j]
        preps.append({"prep": prep, "kind": kind, "genotype": genotype[(prep, kind)],
                      "p_key": float(s),
                      "top1": cand[names[top]], "called_key": False,
                      "effect_key": float(z[j]),
                      "p_active": [float(abs(z[i])) for i in idx]})
    return {"preps": preps, "metrics": contrast(matched(preps), key, direction),
            "pooled_null": contrast(preps, key, direction)}


if __name__ == "__main__":
    main()
