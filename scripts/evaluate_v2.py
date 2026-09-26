"""Every number of the second pre-registration, from one command.

    python scripts/evaluate_v2.py

Reads the answer keys, the metrics and the operating points from the JSON
block in PREREGISTRATION_v2.md, checks that file against its recorded hash and
the frozen models against the hashes it lists, and refuses to run if either
changed. Writes results/v2/results.json; `scripts/render_v2.py` turns it into
results/v2/RESULTS.md and the report tables.

Sections, in the order of the pre-registration:

A. Doorn et al. Dynasore wells, the blind test.
B. The chip readout prediction, scored on Mateus et al. chips
   (`scripts/mateus_check.py` writes its own file; this script reads it).
C. Tampere wells, the development set, scored for the fifth time.
D. The transfer test on Tampere: does the fingerprint of a compound learned on
   one species name the compound on the other?
E. Simulation-side calibration and the guard.
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

from hodgkins_razor import features as F, nde, params as P, ppc, shift as SH
from hodgkins_razor import simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


# ------------------------------------------------------------ pre-registration
def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(path: pathlib.Path) -> str:
    """One hash over every file of a model directory, in name order."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.iterdir() if p.is_file()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def read_prereg() -> dict:
    md = ROOT / "PREREGISTRATION_v2.md"
    rec = (ROOT / "PREREGISTRATION_v2.sha256").read_text().split()[0]
    if sha256(md) != rec:
        raise SystemExit("PREREGISTRATION_v2.md does not match its recorded hash")
    block = re.search(r"```json\n(.*?)\n```", md.read_text(encoding="utf-8"), re.S)
    spec = json.loads(block.group(1))
    for name, digest in spec["frozen_models"].items():
        got = model_digest(ROOT / name)
        if got != digest:
            raise SystemExit(f"{name} changed since the pre-registration "
                             f"({got[:12]} != {digest[:12]})")
    spec["digest"] = rec
    return spec


# ------------------------------------------------------------------- scoring
def readable(ev: np.ndarray, min_events: int = 50, min_elec: int = 3) -> bool:
    return ev.shape[0] >= min_events and len(np.unique(ev[:, 0])) >= min_elec


def score_pairs(twin, pairs, n_samples: int) -> list[dict]:
    rows = []
    for p in pairs:
        if not readable(p.baseline):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        post = twin.posterior(xb, xt, n_samples=n_samples)
        eff = post.get("effect_given_active", post["delta"])
        rows.append({"plate": p.plate, "species": p.species, "well": p.well,
                     "compound": p.compound, "p_active": post["p_active"].tolist(),
                     "effect_med": np.median(eff, axis=0).tolist(),
                     "effect_lo": np.quantile(eff, 0.05, axis=0).tolist(),
                     "effect_hi": np.quantile(eff, 0.95, axis=0).tolist(),
                     "x_base": xb.tolist(), "x_treat": xt.tolist()})
    return rows


def by_well(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["plate"], r["well"], r["compound"]), []).append(r)
    out = []
    for (plate, well, comp), rs in groups.items():
        pa = np.mean([r["p_active"] for r in rs], axis=0)
        em = np.mean([r["effect_med"] for r in rs], axis=0)
        out.append({"plate": plate, "species": rs[0]["species"], "well": well,
                    "compound": comp, "n_windows": len(rs),
                    "p_active": pa.tolist(), "effect_med": em.tolist(),
                    "effect_lo": np.mean([r["effect_lo"] for r in rs], axis=0).tolist(),
                    "effect_hi": np.mean([r["effect_hi"] for r in rs], axis=0).tolist(),
                    "top1": SHIFT_KEYS[int(np.argmax(pa))],
                    "top2": [SHIFT_KEYS[i] for i in np.argsort(-pa)[:2]],
                    "called": bool(pa.max() > 0.5)})
    return out


def class_of_well(w: dict) -> str:
    agg = np.zeros(len(P.CLASS_NAMES))
    for j, k in enumerate(SHIFT_KEYS):
        agg[P.class_index(k)] = max(agg[P.class_index(k)], w["p_active"][j])
    return P.CLASS_NAMES[int(np.argmax(agg))]


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def mechanism_metrics(wells: list[dict], key: dict) -> dict:
    """Accuracy against an answer key {compound: {"accept": [...], "direction": ...}}."""
    scored = [w for w in wells if key.get(w["compound"], {}).get("accept")]
    controls = [w for w in wells if key.get(w["compound"], {}).get("control")]
    hit = [w["top1"] in key[w["compound"]]["accept"] for w in scored]
    hit2 = [bool(set(w["top2"]) & set(key[w["compound"]]["accept"])) for w in scored]
    cls = [class_of_well(w) == P.CLASS_OF[key[w["compound"]]["accept"][0]]
           for w in scored]
    direction = []
    for w, h in zip(scored, hit):
        want = key[w["compound"]].get("direction")
        if not h or want not in ("up", "down"):
            continue
        j = SHIFT_KEYS.index(w["top1"])
        direction.append(("up" if w["effect_med"][j] > 0 else "down") == want)
    per: dict[str, dict] = {}
    for w, h, h2, c in zip(scored, hit, hit2, cls):
        d = per.setdefault(w["compound"], {"n": 0, "top1": 0, "top2": 0, "class": 0,
                                           "species": {}})
        d["n"] += 1; d["top1"] += int(h); d["top2"] += int(h2); d["class"] += int(c)
        sp = d["species"].setdefault(w["species"], {"n": 0, "top1": 0})
        sp["n"] += 1; sp["top1"] += int(h)
    conf = np.zeros((len(SHIFT_KEYS), len(SHIFT_KEYS)), dtype=int)
    for w in scored:
        conf[SHIFT_KEYS.index(key[w["compound"]]["accept"][0]),
             SHIFT_KEYS.index(w["top1"])] += 1
    k, n = int(sum(hit)), len(scored)
    chance = float(np.mean([len(key[w["compound"]]["accept"]) / len(SHIFT_KEYS)
                            for w in scored])) if scored else float("nan")
    out = {"n_wells": n, "top1_hits": k,
           "top1_accuracy": k / max(n, 1), "top1_ci95": wilson(k, n),
           "chance": chance, "p_vs_chance": binom_sf(k, n, chance) if n else float("nan"),
           "top2_accuracy": float(np.mean(hit2)) if n else float("nan"),
           "class_accuracy": float(np.mean(cls)) if n else float("nan"),
           "direction_agreement": float(np.mean(direction)) if direction else float("nan"),
           "direction_n": len(direction),
           "called_treated": float(np.mean([w["called"] for w in scored])) if n else float("nan"),
           "called_accuracy": (float(np.mean([h for w, h in zip(scored, hit) if w["called"]]))
                               if any(w["called"] for w in scored) else float("nan")),
           "per_compound": per, "confusion": conf.tolist(),
           "confusion_labels": SHIFT_KEYS}
    if controls:
        out["n_controls"] = len(controls)
        out["control_false_mechanism_rate"] = float(np.mean([w["called"] for w in controls]))
        s = [max(w["p_active"]) for w in scored] + [max(w["p_active"]) for w in controls]
        y = [1] * len(scored) + [0] * len(controls)
        out["detection_auroc"] = auroc(np.array(s), np.array(y))
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- baselines
def unpaired_wells(model, pairs, n_samples: int) -> list[dict]:
    rows = []
    for p in pairs:
        if not readable(p.baseline):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        post = model.posterior(xb, xt, n_samples=n_samples)
        rows.append({"plate": p.plate, "species": p.species, "well": p.well,
                     "compound": p.compound, "p_active": post["p_active"].tolist(),
                     "effect_med": np.median(post["delta"], axis=0).tolist(),
                     "effect_lo": [0] * P.N_SHIFT, "effect_hi": [0] * P.N_SHIFT})
    return by_well(rows)


PRIOR_ART_MAP = {"g_Na": "g_na", "g_K": "g_kdr", "g_AHP": "g_ahp", "g_AMPA": "g_ampa",
                 "g_NMDA": "g_nmda", "tau_D": "tau_d", "U_STD": "u_rel",
                 "U_asyn": "u_asyn"}


def prior_art_wells(path: pathlib.Path) -> list[dict] | None:
    """Doorn et al.'s estimator: the parameter whose posterior median moved most,
    in units of the two posteriors' spread, among its shiftable parameters."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    names = d["params"]
    cand = [names.index(k) for k in PRIOR_ART_MAP]
    groups: dict[tuple, list] = {}
    for r in d["rows"]:
        if not r.get("baseline") or not r.get("treated"):
            continue
        mb, mt = np.array(r["baseline"]["median"]), np.array(r["treated"]["median"])
        sb, st = np.array(r["baseline"]["sd"]), np.array(r["treated"]["sd"])
        z = (mt - mb) / np.sqrt(sb ** 2 + st ** 2 + 1e-12)
        groups.setdefault((r["plate"], r["well"], r["compound"], r["species"]), []).append(z)
    out = []
    for (plate, well, comp, sp), zs in groups.items():
        z = np.mean(zs, axis=0)
        order = sorted(cand, key=lambda i: -abs(z[i]))
        out.append({"plate": plate, "well": well, "compound": comp, "species": sp,
                    "top1": PRIOR_ART_MAP[names[order[0]]],
                    "top2": [PRIOR_ART_MAP[names[i]] for i in order[:2]],
                    "direction": "up" if z[order[0]] > 0 else "down"})
    return out


def prior_art_metrics(wells: list[dict] | None, key: dict) -> dict:
    if wells is None:
        return {"available": False}
    scored = [w for w in wells if key.get(w["compound"], {}).get("accept")]
    hit = [w["top1"] in key[w["compound"]]["accept"] for w in scored]
    per: dict[str, list] = {}
    for w, h in zip(scored, hit):
        per.setdefault(w["compound"], []).append(h)
    return {"available": True, "n_wells": len(scored),
            "top1_accuracy": float(np.mean(hit)) if hit else float("nan"),
            "per_compound": {c: f"{sum(v)}/{len(v)}" for c, v in per.items()},
            "calls": [{"well": w["well"], "compound": w["compound"],
                       "top1": w["top1"], "direction": w["direction"]} for w in scored]}


def supervised_lowo(rows: list[dict], key: dict) -> dict:
    """Nearest centroid on feature changes, given the labels, leave one well out."""
    wells = {}
    for r in rows:
        k = (r["plate"], r["well"], r["compound"])
        wells.setdefault(k, []).append(nde.phi(np.array(r["x_treat"])) - nde.phi(np.array(r["x_base"])))
    items = [(k, np.mean(v, axis=0)) for k, v in wells.items()
             if key.get(k[2], {}).get("accept")]
    if len(items) < 4:
        return {"top1_accuracy": float("nan"), "n": len(items)}
    D = np.stack([v for _, v in items])
    D = (D - D.mean(0)) / (D.std(0) + 1e-9)
    lab = [key[k[2]]["accept"][0] for k, _ in items]
    ok = 0
    for i in range(len(items)):
        keep = [j for j in range(len(items)) if j != i]
        names = sorted(set(lab[j] for j in keep))
        cents = [D[[j for j in keep if lab[j] == n]].mean(0) for n in names]
        ok += names[int(np.argmin([np.linalg.norm(D[i] - c) for c in cents]))] == lab[i]
    return {"top1_accuracy": ok / len(items), "n": len(items),
            "note": "given the compound labels, leave one well out"}


def transfer(rows: list[dict], wells: list[dict]) -> dict:
    """Does a compound's fingerprint learned on one species name it on the other?

    Nearest centroid over compounds, fitted on every well of one species and
    scored on every well of the other, in two representations: the raw
    feature change, and the twin's reading of it (presence probabilities and
    effect sizes). Nothing is tuned; each representation is standardised on
    the training species only.
    """
    raw: dict[tuple, list] = {}
    for r in rows:
        raw.setdefault((r["plate"], r["well"], r["compound"]), []).append(
            nde.phi(np.array(r["x_treat"])) - nde.phi(np.array(r["x_base"])))
    items = []
    for w in wells:
        k = (w["plate"], w["well"], w["compound"])
        if w["compound"] in ("Control",) or k not in raw:
            continue
        pa = np.clip(np.array(w["p_active"]), 1e-4, 1 - 1e-4)
        twin_rep = np.concatenate([np.log(pa / (1 - pa)), np.array(w["effect_med"])])
        items.append({"species": w["species"], "compound": w["compound"],
                      "raw": np.mean(raw[k], axis=0), "twin": twin_rep})
    out = {}
    species = sorted({i["species"] for i in items})
    for rep in ("raw", "twin"):
        res = {}
        for tr in species:
            te = [s for s in species if s != tr]
            A = [i for i in items if i["species"] == tr]
            B = [i for i in items if i["species"] in te]
            XA = np.stack([i[rep] for i in A]); XB = np.stack([i[rep] for i in B])
            mu, sd = XA.mean(0), XA.std(0) + 1e-9
            XA, XB = (XA - mu) / sd, (XB - mu) / sd
            names = sorted({i["compound"] for i in A})
            cents = np.stack([XA[[k for k, i in enumerate(A) if i["compound"] == n]].mean(0)
                              for n in names])
            pred = [names[int(np.argmin(np.linalg.norm(cents - x, axis=1)))] for x in XB]
            hits = [p == i["compound"] for p, i in zip(pred, B)]
            res[f"{tr}_to_other"] = {"n": len(B), "accuracy": float(np.mean(hits)),
                                     "chance": 1.0 / len(names)}
        out[rep] = res
    return out


# -------------------------------------------------------------------- guard
def shuffle_events(events: np.ndarray, duration: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Circularly shift each electrode independently: rates survive, structure does not."""
    if events.size == 0:
        return events
    out = events.copy()
    for e in np.unique(out[:, 0]):
        m = out[:, 0] == e
        out[m, 1] = np.mod(out[m, 1] + rng.uniform(0, duration), duration)
    return out


class Guard:
    """Two tests, each at its own quantile of held-out simulated records.

    Typicality (no simulation) and the predictive check (re-simulation). A
    pair is outside the model when either fires.
    """

    def __init__(self, twin, sim, ppc_threshold: float, duration: float,
                 transient: float, n_draws: int):
        self.twin, self.sim = twin, sim
        self.typ = ppc.Typicality.for_twin(twin)
        self.ppc_threshold = ppc_threshold
        self.duration, self.transient, self.n_draws = duration, transient, n_draws

    def __call__(self, xb, xt, seed: int, post=None) -> dict:
        t = self.typ.of_pair(self.twin, xb, xt)
        g = ppc.check(self.twin, self.sim, xb, xt, self.ppc_threshold, self.duration,
                      self.transient, n_draws=self.n_draws, seed=seed, post=post)
        fire_t = t > self.typ.threshold
        fire_p = g["discrepancy"] > self.ppc_threshold
        return {"typicality": t, "discrepancy": g["discrepancy"],
                "fires": bool(fire_t or fire_p), "fires_typicality": bool(fire_t),
                "fires_ppc": bool(fire_p)}


def calibrate_ppc(twin, sim, bank, idx, duration, transient, n_records, n_draws,
                  quantile, view) -> dict:
    return ppc.calibrate_threshold(twin, sim, bank, idx, duration, transient,
                                   n_records=n_records, n_draws=n_draws,
                                   quantile=quantile, seed=7, view=view)


def variant_pairs(view: str, n: int, duration: float, transient: float,
                  seed: int) -> list[tuple]:
    """Simulations from receptor kinetics the twin has no parameter for."""
    import json as _j
    from fit_domain import GaussianProposal
    dom = _j.loads((ROOT / "models" / "domain.json").read_text())
    rng = np.random.default_rng(seed)
    variant = S.Simulator(kinetics={"TAU_AMPA": 20.0, "TAU_GABA": 40.0})
    theta_c = GaussianProposal.load(dom["views"][view]["proposal"]).draw(n, rng)
    delta, active = SH.sample_shift(n, rng, theta_c=theta_c)
    theta_t, _, _ = SH.apply_shift(theta_c, delta, active)
    res = variant.run(SH.interleave(theta_c, theta_t), duration_s=duration,
                      transient_s=transient, seed=seed, pair=True)
    out = []
    for k in range(n):
        eb, nb = S.view_events(res.raw_events(2 * k), view)
        et, _ = S.view_events(res.raw_events(2 * k + 1), view)
        if eb.shape[0] < 50:
            continue
        out.append((F.compute(eb, nb, duration), F.compute(et, nb, duration)))
    return out


def guard_tests(guard: Guard, bank, idx_val, pairs, view: str, n_cases: int,
                seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    pick = rng.choice(idx_val, size=min(n_cases, len(idx_val)), replace=False)
    r = [guard(bank["x_base"][i], bank["x_treat"][i], seed=int(1e4 + k))
         for k, i in enumerate(pick)]
    out["bank_holdout"] = _rate(r)
    var = variant_pairs(view, n_cases, guard.duration, guard.transient, seed + 1)
    out["variant_kinetics"] = _rate([guard(xb, xt, seed=int(2e4 + k))
                                     for k, (xb, xt) in enumerate(var)])
    usable = [p for p in pairs if readable(p.baseline)]
    rng.shuffle(usable)
    usable = usable[:n_cases]
    shuf = []
    for k, p in enumerate(usable):
        xb = F.compute(shuffle_events(p.baseline, p.duration, rng), p.n_elec, p.duration)
        xt = F.compute(shuffle_events(p.treated, p.duration, rng), p.n_elec, p.duration)
        shuf.append(guard(xb, xt, seed=int(3e4 + k)))
    out["shuffled_real"] = _rate(shuf)
    return out


def _rate(r: list[dict]) -> dict:
    if not r:
        return {"n": 0}
    return {"n": len(r), "fire_rate": float(np.mean([x["fires"] for x in r])),
            "fire_rate_typicality": float(np.mean([x["fires_typicality"] for x in r])),
            "fire_rate_ppc": float(np.mean([x["fires_ppc"] for x in r])),
            "median_typicality": float(np.median([x["typicality"] for x in r])),
            "median_discrepancy": float(np.median([x["discrepancy"] for x in r]))}


def guard_on_wells(guard: Guard, pairs, seed: int) -> dict:
    """Guard verdict per well: outside the model when most windows fire."""
    status: dict[tuple, list[bool]] = {}
    for k, p in enumerate(pairs):
        if not readable(p.baseline):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        g = guard(xb, xt, seed=int(seed + k))
        status.setdefault((p.plate, p.well, p.compound), []).append(g["fires"])
    return {k: bool(np.mean(v) > 0.5) for k, v in status.items()}


# -------------------------------------------------------------- simulations
def simulation_side(twin, bank, idx_val, n: int = 20000, n_cal: int = 800,
                    seed: int = 2) -> dict:
    import torch
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx_val, size=min(n, len(idx_val)), replace=False)
    c = twin.scaler(nde.context(bank["x_base"][pick], bank["x_treat"][pick]))
    with torch.no_grad():
        logits = twin.presence(torch.as_tensor(c, dtype=torch.float32,
                                               device=twin.device)).cpu().numpy()
    p = twin.calibrate(1.0 / (1.0 + np.exp(-logits)))
    y = bank["active"][pick]
    pres = {}
    for j, k in enumerate(SHIFT_KEYS):
        bins = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(p[:, j], bins) - 1, 0, 9)
        ece = sum(abs(p[which == b, j].mean() - y[which == b, j].mean()) * (which == b).sum()
                  for b in range(10) if (which == b).sum() >= 20) / len(pick)
        pres[k] = {"auroc": auroc(p[:, j], y[:, j]), "ece": float(ece)}
    delta = bank["delta"][pick][:, P.SHIFT_IDX]
    mb = bank["x_base"][pick][:, F.NAMES.index("mfr")]
    mt = bank["x_treat"][pick][:, F.NAMES.index("mfr")]
    observable = np.abs(np.log1p(mt) - np.log1p(mb))
    single = y.sum(axis=1) == 1
    cases = {"all_single_mechanism": single,
             "saturating": single & (np.abs(delta).max(axis=1) > np.log(5.0))
                           & (observable > np.log(2.0))}
    rec = {}
    for name, m in cases.items():
        truth = np.argmax(y[m], axis=1)
        prob = p[m]
        agg = np.zeros((int(m.sum()), len(P.CLASS_NAMES)))
        for j, k in enumerate(SHIFT_KEYS):
            agg[:, P.class_index(k)] = np.maximum(agg[:, P.class_index(k)], prob[:, j])
        rec[name] = {"n": int(m.sum()),
                     "top1": float(np.mean(np.argmax(prob, 1) == truth)),
                     "top2": float(np.mean([t in np.argsort(-prob[i])[:2]
                                            for i, t in enumerate(truth)])),
                     "class_top1": float(np.mean(np.argmax(agg, 1) == np.array(
                         [P.class_index(SHIFT_KEYS[t]) for t in truth])))}
    # Coverage of the joint posterior, and recovery of the effect given active.
    sub = rng.choice(idx_val, size=min(n_cal, len(idx_val)), replace=False)
    levels = (0.5, 0.8, 0.9)
    cov = {k: {str(l): [] for l in levels} for k in SHIFT_KEYS}
    eff_t, eff_p = {k: [] for k in SHIFT_KEYS}, {k: [] for k in SHIFT_KEYS}
    for i in sub:
        post = twin.posterior(bank["x_base"][i], bank["x_treat"][i], n_samples=1500)
        for j, k in enumerate(SHIFT_KEYS):
            s = post["delta"][:, j]
            t = bank["delta"][i, P.SHIFT_IDX[j]]
            for l in levels:
                lo, hi = np.quantile(s, [(1 - l) / 2, 1 - (1 - l) / 2])
                cov[k][str(l)].append(bool(lo <= t <= hi))
            if bank["active"][i, j] and "effect_given_active" in post:
                eff_t[k].append(t)
                eff_p[k].append(float(np.median(post["effect_given_active"][:, j])))
    coverage = {k: {l: float(np.mean(v)) for l, v in d.items()} for k, d in cov.items()}
    recovery = {k: {"n": len(eff_t[k]),
                    "pearson_r": float(np.corrcoef(eff_t[k], eff_p[k])[0, 1])
                    if len(eff_t[k]) > 10 else float("nan"),
                    "sign_agreement": float(np.mean(np.sign(eff_t[k]) == np.sign(eff_p[k])))
                    if eff_t[k] else float("nan")}
                for k in SHIFT_KEYS}
    return {"presence": pres, "recovery_top1": rec, "coverage": coverage,
            "effect_given_active": recovery, "chance": 1.0 / len(SHIFT_KEYS),
            "class_chance": 1.0 / len(P.CLASS_NAMES)}


# ------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="A,C,D,E")
    ap.add_argument("--n-guard", type=int, default=60)
    args = ap.parse_args()

    from train import load_bank
    spec = read_prereg()
    ops = spec["operating_points"]
    n_samples, n_draws = ops["posterior_samples"], ops["predictive_draws"]
    duration, transient = ops["window_s"], 5.0
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["prereg_sha256"] = spec["digest"]
    t0 = time.time()

    twins = {v: nde.Twin.load(ROOT / spec["twins"][v], device="cuda") for v in spec["twins"]}
    unpaired = {v: nde.UnpairedTwin.load(ROOT / spec["unpaired"][v], device="cuda")
                for v in spec.get("unpaired", {})}
    sim = S.Simulator()
    banks, vals, guards = {}, {}, {}
    for v, twin in twins.items():
        banks[v] = load_bank(ROOT / spec["bank"], v)
        keep = banks[v]["domain"] == list(S.VIEWS).index(v)
        banks[v] = {k: a[keep] for k, a in banks[v].items()}
        vals[v] = np.load(ROOT / spec["twins"][v] / "val_index.npy")
        cal = calibrate_ppc(twin, sim, banks[v], vals[v], duration, transient,
                            ops["ppc_calibration_records"], n_draws,
                            ops["ppc_quantile"], v)
        res.setdefault("guard_thresholds", {})[v] = {
            "ppc": cal["threshold"], "ppc_quantile": ops["ppc_quantile"],
            "typicality": twin.meta.get("typicality_threshold"),
            "typicality_quantile": twin.meta.get("typicality_quantile")}
        guards[v] = Guard(twin, sim, cal["threshold"], duration, transient, n_draws)
        print(f"[{v}] guard thresholds {res['guard_thresholds'][v]}", flush=True)

    sections = set(args.sections.split(","))
    if "A" in sections:
        from hodgkins_razor import doorn as D
        pairs = D.load(treated=True)
        twin = twins["grid12"]
        rows = score_pairs(twin, pairs, n_samples)
        wells = by_well(rows)
        key = spec["answer_keys"]["doorn"]
        gw = guard_on_wells(guards["grid12"], pairs, seed=50000)
        for w in wells:
            w["outside_model"] = gw.get((w["plate"], w["well"], w["compound"]), None)
        inside = [w for w in wells if w["outside_model"] is False]
        res["A_doorn"] = {"wells": wells, "metrics": mechanism_metrics(wells, key),
                          "inside_model": {"n": len(inside),
                                           "metrics": mechanism_metrics(inside, key)},
                          "unpaired": mechanism_metrics(
                              unpaired_wells(unpaired["grid12"], pairs, n_samples), key)
                          if "grid12" in unpaired else None,
                          "prior_art": prior_art_metrics(
                              prior_art_wells(ROOT / "results" / "prior_art_doorn_doorn.json"),
                              key)}
        print(f"A done ({time.time() - t0:.0f}s): "
              f"{res['A_doorn']['metrics']['top1_hits']}/{res['A_doorn']['metrics']['n_wells']}",
              flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    if "C" in sections or "D" in sections:
        from hodgkins_razor import tampere as T
        pairs = T.load_all() + [p for n in T.PLATES for p in T.load_ttx(n)]
        for p in pairs:
            p.compound = {"DAP5": "D-AP5", "KainicAcid": "Kainic acid"}.get(p.compound, p.compound)
        twin = twins["grid16"]
        rows = score_pairs(twin, pairs, n_samples)
        wells = by_well(rows)
        if "C" in sections:
            keys = spec["answer_keys"]
            gw = guard_on_wells(guards["grid16"], pairs, seed=60000)
            for w in wells:
                w["outside_model"] = gw.get((w["plate"], w["well"], w["compound"]), None)
            inside = [w for w in wells if w["outside_model"] is False]
            res["C_tampere"] = {
                "wells": wells,
                "metrics_v1_key": mechanism_metrics(wells, keys["tampere_v1"]),
                "metrics_v2_key": mechanism_metrics(wells, keys["tampere_v2"]),
                "inside_model": {"n": len(inside),
                                 "metrics_v2_key": mechanism_metrics(inside, keys["tampere_v2"])},
                "supervised_lowo": supervised_lowo(rows, keys["tampere_v1"]),
                "unpaired": mechanism_metrics(
                    unpaired_wells(unpaired["grid16"], pairs, n_samples), keys["tampere_v1"])
                if "grid16" in unpaired else None,
                "prior_art": prior_art_metrics(
                    prior_art_wells(ROOT / "results" / "prior_art_doorn_tampere.json"),
                    keys["tampere_v1"])}
            print(f"C done ({time.time() - t0:.0f}s)", flush=True)
        if "D" in sections:
            res["D_transfer"] = transfer(rows, wells)
            print(f"D done: {json.dumps(res['D_transfer'])}", flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    if "E" in sections:
        res["E_simulation"] = {v: simulation_side(twins[v], banks[v], vals[v]) for v in twins}
        from hodgkins_razor import doorn as D, tampere as T
        real = {"grid16": T.load_all(), "grid12": D.load(treated=True)}
        res["E_guard"] = {v: guard_tests(guards[v], banks[v], vals[v], real[v], v,
                                         args.n_guard, seed=90 + k)
                          for k, v in enumerate(twins)}
        print(f"E done ({time.time() - t0:.0f}s)", flush=True)
        res_path.write_text(json.dumps(res, indent=1))

    mat = ROOT / "results" / "v2" / "mateus.json"
    if mat.exists():
        res["B_chips"] = json.loads(mat.read_text())
        res_path.write_text(json.dumps(res, indent=1))
    print("wrote", res_path)


if __name__ == "__main__":
    main()
