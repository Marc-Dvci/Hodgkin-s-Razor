"""Run every pre-registered test and write the results.

    python scripts/evaluate.py --twin models/twin --bank data/bank

Reads the answer key from PREREGISTRATION.md, checks the file against its
recorded hash, and writes results/*.json plus results/RESULTS.md. Nothing here
chooses a threshold, a feature or an operating point: those are all fixed
before this script runs.
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
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, nde, params as P, ppc, shift as SH
from hodgkins_razor import simulator as S, tampere as T

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_EVENTS = 50
MIN_ELEC = 3
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


# ---------------------------------------------------------------- answer key
def read_key() -> dict[str, tuple[str | None, str | None]]:
    """Answer key parsed from the pre-registration, with its hash checked."""
    md = ROOT / "PREREGISTRATION.md"
    digest = hashlib.sha256(md.read_bytes()).hexdigest()
    recorded = (ROOT / "PREREGISTRATION.sha256").read_text().split()[0]
    if digest != recorded:
        raise SystemExit("PREREGISTRATION.md does not match its recorded hash")
    key: dict[str, tuple[str | None, str | None]] = {}
    for line in md.read_text().splitlines():
        m = re.match(r"\|\s*([A-Za-z0-9 .\-]+?)\s*\|\s*[^|]*\|\s*`?(\w+)`?\s*\|\s*(\w+)\s*\|", line)
        if not m:
            continue
        name, param, direction = m.group(1).strip(), m.group(2), m.group(3)
        if param == "Parameter" or name == "Compound":
            continue
        key[name] = (None, None) if param == "none" else (param, direction)
    return {"digest": digest, "key": key}


COMPOUND_ALIAS = {"CNQX": "CNQX", "DAP5": "D-AP5", "D-AP5": "D-AP5",
                  "GABA": "GABA", "Gabazine": "Gabazine",
                  "KainicAcid": "Kainic acid", "Control": "Control", "TTX": "TTX"}


def readable(pair: T.Pair) -> bool:
    ev = pair.baseline
    if ev.shape[0] < MIN_EVENTS:
        return False
    return len(np.unique(ev[:, 0])) >= MIN_ELEC


# ------------------------------------------------------- simulation-side tests
def calibration_tests(twin, bank: dict, idx: np.ndarray, n: int = 4000,
                      n_samples: int = 2000, seed: int = 0) -> dict:
    """Interval coverage, rank uniformity and shift recovery on held-out records."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    levels = (0.50, 0.80, 0.90)
    cov = {k: {f"{int(l * 100)}": [] for l in levels} for k in list(P.KEYS) + SHIFT_KEYS}
    ranks = {k: [] for k in list(P.KEYS) + SHIFT_KEYS}
    true_d, post_d, act = [], [], []
    for i in pick:
        post = twin.posterior(bank["x_base"][i], bank["x_treat"][i], n_samples=n_samples)
        for j, k in enumerate(P.KEYS):
            s = post["theta_c"][:, j]
            t = bank["theta_c"][i, j]
            for l in levels:
                lo, hi = np.quantile(s, [(1 - l) / 2, 1 - (1 - l) / 2])
                cov[k][f"{int(l * 100)}"].append(bool(lo <= t <= hi))
            ranks[k].append(float((s < t).mean()))
        for j, k in enumerate(SHIFT_KEYS):
            s = post["delta"][:, j]
            t = bank["delta"][i, P.SHIFT_IDX[j]]
            for l in levels:
                lo, hi = np.quantile(s, [(1 - l) / 2, 1 - (1 - l) / 2])
                cov[k][f"{int(l * 100)}"].append(bool(lo <= t <= hi))
            ranks[k].append(float((s < t).mean()))
        true_d.append(bank["delta"][i, P.SHIFT_IDX])
        post_d.append(np.median(post["delta"], axis=0))
        act.append(bank["active"][i])
    true_d, post_d, act = np.array(true_d), np.array(post_d), np.array(act)

    recovery = {}
    for j, k in enumerate(SHIFT_KEYS):
        m = act[:, j]
        if m.sum() > 10:
            r = float(np.corrcoef(true_d[m, j], post_d[m, j])[0, 1])
            mae = float(np.mean(np.abs(true_d[m, j] - post_d[m, j])))
        else:
            r, mae = float("nan"), float("nan")
        recovery[k] = {"n_active": int(m.sum()), "pearson_r": r, "mae_log": mae}

    return {"n_records": int(len(pick)),
            "coverage": {k: {l: float(np.mean(v)) for l, v in d.items()}
                         for k, d in cov.items()},
            "rank_ks": {k: float(_ks_uniform(np.array(v))) for k, v in ranks.items()},
            "recovery": recovery}


def _ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of ranks from uniform."""
    u = np.sort(np.clip(u, 0, 1))
    n = u.size
    if n == 0:
        return float("nan")
    grid = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(u - grid)))


def presence_reliability(twin, bank: dict, idx: np.ndarray, n: int = 20000,
                         seed: int = 1) -> dict:
    """Calibration of the presence probability, per mechanism."""
    import torch
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    c = twin.scaler(nde.context(bank["x_base"][pick], bank["x_treat"][pick]))
    with torch.no_grad():
        logits = twin.presence(torch.as_tensor(c, dtype=torch.float32,
                                               device=twin.device)).cpu().numpy()
    p = twin.calibrate(1.0 / (1.0 + np.exp(-logits)))
    y = bank["active"][pick]
    out = {}
    for j, k in enumerate(SHIFT_KEYS):
        bins = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(p[:, j], bins) - 1, 0, 9)
        rows = []
        for b in range(10):
            m = which == b
            if m.sum() < 20:
                continue
            rows.append({"p_mean": float(p[m, j].mean()),
                         "observed": float(y[m, j].mean()), "n": int(m.sum())})
        ece = sum(r["n"] * abs(r["p_mean"] - r["observed"]) for r in rows) / max(
            sum(r["n"] for r in rows), 1)
        auc = _auc(p[:, j], y[:, j])
        out[k] = {"bins": rows, "ece": float(ece), "auroc": float(auc)}
    return out


def simulated_recovery(twin, bank: dict, idx: np.ndarray, n: int = 20000,
                       seed: int = 2) -> dict:
    """Top-1 on held-out simulations, and on the regime the compounds occupy.

    A recorded compound is applied at a saturating concentration and produces a
    large change in the recording. Reporting the same metric on simulations
    restricted that way says how much of any gap to the recorded result is the
    simulator and how much is the question.
    """
    import torch
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    c = twin.scaler(nde.context(bank["x_base"][pick], bank["x_treat"][pick]))
    with torch.no_grad():
        logits = twin.presence(torch.as_tensor(c, dtype=torch.float32,
                                               device=twin.device)).cpu().numpy()
    p = twin.calibrate(1.0 / (1.0 + np.exp(-logits)))
    y = bank["active"][pick]
    delta = bank["delta"][pick][:, P.SHIFT_IDX]
    mb = bank["x_base"][pick][:, F.NAMES.index("mfr")]
    mt = bank["x_treat"][pick][:, F.NAMES.index("mfr")]
    observable = np.abs(np.log1p(mt) - np.log1p(mb))
    single = y.sum(axis=1) == 1

    out = {}
    cases = {
        "all_single_mechanism": single,
        "saturating": single & (np.abs(delta).max(axis=1) > np.log(5.0))
                      & (observable > np.log(2.0)),
    }
    for name, mask in cases.items():
        if mask.sum() < 40:
            out[name] = {"n": int(mask.sum())}
            continue
        truth = np.argmax(y[mask], axis=1)
        prob = p[mask]
        top1 = float(np.mean(np.argmax(prob, axis=1) == truth))
        top2 = float(np.mean([t in np.argsort(-prob[i])[:2]
                              for i, t in enumerate(truth)]))
        agg = np.zeros((int(mask.sum()), len(P.CLASS_NAMES)))
        for j, k in enumerate(SHIFT_KEYS):
            agg[:, P.class_index(k)] += prob[:, j]
        cls = np.array([P.class_index(SHIFT_KEYS[t]) for t in truth])
        out[name] = {"n": int(mask.sum()), "top1": top1, "top2": top2,
                     "class_top1": float(np.mean(np.argmax(agg, axis=1) == cls))}
    out["chance"] = 1.0 / len(SHIFT_KEYS)
    out["class_chance"] = 1.0 / len(P.CLASS_NAMES)
    return out


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    label = label.astype(bool)
    if label.all() or not label.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------- real recordings
def score_pairs(model, pairs: list[T.Pair], n_samples: int = 4000) -> list[dict]:
    """Posterior summary for every readable pair."""
    rows = []
    for p in pairs:
        if not readable(p):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        post = model.posterior(xb, xt, n_samples=n_samples)
        rows.append({"plate": p.plate, "species": p.species, "well": p.well,
                     "compound": COMPOUND_ALIAS.get(p.compound, p.compound),
                     "dose": p.dose,
                     "p_active": post["p_active"].tolist(),
                     "delta_med": np.median(post["delta"], axis=0).tolist(),
                     "x_base": xb.tolist(), "x_treat": xt.tolist()})
    return rows


def by_well(rows: list[dict]) -> list[dict]:
    """Collapse windows to wells, the unit the metric is defined on."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["plate"], r["well"], r["compound"]), []).append(r)
    out = []
    for (plate, well, comp), rs in groups.items():
        pa = np.mean([r["p_active"] for r in rs], axis=0)
        dm = np.mean([r["delta_med"] for r in rs], axis=0)
        out.append({"plate": plate, "species": rs[0]["species"], "well": well,
                    "compound": comp, "n_windows": len(rs),
                    "p_active": pa.tolist(), "delta_med": dm.tolist(),
                    "top1": SHIFT_KEYS[int(np.argmax(pa))],
                    "top2": [SHIFT_KEYS[i] for i in np.argsort(-pa)[:2]]})
    return out


def mechanism_metrics(wells: list[dict], key: dict) -> dict:
    scored = [w for w in wells if key.get(w["compound"], (None,))[0]]
    controls = [w for w in wells if w["compound"] == "Control"]
    n_top1 = sum(w["top1"] == key[w["compound"]][0] for w in scored)
    n_top2 = sum(key[w["compound"]][0] in w["top2"] for w in scored)
    dir_ok = 0
    dir_n = 0
    for w in scored:
        param, want = key[w["compound"]]
        if w["top1"] != param:
            continue
        dir_n += 1
        j = SHIFT_KEYS.index(param)
        got = "up" if w["delta_med"][j] > 0 else "down"
        dir_ok += int(got == want)
    per_comp: dict[str, dict] = {}
    for w in scored:
        d = per_comp.setdefault(w["compound"], {"n": 0, "top1": 0, "top2": 0,
                                                "species": {}})
        d["n"] += 1
        d["top1"] += int(w["top1"] == key[w["compound"]][0])
        d["top2"] += int(key[w["compound"]][0] in w["top2"])
        sp = d["species"].setdefault(w["species"], {"n": 0, "top1": 0})
        sp["n"] += 1
        sp["top1"] += int(w["top1"] == key[w["compound"]][0])
    per_species: dict[str, dict] = {}
    for w in scored:
        d = per_species.setdefault(w["species"], {"n": 0, "top1": 0})
        d["n"] += 1
        d["top1"] += int(w["top1"] == key[w["compound"]][0])

    confusion = np.zeros((len(SHIFT_KEYS), len(SHIFT_KEYS)), dtype=int)
    for w in scored:
        confusion[SHIFT_KEYS.index(key[w["compound"]][0]),
                  SHIFT_KEYS.index(w["top1"])] += 1

    ctrl_flag = [max(w["p_active"]) > 0.5 for w in controls]
    return {"n_wells_scored": len(scored),
            "top1_accuracy": n_top1 / max(len(scored), 1),
            "top2_accuracy": n_top2 / max(len(scored), 1),
            "direction_agreement": dir_ok / max(dir_n, 1),
            "direction_n": dir_n,
            "chance": 1.0 / len(SHIFT_KEYS),
            "per_compound": per_comp, "per_species": per_species,
            "confusion": confusion.tolist(), "confusion_labels": SHIFT_KEYS,
            "n_controls": len(controls),
            "control_false_mechanism_rate": float(np.mean(ctrl_flag)) if controls else float("nan")}


def class_metrics(wells: list[dict], key: dict) -> dict:
    """Accuracy at the level of the mechanism class.

    Secondary and post-hoc. A compound identified as acting on inhibition
    rather than on excitatory transmission is a useful answer even when the
    exact conductance is not resolved, and the two errors are worth separating.
    """
    scored = [w for w in wells if key.get(w["compound"], (None,))[0]]
    if not scored:
        return {"n": 0}
    hits = 0
    conf = np.zeros((len(P.CLASS_NAMES), len(P.CLASS_NAMES)), dtype=int)
    per: dict[str, dict] = {}
    for w in scored:
        want = P.class_index(key[w["compound"]][0])
        agg = np.zeros(len(P.CLASS_NAMES))
        for j, k in enumerate(SHIFT_KEYS):
            agg[P.class_index(k)] += w["p_active"][j]
        got = int(np.argmax(agg))
        conf[want, got] += 1
        hits += int(got == want)
        d = per.setdefault(w["compound"], {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += int(got == want)
    return {"n": len(scored), "top1_accuracy": hits / len(scored),
            "chance": 1.0 / len(P.CLASS_NAMES),
            "labels": list(P.CLASS_NAMES), "confusion": conf.tolist(),
            "per_compound": per, "note": "secondary, not pre-registered"}


def supervised_baseline(rows: list[dict], key: dict) -> dict:
    """Nearest-centroid on feature differences, with the labels the twin never sees.

    Leave-one-well-out, so a well never contributes to its own centroid.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["plate"], r["well"], r["compound"]), []).append(r)
    items = []
    for (plate, well, comp), rs in groups.items():
        if not key.get(comp, (None,))[0]:
            continue
        d = np.mean([nde.phi(np.array(r["x_treat"])) - nde.phi(np.array(r["x_base"]))
                     for r in rs], axis=0)
        items.append({"plate": plate, "well": well, "compound": comp,
                      "param": key[comp][0], "d": d, "species": rs[0]["species"]})
    if len(items) < 4:
        return {"top1_accuracy": float("nan"), "n": len(items)}
    D = np.stack([i["d"] for i in items])
    D = (D - D.mean(0)) / (D.std(0) + 1e-9)
    labels = [i["param"] for i in items]
    correct = 0
    for k in range(len(items)):
        keep = [j for j in range(len(items)) if j != k]
        cents, names = [], []
        for lab in sorted(set(labels[j] for j in keep)):
            sel = [j for j in keep if labels[j] == lab]
            cents.append(D[sel].mean(0))
            names.append(lab)
        dist = [np.linalg.norm(D[k] - c) for c in cents]
        correct += int(names[int(np.argmin(dist))] == labels[k])
    return {"top1_accuracy": correct / len(items), "n": len(items),
            "note": "sees the compound labels under leave-one-well-out"}


# ------------------------------------------------------------------- the guard
def shuffle_events(events: np.ndarray, duration: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Circularly shift each electrode independently.

    Every per-electrode rate and interval distribution survives; the network
    structure between electrodes does not.
    """
    if events.size == 0:
        return events
    out = events.copy()
    for e in np.unique(out[:, 0]):
        m = out[:, 0] == e
        out[m, 1] = np.mod(out[m, 1] + rng.uniform(0, duration), duration)
    return out


def guard_tests(twin, sim, bank: dict, idx_val: np.ndarray, pairs: list[T.Pair],
                duration: float, transient: float, threshold: float,
                n_cases: int = 40, n_draws: int = 48, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {"threshold": threshold}

    # Must pass: held-out bank records.
    pick = rng.choice(idx_val, size=min(n_cases, len(idx_val)), replace=False)
    d_bank = [ppc.check(twin, sim, bank["x_base"][i], bank["x_treat"][i], threshold,
                        duration, transient, n_draws=n_draws, seed=int(1e4 + k))["discrepancy"]
              for k, i in enumerate(pick)]
    out["bank_holdout"] = {"n": len(d_bank),
                           "fire_rate": float(np.mean(np.array(d_bank) > threshold)),
                           "median_discrepancy": float(np.median(d_bank))}

    # Must fire: receptor kinetics the twin has no parameter for.
    variant = S.Simulator(kinetics={"TAU_AMPA": 20.0, "TAU_GABA": 40.0})
    n_v = n_cases
    theta_c = P.sample_prior(n_v, rng)
    delta, _ = SH.sample_shift(n_v, rng)
    theta_t, _ = SH.apply_shift(theta_c, delta)
    res = variant.run(SH.interleave(theta_c, theta_t), duration_s=duration,
                      transient_s=transient, seed=77, pair=True)
    d_var = []
    for k in range(n_v):
        xb = F.compute(res.as_events(2 * k), S.NELEC, duration)
        xt = F.compute(res.as_events(2 * k + 1), S.NELEC, duration)
        if xb[0] < 0.05:
            continue
        d_var.append(ppc.check(twin, sim, xb, xt, threshold, duration, transient,
                               n_draws=n_draws, seed=int(2e4 + k))["discrepancy"])
    out["variant_kinetics"] = {"n": len(d_var),
                               "fire_rate": float(np.mean(np.array(d_var) > threshold)) if d_var else float("nan"),
                               "median_discrepancy": float(np.median(d_var)) if d_var else float("nan")}

    # Must fire: real recordings with the network structure destroyed.
    usable = [p for p in pairs if readable(p)][:n_cases]
    d_shuf = []
    for k, p in enumerate(usable):
        xb = F.compute(shuffle_events(p.baseline, p.duration, rng), p.n_elec, p.duration)
        xt = F.compute(shuffle_events(p.treated, p.duration, rng), p.n_elec, p.duration)
        d_shuf.append(ppc.check(twin, sim, xb, xt, threshold, duration, transient,
                                n_draws=n_draws, seed=int(3e4 + k))["discrepancy"])
    out["shuffled_real"] = {"n": len(d_shuf),
                            "fire_rate": float(np.mean(np.array(d_shuf) > threshold)) if d_shuf else float("nan"),
                            "median_discrepancy": float(np.median(d_shuf)) if d_shuf else float("nan")}

    # Reported, not pre-registered as pass or fail: the real pairs themselves.
    d_real = []
    for k, p in enumerate(usable):
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        d_real.append(ppc.check(twin, sim, xb, xt, threshold, duration, transient,
                                n_draws=n_draws, seed=int(4e4 + k))["discrepancy"])
    out["real_pairs"] = {"n": len(d_real),
                         "fire_rate": float(np.mean(np.array(d_real) > threshold)) if d_real else float("nan"),
                         "median_discrepancy": float(np.median(d_real)) if d_real else float("nan")}
    return out


def inside_model_metrics(twin, sim, pairs, wells, key, threshold, duration,
                         transient, n_draws: int = 24) -> dict:
    """Mechanism accuracy restricted to wells the predictive check passes.

    Secondary and post-hoc. It is reported so that a high refusal rate cannot
    be read as a high error rate, and so the two can be told apart.
    """
    status: dict[tuple, list[bool]] = {}
    for k, p in enumerate(pairs):
        if not readable(p):
            continue
        xb = F.compute(p.baseline, p.n_elec, p.duration)
        xt = F.compute(p.treated, p.n_elec, p.duration)
        g = ppc.check(twin, sim, xb, xt, threshold, duration, transient,
                      n_draws=n_draws, seed=int(5e4 + k))
        comp = COMPOUND_ALIAS.get(p.compound, p.compound)
        status.setdefault((p.plate, p.well, comp), []).append(g["inside_model"])
    inside = {k: (np.mean(v) >= 0.5) for k, v in status.items()}
    kept = [w for w in wells
            if inside.get((w["plate"], w["well"], w["compound"]), False)]
    out = mechanism_metrics(kept, key)
    out["n_wells_total"] = len(wells)
    out["n_wells_inside"] = len(kept)
    out["inside_fraction"] = len(kept) / max(len(wells), 1)
    out["note"] = "secondary, not pre-registered"
    return out


# ------------------------------------------------------------------ driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin")
    ap.add_argument("--unpaired-twin", default="models/twin_unpaired")
    ap.add_argument("--bank", default="data/bank")
    ap.add_argument("--out", default="results")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--n-calibration", type=int, default=1500)
    ap.add_argument("--guard-cases", type=int, default=40)
    ap.add_argument("--ppc-records", type=int, default=100)
    ap.add_argument("--skip-guard", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = read_key()
    key = pre["key"]
    print("answer key:", {k: v[0] for k, v in key.items()})

    twin = nde.Twin.load(args.twin, device="cuda")
    bankdir = pathlib.Path(args.bank)
    shards = sorted(bankdir.glob("shard_*.npz"))
    acc = {k: [] for k in ("theta_c", "delta", "active", "x_base", "x_treat")}
    for s in shards:
        d = np.load(s)
        for k in acc:
            acc[k].append(d[k])
    bank = {k: np.concatenate(v) for k, v in acc.items()}
    idx_val = np.load(pathlib.Path(args.twin) / "val_index.npy")
    print(f"bank {bank['theta_c'].shape[0]} pairs, {len(idx_val)} held out")

    results: dict = {"prereg_sha256": pre["digest"],
                     "twin_meta": twin.meta,
                     "duration_s": args.duration}

    t0 = time.time()
    print("calibration and recovery on held-out simulations ...", flush=True)
    results["calibration"] = calibration_tests(twin, bank, idx_val,
                                               n=args.n_calibration)
    results["presence"] = presence_reliability(twin, bank, idx_val)
    results["simulated_recovery"] = simulated_recovery(twin, bank, idx_val)
    print(f"  {time.time() - t0:.0f}s", flush=True)

    pairs = T.load_all(window_s=args.duration, n_windows=3)
    pairs += T.load_ttx("rat", window_s=args.duration, n_windows=3)
    pairs += T.load_ttx("human", window_s=args.duration, n_windows=3)
    n_excluded = sum(1 for p in pairs if not readable(p))
    print(f"real pairs: {len(pairs)} windows, {n_excluded} unreadable", flush=True)

    rows = score_pairs(twin, pairs)
    wells = by_well(rows)
    results["real"] = {"n_windows": len(rows), "n_excluded": n_excluded,
                       "wells": wells}
    results["mechanism"] = mechanism_metrics(wells, key)
    results["supervised_baseline"] = supervised_baseline(rows, key)
    results["mechanism_class"] = class_metrics(wells, key)
    print("top-1 accuracy:", round(results["mechanism"]["top1_accuracy"], 3),
          " controls flagged:", round(results["mechanism"]["control_false_mechanism_rate"], 3),
          flush=True)

    up = pathlib.Path(args.unpaired_twin)
    if (up / "flow.pt").exists():
        um = nde.UnpairedTwin.load(up, device="cuda")
        urows = score_pairs(um, pairs)
        uwells = by_well(urows)
        results["unpaired_baseline"] = mechanism_metrics(uwells, key)
        print("unpaired baseline top-1:",
              round(results["unpaired_baseline"]["top1_accuracy"], 3), flush=True)

    if not args.skip_guard:
        sim = S.Simulator()
        results["guard_on_real"] = {}
        print("calibrating the guard on held-out records ...", flush=True)
        cal = ppc.calibrate_threshold(twin, sim, bank, idx_val, args.duration,
                                      args.transient, n_records=args.ppc_records)
        results["guard_calibration"] = {k: v for k, v in cal.items() if k != "values"}
        results["guard_calibration"]["values"] = cal["values"]
        print("  threshold", round(cal["threshold"], 2), flush=True)
        results["guard"] = guard_tests(twin, sim, bank, idx_val, pairs,
                                       args.duration, args.transient,
                                       cal["threshold"], n_cases=args.guard_cases)
        # Not pre-registered, and declared as such: the same metric restricted
        # to the wells the twin can reproduce. Written before any result was
        # seen, and reported whichever way it falls.
        results["mechanism_inside_model"] = inside_model_metrics(
            twin, sim, pairs, wells, key, cal["threshold"], args.duration,
            args.transient)
        print("  guard:", {k: round(v["fire_rate"], 3) for k, v in results["guard"].items()
                           if isinstance(v, dict)}, flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "results.json", f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
