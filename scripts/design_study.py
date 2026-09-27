"""Does the recommended follow-up experiment resolve a tie? Simulations only.

    python scripts/design_study.py --twin models/twin_v2_grid16 --bank data/bank_v2

When a recording leaves two mechanisms tied, `hodgkins_razor.design` ranks
tool compounds by how far apart the two hypotheses predict their effect. This
script tests that on held-out simulated pairs whose true mechanism is known:

1. Take pairs with one true mechanism, where the twin's two leading
   mechanisms are within the app's tie margin and the truth is one of them.
2. Rank the candidate follow-ups exactly as the app does.
3. Run every candidate follow-up on the true culture, then score the choices:
   the recommended one; a candidate drawn at random (and the mean over all of
   them); recording the same well again; the **fixed best** follow-up, the one
   compound that resolves ties best on average, chosen on the other half of the
   cases (two-fold cross-fitting, so it is never scored on the cases it was
   chosen on); and the per-case oracle, an upper bound no policy can reach.
4. Decide between the two hypotheses from the follow-up recording alone, by
   which hypothesis's predicted feature distribution explains it better.

The score is how often each follow-up picks the true mechanism. Beating random
shows little; the recommendation earns its place only if it beats a fixed
protocol. The true
culture's follow-up is simulated on a fresh network with the true parameters,
as are the predictions, so neither side sees the original wiring.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import design as DS, features as F, nde, params as P
from hodgkins_razor import simulator as S
from train import load_bank

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def _feat(args):
    events, view, duration = args
    return F.compute(*S.view_events(events, view), duration)


def features(sim, theta_unit: np.ndarray, view: str, duration: float, seed: int, pool):
    res = sim.run(P.from_unit(theta_unit), duration_s=duration, transient_s=5.0, seed=seed)
    x = np.stack(list(pool.map(_feat, [(res.raw_events(i), view, duration)
                                       for i in range(theta_unit.shape[0])], chunksize=4)))
    return nde.phi(x)


def decide(x: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> str:
    """'a' or 'b': the hypothesis whose predicted distribution explains x better."""
    sd = np.sqrt(0.5 * (fa.var(0) + fb.var(0))) + 1e-3
    la = -0.5 * np.sum(((x - fa.mean(0)) / sd) ** 2)
    lb = -0.5 * np.sum(((x - fb.mean(0)) / sd) ** 2)
    return "a" if la >= lb else "b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin_v2_grid16")
    ap.add_argument("--bank", default="data/bank_v2")
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    twin = nde.Twin.load(ROOT / args.twin, device="cuda")
    view = twin.meta.get("view", "grid16")
    bank = load_bank(ROOT / args.bank, view)
    keep = bank["domain"] == list(S.VIEWS).index(twin.meta.get("domain", view))
    bank = {k: v[keep] for k, v in bank.items()}
    idx_val = np.load(ROOT / args.twin / "val_index.npy")
    rng = np.random.default_rng(args.seed)
    sim = S.Simulator()
    pool = cf.ProcessPoolExecutor(max_workers=8)
    cands = [c for c in DS.CANDIDATES]
    vecs = [DS._shift_vector(c) for c in cands]
    repeat = next(i for i, c in enumerate(cands) if not c.shift)
    active_only = [i for i in range(len(cands)) if i != repeat]

    single = idx_val[bank["active"][idx_val].sum(1) == 1]
    rng.shuffle(single)
    rows = []
    for n_seen, i in enumerate(single):
        if len(rows) >= args.cases:
            break
        truth = SHIFT_KEYS[int(np.argmax(bank["active"][i]))]
        pa, _ = twin.presence_probs(bank["x_base"][i], bank["x_treat"][i])
        order = np.argsort(-pa)
        a, b = SHIFT_KEYS[order[0]], SHIFT_KEYS[order[1]]
        if pa[order[0]] - pa[order[1]] >= args.margin or truth not in (a, b):
            continue
        post = twin.posterior(bank["x_base"][i], bank["x_treat"][i], n_samples=2000)
        ia, ib = DS.split_hypotheses(post, a, b)
        if len(ia) < 10 or len(ib) < 10:
            continue
        # Predicted follow-ups under each hypothesis, for every candidate.
        seed = int(args.seed * 1000 + len(rows) * 37)
        preds, seps = [], []
        for k, vec in enumerate(vecs):
            th = np.concatenate([DS._treated_theta(post, ia, vec, args.draws, rng),
                                 DS._treated_theta(post, ib, vec, args.draws, rng)])
            f = features(sim, th, view, args.duration, seed + k, pool)
            fa, fb = f[:args.draws], f[args.draws:]
            pooled = np.sqrt(0.5 * (fa.var(0) + fb.var(0))) + 1e-6
            seps.append(float(np.sqrt(np.mean((np.abs(fa.mean(0) - fb.mean(0)) / pooled) ** 2))))
            preds.append((fa, fb))
        rec = int(np.argmax(seps))
        rand = int(rng.choice(active_only))
        # The truth: the culture that produced the pair, after every follow-up.
        t_true = bank["theta_c"][i] + bank["delta"][i]
        th = np.stack([np.clip(t_true + v, P.TLO, P.THI) for v in vecs])
        obs = features(sim, th, view, args.duration, seed + 999, pool)
        correct = [bool((a if decide(x, *preds[k]) == "a" else b) == truth)
                   for k, x in enumerate(obs)]
        chosen = {"recommended": rec, "random": rand, "repeat": repeat}
        row = {"truth": truth, "tied": [a, b], "p_gap": float(pa[order[0]] - pa[order[1]]),
               "recommended": cands[rec].name, "random": cands[rand].name,
               "separation": dict(zip([c.name for c in cands], seps)),
               "correct_by_candidate": dict(zip([c.name for c in cands], correct))}
        for name, k in chosen.items():
            row[f"correct_{name}"] = correct[k]
        row["correct_oracle"] = any(correct)
        rows.append(row)
        print(f"{len(rows):3d}/{args.cases} (seen {n_seen + 1})  truth {truth:12s} tied {a}/{b}  "
              f"rec {cands[rec].name:28s} "
              + " ".join(f"{n}={'+' if row[f'correct_{n}'] else '.'}" for n in chosen), flush=True)
    pool.shutdown()

    from scipy.stats import binomtest
    # Fixed best follow-up, two-fold cross-fitted: chosen on one half of the
    # cases, scored on the other.
    names = [c.name for c in cands if c.shift]
    half = np.arange(len(rows)) % 2
    fixed_pick = {}
    for h in (0, 1):
        fit = [r for r, hh in zip(rows, half) if hh != h]
        best = max(names, key=lambda n: sum(r["correct_by_candidate"][n] for r in fit))
        fixed_pick[h] = best
        for r, hh in zip(rows, half):
            if hh == h:
                r["fixed_best"] = best
                r["correct_fixed_best"] = r["correct_by_candidate"][best]
    summ = {}
    for name in ("recommended", "fixed_best", "random", "repeat", "oracle"):
        k = sum(r[f"correct_{name}"] for r in rows)
        ci = binomtest(k, len(rows)).proportion_ci(method="wilson")
        summ[name] = {"correct": k, "n": len(rows), "rate": k / max(len(rows), 1),
                      "ci95": [float(ci.low), float(ci.high)]}
    summ["random_expected"] = {"rate": float(np.mean(
        [np.mean([r["correct_by_candidate"][n] for n in names]) for r in rows])) if rows else 0.0,
        "note": "mean over every active candidate, per case"}

    def mcnemar(x: str, y: str) -> dict:
        b = sum(r[f"correct_{x}"] and not r[f"correct_{y}"] for r in rows)
        c = sum(r[f"correct_{y}"] and not r[f"correct_{x}"] for r in rows)
        p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else float("nan")
        return {"b": b, "c": c, "p": float(p)}
    out = {"twin": args.twin, "view": view, "margin": args.margin, "draws": args.draws,
           "summary": summ, "fixed_best_choice": fixed_pick,
           "mcnemar_recommended_over_random": mcnemar("recommended", "random"),
           "mcnemar_recommended_over_fixed_best": mcnemar("recommended", "fixed_best"),
           "mcnemar_recommended_over_repeat": mcnemar("recommended", "repeat"),
           "rows": rows}
    path = pathlib.Path(args.out or ROOT / "results" / f"design_study_{view}.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summ, "fixed_best": fixed_pick,
                      "vs_random": out["mcnemar_recommended_over_random"],
                      "vs_fixed_best": out["mcnemar_recommended_over_fixed_best"]}, indent=1))


if __name__ == "__main__":
    main()
