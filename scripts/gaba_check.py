"""Why is bath GABA 0/11 on Tampere? A check on simulations only.

    python scripts/gaba_check.py

On the recorded Tampere wells the version 2 twin read bath GABA as `g_na`
(5 of 7 rat wells). Two explanations predict different things on
simulations:

* **identifiability**: a tonic GABA-A conductance and a sodium block both
  silence the culture, and the features cannot tell them apart. Then the twin
  also calls `g_na` on simulated bath GABA.
* **misspecification**: the simulator's bath GABA differs from the real one.
  Then the twin reads simulated bath GABA correctly, and the recorded wells
  differ.

Simulated bath GABA (tonic GABA-A conductance raised, the pharmacology check's
graded series) is run on untreated cultures of the Tampere domain and read by
the version 2 grid16 twin. The same is done for a partial sodium block, as the
contrast. No recording enters it.
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

from hodgkins_razor import features as F, nde, params as P, simulator as S
from pharmacology_check import _feats, bank_cultures

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]
CONDITIONS = (
    {"name": "bath GABA, tonic +0.01", "key": "g_tonic_inh", "add": 0.01},
    {"name": "bath GABA, tonic +0.02", "key": "g_tonic_inh", "add": 0.02},
    {"name": "bath GABA, tonic +0.04", "key": "g_tonic_inh", "add": 0.04},
    {"name": "bath GABA, saturating (0.25)", "key": "g_tonic_inh", "set": 0.25},
    {"name": "synaptic GABA-A x3", "key": "g_gaba", "fold": 3.0},
    {"name": "sodium block x0.4", "key": "g_na", "fold": 0.4},
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin_v2_grid16")
    ap.add_argument("--bank", default="data/bank_v2")
    ap.add_argument("--view", default="grid16")
    ap.add_argument("--cultures", type=int, default=150)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--out", default="results/v2/gaba_check.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    twin = nde.Twin.load(ROOT / args.twin, device="cuda")
    base = bank_cultures(ROOT / args.bank, args.view, args.cultures, rng)
    sim = S.Simulator()
    pool = cf.ProcessPoolExecutor(max_workers=6)
    out = {"twin": args.twin, "view": args.view, "cultures": len(base), "conditions": []}
    mfr = F.NAMES.index("mfr")
    for cond in CONDITIONS:
        j = P.index(cond["key"])
        pairs = []
        for th in base:
            t = th.copy()
            if "set" in cond:
                t[j] = cond["set"]
            elif "fold" in cond:
                t[j] = th[j] * cond["fold"]
            else:
                t[j] = th[j] + cond["add"]
            t[j] = float(np.clip(t[j], P.LO[j], P.HI[j]))
            pairs.extend([th, t])
        arr = np.array(pairs)
        feats = []
        for a in range(0, arr.shape[0], 384):
            chunk = arr[a:a + 384]
            res = sim.run(chunk, duration_s=args.duration, transient_s=5.0,
                          seed=args.seed * 100 + a, pair=True)
            feats.extend(pool.map(_feats, [(res.raw_events(i), args.view, args.duration)
                                           for i in range(chunk.shape[0])], chunksize=8))
        feats = np.array(feats)
        fb, ft = feats[0::2], feats[1::2]
        top, called, pk = [], [], []
        for xb, xt in zip(fb, ft):
            if xb[mfr] <= 0:
                continue
            pa, _ = twin.presence_probs(xb, xt)
            top.append(SHIFT_KEYS[int(np.argmax(pa))])
            called.append(bool(pa.max() > 0.5))
            pk.append(float(pa[SHIFT_KEYS.index(cond["key"])]))
        counts: dict[str, int] = {}
        for k in top:
            counts[k] = counts.get(k, 0) + 1
        n = len(top)
        row = {"condition": cond["name"], "true": cond["key"], "n": n,
               "rate_ratio_median": float(np.median(ft[:, mfr] / np.maximum(fb[:, mfr], 1e-9))),
               "top1_true": sum(k == cond["key"] for k in top) / max(n, 1),
               "top1_g_na": sum(k == "g_na" for k in top) / max(n, 1),
               "top1_inhibition": sum(k in ("g_tonic_inh", "g_gaba") for k in top) / max(n, 1),
               "called": float(np.mean(called)) if called else float("nan"),
               "median_p_true": float(np.median(pk)) if pk else float("nan"),
               "top1_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
        out["conditions"].append(row)
        print(f"{cond['name']:30s} n {n:3d}  rate x{row['rate_ratio_median']:.2f}  "
              f"top-1 true {row['top1_true']:.2f}  g_na {row['top1_g_na']:.2f}  "
              f"inhibition {row['top1_inhibition']:.2f}  {row['top1_counts']}", flush=True)
    pool.shutdown()
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
