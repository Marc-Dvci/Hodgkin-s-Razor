"""Does the twin track a culture as it matures?

    python scripts/development.py

The Tampere development plates record the same wells from day 2 to day 66 in
vitro with no compound applied. The day is an ordering the model has never
seen, and there is no drug label anywhere in it, so it is an independent test
of the parameter axis: a maturing culture should show rising synaptic
conductance and connectivity, and the twin should say so without being told
what day it is.

Maturation is not a wash-on: it changes the wiring, and the paired model holds
wiring fixed across a pair precisely because a drug cannot change it. So this
uses the unpaired model, which reads absolute parameters from one recording,
and correlates them with the day. The day never enters the inference.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, nde, params as P, tampere as T

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]

# What a maturing culture is expected to do, from the developmental literature:
# synapses form, so excitatory and inhibitory conductance and connectivity all
# rise over the first weeks in vitro.
EXPECTED_UP = ("g_ampa", "g_nmda", "g_gaba", "p_conn")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def permutation_p(div: np.ndarray, value: np.ndarray, wells: np.ndarray,
                  observed: float, n: int = 2000, seed: int = 0) -> float:
    """Shuffle the day within each well, so the null keeps the well structure."""
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        shuffled = div.copy()
        for w in np.unique(wells):
            m = wells == w
            shuffled[m] = rng.permutation(div[m])
        if abs(spearman(shuffled, value)) >= abs(observed):
            count += 1
    return (count + 1) / (n + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin_unpaired")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--min-div", type=int, default=7)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--out", default="results/development.json")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    twin = nde.UnpairedTwin.load(ROOT / args.twin, device=dev)
    recs = T.load_development(window_s=args.window, n_windows=1)
    print(f"{len(recs)} development recordings")

    by_well: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if r.events.shape[0] < 50 or len(np.unique(r.events[:, 0])) < 3:
            continue
        by_well[(r.plate, r.well)].append(r)
    print(f"{len(by_well)} wells with readable recordings")

    rows = []
    for (plate, well), items in sorted(by_well.items()):
        items.sort(key=lambda r: r.div)
        early = [r for r in items if r.div >= args.min_div]
        if len(early) < 3:
            continue
        for r in early:
            x = F.compute(r.events, r.n_elec, r.duration)
            theta = twin.theta(x, n_samples=args.n_samples)
            rows.append({"plate": plate, "species": r.species, "well": well,
                         "div": r.div,
                         "theta": np.median(theta, axis=0).tolist()})
        print(f"  {plate} {well}: {len(early)} recordings from DIV {early[0].div}",
              flush=True)

    if not rows:
        raise SystemExit("no usable development pairs")

    div = np.array([r["div"] for r in rows], dtype=float)
    wells = np.array([f"{r['plate']}/{r['well']}" for r in rows])
    theta = np.array([r["theta"] for r in rows])

    out = {"n_pairs": len(rows), "n_wells": int(len(np.unique(wells))),
           "div_range": [float(div.min()), float(div.max())],
           "window_s": args.window, "parameters": {}}
    print()
    print(f"{'parameter':13s} {'rho with DIV':>13s} {'p (well-permuted)':>19s}  expected")
    for j, key in enumerate(P.KEYS):
        rho = spearman(div, theta[:, j])
        p = permutation_p(div, theta[:, j], wells, rho)
        want = "up" if key in EXPECTED_UP else ""
        agrees = (rho > 0) if key in EXPECTED_UP else None
        out["parameters"][key] = {"spearman_with_div": rho, "p_value": p,
                                  "expected": want, "agrees": agrees}
        mark = "" if agrees is None else ("  +" if agrees else "  .")
        print(f"{key:13s} {rho:13.3f} {p:19.4f}  {want:8s}{mark}")

    hits = [k for k in EXPECTED_UP
            if out["parameters"][k]["spearman_with_div"] > 0
            and out["parameters"][k]["p_value"] < 0.05]
    out["expected_up"] = list(EXPECTED_UP)
    out["expected_up_recovered"] = hits
    print(f"\n{len(hits)} of {len(EXPECTED_UP)} conductances rise with days in vitro")
    pathlib.Path(ROOT / args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
