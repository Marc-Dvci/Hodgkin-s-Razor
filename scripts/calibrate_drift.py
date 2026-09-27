"""How far apart are two sister cultures when neither was treated?

    python scripts/calibrate_drift.py --domain-file models/domain_v3.json --view mcs60q

Sister arrays plated from one preparation differ in wiring, in electrode
pickup and, a little, in every culture parameter. The simulator's unpaired
mode gives the first two; the third is the drift that `make_bank.py --design
sister` adds to the second sister. This script sets its size from sister pairs
recorded before any drug was applied (Charlesworth et al., 6 and 7 days in
vitro), so no treated recording and no drug label enters it.

For each candidate drift, simulated sister pairs are drawn from the fitted
domain and read through the recording system. The chosen drift is the one
whose feature differences between sisters match the recorded ones: the median
over statistics of log(recorded spread / simulated spread) closest to zero.
Only recorded pairs whose first sister lies inside the domain box are used, so
the comparison is made on the preparations the twin is trained on.
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

from hodgkins_razor import charlesworth as C, features as F, nde, simulator as S
from fit_domain import GaussianProposal, violation
from make_bank import sister_drift

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _feat(args):
    events, view, duration = args
    ev, n = S.view_events(events, view)
    return F.compute(ev, n, duration)


def recorded_pairs(kind: str, n_windows: int, min_div: float, max_div: float) -> list:
    """Untreated pairs of one kind. Sister kinds come from Charlesworth et al.;
    the within-well kinds are one well recorded twice with no compound:
    `tampere-vehicle` (vehicle added, the same time gap as a compound) and
    `doorn-null` (two pre-drug stretches of each Dynasore well, 4 minutes apart)."""
    if kind == "tampere-vehicle":
        from hodgkins_razor import tampere as T
        return [q for q in T.load_all() if q.is_control]
    if kind == "doorn-null":
        from null_controls_v2 import null_pairs
        return null_pairs()
    return C.load(kinds=(kind,), min_div=min_div, max_div=max_div, n_windows=n_windows)


def recorded_differences(box: dict, n_windows: int, kind: str = "pre-drug",
                         min_div: float = 0.0, max_div: float = 99.0) -> np.ndarray:
    rows = []
    for p in recorded_pairs(kind, n_windows, min_div, max_div):
        if p.baseline.shape[0] < 50 or p.treated.shape[0] < 50:
            continue
        xa = F.compute(p.baseline, p.n_elec, p.duration)
        if violation(xa, box)[0] > 0:
            continue
        xb = F.compute(p.treated, p.n_elec, p.duration)
        rows.append(nde.phi(xb) - nde.phi(xa))
    return np.array(rows)


def simulated_differences(sim, prop, box, view, drift, n, seed, pool, duration,
                          paired: bool = False):
    rng = np.random.default_rng(seed)
    th = prop.draw(n, rng)
    tw = sister_drift(th, drift, rng)
    res = sim.run(np.stack([th, tw], 1).reshape(-1, th.shape[1]), duration_s=duration,
                  transient_s=5.0, seed=seed, pair=paired)
    x = np.stack(list(pool.map(_feat, [(res.raw_events(k), view, duration)
                                       for k in range(2 * n)], chunksize=8)))
    xa, xb = x[0::2], x[1::2]
    ok = (violation(xa, box) == 0) & ~res.truncated[0::2] & ~res.truncated[1::2]
    return nde.phi(xb[ok]) - nde.phi(xa[ok])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-file", default="models/domain_v3.json")
    ap.add_argument("--view", default="mcs60q")
    ap.add_argument("--drifts", default="0,0.02,0.04,0.06,0.08,0.12")
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--kind", default="pre-drug",
                    choices=("pre-drug", "a-c", "tampere-vehicle", "doorn-null"),
                    help="pre-drug: A-B sisters at 6-7 days; a-c: the never-treated A and C "
                         "sisters of four-array preparations, at the scored ages")
    ap.add_argument("--paired", action="store_true",
                    help="one network recorded twice (within-well kinds), not two sisters")
    ap.add_argument("--min-div", type=float, default=0.0)
    ap.add_argument("--max-div", type=float, default=99.0)
    ap.add_argument("--out", default="models/drift_v3.json")
    args = ap.parse_args()

    dom = json.loads((ROOT / args.domain_file).read_text())["views"][args.view]
    box, prop = dom["box"], GaussianProposal.load(dom["proposal"])
    real = recorded_differences(box, args.windows, args.kind, args.min_div, args.max_div)
    print(f"recorded {args.kind} sister windows inside the domain: {len(real)}", flush=True)
    r_spread = np.nanmedian(np.abs(real), axis=0)

    sim = S.Simulator()
    pool = cf.ProcessPoolExecutor(max_workers=8)
    table = []
    for d in [float(v) for v in args.drifts.split(",")]:
        diff = simulated_differences(sim, prop, box, args.view, d, args.n,
                                     args.seed, pool, args.duration, args.paired)
        s_spread = np.nanmedian(np.abs(diff), axis=0)
        ok = (r_spread > 0) & (s_spread > 0)
        score = float(np.median(np.log(r_spread[ok] / s_spread[ok])))
        lo, hi = np.nanquantile(diff, [0.025, 0.975], axis=0)
        outside = float(np.nanmean((real < lo) | (real > hi)))
        table.append({"drift": d, "n_sim": int(len(diff)), "log_ratio": score,
                      "recorded_outside_95": outside})
        print(f"drift {d:.3f}: simulated pairs {len(diff)}, median log(recorded/simulated) "
              f"{score:+.3f}, recorded outside the simulated 95% band {outside:.3f}", flush=True)
    pool.shutdown()
    best = min(table, key=lambda r: abs(r["log_ratio"]))
    source = {"pre-drug": "Charlesworth et al., sister arrays at 6 and 7 days in vitro, "
                          "before any drug",
              "a-c": "Charlesworth et al., never-treated A and C sisters of four-array "
                     f"preparations, {args.min_div:g} to {args.max_div:g} days in vitro",
              "tampere-vehicle": "Tampere vehicle wells, one well before and after vehicle",
              "doorn-null": "Doorn et al. Dynasore wells, two pre-drug stretches"}[args.kind]
    out = {"view": args.view, "chosen": best["drift"], "table": table, "kind": args.kind,
           "paired": bool(args.paired),
           "divs": [args.min_div, args.max_div],
           "n_recorded": int(len(real)), "source": source}
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print("chosen drift", best["drift"], "->", args.out)


if __name__ == "__main__":
    main()
