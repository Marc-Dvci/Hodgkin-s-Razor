"""The chip twin's prediction for a four-compartment chip, from simulation only.

    python scripts/brewer_prediction.py --chips 3000

Lassers, Vakilna, Tang and Brewer (2023) recorded hippocampal cultures in four
compartments (EC, DG, CA3, CA1) joined in a loop by 51 microfluidic tunnels,
on a 120-electrode array: 19 electrodes under each compartment and an
electrode pair spanning each of 20 monitored tunnels (five per boundary). The
pair's delay gives each axon's direction: feed-forward (EC > DG > CA3 > CA1 >
EC) or feedback. That chip carries both readouts the chip twin compares, on the
same device: compartment electrodes and channel electrodes.

This script asks the twin, before any spike of that dataset is counted, what
the relation between them should be. Each boundary is simulated as one
two-chamber chip: both chambers self-active, axons growing either way through
the channels, five of them monitored. For every simulated boundary it computes

* `asym`: the asymmetry of the two chambers' population-rate
  cross-correlation, positive when the upstream chamber leads
  (`chip.rate_xcorr`, the same statistic scored on the Mateus chips);
* `traffic_ff`: the share of the monitored axons' spikes that run
  feed-forward (upstream to downstream);
* `axons_ff`: the share of the monitored axons that run feed-forward;
* `true_ff`: the share of all projecting axons that run feed-forward,
* `home`: for the monitored axons of each direction, the peak correlation of
  their spikes with the population rate of the compartment they grow from,
  minus the same with the compartment they grow into,

and the Spearman correlation of `asym` with each, at every number of scored
boundaries from 12 to 36, by resampling simulated boundaries. Which chamber is
upstream is drawn at random, so feed-forward shares span 0 to 1.

Writes results/v4/brewer_prediction.json. PREREGISTRATION_v4.md quotes it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import chip as C
from hodgkins_razor import simulator as S
from chip_study import cultures

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_MONITORED = 5            # monitored tunnels per boundary, one axon each
AUTONOMY = (0.7, 1.0)      # every compartment is a self-active culture


def simulate(args) -> dict:
    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    rows = {k: [] for k in ("asym", "traffic_ff", "axons_ff", "true_ff", "spikes",
                            "rate_up", "rate_down", "chip", "upstream")}
    home = []     # one per (boundary, direction) with enough axon spikes
    tried = step = truncated = 0
    t0 = time.time()
    while len(rows["asym"]) < args.chips:
        theta = cultures(args.batch, rng)
        chips = C.sample_chip_prior(args.batch, rng)
        chips[:, 3] = rng.uniform(*AUTONOMY, size=args.batch)
        seed = int(args.seed + 7919 * step)
        res, mon_fwd, mon_n = C.run_chip(sim, theta, chips, duration_s=args.duration,
                                         transient_s=args.transient, seed=seed,
                                         monitor=N_MONITORED)
        up = rng.integers(0, 2, size=args.batch)     # which chamber is upstream
        for k in range(args.batch):
            if res.truncated[k]:
                truncated += 1
                continue
            ev = res.as_events(k)      # 2 ms dead time, as in chip_study
            a = C.chamber_events(ev, 0)[:, 1]
            b = C.chamber_events(ev, 1)[:, 1]
            e = ev[:, 0].astype(int)
            fwd = float((e == C.ELEC_FWD).sum())
            bwd = float((e == C.ELEC_BWD).sum())
            if a.size < 30 or b.size < 30 or fwd + bwd < args.min_axon_spikes or mon_n[k] < 2:
                continue
            for el, own, other in ((C.ELEC_FWD, a, b), (C.ELEC_BWD, b, a)):
                ax = ev[e == el, 1]
                if ax.size >= args.min_axon_spikes:
                    home.append(C.rate_xcorr(own, ax, args.duration)[0]
                                - C.rate_xcorr(other, ax, args.duration)[0])
            dsel = chips[k, 1]
            if up[k] == 0:
                asym = C.rate_xcorr(a, b, args.duration)[2]
                tff, aff, trff = fwd / (fwd + bwd), mon_fwd[k] / mon_n[k], dsel
                r_up, r_dn = a.size, b.size
            else:
                asym = C.rate_xcorr(b, a, args.duration)[2]
                tff, aff, trff = bwd / (fwd + bwd), 1 - mon_fwd[k] / mon_n[k], 1 - dsel
                r_up, r_dn = b.size, a.size
            rows["asym"].append(asym); rows["traffic_ff"].append(tff)
            rows["axons_ff"].append(aff); rows["true_ff"].append(trff)
            rows["spikes"].append(fwd + bwd)
            rows["rate_up"].append(r_up / args.duration / C.N_COMP_ELEC)
            rows["rate_down"].append(r_dn / args.duration / C.N_COMP_ELEC)
            rows["chip"].append(chips[k]); rows["upstream"].append(up[k])
        tried += args.batch
        step += 1
        print(f"  {len(rows['asym'])}/{args.chips} usable of {tried} "
              f"({truncated} truncated, {time.time() - t0:.0f}s)", flush=True)
    out = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    out["home"] = np.asarray(home, dtype=float)
    out["tried"], out["truncated"] = tried, truncated
    return out


def predictive(x: np.ndarray, y: np.ndarray, n: int, reps: int,
               rng: np.random.Generator) -> dict:
    """Spearman of x with y over `reps` draws of n simulated boundaries."""
    vals = np.array([spearmanr(x[i], y[i])[0]
                     for i in (rng.choice(x.size, size=n, replace=False)
                               for _ in range(reps))])
    vals = vals[np.isfinite(vals)]
    return {"median": float(np.median(vals)),
            "pi90": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))],
            "pi95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "p_positive": float((vals > 0).mean())}


def home_predictive(h: np.ndarray, n: int, reps: int, rng: np.random.Generator) -> dict:
    """Share of units whose axons follow their own compartment, over n units."""
    vals = np.array([(h[rng.choice(h.size, size=n, replace=False)] > 0).mean()
                     for _ in range(reps)])
    return {"median": float(np.median(vals)),
            "pi90": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))],
            "pi95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--duration", type=float, default=300.0)   # Brewer: 300 s
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--min-axon-spikes", type=int, default=50)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default="results/v4/brewer_prediction.json")
    args = ap.parse_args()

    d = simulate(args)
    rng = np.random.default_rng(args.seed + 1)
    whole = {f"asym_vs_{y}": float(spearmanr(d["asym"], d[y])[0])
             for y in ("traffic_ff", "axons_ff", "true_ff")}
    by_n = {}
    for n in range(12, 37):
        by_n[str(n)] = {f"asym_vs_{y}": predictive(d["asym"], d[y], n, args.reps, rng)
                        for y in ("traffic_ff", "axons_ff")}
    home_by_n = {str(n): home_predictive(d["home"], n, args.reps, rng)
                 for n in range(12, 73)}
    out = {"settings": {k: v for k, v in vars(args).items()},
           "n_monitored_axons": N_MONITORED, "target_autonomy": list(AUTONOMY),
           "simulated_boundaries": int(d["asym"].size), "tried": int(d["tried"]),
           "truncated": int(d["truncated"]),
           "whole_bank_spearman": whole,
           "median_rate_hz_per_electrode": float(np.median(d["rate_up"])),
           "median_monitored_axon_spikes": float(np.median(d["spikes"])),
           "predictive_by_n": by_n,
           "home": {"units": int(d["home"].size),
                    "share_positive": float((d["home"] > 0).mean()),
                    "median": float(np.median(d["home"])),
                    "predictive_by_n": home_by_n}}
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "data" / "chip_bank_brewer.npz",
                        **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})
    print(json.dumps({k: out[k] for k in ("simulated_boundaries", "tried", "truncated",
                                          "whole_bank_spearman")}, indent=1))
    print(json.dumps(out["predictive_by_n"]["36"], indent=1))
    print(json.dumps({k: out["home"][k] for k in ("units", "share_positive", "median")}),
          json.dumps(out["home"]["predictive_by_n"]["60"]))


if __name__ == "__main__":
    main()
