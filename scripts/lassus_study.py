"""The cortico-striatal chip of Lassus et al. (2018), in the chip twin.

    python scripts/lassus_study.py

Lassus B., Naudé J., Faure P., Guedin D., Von Boxberg Y., Mannoury la Cour C.,
Millan M.J., Peyrin J.-M. "Glutamatergic and dopaminergic modulation of
cortico-striatal circuits probed by dynamic calcium imaging of networks
reconstructed in microfluidic chips." Sci Rep 8:17461 (2018).

Their protocol, as the paper states it:

* cortical neurons in one chamber, striatal neurons in the other, joined by
  axon diodes; unconnected striatal neurons show no spontaneous oscillations;
* the cortical chamber is stimulated throughout with bicuculline, 4-AP and
  nimodipine;
* drugs are perfused into the striatal chamber only;
* each neuron is imaged at 2 Hz; a calcium event is a crossing of the mean
  plus one standard deviation of its detrended signal; synchrony is the
  SPIKE-synchronisation index of Kreuz et al. over those events.

Their result for GluN2B antagonists (ifenprodil, RO256981) in the striatal
chamber: striatal event frequency down, striato-striatal synchrony down and
cortico-striatal synchrony down. That direction is fixed here before the
simulation is run.

In the twin: the source is a cortical culture from the Tampere domain with
bicuculline (synaptic GABA-A x0.05) and 4-AP (delayed rectifier x0.25) in its
chamber; nimodipine has no counterpart. The target is striatal: every neuron
GABAergic, with little drive of its own (`tgt_autonomy` 0.05 to 0.3). The
drug is an NMDA reduction on the target chamber's neurons only. The model has
no NMDA subunits and no dopamine receptors, so GluN2A against GluN2B and the
D1 and D2/D3 results are outside it. Electrodes stand in for imaged neurons,
seven per chamber. The same chips with a generic, partly excitatory target are
run as the contrast: the published effect should need the striatal identity.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import chip as C, params as P, simulator as S
from fit_domain import GaussianProposal

ROOT = pathlib.Path(__file__).resolve().parents[1]
COCKTAIL = {0: {"g_gaba": 0.05, "g_kdr": 0.25}}      # cortical chamber, throughout


def roi_traces(events: np.ndarray, duration: float, compartment: int,
               rng: np.random.Generator) -> np.ndarray:
    """One 2 Hz calcium trace per electrode of a chamber (the ROIs)."""
    n_frames = int(duration * C.CA_FRAME_HZ)
    fine = 100.0
    k_len = int(6 * C.CA_DECAY_S * fine)
    tk = np.arange(k_len) / fine
    kernel = np.clip(np.exp(-tk / C.CA_DECAY_S) - np.exp(-tk / C.CA_RISE_S), 0, None)
    kernel /= kernel.sum()
    step = int(fine / C.CA_FRAME_HZ)
    out = np.zeros((C.N_COMP_ELEC, n_frames))
    ev = C.chamber_events(events, compartment)
    for r in range(C.N_COMP_ELEC):
        train = np.zeros(int(duration * fine))
        t = ev[ev[:, 0] == r, 1]
        np.add.at(train, np.clip((t * fine).astype(int), 0, train.size - 1), 1.0)
        tr = np.convolve(train, kernel)[:train.size]
        fr = tr[:n_frames * step].reshape(n_frames, step).mean(1)
        out[r] = fr + rng.normal(0, 0.02 * max(fr.std(), 1e-3), n_frames)
    return out


def peaks(trace: np.ndarray) -> np.ndarray:
    """Frames where the trace crosses the mean plus one standard deviation upward."""
    th = trace.mean() + trace.std()
    above = trace > th
    return np.flatnonzero(above & ~np.r_[False, above[:-1]]) / C.CA_FRAME_HZ


def spike_sync(a: np.ndarray, b: np.ndarray) -> float:
    """SPIKE-synchronisation of two event trains (Kreuz et al. 2015)."""
    if a.size == 0 or b.size == 0:
        return np.nan

    def gaps(x: np.ndarray, k: int) -> list[float]:
        return [x[k] - x[k - 1] if k > 0 else np.inf,
                x[k + 1] - x[k] if k + 1 < x.size else np.inf]

    def coincident(x: np.ndarray, y: np.ndarray) -> int:
        # An event of x is coincident when the nearest event of y lies within
        # half the smallest neighbouring interval of either train.
        c = 0
        for k, t in enumerate(x):
            j = int(np.searchsorted(y, t))
            q = min((q for q in (j - 1, j) if 0 <= q < y.size), key=lambda q: abs(y[q] - t))
            tau = 0.5 * min(gaps(x, k) + gaps(y, q))
            c += abs(y[q] - t) <= tau
        return c

    return (coincident(a, b) + coincident(b, a)) / (a.size + b.size)


def readout(events: np.ndarray, duration: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    src, tgt = roi_traces(events, duration, 0, rng), roi_traces(events, duration, 1, rng)
    ps, pt = [peaks(x) for x in src], [peaks(x) for x in tgt]
    ss = [spike_sync(pt[i], pt[j]) for i in range(len(pt)) for j in range(i + 1, len(pt))]
    cs = [spike_sync(a, b) for a in ps for b in pt]
    return {"tgt_freq": float(np.mean([p.size for p in pt]) * 60.0 / duration),
            "tgt_rate_hz": float(C.chamber_events(events, 1).shape[0] / duration / C.N_COMP_ELEC),
            "striato_striatal_sync": float(np.nanmean(ss)) if np.isfinite(ss).any() else np.nan,
            "cortico_striatal_sync": float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan}


def compare(base: list[dict], drug: list[dict]) -> dict:
    from scipy.stats import wilcoxon
    out = {}
    for k in ("tgt_freq", "tgt_rate_hz", "striato_striatal_sync", "cortico_striatal_sync"):
        a = np.array([r[k] for r in base]); b = np.array([r[k] for r in drug])
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
        a, b = a[ok], b[ok]
        if ok.sum() < 5:
            out[k] = {"n": int(ok.sum())}
            continue
        d = b - a
        out[k] = {"n": int(ok.sum()), "fraction_lower": float(np.mean(d < 0)),
                  "median_change_percent": float(np.median(d / a) * 100.0),
                  "wilcoxon_p_lower": float(wilcoxon(d, alternative="less")[1])
                  if np.any(d != 0) else 1.0}
    return out


SILENT_FRACTION = 0.05     # unconnected striatum: at most 5% of its driven rate
ACTIVE_SHARE = 0.5         # and still active when driven, in at least half the chips
MIN_CHIPS = 40             # chips meeting the precondition chip by chip, for a paired test


def calibrate(sim, theta, chips, args) -> None:
    """Set the down-state bias from a published precondition, with no drug.

    Lassus et al. report that striatal neurons with no cortical partner show
    no spontaneous oscillations. The chosen bias is the weakest for which the
    median striatal rate with the cortex held silent is at most 5% of the
    median rate with the cortex driving, while the striatum is still active
    (above 0.05 Hz) when driven in at least half the chips. No NMDA condition
    is simulated here.
    """
    table = []
    for b in [float(x) for x in args.calibrate.split(",")]:
        kw = dict(duration_s=args.duration, transient_s=5.0, seed=args.seed,
                  striatal=True, perfuse=COCKTAIL, down_state_pa=b)
        on = C.run_chip(sim, theta, chips, **kw)
        off = C.run_chip(sim, theta, chips, silence=np.zeros(args.chips), **kw)
        n = C.N_COMP_ELEC * args.duration
        r_on = np.array([C.chamber_events(on.as_events(k), 1).shape[0] for k in range(args.chips)]) / n
        r_off = np.array([C.chamber_events(off.as_events(k), 1).shape[0] for k in range(args.chips)]) / n
        act = r_on > 0.05
        cond = act & (r_off <= SILENT_FRACTION * r_on)
        row = {"down_state_pa": b, "active_share": float(act.mean()),
               "precondition_share": float(cond.mean()),
               "precondition_chips": int(cond.sum()),
               "rate_driven_median": float(np.median(r_on[act])) if act.any() else 0.0,
               "rate_silent_median": float(np.median(r_off[act])) if act.any() else 0.0}
        row["silent_over_driven"] = (row["rate_silent_median"] / row["rate_driven_median"]
                                     if row["rate_driven_median"] > 0 else float("nan"))
        row["meets"] = bool(row["active_share"] >= ACTIVE_SHARE
                            and row["silent_over_driven"] <= SILENT_FRACTION)
        table.append(row)
        print(json.dumps(row), flush=True)
    # Chip by chip: the bias that leaves the most chips meeting the precondition
    # (striatum active when the cortex drives it, silent when it does not).
    chosen = max(table, key=lambda r: (r["precondition_share"], r["down_state_pa"]))
    if chosen["precondition_share"] <= 0:
        chosen = None
    out = {"rule": {"silent_fraction": SILENT_FRACTION, "active_share": ACTIVE_SHARE,
                    "per_chip": "active when driven (> 0.05 Hz) and silent rate at most "
                                "5% of driven", "min_chips": MIN_CHIPS},
           "chips": args.chips, "duration_s": args.duration, "table": table,
           "chosen_down_state_pa": chosen["down_state_pa"] if chosen else None,
           "chips_to_draw": (int(np.ceil(1.25 * MIN_CHIPS / chosen["precondition_share"]))
                             if chosen else None)}
    (ROOT / "results" / "lassus_calibration.json").write_text(json.dumps(out, indent=1))
    print("chosen", out["chosen_down_state_pa"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=192)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--nmda", default="0.3,0.5,0.15",
                    help="NMDA factors on the striatal chamber; the first is the headline")
    ap.add_argument("--seed", type=int, default=2018)
    ap.add_argument("--down-state", type=float, default=0.0,
                    help="hyperpolarising bias on striatal neurons, pA (0: the first run)")
    ap.add_argument("--calibrate", default="",
                    help="comma-separated down-state biases: set the bias from the "
                         "paper's precondition (unconnected striatum silent) with no drug")
    ap.add_argument("--precondition", action="store_true",
                    help="score only chips whose striatum is active when driven and "
                         "silent (at most 5% of its driven rate) with the cortex held silent")
    ap.add_argument("--out", default="results/lassus_study.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dom = json.loads((ROOT / "models" / "domain.json").read_text())
    theta = GaussianProposal.load(dom["views"]["grid16"]["proposal"]).draw(args.chips, rng)
    chips = C.sample_chip_prior(args.chips, rng)
    chips[:, 1] = rng.uniform(0.85, 1.0, args.chips)       # axon diodes
    chips[:, 3] = rng.uniform(0.05, 0.3, args.chips)       # striatum not self-active
    sim = S.Simulator()
    if args.calibrate:
        calibrate(sim, theta, chips, args)
        return
    res = {"protocol": __doc__.split("\n\n")[2:6], "chips": args.chips,
           "duration_s": args.duration, "down_state_pa": args.down_state,
           "precondition": bool(args.precondition)}
    for label, striatal in (("striatal_target", True), ("generic_target", False)):
        def run(perf, silence=None):
            r = C.run_chip(sim, theta, chips, duration_s=args.duration, transient_s=5.0,
                           seed=args.seed, striatal=striatal, perfuse=perf,
                           silence=silence, down_state_pa=args.down_state)
            return [readout(r.as_events(k), args.duration, k) for k in range(args.chips)]
        base = run(COCKTAIL)
        alone = run(COCKTAIL, silence=np.zeros(args.chips))   # cortex held silent
        keep = [k for k in range(args.chips) if base[k]["tgt_rate_hz"] > 0.05
                and (not args.precondition or not striatal
                     or alone[k]["tgt_rate_hz"] <= SILENT_FRACTION * base[k]["tgt_rate_hz"])]
        sec = {"n_active_chips": len(keep),
               "target_rate_hz_median": float(np.median([base[k]["tgt_rate_hz"] for k in keep])),
               "target_rate_without_cortex_hz_median":
                   float(np.median([alone[k]["tgt_rate_hz"] for k in keep])),
               "nmda": {}}
        for f in [float(x) for x in args.nmda.split(",")]:
            perf = {0: COCKTAIL[0], 1: {"g_nmda": f}}
            drug = run(perf)
            sec["nmda"][f"{f:g}"] = compare([base[k] for k in keep], [drug[k] for k in keep])
        res[label] = sec
        h = sec["nmda"][args.nmda.split(",")[0]]
        print(label, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()}
                                 for k, v in h.items()}), flush=True)
    res["published"] = {"source": "Lassus et al. 2018, GluN2B antagonists in the striatal chamber",
                        "tgt_freq": "lower", "striato_striatal_sync": "lower",
                        "cortico_striatal_sync": "lower"}
    out = ROOT / args.out
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
