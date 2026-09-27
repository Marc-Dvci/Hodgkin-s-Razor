"""Score the published estimator of Doorn et al. (Commun Biol 2025) on our pairs.

    .venv-doorn/Scripts/python scripts/prior_art_doorn.py --dataset tampere
    .venv-doorn/Scripts/python scripts/prior_art_doorn.py --dataset doorn
    .venv-doorn/Scripts/python scripts/prior_art_doorn.py --dataset charlesworth

Runs in its own environment (`requirements-doorn.txt`: Python 3.9, sbi 0.21,
torch 1.13, brian2), because the trained estimator is a pickled sbi 0.21
object. It uses their own feature code (`FeatureExtraction.compute_features`)
and their own trained network (`TrainedNDE`), both from
gitlab.utwente.nl/m7706783/SBI_MEA_model (Apache-2.0), unchanged.

Their model reads one recording at a time, on 12 electrodes. Every window of
a pair is read on the 12-electrode layout (the 4 x 4 grid without its
corners for the 16-electrode Tampere wells), and the posterior median and
spread of every parameter are written for the baseline and the treated
recording. Scoring is done by `scripts/evaluate_v2.py` with the mapping fixed
in PREREGISTRATION_v2.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DOORN = ROOT / "data" / "raw" / "doorn_sbi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DOORN))

PARAMS = ["noise", "g_Na", "g_K", "g_AHP", "g_AMPA", "g_NMDA", "conn",
          "tau_D", "U_STD", "U_asyn"]
FS = 10000
KEEP16 = [e for e in range(16) if e not in (0, 3, 12, 15)]


def to_samples(events: np.ndarray, n_elec: int, duration: float) -> np.ndarray:
    """[electrode, time in s] to their [electrode, sample index] on 12 electrodes.

    Their feature code sizes its arrays by the electrodes that fired and then
    indexes them by electrode number, so it fails when an electrode is silent.
    Such an electrode is given one spike at the last sample of the window,
    which leaves their code unchanged.
    """
    e = events[:, 0].astype(int) if events.size else np.zeros(0, dtype=int)
    t = events[:, 1] if events.size else np.zeros(0)
    if n_elec == 16:
        remap = np.full(16, -1)
        remap[KEEP16] = np.arange(12)
        e = remap[e]
    ok = e >= 0
    e, s = e[ok], np.round(t[ok] * FS).astype(int)
    silent = sorted(set(range(12)) - set(e.tolist()))
    last = int(duration * FS) - 1
    e = np.concatenate([e, np.array(silent, dtype=int)])
    s = np.concatenate([s, np.full(len(silent), last, dtype=int)])
    out = np.stack([e, s], axis=1)
    return out[np.argsort(out[:, 1], kind="stable")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tampere", "doorn", "charlesworth"], required=True)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from brian2 import second
    from FeatureExtraction import compute_features

    with open(DOORN / "TrainedNDE", "rb") as f:
        posterior = pickle.load(f)

    if args.dataset == "tampere":
        from hodgkins_razor import tampere as T
        pairs = T.load_all() + [p for n in T.PLATES for p in T.load_ttx(n)]
    elif args.dataset == "doorn":
        from hodgkins_razor import doorn as D
        pairs = D.load(treated=True)
    else:
        # The protocol of PREREGISTRATION_v3: sister pairs at 9 to 15 days, the
        # first 60 s window of two diagonal quadrants, because their estimator
        # takes seconds per recording.
        from hodgkins_razor import charlesworth as C
        pairs = C.load(kinds=("treated", "null"), min_div=9.0, max_div=15.0,
                       n_windows=1, quadrants=(0, 3))

    rows = []
    for k, p in enumerate(pairs):
        rec = {"plate": p.plate, "species": p.species, "well": p.well,
               "compound": p.compound}
        if args.dataset == "charlesworth":
            rec.update({"prep": p.prep, "div": p.div, "kind": p.kind})
        for side, ev in (("baseline", p.baseline), ("treated", p.treated)):
            aps = to_samples(ev, p.n_elec, p.duration)
            if aps.shape[0] < 20:
                rec[side] = None
                continue
            # Their code drops everything before `transient`; the windows are
            # already cut, so a transient of one sample keeps all of it.
            x = compute_features(aps.copy(), p.duration * second, (1.0 / FS) * second, FS)
            x = np.nan_to_num(np.asarray(x, dtype=float))
            posterior.set_default_x(torch.as_tensor(x, dtype=torch.float32))
            s = posterior.sample((args.samples,), show_progress_bars=False).numpy()
            rec[side] = {"features": x.tolist(),
                         "median": np.median(s, 0).tolist(),
                         "sd": s.std(0).tolist()}
        rows.append(rec)
        if k % 20 == 0:
            print(f"  {k + 1}/{len(pairs)}", flush=True)
    out = ROOT / "results" / f"prior_art_doorn_{args.dataset}.json"
    out.write_text(json.dumps({"params": PARAMS, "rows": rows}))
    print("wrote", out)


if __name__ == "__main__":
    main()
