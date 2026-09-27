"""What each version 2 method names when nothing was applied. Post hoc.

    python scripts/null_controls_v2.py                       # both twins
    .venv-doorn/Scripts/python scripts/null_controls_v2.py --prior-art

Run after the second pre-registration was scored, so none of this is
pre-registered. The blind Dynasore test scored top-1 against a chance of 2 in
10. A ranking with a fixed preference can land on an answer key without
reading the recording, so the fair chance level of a method is its own hit
rate on pairs where nothing was applied.

Null pairs come from the pre-drug stretch of every Doorn well: windows from 0
to 180 s against windows from 240 to 420 s. The per-minute event counts show
the drug going on after 420 s in every well. Results go to
results/v2/null_controls.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "v2" / "null_controls.json"
ACCEPT = {"u_rel", "tau_d"}


def null_pairs():
    from hodgkins_razor import doorn as D
    out = []
    for well, (fname, _, _) in D.WELLS.items():
        ev = D._load(fname)
        for w in range(3):
            out.append(D.Pair(plate=D.PLATE_OF[well], species="hiPSC-Ngn2", well=well,
                              compound="none", dose="",
                              baseline=D._cut(ev, 60.0 * w, 60.0 * (w + 1)),
                              treated=D._cut(ev, 240.0 + 60.0 * w, 300.0 + 60.0 * w),
                              duration=60.0, n_elec=12))
    return out


def twins() -> dict:
    import evaluate_v2 as E
    from hodgkins_razor import nde
    pairs = null_pairs()
    twin = nde.Twin.load(ROOT / "models/twin_v2_grid12", device="cuda")
    unp = nde.UnpairedTwin.load(ROOT / "models/twin_v2_unpaired_grid12", device="cuda")
    out = {}
    for name, wells in (("paired", E.by_well(E.score_pairs(twin, pairs, 4000))),
                        ("unpaired", E.unpaired_wells(unp, pairs, 4000))):
        out[name] = {"wells": [{"well": w["well"], "top1": w["top1"],
                                "p_top1": float(max(w["p_active"])),
                                "called": bool(w["called"])} for w in wells]}
    return out


def prior_art() -> dict:
    import pickle
    import torch
    from brian2 import second
    sys.path.insert(0, str(ROOT / "data" / "raw" / "doorn_sbi"))
    from FeatureExtraction import compute_features
    from prior_art_doorn import FS, PARAMS, to_samples
    # The mapping of scripts/evaluate_v2.py, which this environment cannot import.
    PRIOR_ART_MAP = {"g_Na": "g_na", "g_K": "g_kdr", "g_AHP": "g_ahp", "g_AMPA": "g_ampa",
                     "g_NMDA": "g_nmda", "tau_D": "tau_d", "U_STD": "u_rel",
                     "U_asyn": "u_asyn"}
    post = pickle.load(open(ROOT / "data/raw/doorn_sbi/TrainedNDE", "rb"))
    cand = [PARAMS.index(k) for k in PRIOR_ART_MAP]
    zs: dict[str, list] = {}
    for p in null_pairs():
        r = {}
        for side, ev in (("a", p.baseline), ("b", p.treated)):
            x = compute_features(to_samples(ev, 12, 60.0).copy(), 60 * second,
                                 (1.0 / FS) * second, FS)
            post.set_default_x(torch.as_tensor(np.nan_to_num(np.asarray(x, float)),
                                               dtype=torch.float32))
            s = post.sample((2000,), show_progress_bars=False).numpy()
            r[side] = (np.median(s, 0), s.std(0))
        zs.setdefault(p.well, []).append(
            (r["b"][0] - r["a"][0]) / np.sqrt(r["a"][1] ** 2 + r["b"][1] ** 2 + 1e-12))
    wells = []
    for well, z in zs.items():
        z = np.mean(z, 0)
        top = max(cand, key=lambda i: abs(z[i]))
        wells.append({"well": well, "top1": PRIOR_ART_MAP[PARAMS[top]],
                      "direction": "up" if z[top] > 0 else "down",
                      "abs_z": float(abs(z[top]))})
    return {"wells": wells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-art", action="store_true",
                    help="score the Doorn et al. estimator (needs .venv-doorn)")
    args = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["note"] = ("Post hoc. Pre-drug windows 0-180 s against 240-420 s of each "
                   "Doorn well; chance of a method = its hit rate here.")
    res.update({"prior_art": prior_art()} if args.prior_art else twins())
    for name in ("paired", "unpaired", "prior_art"):
        if name in res:
            w = res[name]["wells"]
            res[name]["null_hits"] = int(sum(x["top1"] in ACCEPT for x in w))
            res[name]["n"] = len(w)
            if "called" in w[0]:
                res[name]["null_called"] = int(sum(x["called"] for x in w))
    OUT.write_text(json.dumps(res, indent=1))
    print({k: {kk: v[kk] for kk in ("null_hits", "n", "null_called") if kk in v}
           for k, v in res.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
