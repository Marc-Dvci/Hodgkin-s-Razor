"""Bundle recording pairs for the app, optionally with their analysis cached.

    python scripts/make_examples.py --cache

Without --cache the files hold only the events, and the app analyses them live.
With --cache each file also holds the finished analysis, so the page works on a
host with no CUDA device.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import tampere as T

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "examples"

WANTED = [("rat", "CNQX"), ("rat", "GABA"), ("rat", "Gabazine"),
          ("rat", "DAP5"), ("rat", "KainicAcid"), ("rat", "Control"),
          ("human", "CNQX"), ("human", "Gabazine"), ("human", "GABA"),
          ("human", "D-AP5"), ("human", "KainicAcid"), ("human", "Control")]

PRETTY = {"DAP5": "D-AP5", "D-AP5": "D-AP5", "KainicAcid": "kainic acid",
          "CNQX": "CNQX", "GABA": "GABA", "Gabazine": "gabazine",
          "Control": "vehicle control", "TTX": "TTX"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", default="models/twin_v2_grid16,models/twin_v2_grid12")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--n-draws", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for plate in ("rat", "human"):
        for p in T.load_plate(plate, window_s=args.duration, n_windows=1):
            pairs.setdefault((plate, p.compound), p)
        for p in T.load_ttx(plate, window_s=args.duration, n_windows=1):
            pairs.setdefault((plate, "TTX"), p)

    analyse = None
    if args.cache:
        sys.path.insert(0, str(ROOT / "app"))
        import server
        from hodgkins_razor import nde, simulator as S
        dev = "cuda" if S.available() else "cpu"
        server.STATE["twins"] = {}
        for path in args.twins.split(","):
            t = nde.Twin.load(ROOT / path, device=dev)
            server.STATE["twins"][t.meta.get("view", "grid16")] = t
        server.STATE["n_samples"] = 4000
        server.STATE["n_draws"] = args.n_draws
        server.STATE["transient"] = 5.0
        server.STATE["sim"] = S.Simulator() if S.available() else None
        res = ROOT / "results" / "v2" / "results.json"
        server.STATE["threshold"] = (
            {v: g["ppc"] for v, g in json.loads(res.read_text()).get("guard_thresholds", {}).items()}
            if res.exists() else {})
        analyse = server.analyse

    wanted = [w for w in WANTED if w in pairs] + \
             [(pl, "TTX") for pl in ("rat", "human") if (pl, "TTX") in pairs]
    for plate, comp in wanted:
        p = pairs[(plate, comp)]
        name = PRETTY.get(comp, comp)
        label = f"{'Rat cortical DIV22' if plate == 'rat' else 'Human iPSC DIV29'} - {name}"
        rec = {"label": label, "compound": name, "species": p.species,
               "well": p.well, "dose": p.dose, "duration": args.duration,
               "source": "Tampere comparative MEA dataset, CC BY 4.0",
               "n_elec": 16, "system": "Axion Maestro, 16 electrodes",
               "baseline": np.round(p.baseline, 5).tolist(),
               "treated": np.round(p.treated, 5).tolist()}
        if analyse is not None:
            rec["analysis"] = analyse(p.baseline, p.treated, args.duration, label, n_elec=16)
        path = OUT / f"{plate}_{comp.lower()}.json"
        path.write_text(json.dumps(rec))
        print(f"{path.name}  {len(p.baseline)} -> {len(p.treated)} events")
    # The blind-test compound, available once the second pre-registration is in
    # place (the reader refuses before that).
    try:
        from hodgkins_razor import doorn as D
        dyn = [p for p in D.load(treated=True) if p.well == "FB2t_B1"]
    except SystemExit:
        dyn = []
    if dyn:
        p = dyn[0]
        label = "Human iPSC Ngn2 DIV35 (Doorn et al.) - Dynasore"
        rec = {"label": label, "compound": "Dynasore", "species": p.species,
               "well": p.well, "dose": p.dose, "duration": p.duration, "n_elec": 12,
               "system": "MCS 24-well, 12 electrodes",
               "source": "Doorn et al. 2024, gitlab.utwente.nl/m7706783/fb_model, Apache-2.0",
               "baseline": np.round(p.baseline, 5).tolist(),
               "treated": np.round(p.treated, 5).tolist()}
        if analyse is not None:
            rec["analysis"] = analyse(p.baseline, p.treated, p.duration, label, n_elec=12)
        (OUT / "human_dynasore.json").write_text(json.dumps(rec))
        print("human_dynasore.json")
    print("wrote", len(wanted) + (1 if dyn else 0), "examples to", OUT)


if __name__ == "__main__":
    main()
