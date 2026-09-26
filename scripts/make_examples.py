"""Bundle recording pairs for the app, optionally with their analysis cached.

    python scripts/make_examples.py --twin models/twin --cache

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
    ap.add_argument("--twin", default="models/twin")
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
        server.STATE["twin"] = nde.Twin.load(args.twin,
                                             device="cuda" if S.available() else "cpu")
        server.STATE["n_samples"] = 4000
        server.STATE["n_draws"] = args.n_draws
        server.STATE["transient"] = 5.0
        server.STATE["sim"] = S.Simulator() if S.available() else None
        res = ROOT / "results" / "results.json"
        thr = float("inf")
        if res.exists():
            thr = json.loads(res.read_text()).get(
                "guard_calibration", {}).get("threshold", float("inf"))
        server.STATE["threshold"] = thr
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
               "baseline": np.round(p.baseline, 5).tolist(),
               "treated": np.round(p.treated, 5).tolist()}
        if analyse is not None:
            rec["analysis"] = analyse(p.baseline, p.treated, args.duration, label)
        path = OUT / f"{plate}_{comp.lower()}.json"
        path.write_text(json.dumps(rec))
        print(f"{path.name}  {len(p.baseline)} -> {len(p.treated)} events")
    print("wrote", len(wanted), "examples to", OUT)


if __name__ == "__main__":
    main()
