"""Fit a proposal that finds parameter sets producing a living culture.

    python scripts/fit_regime.py --screen 12000

Most of the prior produces networks that are silent, saturated, or driven by
membrane noise rather than by their own synapses. In those, blocking a synaptic
conductance changes nothing, so no method can identify what a compound did.
This screens the prior once and fits a classifier over parameters, which the
bank generator then uses as a proposal. The criterion is measured on the
recording, never on the parameters, and the same criterion is reported for the
recorded data so the twin's domain of validity is explicit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P, simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A living, network-driven preparation, judged on the recording alone.
CRITERION = {"mfr_min": 0.3, "mfr_max": 40.0, "active_frac_min": 0.6,
             "nbr_min": 1.0, "psib_min": 5.0}


def is_living(f) -> bool:
    """True for a living, network-driven preparation.

    Accepts either a full feature vector or the dictionary from
    `features.regime_stats`, which computes the same four numbers.
    """
    g = ((lambda n: float(f[n])) if isinstance(f, dict)
         else (lambda n: float(f[F.NAMES.index(n)])))
    c = CRITERION
    return (c["mfr_min"] < g("mfr") < c["mfr_max"]
            and g("active_frac") >= c["active_frac_min"]
            and g("nbr") >= c["nbr_min"]
            and g("psib") >= c["psib_min"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--out", default="models/regime.json")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    TH, Y = [], []
    t0 = time.time()
    for a in range(0, args.screen, args.batch):
        nb = min(args.batch, args.screen - a)
        th = P.sample_prior(nb, rng)
        r = sim.run(th, duration_s=args.duration, transient_s=args.transient,
                    seed=args.seed + a)
        TH.append(th)
        Y.append(np.array([is_living(F.regime_stats(r.as_events(k), S.NELEC,
                                                    args.duration)) for k in range(nb)]))
        done = sum(len(y) for y in Y)
        print(f"  screened {done}/{args.screen}, living {np.concatenate(Y).mean():.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    TH = np.concatenate(TH)
    Y = np.concatenate(Y)

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08)
    model.fit(P.to_unit(TH), Y.astype(int))
    score = model.predict_proba(P.to_unit(TH))[:, 1]
    out = pathlib.Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out.with_suffix(".joblib"))

    stats = {"screened": int(len(Y)), "living_rate": float(Y.mean()),
             "criterion": CRITERION, "duration_s": args.duration,
             "quantiles": {}}
    for q in (0.5, 0.7, 0.8, 0.85, 0.9):
        t = float(np.quantile(score, q))
        sel = score >= t
        stats["quantiles"][f"{q:.2f}"] = {"threshold": t,
                                          "acceptance": float(Y[sel].mean())}
        print(f"  keep top {1 - q:.0%}: acceptance {Y[sel].mean():.3f}")
    out.write_text(json.dumps(stats, indent=1))
    print("wrote", out, "and", out.with_suffix(".joblib"))


if __name__ == "__main__":
    main()
