"""Posterior predictive check, and the guard that refuses to name a mechanism.

A posterior is only worth reading if the twin, run at those parameters, produces
a recording like the one that was measured. `check` re-simulates from posterior
draws and scores how far the measured features sit from the predicted spread.

The threshold is calibrated on held-out bank records, where the model generated
the data and is correct by construction. A recording scoring past that
threshold is outside the model, and the report then names no mechanism. This is
what a compound acting through a target the twin does not carry looks like.
"""
from __future__ import annotations

import concurrent.futures as cf
import os

import numpy as np

from . import features as F
from . import nde
from . import params as P
from . import shift as SH

DEFAULT_DRAWS = 48

_POOL: cf.ProcessPoolExecutor | None = None


def _feature_job(args):
    events, n_elec, duration = args
    return F.compute(events, n_elec, duration)


def _pool() -> cf.ProcessPoolExecutor | None:
    """Shared worker pool for the predictive draws.

    A check extracts features from roughly a hundred recordings, and the
    evaluation runs hundreds of checks, so this is where the wall-clock goes.
    """
    global _POOL
    if _POOL is None and os.environ.get("HR_NO_POOL") != "1":
        try:
            _POOL = cf.ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1))
        except Exception:
            return None
    return _POOL


def theta_pair_from_posterior(post: dict, n_draws: int,
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw matched (baseline, treated) parameter sets from a posterior."""
    n = post["theta_c"].shape[0]
    pick = rng.choice(n, size=min(n_draws, n), replace=n < n_draws)
    theta_c = post["theta_c"][pick]
    delta = np.zeros((len(pick), P.N_PARAM))
    delta[:, P.SHIFT_IDX] = post["delta"][pick]
    t_t = np.clip(theta_c + delta, P.TLO, P.THI)
    return P.from_unit(theta_c), P.from_unit(t_t)


def predict(sim, post: dict, duration: float, transient: float,
            n_draws: int = DEFAULT_DRAWS, seed: int = 0) -> dict:
    """Simulate the twin at posterior draws and return predicted features."""
    rng = np.random.default_rng(seed)
    nat_c, nat_t = theta_pair_from_posterior(post, n_draws, rng)
    stacked = SH.interleave(nat_c, nat_t)
    res = sim.run(stacked, duration_s=duration, transient_s=transient,
                  seed=seed, pair=True)
    n_elec = sim_n_elec(sim)
    jobs = [(res.as_events(i), n_elec, duration) for i in range(stacked.shape[0])]
    pool = _pool()
    if pool is not None:
        feats = np.stack(list(pool.map(_feature_job, jobs, chunksize=8)))
    else:
        feats = np.stack([_feature_job(j) for j in jobs])
    return {"x_base": feats[0::2], "x_treat": feats[1::2], "result": res,
            "theta_base": nat_c, "theta_treat": nat_t}


def sim_n_elec(sim) -> int:
    from . import simulator as S
    return S.NELEC


TAIL = 0.02


def discrepancy(x_base: np.ndarray, x_treat: np.ndarray,
                pred_base: np.ndarray, pred_treat: np.ndarray,
                tail: float = TAIL) -> float:
    """How far the measured recording sits from its own predictive spread.

    The statistic counts how many of the eighty numbers fall outside the
    central interval of what the fitted twin predicts. Counting is used rather
    than a distance because some features barely vary across predictive draws,
    and dividing by that spread produced scores in the hundreds for a feature
    that was merely constant. Two earlier statistics failed on this: a mean
    over all eighty buried a mismatch confined to a few, and a mean over the
    worst few inherited the scale explosion. A count has neither problem and is
    bounded by the number of features.
    """
    obs = np.concatenate([nde.phi(np.atleast_2d(x_base)),
                          nde.phi(np.atleast_2d(x_treat))], axis=1).ravel()
    pred = np.concatenate([nde.phi(pred_base), nde.phi(pred_treat)], axis=1)
    lo = np.quantile(pred, tail, axis=0)
    hi = np.quantile(pred, 1.0 - tail, axis=0)
    # A feature that never moves across draws would flag on any rounding, so a
    # band is opened around it in proportion to its own size.
    pad = 1e-3 + 0.01 * np.abs(np.median(pred, axis=0))
    return float(np.sum((obs < lo - pad) | (obs > hi + pad)))


def calibrate_threshold(twin, sim, bank: dict, index: np.ndarray,
                        duration: float, transient: float,
                        n_records: int = 120, n_draws: int = DEFAULT_DRAWS,
                        quantile: float = 0.95, seed: int = 0) -> dict:
    """Null distribution of the discrepancy on held-out, well-specified records."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(index, size=min(n_records, len(index)), replace=False)
    vals = []
    for k, i in enumerate(pick):
        post = twin.posterior(bank["x_base"][i], bank["x_treat"][i], n_samples=1500)
        pred = predict(sim, post, duration, transient, n_draws=n_draws, seed=1000 + k)
        vals.append(discrepancy(bank["x_base"][i], bank["x_treat"][i],
                                pred["x_base"], pred["x_treat"]))
    vals = np.array(vals)
    return {"values": vals.tolist(),
            "threshold": float(np.quantile(vals, quantile)),
            "quantile": quantile, "n_records": int(len(vals))}


def check(twin, sim, x_base: np.ndarray, x_treat: np.ndarray, threshold: float,
          duration: float, transient: float, n_draws: int = DEFAULT_DRAWS,
          seed: int = 0, post: dict | None = None) -> dict:
    """Run the check for one paired recording and return the verdict."""
    post = post if post is not None else twin.posterior(x_base, x_treat)
    pred = predict(sim, post, duration, transient, n_draws=n_draws, seed=seed)
    d = discrepancy(x_base, x_treat, pred["x_base"], pred["x_treat"])
    # The posterior over a sixty second recording is wide, so a draw taken at
    # random rarely looks like the measurement even when the fit is sound.
    # For display, the draw closest in firing rate is named; the verdict is
    # still computed from the whole predictive spread.
    obs = np.array([np.log1p(max(np.atleast_1d(x_base)[0], 0.0)),
                    np.log1p(max(np.atleast_1d(x_treat)[0], 0.0))])
    got = np.stack([np.log1p(np.clip(pred["x_base"][:, 0], 0, None)),
                    np.log1p(np.clip(pred["x_treat"][:, 0], 0, None))], axis=1)
    closest = int(np.argmin(np.abs(got - obs).sum(axis=1)))
    return {"discrepancy": d, "threshold": float(threshold),
            "inside_model": bool(d <= threshold),
            "pred_base": pred["x_base"], "pred_treat": pred["x_treat"],
            "result": pred["result"], "theta_base": pred["theta_base"],
            "theta_treat": pred["theta_treat"],
            "closest_draw": closest, "n_draws": int(pred["x_base"].shape[0])}
