"""The prior over what a compound does to a network.

A compound acts on one or two targets, not on everything at once, so the prior
over the parameter shift is sparse: most parameters are drawn from a narrow
component around zero and a small active set is drawn from a wide one. The
narrow component is not a point mass, which keeps the density continuous and
lets a normalising flow represent it.

`inactive_scale` is the width of the narrow component in transformed units. A
parameter whose posterior sits inside that width has not been shown to move.
"""
from __future__ import annotations

import numpy as np

from . import params as P

INACTIVE_SCALE = 0.02          # ~2 percent, below any biologically read change
N_ACTIVE_WEIGHTS = {0: 0.15, 1: 0.45, 2: 0.25, 3: 0.15}
MIN_LOG_FOLD = np.log(1.50)    # smallest active effect a recording can carry
MAX_LOG_FOLD = np.log(25.0)    # largest active effect: 25-fold


def sample_shift(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n` shift vectors in transformed space.

    Returns (delta, active) where delta has one column per model parameter
    (zero for parameters a wash-on cannot change) and active is the boolean
    mask over the shiftable columns only.
    """
    delta = np.zeros((n, P.N_PARAM))
    active = np.zeros((n, P.N_SHIFT), dtype=bool)

    counts = np.array(list(N_ACTIVE_WEIGHTS))
    probs = np.array([N_ACTIVE_WEIGHTS[c] for c in counts], dtype=float)
    probs /= probs.sum()
    k = rng.choice(counts, size=n, p=probs)

    span = P.THI - P.TLO
    for i in range(n):
        # Narrow component everywhere.
        delta[i, P.SHIFT_IDX] = rng.normal(0.0, INACTIVE_SCALE, size=P.N_SHIFT)
        if k[i] == 0:
            continue
        pick = rng.choice(P.N_SHIFT, size=int(k[i]), replace=False)
        for j in pick:
            col = P.SHIFT_IDX[j]
            sign = rng.choice([-1.0, 1.0])
            if P.IS_LOG[col]:
                mag = rng.uniform(MIN_LOG_FOLD, MAX_LOG_FOLD)
            else:
                mag = rng.uniform(0.03, 1.0) * span[col]
            delta[i, col] = sign * mag
            active[i, j] = True
    return delta, active


def apply_shift(theta_c: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift a baseline parameter set and report the shift that was realised.

    The treated parameters are held inside the prior box, so the realised shift
    is what the clipping left, never what was asked for. Labels therefore always
    describe the simulation that was actually run.
    """
    t_c = P.to_unit(theta_c)
    t_t = np.clip(t_c + delta, P.TLO, P.THI)
    return P.from_unit(t_t), t_t - t_c


def interleave(theta_c: np.ndarray, theta_t: np.ndarray) -> np.ndarray:
    """Stack matched parameter sets so consecutive rows form a pair."""
    out = np.empty((theta_c.shape[0] * 2, theta_c.shape[1]))
    out[0::2] = theta_c
    out[1::2] = theta_t
    return out
