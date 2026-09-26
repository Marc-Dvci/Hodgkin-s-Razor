"""The prior over what a compound does to a network.

A compound acts on one or two targets, not on everything at once, so the prior
over the parameter shift is sparse: most parameters are drawn from a narrow
component around zero and a small active set is drawn from a wide one. The
narrow component is not a point mass, which keeps the density continuous and
lets a normalising flow represent it.

`inactive_scale` is the width of the narrow component in transformed units. A
parameter whose posterior sits inside that width has not been shown to move.

Version 2 changes, all found on simulations:

* The direction of an active shift is drawn among the directions the baseline
  leaves room for. Version 1 drew it blindly and clipped at the prior bounds,
  so 28 percent of "active" tonic-inhibition labels and 18 percent of "active"
  AMPA labels described shifts that the clipping had removed. The presence
  head was being taught that a mechanism moved when the simulation shows it
  did not.
* A linear parameter's active magnitude is log-uniform, like the fold change
  of a log parameter, instead of uniform over its span. A uniform draw put
  most tonic-inhibition shifts at a strength that silences the culture, so a
  partial agonist effect had almost no support.
* The stored label is the realised shift, and a mechanism is labelled active
  only when that realised shift is at least half the smallest active effect.
"""
from __future__ import annotations

import numpy as np

from . import params as P

INACTIVE_SCALE = 0.02          # ~2 percent, below any biologically read change
N_ACTIVE_WEIGHTS = {0: 0.15, 1: 0.45, 2: 0.25, 3: 0.15}
MIN_LOG_FOLD = np.log(1.50)    # smallest active effect a recording can carry
MAX_LOG_FOLD = np.log(25.0)    # largest active effect: 25-fold
LIN_MIN_FRAC = 0.03            # smallest linear effect, as a fraction of span
LIN_MAX_FRAC = 1.0


def min_effect() -> np.ndarray:
    """Smallest active effect per shiftable parameter, in transformed units."""
    span = (P.THI - P.TLO)[P.SHIFT_IDX]
    return np.where(P.IS_LOG[P.SHIFT_IDX], MIN_LOG_FOLD, LIN_MIN_FRAC * span)


def _magnitude(col: int, rng: np.random.Generator) -> float:
    if P.IS_LOG[col]:
        return float(rng.uniform(MIN_LOG_FOLD, MAX_LOG_FOLD))
    span = P.THI[col] - P.TLO[col]
    return float(np.exp(rng.uniform(np.log(LIN_MIN_FRAC), np.log(LIN_MAX_FRAC))) * span)


def sample_shift(n: int, rng: np.random.Generator,
                 theta_c: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n` shift vectors in transformed space.

    Returns (delta, active) where delta has one column per model parameter
    (zero for parameters a wash-on cannot change) and active is the boolean
    mask over the shiftable columns only. With `theta_c` given (natural units,
    one row per draw), the direction of each active shift is drawn among the
    directions the baseline leaves room for.
    """
    delta = np.zeros((n, P.N_PARAM))
    active = np.zeros((n, P.N_SHIFT), dtype=bool)

    counts = np.array(list(N_ACTIVE_WEIGHTS))
    probs = np.array([N_ACTIVE_WEIGHTS[c] for c in counts], dtype=float)
    probs /= probs.sum()
    k = rng.choice(counts, size=n, p=probs)
    t_c = P.to_unit(theta_c) if theta_c is not None else None
    floor = min_effect()

    for i in range(n):
        # Narrow component everywhere.
        delta[i, P.SHIFT_IDX] = rng.normal(0.0, INACTIVE_SCALE, size=P.N_SHIFT)
        if k[i] == 0:
            continue
        chosen = 0
        for j in rng.permutation(P.N_SHIFT):
            if chosen == k[i]:
                break
            col = P.SHIFT_IDX[j]
            signs = [-1.0, 1.0]
            if t_c is not None:
                room = {-1.0: t_c[i, col] - P.TLO[col], 1.0: P.THI[col] - t_c[i, col]}
                signs = [s for s in signs if room[s] >= floor[j]]
                if not signs:
                    continue
            sign = float(rng.choice(signs))
            mag = _magnitude(col, rng)
            if t_c is not None:
                room_s = (t_c[i, col] - P.TLO[col]) if sign < 0 else (P.THI[col] - t_c[i, col])
                mag = min(mag, room_s)
            delta[i, col] = sign * mag
            active[i, j] = True
            chosen += 1
    return delta, active


def apply_shift(theta_c: np.ndarray, delta: np.ndarray,
                active: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift a baseline parameter set and report the shift that was realised.

    The treated parameters are held inside the prior box, so the realised shift
    is what the clipping left, never what was asked for. Labels therefore always
    describe the simulation that was actually run. With `active` given, the
    mask is returned too, with any mechanism whose realised shift fell below
    half the smallest active effect relabelled as inactive.
    """
    t_c = P.to_unit(theta_c)
    t_t = np.clip(t_c + delta, P.TLO, P.THI)
    realised = t_t - t_c
    if active is None:
        return P.from_unit(t_t), realised
    real = np.abs(realised[:, P.SHIFT_IDX]) >= 0.5 * min_effect()[None, :]
    return P.from_unit(t_t), realised, active & real


def interleave(theta_c: np.ndarray, theta_t: np.ndarray) -> np.ndarray:
    """Stack matched parameter sets so consecutive rows form a pair."""
    out = np.empty((theta_c.shape[0] * 2, theta_c.shape[1]))
    out[0::2] = theta_c
    out[1::2] = theta_t
    return out
