"""Which experiment to run next.

When a recording leaves two mechanisms tied, the useful output is not a
confident guess. It is the follow-up that separates them. This module takes a
posterior, splits it into the competing hypotheses, simulates what each would
predict under a set of candidate follow-up experiments, and ranks the
candidates by how far apart the two predictions land.

The candidates are tool compounds with known targets, expressed as a shift on
the same parameters the twin already carries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as F
from . import nde
from . import params as P
from . import shift as SH
from . import simulator as S


@dataclass
class Candidate:
    name: str
    note: str
    shift: dict[str, float]   # parameter key -> fold change (linear delta for
                              # a linear parameter)


# Tool compounds a laboratory already has, with the published direction of
# their action. Magnitudes are the effect of a saturating concentration.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("gabazine", "GABA-A antagonist, removes inhibition",
              {"g_gaba": 0.05}),
    Candidate("CNQX", "AMPA antagonist", {"g_ampa": 0.05}),
    Candidate("D-AP5", "NMDA antagonist", {"g_nmda": 0.05}),
    Candidate("TTX", "sodium channel blocker", {"g_na": 0.12}),
    Candidate("4-aminopyridine", "potassium channel blocker", {"g_kdr": 0.2}),
    Candidate("apamin", "small-conductance calcium-activated K blocker",
              {"g_ahp": 0.15}),
    Candidate("baclofen-like drive reduction", "lowers tonic excitability",
              {"i_drive": -8.0}),
    Candidate("no follow-up", "record the same well again", {}),
)


def _shift_vector(c: Candidate) -> np.ndarray:
    d = np.zeros(P.N_PARAM)
    for key, value in c.shift.items():
        j = P.index(key)
        d[j] = np.log(value) if P.PARAMS[j].log else value
    return d


def split_hypotheses(post: dict, a: str, b: str,
                     quantile: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Posterior draws that favour mechanism `a`, and those that favour `b`.

    A draw favours whichever of the two it moved further, measured in units of
    the narrow prior component.
    """
    ja = [P.KEYS[i] for i in P.SHIFT_IDX].index(a)
    jb = [P.KEYS[i] for i in P.SHIFT_IDX].index(b)
    da = np.abs(post["delta"][:, ja]) / SH.INACTIVE_SCALE
    db = np.abs(post["delta"][:, jb]) / SH.INACTIVE_SCALE
    return np.flatnonzero(da > db), np.flatnonzero(db >= da)


def _treated_theta(post: dict, idx: np.ndarray, cand: np.ndarray,
                   n: int, rng: np.random.Generator) -> np.ndarray:
    """Parameters after the recorded compound and then the candidate."""
    pick = rng.choice(idx, size=n, replace=len(idx) < n)
    theta = post["theta_c"][pick].copy()
    delta = np.zeros((n, P.N_PARAM))
    delta[:, P.SHIFT_IDX] = post["delta"][pick]
    return np.clip(theta + delta + cand, P.TLO, P.THI)


def separation(sim, post: dict, a: str, b: str, duration: float,
               transient: float, n_draws: int = 24, seed: int = 0,
               candidates: tuple[Candidate, ...] = CANDIDATES,
               view: str = "grid16") -> list[dict]:
    """Rank follow-up experiments by how far apart the two hypotheses predict.

    The score is a symmetric standardised distance between the predicted
    feature distributions, so it is comparable across candidates.
    """
    idx_a, idx_b = split_hypotheses(post, a, b)
    if len(idx_a) < 10 or len(idx_b) < 10:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for k, cand in enumerate(candidates):
        vec = _shift_vector(cand)
        th_a = _treated_theta(post, idx_a, vec, n_draws, rng)
        th_b = _treated_theta(post, idx_b, vec, n_draws, rng)
        stacked = P.from_unit(np.concatenate([th_a, th_b]))
        res = sim.run(stacked, duration_s=duration, transient_s=transient,
                      seed=seed * 131 + k)
        feats = nde.phi(np.stack([F.compute(*S.view_events(res.raw_events(i), view), duration)
                                  for i in range(stacked.shape[0])]))
        fa, fb = feats[:n_draws], feats[n_draws:]
        pooled = np.sqrt(0.5 * (fa.var(0) + fb.var(0))) + 1e-6
        d = np.abs(fa.mean(0) - fb.mean(0)) / pooled
        out.append({"name": cand.name, "note": cand.note,
                    "separation": float(np.sqrt(np.mean(d ** 2))),
                    "best_feature": F.NAMES[int(np.argmax(d))],
                    "best_feature_d": float(d.max())})
    base = next((o["separation"] for o in out if o["name"] == "no follow-up"), 0.0)
    for o in out:
        o["gain_over_repeat"] = o["separation"] - base
    out.sort(key=lambda o: -o["separation"])
    return out


def recommend(sim, post: dict, rows: list[dict], duration: float,
              transient: float, margin: float = 0.15, **kw) -> dict:
    """Recommend a follow-up when the top two mechanisms are not separated."""
    if len(rows) < 2 or rows[0]["p_active"] - rows[1]["p_active"] >= margin:
        return {"needed": False,
                "reason": "the top mechanism is already separated"}
    a, b = rows[0]["key"], rows[1]["key"]
    ranked = separation(sim, post, a, b, duration, transient, **kw)
    if not ranked:
        return {"needed": True, "reason": "posterior does not split cleanly",
                "candidates": []}
    return {"needed": True, "tied": [a, b], "candidates": ranked,
            "recommended": ranked[0]["name"],
            "sentence": (f"{rows[0]['label']} and {rows[1]['label']} are not "
                         f"separated by this recording. Adding "
                         f"{ranked[0]['name']} separates them best, mostly "
                         f"through {ranked[0]['best_feature']}.")}
