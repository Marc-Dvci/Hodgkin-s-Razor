"""Parameter table for the network twin.

One row per model parameter. The simulator, the inference code, the reports and
the web app all read this table, so the column order and the meaning of each
column are defined once.

`shiftable` marks the parameters a compound is allowed to move in the paired
model. Network wiring (connection probability, inhibitory fraction) and the
recording noise are culture properties, so an acute wash-on cannot change them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    """One model parameter.

    `lo`/`hi` bound what the parameter can ever be, including after a
    saturating antagonist. `blo`/`bhi` bound what an untreated culture is drawn
    from. The two differ for the receptor conductances: a healthy culture never
    sits at the blocked extreme, but a compound must be able to take it there,
    and sampling baselines across the whole range leaves almost every draw dead.
    """
    key: str
    label: str
    unit: str
    lo: float
    hi: float
    log: bool
    shiftable: bool
    target: str
    blo: float | None = None
    bhi: float | None = None

    @property
    def base_lo(self) -> float:
        return self.lo if self.blo is None else self.blo

    @property
    def base_hi(self) -> float:
        return self.hi if self.bhi is None else self.bhi


PARAMS: tuple[Param, ...] = (
    Param("noise", "Membrane noise", "mV", 1.5, 7.0, False, False, ""),
    # Log-scaled and reaching low enough to represent a channel block: the
    # network only silences below about 0.2, and a linear range from 0.5 put
    # every draw in the flat region above it.
    Param("g_na", "Na conductance", "x50 mS/cm2", 0.08, 2.0, True, True,
          "voltage-gated Na channels", blo=0.45, bhi=2.0),
    Param("g_kdr", "Kdr conductance", "x5 mS/cm2", 0.3, 4.0, True, True,
          "delayed-rectifier K channels"),
    Param("g_ahp", "Slow AHP conductance", "nS", 0.5, 10.0, True, True,
          "Ca-activated K channels"),
    # Every receptor range reaches the extreme a saturating antagonist
    # produces. A floor above that leaves a block unrepresentable, which is
    # what the pharmacology check caught for sodium first and here second.
    Param("g_ampa", "AMPA conductance", "nS", 0.004, 1.2, True, True,
          "AMPA receptors", blo=0.06, bhi=1.2),
    # At this model's resting potential the magnesium block leaves about a
    # quarter of the NMDA conductance open, and its decay is fifty times slower
    # than AMPA, so equal conductances give NMDA twelve times the charge. The
    # range is scaled so that AMPA carries fast transmission, as it does in
    # cortical culture, and an AMPA block therefore silences the network.
    Param("g_nmda", "NMDA conductance", "nS", 0.0004, 0.12, True, True,
          "NMDA receptors", blo=0.006, bhi=0.12),
    Param("g_gaba", "Synaptic GABA-A conductance", "nS", 0.004, 8.0, True, True,
          "synaptic GABA-A receptors", blo=0.06, bhi=8.0),
    # A bath-applied agonist opens receptors on every cell, not only where an
    # inhibitory neuron happens to synapse. Without this column, GABA and
    # muscimol have no way to act, which the pharmacology check caught.
    Param("g_tonic_inh", "Tonic GABA-A conductance", "nS", 0.0, 6.0, False, True,
          "extrasynaptic GABA-A receptors", blo=0.0, bhi=0.35),
    Param("p_conn", "Connection probability", "", 0.10, 0.60, False, False, ""),
    Param("f_inh", "Inhibitory fraction", "", 0.05, 0.40, False, False, ""),
    Param("tau_d", "Vesicle recovery time", "ms", 150.0, 1200.0, True, True,
          "vesicle recycling"),
    Param("u_rel", "Release fraction per spike", "", 0.02, 0.60, True, True,
          "release probability"),
    Param("i_drive", "Tonic drive", "pA", 0.0, 22.0, False, True,
          "resting excitability"),
    # Recording nuisance: the fraction of candidate events an electrode reports.
    # It is a property of the amplifier and the detection threshold, so a
    # wash-on cannot change it, and the paired model holds it fixed.
    Param("p_detect", "Event detection fraction", "", 0.02, 1.0, True, False, ""),
    # Electrodes in one well differ in how many units they pick up. This is the
    # spread of that pickup, and it is a property of the plating, so it is held
    # fixed across a pair as well.
    Param("elec_het", "Electrode pickup spread", "", 0.01, 1.6, False, False, ""),
)

N_PARAM = len(PARAMS)
KEYS: tuple[str, ...] = tuple(p.key for p in PARAMS)
LABELS: tuple[str, ...] = tuple(p.label for p in PARAMS)
LO = np.array([p.lo for p in PARAMS], dtype=np.float64)
HI = np.array([p.hi for p in PARAMS], dtype=np.float64)
BLO = np.array([p.base_lo for p in PARAMS], dtype=np.float64)
BHI = np.array([p.base_hi for p in PARAMS], dtype=np.float64)
IS_LOG = np.array([p.log for p in PARAMS], dtype=bool)
SHIFTABLE = np.array([p.shiftable for p in PARAMS], dtype=bool)
SHIFT_IDX = np.flatnonzero(SHIFTABLE)
N_SHIFT = int(SHIFT_IDX.size)

# Bound in the transformed space, where the prior is uniform on [lo, hi].
TLO = np.where(IS_LOG, np.log(np.maximum(LO, 1e-12)), LO)
THI = np.where(IS_LOG, np.log(np.maximum(HI, 1e-12)), HI)
TBLO = np.where(IS_LOG, np.log(np.maximum(BLO, 1e-12)), BLO)
TBHI = np.where(IS_LOG, np.log(np.maximum(BHI, 1e-12)), BHI)


# Mechanisms grouped by what a pharmacologist would do next. A wrong parameter
# inside the right class is a different kind of error from a wrong class, and
# the two are reported separately.
CLASSES: dict[str, tuple[str, ...]] = {
    "excitatory transmission": ("g_ampa", "g_nmda"),
    "inhibitory transmission": ("g_gaba", "g_tonic_inh"),
    "intrinsic excitability": ("g_na", "g_kdr", "i_drive"),
    "adaptation and short-term plasticity": ("g_ahp", "tau_d", "u_rel"),
}
CLASS_NAMES: tuple[str, ...] = tuple(CLASSES)
CLASS_OF: dict[str, str] = {k: name for name, keys in CLASSES.items() for k in keys}


def class_index(key: str) -> int:
    """Index of the class a mechanism belongs to."""
    return CLASS_NAMES.index(CLASS_OF[key])


def index(key: str) -> int:
    """Column index of a parameter."""
    return KEYS.index(key)


def to_unit(theta: np.ndarray) -> np.ndarray:
    """Natural units -> transformed space (log where the prior is log-uniform)."""
    theta = np.asarray(theta, dtype=np.float64)
    out = theta.copy()
    out[..., IS_LOG] = np.log(np.clip(theta[..., IS_LOG], 1e-12, None))
    return out


def from_unit(t: np.ndarray) -> np.ndarray:
    """Transformed space -> natural units."""
    t = np.asarray(t, dtype=np.float64)
    out = t.copy()
    out[..., IS_LOG] = np.exp(t[..., IS_LOG])
    return out


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` untreated parameter sets, in natural units."""
    u = rng.uniform(size=(n, N_PARAM))
    return from_unit(TBLO + u * (TBHI - TBLO))


def sample_support(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw across the full support, including states only a drug reaches."""
    u = rng.uniform(size=(n, N_PARAM))
    return from_unit(TLO + u * (THI - TLO))


def describe_shift(delta_t: np.ndarray) -> list[dict]:
    """Turn a shift vector in transformed space into per-parameter effects.

    For a log parameter the shift is a multiplicative factor; for a linear one
    it is an additive change in the parameter's own unit.
    """
    out = []
    for j, p in enumerate(PARAMS):
        d = float(delta_t[j])
        if p.log:
            out.append({"key": p.key, "label": p.label, "kind": "fold",
                        "value": float(np.exp(d)),
                        "percent": float((np.exp(d) - 1.0) * 100.0),
                        "target": p.target})
        else:
            span = p.hi - p.lo
            out.append({"key": p.key, "label": p.label, "kind": "absolute",
                        "value": d, "percent": float(100.0 * d / span),
                        "target": p.target})
    return out
