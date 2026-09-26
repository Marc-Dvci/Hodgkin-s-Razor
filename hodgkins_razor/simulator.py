"""GPU simulator for cultured neuronal networks recorded on a microelectrode array.

The model is a conductance-based Hodgkin-Huxley network with excitatory and
inhibitory populations, short-term depression, slow after-hyperpolarisation and
distance-dependent conduction delays. It extends the excitatory-only model of
Doorn et al. (Commun Biol 2025) with inhibition and a tonic drive term, which
is what makes GABAergic compounds readable.

One CUDA block runs one network, so a batch of parameter sets is simulated in
one launch. `Simulator.run` returns detected electrode events, which is what a
real MEA records.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import numpy as np

from . import params as P

NN = 256          # neurons, 16 x 16 grid
NELEC = 16        # electrodes, 4 x 4 grid
RB_BINS = 16
RB_STEP = 8
PER_ELEC = 16384  # event capacity per electrode, must match kernel.cu
GRID = 16
SPACING = 45.0    # um between neurons
MAX_DELAY = 12.8  # ms across the longest connection
REF_N = 100       # reference network size the conductance scale is defined at
DEFAULT_DT = 0.1  # ms

_KERNEL_SRC = pathlib.Path(__file__).with_name("kernel.cu")


@dataclass
class SimResult:
    """Electrode events for one batch.

    times: list of int arrays, one per network, in simulation steps.
    elecs: list of uint8 arrays, one per network.
    truncated: networks whose event buffer overflowed.
    """
    times: list[np.ndarray]
    elecs: list[np.ndarray]
    counts: np.ndarray
    truncated: np.ndarray
    duration_s: float
    dt_ms: float

    def raw_events(self, b: int) -> np.ndarray:
        """(n, 2) [electrode, time_in_seconds] at the kernel's resolution."""
        t = self.times[b] * (self.dt_ms / 1000.0)
        return np.stack([self.elecs[b].astype(np.float64), t], axis=1)

    def as_events(self, b: int, view: str = "grid16") -> np.ndarray:
        """Events for network b as one recording system would report them."""
        return view_events(self.raw_events(b), view)[0]


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    """Delay slot table and neuron-to-electrode assignment.

    Both are fixed by the grid, so they are shared across every network.
    """
    ix, iy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    x = (ix.ravel() * SPACING).astype(np.float64)
    y = (iy.ravel() * SPACING).astype(np.float64)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    delay_ms = MAX_DELAY * d / d.max()
    slot = np.clip(np.round(delay_ms / (RB_STEP * DEFAULT_DT)), 0, RB_BINS - 1)
    slot = slot.astype(np.uint8)

    span = (GRID - 1) * SPACING
    step = span / 4.0
    ex, ey = np.meshgrid(np.arange(4) * step + step / 2.0,
                         np.arange(4) * step + step / 2.0, indexing="ij")
    ex, ey = ex.ravel(), ey.ravel()
    dist = np.hypot(x[:, None] - ex[None, :], y[:, None] - ey[None, :])
    nearest = dist.argmin(axis=1)
    radius = 1.45 * SPACING
    elec = np.where(dist.min(axis=1) <= radius, nearest, -1).astype(np.int32)
    return slot, elec


DELAY_SLOT, NEURON_ELEC = _geometry()


class Simulator:
    """Batched network simulator. Requires a CUDA device."""

    def __init__(self, dt_ms: float = DEFAULT_DT,
                 kinetics: dict | None = None):
        """kinetics overrides receptor time constants the twin holds fixed.

        The twin has no parameter for them, so a simulator built with different
        values produces recordings it cannot represent. That is what the guard
        is tested against.
        """
        import cupy as cp
        self.cp = cp
        self.dt = float(dt_ms)
        self.max_events = NELEC * PER_ELEC
        self.kinetics = dict(kinetics or {})
        src = _KERNEL_SRC.read_text()
        for name, value in self.kinetics.items():
            pattern = r"#define " + name + r" [^/\n]+"
            src = re.sub(pattern, f"#define {name} {float(value)}f ", src)
        self.module = cp.RawModule(code=src, backend="nvrtc",
                                   options=("--use_fast_math", "-std=c++14"))
        self.kern = self.module.get_function("simulate")
        self.shared = ((10 * NN + 2 * RB_BINS * NN) * 4
                       + (NN // 32) * 4 + NELEC * 4 + 4 + NELEC * 4 + NN)
        self.kern.max_dynamic_shared_size_bytes = self.shared
        self._db = cp.asarray(DELAY_SLOT.ravel())
        self._elec = cp.asarray(NEURON_ELEC)

    def run(self, theta: np.ndarray, duration_s: float = 160.0,
            transient_s: float = 5.0, seed: int = 0,
            pair: bool = False, structure: dict | None = None) -> SimResult:
        """Simulate one batch of parameter sets.

        theta: (B, n_param) in natural units, column order given by
        params.PARAMS. With `pair=True` consecutive rows are a matched
        baseline/treated pair: they share the wiring, the inhibitory
        assignment, the tonic drive and the electrode pickup, and differ only
        in the parameters and in the recording noise, which is what recording
        the same well twice gives.

        `structure` overrides the wiring, the delay table and the electrode map,
        which is how a compartmented chip is simulated on the same kernel.
        """
        cp = self.cp
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if theta.shape[1] != P.N_PARAM:
            raise ValueError(f"theta must have {P.N_PARAM} columns")
        B = theta.shape[0]
        n_steps = int(round((duration_s + transient_s) * 1000.0 / self.dt))
        n_trans = int(round(transient_s * 1000.0 / self.dt))

        rng = cp.random.RandomState(seed)
        th = cp.asarray(theta, dtype=cp.float32)
        p_conn = th[:, P.index("p_conn")][:, None, None]
        f_inh = th[:, P.index("f_inh")][:, None]
        i_drive = th[:, P.index("i_drive")][:, None]

        if pair and B % 2:
            raise ValueError("pair mode needs an even number of rows")
        n_struct = B // 2 if pair else B
        rep = (lambda a: cp.repeat(a, 2, axis=0)) if pair else (lambda a: a)
        mask = rng.random_sample((n_struct, NN, NN), dtype=cp.float32) < p_conn[::2 if pair else 1]
        w = cp.clip(1.0 + 0.7 * rng.standard_normal((n_struct, NN, NN), dtype=cp.float32), 0.0, 2.0)
        # Synaptic conductances are per connection, so total drive would grow with
        # network size. Scaling to a 100-neuron reference keeps the conductance
        # parameters comparable to Doorn et al. and keeps the network off
        # depolarisation block.
        w *= mask * np.float32(REF_N / NN)
        idx = cp.arange(NN)
        w[:, idx, idx] = 0.0
        w = rep(w)
        isinh = rep((rng.random_sample((n_struct, NN), dtype=cp.float32)
                     < f_inh[::2 if pair else 1]).astype(cp.uint8))
        ibias = rep(rng.random_sample((n_struct, NN), dtype=cp.float32) - 0.5) * i_drive

        db, elec = self._db, self._elec
        if structure is not None:
            if "w" in structure:
                w = cp.asarray(structure["w"], dtype=cp.float32)
            if "isinh" in structure:
                isinh = cp.asarray(structure["isinh"], dtype=cp.uint8)
            if "delay" in structure:
                db = cp.asarray(structure["delay"].ravel(), dtype=cp.uint8)
            if "elec" in structure:
                elec = cp.asarray(structure["elec"], dtype=cp.int32)
            if "ibias" in structure:
                ibias = cp.asarray(structure["ibias"], dtype=cp.float32)

        out_t = cp.zeros((B, NELEC, PER_ELEC), dtype=cp.int32)
        out_e = cp.zeros((B, NELEC, PER_ELEC), dtype=cp.uint8)
        out_c = cp.zeros((B, NELEC), dtype=cp.int32)

        # A per-network electrode map lets a readout follow wiring that
        # differs between networks, such as the neurons whose axons run
        # through a chip's microchannels.
        stride = np.int32(NN if elec.ndim == 2 else 0)
        elec = elec.ravel() if elec.ndim == 2 else elec
        self.kern((B,), (NN,),
                  (th, w, db, isinh, ibias, elec, stride,
                   np.int32(n_steps), np.int32(n_trans), np.float32(self.dt),
                   out_t, out_e, out_c, np.int32(PER_ELEC),
                   np.uint32((seed * 2654435761 + 12345) & 0xFFFFFFFF)),
                  shared_mem=self.shared)
        cp.cuda.Stream.null.synchronize()

        counts = cp.asnumpy(out_c)
        keep = np.minimum(counts, PER_ELEC)
        ts = cp.asnumpy(out_t)
        es = cp.asnumpy(out_e)
        # Thin to the detection fraction: an electrode reports only the events
        # that clear its threshold, which is what sets the absolute event rate
        # of a recording system.
        p_det = theta[:, P.index("p_detect")]
        het = theta[:, P.index("elec_het")]
        times, elecs = [], []
        for b in range(B):
            tb = np.concatenate([ts[b, e, :keep[b, e]] for e in range(NELEC)])
            eb = np.concatenate([es[b, e, :keep[b, e]] for e in range(NELEC)])
            order = np.lexsort((eb, tb))
            tb, eb = tb[order], eb[order]
            # Per-electrode pickup is fixed by the well, so it is drawn from the
            # structure seed and is identical for both halves of a pair.
            struct = np.random.default_rng(1000003 * (seed + 1) + (b // 2 if pair else b))
            gain = np.clip(p_det[b] * np.exp(struct.normal(0.0, het[b], size=NELEC)), 0.0, 1.0)
            if tb.size:
                draw = np.random.default_rng(7919 * (seed + 1) + b).random(tb.size)
                sel = draw < gain[eb]
                tb, eb = tb[sel], eb[sel]
            times.append(tb.copy())
            elecs.append(eb.copy())
        del w, mask, out_t, out_e, th
        cp.get_default_memory_pool().free_all_blocks()
        return SimResult(times=times, elecs=elecs, counts=counts.sum(axis=1),
                         truncated=(counts >= PER_ELEC).any(axis=1),
                         duration_s=duration_s, dt_ms=self.dt)


def available() -> bool:
    """True when a CUDA device is usable."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# Recording systems the twin reads. The simulated array is a 4 x 4 grid whose
# electrodes report multi-unit events at the kernel's 0.2 ms resolution. A real
# system adds its own detection dead time and may record fewer electrodes:
#
# grid16  Axion Maestro 48-well plate, 16 electrodes per well (Tampere dataset),
#         one event per electrode per 2 ms.
# grid12  Multi Channel Systems 24-well plate, a 4 x 4 grid whose four corner
#         positions are reference electrodes, so 12 record (Doorn et al.).
#         Their peak trains carry intervals down to 0.2 ms, so 0.3 ms.
VIEWS: dict[str, dict] = {
    "grid16": {"electrodes": tuple(range(NELEC)), "dead_ms": 2.0},
    "grid12": {"electrodes": tuple(e for e in range(NELEC) if e not in (0, 3, 12, 15)),
               "dead_ms": 0.3},
}


def n_electrodes(view: str) -> int:
    return len(VIEWS[view]["electrodes"])


try:
    from numba import njit
except ImportError:          # pragma: no cover - numba is in requirements
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)


@njit(cache=True)
def _dead_time(elec, t, n_elec, dead_s):
    keep = np.zeros(t.size, dtype=np.bool_)
    last = np.full(n_elec, -1e9)
    for k in range(t.size):
        e = int(elec[k])
        if t[k] - last[e] >= dead_s:
            keep[k] = True
            last[e] = t[k]
    return keep


def view_events(events: np.ndarray, view: str) -> tuple[np.ndarray, int]:
    """Read simulated events through one recording system.

    Keeps the system's electrodes, renumbers them from 0, and applies its
    detection dead time. Returns the events and the electrode count.
    """
    spec = VIEWS[view]
    keep = spec["electrodes"]
    remap = np.full(NELEC, -1)
    remap[list(keep)] = np.arange(len(keep))
    if events.size == 0:
        return np.zeros((0, 2)), len(keep)
    e = remap[events[:, 0].astype(np.int64)]
    sel = e >= 0
    e, t = e[sel], events[sel, 1]
    order = np.argsort(t, kind="stable")
    e, t = e[order], t[order]
    ok = _dead_time(e.astype(np.int64), t.astype(np.float64), len(keep),
                    spec["dead_ms"] / 1000.0)
    return np.stack([e[ok].astype(float), t[ok]], axis=1), len(keep)
