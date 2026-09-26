"""Two-compartment chip twin with directional connectivity.

This is the geometry CellShells works in: two neuron populations in separate
chambers joined by asymmetric microchannels that let axons grow one way.
Peyrin et al. (Lab Chip 2011) report about 97 percent directional selectivity
for that design, and Lassus et al. (Sci Rep 2018) read such chips with calcium
imaging at 2 Hz rather than with electrodes.

The network kernel is unchanged. A chip is a wiring matrix, a delay table and
an electrode map, so the same simulator runs it. What the chip adds is three
parameters the single-chamber model has no place for, and a calcium
observation model, so the same twin can be asked what a 2 Hz imaging readout
can and cannot resolve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import params as P
from . import simulator as S

NN = S.NN
HALF = NN // 2
GRID_X, GRID_Y = 16, 8      # neurons per compartment
SPACING = 45.0              # um
CHANNEL_LEN = 500.0         # um, the microchannel array
AXON_SPEED = 0.15           # m/s, so 500 um takes about 3.3 ms

# The target chamber of a cortico-striatal chip is far less self-active than
# the source. `tgt_autonomy` scales both its tonic drive and its own recurrent
# weights, so at the low end the target only fires when the source drives it,
# which is the regime the device is built to create.
CHIP_KEYS = ("p_cross", "direction_sel", "g_cross", "tgt_autonomy")
CHIP_LO = np.array([0.02, 0.50, 0.05, 0.05])
CHIP_HI = np.array([0.50, 1.00, 1.50, 1.00])
CHIP_LOG = np.array([True, False, True, True])
N_CHIP = len(CHIP_KEYS)


@dataclass
class ChipGeometry:
    delay: np.ndarray       # (NN, NN) uint8 delay slots
    elec: np.ndarray        # (NN,) electrode index
    compartment: np.ndarray  # (NN,) 0 source, 1 target
    elec_compartment: np.ndarray  # (NELEC,) compartment of each electrode


def geometry() -> ChipGeometry:
    """Two chambers, eight electrodes each, joined by a microchannel array."""
    ix, iy = np.meshgrid(np.arange(GRID_X), np.arange(GRID_Y), indexing="ij")
    x0 = ix.ravel() * SPACING
    y0 = iy.ravel() * SPACING
    width = (GRID_X - 1) * SPACING
    comp = np.concatenate([np.zeros(HALF, dtype=int), np.ones(HALF, dtype=int)])
    x = np.concatenate([x0, x0 + width + CHANNEL_LEN])
    y = np.concatenate([y0, y0])

    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    delay_ms = d / (AXON_SPEED * 1000.0)     # um / (um per ms)
    slot = np.clip(np.round(delay_ms / (S.RB_STEP * S.DEFAULT_DT)),
                   0, S.RB_BINS - 1).astype(np.uint8)

    # Four electrodes per chamber row, eight per chamber.
    elec = np.full(NN, -1, dtype=np.int32)
    ecomp = np.zeros(S.NELEC, dtype=int)
    for c in (0, 1):
        sel = np.flatnonzero(comp == c)
        ex = np.linspace(x[sel].min() + SPACING, x[sel].max() - SPACING, 4)
        ey = np.linspace(y[sel].min() + SPACING * 0.5, y[sel].max() - SPACING * 0.5, 2)
        gx, gy = np.meshgrid(ex, ey, indexing="ij")
        gx, gy = gx.ravel(), gy.ravel()
        dist = np.hypot(x[sel][:, None] - gx[None, :], y[sel][:, None] - gy[None, :])
        near = dist.argmin(axis=1)
        within = dist.min(axis=1) <= 1.6 * SPACING
        elec[sel] = np.where(within, near + c * 8, -1)
        ecomp[c * 8:(c + 1) * 8] = c
    return ChipGeometry(delay=slot, elec=elec, compartment=comp,
                        elec_compartment=ecomp)


GEOMETRY = geometry()


def wiring(theta: np.ndarray, chip: np.ndarray, rng: np.random.Generator,
           pair: bool = False) -> dict:
    """Wiring for a batch of chips.

    Within a chamber the network is wired as usual. Across the channel only a
    fraction of connections exist, and `direction_sel` sets how many of them run
    the intended way: 1.0 is a perfect diode, 0.5 is a channel with no
    selectivity at all.
    """
    theta = np.atleast_2d(theta)
    chip = np.atleast_2d(chip)
    B = theta.shape[0]
    n_struct = B // 2 if pair else B
    take = slice(None, None, 2) if pair else slice(None)
    p_conn = theta[take, P.index("p_conn")]
    f_inh = theta[take, P.index("f_inh")]
    p_cross, dsel, g_cross = chip[take, 0], chip[take, 1], chip[take, 2]
    autonomy = chip[take, 3]
    i_drive_full = theta[:, P.index("i_drive")]

    comp = GEOMETRY.compartment
    same = comp[:, None] == comp[None, :]
    forward = (comp[:, None] == 0) & (comp[None, :] == 1)   # source j -> target i
    backward = (comp[:, None] == 1) & (comp[None, :] == 0)

    w = np.zeros((n_struct, NN, NN), dtype=np.float32)
    isinh = np.zeros((n_struct, NN), dtype=np.uint8)
    for b in range(n_struct):
        weight = np.clip(1.0 + 0.7 * rng.standard_normal((NN, NN)), 0.0, 2.0)
        m = np.zeros((NN, NN), dtype=bool)
        u = rng.random((NN, NN))
        m |= same & (u < p_conn[b])
        m |= forward & (u < p_cross[b] * dsel[b])
        m |= backward & (u < p_cross[b] * (1.0 - dsel[b]))
        tgt_tgt = (comp[:, None] == 1) & (comp[None, :] == 1)
        scale = np.where(same, 1.0, g_cross[b])
        scale = np.where(tgt_tgt, autonomy[b], scale)
        w[b] = (weight * m * scale * (S.REF_N / NN)).astype(np.float32)
        np.fill_diagonal(w[b], 0.0)
        isinh[b] = (rng.random(NN) < f_inh[b]).astype(np.uint8)
    # Heterogeneous drive, scaled down in the target chamber.
    shape = rng.random((n_struct, NN)) - 0.5
    if pair:
        w = np.repeat(w, 2, axis=0)
        isinh = np.repeat(isinh, 2, axis=0)
        shape = np.repeat(shape, 2, axis=0)
        ratio = np.repeat(autonomy, 2)
    else:
        ratio = autonomy
    scale = np.where(comp[None, :] == 1, ratio[:, None], 1.0)
    ibias = (shape * i_drive_full[:, None] * scale).astype(np.float32)
    return {"w": w, "isinh": isinh, "delay": GEOMETRY.delay,
            "elec": GEOMETRY.elec, "ibias": ibias}


def sample_chip_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    lo = np.where(CHIP_LOG, np.log(CHIP_LO), CHIP_LO)
    hi = np.where(CHIP_LOG, np.log(CHIP_HI), CHIP_HI)
    u = lo + rng.uniform(size=(n, N_CHIP)) * (hi - lo)
    return np.where(CHIP_LOG, np.exp(u), u)


def run_chip(sim: S.Simulator, theta: np.ndarray, chip: np.ndarray,
             duration_s: float = 60.0, transient_s: float = 5.0,
             seed: int = 0, pair: bool = False) -> S.SimResult:
    rng = np.random.default_rng(seed * 104729 + 7)
    struct = wiring(theta, chip, rng, pair=pair)
    return sim.run(theta, duration_s=duration_s, transient_s=transient_s,
                   seed=seed, pair=pair, structure=struct)


# --------------------------------------------------------------- calcium view
CA_RISE_S = 0.08
CA_DECAY_S = 0.55       # Fluo-4 bulk-loaded population signal
CA_FRAME_HZ = 2.0       # Lassus et al. imaged every 500 ms


def calcium_trace(events: np.ndarray, duration: float, compartment: int,
                  geom: ChipGeometry = GEOMETRY, frame_hz: float = CA_FRAME_HZ,
                  noise_sd: float = 0.02,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Population calcium signal for one chamber, sampled at the frame rate.

    Spikes from the chamber's electrodes are convolved with a double-exponential
    indicator kernel and then sampled at the imaging rate, which is what
    discards the fast structure an electrode array keeps.
    """
    n_frames = int(duration * frame_hz)
    if n_frames <= 0:
        return np.zeros(0)
    fine_hz = 100.0
    n_fine = int(duration * fine_hz)
    train = np.zeros(n_fine)
    if events.size:
        sel = geom.elec_compartment[events[:, 0].astype(int)] == compartment
        t = events[sel, 1]
        idx = np.clip((t * fine_hz).astype(int), 0, n_fine - 1)
        np.add.at(train, idx, 1.0)
    k_len = int(6 * CA_DECAY_S * fine_hz)
    tk = np.arange(k_len) / fine_hz
    kernel = np.exp(-tk / CA_DECAY_S) - np.exp(-tk / CA_RISE_S)
    kernel = np.clip(kernel, 0, None)
    s = kernel.sum()
    if s > 0:
        kernel /= s
    trace = np.convolve(train, kernel)[:n_fine]
    step = max(int(fine_hz / frame_hz), 1)
    frames = trace[: n_frames * step].reshape(n_frames, step).mean(axis=1)
    if noise_sd > 0:
        rng = rng or np.random.default_rng(0)
        frames = frames + rng.normal(0, noise_sd * max(frames.std(), 1e-6), n_frames)
    return frames


CA_NAMES = ("ca_rate_src", "ca_rate_tgt", "ca_amp_src", "ca_amp_tgt",
            "ca_cv_src", "ca_cv_tgt", "ca_corr", "ca_lag_s",
            "ca_lead_frac", "ca_sync_src", "ca_sync_tgt", "ca_ratio")
N_CA = len(CA_NAMES)


def calcium_features(events: np.ndarray, duration: float,
                     geom: ChipGeometry = GEOMETRY,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Summary of what a two-chamber calcium movie shows.

    Includes the cross-chamber lag, which is the quantity a directional chip is
    built to produce and the one an imaging readout is expected to resolve.
    """
    a = calcium_trace(events, duration, 0, geom, rng=rng)
    b = calcium_trace(events, duration, 1, geom, rng=rng)
    out = np.zeros(N_CA)
    if a.size < 4 or b.size < 4:
        return out

    def peaks(x):
        th = x.mean() + 2.0 * x.std()
        return np.flatnonzero((x > th) & (np.r_[True, np.diff(x) > 0][:x.size]))

    pa, pb = peaks(a), peaks(b)
    out[0] = len(pa) * 60.0 / duration
    out[1] = len(pb) * 60.0 / duration
    out[2] = float(a.max() - a.mean()) if a.size else 0.0
    out[3] = float(b.max() - b.mean()) if b.size else 0.0
    out[4] = float(a.std() / max(abs(a.mean()), 1e-9))
    out[5] = float(b.std() / max(abs(b.mean()), 1e-9))
    if a.std() > 0 and b.std() > 0:
        out[6] = float(np.corrcoef(a, b)[0, 1])
        n = a.size
        lags = np.arange(-6, 7)
        cc = []
        for l in lags:
            if l < 0:
                cc.append(np.corrcoef(a[-l:], b[:n + l])[0, 1])
            elif l > 0:
                cc.append(np.corrcoef(a[:n - l], b[l:])[0, 1])
            else:
                cc.append(out[6])
        cc = np.nan_to_num(np.array(cc))
        out[7] = float(lags[int(np.argmax(cc))] / CA_FRAME_HZ)
    if len(pa) and len(pb):
        # Fraction of target events preceded by a source event within 2 frames.
        lead = sum(np.any((pb[k] - pa >= 0) & (pb[k] - pa <= 2)) for k in range(len(pb)))
        out[8] = lead / len(pb)
    out[9] = float((a > a.mean() + 2 * a.std()).mean())
    out[10] = float((b > b.mean() + 2 * b.std()).mean())
    out[11] = float(np.log1p(max(b.mean(), 0)) - np.log1p(max(a.mean(), 0)))
    return np.nan_to_num(out)
