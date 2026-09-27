"""Two-compartment chip twin with directional connectivity.

This is the geometry CellShells works in: two neuron populations in separate
chambers joined by asymmetric microchannels that let axons grow one way.
Peyrin et al. (Lab Chip 2011) report about 97 percent directional selectivity
for that design, Lassus et al. (Sci Rep 2018) read such chips with calcium
imaging at 2 Hz, and Mateus et al. (2024) record them on a 256-electrode MEA
with five electrodes inside each microchannel.

The network kernel is unchanged. A chip is a wiring matrix, a delay table and
an electrode map, so the same simulator runs it. What the chip adds is four
parameters the single-chamber model has no place for, and four ways of reading
it, so the twin can be asked which readout resolves which property before an
experiment is run:

* compartment electrodes: seven per chamber, the population a standard MEA
  under each chamber sees;
* a population calcium movie at 2 Hz per chamber;
* channel electrodes: the axons that run through the microchannels, recorded
  separately by the direction they carry, which is what electrodes inside the
  channels report after a propagation-sequence analysis;
* compartment electrodes across three perfusions: untreated, the source
  chamber silenced, the target chamber silenced. Fluidic isolation between the
  chambers is what these devices are built for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as F
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
# p_cross is the fraction of a chamber's neurons with an axon in the channels.
CHIP_LOG = np.array([True, False, True, True])
N_CHIP = len(CHIP_KEYS)

N_COMP_ELEC = 7             # electrodes per chamber
ELEC_FWD = 2 * N_COMP_ELEC  # channel electrode, axons running source -> target
ELEC_BWD = ELEC_FWD + 1     # channel electrode, axons running target -> source
SILENCE_PA = -60.0          # bias that holds a chamber below threshold

# Readout names used by the study and the report.
READOUTS = ("compartment", "calcium_2hz", "channel", "perfusion")


@dataclass
class ChipGeometry:
    delay: np.ndarray            # (NN, NN) uint8 delay slots
    elec: np.ndarray             # (NN,) compartment electrode index, -1 if unseen
    compartment: np.ndarray      # (NN,) 0 source, 1 target
    elec_compartment: np.ndarray  # (NELEC,) 0 source, 1 target, 2 channel


def geometry() -> ChipGeometry:
    """Two chambers, seven electrodes each, joined by a microchannel array."""
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

    elec = np.full(NN, -1, dtype=np.int32)
    ecomp = np.full(S.NELEC, 2, dtype=int)
    for c in (0, 1):
        sel = np.flatnonzero(comp == c)
        ex = np.linspace(x[sel].min() + SPACING, x[sel].max() - SPACING, N_COMP_ELEC)
        ey = np.full(N_COMP_ELEC, y[sel].mean())
        ey[1::2] += SPACING * 1.5
        ey[0::2] -= SPACING * 1.5
        dist = np.hypot(x[sel][:, None] - ex[None, :], y[sel][:, None] - ey[None, :])
        near = dist.argmin(axis=1)
        within = dist.min(axis=1) <= 1.6 * SPACING
        elec[sel] = np.where(within, near + c * N_COMP_ELEC, -1)
        ecomp[c * N_COMP_ELEC:(c + 1) * N_COMP_ELEC] = c
    return ChipGeometry(delay=slot, elec=elec, compartment=comp,
                        elec_compartment=ecomp)


GEOMETRY = geometry()


RSCALE_KEYS = ("g_ampa", "g_nmda", "g_gaba", "g_kdr")   # order of the kernel's RSCALE


def wiring(theta: np.ndarray, chip: np.ndarray, rng: np.random.Generator,
           pair: bool = False, silence: np.ndarray | None = None,
           striatal: bool = False, perfuse: dict | None = None) -> dict:
    """Wiring, drive and electrode maps for a batch of chips.

    Within a chamber the network is wired as usual. `p_cross` is the fraction
    of a chamber's neurons whose axons grow through the microchannels, and
    `direction_sel` how many of those run the intended way: 1.0 is a perfect
    diode, 0.5 a channel with no selectivity. A projecting neuron connects to
    the other chamber with the culture's own connection probability, at a
    strength scaled by `g_cross`.

    Projecting neurons are drawn among the somata no compartment electrode
    picks up, and they are recorded by the channel electrode for their
    direction, so that readout follows each chip's own wiring. `silence`
    (one of -1, 0, 1 per row) holds the source (0) or the target (1) chamber
    below threshold, which is what perfusing that chamber with TTX does.

    `striatal` makes every target neuron GABAergic, as the medium spiny
    neurons of a cortico-striatal chip are: they excite nothing, and they
    fire only when the source drives them if `tgt_autonomy` is low.
    `perfuse` maps a chamber (0 source, 1 target) to conductance factors,
    for example {1: {"g_nmda": 0.3}}, applied to that chamber's neurons
    only: a drug perfused into one compartment of a fluidically isolated
    device.
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
    tgt_tgt = (comp[:, None] == 1) & (comp[None, :] == 1)
    unseen = [np.flatnonzero((comp == c) & (GEOMETRY.elec < 0)) for c in (0, 1)]

    w = np.zeros((n_struct, NN, NN), dtype=np.float32)
    isinh = np.zeros((n_struct, NN), dtype=np.uint8)
    elec = np.zeros((n_struct, NN), dtype=np.int32)
    for b in range(n_struct):
        weight = np.clip(1.0 + 0.7 * rng.standard_normal((NN, NN)), 0.0, 2.0)
        u = rng.random((NN, NN))
        m = same & (u < p_conn[b])
        n_fwd = min(int(round(p_cross[b] * dsel[b] * HALF)), unseen[0].size)
        n_bwd = min(int(round(p_cross[b] * (1.0 - dsel[b]) * HALF)), unseen[1].size)
        fwd = rng.choice(unseen[0], size=n_fwd, replace=False)
        bwd = rng.choice(unseen[1], size=n_bwd, replace=False)
        proj = np.zeros(NN, dtype=bool)
        proj[fwd] = True
        proj[bwd] = True
        # W[j][i] is the synapse from j to i.
        m |= proj[:, None] & ~same & (u < p_conn[b])
        scale = np.where(same, 1.0, g_cross[b])
        scale = np.where(tgt_tgt, autonomy[b], scale)
        w[b] = (weight * m * scale * (S.REF_N / NN)).astype(np.float32)
        np.fill_diagonal(w[b], 0.0)
        isinh[b] = (rng.random(NN) < f_inh[b]).astype(np.uint8)
        if striatal:
            isinh[b, comp == 1] = 1
        e = GEOMETRY.elec.copy()
        e[fwd] = ELEC_FWD
        e[bwd] = ELEC_BWD
        elec[b] = e
    shape = rng.random((n_struct, NN)) - 0.5
    if pair:
        w = np.repeat(w, 2, axis=0)
        isinh = np.repeat(isinh, 2, axis=0)
        elec = np.repeat(elec, 2, axis=0)
        shape = np.repeat(shape, 2, axis=0)
        ratio = np.repeat(autonomy, 2)
    else:
        ratio = autonomy
    scale = np.where(comp[None, :] == 1, ratio[:, None], 1.0)
    ibias = (shape * i_drive_full[:, None] * scale).astype(np.float32)
    if silence is not None:
        silence = np.asarray(silence)
        for c in (0, 1):
            rows = silence == c
            ibias[np.ix_(rows, comp == c)] = SILENCE_PA
    out = {"w": w, "isinh": isinh, "delay": GEOMETRY.delay,
           "elec": elec, "ibias": ibias}
    if perfuse:
        rscale = np.ones((B, NN, len(RSCALE_KEYS)), dtype=np.float32)
        for c, factors in perfuse.items():
            for key, f in factors.items():
                rscale[:, comp == c, RSCALE_KEYS.index(key)] = f
        out["rscale"] = rscale
    return out


def sample_chip_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    lo = np.where(CHIP_LOG, np.log(CHIP_LO), CHIP_LO)
    hi = np.where(CHIP_LOG, np.log(CHIP_HI), CHIP_HI)
    u = lo + rng.uniform(size=(n, N_CHIP)) * (hi - lo)
    return np.where(CHIP_LOG, np.exp(u), u)


def run_chip(sim: S.Simulator, theta: np.ndarray, chip: np.ndarray,
             duration_s: float = 60.0, transient_s: float = 5.0,
             seed: int = 0, pair: bool = False,
             silence: np.ndarray | None = None, striatal: bool = False,
             perfuse: dict | None = None) -> S.SimResult:
    """Simulate chips. The wiring depends only on `seed`, so the same seed
    with a different `silence` or `perfuse` is the same device under another
    perfusion."""
    rng = np.random.default_rng(seed * 104729 + 7)
    struct = wiring(theta, chip, rng, pair=pair, silence=silence,
                    striatal=striatal, perfuse=perfuse)
    return sim.run(theta, duration_s=duration_s, transient_s=transient_s,
                   seed=seed, pair=pair, structure=struct)


# ---------------------------------------------------------------- readouts
def chamber_events(events: np.ndarray, compartment: int) -> np.ndarray:
    """Events of one chamber's electrodes, renumbered from 0."""
    if events.size == 0:
        return np.zeros((0, 2))
    ec = GEOMETRY.elec_compartment[events[:, 0].astype(int)]
    ev = events[ec == compartment].copy()
    ev[:, 0] -= compartment * N_COMP_ELEC
    return ev


def cross_stats(events: np.ndarray, duration: float) -> np.ndarray:
    """Cross-chamber coupling a compartment array shows without channel access.

    Peak and lag of the cross-correlation of the two chambers' population
    rates at 5 ms resolution, its asymmetry (how much of the correlation mass
    sits at positive lags, source leading), and the rate ratio.
    """
    out = np.zeros(4)
    a = chamber_events(events, 0)[:, 1]
    b = chamber_events(events, 1)[:, 1]
    if a.size > 5 and b.size > 5:
        bins = np.arange(0, duration + 0.005, 0.005)
        ha, _ = np.histogram(a, bins)
        hb, _ = np.histogram(b, bins)
        ha = ha - ha.mean()
        hb = hb - hb.mean()
        n = 60   # +/- 300 ms
        cc = np.array([np.dot(ha[max(0, -l):len(ha) - max(0, l)],
                              hb[max(0, l):len(hb) - max(0, -l)])
                       for l in range(-n, n + 1)], dtype=float)
        cc /= np.sqrt(np.dot(ha, ha) * np.dot(hb, hb)) + 1e-9
        out[0] = float(cc.max())
        out[1] = float((np.argmax(cc) - n) * 0.005)
        pos, neg = np.clip(cc[n + 1:], 0, None).sum(), np.clip(cc[:n], 0, None).sum()
        out[2] = float((pos - neg) / (pos + neg + 1e-9))
    out[3] = float(np.log1p(b.size) - np.log1p(a.size))
    return out


def compartment_view(events: np.ndarray, duration: float) -> np.ndarray:
    """Seven electrodes per chamber, and the cross-chamber statistics."""
    return np.concatenate([F.compute(chamber_events(events, 0), N_COMP_ELEC, duration),
                           F.compute(chamber_events(events, 1), N_COMP_ELEC, duration),
                           cross_stats(events, duration)])


def channel_view(events: np.ndarray, duration: float) -> np.ndarray:
    """What electrodes inside the channels report.

    The forward fraction of propagating events, the two directions' event
    rates, and how many axons of each direction were active at all.
    """
    if events.size == 0:
        return np.zeros(4)
    e = events[:, 0].astype(int)
    fwd, bwd = float((e == ELEC_FWD).sum()), float((e == ELEC_BWD).sum())
    tot = fwd + bwd
    return np.array([fwd / tot if tot > 0 else 0.5,
                     np.log1p(fwd / duration), np.log1p(bwd / duration),
                     np.log1p(tot / duration)])


def perfusion_view(ev_base: np.ndarray, ev_src_off: np.ndarray,
                   ev_tgt_off: np.ndarray, duration: float) -> np.ndarray:
    """Each chamber's rate with the other chamber silenced, against untreated.

    Silencing the source removes whatever drive runs forward; silencing the
    target removes whatever runs back. The two drops are the directional
    coupling, read through compartment electrodes only.
    """
    def rate(ev, c):
        return chamber_events(ev, c).shape[0] / duration / N_COMP_ELEC
    tb, sb = rate(ev_base, 1), rate(ev_base, 0)
    t_src_off, s_tgt_off = rate(ev_src_off, 1), rate(ev_tgt_off, 0)
    return np.array([np.log1p(t_src_off) - np.log1p(tb),
                     np.log1p(s_tgt_off) - np.log1p(sb),
                     np.log1p(tb), np.log1p(sb)])


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
