"""Summary statistics of a multi-electrode recording.

`compute` is the only path from spike events to numbers, and it is used for
simulated and for recorded data alike. Everything downstream, including the
posterior and the predictive check, sees a recording only through this
function, so a difference between simulation and experiment can never come
from two different feature implementations.

The first fifteen entries follow Doorn et al. (Commun Biol 2025) so that their
published estimator can be scored on the same vector. The remaining entries are
network statistics that respond to inhibition, which their excitatory-only
model had no use for.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import norm

BIN_S = 0.025          # network firing-rate bin
CORR_BIN_S = 0.2       # binarisation bin for pairwise correlation
ISI_GRID_S = 0.002     # grid for the time-resolved ISI statistics

NAMES: tuple[str, ...] = (
    # Descriptors in common use, so a published estimator can be scored on the
    # same vector.
    "mfr", "nbr", "nbd", "psib", "fbs", "cvibi",
    "mean_cc", "sd_cc", "mean_isi_cc", "sd_isi_cc", "isi_dist",
    "mean_isi", "sd_isi_temporal", "sd_isi_electrode", "mac",
    "active_frac", "rate_cv_electrode", "burst_peak_ratio",
    "in_burst_cv", "isi_cv", "burst_rise_s", "silent_frac",
    # Descriptors that separate one conductance from another. Removing
    # excitatory drive and adding inhibition both lower the firing rate; they
    # do different things to the shape of a burst, to how much activity
    # survives between bursts, and to how tightly electrodes tile in time.
    "sttc_mean", "sttc_sd",
    "tonic_rate", "burst_tonic_ratio",
    "burst_participation", "burst_participation_sd",
    "burst_onset_jitter", "burst_decay_ratio", "burst_skew",
    "fano_25ms", "fano_250ms", "fano_1s",
    "acorr_50ms", "acorr_200ms", "acorr_1s",
    "cv2", "log_isi_p10", "log_isi_p90",
)
N_FEATURE = len(NAMES)


def _gauss_kernel(width: int = 11, sigma: float = 3.0) -> np.ndarray:
    x = np.arange(width, dtype=float) - width // 2
    k = norm.pdf(x, scale=sigma)
    return k / k.sum()


_KERNEL = _gauss_kernel()


def _binned(elec: np.ndarray, t: np.ndarray, n_elec: int,
            duration: float, bin_s: float) -> np.ndarray:
    n_bins = max(int(np.floor(duration / bin_s)), 1)
    idx = np.clip((t / bin_s).astype(np.int64), 0, n_bins - 1)
    flat = elec.astype(np.int64) * n_bins + idx
    counts = np.bincount(flat, minlength=n_elec * n_bins)
    return counts.reshape(n_elec, n_bins)


MIN_BURST_ELEC = 3
MIN_BURST_S = 0.05
MERGE_GAP_S = 0.10


def _detect_bursts(counts: np.ndarray, rate_total: np.ndarray,
                   n_active: int) -> tuple[np.ndarray, np.ndarray]:
    """Network bursts as runs where the population rate exceeds its own baseline.

    The threshold is three standard deviations above the mean population rate,
    floored at a tenth of the 99th percentile so that a recording with one
    outsized event cannot hide its smaller ones. A run counts as a network
    burst when it lasts at least 50 ms and at least three electrodes take part.
    Runs separated by less than 100 ms are merged, which keeps a fragmented
    burst from being counted several times.
    """
    if rate_total.size == 0 or n_active == 0 or rate_total.max() <= 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0)
    th = max(rate_total.mean() + 3.0 * rate_total.std(),
             0.10 * np.percentile(rate_total, 99), 1.0 / BIN_S)
    above = rate_total > th
    if not above.any():
        return np.zeros((0, 2), dtype=int), np.zeros(0)

    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(above.size)

    merged: list[list[int]] = []
    gap = max(int(round(MERGE_GAP_S / BIN_S)), 1)
    for a, b in zip(starts, stops):
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    min_bins = max(int(round(MIN_BURST_S / BIN_S)), 1)
    bursts, spikes_in = [], []
    for a, b in merged:
        if b - a < min_bins:
            continue
        block = counts[:, a:b]
        if (block.sum(axis=1) > 0).sum() < MIN_BURST_ELEC:
            continue
        bursts.append((a, b))
        spikes_in.append(float(block.sum()))
    return np.array(bursts, dtype=int).reshape(-1, 2), np.array(spikes_in, dtype=float)


def _pairwise(binary: np.ndarray) -> tuple[float, float]:
    """Mean and spread of the pairwise Pearson correlation between electrodes."""
    live = binary[binary.std(axis=1) > 0]
    if live.shape[0] < 2:
        return 1.0, 0.0
    c = np.corrcoef(live)
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0, 0.0
    return float(vals.mean()), float(vals.std())


def _isi_arrays(elec: np.ndarray, t: np.ndarray, n_elec: int,
                duration: float) -> np.ndarray:
    """Time-resolved inter-spike interval per electrode on a common grid."""
    n = max(int(duration / ISI_GRID_S), 1)
    out = np.full((n_elec, n), np.nan)
    grid = np.arange(n) * ISI_GRID_S
    for e in range(n_elec):
        ts = np.sort(t[elec == e])
        if ts.size < 3:
            continue
        isi = np.diff(ts)
        pos = np.searchsorted(ts, grid, side="right") - 1
        valid = (pos >= 0) & (pos < isi.size)
        out[e, valid] = isi[pos[valid]]
    return out


def compute(events: np.ndarray, n_elec: int, duration: float) -> np.ndarray:
    """Feature vector for one recording.

    events: (n, 2) array of [electrode index, spike time in seconds].
    Returns a vector of length `N_FEATURE`; a recording with no spikes returns
    zeros, which is a valid observation and not a failure.
    """
    out = np.zeros(N_FEATURE)
    if events.size == 0:
        out[NAMES.index("silent_frac")] = 1.0
        return out
    elec = events[:, 0].astype(np.int64)
    t = events[:, 1].astype(np.float64)
    keep = (t >= 0) & (t < duration) & (elec >= 0) & (elec < n_elec)
    elec, t = elec[keep], t[keep]
    if t.size == 0:
        out[NAMES.index("silent_frac")] = 1.0
        return out

    counts = _binned(elec, t, n_elec, duration, BIN_S)
    rate = counts / BIN_S
    rate_total = rate.sum(axis=0)
    per_elec_rate = counts.sum(axis=1) / duration
    active = per_elec_rate > 0.02
    n_active = int(active.sum())

    smooth = np.convolve(rate_total, _KERNEL, mode="same")
    peak = rate_total.max()
    frag_peaks = np.array([], dtype=int)
    if peak > 0:
        frag_peaks, _ = find_peaks(smooth, height=peak / 16.0, prominence=peak / 20.0)

    bursts, spikes_in = _detect_bursts(counts, rate_total, n_active)
    n_burst = bursts.shape[0]

    out[NAMES.index("mfr")] = t.size / duration / n_elec
    out[NAMES.index("nbr")] = n_burst * 60.0 / duration
    if n_burst:
        dur = (bursts[:, 1] - bursts[:, 0]) * BIN_S
        out[NAMES.index("nbd")] = float(dur.mean())
        out[NAMES.index("psib")] = float(spikes_in.sum() / t.size * 100.0)
        frag = [((frag_peaks > a) & (frag_peaks < b)).sum() for a, b in bursts]
        out[NAMES.index("fbs")] = float(np.mean(frag))
        out[NAMES.index("burst_peak_ratio")] = float(
            np.mean([rate_total[a:b].max() for a, b in bursts]) / max(peak, 1e-9))
        out[NAMES.index("burst_rise_s")] = float(np.mean(
            [(np.argmax(rate_total[a:b]) + 1) * BIN_S for a, b in bursts]))
        in_mask = np.zeros(rate_total.size, dtype=bool)
        for a, b in bursts:
            in_mask[a:b] = True
        inb = rate_total[in_mask]
        out[NAMES.index("in_burst_cv")] = float(inb.std() / max(inb.mean(), 1e-9))
    if n_burst > 1:
        ibi = (bursts[1:, 0] - bursts[:-1, 1]) * BIN_S
        out[NAMES.index("cvibi")] = float(ibi.std() / max(ibi.mean(), 1e-9))

    binary = (_binned(elec, t, n_elec, duration, CORR_BIN_S) > 0).astype(float)
    mean_cc, sd_cc = _pairwise(binary)
    out[NAMES.index("mean_cc")] = mean_cc
    out[NAMES.index("sd_cc")] = sd_cc

    if peak > 0:
        yf = np.abs(fft(rate_total))
        out[NAMES.index("mac")] = float(yf[1:].max() / max(yf[0], 1e-9))

    isi = _isi_arrays(elec, t, n_elec, duration)
    with np.errstate(invalid="ignore"):
        per_time = np.nanmean(isi, axis=0)
        out[NAMES.index("mean_isi")] = float(np.nanmean(per_time)) if np.isfinite(per_time).any() else 0.0
        out[NAMES.index("sd_isi_temporal")] = float(np.nanstd(per_time)) if np.isfinite(per_time).any() else 0.0
        across = np.nanstd(isi, axis=0)
        out[NAMES.index("sd_isi_electrode")] = float(np.nanmean(across)) if np.isfinite(across).any() else 0.0
    mean_icc, sd_icc = _pairwise(np.nan_to_num(isi, nan=0.0))
    out[NAMES.index("mean_isi_cc")] = mean_icc
    out[NAMES.index("sd_isi_cc")] = sd_icc

    live = [e for e in range(n_elec) if (elec == e).sum() > 2]
    dists = []
    for a_i in range(len(live)):
        for b_i in range(a_i + 1, len(live)):
            x = isi[live[a_i]]
            y = isi[live[b_i]]
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if ok.sum() < 10:
                continue
            r = x[ok] / y[ok]
            dists.append(np.mean(np.where(r <= 1, np.abs(r - 1), -(1.0 / r - 1))))
    out[NAMES.index("isi_dist")] = float(np.mean(dists)) if dists else 0.0

    out[NAMES.index("active_frac")] = n_active / n_elec
    if n_active:
        act = per_elec_rate[active]
        out[NAMES.index("rate_cv_electrode")] = float(act.std() / max(act.mean(), 1e-9))
    all_isi = []
    for e in range(n_elec):
        ts = np.sort(t[elec == e])
        if ts.size > 2:
            all_isi.append(np.diff(ts))
    if all_isi:
        a = np.concatenate(all_isi)
        out[NAMES.index("isi_cv")] = float(a.std() / max(a.mean(), 1e-9))
    out[NAMES.index("silent_frac")] = float((rate_total == 0).mean())

    _discriminative(out, elec, t, counts, rate_total, bursts, n_elec, duration)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sttc(elec: np.ndarray, t: np.ndarray, n_elec: int, duration: float,
          dt: float = 0.02) -> tuple[float, float]:
    """Spike time tiling coefficient, averaged over electrode pairs.

    Cutts and Eglen's measure of how tightly two trains tile in time. It is
    close to independent of firing rate, so it separates a change in synchrony
    from a change in how much the network fires.
    """
    trains = [np.sort(t[elec == e]) for e in range(n_elec)]
    live = [x for x in trains if x.size > 2]
    if len(live) < 2:
        return 0.0, 0.0

    def tiled(x: np.ndarray) -> float:
        # Fraction of the recording within dt of a spike of this train. The
        # windows are sorted and of equal width, so their union is the sum of
        # each window cut at the start of the next one.
        lo = np.clip(x - dt, 0, duration)
        hi = np.clip(x + dt, 0, duration)
        return float((np.minimum(hi[:-1], lo[1:]) - lo[:-1]).clip(0).sum()
                     + (hi[-1] - lo[-1])) / duration

    def near(a: np.ndarray, b: np.ndarray) -> float:
        # Fraction of spikes in a with a spike of b within dt on either side.
        k = np.searchsorted(b, a)
        after = np.abs(b[k.clip(0, b.size - 1)] - a)
        before = np.abs(a - b[(k - 1).clip(0, b.size - 1)])
        return float(np.mean(np.minimum(after, before) <= dt))

    tiles = [tiled(x) for x in live]
    vals = []
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            ta, tb = tiles[i], tiles[j]
            pa, pb = near(a, b), near(b, a)
            da = (pa - tb) / max(1 - pa * tb, 1e-9)
            db = (pb - ta) / max(1 - pb * ta, 1e-9)
            vals.append(0.5 * (da + db))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _discriminative(out: np.ndarray, elec: np.ndarray, t: np.ndarray,
                    counts: np.ndarray, rate_total: np.ndarray,
                    bursts: np.ndarray, n_elec: int, duration: float) -> None:
    """Fill the descriptors that separate one conductance from another."""
    s_mean, s_sd = _sttc(elec, t, n_elec, duration)
    out[NAMES.index("sttc_mean")] = s_mean
    out[NAMES.index("sttc_sd")] = s_sd

    in_burst = np.zeros(rate_total.size, dtype=bool)
    for a, b in bursts:
        in_burst[a:b] = True
    tonic_bins = (~in_burst).sum()
    tonic = (rate_total[~in_burst].sum() * BIN_S / max(tonic_bins * BIN_S, 1e-9)
             / n_elec) if tonic_bins else 0.0
    out[NAMES.index("tonic_rate")] = float(tonic)
    if bursts.shape[0]:
        peak = float(np.mean([rate_total[a:b].max() for a, b in bursts])) / n_elec
        out[NAMES.index("burst_tonic_ratio")] = float(np.log1p(peak)
                                                      - np.log1p(max(tonic, 0.0)))
        part, jit, dec, skew = [], [], [], []
        for a, b in bursts:
            block = counts[:, a:b]
            part.append(float((block.sum(axis=1) > 0).mean()))
            firsts = [np.argmax(block[e] > 0) for e in range(n_elec)
                      if block[e].sum() > 0]
            if len(firsts) > 1:
                jit.append(float(np.std(firsts) * BIN_S))
            prof = rate_total[a:b]
            k = int(np.argmax(prof))
            dec.append((len(prof) - k) / max(k + 1, 1))
            if prof.sum() > 0:
                idx = np.arange(len(prof))
                mu = float((idx * prof).sum() / prof.sum())
                var = float((((idx - mu) ** 2) * prof).sum() / prof.sum())
                if var > 1e-9:
                    skew.append(float((((idx - mu) ** 3) * prof).sum()
                                      / prof.sum() / var ** 1.5))
        out[NAMES.index("burst_participation")] = float(np.mean(part))
        out[NAMES.index("burst_participation_sd")] = float(np.std(part))
        if jit:
            out[NAMES.index("burst_onset_jitter")] = float(np.mean(jit))
        out[NAMES.index("burst_decay_ratio")] = float(np.mean(dec))
        if skew:
            out[NAMES.index("burst_skew")] = float(np.mean(skew))

    for name, width in (("fano_25ms", 0.025), ("fano_250ms", 0.25), ("fano_1s", 1.0)):
        n_bins = max(int(duration / width), 2)
        h, _ = np.histogram(t, bins=np.linspace(0, duration, n_bins + 1))
        out[NAMES.index(name)] = float(h.var() / max(h.mean(), 1e-9))

    r = rate_total - rate_total.mean()
    denom = float(np.dot(r, r))
    for name, lag_s in (("acorr_50ms", 0.05), ("acorr_200ms", 0.2), ("acorr_1s", 1.0)):
        lag = int(round(lag_s / BIN_S))
        if denom > 1e-9 and lag < r.size:
            out[NAMES.index(name)] = float(np.dot(r[:-lag], r[lag:]) / denom)

    cv2, logs = [], []
    for e in range(n_elec):
        ts = np.sort(t[elec == e])
        if ts.size > 3:
            isi = np.diff(ts)
            logs.append(np.log(np.clip(isi, 1e-6, None)))
            pairs = 2.0 * np.abs(isi[1:] - isi[:-1]) / np.clip(isi[1:] + isi[:-1], 1e-9, None)
            cv2.append(float(np.mean(pairs)))
    if cv2:
        out[NAMES.index("cv2")] = float(np.mean(cv2))
    if logs:
        allg = np.concatenate(logs)
        out[NAMES.index("log_isi_p10")] = float(np.percentile(allg, 10))
        out[NAMES.index("log_isi_p90")] = float(np.percentile(allg, 90))


def regime_stats(events: np.ndarray, n_elec: int, duration: float) -> dict:
    """The four statistics the regime criterion needs, and nothing else.

    Uses the same binning and the same burst detector as `compute`, so a
    recording judged living here is judged living there.
    """
    if events.size == 0:
        return {"mfr": 0.0, "active_frac": 0.0, "nbr": 0.0, "psib": 0.0}
    elec = events[:, 0].astype(np.int64)
    t = events[:, 1].astype(np.float64)
    keep = (t >= 0) & (t < duration) & (elec >= 0) & (elec < n_elec)
    elec, t = elec[keep], t[keep]
    if t.size == 0:
        return {"mfr": 0.0, "active_frac": 0.0, "nbr": 0.0, "psib": 0.0}
    counts = _binned(elec, t, n_elec, duration, BIN_S)
    rate_total = (counts / BIN_S).sum(axis=0)
    per_elec = counts.sum(axis=1) / duration
    n_active = int((per_elec > 0.02).sum())
    bursts, spikes_in = _detect_bursts(counts, rate_total, n_active)
    return {"mfr": t.size / duration / n_elec,
            "active_frac": n_active / n_elec,
            "nbr": bursts.shape[0] * 60.0 / duration,
            "psib": float(spikes_in.sum() / t.size * 100.0) if bursts.shape[0] else 0.0}


def compute_batch(list_of_events: list[np.ndarray], n_elec: int,
                  duration: float) -> np.ndarray:
    return np.stack([compute(e, n_elec, duration) for e in list_of_events])
