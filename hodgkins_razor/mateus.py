"""Reader for the microchannel chip recordings of Mateus et al. (2024).

Mateus J., Melo P., Aroso M., Charlot B., Aguiar P. "Influence of asymmetric
microchannels in the structure and function of engineered neuronal circuits."
bioRxiv 10.1101/2024.07.09.602729. Data: Zenodo 14525182.

The dataset README states a CC BY-NC-ND licence: it may be used for research
with credit, and it may not be redistributed in altered form or used
commercially. This project downloads it with `scripts/fetch_mateus.py`, reads
it in place, and publishes only derived statistics.

Rat hippocampal neurons (E18) seeded in both chambers of a two-compartment
device, recorded at 12 to 21 days in vitro for 10 minutes on a Multi Channel
Systems MEA2100-256. Sixteen microchannels lie along the sixteen electrode
columns; each channel covers five electrodes (rows 7 to 11), and each chamber
is probed by the rows above and below. Channel designs: straight controls, and
the asymmetric Tesla, Tesla v2, Rams and Arrows.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "mateus2024" / "1_MEA_Recordings" / "Recordings"
COLUMNS = "ABCDEFGHJKLMNOPR"          # MCS 256MEA column letters (no I, Q)
CHANNEL_ROWS = (7, 8, 9, 10, 11)      # five electrodes inside each channel
CHAMBER_ROWS = {"upper": tuple(range(1, 7)), "lower": tuple(range(12, 17))}
MAX_STEP_S = 0.001                    # Mateus et al.: <= 1 ms per electrode pair

DESIGNS = {"control": "control", "tesla_v2": "tesla_v2", "tesla": "tesla",
           "rams": "rams", "ovaries": "rams", "arrows": "arrows"}


@dataclass
class Recording:
    path: pathlib.Path
    experiment: str
    design: str
    chip: str
    div: int
    duration: float
    spikes: dict          # label -> spike times in seconds
    unreadable: list | None = None   # electrodes whose data failed to decode


def _design(path: pathlib.Path) -> str:
    folder = path.parent.name.lower()
    name = path.name.lower()
    if "tesla_v2" in name or folder == "tesla_v2":
        return "tesla_v2"
    for key, design in DESIGNS.items():
        if key in name or key == folder:
            return design
    return folder


def load(path: pathlib.Path) -> Recording:
    import h5py
    with h5py.File(path, "r") as h:
        rec = h["Data/Recording_0"]
        duration = float(rec.attrs["Duration"]) / 1e6
        s = rec["TimeStampStream/Stream_0"]
        info = s["InfoTimeStamp"][()]
        spikes = {}
        unreadable: list[str] = []
        for ent, label, exp in zip(info["TimeStampEntityID"], info["Label"],
                                   info["Exponent"]):
            label = label.decode() if isinstance(label, bytes) else str(label)
            # An electrode that recorded no spike has no entity in some files.
            key = f"TimeStampEntity_{ent}"
            if key not in s:
                spikes[label] = np.zeros(0)
                continue
            try:
                t = s[key][()].ravel().astype(np.float64)
            except OSError:
                # A compressed block that fails to decode is counted, not guessed.
                unreadable.append(label)
                spikes[label] = np.zeros(0)
                continue
            spikes[label] = np.sort(t * 10.0 ** int(exp))
    m = re.search(r"chip_(\d+)", path.name)
    d = re.search(r"DIV(\d+)", path.name)
    return Recording(path=path, experiment=path.parent.parent.name,
                     design=_design(path), chip=m.group(1) if m else "?",
                     div=int(d.group(1)) if d else -1, duration=duration,
                     spikes=spikes, unreadable=unreadable)


def recordings() -> list[pathlib.Path]:
    return sorted(DATA.glob("EXP*/*/*.h5"))


def _train(rec: Recording, col: str, row: int) -> np.ndarray:
    return rec.spikes.get(f"{col}{row}", np.zeros(0))


def propagation(rec: Recording) -> dict:
    """Propagating events along each channel, by direction.

    An event is a spike on the first channel electrode followed, electrode by
    electrode along the channel, by a spike within 1 ms on each of the next
    four: the criterion of Mateus et al. "Down" runs from row 7 to row 11,
    "up" from row 11 to row 7. Which chamber is the source is not stored in the
    files, so directionality is reported without a sign: the share of events
    running the channel's dominant way.
    """
    down = up = 0
    per_channel = []
    for col in COLUMNS:
        trains = [_train(rec, col, r) for r in CHANNEL_ROWS]
        if any(t.size == 0 for t in trains):
            per_channel.append((0, 0))
            continue

        def chain(order):
            t = trains[order[0]]
            ok = np.ones(t.size, dtype=bool)
            cur = t.copy()
            for k in order[1:]:
                nxt = trains[k]
                idx = np.searchsorted(nxt, cur)
                idx = np.clip(idx, 0, nxt.size - 1)
                cand = nxt[idx]
                good = (cand >= cur) & (cand - cur <= MAX_STEP_S)
                ok &= good
                cur = np.where(good, cand, cur)
            return int(ok.sum())

        d = chain(range(5))
        u = chain(range(4, -1, -1))
        per_channel.append((d, u))
        down += d
        up += u
    total = down + up
    frac = down / total if total else 0.5
    return {"down": down, "up": up, "events": total,
            "dominant_share": max(frac, 1 - frac) if total else float("nan"),
            "directionality": abs(2 * frac - 1) if total else float("nan"),
            "per_channel": per_channel}


def chamber_rates(rec: Recording, bin_s: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Population spike counts of the two chambers at 5 ms resolution."""
    n = int(np.ceil(rec.duration / bin_s))
    out = []
    for rows in CHAMBER_ROWS.values():
        h = np.zeros(n)
        for col in COLUMNS:
            for r in rows:
                t = _train(rec, col, r)
                if t.size:
                    h += np.bincount(np.clip((t / bin_s).astype(int), 0, n - 1),
                                     minlength=n)[:n]
        out.append(h)
    return out[0], out[1]


def chamber_asymmetry(rec: Recording, max_lag_s: float = 0.3,
                      bin_s: float = 0.005) -> dict:
    """What electrodes under the two chambers show about direction, passively.

    The same statistic as the chip twin's compartment readout
    (`chip.cross_stats`): the asymmetry of the cross-correlation of the two
    chambers' population rates between positive and negative lags, reported
    without a sign for the same reason as above.
    """
    a, b = chamber_rates(rec, bin_s)
    if a.sum() < 20 or b.sum() < 20:
        return {"asymmetry": float("nan"), "peak": float("nan"),
                "lag_s": float("nan"), "rate_upper": float(a.sum() / rec.duration),
                "rate_lower": float(b.sum() / rec.duration)}
    rate_upper, rate_lower = a.sum() / rec.duration, b.sum() / rec.duration
    a = a - a.mean()
    b = b - b.mean()
    n = int(round(max_lag_s / bin_s))
    cc = np.array([np.dot(a[max(0, -l):len(a) - max(0, l)],
                          b[max(0, l):len(b) - max(0, -l)])
                   for l in range(-n, n + 1)], dtype=float)
    cc /= np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-9
    pos, neg = np.clip(cc[n + 1:], 0, None).sum(), np.clip(cc[:n], 0, None).sum()
    asym = (pos - neg) / (pos + neg + 1e-9)
    return {"asymmetry": float(abs(asym)), "signed_asymmetry": float(asym),
            "peak": float(cc.max()), "lag_s": float((np.argmax(cc) - n) * bin_s),
            "rate_upper": float(rate_upper), "rate_lower": float(rate_lower)}
