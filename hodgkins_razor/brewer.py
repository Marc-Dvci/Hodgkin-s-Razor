"""Reader for the four-compartment hippocampal chips of Lassers et al. (2023).

Lassers S., Vakilna Y.S., Tang W., Brewer G.J. The flow of axonal information
among hippocampal subregions: 2. Patterned stimulation sharpens routing of
information transmission. Dryad doi:10.5061/dryad.7h44j1013, Zenodo record
10257483. CC0 1.0.

Each culture grows in four compartments of one PDMS device on an MCS
120-electrode array: entorhinal cortex (EC), dentate gyrus (DG), CA3 and CA1,
joined in a loop by microfluidic tunnels 400 um long. Nineteen electrodes sit
under each compartment. Five tunnels per boundary are spanned by an electrode
pair; the delay between the pair gives each axon's direction, feed-forward
(EC > DG > CA3 > CA1 > EC) or feedback. Recordings are 300 s at 25 kHz:
nine unstimulated cultures, and six of them again after each of two
high-frequency stimulation patterns (HFS 5 and HFS 40).

The files hold, per culture, a table of tunnels (each axon's spike train at
the upstream and downstream electrode, by direction) and a table of well
electrodes (subregion and spike train). This module returns spike times only
once `PREREGISTRATION_v4.md` exists and matches its recorded hash. Before
that, `layout()` returns the labels alone: cultures, boundaries, tunnels and
electrode names.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "brewer2023"
CACHE = DATA / "cache"
FS = 25000.0
DURATION = 300.0
CONDITIONS = {"nostim": "NoStim", "hfs5": "HFS5", "hfs40": "HFS40"}
REGIONS = ("EC", "DG", "CA3", "CA1")
# Boundary label -> (upstream, downstream) along the trisynaptic loop.
BOUNDARIES = {"EC-DG": ("EC", "DG"), "DG-CA3": ("DG", "CA3"),
              "CA3-CA1": ("CA3", "CA1"), "CA1-EC": ("CA1", "EC")}


def _unblinded() -> bool:
    md = ROOT / "PREREGISTRATION_v4.md"
    rec = ROOT / "PREREGISTRATION_v4.sha256"
    if not (md.exists() and rec.exists()):
        return False
    return hashlib.sha256(md.read_bytes()).hexdigest() == rec.read_text().split()[0]


def _tables(condition: str):
    from matio import load_from_mat
    stem = CONDITIONS[condition]
    ax = load_from_mat(str(DATA / f"{stem}SortedAxons.mat"))
    ax = ax[[k for k in ax if not k.startswith("__")][0]]
    return ax


def layout(condition: str = "nostim") -> list[dict]:
    """Labels only: per culture, each tunnel's boundary and electrode pair."""
    ax = _tables(condition)
    out = []
    for i in range(ax.shape[1]):
        t = ax[0, i]
        out.append({"culture": i + 1,
                    "tunnels": [(str(s), str(p)) for s, p in
                                zip(t["Subregion"], t["ElectrodePairs"])]})
    return out


def _trains(cell) -> list[np.ndarray]:
    """Spike times (s) of every axon stored in one table cell."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [t for c in cell for t in _trains(c)]
    a = np.asarray(cell)
    if a.dtype == object and a.ndim == 0:
        inner = a.item()
        return [] if inner is cell or isinstance(inner, str) else _trains(inner)
    if a.dtype == object:
        out = []
        for c in a.ravel():
            out.extend(_trains(c))
        return out
    if a.size == 0 or a.dtype.kind not in "bui":
        return []
    a = np.atleast_2d(a)
    if a.shape[0] > a.shape[1]:
        a = a.T
    return [np.flatnonzero(row) / FS for row in a]


def _build_cache(condition: str) -> pathlib.Path:
    from matio import load_from_mat
    stem = CONDITIONS[condition]
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{condition}.npz"
    ax = _tables(condition)
    wl = load_from_mat(str(DATA / f"{stem}WellSpikes.mat"))
    wl = wl[[k for k in wl if not k.startswith("__")][0]]
    arrays = {}
    for i in range(ax.shape[1]):
        w = wl[0, i]
        for region in REGIONS:
            rows = w[w["Subregion"].astype(str) == region]
            times = [np.flatnonzero(np.asarray(s).ravel()) / FS for s in rows["SpikeTrain"]]
            arrays[f"c{i + 1}/well/{region}"] = (np.sort(np.concatenate(times))
                                                if times else np.zeros(0))
            arrays[f"c{i + 1}/nelec/{region}"] = np.array([len(rows)])
        t = ax[0, i]
        for j in range(len(t)):
            b, pair = str(t["Subregion"].iloc[j]), str(t["ElectrodePairs"].iloc[j])
            for d in ("ff", "fb"):
                for side in ("up", "down"):
                    for k, tr in enumerate(_trains(t[f"{side}_{d}"].iloc[j])):
                        arrays[f"c{i + 1}/axon/{b}/{pair}/{d}/{k}/{side}"] = tr
        del w, t
    np.savez_compressed(dest, **arrays)
    return dest


def load(condition: str = "nostim") -> list[dict]:
    """Every culture of one condition.

    Returns a list of dicts: `culture` (1-based, the order of the files),
    `well` (region -> pooled spike times of its electrodes, s), `nelec`
    (region -> electrode count) and `axons` (one dict per axon: boundary,
    electrode pair, direction 'ff' or 'fb', and spike times at the upstream
    and downstream electrodes).
    """
    if not _unblinded():
        raise SystemExit("the Lassers et al. spike trains are locked until "
                         "PREREGISTRATION_v4.md exists and matches its hash")
    path = CACHE / f"{condition}.npz"
    if not path.exists():
        path = _build_cache(condition)
    cultures: dict[int, dict] = {}
    with np.load(path) as z:
        for key in z.files:
            parts = key.split("/")
            c = cultures.setdefault(int(parts[0][1:]), {"well": {}, "nelec": {}, "axons": {}})
            if parts[1] == "well":
                c["well"][parts[2]] = z[key]
            elif parts[1] == "nelec":
                c["nelec"][parts[2]] = int(z[key][0])
            else:
                b, pair, d, k, side = parts[2:]
                ax = c["axons"].setdefault((b, pair, d, k), {"boundary": b, "pair": pair,
                                                            "direction": d})
                ax[side] = z[key]
    return [{"culture": i, "well": c["well"], "nelec": c["nelec"],
             "axons": list(c["axons"].values())} for i, c in sorted(cultures.items())]


def digest() -> dict[str, str]:
    """SHA-256 of every input file, so the scored data are pinned."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DATA.glob("*.mat"))}
