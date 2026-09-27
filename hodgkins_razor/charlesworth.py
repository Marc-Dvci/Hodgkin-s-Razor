"""Reader for the sister-culture recordings of Charlesworth et al. (2015).

Charlesworth P., Morton A., Eglen S.J., Komiyama N.H., Grant S.G.N.
"Canalization of genetic and pharmacological perturbations in developing
primary neuronal activity patterns." Neuropharmacology 100:47-55 (2015).
Data: Zenodo 31085, CC0 (public domain).

Mouse hippocampal neurons on Multi Channel Systems 60-electrode arrays (8 x 8
grid, 200 um spacing, 30 um electrodes, the four corners absent and electrode
15 the internal reference), recorded twice a week from 6 to 30 days in vitro
for about 15 minutes. Spikes were detected at -20 uV; the shortest interval on
any electrode is 1.08 ms, which is the detector's dead time.

Each preparation was plated on two sister arrays, A and B, recorded together.
After the recording at 7 days, 50 uM APV was added to the medium of the B array
of some preparations and kept there, so every later B recording has NMDA
receptors blocked. Other preparations kept both arrays untreated throughout.
That gives three kinds of pair, each an A array against its B sister recorded
the same day:

* treated: the B sister carries APV;
* null: neither sister was treated, at the same ages as the treated pairs;
* pre-drug: any preparation at 7 days or earlier, before APV was added.

This dataset is the blind test of the third pre-registration. Every recording
from a B or D array at 8 days or later is the second half of a scored pair, and
is returned only once `PREREGISTRATION_v3.md` exists and matches its recorded
hash. Array A and C recordings and anything recorded at 7 days or earlier can be read
before that, which is what the domain and the drift calibration use.

The twin's simulated array is a 4 x 4 grid, so each 8 x 8 recording is read as
its four 4 x 4 quadrants, each through the layout of the 24-well MCS plate (the
four corners dropped). The array's missing corners and its reference electrode
all fall on quadrant corners, so every quadrant keeps twelve recording
electrodes.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "charlesworth2015" / "g2c-1"
META = DATA / "00g2cdata.csv"
# Parsed spike trains, one file per recording; the text files take seconds each.
CACHE = DATA.parent / "cache"
LOCK_FROM_DIV = 8
SECOND = ("B", "D")   # the sister that may carry the drug; A and C never do
DEAD_MS = 1.08
N_ELEC = 12
# Local 4 x 4 index is column * 4 + row, as in simulator._geometry; the corners
# 0, 3, 12 and 15 are dropped, as in the grid12 view.
_CORNERS = (0, 3, 12, 15)
_KEEP = tuple(i for i in range(16) if i not in _CORNERS)
_LOCAL = {i: k for k, i in enumerate(_KEEP)}


@dataclass(frozen=True)
class File:
    name: str
    prep: str
    array: str
    div: float
    genotype: str
    drug: str

    @property
    def treated(self) -> bool:
        return self.drug != ""


@dataclass
class Pair:
    """Sister arrays of one preparation, recorded the same day, one quadrant."""
    prep: str
    kind: str            # "treated", "null" or "pre-drug"
    div: float
    genotype: str
    compound: str        # "APV" for treated pairs, "none" otherwise
    quadrant: int
    window: int
    baseline: np.ndarray
    treated: np.ndarray
    duration: float
    n_elec: int = N_ELEC

    # Field names shared with the other readers, so the evaluation code can
    # score any of them.
    @property
    def plate(self) -> str:
        return self.prep

    @property
    def well(self) -> str:
        return f"{self.prep}_DIV{self.div:g}"

    @property
    def species(self) -> str:
        return "mouse"

    @property
    def is_control(self) -> bool:
        return self.kind != "treated"


def _parse(name: str) -> tuple[str, str] | None:
    """Preparation and array letter from a file name, or None for an unpaired file."""
    a = re.search(r"_([A-D])\.\w+$", name)
    m = re.search(r"TC0*(\d+)", name)
    if not a or not m:
        return None
    prefix = name[:m.start()].strip("_-")
    return f"{prefix + '_' if prefix else ''}TC{int(m.group(1))}", a.group(1)


def catalogue() -> list[File]:
    """Every file with a sister array, from the metadata sheet."""
    import pandas as pd
    d = pd.read_csv(META)
    d.columns = ["file", "age", "genotype", "strain", "drug"]
    out = []
    for r in d.itertuples(index=False):
        key = _parse(str(r.file))
        if key is None:
            continue
        drug = "" if not isinstance(r.drug, str) else r.drug.strip()
        out.append(File(name=str(r.file), prep=key[0], array=key[1], div=float(r.age),
                        genotype=str(r.genotype), drug=drug))
    return out


# GluR1 and GluR-A are two names for the same knockout (Gria1, the GluA1
# subunit), used in different preparations of the same series.
_SAME_GENE = {"GluRAnull": "GluR1"}


def _gene(genotype: str) -> str:
    return _SAME_GENE.get(genotype, genotype)


def _sister(array: str) -> str | None:
    return {"A": "B", "C": "D"}.get(array)


def pairs_index() -> list[dict]:
    """Every same-day sister pair, labelled by kind. Reads only the metadata."""
    files = catalogue()
    by = {(f.prep, f.array, f.div): f for f in files}
    # A preparation is an APV preparation if any of its B recordings carries APV.
    apv_preps = {f.prep for f in files if f.drug == "APV"}
    out = []
    for f in files:
        s = _sister(f.array)
        if s is None or (f.prep, s, f.div) not in by:
            continue
        g = by[(f.prep, s, f.div)]
        if f.treated or _gene(g.genotype) != _gene(f.genotype):
            continue          # the first array is never treated; sisters share a genotype
        if f.div <= LOCK_FROM_DIV - 1:
            kind = "pre-drug"
        elif g.drug == "APV":
            kind = "treated"
        elif not g.treated and f.prep not in apv_preps:
            kind = "null"
        else:
            continue
        out.append({"prep": f.prep, "div": f.div, "genotype": f.genotype,
                    "kind": kind, "first": f.name, "second": g.name,
                    "first_genotype": f.genotype, "second_genotype": g.genotype})
    return out


def _unblinded() -> bool:
    md = ROOT / "PREREGISTRATION_v3.md"
    rec = ROOT / "PREREGISTRATION_v3.sha256"
    if not (md.exists() and rec.exists()):
        return False
    return hashlib.sha256(md.read_bytes()).hexdigest() == rec.read_text().split()[0]


def locked(name: str, div: float) -> bool:
    key = _parse(name)
    return (key is not None and key[1] in SECOND and div >= LOCK_FROM_DIV
            and not _unblinded())


def read(name: str, div: float) -> dict[str, np.ndarray]:
    """Spike times per electrode label ("12" = column 1, row 2), in seconds.

    Two formats occur: `.mct` files lead with start, stop and sweep columns;
    `.nexTimestamps` files hold only the electrode columns. Electrodes with no
    detected spike are absent from both, and are read as silent.
    """
    if locked(name, div):
        raise SystemExit(f"{name} is the second half of a scored pair and is locked "
                         "until PREREGISTRATION_v3.md exists and matches its hash")
    cache = CACHE / (name + ".npz")
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import pandas as pd
    # Shorter columns are padded with blanks or single spaces.
    d = pd.read_csv(DATA / name, sep="\t", engine="c", skipinitialspace=True,
                    dtype=str).apply(pd.to_numeric, errors="coerce")
    out = {}
    for h in d.columns:
        m = re.match(r"ch_(\d\d)", str(h))
        if m:
            c = d[h].to_numpy()
            out[m.group(1)] = np.sort(c[np.isfinite(c)])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def quadrant_events(spikes: dict[str, np.ndarray], q: int, t0: float, t1: float) -> np.ndarray:
    """[local electrode, time from t0] for one quadrant and one window."""
    c0, r0 = (q // 2) * 4, (q % 2) * 4
    rows = []
    for label, t in spikes.items():
        col, row = int(label[0]) - 1, int(label[1]) - 1
        if not (c0 <= col < c0 + 4 and r0 <= row < r0 + 4):
            continue
        local = (col - c0) * 4 + (row - r0)
        if local not in _LOCAL:
            continue
        sel = t[(t >= t0) & (t < t1)] - t0
        rows.append(np.stack([np.full(sel.size, _LOCAL[local], float), sel], axis=1))
    if not rows:
        return np.zeros((0, 2))
    ev = np.concatenate(rows)
    return ev[np.argsort(ev[:, 1], kind="stable")]


def windows(start_s: float = 60.0, window_s: float = 60.0, n: int = 6) -> list[tuple[float, float]]:
    return [(start_s + k * window_s, start_s + (k + 1) * window_s) for k in range(n)]


def load(kinds: tuple[str, ...] = ("treated", "null"), min_div: float = 9.0,
         max_div: float = 99.0, window_s: float = 60.0, n_windows: int = 6,
         quadrants: tuple[int, ...] = (0, 1, 2, 3)) -> list[Pair]:
    """Matched sister pairs, cut into quadrants and windows matched by position."""
    out = []
    for row in pairs_index():
        if row["kind"] not in kinds or not (min_div <= row["div"] <= max_div):
            continue
        a = read(row["first"], row["div"])
        b = read(row["second"], row["div"])
        for q in quadrants:
            for w, (t0, t1) in enumerate(windows(60.0, window_s, n_windows)):
                out.append(Pair(prep=row["prep"], kind=row["kind"], div=row["div"],
                                genotype=row["genotype"],
                                compound="APV" if row["kind"] == "treated" else "none",
                                quadrant=q, window=w,
                                baseline=quadrant_events(a, q, t0, t1),
                                treated=quadrant_events(b, q, t0, t1),
                                duration=window_s))
    return out


def baselines(min_div: float = 9.0, window_s: float = 60.0, n_windows: int = 6,
              kinds: tuple[str, ...] = ("treated", "null")) -> list[np.ndarray]:
    """First-array quadrant windows of every scored pair: the untreated side only."""
    out = []
    for row in pairs_index():
        if row["kind"] not in kinds or row["div"] < min_div:
            continue
        a = read(row["first"], row["div"])
        for q in range(4):
            for t0, t1 in windows(60.0, window_s, n_windows):
                out.append(quadrant_events(a, q, t0, t1))
    return out


def digest() -> str:
    """SHA-256 of the metadata sheet, so the scored set is pinned."""
    return hashlib.sha256(META.read_bytes()).hexdigest()
