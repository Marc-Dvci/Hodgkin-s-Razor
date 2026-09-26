"""Reader for the Tampere comparative MEA dataset (CC BY 4.0).

Marttinen Rossi et al., Scientific Data 9:118 (2022).
gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset

The pharmacology plates hold three recordings of the same wells: a baseline, a
treated recording after wash-on, and a TTX recording. Every well therefore
gives a matched pair, which is what the paired model consumes. Electrodes the
authors marked as noisy are dropped, and the compound names are mapped to the
receptor or channel the answer key scores against.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "tampere" / "Data"

# Compound to the parameter its published mechanism moves. This is the answer
# key: it is fixed before any recording is scored, and the model never sees it.
MECHANISM: dict[str, tuple[str, str]] = {
    "CNQX": ("g_ampa", "down"),
    "D-AP5": ("g_nmda", "down"),
    "DAP5": ("g_nmda", "down"),
    "Gabazine": ("g_gaba", "down"),
    "GABA": ("g_gaba", "up"),
    "KainicAcid": ("g_ampa", "up"),
    "TTX": ("g_na", "down"),
}

PLATES = {
    "rat": {
        "species": "rat",
        "div": 22,
        "baseline": "Rat_MEA2_Pharmacology/Rat_MEA2Baseline_DIV22/Rat_MEA2Baseline_DIV22_spikes_noise_explogs/Rat_50618_MEA2Baseline_DIV22_spikes.csv",
        "treated": "Rat_MEA2_Pharmacology/Rat_MEA2Pharma_DIV22/Rat_MEA2Pharma_DIV22_spikes_noise_explogs/Rat_50618_MEA2Pharma_DIV22_spikes.csv",
        "ttx": "Rat_MEA2_Pharmacology/Rat_MEA2TTX_DIV22/Rat_MEA2TTX_DIV22_spikes_noise_explogs/Rat_50618_MEA2TTX_DIV22_spikes.csv",
        "log": "Rat_MEA2_Pharmacology/Rat_MEA2Pharma_DIV22/Rat_MEA2Pharma_DIV22_spikes_noise_explogs/Rat_50618_MEA2Pharma_DIV22_expLog.csv",
        "noisy": [
            "Rat_MEA2_Pharmacology/Rat_MEA2Baseline_DIV22/Rat_MEA2Baseline_DIV22_spikes_noise_explogs/noisy_electrodes_Rat_MEA2Baseline_DIV22.csv",
            "Rat_MEA2_Pharmacology/Rat_MEA2Pharma_DIV22/Rat_MEA2Pharma_DIV22_spikes_noise_explogs/noisy_electrodes_Rat_MEA2Pharma_DIV22.csv",
        ],
    },
    "human": {
        "species": "hPSC",
        "div": 29,
        "baseline": "hPSC_MEA3_Pharmacology/hPSC_MEA3Baseline_DIV29/hPSC_MEA3Baseline_DIV29_spikes_noise_explogs/hPSC_120618_MEA3Baseline_DIV29_spikes.csv",
        "treated": "hPSC_MEA3_Pharmacology/hPSC_MEA3Pharma_DIV29/hPSC_MEA3Pharma_DIV29_spikes_noise_explogs/hPSC_120618_MEA3Pharma_DIV29_spikes.csv",
        "ttx": "hPSC_MEA3_Pharmacology/hPSC_MEA3TTX_DIV29/hPSC_MEA3TTX_DIV29_spikes_noise_explogs/hPSC_120618_MEA3TTX_DIV29_spikes.csv",
        "log": "hPSC_MEA3_Pharmacology/hPSC_MEA3Pharma_DIV29/hPSC_MEA3Pharma_DIV29_spikes_noise_explogs/hPSC_120618_MEA3Pharma_DIV29_expLog.csv",
        "noisy": [
            "hPSC_MEA3_Pharmacology/hPSC_MEA3Baseline_DIV29/hPSC_MEA3Baseline_DIV29_spikes_noise_explogs/noisy_electrodes_hPSC_MEA3Baseline_DIV29.csv",
            "hPSC_MEA3_Pharmacology/hPSC_MEA3Pharma_DIV29/hPSC_MEA3Pharma_DIV29_spikes_noise_explogs/noisy_electrodes_hPSC_MEA3Pharma_DIV29.csv",
        ],
    },
}

N_ELEC = 16
ELECTRODE_ORDER = tuple(f"{r}{c}" for r in range(1, 5) for c in range(1, 5))


@dataclass
class Pair:
    """One well recorded before and after wash-on."""
    plate: str
    species: str
    well: str
    compound: str
    dose: str
    baseline: np.ndarray   # (n, 2) electrode, time
    treated: np.ndarray
    ttx: np.ndarray | None
    duration: float
    n_elec: int

    @property
    def is_control(self) -> bool:
        return self.compound == "Control"


def _read_spikes(path: pathlib.Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    parts = d["Channel"].str.split("_", n=1, expand=True)
    d["well"] = parts[0]
    d["elec"] = parts[1]
    return d


def _noisy(plate: dict) -> set[str]:
    bad: set[str] = set()
    for rel in plate["noisy"]:
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if "_" in line:
                bad.add(line)
    return bad


def _events(d: pd.DataFrame, well: str, bad: set[str], t0: float,
            t1: float) -> np.ndarray:
    sub = d[(d.well == well) & (d.Time >= t0) & (d.Time < t1)]
    if sub.empty:
        return np.zeros((0, 2))
    keep = ~(sub.well + "_" + sub.elec).isin(bad)
    sub = sub[keep]
    if sub.empty:
        return np.zeros((0, 2))
    idx = sub.elec.map({e: i for i, e in enumerate(ELECTRODE_ORDER)})
    ok = idx.notna()
    return np.stack([idx[ok].to_numpy(dtype=float),
                     sub.Time[ok].to_numpy() - t0], axis=1)


def load_plate(name: str, window_s: float = 60.0, n_windows: int = 3,
               start_s: float = 300.0) -> list[Pair]:
    """Matched baseline/treated pairs for one plate.

    Windows are taken well inside the recording so that the treated window is
    after wash-on has equilibrated, and the same window is used on both sides
    of a pair.
    """
    plate = PLATES[name]
    bad = _noisy(plate)
    base = _read_spikes(ROOT / plate["baseline"])
    treat = _read_spikes(ROOT / plate["treated"])
    ttx_path = ROOT / plate["ttx"]
    ttx = _read_spikes(ttx_path) if ttx_path.exists() else None
    log = pd.read_csv(ROOT / plate["log"])

    out: list[Pair] = []
    for _, row in log.iterrows():
        comp = str(row["Treatment"])
        if comp in ("ExcludedWell", "nan"):
            continue
        well = str(row["Well"])
        for w in range(n_windows):
            t0 = start_s + w * window_s
            t1 = t0 + window_s
            b = _events(base, well, bad, t0, t1)
            t = _events(treat, well, bad, t0, t1)
            x = _events(ttx, well, bad, t0, t1) if ttx is not None else None
            out.append(Pair(plate=name, species=plate["species"], well=well,
                            compound=comp, dose=str(row.get("Dose", "")),
                            baseline=b, treated=t, ttx=x,
                            duration=window_s, n_elec=N_ELEC))
    return out


def load_all(**kw) -> list[Pair]:
    return [p for name in PLATES for p in load_plate(name, **kw)]


def load_ttx(name: str, window_s: float = 60.0, n_windows: int = 3,
             start_s: float = 300.0) -> list[Pair]:
    """Baseline against the TTX recording, on vehicle wells only.

    TTX was applied on top of whatever each well already had, so only the
    control wells give a clean single-compound pair.
    """
    plate = PLATES[name]
    bad = _noisy(plate)
    base = _read_spikes(ROOT / plate["baseline"])
    ttx = _read_spikes(ROOT / plate["ttx"])
    log = pd.read_csv(ROOT / plate["log"])
    out: list[Pair] = []
    for _, row in log.iterrows():
        if str(row["Treatment"]) != "Control":
            continue
        well = str(row["Well"])
        for w in range(n_windows):
            t0 = start_s + w * window_s
            out.append(Pair(plate=name, species=plate["species"], well=well,
                            compound="TTX", dose="100nM",
                            baseline=_events(base, well, bad, t0, t0 + window_s),
                            treated=_events(ttx, well, bad, t0, t0 + window_s),
                            ttx=None, duration=window_s, n_elec=N_ELEC))
    return out


DEVELOPMENT = {
    "rat_MEA1": ("Rat_MEA1", "Rat_MEA1_spikes_noise_explogs", "Rat_190617_MEA1",
                 "noisy_electrodes_Rat_MEA1.csv"),
    "hPSC_MEA1": ("hPSC_MEA1", "hPSC_MEA1_spikes_noise_explogs", "hPSC_20517_MEA1",
                  "noisy_electrodes_hPSC_MEA1.csv"),
    "hPSC_MEA2": ("hPSC_MEA2", "hPSC_MEA2_spikes_noise_explogs", "hPSC_20517_MEA2",
                  "noisy_electrodes_hPSC_MEA2.csv"),
}


@dataclass
class Recording:
    """One development time point of one well."""
    plate: str
    species: str
    well: str
    div: int
    events: np.ndarray
    duration: float
    n_elec: int


def load_development(window_s: float = 60.0, start_s: float = 60.0,
                     n_windows: int = 1) -> list[Recording]:
    """Spontaneous recordings across days in vitro, with no compound applied.

    The day is an ordering the model never sees, which makes it an external
    check on the parameter axis.
    """
    out: list[Recording] = []
    for name, (folder, sub, stem, noisy) in DEVELOPMENT.items():
        base = ROOT / folder / sub
        if not base.exists():
            continue
        bad: set[str] = set()
        npath = base / noisy
        if npath.exists():
            bad = {l.strip() for l in npath.read_text().splitlines()
                   if "_" in l.strip()}
        species = "rat" if name.startswith("rat") else "hPSC"
        for f in sorted(base.glob(f"{stem}_DIV*_spikes.csv")):
            div = int(f.name.split("_DIV")[1].split("_")[0])
            d = _read_spikes(f)
            for well in sorted(d.well.unique()):
                for w in range(n_windows):
                    t0 = start_s + w * window_s
                    ev = _events(d, well, bad, t0, t0 + window_s)
                    out.append(Recording(plate=name, species=species, well=well,
                                         div=div, events=ev, duration=window_s,
                                         n_elec=N_ELEC))
    return out
