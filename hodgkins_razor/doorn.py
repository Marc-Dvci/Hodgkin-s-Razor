"""Reader for the Dynasore recordings of Doorn et al. (Stem Cell Reports 2024).

Doorn N., Voogd E.J.H.F., Levers M.R., van Putten M.J.A.M., Frega M.
"Breaking the burst: unveiling mechanisms behind fragmented network bursts in
patient-derived neurons." Stem Cell Rep. 19:1583 (2024).
Peak trains: gitlab.utwente.nl/m7706783/fb_model (Apache-2.0).

Human iPSC-derived excitatory neurons (Ngn2) with rat astrocytes, recorded on a
Multi Channel Systems 24-well MEA at 12 electrodes per well and 10 kHz, at 35
days in vitro. 10 uM Dynasore was added after a baseline recording. The wells
and the analysis windows are the ones the authors list in their Table S2.

This dataset is the blind test of the second pre-registration. The treated
windows are only returned once `PREREGISTRATION_v2.md` exists and matches its
recorded hash, so no model output on a treated recording can be produced
before the answer key and the metric are fixed.
"""
from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "doorn_fb_model" / "Experimental_peaktrains"
FS = 10_000.0
N_ELEC = 12

# Table S2, "Dynasore in vitro": well -> (file, baseline window, treated window).
WELLS: dict[str, tuple[str, tuple[float, float], tuple[float, float]]] = {
    "FB2_B6": ("APS_FB2_B6.mat", (0.0, 300.0), (1550.0, 1850.0)),
    "FB2_C6": ("APS_FB2_C6.mat", (0.0, 300.0), (1550.0, 1850.0)),
    "FB2_D6": ("APS_FB2_D6.mat", (0.0, 300.0), (1550.0, 1850.0)),
    "FB2t_A2": ("APS_FB2t_Dyn_tot_A2.mat", (0.0, 300.0), (950.0, 1250.0)),
    "FB2t_A3": ("APS_FB2t_Dyn_tot_A3.mat", (0.0, 300.0), (950.0, 1250.0)),
    "FB2t_B1": ("APS_FB2t_Dyn_tot_B1.mat", (0.0, 300.0), (950.0, 1250.0)),
    "FB2t_C1": ("APS_FB2t_Dyn_tot_C1.mat", (0.0, 300.0), (950.0, 1250.0)),
    "FB3t_A3": ("APS_FB3t_Dyn_A3.mat", (0.0, 300.0), (1520.0, 1820.0)),
    "FB3t_C3": ("APS_FB3t_Dyn_C3.mat", (0.0, 300.0), (1520.0, 1820.0)),
    "FB3t_D1": ("APS_FB3t_Dyn_tot_D1.mat", (0.0, 300.0), (1520.0, 1820.0)),
}
PLATE_OF = {w: w.split("_")[0] for w in WELLS}


@dataclass
class Pair:
    """One well before and after Dynasore, cut into matched windows."""
    plate: str
    species: str
    well: str
    compound: str
    dose: str
    baseline: np.ndarray
    treated: np.ndarray | None
    duration: float
    n_elec: int

    @property
    def is_control(self) -> bool:
        return False


def _load(name: str) -> np.ndarray:
    from scipy.io import loadmat
    a = loadmat(DATA / name)["Ts_AP"]
    return np.stack([a[:, 0].astype(float), a[:, 1] / FS], axis=1)


def _cut(ev: np.ndarray, t0: float, t1: float) -> np.ndarray:
    sel = (ev[:, 1] >= t0) & (ev[:, 1] < t1)
    out = ev[sel].copy()
    out[:, 1] -= t0
    return out


def _unblinded() -> bool:
    md = ROOT / "PREREGISTRATION_v2.md"
    rec = ROOT / "PREREGISTRATION_v2.sha256"
    if not (md.exists() and rec.exists()):
        return False
    return hashlib.sha256(md.read_bytes()).hexdigest() == rec.read_text().split()[0]


def load(window_s: float = 60.0, treated: bool = False) -> list[Pair]:
    """Matched windows for every Table S2 well.

    With `treated=False` only the baseline side is returned, which is what the
    label-free domain estimate may read. The treated side needs the second
    pre-registration to be in place.
    """
    if treated and not _unblinded():
        raise SystemExit("treated Dynasore windows are locked until "
                         "PREREGISTRATION_v2.md exists and matches its hash")
    out: list[Pair] = []
    for well, (fname, (b0, b1), (t0, t1)) in WELLS.items():
        ev = _load(fname)
        n_win = int((b1 - b0) // window_s)
        for w in range(n_win):
            base = _cut(ev, b0 + w * window_s, b0 + (w + 1) * window_s)
            trt = (_cut(ev, t0 + w * window_s, t0 + (w + 1) * window_s)
                   if treated else None)
            out.append(Pair(plate=PLATE_OF[well], species="hiPSC-Ngn2",
                            well=well, compound="Dynasore", dose="10uM",
                            baseline=base, treated=trt, duration=window_s,
                            n_elec=N_ELEC))
    return out
