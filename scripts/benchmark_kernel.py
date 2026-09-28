"""Measure the simulator's throughput on this machine.

    python scripts/benchmark_kernel.py

Simulates batches of cultures from the Axion domain for 65 simulated seconds
(5 s transient + 60 s recorded), the length of one bank pair side, and reports
networks per wall-clock second over the timed repeats (the first batch warms up
the kernel and is not timed). Writes results/throughput.json.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import simulator as S
from fit_domain import GaussianProposal

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    dom = json.loads((ROOT / "models" / "domain.json").read_text())
    prop = GaussianProposal.load(dom["views"]["grid16"]["proposal"])
    rng = np.random.default_rng(0)
    sim = S.Simulator()
    batch, repeats = 384, 3
    sim.run(prop.draw(batch, rng), duration_s=60.0, transient_s=5.0, seed=1)
    times = []
    for k in range(repeats):
        theta = prop.draw(batch, rng)
        t0 = time.perf_counter()
        sim.run(theta, duration_s=60.0, transient_s=5.0, seed=10 + k)
        times.append(time.perf_counter() - t0)
    rate = batch * repeats / sum(times)
    try:
        import cupy
        gpu = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        gpu = "unknown"
    out = {"networks_per_s": round(rate), "batch": batch, "repeats": repeats,
           "simulated_seconds_per_network": 65, "gpu": gpu, "python": platform.python_version()}
    (ROOT / "results" / "throughput.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
