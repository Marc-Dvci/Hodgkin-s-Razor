"""How often each mechanism moved in the version 3 training simulations.

    python scripts/base_rate.py

The presence heads were trained on these pairs, so this is the rate a
probability should be read against: a treated preparation at the base rate is
one the twin is undecided about. Writes results/v3/base_rate.json.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hodgkins_razor import params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANKS = ("data/bank_v3", "data/bank_v3b")


def main() -> None:
    keys = [P.KEYS[i] for i in P.SHIFT_IDX]
    n, act, k = 0, np.zeros(len(keys)), []
    for b in BANKS:
        for f in sorted(glob.glob(str(ROOT / b / "shard_*.npz"))):
            with np.load(f) as z:
                a = z["active"]
            n += a.shape[0]
            act += a.sum(0)
            k.append(a.sum(1))
    k = np.concatenate(k)
    out = {"banks": list(BANKS), "pairs": int(n),
           "rate": {key: float(v / n) for key, v in zip(keys, act)},
           "mean_active": float(k.mean()), "share_none_active": float((k == 0).mean())}
    dest = ROOT / "results" / "v3" / "base_rate.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
