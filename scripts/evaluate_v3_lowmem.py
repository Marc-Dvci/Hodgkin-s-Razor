"""Run the frozen version 3 evaluation with one sister pair in memory at a time.

    python scripts/evaluate_v3_lowmem.py --sections A

`scripts/evaluate_v3.py` and `hodgkins_razor/charlesworth.py` are hashed in
PREREGISTRATION_v3.md and are not edited. The evaluation calls
`charlesworth.load` once and holds all 14,808 quadrant windows at the same
time, which ran the machine out of memory. This wrapper replaces that single
call with a lazy sequence. Iterating it runs the frozen `load` on one row of
`pairs_index()` after another, in the same order, and drops each row's
windows before reading the next. The pairs, their order and every number
computed from them are the same; only the peak memory changes.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import charlesworth as C

_load = C.load
_index = C.pairs_index


class LazyPairs:
    """The pairs `charlesworth.load` would return, read one sister pair at a time."""

    def __init__(self, kinds, min_div, max_div, window_s, n_windows, quadrants):
        self.kw = dict(kinds=kinds, min_div=min_div, max_div=max_div, window_s=window_s,
                       n_windows=n_windows, quadrants=quadrants)
        self.rows = [r for r in _index()
                     if r["kind"] in kinds and min_div <= r["div"] <= max_div]

    def __iter__(self):
        for row in self.rows:
            C.pairs_index = lambda row=row: [row]
            try:
                yield from _load(**self.kw)
            finally:
                C.pairs_index = _index

    def __len__(self) -> int:
        return len(self.rows) * len(self.kw["quadrants"]) * self.kw["n_windows"]


def lazy_load(kinds=("treated", "null"), min_div=9.0, max_div=99.0, window_s=60.0,
              n_windows=6, quadrants=(0, 1, 2, 3)):
    return LazyPairs(kinds, min_div, max_div, window_s, n_windows, quadrants)


if __name__ == "__main__":
    import evaluate_v3 as E
    E.C.load = lazy_load
    E.main()
