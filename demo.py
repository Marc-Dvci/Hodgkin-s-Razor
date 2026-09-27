"""One-command demonstration.

    python demo.py                 # bundled recordings, cached analyses, no GPU
    python demo.py --live          # re-run the twin on a CUDA device
    python demo.py --serve         # open the web application instead

Prints, for each bundled pair of recordings, what the twin says the compound
did, and whether that matches the published pharmacology. The answer key is
read from PREREGISTRATION.md and is not used by the model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, nde, params as P

EXAMPLES = ROOT / "data" / "examples"
KEY = {"CNQX": {"g_ampa"}, "D-AP5": {"g_nmda"}, "GABA": {"g_gaba", "g_tonic_inh"},
       "gabazine": {"g_gaba"}, "kainic acid": {"g_ampa"}, "TTX": {"g_na"},
       "Dynasore": {"u_rel", "tau_d"}, "vehicle control": None}
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


def line(char: str = "-", n: int = 92) -> None:
    print(char * n)


def run(live: bool) -> int:
    files = sorted(EXAMPLES.glob("*.json"))
    if not files:
        print("No examples found. Run: python scripts/make_examples.py")
        return 1

    twin = sim = None
    if live:
        from hodgkins_razor import simulator as S
        if not S.available():
            print("No CUDA device found; falling back to cached analyses.")
            live = False
        else:
            twin = {v: nde.Twin.load(ROOT / "models" / f"twin_v2_{v}", device="cuda")
                    for v in ("grid16", "grid12")}
            sim = S.Simulator()

    print()
    print("Hodgkin's Razor - which mechanism did the compound move?")
    print(f"{len(files)} recording pairs: Tampere comparative MEA dataset (CC BY 4.0)"
          + (" and Doorn et al. 2024 (Apache-2.0)" if any("dynasore" in f.name for f in files) else ""))
    line("=")
    print(f"{'recording':50s} {'top-1':12s} {'prob':>5s} {'verdict':9s} {'expected':20s}")
    line()

    hits = total = 0
    for path in files:
        d = json.loads(path.read_text())
        label = d["label"]
        want = KEY.get(d["compound"])
        if live:
            base = np.array(d["baseline"]).reshape(-1, 2)
            treat = np.array(d["treated"]).reshape(-1, 2)
            dur = float(d["duration"])
            n = int(d.get("n_elec", 16))
            view = "grid12" if n == 12 else "grid16"
            xb = F.compute(base, n, dur)
            xt = F.compute(treat, n, dur)
            rep = _live_report(twin[view], sim, xb, xt, dur, view)
        elif "analysis" in d:
            rep = d["analysis"]["report"]
        else:
            print(f"{label:50s} no cached analysis; run with --live")
            continue

        top = rep["mechanisms"][0]
        verdict = {"outside_model": "outside", "no_mechanism_called": "no call",
                   "mechanism_called": "called"}[rep["verdict"]]
        if want is None:
            expected = "nothing"
            mark = "+" if verdict != "called" else "."
        else:
            total += 1
            good = top["key"] in want
            hits += good
            expected = "/".join(sorted(want))
            mark = "+" if good else "."
        print(f"{label:50s} {top['key']:12s} {top['p_active']:5.2f} {verdict:9s} {expected:20s} {mark}")

    line()
    if total:
        print(f"top-1 on these single 60 s windows: {hits}/{total}   (chance about "
              f"{1 / len(SHIFT_KEYS):.2f}). 'outside': the guard says the twin cannot")
        print("reproduce the pair, so no mechanism is named. The pre-registered scores")
        print("average every window of every well: results/v2/RESULTS.md.")
    print("Full pre-registered evaluation: python scripts/evaluate_v2.py")
    print()
    return 0


def _live_report(twin, sim, xb, xt, dur, view):
    from hodgkins_razor import ppc, report
    post = twin.posterior(xb, xt)
    thr = float("inf")
    res = ROOT / "results" / "v2" / "results.json"
    if res.exists():
        thr = json.loads(res.read_text()).get("guard_thresholds", {}).get(view, {}).get("ppc", thr)
    g = ppc.check(twin, sim, xb, xt, thr, dur, 5.0, n_draws=24, post=post)
    typ = ppc.Typicality.for_twin(twin)
    guard = {"discrepancy": g["discrepancy"], "threshold": thr,
             "inside_model": g["inside_model"]}
    if typ is not None:
        t = typ.of_pair(twin, xb, xt)
        guard.update({"typicality": t, "typicality_threshold": typ.threshold})
        guard["inside_model"] = guard["inside_model"] and t <= typ.threshold
    return report.build(post, guard, xb, xt, meta={"duration_s": dur, "recording_system": view})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="re-run the twin instead of reading cached analyses")
    ap.add_argument("--serve", action="store_true", help="start the web application")
    args = ap.parse_args()
    if args.serve:
        cmd = [sys.executable, str(ROOT / "app" / "server.py")]
        if not args.live:
            cmd.append("--no-gpu")
        print("http://127.0.0.1:8000")
        raise SystemExit(subprocess.call(cmd))
    raise SystemExit(run(args.live))


if __name__ == "__main__":
    main()
