"""Write results/v4/RESULTS.md and the version 4 figure from results/v4/results.json.

    python scripts/render_v4.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import figures_v2 as FV   # noqa: E402,F401  (applies the shared matplotlib style)
from figures_v2 import INK, INK2, MUTED, NEUTRAL, S1, S2, SURFACE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
V4 = ROOT / "results" / "v4"
NAMES = {"nostim": "unstimulated\n(primary, 9 cultures)", "hfs5": "after HFS 5\n(6 cultures)",
         "hfs40": "after HFS 40\n(6 cultures)"}


def conditions(r: dict) -> list[tuple[str, dict]]:
    return [("nostim", r["primary"])] + [(c, r["secondary"][c]) for c in ("hfs5", "hfs40")]


def figure(r: dict, pred: dict) -> None:
    rows = conditions(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    y = np.arange(len(rows))[::-1]
    v = r["verdicts"]
    t1 = "P1. Axons follow their home compartment: " + ("met" if v["P1_home_share"]["met"] else "not met")
    t2 = "P2. Compartment timing vs traffic: " + ("met" if v["P2_asym_vs_traffic"]["met"] else "not met")
    for ax, key, title in ((a1, "home", t1), (a2, "asym_vs_traffic_ff", t2)):
        for yi, (c, s) in zip(y, rows):
            d = s[key]
            v = d["share_positive"] if key == "home" else d["spearman"]
            lo, hi = d["ci95_by_culture"]
            pi = d["twin_pi90"]
            ax.fill_betweenx([yi - 0.28, yi + 0.28], pi[0], pi[1], color=NEUTRAL, alpha=0.55,
                             lw=0, zorder=1)
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.6, zorder=2)
            ok = (v >= pi[0]) if key == "home" else (pi[0] <= v <= pi[1])
            ax.scatter([v], [yi], s=70, color=S1 if ok else S2, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
            n = d["n"] if key == "home" else s["boundary_units_scored"]
            lab = f"{v:.2f}" + (f"  ({d['k']}/{n})" if key == "home" else f"  (n {n})")
            ax.text(hi + 0.03, yi, lab, va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [NAMES[c] for c, _ in rows] if ax is a1 else [""] * len(rows))
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(rows) - 0.3)
    a1.axvline(0.5, color=MUTED, lw=1, ls=":")
    a1.text(0.505, -0.62, "no preference", color=MUTED, fontsize=8)
    a1.set_xlim(0.4, 1.12)
    a1.set_xticks(np.arange(0.4, 1.01, 0.1))
    a1.set_xlabel("share of axon pools with home > 0 (95% CI, cultures resampled)")
    a2.axvline(0.0, color=MUTED, lw=1, ls=":")
    a2.set_xlim(-0.75, 0.85)
    a2.set_xlabel("Spearman, compartment lead vs feed-forward share (95% CI)")
    a2.text(0.98, 0.02, "grey band: the twin's 90% interval,\nfixed before the data were read",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    out = V4 / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "brewer_chip.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / "brewer_chip.png")


def f(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def results_md(r: dict, pred: dict) -> str:
    p = r["primary"]
    v = r["verdicts"]
    h = p["home"]
    t = p["asym_vs_traffic_ff"]
    a = p["asym_vs_axons_ff"]
    L = ["# Version 4 results: the chip twin on a four-compartment chip", "",
         f"Pre-registration `PREREGISTRATION_v4.md`, sha256 `{r['preregistration'][:16]}`, "
         "frozen and pushed (commit `07bc64c`) before any spike of the dataset was counted. "
         "Data: Lassers, Vakilna, Tang and Brewer (2023), Zenodo 10257483, CC0 1.0: "
         "hippocampal EC, DG, CA3 and CA1 grown in four compartments of one device on a "
         "120-electrode array, 19 electrodes per compartment, five monitored tunnels per "
         "boundary with an electrode pair that gives each axon's direction. "
         "Scored by `scripts/evaluate_v4.py`; every number below is in `results/v4/results.json`.", "",
         "![P1 and P2 on the unstimulated and stimulated recordings](figures/brewer_chip.png)", "",
         "## Primary outcomes (unstimulated recordings, 9 cultures)", "",
         "| Outcome | Twin, fixed before scoring | Recorded | Verdict |", "|---|---|---|---|",
         f"| P1. Share of axon pools whose spikes follow the compartment they grow from more "
         f"closely than the one they grow into | {f(pred['home']['share_positive'])}; at least "
         f"{f(h['twin_pi90'][0])} at n = {h['n']} | **{f(h['share_positive'])}** ({h['k']}/{h['n']}; "
         f"cultures resampled {f(h['ci95_by_culture'][0])} to {f(h['ci95_by_culture'][1])}; "
         f"sign test against 0.5 p = {h['sign_test_p_vs_half']:.1g}) | "
         f"**{'met' if v['P1_home_share']['met'] else 'not met'}** |",
         f"| P2. Spearman of the compartments' timing asymmetry (upstream leading) with the "
         f"feed-forward share of axonal spikes | {f(pred['whole_bank_spearman']['asym_vs_traffic_ff'])}; "
         f"inside [{f(t['twin_pi90'][0])}, {f(t['twin_pi90'][1])}] at n = {p['boundary_units_scored']} | "
         f"**{f(t['spearman'])}** (cultures resampled {f(t['ci95_by_culture'][0])} to "
         f"{f(t['ci95_by_culture'][1])}; permutation p = {f(t['permutation_p'])}) | "
         f"**{'met' if v['P2_asym_vs_traffic']['met'] else 'not met'}**: below the interval |", "",
         "## Secondary outcomes", "",
         "| Condition | Boundaries / axon pools scored | P1 share | P2 Spearman | Spearman with axon share |",
         "|---|---|---|---|---|"]
    for c, s in conditions(r):
        L.append(f"| {NAMES[c].split(chr(10))[0]} | {s['boundary_units_scored']} / {s['home']['n']} | "
                 f"{f(s['home']['share_positive'])} ({'at or above' if s['home']['at_or_above_lower_bound'] else 'below'} "
                 f"{f(s['home']['twin_pi90'][0])}) | {f(s['asym_vs_traffic_ff']['spearman'])} "
                 f"({'inside' if s['asym_vs_traffic_ff']['inside'] else 'outside'}) | "
                 f"{f(s['asym_vs_axons_ff']['spearman'])} ({'inside' if s['asym_vs_axons_ff']['inside'] else 'outside'}) |")
    L += ["", f"Excluded boundaries (fewer than two counted axons): "
          f"{p['boundary_units'] - p['boundary_units_scored']} of {p['boundary_units']}.", "",
          "## Reading", "",
          "* **P1 met, on all three conditions.** An axon's spikes follow its home compartment, which "
          "is how the twin wires a chip. The effect is far larger on the device than in the twin: the "
          f"median home index is {f(h['median_home_index'])} recorded against "
          f"{f(pred['home']['median'], 3)} simulated. The twin's compartments are more tightly "
          "coupled to each other, relative to their own axons, than these are.",
          "* **P2 not met.** The pre-registration read a result below the interval as: compartment "
          "timing runs against axonal traffic, which the twin has no mechanism for. On unstimulated "
          f"cultures the compartment that leads tends to be the one receiving more traffic "
          f"(Spearman {f(t['spearman'])}), though the permutation test does not exclude zero "
          f"(p = {f(t['permutation_p'])}). After stimulation the relation is near zero and inside "
          "the interval (secondary).",
          "* **What survives for chip design.** Neither the twin nor the device lets compartment "
          "timing stand in for the direction of axonal traffic: its relation with the measured "
          "direction is weak and, unstimulated, of the opposite sign. Direction has to be read "
          "from electrodes in the channels, which is the planner's advice. The twin's predicted "
          "relation fell outside its interval, and that is reported as a failed prediction.", "",
          "Known mismatches, fixed in the pre-registration and not tuned: a loop of four "
          "compartments read one boundary at a time as two-chamber chips; tunnels 400 um against "
          "500 um; cortical against hippocampal cultures; detection pipelines differ."]
    return "\n".join(L) + "\n"


def main() -> None:
    r = json.loads((V4 / "results.json").read_text())
    pred = json.loads((V4 / "brewer_prediction.json").read_text())
    figure(r, pred)
    (V4 / "RESULTS.md").write_text(results_md(r, pred), encoding="utf-8")
    print("wrote", V4 / "RESULTS.md")


if __name__ == "__main__":
    main()
