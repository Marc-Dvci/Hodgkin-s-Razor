"""Figures for the version 3 results, the chip sample size and the report.

    python scripts/figures_v3.py

Reads results/v3/results.json and results/chip_power.json; writes
results/v3/figures/*.png in the same style as scripts/figures_v2.py.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import figures_v2 as FV   # noqa: E402  (applies the shared matplotlib style)
from figures_v2 import INK, INK2, MUTED, NEUTRAL, S1, S2, S3, SURFACE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3" / "figures"
MATCHED = {"WT", "GluR1", "GluRAnull"}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def load(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def fig_primary(r: dict) -> None:
    A = r["A_blind"]
    preps = [p for p in A["early"]["preps"] if p["kind"] == "treated" or p["genotype"] in MATCHED]
    t = np.array([p["p_key"] for p in preps if p["kind"] == "treated"])
    z = np.array([p["p_key"] for p in preps if p["kind"] == "null"])
    rng = np.random.default_rng(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for x, v, c, name in ((0, z, NEUTRAL, "untreated sisters"), (1, t, S1, "APV sister")):
        a1.scatter(x + rng.uniform(-0.12, 0.12, v.size), v, s=34, color=c, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3)
        a1.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    a1.set_xticks([0, 1], [f"untreated, same genotypes\n(n = {z.size})",
                           f"chronic APV\n(n = {t.size})"])
    a1.set_xlim(-0.6, 1.6)
    a1.set_ylabel("p(g_nmda moved), per preparation")
    m = A["early"]["metrics"]
    a1.set_title(f"Blind: AUROC {m['auroc_key']:.2f} [{m['auroc_key_ci95'][0]:.2f}, "
                 f"{m['auroc_key_ci95'][1]:.2f}]", loc="left")
    a1.grid(axis="x", visible=False)

    B = r.get("B_comparators", {})
    rows = [("Hodgkin's Razor", m, S1)]
    if "unpaired" in B:
        rows.append(("same twin,\npairing removed", B["unpaired"]["metrics"], NEUTRAL))
    if "prior_art" in B:
        rows.append(("Doorn et al. 2025\nestimator", B["prior_art"]["metrics"], NEUTRAL))
    y = np.arange(len(rows))[::-1]
    for yi, (name, mm, c) in zip(y, rows):
        lo, hi = mm["auroc_key_ci95"]
        a2.barh(yi, mm["auroc_key"], height=0.5, color=c, edgecolor=SURFACE, linewidth=2)
        a2.plot([lo, hi], [yi, yi], color=INK, lw=1.6)
        a2.text(hi + 0.015, yi, f"{mm['auroc_key']:.2f}", va="center", color=INK, fontsize=9)
    a2.set_yticks(y, [n for n, _, _ in rows])
    a2.axvline(0.5, color=MUTED, lw=1, ls=":")
    a2.axvline(0.70, color=S2, lw=1.2, ls="--")
    a2.text(0.705, y.max() + 0.42, "pre-registered bar 0.70", color=INK2, fontsize=8, va="bottom")
    a2.text(0.505, -0.55, "chance", color=MUTED, fontsize=8)
    a2.set_xlim(0.3, 1.05)
    a2.set_ylim(-0.7, y.max() + 0.8)
    a2.set_xlabel("AUROC, treated vs untreated preparations (95% CI)")
    a2.set_title("Same preparations, three methods", loc="left")
    a2.grid(axis="y", visible=False)
    save(fig, "charlesworth_primary.png")


def fig_canalization(r: dict) -> None:
    A = r["A_blind"]
    early = {p["prep"]: p["p_key"] for p in A["early"]["preps"] if p["kind"] == "treated"}
    late = {p["prep"]: p["p_key"] for p in A["late"]["preps"] if p["kind"] == "treated"}
    both = sorted(set(early) & set(late))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    for k in both:
        a1.plot([0, 1], [early[k], late[k]], color=S1, alpha=0.45, lw=1.2)
    a1.plot([0, 1], [np.median([early[k] for k in both]), np.median([late[k] for k in both])],
            color=INK, lw=2.6, marker="o", ms=7, zorder=4)
    a1.set_xticks([0, 1], ["10–14 days", "15 days and after"])
    a1.set_xlim(-0.3, 1.3)
    a1.set_ylabel("p(g_nmda moved), treated preparation")
    c = A["canalization"]
    a1.set_title(f"Same {len(both)} treated preparations: p = {c['wilcoxon_p_early_gt_late']:.0e}",
                 loc="left")
    a1.grid(axis="x", visible=False)
    vals = [A["early"]["metrics"]["auroc_key"], A["late"]["metrics"]["auroc_key"]]
    cis = [A["early"]["metrics"]["auroc_key_ci95"], A["late"]["metrics"]["auroc_key_ci95"]]
    for x, v, ci, col in zip((0, 1), vals, cis, (S1, NEUTRAL)):
        a2.bar(x, v, width=0.5, color=col, edgecolor=SURFACE, linewidth=2)
        a2.plot([x, x], ci, color=INK, lw=1.6)
        a2.text(x, ci[1] + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    a2.axhline(0.5, color=MUTED, lw=1, ls=":")
    a2.set_xticks([0, 1], ["10–14 days", "15 days and after"])
    a2.set_ylim(0.2, 1.08)
    a2.set_ylabel("AUROC, treated vs untreated")
    a2.set_title("The contrast fades as the cultures compensate", loc="left")
    a2.grid(axis="x", visible=False)
    save(fig, "canalization.png")


def fig_profile(r: dict) -> None:
    prof = r["A_blind"]["profile_exploratory"]
    items = sorted(prof.items(), key=lambda kv: kv[1]["auroc"])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (k, v) in enumerate(items):
        ax.barh(i, v["auroc"] - 0.5, left=0.5, height=0.6,
                color=S1 if k == "g_nmda" else NEUTRAL, edgecolor=SURFACE, linewidth=2)
        ax.text(max(v["auroc"], 0.5) + 0.012, i, f"{v['auroc']:.2f}", va="center",
                color=INK if k == "g_nmda" else INK2, fontsize=8)
    ax.set_yticks(range(len(items)), [k for k, _ in items])
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("AUROC of each mechanism's presence, treated vs untreated")
    ax.set_title("Exploratory: the answer key (g_nmda) separates best", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "mechanism_profile.png")


def fig_chip_power(cp: dict) -> None:
    ch = cp["readouts"]["channel_dominant_share"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    tw = ch["twin"]["curve"]
    ob = ch["observed"]["curve"]
    ax.plot([c["chips_per_design"] for c in tw], [c["power_ci_clears_0.5"] for c in tw],
            color=S1, lw=2, marker="o", ms=5)
    ax.plot([c["chips_per_design"] for c in ob], [c["power_ci_clears_0.5"] for c in ob],
            color=S2, lw=2, marker="s", ms=5)
    ax.axhline(0.8, color=MUTED, lw=1, ls="--")
    ax.axvline(8, color=INK2, lw=1, ls=":")
    ax.text(8.6, 0.05, "recorded test\n(about 8 per design)", color=INK2, fontsize=8)
    ax.text(tw[-1]["chips_per_design"], tw[-1]["power_ci_clears_0.5"] + 0.03,
            f"if the twin is right (AUROC {ch['simulated_auroc']:.2f})", color=INK, fontsize=8, ha="right")
    ax.text(36, 0.56, "at the recorded\nseparation (0.62)", color=INK, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 20, 50, 100, 200], ["4", "8", "20", "50", "100", "200"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("chips per design (log scale)")
    ax.set_ylabel("power: 95% CI of AUROC clears 0.5")
    ax.set_title(f"How many chips a directionality claim needs: "
                 f"{ch['twin']['chips_needed']} per design", loc="left")
    save(fig, "chip_power.png")


def main() -> None:
    r = load("results/v3/results.json")
    if r and "A_blind" in r:
        fig_primary(r)
        fig_canalization(r)
        fig_profile(r)
    cp = load("results/chip_power.json")
    if cp:
        fig_chip_power(cp)


if __name__ == "__main__":
    main()
