"""Every figure of the version 2 report, from the saved results.

    python scripts/figures_v2.py [--raster]

Reads results/v2/results.json, results/chip_study.json and
results/v2/pharmacology.json, and writes results/v2/figures/*.png. `--raster`
also draws a recorded pair beside the twin re-simulated at its posterior, which
needs a CUDA device.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"
OUT = V2 / "figures"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]

# Reference palette (dataviz skill): categorical slots in fixed order, a
# one-hue sequential ramp, recessive ink and grid.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, NEUTRAL = "#e7e6e2", "#fcfcfb", "#c9c8c2"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def bars(ax, x, h, color, label=None, width=0.36):
    ax.bar(x, h, width=width, color=color, label=label, edgecolor=SURFACE, linewidth=2)


def fig_doorn(r: dict) -> None:
    a = r.get("A_doorn")
    if not a:
        return
    wells = sorted(a["wells"], key=lambda w: w["well"])
    M = np.array([w["p_active"] for w in wells])
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(wells) + 1.8))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SHIFT_KEYS)), SHIFT_KEYS, rotation=40, ha="right")
    ax.set_yticks(range(len(wells)), [w["well"] for w in wells])
    for j in (SHIFT_KEYS.index("u_rel"), SHIFT_KEYS.index("tau_d")):
        ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(wells), fill=False,
                                   edgecolor=S2, linewidth=2))
    for i, w in enumerate(wells):
        j = SHIFT_KEYS.index(w["top1"])
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if M[i, j] > .55 else INK)
    ax.grid(False)
    m = a["metrics"]
    ax.set_title(f"Blind test, Dynasore (Doorn et al.): named {m['top1_hits']} of "
                 f"{m['n_wells']} wells; boxed columns are the answer key")
    ax.set_xlabel("presence probability per mechanism (top call labelled)")
    fig.colorbar(im, ax=ax, shrink=.8, label="probability the mechanism moved")
    save(fig, "doorn_wells.png")


def fig_tampere_confusion(r: dict) -> None:
    c = r.get("C_tampere")
    if not c:
        return
    m = c["metrics_v1_key"]
    conf = np.array(m["confusion"], dtype=float)
    keep = conf.sum(1) > 0
    labels = [l for l, k in zip(m["confusion_labels"], keep) if k]
    sub = conf[keep]
    fr = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.4, 1.6 + .45 * len(labels)))
    im = ax.imshow(fr, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SHIFT_KEYS)), SHIFT_KEYS, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if sub[i, j]:
                ax.text(j, i, int(sub[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if fr[i, j] > .55 else INK)
    ax.grid(False)
    ax.set_xlabel("mechanism named"); ax.set_ylabel("mechanism the compound acts on")
    ax.set_title(f"Tampere development set, {int(sub.sum())} wells")
    fig.colorbar(im, ax=ax, shrink=.8, label="fraction of wells")
    save(fig, "tampere_confusion.png")


def fig_transfer(r: dict) -> None:
    d = r.get("D_transfer")
    if not d:
        return
    dirs = sorted(d["raw"])
    names = {"rat_to_other": "trained on rat,\nscored on human",
             "hPSC_to_other": "trained on human,\nscored on rat"}
    x = np.arange(len(dirs))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars(ax, x - .19, [d["raw"][k]["accuracy"] for k in dirs], S2, "raw feature change")
    bars(ax, x + .19, [d["twin"][k]["accuracy"] for k in dirs], S1, "twin's reading")
    ch = d["raw"][dirs[0]]["chance"]
    ax.axhline(ch, color=MUTED, lw=1, ls="--")
    ax.text(x[-1] + .5, ch, "chance", color=INK2, va="bottom", ha="right", fontsize=8)
    ax.set_xticks(x, [names.get(k, k) for k in dirs])
    ax.set_ylim(0, 1); ax.set_ylabel("compound named correctly")
    ax.set_title("Does a compound's fingerprint transfer across species?")
    ax.legend(loc="upper left")
    save(fig, "transfer.png")


def fig_simulation(r: dict) -> None:
    e = r.get("E_simulation")
    if not e:
        return
    views = list(e)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax = axes[0]
    x = np.arange(len(views))
    bars(ax, x - .19, [e[v]["recovery_top1"]["all_single_mechanism"]["top1"] for v in views], S2,
         "every single-mechanism pair")
    bars(ax, x + .19, [e[v]["recovery_top1"]["saturating"]["top1"] for v in views], S1,
         "saturating, large change")
    ax.axhline(e[views[0]]["chance"], color=MUTED, lw=1, ls="--")
    ax.set_xticks(x, views); ax.set_ylim(0, 1); ax.set_ylabel("top-1 on held-out simulations")
    ax.set_title("Exact mechanism, simulations"); ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    for v, col in zip(views, (S1, S3)):
        cov = [e[v]["coverage"][k]["0.9"] for k in SHIFT_KEYS]
        ax.plot(range(len(SHIFT_KEYS)), cov, "o-", color=col, lw=2, ms=5, label=v)
    ax.axhline(.9, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(SHIFT_KEYS)), SHIFT_KEYS, rotation=40, ha="right")
    ax.set_ylim(.5, 1.0); ax.set_ylabel("90% interval coverage")
    ax.set_title("Calibration of the shift"); ax.legend(loc="lower left")
    save(fig, "simulation.png")


def fig_guard(r: dict) -> None:
    g = r.get("E_guard")
    if not g:
        return
    cases = ["bank_holdout", "variant_kinetics", "shuffled_real"]
    labels = ["held-out\nsimulations\n(must pass)", "unmodelled\nkinetics\n(must fire)",
              "structure\ndestroyed\n(must fire)"]
    views = list(g)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(cases))
    for k, (v, col) in enumerate(zip(views, (S1, S3))):
        bars(ax, x + (k - .5) * .38, [g[v].get(c, {}).get("fire_rate", np.nan) for c in cases],
             col, v, width=.36)
    ax.plot([-.45, .45], [.1, .1], color=S2, lw=2)
    ax.plot([.55, 2.45], [.8, .8], color=S2, lw=2)
    ax.text(2.45, .82, "pre-registered bar", color=INK2, fontsize=7.5, ha="right")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_ylabel("fraction the guard fired on")
    ax.set_title("The guard"); ax.legend(loc="upper left")
    save(fig, "guard.png")


def fig_chip(chip: dict | None, r: dict) -> None:
    if chip:
        names = list(chip["readouts"])
        keys = [k for k in chip["readouts"][names[0]] if k not in ("epochs", "val_nll", "n_test")]
        M = np.array([[max(chip["readouts"][n][k]["pearson_r"], 0) for k in keys] for n in names])
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(keys)), keys); ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(keys)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > .55 else INK)
        ax.grid(False)
        ax.set_title("What each chip readout recovers (simulated, held out)")
        fig.colorbar(im, ax=ax, shrink=.85, label="recovery r")
        save(fig, "chip_readouts.png")
    b = r.get("B_chips")
    if b:
        order = ["control", "tesla", "tesla_v2", "arrows", "rams"]
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
        for ax, stat, title in ((axes[0], "dominant_share", "Channel electrodes"),
                                (axes[1], "chamber_asymmetry", "Chamber electrodes")):
            for i, d in enumerate(order):
                v = [row[stat] for row in b["rows"] if row["design"] == d
                     and np.isfinite(row[stat])
                     and (stat != "dominant_share" or row["events"] >= 20)]
                col = S1 if d in ("rams", "arrows") else (S2 if d == "control" else MUTED)
                jit = np.random.default_rng(i).uniform(-.12, .12, len(v))
                ax.scatter(np.full(len(v), i) + jit, v, s=26, color=col,
                           edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.set_xticks(range(len(order)), order)
            s = b[stat]
            ax.set_title(f"{title}: AUROC {s['auroc']:.2f}")
        axes[0].set_ylabel("share of propagation, dominant way")
        axes[1].set_ylabel("|cross-correlation asymmetry|")
        save(fig, "mateus_chips.png")


def fig_pharmacology() -> None:
    p = V2 / "pharmacology.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, axes = plt.subplots(1, len(d["views"]), figsize=(4.2 * len(d["views"]), 3.0),
                             squeeze=False)
    for ax, (view, v) in zip(axes[0], d["views"].items()):
        rows = [c for c in v["checks"] if "expected_rate_lo" in c]
        y = np.arange(len(rows))
        for i, c in enumerate(rows):
            lo, hi = c["expected_rate_lo"], min(c["expected_rate_hi"], 3.0)
            ax.plot([lo, hi], [i, i], color=GRID, lw=8, solid_capstyle="round", zorder=1)
            col = S1 if c["passes"] else S2
            ax.plot([c["rate_ratio_p25"], c["rate_ratio_p75"]], [i, i], color=col, lw=2, zorder=2)
            ax.scatter([c["rate_ratio_median"]], [i], color=col, s=30, zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
        ax.set_yticks(y, [c["drug"] for c in rows]); ax.set_xlim(0, 3)
        ax.set_xlabel("firing rate after / before (grey band: published range)")
        ax.set_title(f"{view}: {len(rows) - v['n_failed']} of {len(rows)} pass")
    save(fig, "pharmacology.png")


def fig_domain() -> None:
    """Recorded baselines against the simulated cultures the bank admitted."""
    import glob
    sys.path.insert(0, str(ROOT / "scripts"))
    from hodgkins_razor import simulator as S
    keys = ["mfr", "nbr", "psib", "sttc_mean"]
    try:
        from hodgkins_razor import doorn as D, tampere as T
        real = {"grid16": np.array([F.compute(p.baseline, 16, 60.0) for p in T.load_all()]),
                "grid12": np.array([F.compute(p.baseline, 12, 60.0) for p in D.load()])}
    except Exception:
        return
    sim = {v: [] for v in real}
    for f in sorted(glob.glob(str(ROOT / "data" / "bank_v2" / "shard_*.npz")))[:20]:
        d = np.load(f)
        for vi, v in enumerate(S.VIEWS):
            m = d["domain"] == vi
            sim[v].append(d["x_base"][m][:, vi])
    fig, axes = plt.subplots(2, len(keys), figsize=(9.6, 4.4))
    for i, v in enumerate(real):
        s = np.concatenate(sim[v]) if sim[v] else np.zeros((0, F.N_FEATURE))
        for j, k in enumerate(keys):
            ax = axes[i, j]
            c = F.NAMES.index(k)
            a, b = s[:, c], real[v][:, c]
            lo, hi = np.nanpercentile(np.r_[a, b], [1, 99])
            bins = np.linspace(lo, hi, 30)
            ax.hist(a, bins=bins, density=True, color=S1, alpha=.55, label="simulated, admitted")
            ax.hist(b, bins=bins, density=True, histtype="step", color=S2, lw=2, label="recorded")
            ax.set_yticks([])
            ax.set_title(f"{v} · {k}", fontsize=8.5)
    axes[0, 0].legend(fontsize=7)
    save(fig, "domain.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    r = json.loads((V2 / "results.json").read_text()) if (V2 / "results.json").exists() else {}
    chip_path = ROOT / "results" / "chip_study.json"
    chip = json.loads(chip_path.read_text()) if chip_path.exists() else None
    fig_doorn(r)
    fig_tampere_confusion(r)
    fig_transfer(r)
    fig_simulation(r)
    fig_guard(r)
    fig_chip(chip, r)
    fig_pharmacology()
    fig_domain()


if __name__ == "__main__":
    main()
