"""Every figure in the technical report, from the saved results.

    python scripts/figures.py

Reads results/results.json and writes results/figures/*.png. Figures that need
a simulation are skipped when no CUDA device is present.
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
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hodgkins_razor import features as F, params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]

INK = "#1b1f27"
DIM = "#6b7684"
ACCENT = "#2f6fdb"
WARM = "#d1662b"
GOOD = "#2f8f5b"
BAD = "#c0392b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": "#cbd2da", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": DIM, "ytick.color": DIM, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#e8ecf1", "grid.linewidth": .8,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_confusion(r: dict) -> None:
    m = np.array(r["mechanism"]["confusion"], dtype=float)
    keep = m.sum(axis=1) > 0
    labels = [l for l, k in zip(r["mechanism"]["confusion_labels"], keep) if k]
    sub = m[keep]
    frac = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 2.6 + 0.42 * len(labels)))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SHIFT_KEYS)), SHIFT_KEYS, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("mechanism the twin named")
    ax.set_ylabel("mechanism the compound acts on")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if sub[i, j]:
                ax.text(j, i, f"{int(sub[i, j])}", ha="center", va="center",
                        color="white" if frac[i, j] > .5 else INK, fontsize=8)
    ax.set_title(f"Mechanism recovery, {int(sub.sum())} wells")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label="fraction of wells")
    save(fig, "confusion.png")


def fig_per_compound(r: dict) -> None:
    per = r["mechanism"]["per_compound"]
    names = sorted(per, key=lambda k: -per[k]["top1"] / max(per[k]["n"], 1))
    acc = [per[k]["top1"] / max(per[k]["n"], 1) for k in names]
    n = [per[k]["n"] for k in names]
    fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y, acc, color=[GOOD if a >= .5 else WARM for a in acc], height=.6)
    ax.axvline(r["mechanism"]["chance"], color=BAD, ls="--", lw=1,
               label=f"chance {r['mechanism']['chance']:.2f}")
    ax.set_yticks(y, [f"{k}  (n={c})" for k, c in zip(names, n)])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("top-1 accuracy, wells")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Per compound")
    save(fig, "per_compound.png")


def fig_coverage(r: dict) -> None:
    cov = r["calibration"]["coverage"]
    keys = [k for k in cov if k in SHIFT_KEYS]
    levels = ["50", "80", "90"]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(keys))
    for i, lv in enumerate(levels):
        vals = [cov[k][lv] for k in keys]
        ax.bar(x + (i - 1) * .27, vals, width=.26,
               color=[ACCENT, WARM, GOOD][i], label=f"{lv}% interval")
        ax.axhline(int(lv) / 100, color=[ACCENT, WARM, GOOD][i], lw=.8, ls=":")
    ax.set_xticks(x, keys, rotation=40, ha="right")
    ax.set_ylabel("observed coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    ax.set_title("Credible-interval coverage on held-out simulations")
    save(fig, "coverage.png")


def fig_reliability(r: dict) -> None:
    pres = r["presence"]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=DIM, ls="--", lw=1)
    for k, d in pres.items():
        if not d["bins"]:
            continue
        ax.plot([b["p_mean"] for b in d["bins"]],
                [b["observed"] for b in d["bins"]], marker="o", ms=3, lw=1.2,
                label=f"{k} ({d['auroc']:.2f})")
    ax.set_xlabel("stated probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Presence probability, after calibration")
    ax.legend(frameon=False, fontsize=7, title="AUROC", title_fontsize=7)
    save(fig, "reliability.png")


def fig_guard(r: dict) -> None:
    if "guard" not in r:
        return
    g = r["guard"]
    cal = np.array(r["guard_calibration"]["values"])
    thr = g["threshold"]
    names = [("bank_holdout", "held-out simulations\n(must pass)"),
             ("real_pairs", "real recordings\n(reported)"),
             ("variant_kinetics", "changed receptor kinetics\n(must fire)"),
             ("shuffled_real", "structure destroyed\n(must fire)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    vals = [g[k]["fire_rate"] for k, _ in names]
    cols = [GOOD, ACCENT, BAD, BAD]
    ax.bar(range(4), vals, color=cols, width=.6)
    for i, (k, _) in enumerate(names):
        ax.text(i, vals[i] + .03, f"{vals[i]:.2f}\nn={g[k]['n']}", ha="center",
                fontsize=8, color=INK)
    ax.set_xticks(range(4), [n for _, n in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction called outside the model")
    ax.set_title(f"The guard, threshold {thr:.1f} "
                 f"(95th percentile of {len(cal)} held-out records)")
    save(fig, "guard.png")


def fig_effects(r: dict) -> None:
    """Recovered effect size per compound on the mechanism it acts on."""
    wells = r["real"]["wells"]
    key = {"CNQX": "g_ampa", "D-AP5": "g_nmda", "GABA": "g_gaba",
           "Gabazine": "g_gaba", "Kainic acid": "g_ampa", "TTX": "g_na"}
    groups: dict[str, list[float]] = {}
    for w in wells:
        k = key.get(w["compound"])
        if not k:
            continue
        groups.setdefault(w["compound"], []).append(
            w["delta_med"][SHIFT_KEYS.index(k)])
    if not groups:
        return
    names = sorted(groups)
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(names) + 1.6))
    for i, n in enumerate(names):
        v = np.array(groups[n])
        ax.scatter(np.exp(v), [i] * len(v), s=26, alpha=.75,
                   color=ACCENT if np.median(v) < 0 else WARM,
                   edgecolor="white", linewidth=.5, zorder=3)
        ax.scatter([np.exp(np.median(v))], [i], marker="|", s=420, color=INK, zorder=4)
    ax.axvline(1.0, color=DIM, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)),
                  [f"{n}\n{key[n]}" for n in names], fontsize=8)
    ax.set_xlabel("recovered fold change on the expected mechanism")
    ax.invert_yaxis()
    ax.set_title("Effect size per well, on the target from the answer key")
    save(fig, "effects.png")


def fig_class(r: dict) -> None:
    cm = r.get("mechanism_class")
    if not cm or not cm.get("n"):
        return
    conf = np.array(cm["confusion"], dtype=float)
    labels = [l.replace(" and ", "\nand ").replace(" transmission", "\ntransmission")
              for l in cm["labels"]]
    frac = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            if conf[i, j]:
                ax.text(j, i, f"{int(conf[i, j])}", ha="center", va="center",
                        color="white" if frac[i, j] > .5 else INK, fontsize=9)
    ax.set_xlabel("class the twin named")
    ax.set_ylabel("class the compound acts on")
    ax.set_title(f"Mechanism class, {cm['top1_accuracy']:.2f} of {cm['n']} wells "
                 f"(chance {cm['chance']:.2f})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.75, label="fraction of wells")
    save(fig, "class_confusion.png")


def fig_raster(args) -> None:
    """Measured recording beside the twin simulated at the fitted parameters."""
    from hodgkins_razor import nde, ppc, simulator as S
    if not S.available():
        print("skipping raster figure: no CUDA device")
        return
    ex = ROOT / "data" / "examples" / f"{args.raster}.json"
    if not ex.exists():
        print("skipping raster figure: no example", ex)
        return
    d = json.loads(ex.read_text())
    base = np.array(d["baseline"]).reshape(-1, 2)
    treat = np.array(d["treated"]).reshape(-1, 2)
    dur = float(d["duration"])
    twin = nde.Twin.load(ROOT / "models" / "twin", device="cuda")
    sim = S.Simulator()
    xb, xt = F.compute(base, 16, dur), F.compute(treat, 16, dur)
    post = twin.posterior(xb, xt)
    chk = ppc.check(twin, sim, xb, xt, np.inf, dur, 5.0, n_draws=8, post=post)
    res = chk["result"]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 4.4), sharex=True, sharey=True)
    panels = [(base, "measured, baseline", ACCENT),
              (res.as_events(0), "twin, baseline", "#7aa7e8"),
              (treat, f"measured, {d['compound']}", WARM),
              (res.as_events(1), f"twin, {d['compound']}", "#e8ab7a")]
    for ax, (ev, title, col) in zip(axes.T.ravel(), panels):
        if len(ev):
            ax.scatter(ev[:, 1], ev[:, 0], s=.7, color=col, marker="|",
                       linewidths=.6)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-.5, 15.5)
        ax.set_xlim(0, dur)
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("electrode")
    fig.suptitle(d["label"], fontsize=10)
    save(fig, "raster.png")


def fig_chip() -> None:
    path = ROOT / "results" / "chip_study.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    keys = list(d["readouts"]["electrode_array"])
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(keys))
    for i, (name, col, lab) in enumerate([
            ("electrode_array", ACCENT, "electrode array"),
            ("calcium_2hz", WARM, f"calcium at {d['calcium_frame_hz']:.0f} Hz")]):
        vals = [max(d["readouts"][name][k]["pearson_r"], 0) for k in keys]
        ax.bar(x + (i - .5) * .34, vals, width=.32, color=col, label=lab)
    ax.set_xticks(x, keys, rotation=25, ha="right")
    ax.set_ylabel("recovery correlation")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("What each chip readout resolves")
    save(fig, "chip_readout.png")


def fig_sensitivity() -> None:
    path = ROOT / "results" / "sensitivity.json"
    if not path.exists():
        path = ROOT / "results" / "sensitivity_before.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    keys = list(d["parameters"])
    fig, axes = plt.subplots(2, 7, figsize=(13.5, 4.2), sharey=False)
    for ax, k in zip(axes.ravel(), keys):
        rec = d["parameters"][k]
        ax.plot(rec["grid"], rec["watch"]["mfr"]["median_curve"],
                marker="o", ms=3, color=ACCENT, lw=1.3)
        if rec["log"]:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(k, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Firing rate against each parameter, over its prior range", fontsize=10)
    save(fig, "sensitivity.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--raster", default="rat_cnqx")
    ap.add_argument("--skip-raster", action="store_true")
    args = ap.parse_args()

    fig_sensitivity()
    fig_chip()
    path = ROOT / args.results
    if path.exists():
        r = json.loads(path.read_text())
        fig_confusion(r)
        fig_per_compound(r)
        fig_coverage(r)
        fig_reliability(r)
        fig_guard(r)
        fig_effects(r)
        fig_class(r)
    else:
        print("no results.json yet; skipping the evaluation figures")
    if not args.skip_raster:
        fig_raster(args)


if __name__ == "__main__":
    main()
