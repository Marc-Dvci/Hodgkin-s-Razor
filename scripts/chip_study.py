"""Which chip readout resolves which property, answered before the experiment.

    python scripts/chip_study.py --chips 30000

Simulates two-compartment chips over their prior and reads each one four
ways (see `hodgkins_razor.chip`): compartment electrodes, a 2 Hz calcium
movie, channel electrodes, and compartment electrodes across a three-step
perfusion (untreated, source silenced, target silenced). For each readout a
posterior over the four chip parameters is fitted on simulations and scored on
held-out ones: recovery, 90 percent interval coverage, and how much narrower
than the prior the interval is, which is the expected information that readout
buys about each parameter.

It also writes the twin's prediction for the statistics
`scripts/mateus_check.py` computes on recorded chips, and a qualitative check
against Lassus et al. (2018): an NMDA reduction applied to both chambers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import torch
import zuko

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from hodgkins_razor import chip as C, features as F, params as P
from hodgkins_razor import simulator as S
from fit_domain import GaussianProposal

ROOT = pathlib.Path(__file__).resolve().parents[1]


def chip_to_z(x: np.ndarray) -> np.ndarray:
    lo = np.where(C.CHIP_LOG, np.log(C.CHIP_LO), C.CHIP_LO)
    hi = np.where(C.CHIP_LOG, np.log(C.CHIP_HI), C.CHIP_HI)
    u = np.where(C.CHIP_LOG, np.log(np.clip(x, 1e-9, None)), x)
    return 2.0 * (u - lo) / (hi - lo) - 1.0


def _views(args):
    base, src_off, tgt_off, duration, seed = args
    comp = C.compartment_view(base, duration)
    ca = C.calcium_features(base, duration, rng=np.random.default_rng(seed))
    chan = C.channel_view(base, duration)
    perf = C.perfusion_view(base, src_off, tgt_off, duration)
    return comp, ca, chan, perf


def cultures(n: int, rng: np.random.Generator) -> np.ndarray:
    """Living cultures from the grid16 domain proposal of the main bank."""
    dom = json.loads((ROOT / "models" / "domain.json").read_text())
    return GaussianProposal.load(dom["views"]["grid16"]["proposal"]).draw(n, rng)


def build_bank(args) -> dict:
    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    pool = cf.ProcessPoolExecutor(max_workers=args.workers)
    out = {k: [] for k in ("chip", "theta", "compartment", "calcium_2hz",
                           "channel", "perfusion")}
    done = tried = 0
    t0 = time.time()
    step = 0
    while done < args.chips:
        theta = cultures(args.batch, rng)
        chips = C.sample_chip_prior(args.batch, rng)
        seed = int(args.seed + 7919 * step)
        runs = [C.run_chip(sim, theta, chips, duration_s=args.duration,
                           transient_s=args.transient, seed=seed,
                           silence=np.full(args.batch, s))
                for s in (-1, 0, 1)]
        jobs = [(runs[0].as_events(k), runs[1].as_events(k), runs[2].as_events(k),
                 args.duration, seed + k) for k in range(args.batch)]
        res = list(pool.map(_views, jobs, chunksize=4))
        for k, (comp, ca, chan, perf) in enumerate(res):
            src = C.chamber_events(jobs[k][0], 0).shape[0]
            if src < 30 or runs[0].truncated[k]:
                continue
            out["chip"].append(chips[k]); out["theta"].append(theta[k])
            out["compartment"].append(comp); out["calcium_2hz"].append(ca)
            out["channel"].append(chan); out["perfusion"].append(perf)
            done += 1
        tried += args.batch
        step += 1
        print(f"  {done}/{args.chips} usable of {tried} ({time.time() - t0:.0f}s)",
              flush=True)
    pool.shutdown()
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_readout(x: np.ndarray, y: np.ndarray, seed: int, max_epochs: int = 300,
                patience: int = 12) -> dict:
    """Posterior over chip parameters from one readout, scored on held-out chips.

    Train, validation and test chips are disjoint; the flow keeps the state at
    its best validation likelihood, so the reported coverage is that of a model
    that was not fitted to the chips it is scored on.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_te, n_va = int(0.15 * n), int(0.15 * n)
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    xt = np.nan_to_num(signed_log(x))
    mu, sd = xt[tr].mean(0), xt[tr].std(0) + 1e-6
    X = torch.as_tensor((xt - mu) / sd, dtype=torch.float32)
    Y = torch.as_tensor(chip_to_z(y), dtype=torch.float32)
    flow = zuko.flows.MAF(features=C.N_CHIP, context=X.shape[1], transforms=5,
                          hidden_features=[256, 256]).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        flow.train()
        order = rng.permutation(tr)
        for a in range(0, len(order), 512):
            sel = torch.as_tensor(order[a:a + 512])
            opt.zero_grad(set_to_none=True)
            loss = -flow(X[sel].to(dev)).log_prob(Y[sel].to(dev)).mean()
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            v = float(-flow(X[torch.as_tensor(va)].to(dev))
                      .log_prob(Y[torch.as_tensor(va)].to(dev)).mean())
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in flow.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    flow.load_state_dict(best_state)
    flow.eval()
    with torch.no_grad():
        samples = flow(X[torch.as_tensor(te)].to(dev)).sample((800,)).cpu().numpy()
    med = np.median(samples, axis=0)
    lo, hi = np.quantile(samples, 0.05, axis=0), np.quantile(samples, 0.95, axis=0)
    truth = chip_to_z(y[te])
    out = {"epochs": ep + 1, "val_nll": best, "n_test": int(n_te)}
    for j, key in enumerate(C.CHIP_KEYS):
        width = float(np.mean(hi[:, j] - lo[:, j]))
        out[key] = {"pearson_r": float(np.corrcoef(truth[:, j], med[:, j])[0, 1]),
                    "interval_width": width, "prior_width": 1.8,
                    "information": float(max(0.0, 1.0 - width / 1.8)),
                    "coverage_90": float(np.mean((truth[:, j] >= lo[:, j])
                                                 & (truth[:, j] <= hi[:, j])))}
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


SYMMETRIC_SEEDING = 0.7   # target autonomy at or above: both chambers self-active


def prediction_for_recorded_chips(bank: dict, symmetric: bool = True) -> dict:
    """What the twin expects the two recorded-chip statistics to show.

    Strong diodes (direction_sel above 0.85) against symmetric channels (below
    0.6), scored with the same unsigned statistics `mateus_check.py` computes:
    the dominant share of channel propagation, and the asymmetry of the
    chambers' rate cross-correlation. Mateus et al. seed both chambers with the
    same neurons at the same density, so the prediction for their chips is made
    on simulated chips whose two chambers are both self-active; a weakly active
    target makes forward events dominate whatever the channel does.
    """
    if symmetric:
        keep = bank["chip"][:, 3] >= SYMMETRIC_SEEDING
        bank = {k: v[keep] for k, v in bank.items()}
    d = bank["chip"][:, 1]
    strong, sym = d > 0.85, d < 0.6
    keep = strong | sym
    chan_share = np.maximum(bank["channel"][:, 0], 1 - bank["channel"][:, 0])
    comp_asym = np.abs(bank["compartment"][:, 2 * F.N_FEATURE + 2])
    return {"n_strong": int(strong.sum()), "n_symmetric": int(sym.sum()),
            "channel_dominant_share_auroc": auroc(chan_share[keep], strong[keep]),
            "compartment_asymmetry_auroc": auroc(comp_asym[keep], strong[keep])}


def lassus_check(args, n: int = 192) -> dict:
    """An NMDA reduction on both chambers, read at the target chamber.

    Lassus et al. report that blocking GluN2B-containing NMDA receptors lowers
    striatal (target) calcium event frequency and synchrony. The twin has no
    subunit resolution, so the check is a partial NMDA reduction to 30 percent,
    on chips whose target is driven by the source (low autonomy), read with
    the 2 Hz calcium readout they used.
    """
    sim = S.Simulator()
    rng = np.random.default_rng(args.seed + 99)
    theta = cultures(n, rng)
    chips = C.sample_chip_prior(n, rng)
    chips[:, 3] = rng.uniform(0.05, 0.3, n)
    chips[:, 1] = rng.uniform(0.85, 1.0, n)
    treated = theta.copy()
    treated[:, P.index("g_nmda")] *= 0.3
    seed = args.seed + 4242
    r0 = C.run_chip(sim, theta, chips, duration_s=args.duration,
                    transient_s=args.transient, seed=seed)
    r1 = C.run_chip(sim, treated, chips, duration_s=args.duration,
                    transient_s=args.transient, seed=seed)
    freq_down = sync_down = used = 0
    for k in range(n):
        a = C.calcium_features(r0.as_events(k), args.duration, rng=np.random.default_rng(k))
        b = C.calcium_features(r1.as_events(k), args.duration, rng=np.random.default_rng(k))
        if a[1] <= 0:
            continue
        used += 1
        freq_down += b[1] < a[1]
        sync_down += b[10] < a[10]
    return {"chips": used, "target_frequency_lower": float(freq_down / max(used, 1)),
            "target_synchrony_lower": float(sync_down / max(used, 1)),
            "published_direction": "both lower (Lassus et al. 2018, ifenprodil)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default="data/chip_bank_v2.npz")
    ap.add_argument("--out", default="results/chip_study.json")
    args = ap.parse_args()

    cache = ROOT / args.cache
    if cache.exists():
        d = np.load(cache)
        bank = {k: d[k] for k in d.files}
        print(f"loaded {bank['chip'].shape[0]} chips from {cache}")
    else:
        print(f"simulating {args.chips} chips, three perfusions each ...")
        bank = build_bank(args)
        np.savez_compressed(cache, **bank)

    readouts = {
        "compartment": bank["compartment"],
        "calcium_2hz": bank["calcium_2hz"],
        "channel": bank["channel"],
        "perfusion": np.concatenate([bank["compartment"], bank["perfusion"]], axis=1),
        "compartment+channel": np.concatenate([bank["compartment"], bank["channel"]], axis=1),
    }
    res = {"n_chips": int(bank["chip"].shape[0]), "duration_s": args.duration,
           "calcium_frame_hz": C.CA_FRAME_HZ, "readouts": {}}
    for i, (name, x) in enumerate(readouts.items()):
        print(f"fitting {name} ({x.shape[1]} statistics) ...", flush=True)
        res["readouts"][name] = fit_readout(x, bank["chip"], seed=10 + i)
    res["prediction_for_recorded_chips"] = prediction_for_recorded_chips(bank)
    res["prediction_all_chips"] = prediction_for_recorded_chips(bank, symmetric=False)
    res["prediction_condition"] = (f"target autonomy >= {SYMMETRIC_SEEDING} (both chambers "
                                   "seeded alike, as in Mateus et al.)")
    print("NMDA check against Lassus et al. ...", flush=True)
    res["lassus_nmda"] = lassus_check(args)
    (ROOT / args.out).write_text(json.dumps(res, indent=1))

    print()
    head = "".join(f"{k:>22s}" for k in readouts)
    print(f"{'recovery r':16s}{head}")
    for key in C.CHIP_KEYS:
        print(f"{key:16s}" + "".join(f"{res['readouts'][k][key]['pearson_r']:22.3f}"
                                     for k in readouts))
    print(f"{'coverage 90':16s}")
    for key in C.CHIP_KEYS:
        print(f"{key:16s}" + "".join(f"{res['readouts'][k][key]['coverage_90']:22.2f}"
                                     for k in readouts))
    print(json.dumps(res["prediction_for_recorded_chips"]))
    print(json.dumps(res["lassus_nmda"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
