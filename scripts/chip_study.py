"""What a two-compartment chip readout can resolve, and what it cannot.

    python scripts/chip_study.py --pairs 20000

Simulates directional chips over their parameter prior, then asks how well each
chip parameter can be recovered from two readouts of the same experiment: the
full electrode array, and a population calcium movie at 2 Hz, which is how such
chips are usually read. The gap between the two is the answer to "is my
readout good enough for the question I am asking".

Also scores the experiment-design question: given a posterior, which next
experiment reduces uncertainty most.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import torch
import zuko

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hodgkins_razor import chip as C, features as F, params as P
from hodgkins_razor import simulator as S
from fit_regime import is_living

ROOT = pathlib.Path(__file__).resolve().parents[1]


def chip_to_z(x: np.ndarray) -> np.ndarray:
    lo = np.where(C.CHIP_LOG, np.log(C.CHIP_LO), C.CHIP_LO)
    hi = np.where(C.CHIP_LOG, np.log(C.CHIP_HI), C.CHIP_HI)
    u = np.where(C.CHIP_LOG, np.log(np.clip(x, 1e-9, None)), x)
    return 2.0 * (u - lo) / (hi - lo) - 1.0


def z_to_chip(z: np.ndarray) -> np.ndarray:
    lo = np.where(C.CHIP_LOG, np.log(C.CHIP_LO), C.CHIP_LO)
    hi = np.where(C.CHIP_LOG, np.log(C.CHIP_HI), C.CHIP_HI)
    u = lo + (np.asarray(z) + 1.0) * 0.5 * (hi - lo)
    return np.where(C.CHIP_LOG, np.exp(u), u)


def electrode_view(events: np.ndarray, duration: float) -> np.ndarray:
    """Per-compartment electrode features, plus the cross-compartment lag."""
    geom = C.GEOMETRY
    out = []
    for comp in (0, 1):
        sel = (geom.elec_compartment[events[:, 0].astype(int)] == comp
               if events.size else np.zeros(0, bool))
        ev = events[sel].copy() if events.size else np.zeros((0, 2))
        if ev.size:
            ev[:, 0] = ev[:, 0] % 8
        out.append(F.compute(ev, 8, duration))
    extra = np.zeros(3)
    if events.size:
        a = events[geom.elec_compartment[events[:, 0].astype(int)] == 0][:, 1]
        b = events[geom.elec_compartment[events[:, 0].astype(int)] == 1][:, 1]
        if a.size > 5 and b.size > 5:
            bins = np.arange(0, duration, 0.005)
            ha, _ = np.histogram(a, bins)
            hb, _ = np.histogram(b, bins)
            ha = ha - ha.mean()
            hb = hb - hb.mean()
            n = 60   # +/- 300 ms
            cc = np.array([np.dot(ha[max(0, -l):len(ha) - max(0, l)],
                                  hb[max(0, l):len(hb) - max(0, -l)])
                           for l in range(-n, n + 1)], dtype=float)
            denom = np.sqrt(np.dot(ha, ha) * np.dot(hb, hb)) + 1e-9
            cc /= denom
            extra[0] = float(cc.max())
            extra[1] = float((np.argmax(cc) - n) * 0.005)
            extra[2] = float(np.log1p(b.size) - np.log1p(a.size))
    return np.concatenate([out[0], out[1], extra])


def build_bank(args) -> dict:
    """Simulate chips that are alive.

    Culture parameters are proposed by the same classifier the main bank uses,
    so the question being asked is what a readout resolves on a working device
    rather than on a dead one. A chip with a silent source chamber says nothing
    about its channels whatever the readout.
    """
    import joblib
    sim = S.Simulator()
    rng = np.random.default_rng(args.seed)
    model = joblib.load(ROOT / "models" / "regime.joblib")
    thresh = json.loads((ROOT / "models" / "regime.json").read_text())
    thresh = thresh["quantiles"]["0.90"]["threshold"]

    E, CA, TH = [], [], []
    done = tried = 0
    t0 = time.time()
    while done < args.pairs and time.time() - t0 < args.budget:
        cand = P.sample_prior(args.batch * 24, rng)
        score = model.predict_proba(P.to_unit(cand))[:, 1]
        theta = cand[score >= thresh][:args.batch]
        if theta.shape[0] < 8:
            continue
        chips = C.sample_chip_prior(theta.shape[0], rng)
        res = C.run_chip(sim, theta, chips, duration_s=args.duration,
                         transient_s=args.transient, seed=int(args.seed + tried))
        tried += theta.shape[0]
        for k in range(theta.shape[0]):
            ev = res.as_events(k)
            if ev.shape[0] < 60:
                continue
            geom = C.GEOMETRY
            comp = geom.elec_compartment[ev[:, 0].astype(int)]
            if (comp == 0).sum() < 30:
                continue
            E.append(electrode_view(ev, args.duration))
            CA.append(C.calcium_features(ev, args.duration,
                                         rng=np.random.default_rng(done + k)))
            TH.append(chips[k])
            done += 1
        print(f"  {done}/{args.pairs} usable, {tried} simulated "
              f"({done / max(tried, 1):.2f} kept, {time.time() - t0:.0f}s)",
              flush=True)
    return {"elec": np.array(E, dtype=np.float32),
            "ca": np.array(CA, dtype=np.float32),
            "chip": np.array(TH, dtype=np.float32)}


def fit_flow(x: np.ndarray, y: np.ndarray, epochs: int, seed: int = 0) -> dict:
    """Posterior over chip parameters from one readout."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = int(0.15 * n)
    te, tr = perm[:n_te], perm[n_te:]

    xt = np.log1p(np.clip(x, 0, None)) * np.sign(x) + np.where(x < 0, x, 0)
    mu, sd = xt[tr].mean(0), xt[tr].std(0) + 1e-6
    X = torch.as_tensor((xt - mu) / sd, dtype=torch.float32)
    Y = torch.as_tensor(chip_to_z(y), dtype=torch.float32)

    flow = zuko.flows.MAF(features=C.N_CHIP, context=X.shape[1], transforms=5,
                          hidden_features=[256, 256]).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
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
        ctx = X[torch.as_tensor(te)].to(dev)
        samples = flow(ctx).sample((600,)).cpu().numpy()
    med = np.median(samples, axis=0)
    lo, hi = np.quantile(samples, 0.05, axis=0), np.quantile(samples, 0.95, axis=0)
    truth = chip_to_z(y[te])

    out = {}
    for j, key in enumerate(C.CHIP_KEYS):
        r = float(np.corrcoef(truth[:, j], med[:, j])[0, 1])
        width = float(np.mean(hi[:, j] - lo[:, j]))
        cover = float(np.mean((truth[:, j] >= lo[:, j]) & (truth[:, j] <= hi[:, j])))
        # A prior-width interval means nothing was learned: the prior spans 2.
        out[key] = {"pearson_r": r, "interval_width": width,
                    "prior_width": 2.0 * 0.9,
                    "information": float(max(0.0, 1.0 - width / (2.0 * 0.9))),
                    "coverage_90": cover}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transient", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--budget", type=float, default=2400.0,
                    help="seconds to spend simulating chips")
    ap.add_argument("--cache", default="data/chip_bank.npz")
    ap.add_argument("--out", default="results/chip_study.json")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    if cache.exists():
        d = np.load(cache)
        bank = {k: d[k] for k in ("elec", "ca", "chip")}
        print(f"loaded {bank['chip'].shape[0]} chips from {cache}")
    else:
        print(f"simulating {args.pairs} chips ...")
        bank = build_bank(args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **bank)

    print("fitting the electrode-array readout ...")
    elec = fit_flow(bank["elec"], bank["chip"], args.epochs, seed=1)
    print("fitting the 2 Hz calcium readout ...")
    ca = fit_flow(bank["ca"], bank["chip"], args.epochs, seed=2)
    print("fitting both together ...")
    both = fit_flow(np.concatenate([bank["elec"], bank["ca"]], axis=1),
                    bank["chip"], args.epochs, seed=3)

    res = {"n_chips": int(bank["chip"].shape[0]),
           "duration_s": args.duration,
           "calcium_frame_hz": C.CA_FRAME_HZ,
           "readouts": {"electrode_array": elec, "calcium_2hz": ca,
                        "both": both}}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))

    print()
    print(f"{'chip parameter':16s} {'electrodes r':>13s} {'calcium r':>10s} "
          f"{'elec info':>10s} {'ca info':>8s}")
    for key in C.CHIP_KEYS:
        print(f"{key:16s} {elec[key]['pearson_r']:13.3f} {ca[key]['pearson_r']:10.3f} "
              f"{elec[key]['information']:10.2f} {ca[key]['information']:8.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
