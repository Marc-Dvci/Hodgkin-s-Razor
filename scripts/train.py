"""Train the flow and the presence head on the simulation bank.

    python scripts/train.py --bank data/bank --out models/twin

The bank is split by record so that calibration and the reported validation
numbers never touch a record used for fitting. Nothing from any recorded
experiment enters this script.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import zuko

from hodgkins_razor import features as F, nde, params as P


def load_bank(path: pathlib.Path) -> dict:
    shards = sorted(path.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {path}")
    keys = ("theta_c", "delta", "active", "x_base", "x_treat")
    acc = {k: [] for k in keys}
    for s in shards:
        d = np.load(s)
        for k in keys:
            acc[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"bank: {len(shards)} shards, {out['theta_c'].shape[0]} pairs")
    return out


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 12) -> dict:
    """Isotonic-style calibration map from a held-out split."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(float(p[a:b].mean()))
        ys.append(float(y[a:b].mean()))
    # Enforce monotonicity so the map cannot invert an ordering.
    ys = list(np.maximum.accumulate(ys))
    return {"x": [0.0] + xs + [1.0], "y": [0.0] + ys + [1.0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/bank")
    ap.add_argument("--out", default="models/twin")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--transforms", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--cal-frac", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reweight-domain", action="store_true",
                    help="weight the loss toward the recorded baseline "
                         "distribution instead of discarding what falls "
                         "outside it")
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--match-domain", action="store_true",
                    help="keep only simulated baselines inside the range the "
                         "recorded baselines span")
    ap.add_argument("--unpaired", action="store_true",
                    help="baseline: one flow over parameters from a single "
                         "recording, with no pairing and no shift prior")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bank = load_bank(pathlib.Path(args.bank))
    if args.match_domain:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from fit_regime import BASELINE_BOX, domain_mask
        keep = domain_mask(bank["x_base"])
        bank = {k: v[keep] for k, v in bank.items()}
        print(f"domain match: kept {keep.sum()} of {keep.size} pairs "
              f"({keep.mean():.1%}) inside {BASELINE_BOX}")
    n = bank["theta_c"].shape[0]
    rng = np.random.default_rng(args.seed)
    weights = np.ones(n)
    if args.reweight_domain:
        # A classifier separating recorded baselines from simulated ones gives
        # the density ratio, and weighting by it targets the recorded regime
        # while keeping every simulation. Only unlabelled baseline recordings
        # are used; no compound label enters this.
        from sklearn.ensemble import HistGradientBoostingClassifier
        from hodgkins_razor import tampere as T
        real = []
        for plate in ("rat", "human"):
            for pair in T.load_plate(plate, window_s=60.0, n_windows=3):
                real.append(F.compute(pair.baseline, 16, 60.0))
        real = nde.phi(np.array(real))
        sim_x = nde.phi(bank["x_base"])
        X = np.vstack([sim_x, real])
        y = np.r_[np.zeros(len(sim_x)), np.ones(len(real))]
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                             max_leaf_nodes=15).fit(X, y)
        pr = np.clip(clf.predict_proba(sim_x)[:, 1], 1e-4, 1 - 1e-4)
        ratio = pr / (1.0 - pr)
        weights = np.clip(ratio / np.median(ratio), 1.0 / args.weight_cap,
                          args.weight_cap)
        weights *= len(weights) / weights.sum()
        ess = weights.sum() ** 2 / np.square(weights).sum()
        print(f"domain reweighting: effective sample {ess:.0f} of {n} "
              f"({ess / n:.1%}), weight range "
              f"{weights.min():.3f} to {weights.max():.1f}")
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    n_cal = int(args.cal_frac * n)
    idx_val, idx_cal, idx_tr = perm[:n_val], perm[n_val:n_val + n_cal], perm[n_val + n_cal:]

    if args.unpaired:
        # Every recording becomes its own record: the baseline with theta_c and
        # the treated one with theta_c + delta. The shift is then recovered as a
        # difference of two independent posteriors.
        theta_t = bank["theta_c"] + bank["delta"]
        ctx = nde.phi(np.concatenate([bank["x_base"], bank["x_treat"]]))
        lat = nde.theta_to_z(np.concatenate([bank["theta_c"], theta_t]))
        idx_val = np.concatenate([idx_val, idx_val + n])
        idx_cal = np.concatenate([idx_cal, idx_cal + n])
        idx_tr = np.concatenate([idx_tr, idx_tr + n])
        scaler = nde.Standardiser.fit(ctx[idx_tr])
        C = torch.as_tensor(scaler(ctx), dtype=torch.float32)
        Z = torch.as_tensor(lat, dtype=torch.float32)
        Y = torch.zeros((C.shape[0], P.N_SHIFT))
        flow = zuko.flows.MAF(features=P.N_PARAM, context=F.N_FEATURE,
                              transforms=args.transforms,
                              hidden_features=[args.hidden] * args.depth).to(dev)
        presence = nde.Presence(n_context=F.N_FEATURE).to(dev)
    else:
        ctx = nde.context(bank["x_base"], bank["x_treat"])
        scaler = nde.Standardiser.fit(ctx[idx_tr])
        C = torch.as_tensor(scaler(ctx), dtype=torch.float32)
        Z = torch.as_tensor(np.concatenate(
            [nde.theta_to_z(bank["theta_c"]),
             nde.delta_to_z(bank["delta"][:, P.SHIFT_IDX])], axis=1),
            dtype=torch.float32)
        Y = torch.as_tensor(bank["active"].astype(np.float32))
        flow, presence = nde.Twin.build(device=dev, transforms=args.transforms,
                                        hidden=args.hidden, depth=args.depth)
    opt = torch.optim.AdamW(list(flow.parameters()) + list(presence.parameters()),
                            lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    bce_raw = torch.nn.BCEWithLogitsLoss(reduction="none")

    W = torch.as_tensor(weights, dtype=torch.float32)

    def batches(idx, shuffle=True):
        order = rng.permutation(idx) if shuffle else idx
        for a in range(0, len(order), args.batch):
            sel = torch.as_tensor(order[a:a + args.batch])
            yield (C[sel].to(dev), Z[sel].to(dev), Y[sel].to(dev),
                   W[sel].to(dev))

    # The two heads have disjoint parameters and different loss scales, so each
    # keeps the state at its own best validation loss rather than at the best
    # of their sum.
    best_nll, best_bce, bad = np.inf, np.inf, 0
    best_flow = best_pres = None
    t0 = time.time()
    for epoch in range(args.epochs):
        flow.train(); presence.train()
        tot = nb = 0.0
        for c, z, y, w in batches(idx_tr):
            opt.zero_grad(set_to_none=True)
            wn = w / w.mean()
            nll = -(flow(c).log_prob(z) * wn).mean()
            if args.unpaired:
                cls = torch.zeros((), device=dev)
            else:
                per = bce_raw(presence(c), y).mean(dim=1)
                cls = (per * wn).mean()
            loss = nll + cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(flow.parameters()) + list(presence.parameters()), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()

        flow.eval(); presence.eval()
        with torch.no_grad():
            vn = vc = vb = 0.0
            for c, z, y, w in batches(idx_val, shuffle=False):
                wn = w / w.mean()
                vn += float(-(flow(c).log_prob(z) * wn).mean())
                vc += (float((bce_raw(presence(c), y).mean(dim=1) * wn).mean())
                       if not args.unpaired else 0.0)
                vb += 1
        v = (vn + vc) / vb
        print(f"epoch {epoch + 1:3d}  train {tot / nb:8.3f}  val {v:8.3f} "
              f"(nll {vn / vb:7.3f}  bce {vc / vb:6.4f})  {time.time() - t0:5.0f}s",
              flush=True)
        improved = False
        if vn / vb < best_nll - 1e-4:
            best_nll, improved = vn / vb, True
            best_flow = {k: t.detach().cpu().clone()
                         for k, t in flow.state_dict().items()}
        if not args.unpaired and vc / vb < best_bce - 1e-5:
            best_bce, improved = vc / vb, True
            best_pres = {k: t.detach().cpu().clone()
                         for k, t in presence.state_dict().items()}
        bad = 0 if improved else bad + 1
        if bad >= args.patience:
            print("early stop")
            break

    if best_flow is not None:
        flow.load_state_dict(best_flow)
    if best_pres is not None:
        presence.load_state_dict(best_pres)
    best = best_nll + (0.0 if args.unpaired else best_bce)

    # Calibrate presence probabilities on the untouched calibration split. The
    # unpaired baseline has no presence head: it ranks mechanisms by the size
    # of a difference, which is the only score that design can offer.
    flow.eval(); presence.eval()
    calib: dict[str, dict] = {}
    if not args.unpaired:
        with torch.no_grad():
            logits = []
            for a in range(0, len(idx_cal), 4096):
                sel = torch.as_tensor(idx_cal[a:a + 4096])
                logits.append(presence(C[sel].to(dev)).cpu().numpy())
        pr = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
        yc = bank["active"][idx_cal]
        calib = {str(j): reliability(pr[:, j], yc[:, j].astype(float))
                 for j in range(P.N_SHIFT)}

    meta = {"unpaired": bool(args.unpaired),
            "transforms": args.transforms, "hidden": args.hidden,
            "depth": args.depth, "bank": str(args.bank), "pairs": int(n),
            "val_loss": float(best), "val_nll": float(best_nll),
            "val_bce": float(best_bce) if not args.unpaired else None,
            "epochs_run": epoch + 1,
            "param_keys": list(P.KEYS),
            "shift_keys": [P.KEYS[i] for i in P.SHIFT_IDX],
            "n_val": int(n_val), "n_cal": int(n_cal), "seed": args.seed,
            "match_domain": bool(args.match_domain),
            "reweight_domain": bool(args.reweight_domain)}
    twin = nde.Twin(flow, presence, scaler, device=dev, calibration=calib, meta=meta)
    twin.save(args.out)
    np.save(pathlib.Path(args.out) / "val_index.npy", idx_val)
    print("saved", args.out, json.dumps({k: meta[k] for k in ("pairs", "val_loss", "epochs_run")}))


if __name__ == "__main__":
    main()
