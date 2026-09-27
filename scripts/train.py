"""Train the flow and the presence head on the simulation bank.

    python scripts/train.py --bank data/bank_v2 --view grid16 --out models/twin_v2_grid16

The bank is split by simulated culture, not by record: the shifts drawn on one
baseline share its wiring and its baseline recording, so a record-level split
lets the validation and calibration numbers see cultures the model was fitted
on. Nothing from any recorded experiment enters this script.
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

from hodgkins_razor import features as F, nde, params as P, simulator as S


def load_bank(path: pathlib.Path, view: str = "grid16") -> dict:
    """Load every shard, reading the recordings through one electrode layout.

    Version 1 shards hold one layout and no culture index; the culture index is
    then recovered from runs of identical baseline parameters, which is how
    they were written. Several banks of one design can be read together by
    passing their directories joined with commas; relative parts after the
    first resolve against the repository root.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    parts = str(path).split(",")
    dirs = [pathlib.Path(parts[0])] + [pathlib.Path(q) if pathlib.Path(q).is_absolute()
                                       else root / q for q in parts[1:]]
    shards = [s for d in dirs for s in sorted(d.glob("shard_*.npz"))]
    if not shards:
        raise SystemExit(f"no shards in {path}")
    views = list(S.VIEWS)
    acc = {k: [] for k in ("theta_c", "delta", "active", "x_base", "x_treat",
                           "group", "domain")}
    offset = 0
    for s in shards:
        d = np.load(s)
        xb, xt = d["x_base"], d["x_treat"]
        if xb.ndim == 3:
            v = views.index(view)
            xb, xt = xb[:, v], xt[:, v]
        elif view != "grid16":
            raise SystemExit(f"{s.name} holds only the grid16 layout")
        if "group" in d:
            g = d["group"].astype(np.int64)
        else:
            tc = d["theta_c"]
            new = np.r_[True, np.any(tc[1:] != tc[:-1], axis=1)]
            g = np.cumsum(new) - 1
        acc["group"].append(g + offset)
        acc["domain"].append(d["domain"].astype(np.int8) if "domain" in d
                             else np.zeros(g.size, dtype=np.int8))
        offset += int(g.max()) + 1
        for k in ("theta_c", "delta", "active"):
            acc[k].append(d[k])
        # Sister banks store the second recording's own parameters, which
        # carry the drift between sisters as well as the shift.
        acc.setdefault("theta_t", []).append(
            d["theta_t"] if "theta_t" in d else d["theta_c"] + d["delta"])
        acc["x_base"].append(xb)
        acc["x_treat"].append(xt)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"bank: {len(shards)} shards, {out['theta_c'].shape[0]} pairs, "
          f"{len(np.unique(out['group']))} cultures, view {view}")
    return out


def split_groups(group: np.ndarray, val_frac: float, cal_frac: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Validation, calibration and training indices, disjoint by culture."""
    ids = rng.permutation(np.unique(group))
    n_val = int(round(val_frac * ids.size))
    n_cal = int(round(cal_frac * ids.size))
    in_val = np.isin(group, ids[:n_val])
    in_cal = np.isin(group, ids[n_val:n_val + n_cal])
    idx = np.arange(group.size)
    return idx[in_val], idx[in_cal], idx[~in_val & ~in_cal]


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 12) -> dict:
    """Monotone calibration map from a held-out split."""
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
    ap.add_argument("--view", default="grid16", choices=list(S.VIEWS))
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
    ap.add_argument("--conditional", action="store_true",
                    help="condition the flow on the active mechanism set")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="number of presence heads averaged")
    ap.add_argument("--domain", default="",
                    help="train only on cultures admitted to this recording "
                         "system's domain (version 2 banks)")
    ap.add_argument("--limit", type=int, default=0,
                    help="train on at most this many pairs, for a size ablation")
    ap.add_argument("--match-domain", action="store_true",
                    help="version 1: keep only simulated baselines inside the "
                         "box the Tampere baselines span")
    ap.add_argument("--unpaired", action="store_true",
                    help="baseline: one flow over parameters from a single "
                         "recording, with no pairing and no shift prior")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    bank = load_bank(pathlib.Path(args.bank), args.view)
    if args.match_domain:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from fit_regime import BASELINE_BOX, domain_mask
        keep = domain_mask(bank["x_base"])
        bank = {k: v[keep] for k, v in bank.items()}
        print(f"domain match: kept {keep.sum()} of {keep.size} pairs "
              f"({keep.mean():.1%}) inside {BASELINE_BOX}")
    if args.domain:
        keep = bank["domain"] == list(S.VIEWS).index(args.domain)
        bank = {k: v[keep] for k, v in bank.items()}
        print(f"domain {args.domain}: {keep.sum()} pairs")
    n = bank["theta_c"].shape[0]
    rng = np.random.default_rng(args.seed)
    idx_val, idx_cal, idx_tr = split_groups(bank["group"], args.val_frac,
                                            args.cal_frac, rng)
    if args.limit and idx_tr.size > args.limit:
        idx_tr = np.sort(rng.choice(idx_tr, size=args.limit, replace=False))
    print(f"split by culture: train {idx_tr.size}, val {idx_val.size}, "
          f"cal {idx_cal.size}")

    if args.unpaired:
        # Every recording becomes its own record: the baseline with theta_c and
        # the treated one with theta_c + delta. The shift is then recovered as a
        # difference of two independent posteriors.
        theta_t = bank["theta_t"]
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
                                        hidden=args.hidden, depth=args.depth,
                                        conditional=args.conditional,
                                        ensemble=args.ensemble)
    opt = torch.optim.AdamW(list(flow.parameters()) + list(presence.parameters()),
                            lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = torch.nn.BCEWithLogitsLoss()
    ens = isinstance(presence, nde.PresenceEnsemble)

    def flow_ctx(c, y):
        return torch.cat([c, y], dim=1) if args.conditional else c

    def presence_loss(c, y):
        if ens:
            return torch.stack([bce(h, y) for h in presence.each(c)]).mean()
        return bce(presence(c), y)

    def batches(idx, shuffle=True):
        order = rng.permutation(idx) if shuffle else idx
        for a in range(0, len(order), args.batch):
            sel = torch.as_tensor(order[a:a + args.batch])
            yield C[sel].to(dev), Z[sel].to(dev), Y[sel].to(dev)

    # The two heads have disjoint parameters and different loss scales, so each
    # keeps the state at its own best validation loss rather than at the best
    # of their sum.
    best_nll, best_bce, bad = np.inf, np.inf, 0
    best_flow = best_pres = None
    t0 = time.time()
    for epoch in range(args.epochs):
        flow.train(); presence.train()
        tot = nb = 0.0
        for c, z, y in batches(idx_tr):
            opt.zero_grad(set_to_none=True)
            nll = -flow(flow_ctx(c, y)).log_prob(z).mean()
            cls = (torch.zeros((), device=dev) if args.unpaired
                   else presence_loss(c, y))
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
            for c, z, y in batches(idx_val, shuffle=False):
                vn += float(-flow(flow_ctx(c, y)).log_prob(z).mean())
                vc += float(bce(presence(c), y)) if not args.unpaired else 0.0
                vb += 1
        print(f"epoch {epoch + 1:3d}  train {tot / nb:8.3f}  val {(vn + vc) / vb:8.3f} "
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

    # Calibrate presence probabilities on the untouched calibration cultures.
    # The unpaired baseline has no presence head: it ranks mechanisms by the
    # size of a difference, which is the only score that design can offer.
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

    meta = {"unpaired": bool(args.unpaired), "view": args.view,
            "conditional": bool(args.conditional), "ensemble": int(args.ensemble),
            "transforms": args.transforms, "hidden": args.hidden,
            "depth": args.depth, "bank": str(args.bank), "pairs": int(n),
            "pairs_trained": int(idx_tr.size),
            "cultures": int(len(np.unique(bank["group"]))),
            "val_loss": float(best), "val_nll": float(best_nll),
            "val_bce": float(best_bce) if not args.unpaired else None,
            "epochs_run": epoch + 1,
            "param_keys": list(P.KEYS),
            "shift_keys": [P.KEYS[i] for i in P.SHIFT_IDX],
            "n_val": int(idx_val.size), "n_cal": int(idx_cal.size),
            "split": "by culture", "seed": args.seed,
            "domain": args.domain or "all",
            "match_domain": bool(args.match_domain)}
    if not args.unpaired:
        # A fixed sample of training contexts for the typicality guard, and its
        # threshold: the 97.5th percentile over the validation cultures, which
        # the reference never contains.
        from hodgkins_razor.ppc import Typicality
        ref_idx = np.sort(rng.choice(idx_tr, size=min(20000, idx_tr.size), replace=False))
        ref = C[torch.as_tensor(ref_idx)].numpy().astype(np.float16)
        typ = Typicality(ref.astype(np.float32))
        v_idx = rng.choice(idx_val, size=min(4000, idx_val.size), replace=False)
        vs = typ.score(C[torch.as_tensor(v_idx)].numpy())
        meta["typicality_threshold"] = float(np.quantile(vs, 0.975))
        meta["typicality_quantile"] = 0.975
    twin = nde.Twin(flow, presence, scaler, device=dev, calibration=calib, meta=meta)
    twin.save(args.out)
    if not args.unpaired:
        np.save(pathlib.Path(args.out) / "reference_context.npy", ref)
    out = pathlib.Path(args.out)
    np.save(out / "val_index.npy", idx_val)
    np.save(out / "cal_index.npy", idx_cal)
    print("saved", args.out, json.dumps({k: meta[k] for k in ("pairs", "val_loss", "epochs_run")}))


if __name__ == "__main__":
    main()
