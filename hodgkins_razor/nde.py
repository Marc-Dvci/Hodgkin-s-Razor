"""Amortised inference over culture parameters and the shift a compound caused.

Two heads read the same paired recording:

* a conditional masked autoregressive flow over the joint vector of baseline
  parameters and shift, which gives effect sizes with credible intervals and
  lets the twin be re-simulated for the predictive check;
* a classifier that gives, per mechanism, the probability that it was in the
  active set at all. Sizes and presence are different questions, and reading
  presence off a size posterior would hide how sharp the sparse prior is.

Both are trained once on the simulation bank and then applied without any
further simulation, so scoring a new recording costs a forward pass.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import zuko

from . import features as F
from . import params as P
from . import shift as SH

# Features whose bank distribution is heavy-tailed are compressed before
# standardising, so that one runaway network cannot dominate the scaling.
LOG_FEATURES = ("mfr", "nbr", "nbd", "mean_isi", "sd_isi_temporal",
                "sd_isi_electrode", "isi_cv", "burst_rise_s", "cvibi",
                "tonic_rate", "fano_25ms", "fano_250ms", "fano_1s",
                "burst_decay_ratio", "burst_onset_jitter")
_LOG_MASK = np.array([n in LOG_FEATURES for n in F.NAMES])


def phi(x: np.ndarray) -> np.ndarray:
    """Per-recording feature transform, before standardisation."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    out[..., _LOG_MASK] = np.log1p(np.clip(x[..., _LOG_MASK], 0.0, None))
    return out


def context(x_base: np.ndarray, x_treat: np.ndarray) -> np.ndarray:
    """Conditioning vector: both recordings and their difference."""
    a, b = phi(x_base), phi(x_treat)
    return np.concatenate([a, b, b - a], axis=-1)


def delta_to_z(delta_shift: np.ndarray) -> np.ndarray:
    """Compress the shift so the narrow prior component is O(1) wide."""
    return np.arcsinh(np.asarray(delta_shift) / SH.INACTIVE_SCALE)


def z_to_delta(z: np.ndarray) -> np.ndarray:
    # The inverse is a sinh, so a far-out draw would overflow before the prior
    # box clips it. The bound is far outside any shift the prior can produce.
    return np.sinh(np.clip(np.asarray(z), -25.0, 25.0)) * SH.INACTIVE_SCALE


def theta_to_z(theta_t: np.ndarray) -> np.ndarray:
    """Map the prior box onto [-1, 1]."""
    return 2.0 * (np.asarray(theta_t) - P.TLO) / (P.THI - P.TLO) - 1.0


def z_to_theta(z: np.ndarray) -> np.ndarray:
    return P.TLO + (np.asarray(z) + 1.0) * 0.5 * (P.THI - P.TLO)


N_LATENT = P.N_PARAM + P.N_SHIFT
N_CONTEXT = 3 * F.N_FEATURE


class Standardiser:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.where(np.asarray(scale) > 1e-8, scale, 1.0)

    @classmethod
    def fit(cls, c: np.ndarray) -> "Standardiser":
        return cls(c.mean(0), c.std(0))

    def __call__(self, c: np.ndarray) -> np.ndarray:
        return (np.asarray(c, dtype=np.float64) - self.mean) / self.scale

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": np.asarray(self.scale).tolist()}

    @classmethod
    def load(cls, d: dict) -> "Standardiser":
        return cls(np.array(d["mean"]), np.array(d["scale"]))


class Presence(nn.Module):
    """Per-mechanism probability that the compound moved it."""

    def __init__(self, n_context: int = N_CONTEXT, hidden: int = 256,
                 n_out: int = P.N_SHIFT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_context, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class PresenceEnsemble(nn.Module):
    """Several presence heads trained side by side; their logits are averaged.

    Each head sees a different ordering of the same batches and starts from a
    different initialisation, so the spread across heads is a measure of how
    much a call depends on the fit rather than on the recording.
    """

    def __init__(self, k: int = 5, n_context: int = N_CONTEXT, hidden: int = 256,
                 n_out: int = P.N_SHIFT):
        super().__init__()
        self.heads = nn.ModuleList(Presence(n_context, hidden, n_out) for _ in range(k))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(c) for h in self.heads]).mean(0)

    def each(self, c: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(c) for h in self.heads])


class Twin:
    """Trained inference model: flow, presence head and the feature scaling.

    In the conditional form (`meta["conditional"]`), the flow is conditioned on
    the set of mechanisms the compound acted on as well as on the recording.
    Presence and size are then factorised the way the prior generates them: the
    presence head says which mechanisms moved, and the flow says by how much
    given that they did. A size read off an unconditional flow is dominated by
    the narrow prior component whenever presence is uncertain, which pulls
    every effect toward no change.
    """

    def __init__(self, flow, presence: Presence, scaler: Standardiser,
                 device: str = "cpu", calibration: dict | None = None,
                 meta: dict | None = None):
        self.flow = flow
        self.presence = presence
        self.scaler = scaler
        self.device = device
        self.calibration = calibration or {}
        self.meta = meta or {}

    # ---- construction -------------------------------------------------
    @staticmethod
    def build(device: str = "cuda", transforms: int = 6,
              hidden: int = 384, depth: int = 3, conditional: bool = False,
              ensemble: int = 1, n_feature: int = F.N_FEATURE) -> tuple:
        n_ctx = 3 * n_feature
        flow = zuko.flows.MAF(features=N_LATENT,
                              context=n_ctx + (P.N_SHIFT if conditional else 0),
                              transforms=transforms,
                              hidden_features=[hidden] * depth).to(device)
        presence = (PresenceEnsemble(ensemble, n_context=n_ctx) if ensemble > 1
                    else Presence(n_context=n_ctx)).to(device)
        return flow, presence

    @property
    def conditional(self) -> bool:
        return bool(self.meta.get("conditional", False))

    # ---- inference ----------------------------------------------------
    def _ctx(self, x_base: np.ndarray, x_treat: np.ndarray) -> torch.Tensor:
        c = self.scaler(context(np.atleast_2d(x_base), np.atleast_2d(x_treat)))
        return torch.as_tensor(c, dtype=torch.float32, device=self.device)

    def _finish(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = z_to_theta(z[:, :P.N_PARAM])
        inside = np.all((theta >= P.TLO - 1e-9) & (theta <= P.THI + 1e-9), axis=1)
        if inside.sum() >= 50:
            z, theta = z[inside], theta[inside]
        else:
            theta = np.clip(theta, P.TLO, P.THI)
        # Training labels are shifts that were realised inside the prior box,
        # so a draw implying a treated parameter outside it is outside the
        # prior support. Constraining it here keeps reported effect sizes on
        # the same scale the model was trained to produce.
        delta = z_to_delta(z[:, P.N_PARAM:])
        full = np.zeros((theta.shape[0], P.N_PARAM))
        full[:, P.SHIFT_IDX] = delta
        delta = (np.clip(theta + full, P.TLO, P.THI) - theta)[:, P.SHIFT_IDX]
        return theta, delta

    @torch.no_grad()
    def presence_probs(self, x_base: np.ndarray, x_treat: np.ndarray) -> tuple:
        """Calibrated and raw presence probabilities for one paired recording."""
        self.presence.eval()
        c = self._ctx(x_base, x_treat)
        logits = self.presence(c).cpu().numpy().ravel()
        p_raw = 1.0 / (1.0 + np.exp(-logits))
        return self.calibrate(p_raw), p_raw

    @torch.no_grad()
    def _sample(self, c: torch.Tensor, masks: np.ndarray | None,
                n_samples: int) -> np.ndarray:
        self.flow.eval()
        if masks is None:
            dist = self.flow(c.expand(1, -1))
            return dist.sample((n_samples,)).reshape(n_samples, N_LATENT).cpu().numpy()
        m = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        cc = torch.cat([c.expand(m.shape[0], -1), m], dim=1)
        return self.flow(cc).sample().reshape(m.shape[0], N_LATENT).cpu().numpy()

    @torch.no_grad()
    def posterior(self, x_base: np.ndarray, x_treat: np.ndarray,
                  n_samples: int = 4000, seed: int = 0) -> dict:
        """Posterior samples for one paired recording.

        In the conditional form the active set of each draw is sampled from the
        calibrated presence probabilities, so the joint samples mix the
        hypotheses in proportion to how probable each one is, and
        `effect_given_active` holds, per mechanism, the size of the shift if
        that mechanism is the one that moved.
        """
        c = self._ctx(x_base, x_treat)
        p_active, p_raw = self.presence_probs(x_base, x_treat)
        if not self.conditional:
            theta, delta = self._finish(self._sample(c, None, n_samples))
            return {"theta_c": theta, "delta": delta,
                    "p_active": p_active, "p_active_raw": p_raw}
        rng = np.random.default_rng(seed)
        masks = (rng.random((n_samples, P.N_SHIFT)) < p_active[None, :]).astype(np.float32)
        theta, delta = self._finish(self._sample(c, masks, n_samples))
        n_each = max(n_samples // 4, 400)
        given = []
        for j in range(P.N_SHIFT):
            m = np.zeros((n_each, P.N_SHIFT), dtype=np.float32)
            m[:, j] = 1.0
            _, d = self._finish(self._sample(c, m, n_each))
            given.append(d[:, j])
        n_min = min(len(g) for g in given)
        return {"theta_c": theta, "delta": delta, "p_active": p_active,
                "p_active_raw": p_raw,
                "effect_given_active": np.stack([g[:n_min] for g in given], axis=1)}

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the stored per-mechanism calibration."""
        if not self.calibration:
            return p
        out = np.asarray(p, dtype=float).copy()
        for j in range(out.shape[-1]):
            cal = self.calibration.get(str(j))
            if not cal:
                continue
            out[..., j] = np.interp(out[..., j], cal["x"], cal["y"])
        return out

    # ---- persistence --------------------------------------------------
    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.flow.state_dict(), path / "flow.pt")
        torch.save(self.presence.state_dict(), path / "presence.pt")
        (path / "scaler.json").write_text(json.dumps(self.scaler.state()))
        (path / "calibration.json").write_text(json.dumps(self.calibration))
        (path / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @classmethod
    def load(cls, path: str | pathlib.Path, device: str = "cpu") -> "Twin":
        path = pathlib.Path(path)
        meta = json.loads((path / "meta.json").read_text())
        flow, presence = cls.build(device=device,
                                   transforms=meta.get("transforms", 6),
                                   hidden=meta.get("hidden", 384),
                                   depth=meta.get("depth", 3),
                                   conditional=meta.get("conditional", False),
                                   ensemble=meta.get("ensemble", 1))
        flow.load_state_dict(torch.load(path / "flow.pt", map_location=device))
        presence.load_state_dict(torch.load(path / "presence.pt", map_location=device))
        scaler = Standardiser.load(json.loads((path / "scaler.json").read_text()))
        cal = json.loads((path / "calibration.json").read_text())
        meta["path"] = str(path)
        return cls(flow, presence, scaler, device=device, calibration=cal, meta=meta)


class UnpairedTwin:
    """Baseline: parameters inferred from one recording, shift as a difference.

    This is the paired design removed and nothing else. Same simulator, same
    features, same bank, same flow size.
    """

    def __init__(self, flow, scaler: Standardiser, device: str = "cpu",
                 meta: dict | None = None):
        self.flow = flow
        self.scaler = scaler
        self.device = device
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | pathlib.Path, device: str = "cpu") -> "UnpairedTwin":
        path = pathlib.Path(path)
        meta = json.loads((path / "meta.json").read_text())
        flow = zuko.flows.MAF(features=P.N_PARAM, context=F.N_FEATURE,
                              transforms=meta.get("transforms", 6),
                              hidden_features=[meta.get("hidden", 384)]
                              * meta.get("depth", 3)).to(device)
        flow.load_state_dict(torch.load(path / "flow.pt", map_location=device))
        scaler = Standardiser.load(json.loads((path / "scaler.json").read_text()))
        return cls(flow, scaler, device=device, meta=meta)

    @torch.no_grad()
    def theta(self, x: np.ndarray, n_samples: int = 4000) -> np.ndarray:
        self.flow.eval()
        c = torch.as_tensor(self.scaler(phi(np.atleast_2d(x))),
                            dtype=torch.float32, device=self.device)
        z = self.flow(c).sample((n_samples,)).reshape(n_samples, P.N_PARAM)
        return z_to_theta(z.cpu().numpy())

    def posterior(self, x_base: np.ndarray, x_treat: np.ndarray,
                  n_samples: int = 4000) -> dict:
        """Shift as the difference of two independent posterior medians.

        The presence score is the standardised size of that difference, which is
        the only ranking this design can offer.
        """
        tb = self.theta(x_base, n_samples)
        tt = self.theta(x_treat, n_samples)
        med = np.median(tt, axis=0) - np.median(tb, axis=0)
        spread = np.sqrt(np.var(tb, axis=0) + np.var(tt, axis=0)) + 1e-9
        score = np.abs(med / spread)[P.SHIFT_IDX]
        delta = (tt - tb)[:, P.SHIFT_IDX]
        return {"theta_c": tb, "delta": delta,
                "p_active": score / (1.0 + score), "p_active_raw": score}


def summarise(post: dict, credible: float = 0.90) -> list[dict]:
    """Per-mechanism effect size, interval and presence probability."""
    # The size a mechanism would have if it is the one that moved; the joint
    # samples mix in the draws where it did not, which drags it toward zero.
    delta = post.get("effect_given_active", post["delta"])
    lo_q, hi_q = (1 - credible) / 2, 1 - (1 - credible) / 2
    rows = []
    for j, col in enumerate(P.SHIFT_IDX):
        p = P.PARAMS[col]
        d = delta[:, j]
        med, lo, hi = np.median(d), np.quantile(d, lo_q), np.quantile(d, hi_q)
        if p.log:
            fold = float(np.exp(med))
            rows.append({"key": p.key, "label": p.label, "target": p.target,
                         "kind": "fold", "effect": fold,
                         "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
                         "percent": float((fold - 1.0) * 100.0),
                         "p_active": float(post["p_active"][j]),
                         "direction": "up" if med > 0 else "down"})
        else:
            rows.append({"key": p.key, "label": p.label, "target": p.target,
                         "kind": "absolute", "effect": float(med),
                         "lo": float(lo), "hi": float(hi),
                         "percent": float(100.0 * med / (p.hi - p.lo)),
                         "p_active": float(post["p_active"][j]),
                         "direction": "up" if med > 0 else "down"})
    rows.sort(key=lambda r: -r["p_active"])
    return rows
