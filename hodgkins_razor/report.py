"""Turn a posterior into a report a pharmacologist can act on.

Every sentence here is generated from a number produced upstream. The report
never states a mechanism when the guard says the recording is outside the
model, and it always names what it could not distinguish.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from . import features as F
from . import nde
from . import params as P

CALL_THRESHOLD = 0.5
CONFUSABLE = 0.15   # presence probabilities this close are not separated


def _fmt_effect(row: dict) -> str:
    if row["kind"] == "fold":
        return (f"{row['effect']:.2f}x "
                f"[{row['lo']:.2f}, {row['hi']:.2f}]")
    return f"{row['effect']:+.2f} [{row['lo']:+.2f}, {row['hi']:+.2f}]"


def build(post: dict, guard: dict | None, x_base: np.ndarray,
          x_treat: np.ndarray, meta: dict | None = None) -> dict:
    """Assemble the structured report for one paired recording."""
    rows = nde.summarise(post)
    called = [r for r in rows if r["p_active"] >= CALL_THRESHOLD]
    inside = True if guard is None else bool(guard["inside_model"])

    verdict: str
    if not inside:
        verdict = "outside_model"
    elif not called:
        verdict = "no_mechanism_called"
    else:
        verdict = "mechanism_called"

    rivals: list[str] = []
    if len(rows) > 1 and called:
        top = rows[0]["p_active"]
        rivals = [r["key"] for r in rows[1:] if top - r["p_active"] < CONFUSABLE]

    # The class is the coarser question, and it is answered better than the
    # exact conductance, so it is reported alongside rather than instead.
    agg = np.zeros(len(P.CLASS_NAMES))
    for r in rows:
        agg[P.class_index(r["key"])] += r["p_active"]
    total = float(agg.sum()) or 1.0
    classes = [{"name": n, "probability": float(v / total)}
               for n, v in zip(P.CLASS_NAMES, agg)]
    classes.sort(key=lambda c: -c["probability"])

    obs = {n: float(v) for n, v in zip(F.NAMES, np.asarray(x_base).ravel())}
    obs_t = {n: float(v) for n, v in zip(F.NAMES, np.asarray(x_treat).ravel())}

    culture = []
    for j, p in enumerate(P.PARAMS):
        s = post["theta_c"][:, j]
        med = float(np.median(s))
        lo, hi = float(np.quantile(s, 0.05)), float(np.quantile(s, 0.95))
        conv = (lambda v: float(np.exp(v))) if p.log else (lambda v: float(v))
        culture.append({"key": p.key, "label": p.label, "unit": p.unit,
                        "median": conv(med), "lo": conv(lo), "hi": conv(hi)})

    body = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "inside_model": inside,
        "guard": None if guard is None else {
            "discrepancy": float(guard["discrepancy"]),
            "threshold": float(guard["threshold"])},
        "mechanisms": rows,
        "classes": classes,
        "called": [r["key"] for r in called],
        "not_separated_from_top": rivals,
        "culture": culture,
        "observed_baseline": obs,
        "observed_treated": obs_t,
        "meta": meta or {},
    }
    body["sentence"] = sentence(body)
    body["content_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "content_sha256"},
                   sort_keys=True, default=str).encode()).hexdigest()
    return body


def sentence(body: dict) -> str:
    """One line stating what the recording supports."""
    if body["verdict"] == "outside_model":
        d, t = body["guard"]["discrepancy"], body["guard"]["threshold"]
        return (f"This recording is outside the twin: the fitted model cannot "
                f"reproduce it (discrepancy {d:.1f} against a threshold of "
                f"{t:.1f}). No mechanism is named.")
    if body["verdict"] == "no_mechanism_called":
        top = body["mechanisms"][0]
        return (f"No mechanism reaches the calling threshold. The closest is "
                f"{top['label']} at probability {top['p_active']:.2f}.")
    parts = []
    for r in body["mechanisms"]:
        if r["p_active"] < CALL_THRESHOLD:
            continue
        parts.append(f"{r['target'] or r['label']} {r['direction']} "
                     f"({_fmt_effect(r)}, probability {r['p_active']:.2f})")
    s = "This compound moved " + "; and ".join(parts) + "."
    top_class = body["classes"][0]
    s += (f" The mechanism class is {top_class['name']}, "
          f"at probability {top_class['probability']:.2f}.")
    if body["not_separated_from_top"]:
        s += (" On this recording the twin does not separate it from "
              + ", ".join(body["not_separated_from_top"]) + ".")
    return s


def to_markdown(body: dict) -> str:
    lines = ["# Mechanism report", "", body["sentence"], ""]
    if body["guard"]:
        lines += [f"Predictive check: discrepancy {body['guard']['discrepancy']:.2f}, "
                  f"threshold {body['guard']['threshold']:.2f}, "
                  f"{'inside' if body['inside_model'] else 'outside'} the model.", ""]
    lines += ["| Mechanism | Probability | Effect | Target |", "|---|---|---|---|"]
    for r in body["mechanisms"]:
        lines.append(f"| {r['label']} | {r['p_active']:.2f} | {_fmt_effect(r)} | "
                     f"{r['target'] or '-'} |")
    lines += ["", "## Mechanism class", "", "| Class | Probability |", "|---|---|"]
    for c in body["classes"]:
        lines.append(f"| {c['name']} | {c['probability']:.2f} |")
    lines += ["", "## Culture parameters (baseline)", "",
              "| Parameter | Median | 90% interval | Unit |", "|---|---|---|---|"]
    for c in body["culture"]:
        lines.append(f"| {c['label']} | {c['median']:.3g} | "
                     f"{c['lo']:.3g} to {c['hi']:.3g} | {c['unit'] or '-'} |")
    lines += ["", f"Report hash: `{body['content_sha256'][:16]}`",
              f"Generated {body['generated_utc']}"]
    return "\n".join(lines)
