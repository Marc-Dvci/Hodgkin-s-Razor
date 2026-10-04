"""The known target of each bundled example, and what the twin made of it.

The examples on the page are development recordings: their answers were known
when the twin was built, so they show how a report reads, not how well the
twin does. Each one is labelled with its compound's published target and the
twin's outcome, so a reader sees a miss as a miss.
"""
from __future__ import annotations

from .tampere import MECHANISM

LABEL = {"g_na": "Na channels", "g_ampa": "AMPA receptors", "g_nmda": "NMDA receptors",
         "g_gaba": "GABA-A receptors", "u_rel": "release probability",
         "tau_d": "vesicle recycling"}
DEFAULT = "rat_ttx"


def expected(compound: str) -> tuple[set[str], str, str]:
    """(accepted parameters, direction, text) for a compound, or empty."""
    c = compound.replace(" ", "")
    if c.lower() == "dynasore":
        return {"u_rel", "tau_d"}, "down", "vesicle recycling (dynamin)"
    if c.lower().startswith("vehicle"):
        return set(), "", "none (vehicle)"
    match = {k.lower(): v for k, v in MECHANISM.items()}
    if c.lower() in match:
        key, d = match[c.lower()]
        return {key}, d, f"{LABEL.get(key, key)} {d}"
    return set(), "", ""


def outcome(compound: str, analysis: dict) -> str:
    """One phrase: what the twin returned against the known target."""
    keys, _, text = expected(compound)
    rep = analysis.get("report", analysis)
    if not rep.get("inside_model", True):
        return "outside the model, nothing named"
    called = [m["key"] for m in rep.get("mechanisms", [])
              if m.get("key") in set(rep.get("called", []))] or list(rep.get("called", []))
    if not text.startswith("none") and not keys:
        return ""
    if text.startswith("none"):
        return "correct: nothing called" if not called else "false call"
    if not called:
        return "nothing called"
    if called[0] in keys:
        return "correct"
    for i, k in enumerate(called):
        if k in keys:
            return f"named {i + 1}{'nd' if i == 1 else 'rd' if i == 2 else 'th'}, after a wrong call"
    return "missed"


def describe(compound: str, analysis: dict | None) -> dict:
    _, _, text = expected(compound)
    out = {"target": text}
    if analysis is not None:
        out["outcome"] = outcome(compound, analysis)
    return out
