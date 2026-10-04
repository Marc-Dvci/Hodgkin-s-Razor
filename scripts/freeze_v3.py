"""Freeze the version 3 models and write the third pre-registration.

    python scripts/freeze_v3.py

Fills scripts/templates/PREREGISTRATION_v3.draft.md with a digest of every model directory
and of the files that define the scored set, the chosen drift, and the
simulation evidence the bars were set against. Writes PREREGISTRATION_v3.md and
its SHA-256, and stops if either already exists: a pre-registration is written
once. Reads simulations only.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_v2 import auroc, model_digest
from hodgkins_razor import nde, params as P, simulator as S
from train import load_bank

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEW = "mcs60q"
TWIN = "models/twin_v3_mcs60q"
UNPAIRED = "models/twin_v3_unpaired_mcs60q"
BANK = "data/bank_v3"
DOMAIN = "models/domain_v3.json"
DRIFT = "models/drift_v3.json"
DRIFT_AC = "models/drift_v3_ac.json"
KEY = "g_nmda"
SHIFT_KEYS = [P.KEYS[i] for i in P.SHIFT_IDX]


STOP_RULE = {"alone_min_auroc": 0.80, "coshift_min_auroc": 0.70}   # docs/STOP_RULE_v3.md
EARLY = (10.0, 14.0)      # the ages at which treated pairs exist: 10, 11, 13 and 14 days


def _presence(twin, bank, sel) -> np.ndarray:
    c = twin.scaler(nde.context(bank["x_base"][sel], bank["x_treat"][sel]))
    with torch.no_grad():
        logits = twin.presence(torch.as_tensor(c, dtype=torch.float32,
                                               device=twin.device)).cpu().numpy()
    return twin.calibrate(1.0 / (1.0 + np.exp(-logits)))


def coshift_test(twin, bank, idx_val) -> dict:
    """Stop rule 2: `g_nmda` down at least two-fold, at most one other mechanism moving."""
    j = SHIFT_KEYS.index(KEY)
    act = bank["active"][idx_val]
    d = bank["delta"][idx_val][:, P.SHIFT_IDX]
    t = idx_val[act[:, j] & (d[:, j] < -np.log(2.0)) & (act.sum(1) <= 2)]
    z = idx_val[act.sum(1) == 0]
    p = _presence(twin, bank, np.r_[t, z])
    y = np.r_[np.ones(len(t)), np.zeros(len(z))].astype(bool)
    return {"n_treated": int(len(t)), "n_null": int(len(z)),
            "auroc_key_window": auroc(p[:, j], y)}


def counts() -> dict:
    """Preparations per group in the primary age range, from the metadata alone."""
    from hodgkins_razor import charlesworth as C
    rows = [r for r in C.pairs_index() if EARLY[0] <= r["div"] <= EARLY[1]]
    genes = {C._gene(r["genotype"]) for r in rows if r["kind"] == "treated"}

    def preps(f):
        return sorted({r["prep"] for r in rows if f(r)})
    out = {"treated": preps(lambda r: r["kind"] == "treated"),
           "null_matched": preps(lambda r: r["kind"] == "null"
                                 and C._gene(r["genotype"]) in genes),
           "null_pooled": preps(lambda r: r["kind"] == "null"),
           "a_c": preps(lambda r: r["kind"] == "a-c")}
    by_gene: dict = {}
    for r in rows:
        if r["kind"] in ("treated", "null"):
            by_gene.setdefault(r["kind"], {}).setdefault(C._gene(r["genotype"]), set()).add(r["prep"])
    return {"n": {k: len(v) for k, v in out.items()},
            "by_genotype": {k: {g: len(v) for g, v in sorted(d.items())}
                            for k, d in by_gene.items()},
            "excluded_four_array_preps": sorted(C.four_array_preps()),
            "treated_divs": sorted({r["div"] for r in rows if r["kind"] == "treated"})}


def real_a_c(twin) -> dict:
    """The twin on the real untreated A-C sister pairs at 10 to 14 days (no bar)."""
    from hodgkins_razor import charlesworth as C
    from evaluate_v3 import by_prep, score_windows
    pairs = C.load(kinds=("a-c",), min_div=EARLY[0], max_div=EARLY[1], n_windows=6)
    rows = score_windows(twin, pairs, 1000, KEY, min_events=50, min_elec=3)
    preps = by_prep(rows, EARLY[0], EARLY[1], KEY)
    pk = np.array([p["p_key"] for p in preps])
    return {"n_preps": len(preps), "n_windows": len(rows),
            "median_p_key": float(np.median(pk)) if len(pk) else float("nan"),
            "called_key": float(np.mean(pk > 0.5)) if len(pk) else float("nan"),
            "top1_counts": {k: sum(p["top1"] == k for p in preps)
                            for k in sorted({p["top1"] for p in preps})}}


def simulated_test(twin, bank, idx_val) -> dict:
    """The blind test's contrast on held-out simulated sister pairs.

    Treated: pairs whose only mechanism is `g_nmda` down by at least
    five-fold, the regime of a saturating antagonist. Null: pairs with no
    mechanism, only the drift between sisters. One 60 s quadrant window per
    pair, where the recorded test averages many per preparation.
    """
    j = SHIFT_KEYS.index(KEY)
    act = bank["active"][idx_val]
    d = bank["delta"][idx_val][:, P.SHIFT_IDX]
    t = idx_val[(act.sum(1) == 1) & act[:, j] & (d[:, j] < -np.log(5.0))]
    z = idx_val[act.sum(1) == 0]
    p = _presence(twin, bank, np.r_[t, z])
    y = np.r_[np.ones(len(t)), np.zeros(len(z))].astype(bool)
    top = p.argmax(1)
    return {"n_treated": int(len(t)), "n_null": int(len(z)),
            "auroc_key_window": auroc(p[:, j], y),
            "top1_treated": float(np.mean(top[y] == j)),
            "top1_null": float(np.mean(top[~y] == j)),
            "called_key_treated": float(np.mean(p[y, j] > 0.5)),
            "called_key_null": float(np.mean(p[~y, j] > 0.5))}


def main() -> None:
    md, rec = ROOT / "PREREGISTRATION_v3.md", ROOT / "PREREGISTRATION_v3.sha256"
    if md.exists() or rec.exists():
        raise SystemExit("PREREGISTRATION_v3 already exists; it is written once")
    drift = json.loads((ROOT / DRIFT).read_text())
    sim_path = ROOT / "results" / "v3" / "sim_mcs60q.json"
    sim = json.loads(sim_path.read_text())
    twin = nde.Twin.load(ROOT / TWIN, device="cuda")
    bank_path = twin.meta.get("bank", BANK)      # the bank the frozen twin was trained on
    bank = load_bank(ROOT / bank_path, VIEW)
    keep = bank["domain"] == list(S.VIEWS).index(VIEW)
    bank = {k: v[keep] for k, v in bank.items()}
    idx_val = np.load(ROOT / TWIN / "val_index.npy")
    test = simulated_test(twin, bank, idx_val)
    co = coshift_test(twin, bank, idx_val)
    ok = (test["auroc_key_window"] >= STOP_RULE["alone_min_auroc"]
          and co["auroc_key_window"] >= STOP_RULE["coshift_min_auroc"])
    log = ROOT / "results" / "v3" / "stop_rule_attempts.json"
    attempts = json.loads(log.read_text()) if log.exists() else []
    digest = model_digest(ROOT / TWIN)
    # One entry per twin: a repeated check of the same model is not a new attempt.
    attempts = [x for x in attempts if x["digest"] != digest]
    attempts.append({"twin": TWIN, "digest": digest,
                     "alone": test, "coshift": co, "passed": bool(ok),
                     "date": datetime.datetime.now().isoformat(timespec="seconds")})
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(attempts, indent=1))
    print(json.dumps(attempts[-1], indent=1))
    if not ok or "--check" in sys.argv:
        raise SystemExit("stop rule " + ("met" if ok else "NOT met (docs/STOP_RULE_v3.md): "
                         "revise the twin on simulations") + "; nothing was written")
    groups = counts()
    ac = real_a_c(twin)

    def sha(p: str) -> str:
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

    spec = {
        "view": VIEW, "twin": TWIN, "unpaired": UNPAIRED, "bank": bank_path,
        "domain_file": DOMAIN, "drift": drift["chosen"],
        "frozen_models": {TWIN: model_digest(ROOT / TWIN),
                          UNPAIRED: model_digest(ROOT / UNPAIRED)},
        "frozen_files": {p: sha(p) for p in (DOMAIN, DRIFT, DRIFT_AC,
                                             "hodgkins_razor/charlesworth.py",
                                             "scripts/evaluate_v3.py", "docs/STOP_RULE_v3.md",
                                             "data/raw/charlesworth2015/g2c-1/00g2cdata.csv")},
        "operating_points": {"posterior_samples": 1000, "window_s": 60.0,
                             "windows_per_recording": 6, "presence_call": 0.5,
                             "predictive_draws": 48, "ppc_quantile": 0.975,
                             "ppc_calibration_records": 200,
                             "min_events": 50, "min_electrodes": 3},
        "blind_test": {"dataset": "Charlesworth et al. 2015, Zenodo 31085",
                       "compound": "APV 50 uM, chronic from 7 days in vitro",
                       "accept": KEY, "direction": "down", "early_divs": list(EARLY),
                       "unit": "preparation",
                       "null_group": "untreated preparations of the treated genotypes; "
                                     "four-array preparations excluded",
                       "groups": groups,
                       "top1_hit_rule": "top-1 mechanism is g_nmda whatever its sign; the "
                                        "sign is a secondary outcome",
                       "primary": {"metric": "AUROC of p(g_nmda), treated against "
                                             "genotype-matched null preparations",
                                   "min_auroc": 0.70, "ci_lower_above": 0.50,
                                   "bootstrap": 4000},
                       "co_primary": {"metric": "top-1 g_nmda share, treated against null",
                                      "test": "one-sided Fisher exact", "alpha": 0.05}},
        "guard": {"must_pass_max": 0.10, "must_fire_min": 0.80, "n_cases": 60,
                  "windows_per_quadrant": 2},
        "simulated_test": test,
        "simulated_test_coshift": co,
        "stop_rule": STOP_RULE,
        "real_a_c_null": ac,
    }
    rt = sim["recovery_top1"]
    lines = [
        f"Held-out simulated sister pairs of this recording system (`{sim_path.relative_to(ROOT)}`):",
        "",
        "| Case | n | Top-1 | Top-2 | Class |", "|---|---|---|---|---|"]
    for name, r in rt.items():
        lines.append(f"| {name.replace('_', ' ')} | {r['n']} | {r['top1']:.2f} | "
                     f"{r['top2']:.2f} | {r['class_top1']:.2f} |")
    lines += ["", "Presence AUROC per mechanism: " + ", ".join(
        f"`{k}` {v['auroc']:.2f}" for k, v in sim["presence"].items()) + ".", "",
        "The test's contrast on simulations, one window per pair: "
        f"`g_nmda` blocked at least five-fold alone ({test['n_treated']} pairs) against "
        f"no mechanism ({test['n_null']} pairs): AUROC {test['auroc_key_window']:.2f}; "
        f"top-1 `g_nmda` {test['top1_treated']:.2f} against {test['top1_null']:.2f}; "
        f"called above 0.5 {test['called_key_treated']:.2f} against "
        f"{test['called_key_null']:.2f}.",
        "",
        f"With a co-shift allowed (`g_nmda` down at least two-fold, at most one "
        f"other mechanism moving; {co['n_treated']} pairs): AUROC "
        f"{co['auroc_key_window']:.2f}. Both clear the stop rule "
        f"(`docs/STOP_RULE_v3.md`, bars {STOP_RULE['alone_min_auroc']:.2f} and "
        f"{STOP_RULE['coshift_min_auroc']:.2f}); every attempt is in "
        "`results/v3/stop_rule_attempts.json`:",
        "",
        "| Attempt | Twin | Alone | Co-shift | Passed |", "|---|---|---|---|---|"]
    lines += [f"| {k + 1} | `{a['digest'][:12]}` | {a['alone']['auroc_key_window']:.3f} | "
              f"{a['coshift']['auroc_key_window']:.3f} | {'yes' if a['passed'] else 'no'} |"
              for k, a in enumerate(attempts)]  # noqa: B020
    lines += [
        "",
        "The first twin, trained on 144,000 simulated sister pairs, missed the first bar. "
        "Its presence heads stopped improving after about 16 epochs, so a second bank "
        "of 144,000 pairs (new seed, same design and drift) was simulated and the twin "
        "retrained on both. The bars were not moved. The held-out simulations grow with "
        "the bank, so the two attempts are scored on different held-out sets.",
        "",
        f"The twin on the real untreated A-C sister pairs at 10 to 14 days "
        f"({ac['n_preps']} preparations, {ac['n_windows']} windows, excluded from "
        f"scoring): median presence of `g_nmda` {ac['median_p_key']:.2f}; called above "
        f"0.5 in {ac['called_key']:.2f} of preparations. Top-1 counts: "
        + ", ".join(f"`{k}` {v}" for k, v in ac["top1_counts"].items()) + ".",
        "",
        "The recorded test has more between-preparation variation than the "
        "simulations, the drug acted for days rather than minutes, and a "
        "preparation may compensate, so the recorded AUROC is expected below the "
        "simulated one; the bar of 0.70 sits under it for that reason. With "
        f"{groups['n']['treated']} treated and {groups['n']['null_matched']} "
        "genotype-matched null preparations, an observed AUROC of 0.70 carries a "
        "95 percent interval of roughly 0.56 to 0.84, so the bar is testable with "
        "this sample."]
    draft = (ROOT / "scripts" / "templates" / "PREREGISTRATION_v3.draft.md").read_text(encoding="utf-8")
    text = (draft.replace("{DATE}", datetime.date.today().strftime("%d %B %Y"))
            .replace("{DRIFT}", f"{drift['chosen']:.2f}")
            .replace("{SIMULATION}", "\n".join(lines))
            .replace("{N_TREATED}", str(groups["n"]["treated"]))
            .replace("{N_NULL_MATCHED}", str(groups["n"]["null_matched"]))
            .replace("{N_NULL_POOLED}", str(groups["n"]["null_pooled"]))
            .replace("{N_FOUR}", str(len(groups["excluded_four_array_preps"])))
            .replace("{GROUPS}", json.dumps(groups["by_genotype"]))
            .replace("{SPEC}", json.dumps(spec, indent=1)))
    md.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(md.read_bytes()).hexdigest()
    rec.write_text(f"{digest}  PREREGISTRATION_v3.md\n")
    print("wrote PREREGISTRATION_v3.md", digest)
    print(json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
