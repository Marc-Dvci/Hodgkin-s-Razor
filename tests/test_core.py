"""Tests that hold the guarantees the results depend on."""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import features as F, nde, params as P, report, shift as SH
from hodgkins_razor import simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
needs_gpu = pytest.mark.skipif(not S.available(), reason="needs a CUDA device")


# ---------------------------------------------------------------- parameters
def test_transform_round_trip():
    rng = np.random.default_rng(0)
    th = P.sample_prior(500, rng)
    assert np.allclose(P.from_unit(P.to_unit(th)), th)
    assert (th >= P.LO - 1e-9).all() and (th <= P.HI + 1e-9).all()


def test_nuisances_are_not_shiftable():
    for key in ("noise", "p_conn", "f_inh", "p_detect", "elec_het", "u_asyn"):
        assert not P.PARAMS[P.index(key)].shiftable, key
    for key in ("g_ampa", "g_nmda", "g_gaba", "g_na"):
        assert P.PARAMS[P.index(key)].shiftable, key


def test_shift_stays_in_the_box_and_labels_match_what_was_run():
    rng = np.random.default_rng(1)
    theta_c = P.sample_prior(400, rng)
    delta, active = SH.sample_shift(400, rng)
    theta_t, realised = SH.apply_shift(theta_c, delta)
    assert (theta_t >= P.LO - 1e-9).all() and (theta_t <= P.HI + 1e-9).all()
    # The recorded shift is exactly the difference that was simulated.
    assert np.allclose(P.to_unit(theta_t) - P.to_unit(theta_c), realised, atol=1e-9)
    # Non-shiftable columns never move.
    fixed = np.setdiff1d(np.arange(P.N_PARAM), P.SHIFT_IDX)
    assert np.allclose(realised[:, fixed], 0.0)
    # Inactive entries stay inside the narrow component.
    inactive = realised[:, P.SHIFT_IDX][~active]
    assert np.abs(inactive).max() < 10 * SH.INACTIVE_SCALE


def test_delta_z_round_trip():
    d = np.linspace(-3, 3, 101)
    assert np.allclose(nde.z_to_delta(nde.delta_to_z(d)), d, atol=1e-9)


# ------------------------------------------------------------------ features
def _toy_events(seed: int = 0, duration: float = 30.0, n_elec: int = 16):
    rng = np.random.default_rng(seed)
    t, e = [], []
    for start in np.arange(1.0, duration - 1.0, 3.0):
        for _ in range(rng.integers(40, 90)):
            t.append(start + abs(rng.normal(0, 0.05)))
            e.append(rng.integers(0, n_elec))
    t += list(rng.uniform(0, duration, 200))
    e += list(rng.integers(0, n_elec, 200))
    ev = np.stack([np.array(e, float), np.array(t, float)], axis=1)
    return ev[ev[:, 1] < duration]


def test_features_are_order_invariant():
    ev = _toy_events()
    a = F.compute(ev, 16, 30.0)
    rng = np.random.default_rng(3)
    b = F.compute(ev[rng.permutation(ev.shape[0])], 16, 30.0)
    assert np.allclose(a, b, atol=1e-9)


def test_one_feature_path_for_simulated_and_recorded_events():
    """The same array must give the same vector however it reached the function.

    This is the guarantee that a difference between a simulation and a
    recording cannot come from two feature implementations.
    """
    ev = _toy_events(7)
    from_file = F.compute(np.asarray(ev.tolist(), dtype=float), 16, 30.0)
    from_sim = F.compute(ev.astype(np.float64), 16, 30.0)
    assert np.array_equal(from_file, from_sim)


def test_silent_recording_is_a_valid_observation():
    f = F.compute(np.zeros((0, 2)), 16, 60.0)
    assert f.shape == (F.N_FEATURE,)
    assert np.isfinite(f).all()
    assert f[F.NAMES.index("silent_frac")] == 1.0


def test_features_are_finite_on_degenerate_input():
    one = np.array([[0.0, 1.0]])
    two = np.array([[0.0, 1.0], [0.0, 1.0]])
    for ev in (one, two):
        f = F.compute(ev, 16, 60.0)
        assert np.isfinite(f).all()


def test_burst_detector_finds_planted_bursts():
    ev = _toy_events(11, duration=60.0)
    f = F.compute(ev, 16, 60.0)
    # Bursts were planted every 3 s, so the rate must be in a sane band.
    assert 5.0 < f[F.NAMES.index("nbr")] < 30.0
    assert f[F.NAMES.index("psib")] > 20.0


# ----------------------------------------------------------------- simulator
@needs_gpu
def test_simulator_is_deterministic():
    sim = S.Simulator()
    th = P.sample_prior(8, np.random.default_rng(5))
    a = sim.run(th, duration_s=15.0, transient_s=3.0, seed=42)
    b = sim.run(th, duration_s=15.0, transient_s=3.0, seed=42)
    for i in range(8):
        assert np.array_equal(a.times[i], b.times[i])
        assert np.array_equal(a.elecs[i], b.elecs[i])


@needs_gpu
def test_a_different_seed_changes_the_recording():
    sim = S.Simulator()
    th = P.sample_prior(4, np.random.default_rng(6))
    a = sim.run(th, duration_s=15.0, transient_s=3.0, seed=1)
    b = sim.run(th, duration_s=15.0, transient_s=3.0, seed=2)
    assert any(not np.array_equal(a.times[i], b.times[i]) for i in range(4))


def _living(sim, n: int = 1, seed: int = 3, batch: int = 384,
            duration: float = 30.0) -> np.ndarray:
    """Parameter sets whose simulated baseline is a living culture.

    Most of the prior is silent or saturated, so a test that needs a working
    network has to find one the same way the bank generator does.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_regime import is_living
    rng = np.random.default_rng(seed)
    out = []
    for attempt in range(8):
        th = P.sample_prior(batch, rng)
        r = sim.run(th, duration_s=duration, transient_s=5.0,
                    seed=seed * 100 + attempt)
        for k in range(batch):
            if is_living(F.regime_stats(r.as_events(k), 16, duration)):
                out.append(th[k])
                if len(out) >= n:
                    return np.array(out)
    raise AssertionError("no living baseline found")


@needs_gpu
def test_blocking_sodium_silences_the_network():
    """The model must reproduce the one effect every MEA laboratory knows.

    The cultures are centres of the fitted domain proposals, the preparations
    the twin is trained on. A culture drawn from the whole prior can sit at
    maximal drive and noise, where membrane noise alone fires a cell with no
    sodium current worth the name, and no laboratory records such a culture.
    """
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_domain import from_cube
    dom = json.loads((ROOT / "models" / "domain.json").read_text())
    base = np.concatenate([from_cube(np.array(dom["views"][v]["proposal"]["centres"][:4]))
                           for v in ("grid16", "grid12")])
    th = np.repeat(base, 2, axis=0)
    th[1::2, P.index("g_na")] = P.LO[P.index("g_na")]
    r = S.Simulator().run(th, duration_s=30.0, transient_s=5.0, seed=3)
    hi = np.array([F.compute(r.as_events(k), 16, 30.0)[0] for k in range(0, len(th), 2)])
    lo = np.array([F.compute(r.as_events(k), 16, 30.0)[0] for k in range(1, len(th), 2)])
    alive = hi > 0.2
    assert alive.sum() >= 4, "too few domain cultures produced activity"
    ratio = lo[alive] / hi[alive]
    # The criterion of scripts/pharmacology_check.py: the median culture keeps
    # under a tenth of its firing, and most are silent.
    assert np.median(ratio) < 0.1, f"a sodium block left {ratio.round(2)} of the firing"
    assert np.mean(ratio < 0.01) >= 0.5, f"a sodium block left {ratio.round(2)} of the firing"


@needs_gpu
def test_pair_mode_shares_the_nuisances():
    """Two halves of a pair with identical parameters must agree closely."""
    sim = S.Simulator()
    th = np.repeat(P.sample_prior(6, np.random.default_rng(9)), 2, axis=0)
    r = sim.run(th, duration_s=40.0, transient_s=5.0, seed=8, pair=True)
    diffs = []
    for k in range(6):
        a = F.compute(r.as_events(2 * k), 16, 40.0)[0]
        b = F.compute(r.as_events(2 * k + 1), 16, 40.0)[0]
        if max(a, b) > 1.0:
            diffs.append(abs(a - b) / max(a, b))
    assert diffs, "no active pair to compare"
    assert np.median(diffs) < 0.35


@needs_gpu
def test_pair_mode_rejects_an_odd_batch():
    sim = S.Simulator()
    with pytest.raises(ValueError):
        sim.run(P.sample_prior(3, np.random.default_rng(0)), duration_s=5.0,
                transient_s=1.0, pair=True)


@needs_gpu
def test_variant_kinetics_produce_a_different_recording():
    """The guard is only meaningful if the variant is genuinely out of reach."""
    sim = S.Simulator()
    th = _living(sim, n=8, seed=12)
    n = th.shape[0]
    a = sim.run(th, duration_s=25.0, transient_s=5.0, seed=4)
    b = S.Simulator(kinetics={"TAU_AMPA": 20.0, "TAU_GABA": 40.0}).run(
        th, duration_s=25.0, transient_s=5.0, seed=4)
    changed = sum(not np.array_equal(a.times[i], b.times[i]) for i in range(n))
    assert changed >= max(n // 2, 1)


# -------------------------------------------------------------------- report
def _fake_posterior(n=800, seed=0):
    rng = np.random.default_rng(seed)
    theta = P.TLO + rng.random((n, P.N_PARAM)) * (P.THI - P.TLO)
    delta = rng.normal(0, 0.02, (n, P.N_SHIFT))
    delta[:, 4] = rng.normal(-1.5, 0.2, n)
    p = np.full(P.N_SHIFT, 0.05)
    p[4] = 0.95
    return {"theta_c": theta, "delta": delta, "p_active": p, "p_active_raw": p}


def test_report_names_the_mechanism_when_inside_the_model():
    post = _fake_posterior()
    body = report.build(post, {"inside_model": True, "discrepancy": 1.0,
                               "threshold": 5.0}, np.zeros(F.N_FEATURE),
                        np.zeros(F.N_FEATURE))
    assert body["verdict"] == "mechanism_called"
    assert body["called"]


def test_report_names_nothing_when_outside_the_model():
    post = _fake_posterior()
    body = report.build(post, {"inside_model": False, "discrepancy": 99.0,
                               "threshold": 5.0}, np.zeros(F.N_FEATURE),
                        np.zeros(F.N_FEATURE))
    assert body["verdict"] == "outside_model"
    assert "No mechanism is named" in body["sentence"]


def test_report_hash_changes_with_content():
    a = report.build(_fake_posterior(seed=1), None, np.zeros(F.N_FEATURE),
                     np.zeros(F.N_FEATURE))
    b = report.build(_fake_posterior(seed=2), None, np.zeros(F.N_FEATURE),
                     np.zeros(F.N_FEATURE))
    assert a["content_sha256"] != b["content_sha256"]


# ------------------------------------------------------------ preregistration
def test_preregistration_matches_its_hash():
    md = ROOT / "PREREGISTRATION.md"
    recorded = (ROOT / "PREREGISTRATION.sha256").read_text().split()[0]
    assert hashlib.sha256(md.read_bytes()).hexdigest() == recorded


def test_answer_key_parses_to_the_expected_compounds():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate
    key = evaluate.read_key()["key"]
    assert key["CNQX"] == ("g_ampa", "down")
    assert key["Gabazine"] == ("g_gaba", "down")
    assert key["GABA"] == ("g_gaba", "up")
    assert key["Control"] == (None, None)


# ------------------------------------------------------- the regime criterion
def test_regime_stats_matches_the_full_feature_computation():
    """A recording judged living in screening must be judged living in training."""
    for seed in (0, 5, 9):
        ev = _toy_events(seed, duration=40.0)
        full = F.compute(ev, 16, 40.0)
        quick = F.regime_stats(ev, 16, 40.0)
        for k in ("mfr", "active_frac", "nbr", "psib"):
            assert abs(full[F.NAMES.index(k)] - quick[k]) < 1e-9, k


def test_regime_stats_handles_an_empty_recording():
    q = F.regime_stats(np.zeros((0, 2)), 16, 60.0)
    assert set(q) == {"mfr", "active_frac", "nbr", "psib"}
    assert all(v == 0.0 for v in q.values())


def test_sttc_is_bounded():
    for seed in (1, 2, 3):
        ev = _toy_events(seed, duration=30.0)
        f = F.compute(ev, 16, 30.0)
        for k in ("sttc_mean", "sttc_sd"):
            assert -1.0001 <= f[F.NAMES.index(k)] <= 1.0001, k


def test_discriminative_features_respond_to_burst_structure():
    """Bursting and tonic recordings at the same rate must not look the same."""
    rng = np.random.default_rng(4)
    n = 2400
    bursty = _toy_events(2, duration=60.0)
    flat = np.stack([rng.integers(0, 16, n).astype(float),
                     rng.uniform(0, 60, n)], axis=1)
    a = F.compute(bursty, 16, 60.0)
    b = F.compute(flat, 16, 60.0)
    assert a[F.NAMES.index("fano_250ms")] > b[F.NAMES.index("fano_250ms")]
    assert a[F.NAMES.index("psib")] > b[F.NAMES.index("psib")]


def test_feature_names_are_unique_and_sized():
    assert len(set(F.NAMES)) == len(F.NAMES)
    assert F.N_FEATURE == len(F.NAMES)
    assert F.compute(_toy_events(0), 16, 30.0).shape == (F.N_FEATURE,)


def test_report_gives_a_mechanism_class():
    body = report.build(_fake_posterior(), None, np.zeros(F.N_FEATURE),
                        np.zeros(F.N_FEATURE))
    assert len(body["classes"]) == len(P.CLASS_NAMES)
    assert abs(sum(c["probability"] for c in body["classes"]) - 1.0) < 1e-6
    # The planted shift is on g_ampa, so excitatory transmission must lead.
    assert body["classes"][0]["name"] == "excitatory transmission"


def test_every_shiftable_mechanism_has_a_class():
    for i in P.SHIFT_IDX:
        assert P.KEYS[i] in P.CLASS_OF


# ----------------------------------------------------------------- the guard
def test_discrepancy_sees_a_mismatch_confined_to_a_few_features():
    """A recording that fails on a handful of statistics must score high.

    Averaging over every feature hid exactly this case, which is why the
    statistic takes the worst few instead.
    """
    from hodgkins_razor import ppc
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 1, (48, F.N_FEATURE))
    obs = np.zeros(F.N_FEATURE)
    near = ppc.discrepancy(obs, obs, pred, pred)
    far = obs.copy()
    far[:6] = 30.0
    high = ppc.discrepancy(far, obs, pred, pred)
    assert high >= near + 6, (near, high)


def test_discrepancy_is_low_for_a_recording_the_model_predicts():
    from hodgkins_razor import ppc
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 1, (48, F.N_FEATURE))
    obs = pred.mean(axis=0)
    assert ppc.discrepancy(obs, obs, pred, pred) < 0.1 * F.N_FEATURE


def test_held_out_indices_match_the_bank_they_came_from():
    """A model trained on a restricted bank must be scored on that same bank.

    The indices saved at training time are positions, so applying them to an
    unrestricted bank silently scores different records.
    """
    import json
    meta_path = ROOT / "models" / "twin" / "meta.json"
    idx_path = ROOT / "models" / "twin" / "val_index.npy"
    if not meta_path.exists() or not idx_path.exists():
        pytest.skip("no trained twin in the repository")
    meta = json.loads(meta_path.read_text())
    idx = np.load(idx_path)
    assert idx.max() < meta["pairs"], (int(idx.max()), meta["pairs"])


# ------------------------------------------------------------------ version 2
def test_shift_direction_is_drawn_where_the_baseline_leaves_room():
    """A mechanism already at its floor can only be shifted up.

    Version 1 drew the direction blindly and clipped, so 28 percent of active
    tonic-inhibition labels described shifts the simulation never ran.
    """
    rng = np.random.default_rng(4)
    theta_c = P.sample_prior(2000, rng)
    j = P.index("g_tonic_inh")
    theta_c[:, j] = P.LO[j]
    delta, active = SH.sample_shift(2000, rng, theta_c=theta_c)
    _, realised, active2 = SH.apply_shift(theta_c, delta, active)
    k = list(P.SHIFT_IDX).index(j)
    assert (realised[active2[:, k], j] > 0).all()
    floor = SH.min_effect()
    got = np.abs(realised[:, P.SHIFT_IDX])
    assert (got[active2] >= 0.5 * np.broadcast_to(floor, got.shape)[active2]).all()


def test_a_shift_the_clipping_removed_is_labelled_inactive():
    theta_c = P.sample_prior(1, np.random.default_rng(0))
    j = P.index("g_ampa")
    theta_c[0, j] = P.HI[j]
    delta = np.zeros((1, P.N_PARAM))
    delta[0, j] = np.log(5.0)
    active = np.zeros((1, P.N_SHIFT), dtype=bool)
    active[0, list(P.SHIFT_IDX).index(j)] = True
    _, realised, act = SH.apply_shift(theta_c, delta, active)
    assert abs(realised[0, j]) < 1e-9
    assert not act.any()


def test_views_apply_the_recording_system_electrodes_and_dead_time():
    ev = np.array([[0, 0.0], [0, 0.0005], [0, 0.003], [3, 0.001],
                   [5, 0.0011], [5, 0.0012]])
    g16, n16 = S.view_events(ev, "grid16")
    g12, n12 = S.view_events(ev, "grid12")
    assert n16 == 16 and n12 == 12
    # 2 ms dead time removes the second spike on electrode 0.
    assert g16.shape[0] == 4
    # Corners 0 and 3 are reference positions on the 12-electrode plate;
    # electrode 5 keeps one spike under a 0.3 ms dead time.
    assert g12.shape[0] == 1 and g12[0, 0] == 3


def test_sttc_matches_a_brute_force_reference():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 10, 50))
    b = np.sort(rng.uniform(0, 10, 60))
    dt = 0.02
    grid = np.linspace(0, 10, 200001)
    ta = np.mean(np.any(np.abs(grid[:, None] - a[None, :]) <= dt, axis=1))
    tb = np.mean(np.any(np.abs(grid[:, None] - b[None, :]) <= dt, axis=1))
    pa = np.mean([np.min(np.abs(b - x)) <= dt for x in a])
    pb = np.mean([np.min(np.abs(a - x)) <= dt for x in b])
    want = 0.5 * ((pa - tb) / (1 - pa * tb) + (pb - ta) / (1 - pb * ta))
    got, _ = F._sttc(np.r_[np.zeros(50), np.ones(60)], np.r_[a, b], 2, 10.0, dt)
    assert abs(got - want) < 1e-3


def test_culture_split_never_shares_a_culture():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import split_groups
    group = np.repeat(np.arange(300), 6)
    va, ca, tr = split_groups(group, 0.1, 0.1, np.random.default_rng(0))
    sets = [set(group[i]) for i in (va, ca, tr)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(va) + len(ca) + len(tr) == group.size


def test_conditional_twin_reports_an_effect_per_mechanism():
    flow, presence = nde.Twin.build(device="cpu", transforms=2, hidden=32, depth=1,
                                    conditional=True, ensemble=2)
    n_ctx = 3 * F.N_FEATURE
    scaler = nde.Standardiser(np.zeros(n_ctx), np.ones(n_ctx))
    twin = nde.Twin(flow, presence, scaler, device="cpu",
                    meta={"conditional": True, "ensemble": 2})
    post = twin.posterior(np.ones(F.N_FEATURE), np.ones(F.N_FEATURE), n_samples=400)
    assert post["effect_given_active"].shape[1] == P.N_SHIFT
    rows = nde.summarise(post)
    assert len(rows) == P.N_SHIFT


def test_typicality_is_larger_far_from_the_reference():
    from hodgkins_razor.ppc import Typicality
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (2000, 12))
    t = Typicality(ref)
    near, far = t.score(rng.normal(0, 1, (50, 12))), t.score(rng.normal(6, 1, (50, 12)))
    assert near.max() < far.min()


def test_doorn_treated_windows_are_locked_without_the_second_preregistration(tmp_path, monkeypatch):
    from hodgkins_razor import doorn as D
    monkeypatch.setattr(D, "ROOT", tmp_path)
    assert not D._unblinded()
    (tmp_path / "PREREGISTRATION_v2.md").write_text("x")
    (tmp_path / "PREREGISTRATION_v2.sha256").write_text("0" * 64)
    assert not D._unblinded()
    (tmp_path / "PREREGISTRATION_v2.sha256").write_text(
        hashlib.sha256(b"x").hexdigest())
    assert D._unblinded()


def test_chip_channel_electrodes_record_exactly_the_projecting_neurons():
    from hodgkins_razor import chip as C
    rng = np.random.default_rng(0)
    theta = P.sample_prior(3, rng)
    chips = np.array([[0.3, 1.0, 1.0, 0.5], [0.3, 0.5, 1.0, 0.5], [0.02, 0.9, 1.0, 0.5]])
    st = C.wiring(theta, chips, np.random.default_rng(1))
    comp = C.GEOMETRY.compartment
    for b in range(3):
        w, e = st["w"][b], st["elec"][b]
        same = comp[:, None] == comp[None, :]
        crosses = ((w > 0) & ~same).any(axis=1)
        # Every neuron with an axon across the channel is on a channel electrode.
        assert set(np.flatnonzero(crosses)) <= set(np.flatnonzero(e >= C.ELEC_FWD))
        assert (comp[e == C.ELEC_FWD] == 0).all() and (comp[e == C.ELEC_BWD] == 1).all()
    # A perfect diode has no backward axons.
    assert (st["elec"][0] == C.ELEC_BWD).sum() == 0


def test_mateus_propagation_counts_planted_sequences():
    from hodgkins_razor import mateus as M
    spikes = {}
    t0 = np.arange(1.0, 50.0, 1.0)
    for k, row in enumerate(M.CHANNEL_ROWS):
        spikes[f"A{row}"] = t0 + 0.0004 * k            # 30 forward events
    for k, row in enumerate(reversed(M.CHANNEL_ROWS)):
        spikes[f"B{row}"] = t0[:10] + 0.3 + 0.0004 * k  # 10 backward events
    rec = M.Recording(path=ROOT, experiment="x", design="rams", chip="1", div=12,
                      duration=60.0, spikes=spikes)
    out = M.propagation(rec)
    assert out["down"] == len(t0) and out["up"] == 10


def test_v2_metrics_on_synthetic_wells():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_v2 as E
    keys = E.SHIFT_KEYS
    rng = np.random.default_rng(0)
    rows = []
    truth = {"A": "u_rel", "B": "g_ampa"}
    for comp, want in truth.items():
        for w in range(4):
            for k in range(3):
                p = rng.uniform(0, 0.2, len(keys))
                p[keys.index(want)] = 0.9 if w < 3 else 0.1
                rows.append({"plate": "x", "species": "rat" if w % 2 else "hPSC",
                             "well": f"{comp}{w}", "compound": comp,
                             "p_active": p.tolist(), "effect_med": np.full(len(keys), 0.5).tolist(),
                             "effect_lo": [0] * len(keys), "effect_hi": [1] * len(keys),
                             "x_base": rng.normal(size=40).tolist(),
                             "x_treat": rng.normal(size=40).tolist()})
    for w in range(3):
        rows.append({"plate": "x", "species": "rat", "well": f"C{w}", "compound": "Control",
                     "p_active": [0.05] * len(keys), "effect_med": [0] * len(keys),
                     "effect_lo": [0] * len(keys), "effect_hi": [0] * len(keys),
                     "x_base": [0.0] * 40, "x_treat": [0.0] * 40})
    wells = E.by_well(rows)
    key = {"A": {"accept": ["u_rel", "tau_d"], "direction": "up"},
           "B": {"accept": ["g_ampa"], "direction": "any"},
           "Control": {"control": True}}
    m = E.mechanism_metrics(wells, key)
    assert m["n_wells"] == 8 and m["top1_hits"] == 6
    assert m["control_false_mechanism_rate"] == 0.0
    assert m["detection_auroc"] > 0.5
    t = E.transfer(rows, wells)
    assert set(t) == {"raw", "twin"}
