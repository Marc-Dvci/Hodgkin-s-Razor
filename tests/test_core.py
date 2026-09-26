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
    for key in ("noise", "p_conn", "f_inh", "p_detect", "elec_het"):
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


@needs_gpu
def test_blocking_sodium_silences_the_network():
    """The model must reproduce the one effect every MEA laboratory knows."""
    sim = S.Simulator()
    base = dict(noise=4.0, g_na=1.4, g_kdr=1.0, g_ahp=5.0, g_ampa=0.35,
                g_nmda=0.25, g_gaba=1.0, p_conn=0.25, f_inh=0.2, tau_d=500.0,
                u_rel=0.2, i_drive=18.0, p_detect=0.8, elec_het=0.3)
    th = np.array([[base[k] for k in P.KEYS], [base[k] for k in P.KEYS]])
    th[1, P.index("g_na")] = P.PARAMS[P.index("g_na")].lo
    r = sim.run(th, duration_s=30.0, transient_s=5.0, seed=3)
    hi = F.compute(r.as_events(0), 16, 30.0)[0]
    lo = F.compute(r.as_events(1), 16, 30.0)[0]
    assert hi > 0.2, "control condition produced no activity"
    assert lo < 0.2 * hi, f"a sodium block left {lo:.2f} against {hi:.2f}"


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
    th = P.sample_prior(12, np.random.default_rng(12))
    a = S.Simulator().run(th, duration_s=25.0, transient_s=5.0, seed=4)
    b = S.Simulator(kinetics={"TAU_AMPA": 20.0, "TAU_GABA": 40.0}).run(
        th, duration_s=25.0, transient_s=5.0, seed=4)
    changed = sum(not np.array_equal(a.times[i], b.times[i]) for i in range(12))
    assert changed >= 6


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
