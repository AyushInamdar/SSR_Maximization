"""Tests. Run:  python -m pytest tests   (or)   python tests/test_core.py"""
import os
import sys
import warnings
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from otfs_isac_sim.config import SimParams, C_LIGHT                # noqa: E402
from otfs_isac_sim import waveform as wf                           # noqa: E402
from otfs_isac_sim import crb as crbmod                            # noqa: E402
from otfs_isac_sim.geometry import steering, Scenario              # noqa: E402
from otfs_isac_sim.an import (build_an_basis, an_allocation, null_projector,  # noqa: E402
                              isotropic_null_an, steering_derivative)
from otfs_isac_sim.crb import (crb_matrix, min_illumination_for_crb, crb_from_illumination,  # noqa: E402
                               crb_closed_form, crb_exact, pulse_moments, flat_moments)
from otfs_isac_sim.simulator import run_episode, build_problem, _truth_estimate  # noqa: E402
from otfs_isac_sim.solvers import SOLVERS, LinearModel, directions  # noqa: E402
from otfs_isac_sim.frameworks import get_frameworks                # noqa: E402
from otfs_isac_sim.metrics import evaluate_true                    # noqa: E402
from otfs_isac_sim.sensing import (sense_target, Estimate, ml_estimate_dd_angle,  # noqa: E402
                                   process_noise)

P = SimParams()


# --------------------------------------------------------------------- waveform
def test_isfft_sfft_roundtrip():
    x = wf.qpsk((P.N, P.M), np.random.default_rng(0))
    assert np.allclose(wf.sfft(wf.isfft(x)), x)
    assert np.allclose(wf.otfs_demodulate(wf.otfs_modulate(x), P.N, P.M), x)


def test_otfs_integer_doppler_is_dd_shift():
    for k in (1, 3, -2):
        assert wf.check_otfs_dd_shift(P, k) < 1e-10


def test_ofdm_ici_closed_form_matches_link_level():
    for nu in (100.0, 400.0):
        emp = wf.measure_ofdm_sinr_loss(nu, P)
        assert abs(emp - np.sinc(nu / P.delta_f) ** 2) < 2e-3


# -------------------------------------------------------------------------- CRB
def _legacy_crb(snr_el, theta, p):
    """The former crude formulas (flat spectrum / uniform in time)."""
    snr_arr = p.Nr * snr_el
    b2 = p.bandwidth ** 2 / 12.0
    t2 = p.frame_duration ** 2 / 12.0
    return np.diag([1 / (8 * np.pi ** 2 * b2 * snr_arr), 1 / (8 * np.pi ** 2 * t2 * snr_arr),
                    6 / (np.pi ** 2 * np.cos(theta) ** 2 * snr_el * p.Nr * (p.Nr ** 2 - 1))])


def test_crb_reduces_to_legacy_formulas_for_flat_pulse():
    """(a) Eq. (66) with the moments of a flat-spectrum / uniform-in-time pulse equals the
    former crude formulas, and the full 5x5 FIM (no alpha = beta = 0 assumption) agrees."""
    pf = replace(P, crb_moments="flat")
    for snr, th in [(10.0, 0.3), (1e4, -0.8), (1e6, 0.0)]:
        assert np.allclose(crb_matrix(snr, th, pf), _legacy_crb(snr, th, pf), rtol=1e-10, atol=0)
        m = flat_moments(pf)
        Cm, Ce = crb_matrix(snr, th, pf), crb_exact(snr, th, 2e-7, m, pf.Nr)
        assert np.allclose(np.diag(Ce), np.diag(Cm), rtol=1e-8)
        assert np.abs(Ce - np.diag(np.diag(Ce))).max() < 1e-9 * np.abs(np.diag(Ce)).max()


def test_waveform_moments_match_the_otfs_frame():
    """E0 = sum|s|^2/fs (Parseval); the frame is (nearly) flat in spectrum and time, so the
    waveform moments are close to the flat ones, and Appendix A's assumptions hold."""
    s = wf.probe_waveform(P)
    m = pulse_moments(s, P.fs)
    f = flat_moments(P)
    assert abs(m.E0 / (np.sum(np.abs(s) ** 2) / P.fs) - 1) < 1e-3
    assert abs(m.omega_norm / f.omega_norm - 1) < 0.05
    assert abs(m.tc_norm / f.tc_norm - 1) < 0.05
    d = crbmod.coupling_diagnostics(m, gamma=3e-7, Nr=P.Nr)
    assert d["a1_rel"] < 1e-6                       # a1 = 0 (periodic frame)
    assert d["max_abs_corr"] < 0.05                 # delay-Doppler-angle coupling negligible
    assert d["diag_rel_gap"] < 5e-3                 # Eq. (66) diagonal == exact 5x5 FIM


def test_ml_estimates_approach_crb():
    """(b) Monte-Carlo local ML estimate of (gamma, omega, theta) from the simulated OTFS
    echo (nuisance h concentrated out): MSE / CRB ~ 1 at high SNR and the estimator is
    unbiased to within a fraction of the CRB std."""
    s0, fs = wf.probe_waveform(P), P.fs
    m = pulse_moments(s0, fs)
    Nr, th, g, w = 16, 0.3, 2.3e-7, 150.0
    snr = 10 ** (30 / 10)                           # per-element post-MF SNR |h|^2 E0 / N0
    sig2 = (m.E0 / snr) * fs                        # per-sample noise variance N0 * fs
    C = crb_closed_form(snr, th, Nr, m.omega_norm, m.tc_norm)
    sd = np.sqrt(np.diag(C))
    L = s0.size
    t = np.arange(L) / fs
    mu = np.outer(steering(th, Nr)[:, 0], wf.bandlimited_shift(s0, fs, g) * np.exp(1j * 2 * np.pi * w * t))
    rng = np.random.default_rng(5)
    cell = np.array([1 / P.bandwidth, 1 / P.frame_duration, 2 / Nr])
    errs = []
    for _ in range(80):
        z = np.sqrt(sig2 / 2) * (rng.standard_normal((Nr, L)) + 1j * rng.standard_normal((Nr, L)))
        r = np.exp(1j * rng.uniform(0, 2 * np.pi)) * mu + z
        x0 = np.array([g, w, th]) + rng.uniform(-0.1, 0.1, 3) * cell     # coarse search leaves < 0.1 cell
        errs.append(ml_estimate_dd_angle(r, s0, fs, x0, sd, max_iter=200) - np.array([g, w, th]))
    errs = np.array(errs)
    ratio = np.mean(errs ** 2, 0) / np.diag(C)
    assert np.all(ratio > 0.6) and np.all(ratio < 1.6), ratio        # 80 trials: +/- ~16 %
    assert np.all(np.abs(errs.mean(0)) < 0.4 * sd)


def test_crb_scales_with_Nr_and_snr():
    """(c) var(gamma), var(omega) ~ 1/(Nr snr); var(theta) ~ 1/(Nr (Nr^2-1) snr cos^2);
    CRB ~ 1/illumination."""
    th, snr = 0.4, 123.0
    C = crb_matrix(snr, th, P)
    P2 = replace(P, Nr=2 * P.Nr)
    C2 = crb_matrix(snr, th, P2)
    assert np.isclose(C2[0, 0] / C[0, 0], 0.5) and np.isclose(C2[1, 1] / C[1, 1], 0.5)
    k = P.Nr * (P.Nr ** 2 - 1) / (P2.Nr * (P2.Nr ** 2 - 1))
    assert np.isclose(C2[2, 2] / C[2, 2], k)
    C10 = crb_matrix(10 * snr, th, P)
    assert np.allclose(np.diag(C10), np.diag(C) / 10)
    assert np.isclose(C[2, 2] / crb_matrix(snr, 0.0, P)[2, 2], 1 / np.cos(th) ** 2)
    c1 = crb_from_illumination(th, 40.0, 1e-3, P)
    c2 = crb_from_illumination(th, 40.0, 2e-3, P)
    assert np.allclose(c2, c1 / 2)
    assert np.all(np.linalg.eigvalsh(C) > 0) and np.allclose(C, np.diag(np.diag(C)))


def test_min_illumination_meets_crb_requirement():
    th, d = 0.4, 40.0
    illum = min_illumination_for_crb(th, d, P)
    C = crb_from_illumination(th, d, illum * 1.0001, P)
    assert np.all(np.diag(C) <= np.diag(P.crb_req) * (1 + 1e-6))


# ------------------------------------------------------------- tracking filter
def test_information_filter_update_equations():
    th, d = 0.3, 45.0
    pos = np.array([d * np.sin(th), d * np.cos(th)])
    speed = 12.0
    om = -2 * speed * np.sin(th) * P.fc / C_LIGHT
    prior = Estimate(pos=pos, speed=speed, theta=th, dist=d,
                     C=np.diag([(2e-9) ** 2, 0.5 ** 2, (3e-5) ** 2]))
    est = sense_target(th, d, om, 1e-3, prior, P, np.random.default_rng(0))
    Pp, Cm = np.diag(prior.C), np.diag(est.C_meas)
    assert np.allclose(np.diag(est.C), 1 / (1 / Pp + 1 / Cm), rtol=1e-12)
    assert np.all(np.diag(est.C) < Pp) and np.all(np.diag(est.C) < Cm)
    # tracking OFF -> the region is the raw single-frame CRB
    est0 = sense_target(th, d, om, 1e-3, prior, replace(P, track_accumulate=False), np.random.default_rng(0))
    assert np.allclose(est0.C, est0.C_meas)


def test_region_shrinks_and_covers_over_time():
    r = run_episode(P, "otfs_isac", "grid_split", n_frames=40, seed=0)
    area = r["region_area_e"]
    assert area[-10:].mean() < 0.5 * area[:3].mean()               # E_j shrinks over time
    assert r["cover_e"].mean() > 0.85                              # ~95 % nominal chi^2_3 coverage
    r0 = run_episode(replace(P, track_accumulate=False), "otfs_isac", "grid_split", n_frames=40, seed=0)
    assert r0["region_area_e"][-10:].mean() > 0.5 * r0["region_area_e"][:3].mean()   # never shrinks without fusion


def test_process_noise_positive():
    q = process_noise(12.0, 0.3, 45.0, P)
    assert q.shape == (3,) and np.all(q > 0)


# ------------------------------------------------------------------------- AN
def test_an_su_construction():
    """S = V R_nbar V^H (Su et al. Eqs. 19-20): PSD, zero leakage at the predicted user
    angles, power budget exact, V a projector, isotropic atom = V."""
    th_u = np.deg2rad([-30, 20, 14, 60])
    basis = build_an_basis(th_u, [np.deg2rad([-17, -16]), np.deg2rad([50])], P.Nt)
    assert basis.has_iso
    V = null_projector(th_u, P.Nt)
    assert np.allclose(V @ V, V) and np.allclose(V, V.conj().T)
    for iota in (0.0, 0.4, 1.0):
        q = an_allocation(basis, P.P_an_max, iota, "per_eve")
        S = basis.covariance(q)
        A = steering(th_u, P.Nt)
        leak = np.real(np.einsum("ip,ij,jp->p", A.conj(), S, A))
        assert np.all(np.abs(leak) < 1e-9 * np.real(np.trace(S)))
        assert np.linalg.eigvalsh((S + S.conj().T) / 2).min() > -1e-12 * np.real(np.trace(S))
        assert abs(basis.power(q) - P.P_an_max) < 1e-12 * max(P.P_an_max, 1e-30) + 1e-18
        assert abs(np.real(np.trace(S)) - P.P_an_max) < 1e-9 * P.P_an_max
        # S = V R_nbar V^H with R_nbar = sum_m q_m a a^H + q_0 I
        Rn = np.zeros((P.Nt, P.Nt), complex)
        for qm, o, t in zip(q, basis.owner, basis.theta):
            if o >= 0:
                a = steering(t, P.Nt)[:, 0]
                Rn += qm * np.outer(a, a.conj())
            else:
                Rn += qm * np.eye(P.Nt)
        assert np.allclose(V @ Rn @ V.conj().T, S, atol=1e-12 * np.real(np.trace(S)))
    S_iso = isotropic_null_an(th_u, P.Nt, P.P_an_max)
    assert abs(np.real(np.trace(S_iso)) - P.P_an_max) < 1e-12 * P.P_an_max


def test_an_derivative_nulling_reduces_leakage():
    th_u = np.deg2rad([-30, 20, 14, 60])
    d = 1e-3                                              # angle prediction error (rad)
    A_true = steering(th_u + d, P.Nt)
    out = {}
    for dn in (False, True):
        S = isotropic_null_an(th_u, P.Nt, 1.0, deriv_null=dn)
        out[dn] = np.real(np.einsum("ip,ij,jp->p", A_true.conj(), S, A_true)).max()
    assert out[True] < 1e-3 * out[False]


# ------------------------------------------------------------------- rates etc.
def _truth_problem(seed=3):
    scen = Scenario.from_params(P, np.random.default_rng(seed))
    preds = [_truth_estimate(scen, i) for i in range(scen.P)]
    for e in preds[scen.K:]:
        e.C = np.diag([1e-17, 1.0, 1e-5])
    return scen, preds, build_problem(preds, scen.K, get_frameworks(P)["otfs_isac"], P)


def test_rates_match_hand_computation():
    """Eqs. (27)-(29) written out literally for random F, S."""
    rng = np.random.default_rng(1)
    scen = Scenario.from_params(P, rng)
    K, J, Nt = scen.K, scen.J, P.Nt
    F = (rng.standard_normal((Nt, K)) + 1j * rng.standard_normal((Nt, K))) * 1e-3
    B = rng.standard_normal((Nt, 3)) + 1j * rng.standard_normal((Nt, 3))
    S = 1e-7 * B @ B.conj().T
    m = evaluate_true(F, S, scen, "otfs", 1.0, P)
    A = steering(scen.theta, Nt)
    for k in range(K):
        a, h = A[:, k], scen.h2[k]
        num = h * abs(a.conj() @ F[:, k]) ** 2
        den = sum(h * abs(a.conj() @ F[:, kk]) ** 2 for kk in range(K) if kk != k) \
            + h * np.real(a.conj() @ S @ a) + P.N0
        assert np.isclose(m["sinr_users"][k], num / den, rtol=1e-9)
        assert np.isclose(m["R_users"][k], np.log2(1 + num / den), rtol=1e-9)
        for j in range(J):
            a, h = A[:, K + j], scen.h2[K + j]
            N = h * abs(a.conj() @ F[:, k]) ** 2
            D = sum(h * abs(a.conj() @ F[:, kk]) ** 2 for kk in range(K) if kk != k) \
                + h * np.real(a.conj() @ S @ a) + P.N0
            assert np.isclose(m["N_eve"][j, k], N, rtol=1e-9) and np.isclose(m["D_eve"][j, k], D, rtol=1e-9)
            assert np.isclose(m["R_eve"][j, k], np.log2(1 + N / D), rtol=1e-9)
    assert np.allclose(m["R_sec"], np.maximum(m["R_users"] - m["R_eve"].max(0), 0))


def test_design_rates_match_true_rates():
    """The rates inside the optimiser (LinearModel, predicted states) are the same
    expressions as metrics.evaluate_true when prediction == truth; so are the constraints'
    ingredients (C1 power, C4 sensing)."""
    scen, preds, prob = _truth_problem()
    K, J = scen.K, scen.J
    for name in ("mrt_an", "zf_an", "grid_split"):
        sol = SOLVERS[name](prob)
        W = sol.F / np.maximum(np.linalg.norm(sol.F, axis=0, keepdims=True), 1e-300)
        lm = LinearModel(prob, W)
        R_u, R_e = lm.rates(sol.p, sol.q)
        m = evaluate_true(sol.F, sol.S, scen, "otfs", prob.prefactor, P)
        assert np.allclose(R_u, m["R_users"], rtol=1e-6, atol=1e-9), name
        # first sample of each Eve's region is its centre = the true state
        idx = np.r_[0, np.cumsum(np.bincount(prob.eve_owner))[:-1]]
        assert np.allclose(R_e[idx], m["R_eve"], rtol=1e-6, atol=1e-9), name
        # C4: beampattern of the linear model == illumination of the design
        c = lm.constraints(sol.p, sol.q)
        illum_c4 = c["C4_sensing"] + P.gamma_sens
        assert np.allclose(illum_c4, m["illum"], rtol=1e-6), name
        tot = np.sum(np.abs(sol.F) ** 2) + np.real(np.trace(sol.S))
        assert np.isclose(P.P_max - tot, c["C1_total_power"], rtol=1e-6, atol=1e-12), name


def test_all_working_solvers_respect_power_constraints():
    scen, preds, prob = _truth_problem()
    for name, fn in SOLVERS.items():
        if name in ("sca_sdr", "drl"):
            continue
        sol = fn(prob)
        tot = np.sum(np.abs(sol.F) ** 2) + np.real(np.trace(sol.S))
        assert tot <= P.P_max * (1 + 1e-6), name
        assert sol.info["an_power"] <= P.P_an_max * (1 + 1e-6), name


def test_slsqp_never_worse_than_its_grid_warm_start():
    scen, preds, prob = _truth_problem(seed=4)
    g, s = SOLVERS["grid_split"](prob), SOLVERS["slsqp"](prob)
    assert (-s.info["violation"], s.info["obj_pred"]) >= (-g.info["violation"], g.info["obj_pred"] - 1e-9)


def test_episode_runs_and_an_leak_is_negligible():
    r = run_episode(P, "otfs_isac", "grid_split", n_frames=12)
    assert r["sum_sec"].shape == (11,) and np.all(np.isfinite(r["sum_sec"]))
    assert np.mean(r["an_leak_users"] / P.N0) < 1e-2        # AN leakage << noise at the users


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
