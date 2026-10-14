"""Sanity tests. Run:  python -m pytest tests   (or)   python tests/test_core.py"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from otfs_isac_sim.config import SimParams                       # noqa: E402
from otfs_isac_sim import waveform as wf                          # noqa: E402
from otfs_isac_sim.geometry import steering                       # noqa: E402
from otfs_isac_sim.an import build_an_basis, crude_an_allocation, null_projector  # noqa: E402
from otfs_isac_sim.crb import crb_matrix, min_illumination_for_crb, crb_from_illumination  # noqa: E402
from otfs_isac_sim.simulator import run_episode, build_problem    # noqa: E402
from otfs_isac_sim.solvers import SOLVERS                         # noqa: E402

P = SimParams()


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


def test_an_in_user_null_space_and_budget():
    th_u = np.deg2rad([-30, 20, 14, 60])
    basis = build_an_basis(th_u, [np.deg2rad([-17, -16]), np.deg2rad([50])], P.Nt)
    q = crude_an_allocation(basis, P.P_an_max, "per_eve")
    S = basis.covariance(q)
    A = steering(th_u, P.Nt)
    leak = np.real(np.einsum("ip,ij,jp->p", A.conj(), S, A))
    assert np.all(np.abs(leak) < 1e-9 * np.real(np.trace(S)))
    assert abs(basis.power(q) - P.P_an_max) < 1e-12
    assert abs(np.real(np.trace(S)) - P.P_an_max) < 1e-9
    Pn = null_projector(th_u, P.Nt)
    assert np.allclose(Pn @ Pn, Pn)


def test_crude_crb_psd_and_monotone():
    C1 = crb_matrix(10.0, 0.3, P)
    C2 = crb_matrix(100.0, 0.3, P)
    assert np.all(np.linalg.eigvalsh(C1) > 0)
    assert np.all(np.diag(C2) < np.diag(C1))


def test_min_illumination_meets_crb_requirement():
    th, d = 0.4, 40.0
    illum = min_illumination_for_crb(th, d, P)
    C = crb_from_illumination(th, d, illum * 1.0001, P)
    assert np.all(np.diag(C) <= np.diag(P.crb_req) * (1 + 1e-6))


def test_all_working_solvers_respect_power_constraints():
    from otfs_isac_sim.frameworks import get_frameworks
    from otfs_isac_sim.geometry import Scenario
    from otfs_isac_sim.simulator import _truth_estimate
    scen = Scenario.from_params(P, np.random.default_rng(3))
    preds = [_truth_estimate(scen, i) for i in range(scen.P)]
    for e in preds[scen.K:]:
        e.C = np.diag([1e-17, 1.0, 1e-5])
    prob = build_problem(preds, scen.K, get_frameworks(P)["otfs_isac"], P)
    for name, fn in SOLVERS.items():
        if name in ("sca_sdr", "drl"):
            continue
        sol = fn(prob)
        tot = np.sum(np.abs(sol.F) ** 2) + np.real(np.trace(sol.S))
        assert tot <= P.P_max * (1 + 1e-6), name
        assert sol.info["an_power"] <= P.P_an_max * (1 + 1e-6), name


def test_episode_runs():
    r = run_episode(P, "otfs_isac", "grid_split", n_frames=8)
    assert r["sum_sec"].shape == (7,) and np.all(np.isfinite(r["sum_sec"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
