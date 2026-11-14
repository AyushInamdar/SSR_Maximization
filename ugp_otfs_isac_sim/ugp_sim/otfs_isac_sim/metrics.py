"""
Rates, secrecy rates and constraint checks evaluated on the TRUE geometry
(Report Sec. VII-B, Eqs. (13), (15), (16), (27)-(29)).

With h_k := |h_k|^2 the LoS power gain of user k (config.path_gain; the report's
Eq. (12) writes h_k = (c / 4 pi fc d_k)^2, which is that power gain: using it as an
*amplitude* would square the path loss twice), a(.) the transmit steering vector,
F = [f_1 ... f_K] and S the AN covariance:

  SINR_k   = |h_k a^H(th_k) f_k|^2 ... written with power gains:
           = h_k |a^H(th_k) f_k|^2 /
             ( sum_{k'!=k} h_k |a^H(th_k) f_k'|^2  +  h_k a^H(th_k) S a(th_k)  +  N0 )   (27)
  R_k      = pre * log2(1 + SINR_k)                                                     (28)

  N_{j,k}  = h_j |a^H(th_j) f_k|^2                                      (signal of stream k at Eve j)
  D_{j,k}  = h_j sum_{k'!=k} |a^H(th_j) f_k'|^2 + h_j a^H(th_j) S a(th_j) + N0
  R^Eve_{j,k} = pre * log2(1 + N_{j,k} / D_{j,k})                                       (29)
  R_k^sec  = [ R_k - max_j R^Eve_{j,k} ]^+

(27) and the N, D of (29) are DERIVED from Eqs. (13) and (16) -- the report leaves them as
TODO. N0 is the noise power over the signal bandwidth (config.N0).

Differences w.r.t. the report (deliberate, flagged in CHANGELOG):
  * the user SINR keeps the AN leakage term h_k a^H(th_k) S a(th_k): it is zero only if
    the predicted user angle (the null-space projector's) equals the true one; it is
    returned as `an_leak_users` (W);
  * a waveform-dependent Doppler impairment (OFDM ICI) and the CP overhead `pre` are
    applied through waveform.effective_sinr / rate_prefactor (identity for OTFS);
  * Eve treats other users' streams as noise (as in the report). A stronger Eve
    (SIC / multi-user decoding) is a TODO for a worst-case analysis.
`solvers.LinearModel.rates` implements the same expressions on PREDICTED states for the
design; tests/test_core.py::test_design_rates_match_true_rates checks they coincide.
"""
from __future__ import annotations

import numpy as np

from .config import SimParams
from .geometry import steering, Scenario
from .waveform import effective_sinr


def beam_gains(F: np.ndarray, S: np.ndarray, th: np.ndarray):
    """G[p, k] = |a^H(th_p) f_k|^2 (P x K) and L[p] = a^H(th_p) S a(th_p) (P)."""
    A = steering(th, F.shape[0])
    G = np.abs(A.conj().T @ F) ** 2
    L = np.real(np.einsum("ip,ij,jp->p", A.conj(), S, A))
    return A, G, L


def user_signal_interference(G_u: np.ndarray, L_u: np.ndarray, h2_u: np.ndarray):
    """Numerator and interference (without noise) of Eq. (27), length K each."""
    sig = h2_u * np.diag(G_u)
    interf = h2_u * (G_u.sum(1) - np.diag(G_u)) + h2_u * L_u
    return sig, interf


def eve_N_D(G_e: np.ndarray, L_e: np.ndarray, h2_e: np.ndarray, N0: float):
    """N_{j,k} and D_{j,k} of Eq. (29), each (J, K)."""
    N = h2_e[:, None] * G_e
    D = h2_e[:, None] * (G_e.sum(1, keepdims=True) - G_e) + (h2_e * L_e)[:, None] + N0
    return N, D


def evaluate_true(F: np.ndarray, S: np.ndarray, scen: Scenario, waveform: str,
                  prefactor: float, p: SimParams) -> dict:
    K, J = scen.K, scen.J
    th, h2, nu = scen.theta, scen.h2, scen.nu
    A, G, L = beam_gains(F, S, th)

    # users, Eq. (27)-(28)
    sig_u, int_u = user_signal_interference(G[:K], L[:K], h2[:K])
    sinr_u = effective_sinr(sig_u, int_u, p.N0, nu[:K], waveform, p)
    R_u = prefactor * np.log2(1 + sinr_u)

    # eves, Eq. (29): J x K
    N_e, D_e = eve_N_D(G[K:], L[K:], h2[K:], p.N0)
    sinr_e = effective_sinr(N_e, D_e - p.N0, p.N0, nu[K:, None], waveform, p)
    R_e = prefactor * np.log2(1 + sinr_e)

    R_sec = np.maximum(R_u - (R_e.max(0) if J else 0.0), 0.0)
    illum = np.real(np.einsum("ip,ij,jp->p", A.conj(), F @ F.conj().T + S, A))
    return dict(R_users=R_u, R_eve=R_e, R_sec=R_sec, sum_sec=float(R_sec.sum()),
                sum_rate=float(R_u.sum()), sinr_users=sinr_u, illum=illum,
                N_eve=N_e, D_eve=D_e,
                snr_users=sig_u / p.N0,               # single-stream SNR, no interference
                an_leak_users=L[:K] * h2[:K])
