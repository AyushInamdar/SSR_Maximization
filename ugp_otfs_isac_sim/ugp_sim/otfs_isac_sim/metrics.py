"""
Rates, secrecy rates and constraint checks evaluated on the TRUE geometry
(Draft Sec. VII, "Achievable Rates").

  SINR_k     = |h_k a^H(th_k) f_k|^2 / ( sum_{k'!=k} |h_k a^H f_k'|^2 + h_k^2 a^H S a + N0 )
  R_k        = pre * log2(1 + SINR_k)
  R_{j,k}^Eve = pre * log2(1 + N_{j,k} / D_{j,k})   (other streams treated as noise)
  R_k^sec    = [R_k - max_j R_{j,k}^Eve]^+

Differences w.r.t. the draft (deliberate, flagged):
  * the user SINR includes the AN leakage term h_k^2 a^H(th_k) S a(th_k): it is zero
    only if predicted == true user angle; the draft drops it;
  * a waveform-dependent Doppler impairment (OFDM ICI) and CP overhead `pre` are
    applied through waveform.effective_sinr / rate_prefactor;
  * Eve treats other users' streams as noise (as in the draft). A stronger Eve
    (SIC / multi-user decoding) is a TODO for a worst-case analysis.
"""
from __future__ import annotations

import numpy as np

from .config import SimParams
from .geometry import steering, Scenario
from .waveform import effective_sinr


def evaluate_true(F: np.ndarray, S: np.ndarray, scen: Scenario, waveform: str,
                  prefactor: float, p: SimParams) -> dict:
    K, J = scen.K, scen.J
    th, h2, nu = scen.theta, scen.h2, scen.nu
    A = steering(th, F.shape[0])                       # Nt x P
    G = np.abs(A.conj().T @ F) ** 2                     # P x K beam gains
    L = np.real(np.einsum("ip,ij,jp->p", A.conj(), S, A))   # a^H S a per target

    # users
    sig_u = h2[:K] * np.diag(G[:K])
    int_u = h2[:K] * (G[:K].sum(1) - np.diag(G[:K])) + h2[:K] * L[:K]
    sinr_u = effective_sinr(sig_u, int_u, p.N0, nu[:K], waveform, p)
    R_u = prefactor * np.log2(1 + sinr_u)

    # eves: J x K
    ge = G[K:]                                          # J x K
    sig_e = h2[K:, None] * ge
    int_e = h2[K:, None] * (ge.sum(1, keepdims=True) - ge) + (h2[K:] * L[K:])[:, None]
    sinr_e = effective_sinr(sig_e, int_e, p.N0, nu[K:, None], waveform, p)
    R_e = prefactor * np.log2(1 + sinr_e)

    R_sec = np.maximum(R_u - (R_e.max(0) if J else 0.0), 0.0)
    illum = np.real(np.einsum("ip,ij,jp->p", A.conj(), F @ F.conj().T + S, A))
    return dict(R_users=R_u, R_eve=R_e, R_sec=R_sec, sum_sec=float(R_sec.sum()),
                sum_rate=float(R_u.sum()), sinr_users=sinr_u, illum=illum,
                an_leak_users=L[:K] * h2[:K])
