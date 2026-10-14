"""
#############################################################################
#  DRAFT / PLACEHOLDER — CRUDE CRB MATRIX                                   #
#############################################################################
Crude Fisher information / CRB for eta_j = [gamma_j (s), omega_j (Hz), theta_j (rad)]
used to build the uncertainty ellipsoid E_j (Draft Sec. VI) and the constraint
I^{-1}(eta_j) <= C_j^req (Draft Sec. VII).

What this placeholder DOES:
  * Uses the classical single-target, high-SNR bounds, each parameter treated
    separately (diagonal FIM):
        var(gamma) >= 1 / (8 pi^2 B_rms^2  SNR_arr),  B_rms^2 = (M delta_f)^2 / 12
        var(omega) >= 1 / (8 pi^2 T_rms^2  SNR_arr),  T_rms^2 = (N T)^2 / 12
        var(theta) >= 6 / (pi^2 cos^2(theta) SNR_el Nr (Nr^2 - 1))
    with SNR_el = (Gm |beta|)^2 * a^H(theta) R_x a(theta) / (P_ref sigma^2) (post-MF,
    per receive element; P_ref = P_max normalisation, see config.radar_power_ref) and SNR_arr = Nr * SNR_el.
  * Ties the bound to the transmit covariance R_x = sum_k f_k f_k^H + S, so the
    directed beams AND the AN both illuminate (and help localise) the Eves.

What is MISSING (TODO before using CRB numbers in the paper):
  [ ] full 3x3 FIM from the matched-filter statistic of Draft Sec. IV, incl.
      gamma-omega coupling through the OTFS ambiguity function;
  [ ] complex reflection coefficient beta as nuisance parameter (Schur complement);
  [ ] dependence on the actual waveform s_i(t) (OTFS frame) rather than flat
      spectrum / uniform-in-time energy assumptions;
  [ ] Godrich et al. 2010 style power-allocation dependence (open comment in
      Draft: "verify, taken from power allocation, Godrich") and the Patwari
      derivation note ("verify, add derivation; Patwari");
  [ ] propagation of C_j through the motion model (prediction step) — currently
      the estimation CRB is used unchanged for the next frame.
"""
from __future__ import annotations

import numpy as np

from .config import SimParams, C_LIGHT
from .draft import draft
from .geometry import steer1

_MSG = ("crude diagonal CRB (independent delay/Doppler/angle bounds, no beta nuisance, "
        "no ambiguity-function coupling) — see crb.py header TODO list")


def illumination(theta: float, R_x: np.ndarray) -> float:
    """Transmit beampattern a^H(theta) R_x a(theta)."""
    a = steer1(theta, R_x.shape[0])
    return float(np.real(a.conj() @ R_x @ a))


def echo_snr_element(beta_abs: float, illum: float, p: SimParams) -> float:
    return (p.Gm_eff * beta_abs) ** 2 * (illum / p.radar_power_ref) / p.sigma2_radar


@draft("CRB", _MSG)
def crude_fim(snr_el: float, theta: float, p: SimParams) -> np.ndarray:
    """DRAFT: diagonal 3x3 FIM on [gamma, omega, theta]."""
    snr_el = max(snr_el, 1e-12)
    snr_arr = p.Nr * snr_el
    b2 = p.bandwidth ** 2 / 12.0
    t2 = p.frame_duration ** 2 / 12.0
    fim_gamma = 8 * np.pi ** 2 * b2 * snr_arr
    fim_omega = 8 * np.pi ** 2 * t2 * snr_arr
    fim_theta = (np.pi ** 2 * np.cos(theta) ** 2 * snr_el * p.Nr * (p.Nr ** 2 - 1)) / 6.0
    # --- TODO: off-diagonal terms (gamma-omega coupling, beta nuisance) ---------
    return np.diag([fim_gamma, fim_omega, fim_theta])


def crb_matrix(snr_el: float, theta: float, p: SimParams) -> np.ndarray:
    """C_j = I^{-1}(eta_j)."""
    return np.linalg.inv(crude_fim(snr_el, theta, p))


def crb_from_illumination(theta: float, dist: float, illum: float, p: SimParams) -> np.ndarray:
    beta_abs = p.rcs / (2 * dist)          # Draft Sec. VI: beta = xi / (2 d)
    return crb_matrix(echo_snr_element(beta_abs, illum, p), theta, p)


def crb_constraint_margin(C: np.ndarray, C_req: np.ndarray) -> float:
    """min eig(C_req - C). >= 0  <=>  C <= C_req in the PSD order."""
    return float(np.min(np.linalg.eigvalsh(C_req - C)))


@draft("CRB", _MSG)
def min_illumination_for_crb(theta: float, dist: float, p: SimParams) -> float:
    """Smallest a^H R_x a toward an Eve such that the crude CRB meets C^req.
    Because the crude FIM is diagonal and linear in SNR, this is a closed form,
    which turns the CRB constraint into a linear constraint on the beampattern."""
    beta_abs = p.rcs / (2 * dist)
    C_unit = np.diag(np.linalg.inv(crude_fim(1.0, theta, p)))   # CRB at SNR_el = 1
    snr_needed = float(np.max(C_unit / np.diag(p.crb_req)))
    return snr_needed * p.sigma2_radar * p.radar_power_ref / (p.Gm_eff * beta_abs) ** 2


def range_std(C: np.ndarray) -> float:
    return float(np.sqrt(C[0, 0]) * C_LIGHT / 2)


def uncertainty_samples(theta_hat: float, d_hat: float, C: np.ndarray, p: SimParams,
                        n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points of the ellipsoid E_j = {eta: (eta-eta_hat)^T C^-1 (eta-eta_hat) <= eps}
    projected on the (theta, d) plane: centre + n boundary points.
    Doppler is not sampled (it does not enter the Eve rate for OTFS; for OFDM its
    effect on Eve ICI is second-order) — DRAFT simplification."""
    n = p.n_unc_samples if n is None else n
    s_th = np.sqrt(p.eps_conf * C[2, 2])
    s_d = np.sqrt(p.eps_conf * C[0, 0]) * C_LIGHT / 2
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th = np.concatenate([[theta_hat], theta_hat + s_th * np.cos(phi)])
    d = np.concatenate([[d_hat], np.maximum(d_hat + s_d * np.sin(phi), 1.0)])
    return th, d
