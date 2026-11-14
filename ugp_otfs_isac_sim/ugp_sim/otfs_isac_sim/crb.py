"""
Cramer-Rao bound of the target parameters eta_j = [gamma (s), omega (Hz), theta (rad)]
(Report Sec. VI-B, Eqs. (25)-(26), derived in Appendix A, Eq. (66)):

    C_j = diag( N0 / (2 |h|^2 Nr Omega),
                N0 / (8 pi^2 |h|^2 Nr Tc),
                6 N0 / (pi^2 cos^2(theta) |h|^2 E0 Nr (Nr^2 - 1)) ),     h = beta_j

with the pulse moments of the transmitted sensing waveform s0(t)

    E0 = int |s0|^2,  Omega = int |s0'|^2,  Tm = int u |s0|^2,  T2 = int u^2 |s0|^2,
    Tc = T2 - Tm^2 / E0.

Implementation
  * E0, Omega, Tc are computed numerically from the ACTUAL OTFS waveform of
    `waveform.probe_waveform` (`pulse_moments`), not from flat-spectrum /
    uniform-in-time approximations. (`crb_moments="flat"` in SimParams keeps the old
    approximations B_rms^2 = (M df)^2/12, T_rms^2 = (N T)^2/12 for comparison.)
  * Only the ratios Omega/E0 and Tc/E0 enter once the post-matched-filter element SNR
        snr_el = |h|^2 E0 / N0
    is factored out; snr_el is where the transmit covariance enters: for the echo of the
    directed transmission the radiated power towards the target is the illumination
    a^H(theta) R_x a(theta), R_x = F F^H + S, and
        snr_el = (Gm |beta|)^2 * a^H R_x a / (P_ref sigma^2)        (`echo_snr_element`),
    the normalised radar model of Yuan21 / Draft Eq. (10) (config.radar_power_ref).
    This is the only link between the beamformer/AN and the CRB, so the constraint
    C_j <= C_j^req is a linear constraint on the beampattern.

Assumptions of Appendix A (all inherited here)
  (i)   a1 = Re int s0* s0' = 0 (follows from the periodic frame: |s0|^2 equal at both
        ends) -- the frame is treated as one period of a band-limited signal.
  (ii)  zero phase slope, alpha = Im int s0* s0' = 0 and beta = Im int u s0* s0' = 0, so
        the delay-Doppler and delay-angle FIM couplings vanish and C_j is diagonal.
        `crb_exact` evaluates the full 5x5 FIM (nuisance h = hR + j hI) without (ii);
        `coupling_diagnostics` reports how well (ii) holds for the waveform in use.
  (iii) continuous observation (Ns -> infinity), AWGN with PSD N0, single target
        (echoes of different targets separated by the asymptotically orthogonal b(theta)),
        h constant over the frame, half-wavelength ULA receive array b_n = e^{j pi (n-1) sin(theta)}.
  (iv)  the bound is tight only asymptotically (high SNR): see
        tests/test_core.py::test_ml_estimates_approach_crb.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .config import SimParams, C_LIGHT
from .geometry import steer1


# ----------------------------------------------------------------------------
# pulse moments
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PulseMoments:
    """Moments (38)-(39) of a sensing pulse. Time origin u = 0 at the frame start."""
    E0: float
    Omega: float
    Tm: float
    T2: float
    A1: complex            # int s0* s0'      = a1 + j alpha
    A3: complex            # int u s0* s0'    = a3 + j beta

    @property
    def Tc(self) -> float:
        return self.T2 - self.Tm ** 2 / self.E0

    @property
    def omega_norm(self) -> float:
        """Omega / E0  (= (2 pi B_rms)^2), 1/s^2."""
        return self.Omega / self.E0

    @property
    def tc_norm(self) -> float:
        """Tc / E0  (= T_rms^2), s^2."""
        return self.Tc / self.E0

    @property
    def alpha(self) -> float:
        return float(self.A1.imag)

    @property
    def beta(self) -> float:
        return float(self.A3.imag)


def _spectrum_on_oversampled_grid(s: np.ndarray, oversample: int):
    """Zero-padded (band-limited interpolation) spectrum of one period of s.
    The Nyquist bin (L even) is split symmetrically between -fs/2 and +fs/2.
    Returns (S_os, k_os): padded spectrum and signed bin index of every padded bin."""
    L = s.size
    Lo = L * oversample
    S = np.fft.fft(s)
    k = np.fft.fftfreq(L, 1.0 / L).astype(int)         # -L/2 .. L/2-1
    So = np.zeros(Lo, complex)
    if L % 2 == 0:
        ny = (k == -L // 2)
        So[(k[~ny]) % Lo] += S[~ny]
        So[(-L // 2) % Lo] += S[ny][0] / 2
        So[(L // 2) % Lo] += S[ny][0] / 2
    else:
        So[k % Lo] += S
    k_os = np.fft.fftfreq(Lo, 1.0 / Lo)
    return So, k_os


def pulse_moments(s: np.ndarray, fs: float, oversample: int = 4) -> PulseMoments:
    """Numerically integrate the moments of Eq. (38) for the sampled pulse `s`
    (one period of a band-limited signal, see `waveform.probe_waveform`).
    The signal and its derivative are evaluated on an `oversample`-times finer grid via
    FFT interpolation / spectral differentiation."""
    s = np.asarray(s, dtype=complex)
    L = s.size
    So, k_os = _spectrum_on_oversampled_grid(s, oversample)
    f = k_os * fs / L                                    # Hz (bin spacing fs/L)
    x = np.fft.ifft(So) * oversample
    dx = np.fft.ifft(1j * 2 * np.pi * f * So) * oversample
    dt = 1.0 / (fs * oversample)
    u = np.arange(x.size) * dt
    p2 = np.abs(x) ** 2
    cross = np.conj(x) * dx
    return PulseMoments(
        E0=float(np.sum(p2) * dt),
        Omega=float(np.sum(np.abs(dx) ** 2) * dt),
        Tm=float(np.sum(u * p2) * dt),
        T2=float(np.sum(u ** 2 * p2) * dt),
        A1=complex(np.sum(cross) * dt),
        A3=complex(np.sum(u * cross) * dt),
    )


def flat_moments(p: SimParams) -> PulseMoments:
    """Idealised flat-spectrum / uniform-in-time pulse of unit power:
    B_rms^2 = (M df)^2/12, T_rms^2 = (N T)^2/12  (the former crude approximation)."""
    Tf = p.frame_duration
    E0 = Tf
    return PulseMoments(E0=E0, Omega=(2 * np.pi) ** 2 * p.bandwidth ** 2 / 12.0 * E0,
                        Tm=Tf / 2 * E0, T2=Tf ** 2 / 3 * E0, A1=0j, A3=-0.5 * E0 + 0j)


@functools.lru_cache(maxsize=16)
def _moments_cached(kind: str, N: int, M: int, delta_f: float, seed: int) -> PulseMoments:
    from .waveform import probe_waveform
    p = SimParams(N=N, M=M, delta_f=delta_f)
    if kind == "flat":
        return flat_moments(p)
    if kind == "waveform":
        return pulse_moments(probe_waveform(p, seed), p.fs)
    raise ValueError(f"crb_moments must be 'waveform' or 'flat', got {kind!r}")


def moments_for(p: SimParams) -> PulseMoments:
    return _moments_cached(p.crb_moments, p.N, p.M, p.delta_f, p.crb_waveform_seed)


# ----------------------------------------------------------------------------
# FIM / CRB, Eq. (66)
# ----------------------------------------------------------------------------
def illumination(theta: float, R_x: np.ndarray) -> float:
    """Transmit beampattern a^H(theta) R_x a(theta)."""
    a = steer1(theta, R_x.shape[0])
    return float(np.real(a.conj() @ R_x @ a))


def echo_snr_element(beta_abs: float, illum: float, p: SimParams) -> float:
    """snr_el = |h|^2 E0 / N0 of the matched-filtered echo (per receive element)."""
    return (p.Gm_eff * beta_abs) ** 2 * (illum / p.radar_power_ref) / p.sigma2_radar


def crb_closed_form(snr_el: float, theta: float, Nr: int, omega_norm: float,
                    tc_norm: float) -> np.ndarray:
    """Eq. (66) with snr_el = |h|^2 E0/N0, omega_norm = Omega/E0, tc_norm = Tc/E0."""
    snr_el = max(snr_el, 1e-12)
    return np.diag([1.0 / (2 * Nr * snr_el * omega_norm),
                    1.0 / (8 * np.pi ** 2 * Nr * snr_el * tc_norm),
                    6.0 / (np.pi ** 2 * max(np.cos(theta) ** 2, 1e-12) * snr_el * Nr * (Nr ** 2 - 1))])


def fim_matrix(snr_el: float, theta: float, p: SimParams) -> np.ndarray:
    """J(eta_j) = C_j^{-1} (diagonal under Appendix A's assumptions)."""
    return np.linalg.inv(crb_matrix(snr_el, theta, p))


def crb_matrix(snr_el: float, theta: float, p: SimParams) -> np.ndarray:
    """C_j = J^{-1}(eta_j), Eq. (66), on [gamma (s^2), omega (Hz^2), theta (rad^2)]."""
    m = moments_for(p)
    return crb_closed_form(snr_el, theta, p.Nr, m.omega_norm, m.tc_norm)


def crb_exact(snr_el: float, theta: float, gamma: float, m: PulseMoments, Nr: int) -> np.ndarray:
    """3x3 CRB from the full 5x5 FIM of Appendix A, Eqs. (36)-(54), WITHOUT assuming
    alpha = beta = 0 (a1 is kept too). Units: N0 = 1, |h| = 1, E0 -> snr_el."""
    g = snr_el / m.E0                                    # moment scale: E0/N0 -> snr_el
    E0, Om, Tm, T2 = m.E0 * g, m.Omega * g, m.Tm * g, m.T2 * g
    A1, A3 = m.A1 * g, m.A3 * g
    c = np.cos(theta)
    n = np.arange(Nr)
    bbp = 1j * np.pi * c * np.sum(n)                     # b^H b'  = j Gamma
    bpbp = np.pi ** 2 * c ** 2 * np.sum(n ** 2)          # b'^H b'
    # parameters: gamma, omega, theta, hR, hI  with h = |h| = 1 (real, WLOG)
    # d mu = coef * (vector) * g(u): vectors b / b', scalar functions integrated below
    # int conj(g_m) g_n du for the pairs of scalar functions
    W = Tm + gamma * E0
    I = {}
    I[("ds", "ds")] = Om
    I[("ds", "ts")] = np.conj(A3 + gamma * A1)           # int s0'^* (u+g) s0
    I[("ts", "ts")] = T2 + 2 * gamma * Tm + gamma ** 2 * E0
    I[("ds", "s")] = np.conj(A1)                         # int s0'^* s0
    I[("ts", "s")] = W
    I[("s", "s")] = E0
    # derivative spec: (coef, vector id, scalar fn id)
    spec = [(-1.0, "b", "ds"), (1j * 2 * np.pi, "b", "ts"), (1.0, "bp", "s"),
            (1.0, "b", "s"), (1j, "b", "s")]

    def sc(f1, f2):                                      # int conj(f1) f2
        if (f1, f2) in I:
            return I[(f1, f2)]
        return np.conj(I[(f2, f1)])

    def vv(v1, v2):                                      # v1^H v2
        if v1 == "b" and v2 == "b":
            return Nr
        if v1 == "bp" and v2 == "bp":
            return bpbp
        if v1 == "b" and v2 == "bp":
            return bbp
        return np.conj(bbp)

    J = np.zeros((5, 5))
    for i, (ci, vi, fi) in enumerate(spec):
        for j, (cj, vj, fj) in enumerate(spec):
            J[i, j] = 2 * np.real(np.conj(ci) * cj * vv(vi, vj) * sc(fi, fj))
    return np.linalg.inv(J)[:3, :3]


def coupling_diagnostics(m: PulseMoments, gamma: float = 0.0, Nr: int = 64) -> dict:
    """How well does assumption (ii) of Appendix A hold? Returns the largest |correlation
    coefficient| of the exact 3x3 CRB and the relative gap of its diagonal to Eq. (66)."""
    snr, th = 1e4, 0.3
    Ce = crb_exact(snr, th, gamma, m, Nr)
    Cc = crb_closed_form(snr, th, Nr, m.omega_norm, m.tc_norm)
    d = np.sqrt(np.diag(Ce))
    rho = Ce / np.outer(d, d)
    off = np.abs(rho - np.diag(np.diag(rho))).max()
    return dict(max_abs_corr=float(off),
                diag_rel_gap=float(np.max(np.abs(np.diag(Ce) / np.diag(Cc) - 1))),
                alpha_rel=float(abs(m.alpha) / (m.E0 * np.sqrt(m.omega_norm))),
                a1_rel=float(abs(m.A1.real) / m.E0))


def crb_from_illumination(theta: float, dist: float, illum: float, p: SimParams) -> np.ndarray:
    beta_abs = p.rcs / (2 * dist)          # Draft Sec. VI: beta = xi / (2 d)
    return crb_matrix(echo_snr_element(beta_abs, illum, p), theta, p)


def crb_constraint_margin(C: np.ndarray, C_req: np.ndarray) -> float:
    """min eig(C_req - C). >= 0  <=>  C <= C_req in the PSD order."""
    return float(np.min(np.linalg.eigvalsh(C_req - C)))


def min_illumination_for_crb(theta: float, dist: float, p: SimParams) -> float:
    """Smallest a^H R_x a toward a target such that Eq. (66) meets C^req. Every diagonal
    entry of (66) is inversely proportional to snr_el, so this is closed form and turns
    the CRB constraint into a LINEAR constraint on the beampattern."""
    beta_abs = p.rcs / (2 * dist)
    C_unit = np.diag(crb_matrix(1.0, theta, p))              # CRB at snr_el = 1
    snr_needed = float(np.max(C_unit / np.diag(p.crb_req)))
    return snr_needed * p.sigma2_radar * p.radar_power_ref / (p.Gm_eff * beta_abs) ** 2


def range_std(C: np.ndarray) -> float:
    return float(np.sqrt(C[0, 0]) * C_LIGHT / 2)


# ----------------------------------------------------------------------------
# uncertainty region E_j
# ----------------------------------------------------------------------------
def uncertainty_samples(theta_hat: float, d_hat: float, C: np.ndarray, p: SimParams,
                        n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points of the ellipsoid E_j = {eta: (eta-eta_hat)^T C^-1 (eta-eta_hat) <= eps}
    projected on the (theta, d) plane: centre + n boundary points.
    Doppler is not sampled (it does not enter the Eve rate for OTFS; for OFDM its
    effect on Eve ICI is second-order)."""
    n = p.n_unc_samples if n is None else n
    s_th = np.sqrt(p.eps_conf * C[2, 2])
    s_d = np.sqrt(p.eps_conf * C[0, 0]) * C_LIGHT / 2
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    th = np.concatenate([[theta_hat], theta_hat + s_th * np.cos(phi)])
    d = np.concatenate([[d_hat], np.maximum(d_hat + s_d * np.sin(phi), 1.0)])
    return th, d


def region_area_m2(C: np.ndarray, d: float, p: SimParams) -> float:
    """Area (m^2) of the (cross-range, range) projection of E_j:
    pi * eps * (d sigma_theta) * (c sigma_gamma / 2)."""
    return float(np.pi * p.eps_conf * d * np.sqrt(C[2, 2]) * range_std(C))
