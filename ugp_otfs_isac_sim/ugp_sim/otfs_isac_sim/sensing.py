"""
Radar sensing: echo model, ML angle estimation, delay/Doppler estimation,
localisation and one-step prediction (Draft Secs. III, IV, VI; Yuan21 Sec. III-B).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimParams, C_LIGHT
from .draft import draft_notice
from .geometry import steering, angle_from_pos
from . import crb as crbmod


def simulate_echo(theta: float, dist: float, illum: float, p: SimParams,
                  rng: np.random.Generator, Gm: float | None = None) -> np.ndarray:
    """Matched-filtered echo of one target after array separation (Draft eq. r_i):
        r_i = Gm * beta_i * b(theta_i) * [a^H(theta_i) x_i] + z_i,   z_i ~ CN(0, sigma^2 I)
    The transmit-side term is summarised by its power `illum` = a^H R_x a with an
    unknown phase (noncoherent model, valid for both users and Eves)."""
    Gm = p.Gm_eff if Gm is None else Gm
    amp = Gm * (p.rcs / (2 * dist)) * np.sqrt(max(illum, 0.0) / p.radar_power_ref)
    b = steering(theta, p.Nr)[:, 0]
    z = np.sqrt(p.sigma2_radar / 2) * (rng.standard_normal(p.Nr) + 1j * rng.standard_normal(p.Nr))
    return amp * np.exp(1j * rng.uniform(0, 2 * np.pi)) * b + z


def ml_angle(r: np.ndarray, theta_center: float, p: SimParams) -> float:
    """ML angle with unknown complex amplitude: argmax |b^H(theta) r|^2 over a grid
    around the predicted angle (Draft Sec. VI: search only near theta_bar).
    NOTE: Draft/Yuan21 use the grid Theta = {pi/Nt, ..., pi}, which is too coarse
    (~0.05 rad at Nt=64) and in [0, pi] while theta lies in (-pi/2, pi/2) under the
    sin(theta)=x/d convention; a fine local grid is used here."""
    hw, res = p.theta_search_halfwidth, p.theta_grid_res
    grid = np.arange(theta_center - hw, theta_center + hw + res / 2, res)
    grid = grid[np.abs(grid) < np.pi / 2]
    metric = np.abs(steering(grid, p.Nr).conj().T @ r) ** 2
    return float(grid[np.argmax(metric)])


def estimate_delay_doppler(gamma: float, omega: float, C: np.ndarray,
                           rng: np.random.Generator) -> tuple[float, float]:
    """DRAFT shortcut: instead of running the 2-D (gamma, omega) matched-filter
    search of Draft Sec. IV on every frame, draw the estimate from N(truth, CRB).
    A real DD matched filter is TODO (expensive; see waveform.py for the OTFS
    frame generator it would correlate against)."""
    draft_notice("DD-ESTIMATOR", "delay/Doppler estimates drawn from N(truth, crude CRB) "
                 "instead of a matched-filter grid search")
    return (gamma + rng.normal(0, np.sqrt(C[0, 0])),
            omega + rng.normal(0, np.sqrt(C[1, 1])))


def localise(gamma_hat: float, theta_hat: float, bs: np.ndarray) -> np.ndarray:
    """Draft eqs. for q_x, q_y."""
    r = C_LIGHT * gamma_hat / 2
    return bs + r * np.array([np.sin(theta_hat), np.cos(theta_hat)])


def speed_from_doppler(omega_hat: float, theta_hat: float, fc: float,
                       fallback: float | None) -> float:
    """v = -c * omega / (2 fc sin(theta))  (geometrically consistent; the Draft's
    s = c*omega/(fc cos theta) uses the other angle convention and omits the
    round-trip factor 2). Near broadside (sin theta ~ 0) speed is unobservable from
    Doppler -> keep previous estimate."""
    s = np.sin(theta_hat)
    if abs(s) < 0.05:
        return fallback if fallback is not None else 0.0
    return float(-C_LIGHT * omega_hat / (2 * fc * s))


@dataclass
class Estimate:
    pos: np.ndarray          # (2,)
    speed: float
    theta: float
    dist: float
    C: np.ndarray            # 3x3 crude CRB at estimation time

    def predict(self, p: SimParams) -> "Estimate":
        """Draft: q_bar = q_hat + dT * s_hat * [1, 0]^T, theta_bar = arcsin(...)."""
        bs = np.asarray(p.bs_pos, dtype=float)
        q = self.pos + p.dT * self.speed * np.array([1.0, 0.0])
        return Estimate(pos=q, speed=self.speed, theta=float(angle_from_pos(q, bs)),
                        dist=float(np.linalg.norm(q - bs)), C=self.C)


def sense_target(theta_true: float, dist_true: float, omega_true: float, illum: float,
                 theta_prior: float, prev_speed: float | None, p: SimParams,
                 rng: np.random.Generator, Gm: float | None = None) -> Estimate:
    """One sensing step for one target: echo -> ML angle -> delay/Doppler ->
    location & speed."""
    Gm = p.Gm_eff if Gm is None else Gm
    r = simulate_echo(theta_true, dist_true, illum, p, rng, Gm)
    if theta_prior is None:            # no prediction yet (isotropic probe): coarse -> fine
        coarse = np.arange(-1.45, 1.45, 2.0 / p.Nr)
        theta_prior = float(coarse[np.argmax(np.abs(steering(coarse, p.Nr).conj().T @ r))])
    th = ml_angle(r, theta_prior, p)
    # crude CRB evaluated with the matched-filter gain actually used
    snr_el = (Gm * p.rcs / (2 * dist_true)) ** 2 * (illum / p.radar_power_ref) / p.sigma2_radar
    C = crbmod.crb_matrix(snr_el, theta_true, p)
    g_hat, w_hat = estimate_delay_doppler(2 * dist_true / C_LIGHT, omega_true, C, rng)
    bs = np.asarray(p.bs_pos, dtype=float)
    q = localise(g_hat, th, bs)
    v = speed_from_doppler(w_hat, th, p.fc, prev_speed)
    return Estimate(pos=q, speed=v, theta=th, dist=float(np.linalg.norm(q - bs)), C=C)


def feedback_estimate(theta_true: float, dist_true: float, pilot_snr: float,
                      p: SimParams, rng: np.random.Generator) -> Estimate:
    """DRAFT, stylised feedback benchmark (Yuan21 Sec. IV, ref. [27]):
    the vehicle measures ONE downlink pilot (no matched-filter gain, one-way link),
    picks the best beam from a DFT codebook of oversample*Nt beams and feeds back the
    index. The BS gets a quantised angle, no range/speed -> no kinematic prediction
    (the beam lags the vehicle by one time instant)."""
    draft_notice("FEEDBACK", "stylised pilot/feedback benchmark: one-pilot angle error "
                 "(single-pilot CRB) + DFT-codebook quantisation, no prediction")
    snr = max(pilot_snr, 1e-12)
    var = 6.0 / (np.pi ** 2 * max(np.cos(theta_true) ** 2, 1e-3) * snr * p.Nt * (p.Nt ** 2 - 1))
    th = theta_true + rng.normal(0, np.sqrt(var))
    n_cb = p.feedback_codebook_oversample * p.Nt
    u_grid = -1 + 2 * (np.arange(n_cb) + 0.5) / n_cb          # codebook in sin(theta)
    th_q = float(np.arcsin(u_grid[np.argmin(np.abs(u_grid - np.sin(th)))]))
    bs = np.asarray(p.bs_pos, dtype=float)
    pos = bs + dist_true * np.array([np.sin(th_q), np.cos(th_q)])   # range assumed known
    return Estimate(pos=pos, speed=0.0, theta=th_q, dist=dist_true, C=np.zeros((3, 3)))
