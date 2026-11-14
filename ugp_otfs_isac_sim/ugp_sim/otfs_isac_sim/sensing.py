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
    around the predicted angle (Draft Sec. VI: search only near theta_bar), refined by
    a parabola through the peak and its two neighbours (the grid step is ~10x the CRB
    std at the operating SNR, so the raw grid maximum is quantisation-limited).
    NOTE: Draft/Yuan21 use the grid Theta = {pi/Nt, ..., pi}, which is too coarse
    (~0.05 rad at Nt=64) and in [0, pi] while theta lies in (-pi/2, pi/2) under the
    sin(theta)=x/d convention; a fine local grid is used here."""
    hw, res = p.theta_search_halfwidth, p.theta_grid_res
    grid = np.arange(theta_center - hw, theta_center + hw + res / 2, res)
    grid = grid[np.abs(grid) < np.pi / 2]
    metric = np.abs(steering(grid, p.Nr).conj().T @ r) ** 2
    i = int(np.argmax(metric))
    if p.ml_angle_interp and 0 < i < grid.size - 1:
        y0, y1, y2 = metric[i - 1], metric[i], metric[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(grid[i] + 0.5 * res * (y0 - y2) / den)
    return float(grid[i])


def estimate_delay_doppler(gamma: float, omega: float, C: np.ndarray,
                           rng: np.random.Generator) -> tuple[float, float]:
    """Delay/Doppler estimate modelled as an EFFICIENT estimator: N(truth, CRB), instead
    of running the 2-D (gamma, omega) matched-filter search of Draft Sec. IV every frame.
    Justification (not a placeholder any more): the local ML estimator attains the CRB of
    Eq. (66) (MSE/CRB = 1.0-1.12 at 10-40 dB, tests/test_core.py::test_ml_estimates_approach_crb)
    and the simulated radar SNR is far above that range. Not modelled: ambiguity/sidelobe
    outliers at low SNR."""
    return (gamma + rng.normal(0, np.sqrt(C[0, 0])),
            omega + rng.normal(0, np.sqrt(C[1, 1])))


def ml_estimate_dd_angle(r: np.ndarray, s0: np.ndarray, fs: float, x0, scale,
                        max_iter: int = 400) -> np.ndarray:
    """Local ML estimate of [gamma, omega, theta] from the Nr x L echo r(t) = h b(theta)
    s0(t-gamma) e^{j 2 pi omega t} + z (Appendix A, Eq. 31), nuisance h concentrated out:

        max  | sum_n b^H(theta) r[:, n] conj(s0(t_n - gamma)) e^{-j 2 pi omega t_n} |^2.

    Nelder-Mead from the start point `x0` (in the paper's flow the coarse search leaves
    the maximum within a fraction of a resolution cell), `scale` = parameter scales.
    Used to validate that the CRB of Eq. (66) is attained at high SNR, i.e. that
    `estimate_delay_doppler`'s N(truth, CRB) model is a fair stand-in for the matched
    filter (tests/test_core.py::test_ml_estimates_approach_crb)."""
    from scipy.optimize import minimize
    from .waveform import bandlimited_shift
    Nr, L = r.shape
    t = np.arange(L) / fs
    scale = np.asarray(scale, float)

    def cost(z):
        g, w, th = np.asarray(x0, float) + z * scale
        x = steering(th, Nr)[:, 0].conj() @ r
        y = x * np.conj(bandlimited_shift(s0, fs, g)) * np.exp(-1j * 2 * np.pi * w * t)
        return -np.abs(y.sum()) ** 2

    res = minimize(cost, np.zeros(3), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-12 * abs(cost(np.zeros(3))) + 1e-300,
                                maxiter=max_iter, initial_simplex=np.vstack([np.zeros(3), np.eye(3) * 0.5])))
    return np.asarray(x0, float) + res.x * scale


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


def eta_from_state(pos: np.ndarray, speed: float, theta: float, bs: np.ndarray,
                   fc: float) -> np.ndarray:
    """eta = [gamma (s), omega (Hz), theta (rad)] of a target at `pos` moving along x at
    `speed` (round-trip delay 2d/c, round-trip Doppler 2 nu = -2 v sin(theta) fc / c)."""
    d = float(np.linalg.norm(pos - bs))
    return np.array([2 * d / C_LIGHT, -2 * speed * np.sin(theta) * fc / C_LIGHT, theta])


def process_noise(speed: float, theta: float, dist: float, p: SimParams) -> np.ndarray:
    """Diagonal process noise Q (variances of [gamma, omega, theta]) added per time
    instant, from the motion model of `geometry.Scenario.step`: the speed performs a
    random walk with std `speed_process_std` per instant, so over dT the position is
    uncertain by sigma_x = sigma_v dT along x and the speed by sigma_v. Mapped to
    eta by the local Jacobian (d_gamma = 2 sin(th) dx / c, d_theta = cos(th) dx / d,
    d_omega = -2 fc/c [sin(th) dv + v cos^2(th) dx / d]); scaled by track_q_scale^2."""
    sx = p.speed_process_std * p.dT
    sv = p.speed_process_std
    st, ct = np.sin(theta), np.cos(theta)
    d = max(dist, 1.0)
    q_g = (2.0 / C_LIGHT * st * sx) ** 2
    q_w = (2 * p.fc / C_LIGHT) ** 2 * ((st * sv) ** 2 + (speed * ct ** 2 * sx / d) ** 2)
    q_t = (ct * sx / d) ** 2
    return np.array([q_g, q_w, q_t]) * p.track_q_scale ** 2


@dataclass
class Estimate:
    pos: np.ndarray          # (2,)
    speed: float
    theta: float
    dist: float
    C: np.ndarray            # 3x3 covariance of eta = [gamma, omega, theta]: posterior after
                             # fusing this instant's echo, or predicted (+Q) after predict()
    C_meas: np.ndarray | None = None   # single-instant CRB of the raw measurement (diagnostics)

    def eta_for(self, p: SimParams) -> np.ndarray:
        bs = np.asarray(p.bs_pos, dtype=float)
        return eta_from_state(self.pos, self.speed, self.theta, bs, p.fc)

    def predict(self, p: SimParams) -> "Estimate":
        """Draft: q_bar = q_hat + dT * s_hat * [1, 0]^T, theta_bar = arcsin(...).
        Time update of the information filter: P_pred = P_post + Q (per parameter);
        skipped for genie estimates (C = 0) and when track_accumulate is off."""
        bs = np.asarray(p.bs_pos, dtype=float)
        q = self.pos + p.dT * self.speed * np.array([1.0, 0.0])
        th = float(angle_from_pos(q, bs))
        d = float(np.linalg.norm(q - bs))
        C = self.C
        if p.track_accumulate and np.any(self.C):
            C = self.C + np.diag(process_noise(self.speed, th, d, p))
        return Estimate(pos=q, speed=self.speed, theta=th, dist=d, C=C, C_meas=self.C_meas)


def sense_target(theta_true: float, dist_true: float, omega_true: float, illum: float,
                 prior: "Estimate | None", p: SimParams,
                 rng: np.random.Generator, Gm: float | None = None) -> Estimate:
    """One sensing step for one target: echo -> ML angle -> delay/Doppler -> measurement
    eta_m with CRB C_m (Eq. 66) -> information-filter fusion with the prediction `prior`

        P_pred = P_prev + Q                (done in Estimate.predict)
        1/P_post = 1/P_pred + J_meas,      eta_post = P_post (eta_pred/P_pred + J_meas eta_m)

    per parameter (J_meas = 1/C_m diagonal, Eq. 66) -> location & speed from eta_post.
    With track_accumulate = False (or no prior) the raw single-echo estimate and CRB are
    returned, i.e. the region never shrinks over time."""
    Gm = p.Gm_eff if Gm is None else Gm
    bs = np.asarray(p.bs_pos, dtype=float)
    r = simulate_echo(theta_true, dist_true, illum, p, rng, Gm)
    theta_prior = None if prior is None else prior.theta
    if theta_prior is None:            # no prediction yet (isotropic probe): coarse -> fine
        coarse = np.arange(-1.45, 1.45, 2.0 / p.Nr)
        theta_prior = float(coarse[np.argmax(np.abs(steering(coarse, p.Nr).conj().T @ r))])
    th = ml_angle(r, theta_prior, p)
    # CRB (Eq. 66) evaluated with the matched-filter gain actually used
    snr_el = (Gm * p.rcs / (2 * dist_true)) ** 2 * (illum / p.radar_power_ref) / p.sigma2_radar
    C_m = crbmod.crb_matrix(snr_el, theta_true, p)
    g_hat, w_hat = estimate_delay_doppler(2 * dist_true / C_LIGHT, omega_true, C_m, rng)
    eta_m = np.array([g_hat, w_hat, th])
    if p.track_accumulate and prior is not None and np.any(prior.C):
        P_pred = np.diag(prior.C).copy()
        J_m = 1.0 / np.diag(C_m)
        P_post = 1.0 / (1.0 / P_pred + J_m)
        eta = P_post * (prior.eta_for(p) / P_pred + J_m * eta_m)
        C = np.diag(P_post)
    else:
        eta, C = eta_m, C_m
    g, w, th = float(eta[0]), float(eta[1]), float(eta[2])
    q = localise(g, th, bs)
    v = speed_from_doppler(w, th, p.fc, None if prior is None else prior.speed)
    return Estimate(pos=q, speed=v, theta=th, dist=float(np.linalg.norm(q - bs)), C=C, C_meas=C_m)


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
