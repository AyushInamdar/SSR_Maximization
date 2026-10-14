"""
Simulation parameters for Sum-Secrecy Rate Maximization in OTFS-Assisted ISAC.

Every parameter is tagged with where it comes from:
  [Yuan21]  W. Yuan et al., "Integrated Sensing and Communication-assisted OTFS
            Transmission for Vehicular Networks", arXiv:2105.03125, Sec. IV.
  [Draft]   the UGP draft (system model / problem formulation).
  [ASSUMED] not given in either source -- placeholder value, revisit before
            reporting any numbers.

Notes on [Yuan21]:
  * dT = 0.02 s is the duration of one *time instant* (tracking step). It is NOT
    the OTFS frame length N*T = N/delta_f = 5 ms. The draft currently uses
    "Delta T" for both (Sec. IV uses Delta T = NT, Sec. VI uses it as the
    prediction step). They are kept separate here: `frame_duration` vs `dT`.
  * Transmit power, RCS, noise PSD and the matched-filter gain Gm are not
    reported numerically in [Yuan21]; they are [ASSUMED] below.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

C_LIGHT = 299_792_458.0
K_BOLTZ_DBM_HZ = -174.0  # thermal noise PSD at 290 K, dBm/Hz


@dataclass
class SimParams:
    # ------------------------------------------------------------------ [Yuan21]
    fc: float = 3e9                 # carrier frequency (Hz)
    delta_f: float = 6e3            # subcarrier spacing (Hz)
    M: int = 128                    # subcarriers (delay bins)
    N: int = 30                     # time slots (Doppler bins)
    Nt: int = 64                    # transmit ULA size (paper sweeps 16..128)
    Nr: int = 64                    # receive ULA size
    dT: float = 0.02                # duration of one tracking time instant (s)
    n_frames: int = 200             # number of time instants
    sigma2_radar: float = 1.0       # matched-filtered echo noise variance (normalised)
    bs_pos: tuple = (0.0, 0.0)      # RSU / BS position (m)
    user_init_pos: tuple = ((-30.0, 50.0), (20.0, 50.0), (5.0, 20.0), (40.0, 20.0))
    speed_range: tuple = (10.0, 15.0)   # |speed| ~ U(10, 15) m/s, random sign
    k_max_dl: int = 3               # max downlink Doppler index
    l_max: int = 10                 # max delay index (used for CP / guard sizing)

    # ----------------------------------------------------------------- [ASSUMED]
    # Eavesdroppers (radar targets J). Placed near user angles so that AN matters.
    eve_init_pos: tuple = ((-14.0, 45.0), (30.0, 22.0))
    eve_speed_range: tuple = (8.0, 14.0)
    speed_process_std: float = 0.15     # m/s random walk per time instant (manoeuvres)
    rcs: float = 1.0                    # xi in beta = xi / (2 d)   (Draft Sec. VI)
    Gm: float | None = None             # MF gain; None -> M*N (energy of s_i, unit symbols)
    P_max_dBm: float = 0.0              # total BS transmit power budget (sets link SNR; tune)
    P_an_max_frac: float = 0.3          # NEW: AN power budget as fraction of P_max
    P_probe_dBm: float = 0.0            # isotropic sensing probe power
    noise_figure_dB: float = 9.0        # receiver NF (users and Eves)
    pathloss_model: str = "friis"       # "friis": |h|^2=(c/4 pi fc d)^2 ; "draft": c/(4 pi fc d^2)
    R_min: float = 0.5                  # bits/s/Hz per user QoS
    gamma_sens_dBm: float = 0.0         # min beampattern gain a^H R a toward every target
    # CRB requirement C_j^req on eta_j = [gamma (s), omega (Hz), theta (rad)]
    crb_req_range_m: float = 0.5        # -> std(gamma) <= 2*0.5/c
    crb_req_doppler_hz: float = 5.0
    crb_req_theta_deg: float = 0.5
    eps_conf: float = 7.815             # chi^2_3 95% -> ellipsoid E_j size
    n_unc_samples: int = 6              # boundary samples per uncertainty region
    theta_search_halfwidth: float = 0.15    # rad, ML search window around prediction
    theta_grid_res: float = 2e-4            # rad
    feedback_overhead: float = 0.10     # pilot + feedback fraction (feedback benchmark)
    feedback_codebook_oversample: int = 2   # DFT codebook of oversample*Nt beams (feedback)
    seed: int = 0

    # ------------------------------------------------------------- derived ----
    @property
    def T(self) -> float:
        return 1.0 / self.delta_f

    @property
    def frame_duration(self) -> float:
        return self.N * self.T

    @property
    def bandwidth(self) -> float:
        return self.M * self.delta_f

    @property
    def fs(self) -> float:
        return self.M * self.delta_f

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.fc

    @property
    def Gm_eff(self) -> float:
        return float(self.M * self.N) if self.Gm is None else float(self.Gm)

    @property
    def P_max(self) -> float:
        return dbm_to_w(self.P_max_dBm)

    @property
    def P_an_max(self) -> float:
        return self.P_an_max_frac * self.P_max

    @property
    def radar_power_ref(self) -> float:
        """Radar link uses the normalised model of Yuan21/Liu20 (sigma^2 = 1, unitless
        powers). Physical illumination a^H R_x a (W) is divided by this reference
        (= P_max) before entering the echo SNR. [ASSUMED normalisation]"""
        return self.P_max

    @property
    def P_probe(self) -> float:
        return dbm_to_w(self.P_probe_dBm)

    @property
    def N0(self) -> float:
        """Noise power over the signal bandwidth (W)."""
        return dbm_to_w(K_BOLTZ_DBM_HZ + 10 * np.log10(self.bandwidth) + self.noise_figure_dB)

    @property
    def gamma_sens(self) -> float:
        return dbm_to_w(self.gamma_sens_dBm)

    @property
    def crb_req(self) -> np.ndarray:
        """C_j^req as a diagonal 3x3 matrix on [gamma, omega, theta]."""
        s_gamma = 2 * self.crb_req_range_m / C_LIGHT
        return np.diag([s_gamma ** 2, self.crb_req_doppler_hz ** 2,
                        np.deg2rad(self.crb_req_theta_deg) ** 2])

    @property
    def K(self) -> int:
        return len(self.user_init_pos)

    @property
    def J(self) -> int:
        return len(self.eve_init_pos)

    def as_dict(self) -> dict:
        return asdict(self)


def dbm_to_w(x_dbm: float) -> float:
    return 10 ** ((x_dbm - 30) / 10)


def w_to_dbm(x_w: float) -> float:
    return 10 * np.log10(np.maximum(x_w, 1e-30)) + 30


PARAM_SOURCES = {
    "fc": "Yuan21", "delta_f": "Yuan21", "M": "Yuan21", "N": "Yuan21",
    "Nt": "Yuan21", "Nr": "Yuan21", "dT": "Yuan21", "n_frames": "Yuan21",
    "sigma2_radar": "Yuan21 (from Liu et al. [29])", "bs_pos": "Yuan21",
    "user_init_pos": "Yuan21", "speed_range": "Yuan21", "k_max_dl": "Yuan21",
    "l_max": "Yuan21",
    "eve_init_pos": "ASSUMED", "eve_speed_range": "ASSUMED",
    "speed_process_std": "ASSUMED", "rcs": "ASSUMED", "Gm": "ASSUMED (M*N)",
    "P_max_dBm": "ASSUMED", "P_an_max_frac": "ASSUMED (new AN budget)",
    "P_probe_dBm": "ASSUMED", "noise_figure_dB": "ASSUMED",
    "pathloss_model": "ASSUMED (draft formula available)", "R_min": "ASSUMED",
    "gamma_sens_dBm": "ASSUMED", "crb_req_range_m": "ASSUMED",
    "crb_req_doppler_hz": "ASSUMED", "crb_req_theta_deg": "ASSUMED",
    "eps_conf": "ASSUMED (chi2_3, 95%)", "n_unc_samples": "ASSUMED",
    "theta_search_halfwidth": "ASSUMED", "theta_grid_res": "ASSUMED",
    "feedback_overhead": "ASSUMED", "feedback_codebook_oversample": "ASSUMED", "seed": "-",
}
