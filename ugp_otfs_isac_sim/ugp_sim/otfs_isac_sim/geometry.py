"""
Road geometry, vehicle kinematics and ULA steering vectors.

Conventions (consistent with Draft Sec. III/VI and Yuan21 eqs. 6, 26-27, 30):
  * road = x-axis direction, BS at q_BS, vehicles move along +/- x at fixed y.
  * theta is measured from the array broadside (y-axis): sin(theta) = (x - x_BS)/d.
  * a_n(theta) = exp(j*pi*(n-1)*sin(theta))  (half-wavelength ULA).
  * Communication Doppler nu = -v * sin(theta) * fc / c  (positive when approaching),
    round-trip radar Doppler omega = 2 * nu.
    NOTE: Draft/Yuan21 eq. (28) write s = c*omega/(fc*cos(theta)). With the
    sin(theta)=x/d angle convention used everywhere else, the radial velocity is
    v*sin(theta), and omega is round-trip (factor 2). The geometrically
    consistent form is used here; see sensing.speed_from_doppler.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimParams, C_LIGHT


def steering(theta, n: int) -> np.ndarray:
    """ULA steering matrix, shape (n, len(theta)). Unnormalised (||a||^2 = n)."""
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.exp(1j * np.pi * np.arange(n)[:, None] * np.sin(th)[None, :])


def steer1(theta: float, n: int) -> np.ndarray:
    return steering(theta, n)[:, 0]


def angle_from_pos(pos: np.ndarray, bs: np.ndarray) -> np.ndarray:
    d = pos - bs
    return np.arcsin(np.clip(d[..., 0] / np.linalg.norm(d, axis=-1), -1, 1))


def path_gain(d, p: SimParams) -> np.ndarray:
    """|h|^2 of the LoS downlink channel."""
    d = np.asarray(d, dtype=float)
    if p.pathloss_model == "friis":
        return (C_LIGHT / (4 * np.pi * p.fc * d)) ** 2
    if p.pathloss_model == "draft":
        # Draft / Yuan21 eq. (12): h = sqrt(c / (4 pi fc d^2)). Dimensionally this is
        # not Friis; kept for reproducing the paper's expressions.
        return C_LIGHT / (4 * np.pi * p.fc * d ** 2)
    raise ValueError(p.pathloss_model)


@dataclass
class Scenario:
    """True states of all P = K + J targets. Index 0..K-1 users, K..K+J-1 Eves."""
    pos: np.ndarray        # (P, 2)
    vel: np.ndarray        # (P,)  signed speed along x
    K: int
    J: int
    params: SimParams

    @classmethod
    def from_params(cls, p: SimParams, rng: np.random.Generator) -> "Scenario":
        pos = np.array(list(p.user_init_pos) + list(p.eve_init_pos), dtype=float)
        lo, hi = p.speed_range
        v_u = rng.uniform(lo, hi, p.K) * rng.choice([-1, 1], p.K)
        lo, hi = p.eve_speed_range
        v_e = rng.uniform(lo, hi, p.J) * rng.choice([-1, 1], p.J)
        return cls(pos=pos, vel=np.concatenate([v_u, v_e]), K=p.K, J=p.J, params=p)

    @property
    def P(self) -> int:
        return self.K + self.J

    @property
    def bs(self) -> np.ndarray:
        return np.asarray(self.params.bs_pos, dtype=float)

    @property
    def dist(self) -> np.ndarray:
        return np.linalg.norm(self.pos - self.bs, axis=1)

    @property
    def theta(self) -> np.ndarray:
        return angle_from_pos(self.pos, self.bs)

    @property
    def gamma(self) -> np.ndarray:
        """Round-trip delay (s)."""
        return 2 * self.dist / C_LIGHT

    @property
    def nu(self) -> np.ndarray:
        """One-way (communication) Doppler (Hz)."""
        return -self.vel * np.sin(self.theta) * self.params.fc / C_LIGHT

    @property
    def omega(self) -> np.ndarray:
        """Round-trip radar Doppler (Hz)."""
        return 2 * self.nu

    @property
    def h2(self) -> np.ndarray:
        return path_gain(self.dist, self.params)

    def step(self, rng: np.random.Generator) -> None:
        p = self.params
        self.vel = self.vel + rng.normal(0, p.speed_process_std, self.P)
        self.pos[:, 0] += self.vel * p.dT

    def users(self) -> slice:
        return slice(0, self.K)

    def eves(self) -> slice:
        return slice(self.K, self.K + self.J)
