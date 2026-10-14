"""
#############################################################################
#  DRAFT / PLACEHOLDER — CRUDE ARTIFICIAL-NOISE (AN) MATRIX FORMULATION      #
#############################################################################
The latest draft has the heading "Artificial noise matrix formulation" but no
content yet, plus a TODO "> power budget for artificial noise". This module is a
placeholder implementing the construction from the previous draft revision:

    P_null = I - A_K (A_K^H A_K)^{-1} A_K^H,      A_K = [a(theta_1) ... a(theta_K)]
    a_tilde(theta) = P_null a(theta)
    S = sum_j sum_m q_{j,m} a_tilde(theta_{j,m}) a_tilde^H(theta_{j,m}),  q >= 0

so a^H(theta_k) S a(theta_k) = 0 for every *predicted* user angle (open draft
comment: "By construction, a^H(theta_k) S a(theta_k) = 0"). It leaks onto users
whenever the predicted angle differs from the true one — metrics.py accounts for
that leakage, which the draft SINR currently omits.

NEW constraint added here (requested): AN power budget
    tr(S) = sum_{j,m} q_{j,m} ||a_tilde_{j,m}||^2  <=  P_AN_max
in addition to the total budget sum_k ||f_k||^2 + tr(S) <= P_max.

Placeholders / TODO:
  [ ] allocation rules below are heuristics, not derived from the secrecy objective
      (the optimiser in solvers.py does optimise q, but over these fixed directions);
  [ ] sampling of E_j uses the (theta, d) projection only;
  [ ] when K >= Nt the null space is empty -> no AN possible (handled, warned);
  [ ] general S >= 0 (full PSD design / SDR) instead of rank-one sums;
  [ ] robust version: null space of the *uncertainty region* of users too.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .draft import draft
from .geometry import steering

_MSG = ("crude AN covariance S = sum q a_tilde a_tilde^H in null space of predicted user "
        "steering vectors, heuristic allocation — see an.py header TODO list")


def null_projector(theta_users: np.ndarray, Nt: int) -> np.ndarray:
    A = steering(theta_users, Nt)
    return np.eye(Nt) - A @ np.linalg.pinv(A)       # = I - A (A^H A)^{-1} A^H


@dataclass
class ANBasis:
    """Fixed AN directions a_tilde_m and which Eve (owner) each belongs to."""
    dirs: np.ndarray          # (Nt, n_dir)
    owner: np.ndarray         # (n_dir,) eve index j
    theta: np.ndarray         # (n_dir,) sampled angle
    norms2: np.ndarray        # ||a_tilde_m||^2

    @property
    def n(self) -> int:
        return self.dirs.shape[1]

    def covariance(self, q: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros((self.dirs.shape[0],) * 2, complex)
        return (self.dirs * q[None, :]) @ self.dirs.conj().T

    def power(self, q: np.ndarray) -> float:
        return float(np.sum(q * self.norms2)) if self.n else 0.0


@draft("AN", _MSG)
def build_an_basis(theta_users_pred: np.ndarray, eve_theta_samples: list[np.ndarray],
                   Nt: int, min_norm2: float = 1e-6) -> ANBasis:
    P = null_projector(theta_users_pred, Nt)
    dirs, owner, th = [], [], []
    for j, ths in enumerate(eve_theta_samples):
        for t in np.unique(np.round(ths, 6)):
            a_t = P @ steering(t, Nt)[:, 0]
            if np.vdot(a_t, a_t).real > min_norm2 * Nt:
                dirs.append(a_t); owner.append(j); th.append(t)
    if not dirs:
        return ANBasis(np.zeros((Nt, 0), complex), np.zeros(0, int), np.zeros(0), np.zeros(0))
    D = np.stack(dirs, axis=1)
    return ANBasis(D, np.array(owner), np.array(th), np.sum(np.abs(D) ** 2, axis=0))


@draft("AN", _MSG)
def crude_an_allocation(basis: ANBasis, P_an: float, mode: str = "equal") -> np.ndarray:
    """Heuristic q given an AN power P_an (<= P_AN_max):
       'equal'     : equal *radiated* power per direction
       'per_eve'   : equal power per Eve, split equally over its samples
    """
    if basis.n == 0 or P_an <= 0:
        return np.zeros(basis.n)
    if mode == "equal":
        w = np.ones(basis.n)
    elif mode == "per_eve":
        counts = np.bincount(basis.owner)
        w = 1.0 / counts[basis.owner]
    else:
        raise ValueError(mode)
    return P_an * w / np.sum(w * basis.norms2)


@draft("AN-ISO", "isotropic null-space AN (Goel & Negi style), needs no Eve CSI — benchmark only")
def isotropic_null_an(theta_users_pred: np.ndarray, Nt: int, P_an: float) -> np.ndarray:
    P = null_projector(theta_users_pred, Nt)
    rank = max(int(round(np.real(np.trace(P)))), 1)
    return (P_an / rank) * P
