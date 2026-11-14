"""
Artificial noise (AN) in the null space of the legitimate users' channels, following
N. Su, F. Liu, C. Masouros, "Sensing-Assisted Eavesdropper Estimation: An ISAC
Breakthrough in Physical Layer Security" (arXiv:2210.08286), Sec. III and III-A,
Eqs. (19)-(23):

    n_AN = V n_bar,        V = I - A (A^H A)^{-1} A^H          (orthogonal projector onto
                                                                null{a^H(theta_k)}, k in K)
    S = E[n_AN n_AN^H] = V R_nbar V^H,        R_nbar >= 0

    * R_nbar = I  : isotropic AN, used when the Eves are unknown (Su Sec. III);
    * R_nbar chosen from Eve information : "AN refinement based on Eves' information"
      (Su Sec. III-A). Here the information is the uncertainty region E_j of every Eve
      (Report Sec. VII-A: "construct the AN explicitly in the null space of every
      legitimate user's channel"), and R_nbar is parametrised on the span of the
      steering vectors of the region samples plus an isotropic part:

          R_nbar = sum_{j,m} q_{j,m} a(theta_{j,m}) a^H(theta_{j,m}) + q_0 I,   q >= 0
          =>  S = sum_{j,m} q_{j,m} a~_{j,m} a~_{j,m}^H + q_0 V,   a~ = V a.

    The q are chosen by the optimiser (solvers.py); every atom is an exact PSD piece of
    S, so S >= 0 and a^H(theta_k) S a(theta_k) = 0 for the PREDICTED user angles.

Users' channel h_k = h_k a^H(theta_k) (Report Eq. 12), so A = [a(theta_1) ... a(theta_K)]
and V is exactly Su's H^H(HH^H)^{-1}H complement.

Constraints on the AN (Report Sec. VII, and "power budget for artificial noise"):
    total  sum_k ||f_k||^2 + tr(S) <= P_max,     AN budget  tr(S) <= P_AN,max.
The null space has dimension Nt - K; if K >= Nt there is no null space and no AN
(handled, atoms list is empty).

Leakage. The projector is built from the predicted angles, so a user's TRUE steering
vector is not exactly in the span: the leakage a^H(theta_k) S a(theta_k) is O(delta^2)-
small, and is reported by `metrics.evaluate_true` as `an_leak_users` (W received at
user k). `deriv_null=True` additionally nulls the derivative d a / d theta at the predicted
angles (A <- [A, A']), which makes the leakage fourth order in the angle error at the
price of K more nulled dimensions; see CHANGELOG for the measured numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geometry import steering


def steering_derivative(theta, n: int) -> np.ndarray:
    """d a(theta)/d theta, shape (n, len(theta))."""
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    k = np.arange(n)[:, None]
    return 1j * np.pi * k * np.cos(th)[None, :] * steering(th, n)


def null_projector(theta_users: np.ndarray, Nt: int, deriv_null: bool = False) -> np.ndarray:
    """V = I - A (A^H A)^{-1} A^H  (Su Eq. 19); optionally A = [a, da/dtheta]."""
    A = steering(theta_users, Nt)
    if deriv_null:
        A = np.concatenate([A, steering_derivative(theta_users, Nt)], axis=1)
    if A.shape[1] == 0:
        return np.eye(Nt, dtype=complex)
    return np.eye(Nt) - A @ np.linalg.pinv(A)       # = I - A (A^H A)^{-1} A^H


def null_space_basis(V: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis Z of range(V) (V is a projector), Z Z^H = V."""
    w, U = np.linalg.eigh((V + V.conj().T) / 2)
    return U[:, w > 1 - tol]


@dataclass
class ANBasis:
    """PSD atoms S_m = D_m D_m^H of the AN covariance S = sum_m q_m S_m.
    owner[m] = Eve index j of the region sample the atom comes from, -1 for the
    isotropic atom (D = orthonormal basis of the null space, S_m = V)."""
    factors: list = field(default_factory=list)      # list of (Nt, r_m) arrays
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    norms2: np.ndarray = field(default_factory=lambda: np.zeros(0))   # tr(S_m) = ||D_m||_F^2
    Nt: int = 0

    @property
    def n(self) -> int:
        return len(self.factors)

    @property
    def has_iso(self) -> bool:
        return bool(self.n and np.any(self.owner < 0))

    def covariance(self, q: np.ndarray) -> np.ndarray:
        S = np.zeros((self.Nt, self.Nt), complex)
        for qm, D in zip(q, self.factors):
            if qm:
                S += qm * (D @ D.conj().T)
        return S

    def power(self, q: np.ndarray) -> float:
        return float(np.sum(q * self.norms2)) if self.n else 0.0

    def gains(self, A: np.ndarray) -> np.ndarray:
        """(P, n): a_p^H S_m a_p for the columns a_p of A, i.e. ||A^H D_m||_F^2."""
        if not self.n:
            return np.zeros((A.shape[1], 0))
        return np.stack([np.sum(np.abs(A.conj().T @ D) ** 2, axis=1) for D in self.factors], axis=1)


def build_an_basis(theta_users_pred: np.ndarray, eve_theta_samples: list,
                   Nt: int, min_norm2: float = 1e-6, with_iso: bool = True,
                   deriv_null: bool = False) -> ANBasis:
    """Atoms of S = V R_nbar V^H: one per (rounded) Eve-region sample angle plus,
    if `with_iso`, the isotropic atom V."""
    V = null_projector(theta_users_pred, Nt, deriv_null)
    factors, owner, th = [], [], []
    for j, ths in enumerate(eve_theta_samples):
        for t in np.unique(np.round(ths, 6)):
            a_t = (V @ steering(t, Nt)[:, 0])[:, None]
            if np.vdot(a_t, a_t).real > min_norm2 * Nt:
                factors.append(a_t); owner.append(j); th.append(t)
    if with_iso:
        Z = null_space_basis(V)
        if Z.shape[1] > 0:
            factors.append(Z); owner.append(-1); th.append(np.nan)
    if not factors:
        return ANBasis(Nt=Nt)
    return ANBasis(factors, np.array(owner), np.array(th),
                   np.array([np.sum(np.abs(D) ** 2) for D in factors]), Nt)


def an_allocation(basis: ANBasis, P_an: float, iso_frac: float = 0.0,
                  mode: str = "per_eve") -> np.ndarray:
    """Coefficients q for a total AN power tr(S) = P_an (<= P_AN,max):
    a fraction `iso_frac` goes to the isotropic atom, the rest to the Eve-region atoms
      'per_eve': equal weight per Eve, split equally over its samples
      'equal'  : equal weight per sample
    (heuristic split inside the Eve atoms; the optimiser searches over `iso_frac`
    and P_an, and SLSQP refines every q_m)."""
    q = np.zeros(basis.n)
    if basis.n == 0 or P_an <= 0:
        return q
    iso = basis.owner < 0
    eve = ~iso
    if not eve.any():
        iso_frac = 1.0
    if not iso.any():
        iso_frac = 0.0
    if iso.any() and iso_frac > 0:
        q[iso] = iso_frac * P_an / basis.norms2[iso].sum()
    if eve.any() and iso_frac < 1:
        if mode == "equal":
            w = np.ones(eve.sum())
        elif mode == "per_eve":
            counts = np.bincount(basis.owner[eve])
            w = 1.0 / counts[basis.owner[eve]]
        else:
            raise ValueError(mode)
        q[eve] = (1 - iso_frac) * P_an * w / np.sum(w * basis.norms2[eve])
    return q


def isotropic_null_an(theta_users_pred: np.ndarray, Nt: int, P_an: float,
                      deriv_null: bool = False) -> np.ndarray:
    """Su Sec. III benchmark, R_nbar = (P_an / tr V) I:  S = (P_an / rank V) V.
    Needs no Eve information."""
    V = null_projector(theta_users_pred, Nt, deriv_null)
    rank = max(int(round(np.real(np.trace(V)))), 1)
    return (P_an / rank) * V
