"""
Problem instance (built from PREDICTED states) and a registry of solvers for

  max_{F, q >= 0}  sum_k R_k^sec
  s.t.  sum_k ||f_k||^2 + tr(S)            <= P_max                (C1 total power)
        tr(S) = sum q ||a_tilde||^2         <= P_AN_max             (C2 NEW: AN budget)
        R_k                                 >= R_min      for all k  (C3 QoS)
        a^H(th_i) (sum f f^H + S) a(th_i)   >= Gamma_sens for all i in P (C4 sensing)
        I^{-1}(eta_j)                       <= C_j^req   for all j  (C5 CRB, crude)
  with  R_k^sec = [R_k - max_j max_{eta in E_j} R_{j,k}^Eve]^+.

Parametrisation used by every solver here: f_k = sqrt(p_k) w_k with unit-norm
direction w_k (MRT = draft's a(theta_k)/sqrt(Nt), or ZF), and S over the fixed AN
basis of an.py. Then C1, C2, C4 and the crude C5 are LINEAR in x = (p, q), which is
what makes the simple solvers below possible.

Solver interface:  solve(prob: Problem) -> Solution.   Register with @register.

  name             what it is                                          status
  ---------------  --------------------------------------------------  ----------
  mrt_no_an        draft beams, equal power, no AN                      baseline
  mrt_crude_an     draft beams + crude AN at full AN budget             baseline
  zf_crude_an      ZF beams + crude AN at full AN budget                baseline
  isotropic_an     MRT + isotropic null-space AN (no Eve CSI)           baseline
  grid_split       1-D search of AN fraction x {MRT, ZF}                heuristic
  slsqp            local NLP (SciPy SLSQP) over (p, q), smoothed obj.  heuristic
  sca_sdr          SDR/SCA on full F, S (needs cvxpy)                   STUB
  drl              DDPG/SAC agent                                       STUB
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
from scipy.optimize import minimize

from .config import SimParams
from .geometry import steering
from .waveform import effective_sinr
from .an import ANBasis, crude_an_allocation, isotropic_null_an

SOLVERS: dict = {}


def register(name: str):
    def deco(fn):
        SOLVERS[name] = fn
        return fn
    return deco


def get_solver(name: str):
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(SOLVERS)}")
    return SOLVERS[name]


# -----------------------------------------------------------------------------
@dataclass
class Problem:
    p: SimParams
    waveform: str
    prefactor: float
    th_u: np.ndarray          # predicted user angles (K)
    h2_u: np.ndarray          # predicted user path gains
    nu_u: np.ndarray          # predicted user Doppler
    eve_th: np.ndarray        # worst-case samples over all E_j (S)
    eve_h2: np.ndarray
    eve_nu: np.ndarray
    eve_owner: np.ndarray
    eve_center_th: np.ndarray  # (J)
    sens_th: np.ndarray        # predicted angles of all P targets
    illum_req_eve: np.ndarray  # (J) min a^H R a for crude CRB requirement
    basis: ANBasis

    @property
    def K(self):
        return len(self.th_u)

    @property
    def Nt(self):
        return self.p.Nt


@dataclass
class Solution:
    F: np.ndarray
    S: np.ndarray
    p: np.ndarray
    q: np.ndarray
    name: str
    info: dict = field(default_factory=dict)


def directions(prob: Problem, mode: str) -> np.ndarray:
    A = steering(prob.th_u, prob.Nt)
    if mode == "mrt":
        W = A
    elif mode == "zf":
        W = A @ np.linalg.pinv(A.conj().T @ A)
    else:
        raise ValueError(mode)
    return W / np.linalg.norm(W, axis=0, keepdims=True)


class LinearModel:
    """All received powers as linear maps of x = (p, q) for fixed directions W."""

    def __init__(self, prob: Problem, W: np.ndarray, S_fixed: np.ndarray | None = None):
        self.prob, self.W = prob, W
        D = prob.basis.dirs
        Nt = prob.Nt
        S0 = np.zeros((Nt, Nt), complex) if S_fixed is None else S_fixed
        self.S_fixed = S0

        def blocks(th, h2=None):
            A = steering(th, Nt)
            g = np.abs(A.conj().T @ W) ** 2
            l = np.abs(A.conj().T @ D) ** 2 if D.shape[1] else np.zeros((len(th), 0))
            c = np.real(np.einsum("ip,ij,jp->p", A.conj(), S0, A))
            if h2 is not None:
                g, l, c = g * h2[:, None], l * h2[:, None], c * h2
            return g, l, c

        self.Gu, self.Lu, self.cu = blocks(prob.th_u, prob.h2_u)
        self.Ge, self.Le, self.ce = blocks(prob.eve_th, prob.eve_h2)
        self.Gs, self.Ls, self.cs = blocks(prob.sens_th)
        self.Gc, self.Lc, self.cc = blocks(prob.eve_center_th)
        self.K = prob.K

    # -- rates --------------------------------------------------------------
    def rates(self, p, q):
        P = self.prob
        sig_u = np.diag(self.Gu) * p
        int_u = self.Gu @ p - sig_u + self.Lu @ q + self.cu
        R_u = P.prefactor * np.log2(1 + effective_sinr(sig_u, int_u, P.p.N0, P.nu_u, P.waveform, P.p))
        sig_e = self.Ge * p[None, :]                                 # S x K
        int_e = (self.Ge @ p)[:, None] - sig_e + (self.Le @ q + self.ce)[:, None]
        R_e = P.prefactor * np.log2(1 + effective_sinr(sig_e, int_e, P.p.N0, P.eve_nu[:, None], P.waveform, P.p))
        return R_u, R_e

    def objective(self, p, q, tau: float | None = None):
        """Exact sum secrecy (tau=None) or smoothed version (log-sum-exp/softplus)."""
        R_u, R_e = self.rates(p, q)
        if R_e.size == 0:
            worst = np.zeros_like(R_u)
        elif tau is None:
            worst = R_e.max(0)
        else:
            m = R_e.max(0)
            worst = m + tau * np.log(np.sum(np.exp((R_e - m) / tau), axis=0))
        gap = R_u - worst
        if tau is None:
            return float(np.sum(np.maximum(gap, 0)))
        return float(np.sum(tau * np.logaddexp(0, gap / tau)))

    # -- constraints (>= 0 means satisfied) ------------------------------------
    def constraints(self, p, q) -> dict:
        P = self.prob
        an_pow = P.basis.power(q) + np.real(np.trace(self.S_fixed))
        R_u, _ = self.rates(p, q)
        return {
            "C1_total_power": P.p.P_max - (p.sum() + an_pow),
            "C2_an_budget": P.p.P_an_max - an_pow,
            "C3_qos": R_u - P.p.R_min,
            "C4_sensing": self.Gs @ p + self.Ls @ q + self.cs - P.p.gamma_sens,
            "C5_crb": self.Gc @ p + self.Lc @ q + self.cc - P.illum_req_eve,
        }

    def violation(self, p, q) -> float:
        c = self.constraints(p, q)
        scale = {"C1_total_power": self.prob.p.P_max, "C2_an_budget": self.prob.p.P_max,
                 "C3_qos": 1.0, "C4_sensing": self.prob.p.gamma_sens,
                 "C5_crb": np.maximum(self.prob.illum_req_eve, 1e-12)}
        return float(sum(np.sum(np.maximum(-np.atleast_1d(v) / scale[k], 0)) for k, v in c.items()))


def _finish(prob, W, p, q, name, lm=None, S_fixed=None, t0=None, **info) -> Solution:
    lm = lm or LinearModel(prob, W, S_fixed)
    F = W * np.sqrt(np.maximum(p, 0))[None, :]
    S = prob.basis.covariance(q) + lm.S_fixed
    cons = lm.constraints(p, q)
    info.update(obj_pred=lm.objective(p, q), violation=lm.violation(p, q),
                feasible={k: bool(np.all(np.atleast_1d(v) >= -1e-9)) for k, v in cons.items()},
                an_power=float(prob.basis.power(q) + np.real(np.trace(lm.S_fixed))),
                solve_time=(time.perf_counter() - t0) if t0 else 0.0)
    return Solution(F=F, S=S, p=p, q=q, name=name, info=info)


# ----------------------------------------------------------------- baselines --
def _equal_split(prob, mode, an_frac, name, t0):
    P = prob.p
    W = directions(prob, mode)
    P_an = min(an_frac * P.P_max, P.P_an_max) if prob.basis.n else 0.0
    q = crude_an_allocation(prob.basis, P_an, "per_eve")
    p = np.full(prob.K, (P.P_max - P_an) / prob.K)
    return _finish(prob, W, p, q, name, t0=t0)


@register("mrt_no_an")
def solve_mrt_no_an(prob: Problem) -> Solution:
    return _equal_split(prob, "mrt", 0.0, "mrt_no_an", time.perf_counter())


@register("mrt_crude_an")
def solve_mrt_crude_an(prob: Problem) -> Solution:
    return _equal_split(prob, "mrt", prob.p.P_an_max_frac, "mrt_crude_an", time.perf_counter())


@register("zf_crude_an")
def solve_zf_crude_an(prob: Problem) -> Solution:
    return _equal_split(prob, "zf", prob.p.P_an_max_frac, "zf_crude_an", time.perf_counter())


@register("isotropic_an")
def solve_isotropic_an(prob: Problem) -> Solution:
    t0 = time.perf_counter()
    P = prob.p
    W = directions(prob, "mrt")
    S_iso = isotropic_null_an(prob.th_u, prob.Nt, P.P_an_max)
    p = np.full(prob.K, (P.P_max - P.P_an_max) / prob.K)
    q = np.zeros(prob.basis.n)
    return _finish(prob, W, p, q, "isotropic_an", S_fixed=S_iso, t0=t0)


# ---------------------------------------------------------------- heuristics --
@register("grid_split")
def solve_grid_split(prob: Problem, n_grid: int = 11) -> Solution:
    """Search the AN power fraction rho in [0, P_AN_max/P_max] and beam type;
    pick max predicted objective among least-violating candidates."""
    t0 = time.perf_counter()
    P = prob.p
    best = None
    for mode in ("mrt", "zf"):
        W = directions(prob, mode)
        lm = LinearModel(prob, W)
        for rho in np.linspace(0, P.P_an_max_frac, n_grid):
            P_an = rho * P.P_max if prob.basis.n else 0.0
            q = crude_an_allocation(prob.basis, P_an, "per_eve")
            p = np.full(prob.K, (P.P_max - P_an) / prob.K)
            score = (-lm.violation(p, q), lm.objective(p, q))
            if best is None or score > best[0]:
                best = (score, W, p, q, lm, mode, rho)
    _, W, p, q, lm, mode, rho = best
    return _finish(prob, W, p, q, "grid_split", lm=lm, t0=t0, beam=mode, rho=float(rho))


@register("slsqp")
def solve_slsqp(prob: Problem, tau: float = 0.05, maxiter: int = 150) -> Solution:
    """Local NLP over x = (p, u) with u_m = radiated AN power on direction m
    (q_m = u_m / ||a_tilde_m||^2). Smoothed secrecy objective, all constraints
    C1-C5 imposed. Warm-started from grid_split; returns the better of the two."""
    t0 = time.perf_counter()
    P = prob.p
    warm = solve_grid_split(prob)
    K, nd = prob.K, prob.basis.n
    nrm = prob.basis.norms2 if nd else np.ones(0)
    best = warm
    for mode in ("mrt", "zf"):
        W = directions(prob, mode)
        lm = LinearModel(prob, W)
        s = P.P_max

        def unpack(x):
            x = np.maximum(x, 0)
            return x[:K] * s, (x[K:] * s / nrm) if nd else np.zeros(0)

        def f(x):
            pp, qq = unpack(x)
            return -lm.objective(pp, qq, tau)

        def con_vec(x):
            pp, qq = unpack(x)
            c = lm.constraints(pp, qq)
            return np.concatenate([
                [c["C1_total_power"] / s, c["C2_an_budget"] / s],
                np.atleast_1d(c["C3_qos"]),
                np.atleast_1d(c["C4_sensing"]) / P.gamma_sens,
                np.atleast_1d(c["C5_crb"]) / np.maximum(prob.illum_req_eve, 1e-12),
            ])

        x0 = np.concatenate([warm.p / s, (warm.q * nrm / s) if nd else []])
        res = minimize(f, x0, method="SLSQP", bounds=[(0, 1)] * (K + nd),
                       constraints=[{"type": "ineq", "fun": con_vec}],
                       options=dict(maxiter=maxiter, ftol=1e-7))
        pp, qq = unpack(res.x)
        cand = _finish(prob, W, pp, qq, "slsqp", lm=lm, t0=t0, beam=mode,
                       nit=int(res.nit), status=res.message)
        if (-cand.info["violation"], cand.info["obj_pred"]) > (-best.info["violation"], best.info["obj_pred"]):
            best = cand
    if best is warm:
        best = Solution(warm.F, warm.S, warm.p, warm.q, "slsqp",
                        dict(warm.info, fallback="grid_split", solve_time=time.perf_counter() - t0))
    return best


# ---------------------------------------------------------------------- stubs --
@register("sca_sdr")
def solve_sca_sdr(prob: Problem) -> Solution:
    """STUB. Intended: SDR on F_k = f_k f_k^H and S, with the Eve max over E_j
    handled by S-procedure (ellipsoidal CSI error) or the sampled set, and the
    difference-of-logs secrecy objective handled by SCA (first-order Taylor on the
    concave Eve term). Needs `pip install cvxpy`."""
    raise NotImplementedError("sca_sdr is a stub — see docstring; install cvxpy and implement")


@register("drl")
def solve_drl(prob: Problem) -> Solution:
    """STUB. Intended: DDPG/SAC agent acting on (p, q) (or full F) with state =
    predicted angles/ranges + CRBs, reward = sum secrecy - constraint penalties.
    Hook: call LinearModel(prob, W).objective / .violation as the reward."""
    raise NotImplementedError("drl solver is a stub")
