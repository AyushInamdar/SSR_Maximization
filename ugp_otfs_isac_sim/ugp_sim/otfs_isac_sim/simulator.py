"""
Time-loop simulator. One time instant (coherence interval) t:

  1. vehicles move (true state at t)
  2. design: build Problem from the states PREDICTED at t-1, run the solver
  3. evaluate rates / secrecy on the TRUE state at t
  4. sense with the transmission just made (R_x = F F^H + S) -> estimates at t
     -> predictions for t+1
At t = 0 an isotropic probe (Draft step 1) gives the initial estimates of all
targets. Trajectories use their own RNG stream, so every framework/solver sees the
same vehicle motion for a given seed.
"""
from __future__ import annotations

import time
import numpy as np

from .config import SimParams, C_LIGHT
from .geometry import Scenario, path_gain, steer1
from .sensing import sense_target, feedback_estimate, Estimate
from .crb import uncertainty_samples, min_illumination_for_crb, illumination
from .an import build_an_basis
from .solvers import Problem, get_solver
from .frameworks import get_frameworks
from .metrics import evaluate_true


def _truth_estimate(scen: Scenario, i: int) -> Estimate:
    return Estimate(pos=scen.pos[i].copy(), speed=float(scen.vel[i]), theta=float(scen.theta[i]),
                    dist=float(scen.dist[i]), C=np.zeros((3, 3)))


def build_problem(preds: list[Estimate], K: int, fw, p: SimParams) -> Problem:
    pu = preds[:K]
    pe = preds[K:]
    th_u = np.array([e.theta for e in pu])
    d_u = np.array([e.dist for e in pu])
    nu_u = np.array([-e.speed * np.sin(e.theta) * p.fc / C_LIGHT for e in pu])

    samp_th, samp_d, samp_nu, owner = [], [], [], []
    for j, e in enumerate(pe):
        th, d = uncertainty_samples(e.theta, e.dist, e.C, p)
        samp_th.append(th); samp_d.append(d); owner.append(np.full(th.size, j))
        samp_nu.append(-e.speed * np.sin(th) * p.fc / C_LIGHT)
    cat = (lambda xs: np.concatenate(xs)) if pe else (lambda xs: np.zeros(0))
    eve_th, eve_d = cat(samp_th), cat(samp_d)

    basis = build_an_basis(th_u, samp_th, p.Nt)
    return Problem(
        p=p, waveform=fw.waveform, prefactor=fw.prefactor(p),
        th_u=th_u, h2_u=path_gain(d_u, p), nu_u=nu_u,
        eve_th=eve_th, eve_h2=path_gain(eve_d, p), eve_nu=cat(samp_nu),
        eve_owner=cat(owner).astype(int),
        eve_center_th=np.array([e.theta for e in pe]),
        sens_th=np.array([e.theta for e in preds]),
        illum_req_eve=np.array([min_illumination_for_crb(e.theta, e.dist, p) for e in pe]),
        basis=basis,
    )


def run_episode(p: SimParams, framework: str = "otfs_isac", solver: str = "grid_split",
                n_frames: int | None = None, seed: int | None = None, verbose: bool = False) -> dict:
    seed = p.seed if seed is None else seed
    n_frames = p.n_frames if n_frames is None else n_frames
    rng_traj = np.random.default_rng(seed)
    rng_meas = np.random.default_rng(seed + 10_000)
    fw = get_frameworks(p)[framework]
    solve = get_solver(solver)
    scen = Scenario.from_params(p, rng_traj)
    K, J, P = scen.K, scen.J, scen.P

    # ---- t = 0: isotropic sensing probe (R_x = P_probe/Nt * I -> a^H R a = P_probe)
    est: list[Estimate] = []
    for i in range(P):
        if (i < K and fw.user_sensing == "perfect") or (i >= K and fw.eve_sensing == "perfect"):
            est.append(_truth_estimate(scen, i))
        else:
            est.append(sense_target(scen.theta[i], scen.dist[i], scen.omega[i], p.P_probe,
                                    None, None, p, rng_meas))
    preds = [e.predict(p) for e in est]

    keys = ["sum_sec", "sum_rate", "R_sec", "R_users", "R_eve_max", "ang_err_u", "ang_err_e",
            "crb_theta_std_e", "crb_range_std_e", "an_power", "tx_power", "violation",
            "feas_C1", "feas_C2", "feas_C3", "feas_C4", "feas_C5", "solve_time", "obj_pred",
            "an_leak_users"]
    H = {k: [] for k in keys}

    for t in range(1, n_frames):
        scen.step(rng_traj)
        # genie frameworks get the true current state as "prediction"
        for i in range(P):
            if (i < K and fw.user_sensing == "perfect") or (i >= K and fw.eve_sensing == "perfect"):
                preds[i] = _truth_estimate(scen, i)

        prob = build_problem(preds, K, fw, p)
        t0 = time.perf_counter()
        sol = solve(prob)
        H["solve_time"].append(time.perf_counter() - t0)
        m = evaluate_true(sol.F, sol.S, scen, fw.waveform, prob.prefactor, p)

        H["sum_sec"].append(m["sum_sec"]); H["sum_rate"].append(m["sum_rate"])
        H["R_sec"].append(m["R_sec"]); H["R_users"].append(m["R_users"])
        H["R_eve_max"].append(m["R_eve"].max(0) if J else np.zeros(K))
        H["an_leak_users"].append(m["an_leak_users"])
        th_true = scen.theta
        H["ang_err_u"].append(np.array([preds[i].theta for i in range(K)]) - th_true[:K])
        H["ang_err_e"].append(np.array([preds[i].theta for i in range(K, P)]) - th_true[K:])
        H["crb_theta_std_e"].append(np.array([np.sqrt(preds[i].C[2, 2]) for i in range(K, P)]))
        H["crb_range_std_e"].append(np.array([np.sqrt(preds[i].C[0, 0]) * C_LIGHT / 2 for i in range(K, P)]))
        H["an_power"].append(sol.info["an_power"])
        H["tx_power"].append(float(np.sum(np.abs(sol.F) ** 2) + sol.info["an_power"]))
        H["violation"].append(sol.info["violation"]); H["obj_pred"].append(sol.info["obj_pred"])
        for c in range(1, 6):
            key = [k for k in sol.info["feasible"] if k.startswith(f"C{c}")][0]
            H[f"feas_C{c}"].append(sol.info["feasible"][key])

        # ---- sensing with this transmission -> estimates at t -> predictions t+1
        R_x = sol.F @ sol.F.conj().T + sol.S
        new_est = []
        for i in range(P):
            is_user = i < K
            mode = fw.user_sensing if is_user else fw.eve_sensing
            if mode == "perfect":
                new_est.append(_truth_estimate(scen, i)); continue
            if mode == "feedback":
                # one-way pilot SNR through the current beam toward the vehicle
                pilot_snr = scen.h2[i] * illumination(scen.theta[i], R_x) / p.N0
                new_est.append(feedback_estimate(scen.theta[i], scen.dist[i], pilot_snr, p, rng_meas))
                continue
            if mode == "probe":
                illum = p.P_probe
            else:
                illum = illumination(scen.theta[i], R_x)
            Gm = p.Gm_eff
            new_est.append(sense_target(scen.theta[i], scen.dist[i], scen.omega[i], illum,
                                        preds[i].theta, est[i].speed, p, rng_meas, Gm=Gm))
        est = new_est
        preds = [e.predict(p) for e in est]
        if verbose and t % 20 == 0:
            print(f"[{framework}/{solver}] t={t:3d}  sum_sec={m['sum_sec']:.2f}  "
                  f"viol={sol.info['violation']:.2e}")

    out = {k: np.array(v) for k, v in H.items()}
    out["meta"] = dict(framework=framework, solver=solver, seed=seed, n_frames=n_frames,
                       K=K, J=J, prefactor=fw.prefactor(p))
    return out


def summarise(res: dict) -> dict:
    s = {
        "framework": res["meta"]["framework"], "solver": res["meta"]["solver"],
        "mean_sum_sec": float(np.mean(res["sum_sec"])),
        "mean_sum_rate": float(np.mean(res["sum_rate"])),
        "rmse_angle_users_rad": float(np.sqrt(np.mean(res["ang_err_u"] ** 2))),
        "rmse_angle_eves_rad": float(np.sqrt(np.mean(res["ang_err_e"] ** 2))) if res["ang_err_e"].size else np.nan,
        "mean_an_power_mW": float(np.mean(res["an_power"]) * 1e3),
        "mean_solve_ms": float(np.mean(res["solve_time"]) * 1e3),
    }
    for c in range(1, 6):
        s[f"frac_feasible_C{c}"] = float(np.mean(res[f"feas_C{c}"]))
    return s
