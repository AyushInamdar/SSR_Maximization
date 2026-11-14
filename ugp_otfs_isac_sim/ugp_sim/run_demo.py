"""
Experiments. Every number is mean +/- std (ddof=1) over seeds of the per-seed mean.

  python run_demo.py                      # 10 seeds (0-9), 100 frames, sweeps 50 frames
  python run_demo.py --seeds 0 1 2 --frames 40 --sweep-frames 20 --out results_quick   # smoke test

Writes to ./results (or --out):
  summary.csv                 one row per (experiment, framework, solver, parameter value);
                              <metric>_mean / <metric>_std over seeds
  per_seed.csv                the underlying per-seed numbers
  crb_std_per_frame.csv       CRB std of delay / Doppler / angle per frame (otfs_isac, mean over seeds/Eves)
  fig_frameworks_secrecy.png  sum secrecy vs time (mean +/- std over seeds)
  fig_uncertainty_region.png  size of E_j vs time instant, with / without fusion across frames
  fig_vs_baselines.png        OTFS-ISAC vs baselines (frameworks and solver baselines)
  fig_solvers.png             solvers + constraint feasibility
  fig_secrecy_vs_pmax.png     sum secrecy vs P_max
  fig_secrecy_vs_nt.png       sum secrecy vs number of transmit antennas
  fig_an_budget.png           sum secrecy vs AN budget
  fig_angle_tracking.png      predicted-angle RMSE vs time

A figure carries a "DRAFT" stamp only if a run behind it touched a placeholder
(draft.py registry: currently the stylised feedback benchmark and OFDM sensing).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_isac_sim.config import SimParams, C_LIGHT
from otfs_isac_sim.simulator import run_episode, summarise
from otfs_isac_sim.draft import DraftWarning, DRAFT_ITEMS

warnings.simplefilter("ignore", DraftWarning)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})
LABELS = {"otfs_isac": "OTFS-ISAC (proposed)", "ofdm_isac": "OFDM-ISAC",
          "otfs_feedback": "OTFS, pilot feedback", "otfs_perfect": "OTFS, perfect state (genie)"}
KEEP = ["sum_sec", "sum_rate", "R_users", "R_eve_max", "ang_err_u", "ang_err_e", "region_area_e",
        "pred_std_eta_e", "post_std_eta_e", "meas_std_eta_e", "cover_e", "an_power", "solve_time",
        "feas_C1", "feas_C2", "feas_C3", "feas_C4", "feas_C5", "an_leak_users", "snr_users", "meta"]


def job(spec):
    key, p, fw, solver, frames, seed = spec
    r = run_episode(p, fw, solver, n_frames=frames, seed=seed)
    out = {k: r[k] for k in KEEP}
    out["summary"] = summarise(r)
    return key, out


def stamp(fig, runs):
    tags = sorted({t for r in runs for t in r["meta"].get("drafts", [])})
    if tags:
        fig.text(0.995, 0.005, "DRAFT: placeholder used: " + ", ".join(tags), ha="right", va="bottom",
                 fontsize=8, color=INK2, style="italic")


def ms(vals):
    v = np.asarray(vals, float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if v.size > 1 else 0.0)


def band(ax, t, arrs, color, label, log=False):
    A = np.asarray(arrs)
    m, s = A.mean(0), (A.std(0, ddof=1) if A.shape[0] > 1 else 0 * A.mean(0))
    ax.plot(t, m, color=color, label=label)
    ax.fill_between(t, np.maximum(m - s, 1e-300) if log else m - s, m + s, color=color, alpha=0.15, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--sweep-frames", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = SimParams()
    seeds, F, SF = a.seeds, a.frames, a.sweep_frames
    t0 = time.time()

    # ------------------------------------------------------------------ job list
    specs = []

    def add(exp, tag, pp, fw, solver, frames):
        for s in seeds:
            specs.append(((exp, tag, fw, solver), pp, fw, solver, frames, s))

    fws = ["otfs_isac", "ofdm_isac", "otfs_feedback", "otfs_perfect"]
    for fw in fws:
        add("frameworks", "-", p, fw, "grid_split", F)
    add("tracking", "accumulate", p, "otfs_isac", "grid_split", F)          # same as frameworks/otfs_isac
    add("tracking", "single_frame", replace(p, track_accumulate=False), "otfs_isac", "grid_split", F)
    solvers = ["mrt_no_an", "mrt_an", "zf_an", "isotropic_an", "grid_split", "slsqp"]
    for sv in solvers:
        add("solvers", "-", p, "otfs_isac", sv, F)
    fracs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    for f_ in fracs:
        for sv in ("grid_split", "slsqp"):
            add("an_budget", f"{f_}", replace(p, P_an_max_frac=f_), "otfs_isac", sv, SF)
    pmaxs = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0]
    for pm in pmaxs:
        for fw in fws:
            add("pmax", f"{pm}", replace(p, P_max_dBm=pm), fw, "grid_split", SF)
        add("pmax", f"{pm}", replace(p, P_max_dBm=pm), "otfs_isac", "mrt_no_an", SF)
    nts = [16, 32, 64, 128]
    for nt in nts:
        for fw in fws:
            add("nt", f"{nt}", replace(p, Nt=nt), fw, "grid_split", SF)
        add("nt", f"{nt}", replace(p, Nt=nt), "otfs_isac", "mrt_no_an", SF)

    print(f"{len(specs)} runs on {os.cpu_count()} cores ...", flush=True)
    R: dict = {}
    done = 0
    with ProcessPoolExecutor() as ex:
        for key, out in ex.map(job, specs, chunksize=1):
            R.setdefault(key, []).append(out)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(specs)}  {time.time() - t0:.0f}s", flush=True)

    def runs(exp, tag="-", fw="otfs_isac", solver="grid_split"):
        return R[(exp, tag, fw, solver)]

    def stat(rs, k="mean_sum_sec"):
        return ms([r["summary"][k] for r in rs])

    # ------------------------------------------------------------------- CSVs
    metrics = ["mean_sum_sec", "mean_sum_rate", "mean_user_rate", "mean_best_eve_rate", "mean_user_snr_db",
               "rmse_angle_users_rad", "rmse_angle_eves_rad", "mean_an_power_mW", "an_leak_over_N0",
               "region_area_first5_m2", "region_area_last10_m2", "coverage_E_j", "mean_solve_ms",
               "frac_feasible_C1", "frac_feasible_C2", "frac_feasible_C3", "frac_feasible_C4", "frac_feasible_C5"]
    with open(f"{a.out}/summary.csv", "w", newline="") as fh, open(f"{a.out}/per_seed.csv", "w", newline="") as fp:
        cols = ["experiment", "param", "framework", "solver", "n_seeds", "frames"]
        w = csv.writer(fh); w.writerow(cols + [f"{m}_{x}" for m in metrics for x in ("mean", "std")] + ["placeholders"])
        w2 = csv.writer(fp); w2.writerow(["experiment", "param", "framework", "solver", "seed"] + metrics)
        for (exp, tag, fw, sv), rs in R.items():
            fr = F if exp in ("frameworks", "tracking", "solvers") else SF
            row = [exp, tag, fw, sv, len(rs), fr]
            for m in metrics:
                mu, sd = stat(rs, m)
                row += [f"{mu:.6g}", f"{sd:.6g}"]
            w.writerow(row + [rs[0]["summary"]["drafts"]])
            for r in rs:
                w2.writerow([exp, tag, fw, sv, r["summary"]["seed"]] + [f"{r['summary'][m]:.6g}" for m in metrics])

    tr = runs("tracking", "accumulate")
    pm_ = np.mean([r["post_std_eta_e"] for r in tr], 0).mean(1)       # (T, 3) mean over seeds, Eves
    pr_ = np.mean([r["pred_std_eta_e"] for r in tr], 0).mean(1)
    me_ = np.mean([r["meas_std_eta_e"] for r in tr], 0).mean(1)
    with open(f"{a.out}/crb_std_per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "posterior_delay_std_s", "posterior_range_std_m", "posterior_doppler_std_Hz", "posterior_angle_std_rad",
                    "predicted_delay_std_s", "predicted_doppler_std_Hz", "predicted_angle_std_rad",
                    "single_frame_delay_std_s", "single_frame_doppler_std_Hz", "single_frame_angle_std_rad"])
        for i in range(pm_.shape[0]):
            w.writerow([i + 1, pm_[i, 0], pm_[i, 0] * C_LIGHT / 2, pm_[i, 1], pm_[i, 2], pr_[i, 0], pr_[i, 1], pr_[i, 2],
                        me_[i, 0], me_[i, 1], me_[i, 2]])

    t = np.arange(1, F)

    # -------------------------------------------------- fig: frameworks vs time
    fig, ax = plt.subplots(figsize=(7.5, 4))
    allr = []
    for c, fw in zip(COLORS, fws):
        rs = runs("frameworks", "-", fw); allr += rs
        band(ax, t, [r["sum_sec"] for r in rs], c, LABELS[fw])
    ax.set_xlabel("time instant (dT = 20 ms)"); ax.set_ylabel("sum secrecy rate (bit/s/Hz)")
    ax.set_title(f"Sum secrecy rate over time (mean ± std, {len(seeds)} seeds, P_max = {p.P_max_dBm:g} dBm)", loc="left", fontsize=10)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    stamp(fig, allr); fig.tight_layout(); fig.savefig(f"{a.out}/fig_frameworks_secrecy.png", dpi=150); plt.close(fig)

    # ------------------------------------------ fig: uncertainty region vs time
    sf = runs("tracking", "single_frame")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for c, rs, lab in ((COLORS[0], tr, "fused over frames (information filter)"),
                       (COLORS[1], sf, "single-frame CRB only")):
        band(axs[0], t, [np.mean(r["region_area_e"], 1) for r in rs], c, lab, log=True)
        band(axs[1], t, [np.mean(r["pred_std_eta_e"][:, :, 0], 1) * C_LIGHT / 2 for r in rs], c, lab, log=True)
        band(axs[2], t, [np.mean(r["pred_std_eta_e"][:, :, 1], 1) for r in rs], c, lab, log=True)
        band(axs[3], t, [np.rad2deg(np.mean(r["pred_std_eta_e"][:, :, 2], 1)) for r in rs], c, lab, log=True)
    for ax, ttl, yl in zip(axs, ["size of E_j (cross-range x range)", "range std", "Doppler std", "angle std"],
                           ["area of 95 % region (m²)", "m (c·σγ/2)", "Hz", "deg"]):
        ax.set_yscale("log"); ax.set_title(ttl, loc="left", fontsize=10); ax.set_ylabel(yl); ax.set_xlabel("time instant")
    axs[0].legend(fontsize=8)
    fig.suptitle("Uncertainty region of the eavesdroppers used for the design (CRB-based, incl. process noise Q)", x=0.01, ha="left", fontsize=10)
    stamp(fig, tr + sf); fig.tight_layout(); fig.savefig(f"{a.out}/fig_uncertainty_region.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------ fig: angle tracking
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    ar = []
    for c, fw in zip(COLORS, fws[:3]):
        rs = runs("frameworks", "-", fw); ar += rs
        for ax, k in zip(axs, ("ang_err_u", "ang_err_e")):
            band(ax, t, [np.sqrt(np.mean(r[k] ** 2, 1)) for r in rs], c, LABELS[fw], log=True)
    for ax, ttl in zip(axs, ("Users: predicted-angle RMSE", "Eves: predicted-angle RMSE")):
        ax.set_yscale("log"); ax.set_title(ttl, loc="left", fontsize=10); ax.set_xlabel("time instant")
    axs[0].set_ylabel("RMSE over users/Eves (rad)"); axs[0].legend(fontsize=8)
    stamp(fig, ar); fig.tight_layout(); fig.savefig(f"{a.out}/fig_angle_tracking.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------ fig: vs baselines
    items = [("OTFS-ISAC\n(proposed)", runs("frameworks", "-", "otfs_isac")),
             ("OTFS-ISAC\nno AN", runs("solvers", "-", "otfs_isac", "mrt_no_an")),
             ("OTFS-ISAC\nisotropic AN", runs("solvers", "-", "otfs_isac", "isotropic_an")),
             ("OFDM-ISAC", runs("frameworks", "-", "ofdm_isac")),
             ("OTFS, pilot\nfeedback", runs("frameworks", "-", "otfs_feedback")),
             ("genie\n(perfect state)", runs("frameworks", "-", "otfs_perfect"))]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (lab, rs) in enumerate(items):
        mu, sd = stat(rs)
        ax.bar(i, mu, 0.6, yerr=sd, capsize=4, color=COLORS[0] if i == 0 else COLORS[i % len(COLORS)], error_kw=dict(ecolor=INK2))
        ax.text(i, mu + sd, f"{mu:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(items)), [x[0] for x in items], fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean sum secrecy (bit/s/Hz)")
    ax.set_title(f"OTFS-ISAC vs baselines (mean ± std over {len(seeds)} seeds)", loc="left", fontsize=10)
    stamp(fig, [r for _, rs in items for r in rs]); fig.tight_layout(); fig.savefig(f"{a.out}/fig_vs_baselines.png", dpi=150); plt.close(fig)

    # ----------------------------------------------------------- fig: solvers
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ms_ = [stat(runs("solvers", "-", "otfs_isac", s)) for s in solvers]
    axs[0].barh(range(len(solvers)), [m[0] for m in ms_][::-1] if False else [m[0] for m in ms_], xerr=[m[1] for m in ms_],
                color=COLORS[0], height=0.6, capsize=3, error_kw=dict(ecolor=INK2))
    axs[0].set_yticks(range(len(solvers)), solvers); axs[0].invert_yaxis()
    for i, m in enumerate(ms_):
        axs[0].text(m[0] + m[1], i, f" {m[0]:.2f}", va="center", fontsize=8)
    axs[0].set_xlabel("mean sum secrecy (bit/s/Hz)"); axs[0].grid(axis="y", visible=False)
    axs[0].set_title("OTFS-ISAC: solvers", loc="left", fontsize=10)
    bw = 0.13; x = np.arange(len(solvers))
    for k, cn in enumerate(["C1 total P", "C2 AN budget", "C3 QoS", "C4 sensing", "C5 CRB"]):
        v = [stat(runs("solvers", "-", "otfs_isac", s), f"frac_feasible_C{k+1}")[0] for s in solvers]
        axs[1].bar(x + (k - 2) * bw, v, bw * 0.85, color=COLORS[k], label=cn)
    axs[1].set_xticks(x, solvers, rotation=25, ha="right", fontsize=8); axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("fraction of frames feasible"); axs[1].grid(axis="x", visible=False)
    axs[1].set_title("Constraint satisfaction (predicted states)", loc="left", fontsize=10)
    axs[1].legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    stamp(fig, [r for s in solvers for r in runs("solvers", "-", "otfs_isac", s)]); fig.tight_layout()
    fig.savefig(f"{a.out}/fig_solvers.png", dpi=150); plt.close(fig)

    # ----------------------------------------------------------- fig: AN budget
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); ar = []
    for c, sv in zip(COLORS, ("grid_split", "slsqp")):
        mm = [stat(runs("an_budget", f"{f_}", "otfs_isac", sv)) for f_ in fracs]
        ar += [r for f_ in fracs for r in runs("an_budget", f"{f_}", "otfs_isac", sv)]
        ax.errorbar(np.array(fracs) * 100, [m[0] for m in mm], [m[1] for m in mm], color=c, marker="o", ms=5, capsize=3, label=sv)
    ax.set_xlabel("AN power budget P_AN,max (% of P_max)"); ax.set_ylabel("mean sum secrecy (bit/s/Hz)")
    ax.set_title("Effect of the AN power budget (OTFS-ISAC)", loc="left", fontsize=10); ax.legend(fontsize=9)
    stamp(fig, ar); fig.tight_layout(); fig.savefig(f"{a.out}/fig_an_budget.png", dpi=150); plt.close(fig)

    # ----------------------------------------------------- figs: P_max and Nt
    def sweep_fig(exp, xs, xlabel, fname, title, logx=False):
        fig, ax = plt.subplots(figsize=(7, 4)); ar = []
        series = [(fw, "grid_split", LABELS[fw]) for fw in fws] + [("otfs_isac", "mrt_no_an", "OTFS-ISAC, no AN")]
        for i, (fw, sv, lab) in enumerate(series):
            mm = [stat(runs(exp, f"{x}", fw, sv)) for x in xs]
            ar += [r for x in xs for r in runs(exp, f"{x}", fw, sv)]
            ax.errorbar(xs, [m[0] for m in mm], [m[1] for m in mm], color=COLORS[i], marker="o", ms=5, capsize=3,
                        label=lab, ls="--" if sv == "mrt_no_an" else "-")
        ax.set_xlabel(xlabel); ax.set_ylabel("mean sum secrecy (bit/s/Hz)")
        if logx:
            ax.set_xscale("log", base=2); ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_title(title, loc="left", fontsize=10); ax.legend(fontsize=8)
        stamp(fig, ar); fig.tight_layout(); fig.savefig(f"{a.out}/{fname}", dpi=150); plt.close(fig)

    sweep_fig("pmax", pmaxs, "total transmit power P_max (dBm)", "fig_secrecy_vs_pmax.png",
              f"Sum secrecy rate vs P_max (mean ± std, {len(seeds)} seeds)")
    sweep_fig("nt", nts, "number of transmit antennas Nt", "fig_secrecy_vs_nt.png",
              f"Sum secrecy rate vs Nt (P_max = {p.P_max_dBm:g} dBm, {len(seeds)} seeds)", logx=True)

    print(f"\nwrote {a.out}/summary.csv, per_seed.csv, crb_std_per_frame.csv and 8 figures in {time.time() - t0:.0f}s")
    used = sorted({t for rs in R.values() for r in rs for t in r["meta"].get("drafts", [])})
    print("placeholders touched:", ", ".join(f"{t} ({DRAFT_ITEMS.get(t, "")[:60]})" for t in used) or "none")


if __name__ == "__main__":
    main()
