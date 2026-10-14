"""
Demo / starting experiments.

  python run_demo.py                 # default: 100 time instants, 1 seed
  python run_demo.py --frames 200 --seeds 0 1 2

Produces in ./results/:
  summary.csv                    mean metrics per (framework, solver) and AN sweep
  fig_frameworks_secrecy.png     sum secrecy vs time: OTFS-ISAC / OFDM-ISAC / feedback / genie
  fig_solvers.png                mean sum secrecy + constraint feasibility per solver
  fig_an_budget.png              sum secrecy vs AN power budget (new constraint)
  fig_angle_tracking.png         predicted-angle error vs time (users, Eves)

Every figure is stamped "DRAFT MODELS" because the AN formulation and the CRB are
placeholders (see draft.py report printed at the end).
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_isac_sim.config import SimParams
from otfs_isac_sim.simulator import run_episode, summarise
from otfs_isac_sim.draft import DraftWarning, print_draft_report

warnings.simplefilter("ignore", DraftWarning)   # collected and printed once at the end

# categorical palette, fixed order (dataviz reference palette, light surface)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})


def stamp(fig):
    fig.text(0.995, 0.005, "DRAFT MODELS: crude AN + crude CRB", ha="right", va="bottom",
             fontsize=8, color=INK2, style="italic")


def avg_runs(p, fw, solver, frames, seeds):
    runs = [run_episode(p, fw, solver, n_frames=frames, seed=s) for s in seeds]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = SimParams()
    rows = []

    # ---- A. frameworks (solver fixed) ---------------------------------------
    fws = ["otfs_isac", "ofdm_isac", "otfs_feedback", "otfs_perfect"]
    labels = {"otfs_isac": "OTFS-ISAC (proposed)", "ofdm_isac": "OFDM-ISAC",
              "otfs_feedback": "OTFS, pilot feedback", "otfs_perfect": "OTFS, perfect state (genie)"}
    fw_runs = {}
    for fw in fws:
        fw_runs[fw] = avg_runs(p, fw, "grid_split", a.frames, a.seeds)
        for r in fw_runs[fw]:
            rows.append(dict(experiment="frameworks", **summarise(r)))
        print("frameworks:", fw, "done")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    t = np.arange(1, a.frames)
    for c, fw in zip(COLORS, fws):
        y = np.mean([r["sum_sec"] for r in fw_runs[fw]], axis=0)
        ax.plot(t, y, color=c, label=labels[fw])
    ax.set_xlabel("time instant (ΔT = 20 ms)")
    ax.set_ylabel("sum secrecy rate (bit/s/Hz)")
    ax.set_title("Sum secrecy rate over time — solver: grid_split", loc="left", fontsize=11)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    stamp(fig); fig.tight_layout(); fig.savefig(f"{a.out}/fig_frameworks_secrecy.png", dpi=150)
    plt.close(fig)

    # ---- D. angle tracking --------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for c, fw in zip(COLORS, fws[:3]):
        eu = np.sqrt(np.mean(np.concatenate([r["ang_err_u"] ** 2 for r in fw_runs[fw]], 1), axis=1))
        ee = np.sqrt(np.mean(np.concatenate([r["ang_err_e"] ** 2 for r in fw_runs[fw]], 1), axis=1))
        axs[0].semilogy(t, np.maximum(eu, 1e-6), color=c, label=labels[fw])
        axs[1].semilogy(t, np.maximum(ee, 1e-6), color=c, label=labels[fw])
    axs[0].set_title("Users: predicted-angle RMSE", loc="left", fontsize=11)
    axs[1].set_title("Eves: predicted-angle RMSE", loc="left", fontsize=11)
    for ax in axs:
        ax.set_xlabel("time instant")
    axs[0].set_ylabel("RMSE (rad)")
    axs[0].legend(fontsize=8, loc="center right")
    stamp(fig); fig.tight_layout(); fig.savefig(f"{a.out}/fig_angle_tracking.png", dpi=150)
    plt.close(fig)

    # ---- B. solvers (framework fixed) ---------------------------------------
    solvers = ["mrt_no_an", "mrt_crude_an", "zf_crude_an", "isotropic_an", "grid_split", "slsqp"]
    sres = {}
    for s in solvers:
        runs = avg_runs(p, "otfs_isac", s, a.frames, a.seeds)
        sres[s] = [summarise(r) for r in runs]
        rows += [dict(experiment="solvers", **x) for x in sres[s]]
        print("solvers:", s, "done")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    m = [np.mean([x["mean_sum_sec"] for x in sres[s]]) for s in solvers]
    axs[0].barh(solvers[::-1], m[::-1], color=COLORS[0], height=0.6)
    for i, v in enumerate(m[::-1]):
        axs[0].text(v, i, f" {v:.1f}", va="center", fontsize=9, color=INK)
    axs[0].set_xlabel("mean sum secrecy (bit/s/Hz)")
    axs[0].set_title("OTFS-ISAC: solver comparison", loc="left", fontsize=11)
    axs[0].grid(axis="y", visible=False)
    cons = ["C1", "C2", "C3", "C4", "C5"]
    cnames = ["C1 total P", "C2 AN budget", "C3 QoS", "C4 sensing", "C5 CRB"]
    w = 0.13
    x = np.arange(len(solvers))
    for k, (c, cn) in enumerate(zip(cons, cnames)):
        v = [np.mean([r[f"frac_feasible_{c}"] for r in sres[s]]) for s in solvers]
        axs[1].bar(x + (k - 2) * w, v, w * 0.85, color=COLORS[k], label=cn)
    axs[1].set_xticks(x, solvers, rotation=25, ha="right", fontsize=8)
    axs[1].set_ylabel("fraction of frames feasible")
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Constraint satisfaction (predicted states)", loc="left", fontsize=11)
    axs[1].legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    axs[1].grid(axis="x", visible=False)
    stamp(fig); fig.tight_layout(); fig.savefig(f"{a.out}/fig_solvers.png", dpi=150)
    plt.close(fig)

    # ---- C. AN budget sweep (new constraint) --------------------------------
    fracs = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c, s in zip(COLORS, ["mrt_crude_an", "grid_split", "slsqp"]):
        ys = []
        for f in fracs:
            pf = replace(p, P_an_max_frac=f)
            runs = avg_runs(pf, "otfs_isac", s, max(a.frames // 2, 20), a.seeds)
            sm = [summarise(r) for r in runs]
            rows += [dict(experiment=f"an_budget_{f}", **x) for x in sm]
            ys.append(np.mean([x["mean_sum_sec"] for x in sm]))
        ax.plot(np.array(fracs) * 100, ys, color=c, marker="o", markersize=5, label=s)
        print("AN sweep:", s, "done")
    ax.set_xlabel("AN power budget P_AN,max (% of P_max)")
    ax.set_ylabel("mean sum secrecy (bit/s/Hz)")
    ax.set_title("Effect of the AN power budget (OTFS-ISAC)", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    stamp(fig); fig.tight_layout(); fig.savefig(f"{a.out}/fig_an_budget.png", dpi=150)
    plt.close(fig)

    # ---- summary ------------------------------------------------------------
    keys = list(rows[0].keys())
    with open(f"{a.out}/summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {a.out}/summary.csv and 4 figures\n")
    print_draft_report()


if __name__ == "__main__":
    main()
