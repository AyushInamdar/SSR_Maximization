"""Snapshot of headline numbers used by CHANGELOG.md (before/after each model change).

    python scripts/step_metrics.py <label>

Fixed protocol: default SimParams (or --pmax dBm), grid_split, seeds 0-2, 60 frames,
all four frameworks. Writes results/changelog_steps/<label>.csv (mean +/- std over seeds
of the per-seed mean sum-secrecy rate)."""
import argparse, csv, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")
from otfs_isac_sim.config import SimParams
from otfs_isac_sim.simulator import run_episode

FWS = ["otfs_isac", "ofdm_isac", "otfs_feedback", "otfs_perfect"]


def job(a):
    fw, seed, frames, pmax = a
    p = SimParams() if pmax is None else replace(SimParams(), P_max_dBm=pmax)
    r = run_episode(p, fw, "grid_split", n_frames=frames, seed=seed)
    return fw, seed, float(np.mean(r["sum_sec"])), float(np.mean(r["sum_rate"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    jobs = [(fw, s, a.frames, a.pmax) for fw in FWS for s in a.seeds]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(job, jobs))
    os.makedirs("results/changelog_steps", exist_ok=True)
    with open(f"results/changelog_steps/{a.label}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "framework", "n_seeds", "frames", "sum_sec_mean", "sum_sec_std", "sum_rate_mean"])
        for fw in FWS:
            v = [o[2] for o in out if o[0] == fw]
            sr = [o[3] for o in out if o[0] == fw]
            w.writerow([a.label, fw, len(v), a.frames, f"{np.mean(v):.4f}", f"{np.std(v, ddof=1):.4f}", f"{np.mean(sr):.4f}"])
            print(f"{a.label:12s} {fw:14s} sum_sec {np.mean(v):7.3f} +/- {np.std(v, ddof=1):.3f}")
