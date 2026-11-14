"""Sanity checks for the operating point (CHANGELOG item 5).

    python scripts/sanity_check.py [--pmax 0] [--sweep -30 -20 -10 0 10]

1. per-user SNR / SINR / rate / best-Eve rate at the given P_max (mean over time and seeds);
2. P_max sweep (otfs_isac, grid_split);
3. the angular-crossing diagnostic: minimum user-user angular separation per frame vs the
   array beamwidth 2/Nt, and sum-secrecy in those frames (explains dips such as seed 0, frames 20-24).
"""
import argparse, csv, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")
from otfs_isac_sim.config import SimParams, w_to_dbm
from otfs_isac_sim.simulator import run_episode
from otfs_isac_sim.geometry import Scenario


def job(a):
    pmax, seed, frames, fw = a
    p = replace(SimParams(), P_max_dBm=pmax)
    r = run_episode(p, fw, "grid_split", n_frames=frames, seed=seed)
    return pmax, seed, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=float, default=SimParams().P_max_dBm)
    ap.add_argument("--sweep", type=float, nargs="+", default=[-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    p0 = SimParams()
    print(f"noise power N0 = {w_to_dbm(p0.N0):.1f} dBm over {p0.bandwidth/1e3:.0f} kHz (NF {p0.noise_figure_dB} dB)")
    jobs = [(pm, s, a.frames, "otfs_isac") for pm in sorted(set(a.sweep + [a.pmax])) for s in a.seeds]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(job, jobs))
    rows = []
    print(f"\n{'Pmax dBm':>9} {'SNR dB (mean/user)':>20} {'rate b/s/Hz/user':>17} {'best-Eve rate':>14} {'sum secrecy':>12} {'sum rate':>9}")
    for pm in sorted(set(a.sweep + [a.pmax])):
        rs = [r for (q, s, r) in out if q == pm]
        snr = np.mean([10 * np.log10(np.mean(r["snr_users"], 0)) for r in rs], 0)          # per user
        ru = np.mean([r["R_users"].mean() for r in rs]); re = np.mean([r["R_eve_max"].mean() for r in rs])
        ss = [r["sum_sec"].mean() for r in rs]; sr = np.mean([r["sum_rate"].mean() for r in rs])
        mark = "  <- default" if pm == a.pmax else ""
        print(f"{pm:9.1f} {np.mean(snr):10.1f} [{', '.join(f'{x:5.1f}' for x in snr)}] {ru:9.2f} {re:14.3f} {np.mean(ss):8.2f}+/-{np.std(ss, ddof=1):.2f} {sr:9.2f}{mark}")
        rows.append(dict(P_max_dBm=pm, snr_db_mean=float(np.mean(snr)), snr_db_users=";".join(f"{x:.1f}" for x in snr),
                         rate_per_user=float(ru), best_eve_rate=float(re), sum_sec_mean=float(np.mean(ss)),
                         sum_sec_std=float(np.std(ss, ddof=1)), sum_rate=float(sr)))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/sanity_pmax_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\nper-user detail at the default P_max (seed 0, mean over time):")
    r = [r for (q, s, r) in out if q == a.pmax and s == a.seeds[0]][0]
    for k in range(r["R_users"].shape[1]):
        print(f"  user {k+1}: SNR {10*np.log10(r['snr_users'][:, k].mean()):5.1f} dB   rate {r['R_users'][:, k].mean():5.2f}   "
              f"secrecy {r['R_sec'][:, k].mean():5.2f}   best-Eve {r['R_eve_max'][:, k].mean():5.3f}")

    print(f"\nangular-crossing diagnostic (beamwidth 2/Nt = {2/p0.Nt:.4f} in sin(theta)):")
    for s in a.seeds:
        rt = np.random.default_rng(s); sc = Scenario.from_params(p0, rt)
        sep = []
        for t in range(1, a.frames):
            sc.step(rt)
            st = np.sort(np.sin(sc.theta[:sc.K])); sep.append(np.min(np.diff(st)))
        sep = np.array(sep); bad = np.where(sep < 2 / p0.Nt)[0] + 1
        rr = [r for (q, ss, r) in out if q == a.pmax and ss == s][0]
        dip = rr["sum_sec"][bad - 1] if bad.size else np.array([])
        print(f"  seed {s}: users within one beamwidth at frames {bad.min() if bad.size else '-'}..{bad.max() if bad.size else '-'}; "
              f"min separation {sep.min():.4f}; sum-secrecy there {dip.min() if dip.size else float('nan'):.1f} "
              f"vs median {np.median(rr['sum_sec']):.1f}")
