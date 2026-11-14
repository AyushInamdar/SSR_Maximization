# Results

All numbers: mean ± std (ddof = 1) over **10 seeds (0–9)** of the per-seed mean; default `SimParams` (P_max = −30 dBm,
Nt = Nr = 64, K = 4 users, J = 2 Eves). Frameworks/solvers/tracking: 100 time instants; sweeps (P_max, Nt, AN budget): 50 instants.

## Reproduce (exact commands, from `ugp_otfs_isac_sim/ugp_sim/`)

```
pip install -r requirements.txt
python run_demo.py --seeds 0 1 2 3 4 5 6 7 8 9 --frames 100 --sweep-frames 50 --out results   # ~7.5 min on 12 cores
python scripts/sanity_check.py --pmax -30     # per-user SNR/rate, P_max sweep, crossing diagnostic -> results/sanity_pmax_sweep.csv
python -m pytest tests -q                     # 18 tests
```

## Files
`summary.csv` (mean/std per experiment), `per_seed.csv`, `crb_std_per_frame.csv` (CRB std of delay, Doppler, angle per frame),
`sanity_pmax_sweep.csv`, `changelog_steps/`, `baseline_before/` (old figures and summary), and

| Figure | Content |
|---|---|
| `fig_frameworks_secrecy.png` | sum secrecy vs time, 4 frameworks |
| `fig_uncertainty_region.png` | size of E_j (area; range/Doppler/angle std) vs time instant, fused vs single frame |
| `fig_vs_baselines.png` | OTFS-ISAC vs no-AN, isotropic AN, OFDM-ISAC, feedback, genie |
| `fig_solvers.png` | solver comparison + constraint feasibility |
| `fig_secrecy_vs_pmax.png`, `fig_secrecy_vs_nt.png`, `fig_an_budget.png`, `fig_angle_tracking.png` | sweeps |

Figures with the stylised feedback benchmark or OFDM sensing carry a "DRAFT: placeholder used" stamp (see `draft.py`).

## Key numbers (sum secrecy, bit/s/Hz)

| | mean ± std |
|---|---|
| OTFS-ISAC (grid_split) | 17.08 ± 0.91 |
| OFDM-ISAC (placeholder sensing) | 15.82 ± 0.84 |
| OTFS pilot feedback (placeholder) | 14.10 ± 0.64 |
| genie | 17.08 ± 0.91 |
| OTFS-ISAC, no AN / isotropic AN | 16.30 ± 0.90 / 14.86 ± 0.90 |
| slsqp refinement | 17.14 ± 0.91 |

P_max sweep (OTFS-ISAC): −40 dBm 6.85, −30 17.43, −20 30.06, −10 43.10, 0 56.05, +10 68.57. Nt sweep: 16 → 8.69, 32 → 13.16, 64 → 17.43, 128 → 21.71.
Uncertainty region: 8.0e-3 → 1.2e-3 m² (first 5 → last 10 instants) with fusion, coverage 0.91; coverage 0.22 without fusion.
In `fig_uncertainty_region.png` the fused Doppler/angle std plot *above* the single-frame CRB: the fused region includes the motion
process noise Q (which is what makes it cover the truth); the single-frame CRB alone does not.
See `../CHANGELOG.md` for before/after numbers, the dip explanation and open placeholders.
