# CHANGELOG — model changes and before/after numbers

Branch `feature/crb-tracking-an-experiments`. Every model change is listed here; nothing is changed silently.
Headline numbers use the fixed protocol of `scripts/step_metrics.py` (default parameters, `grid_split`, seeds 0–2,
60 frames, mean ± std over seeds of the per-seed mean sum-secrecy rate, bit/s/Hz) unless stated.
Snapshots: `results/changelog_steps/*.csv`. Old results (1 seed, 100 frames): `results/baseline_before/`.
New results (10 seeds): `results/summary.csv`, `results/README.md`.

| Snapshot (P_max = 0 dBm, old default) | OTFS-ISAC | OFDM-ISAC | OTFS feedback | genie |
|---|---|---|---|---|
| step0: before any change (`step0_baseline.csv`) | 55.18 ± 0.58 | 39.97 ± 0.62 | 36.86 ± 0.17 | 55.45 ± 0.46 |
| step3: after CRB + tracking + AN (`step3_crb_track_an.csv`) | 55.24 ± 0.55 | 39.99 ± 0.61 | 36.84 ± 0.13 | 55.45 ± 0.46 |
| final, 10 seeds, 50 frames (`summary.csv`, experiment `pmax`) | 56.05 ± 1.03 | 41.71 ± 1.71 | 37.16 ± 1.98 | 56.24 ± 0.97 |

(Old `results/summary.csv`, 1 seed × 100 frames: 54.74 / 39.45 / 37.51 / 54.96.)
**At P_max = 0 dBm the sum-secrecy numbers barely moved** (the link is at ~45 dB SNR, so secrecy ≈ user rate and the
CRB/AN details are second order). What moved is the *validity* of the uncertainty region (coverage, below) and the operating point (item 5).

## 1. CRB (`crb.py`, `waveform.py`, `config.py`)
- Eq. (66) of Appendix A implemented: `C_j = diag(N0/(2|h|²NrΩ), N0/(8π²|h|²NrTc), 6N0/(π²cos²θ|h|²E0Nr(Nr²−1)))`.
  `E0, Ω, Tm, T2, Tc` are integrated numerically from the actual OTFS frame (`waveform.probe_waveform`, rectangular pulse,
  QPSK DD symbols; band-limited interpolation ×4, spectral differentiation) — `crb.pulse_moments`.
  `crb_moments="flat"` keeps the old B_rms² = (M Δf)²/12, T_rms² = (NT)²/12 for comparison.
- Link to R_x kept: snr_el = |h|²E0/N0 = (Gm|β|)²·aᴴ(θ)R_xa(θ)/(P_ref σ²) (`echo_snr_element`), so C_j ∝ 1/illumination
  and C5 stays a linear constraint on the beampattern.
- Docstring documents the appendix assumptions (a1 = 0, α = β = 0, continuous observation, single target, ULA).
  `crb_exact` evaluates the full 5×5 FIM (nuisance hR, hI) without α = β = 0; `coupling_diagnostics` measures the assumption:
  for the actual OTFS frame the max |correlation| of the exact CRB is 0.019 and its diagonal differs from (66) by 4e-4.
- DRAFT banner and the `CRB` draft registration removed.
- **Numbers moved:** Ω/E0 is +0.7 %, Tc/E0 −0.9 % vs the flat approximation for the seed-0 frame (other QPSK realisations: ±1–4 %),
  so the CRB std changes by < 1–2 %. Expected: a random-data OTFS frame is nearly flat in spectrum and time.
- Tests: (a) `test_crb_reduces_to_legacy_formulas_for_flat_pulse` (also checks the 5×5 FIM against (66));
  (b) `test_ml_estimates_approach_crb`; (c) `test_crb_scales_with_Nr_and_snr`. Monte-Carlo (150 trials, Nr = 16, run once while developing):
  MSE/CRB = [1.02, 1.10, 1.11] (γ, ω, θ) at 10 dB, [1.02, 1.10, 1.12] at 20 dB, [1.02, 1.10, 1.12] at 30 dB, [1.01, 1.10, 1.13] at 40 dB.
  (The unit test uses 80 trials and accepts 0.6–1.6.)
- **Reclassified (your call to overrule):** `estimate_delay_doppler` (draws γ̂, ω̂ from N(truth, CRB)) was a DRAFT shortcut.
  Since (b) shows the local ML estimator attains the CRB and the simulated radar SNR is far above that range, it is now
  documented as an *efficient-estimator model* and no longer stamps figures. Not modelled: low-SNR outliers/ambiguity peaks.
- `ml_angle`: added parabolic refinement of the grid peak (`ml_angle_interp`). The 2e-4 rad grid step is ~10× the CRB std
  (≈5e-6 rad), so the raw argmax was quantisation-limited and would have contradicted a CRB-sized region.

## 2. Accumulation across frames (`sensing.py`, `simulator.py`)
- Per parameter (γ, ω, θ): `P_pred = P_prev + Q` (`Estimate.predict`), `1/P_post = 1/P_pred + J_meas`
  (J_meas = 1/C from Eq. (66)), and the **point estimate is fused too**, `η_post = P_post(η_pred/P_pred + J_meas η_meas)`
  — a shrinking region around an unfused point estimate would not contain the truth. `E_j` is built from the *predicted*
  covariance (P_post + Q) used for the next design.
- `Q` (`sensing.process_noise`) is derived from the motion model of `Scenario.step` (speed random walk `speed_process_std`
  = 0.15 m/s per instant → σ_x = σ_v dT along x), mapped to (γ, ω, θ) by the local Jacobian, scaled by `track_q_scale` (default 1).
  `track_accumulate=False` reproduces the old behaviour (single-echo CRB, no Q).
- Logged per frame: `pred/post/meas_std_eta_e` (delay, Doppler, angle CRB std), `region_area_e`, `cover_e`;
  `results/crb_std_per_frame.csv`; figure `fig_uncertainty_region.png` (size of E_j vs time instant).
- **Numbers (P_max = −30 dBm, 10 seeds, 100 frames):** region area first 5 → last 10 instants: **8.0e-3 → 1.2e-3 m² (fused)** vs
  6.2e-3 → 9.3e-3 m² (single frame, no shrinkage). Coverage of the true η by the 95 % ellipsoid: **0.91 (fused)** vs
  **0.22 (single-frame CRB)** — the old region ignored motion uncertainty and under-covered badly. Doppler/angle std of the
  fused region are *larger* than the raw single-frame CRB (floor set by Q, ≈7e-5 rad); range std shrinks 0.2 → 0.02 m.
  Sum secrecy is unchanged (17.08 either way) because the Eves are already tightly localised.

## 3. Artificial noise (`an.py`, `solvers.py`)
- You asked me to follow Su, Liu, Masouros (arXiv 2210.08286). Implemented
  `n_AN = V n̄`, `V = I − A(AᴴA)⁻¹Aᴴ` (null space of the users' channels), `S = V R_n̄ Vᴴ`, `R_n̄ ⪰ 0`
  (Su Eqs. 19–20, 23): `R_n̄ = I` is the isotropic case (Sec. III), `R_n̄ = Σ q_{j,m} a aᴴ + q₀ I` uses the Eve regions
  (Sec. III-A). Atoms are exact PSD pieces, so S ⪰ 0 and a(θ_k)ᴴ S a(θ_k) = 0 at the predicted angles (tested, incl. S = V R_n̄ Vᴴ).
  The parametrisation of R_n̄ (span of region samples + identity) is my choice; Su optimise a general R_n̄ ⪰ 0.
- `grid_split` now also searches the isotropic share ι ∈ {0, ½, 1} of the AN power. Optional `an_deriv_null` also nulls da/dθ (my extension, default off).
- Solvers renamed `mrt_crude_an → mrt_an`, `zf_crude_an → zf_an`. DRAFT banners / registrations (`AN`, `AN-ISO`) removed.
- **Leakage to users (`an_leak_users`, W, measured on the true angles; shown as a ratio to the noise power N0):**
  default P_max = −30 dBm: **1.8e-5·N0 (mean over frames/seeds)**; at 0 dBm 3e-2·N0; at +10 dBm 0.3·N0 (leakage ∝ P_AN while N0 is fixed).
  With `an_deriv_null=True` (seed 0, 40 frames, P_max = 0 dBm): 1.5e-4 → 2.5e-9·N0. So leakage is ≈ 0 at the default operating
  point, but not negligible at very high power; enable `an_deriv_null` if you sweep far above 0 dBm.
- Sum secrecy at P_max = 0 dBm (step3 vs step0): 55.18 → 55.24.

## 4. Rates (`metrics.py`, `solvers.py`)
- The report leaves SINR_k, N_{j,k}, D_{j,k} as TODO, so there was nothing to match; **derived from Eqs. (13), (15), (16) (you approved this)**:
  `SINR_k = h_k|aᴴf_k|² / (Σ_{k'≠k} h_k|aᴴf_k'|² + h_k aᴴSa + N0)`, `N_{j,k} = h_j|aᴴ(θ_j)f_k|²`,
  `D_{j,k} = h_j Σ_{k'≠k}|aᴴf_k'|² + h_j aᴴSa + N0`. Now explicit functions (`user_signal_interference`, `eve_N_D`), returned as `N_eve`, `D_eve`.
  The numerical values are unchanged vs the previous code (it already implemented these expressions); only documentation/structure changed.
- **Flag for the paper:** Eq. (12) writes h_k = (c/4πf_c d_k)², used here as the power gain |h_k|² (Friis). Using it as an amplitude squares the path loss twice.
- Consistency with `solvers.py`: tests `test_rates_match_hand_computation` and `test_design_rates_match_true_rates` check that
  `LinearModel.rates` (design) = `evaluate_true` (evaluation) when prediction = truth, and that the C1 and C4 ingredients match
  the power / beampattern of the returned (F, S). C2/C3/C5 use the same quantities. Note C4/C5 are infeasible for solvers with no Eve-directed
  AN/beam (`mrt_no_an`: feasible in 1 % / 8 % of frames) — by design.

## 5. Sanity checks / operating point (`scripts/sanity_check.py`, `config.py`)
- Old default P_max = 0 dBm: mean per-user single-stream SNR **43 dB**, **14.1 bit/s/Hz per user**, sum 55 bit/s/Hz — unrealistic.
  Sweep (4 seeds, 60 frames, `results/sanity_pmax_sweep.csv`): −40 dBm: 3 dB / 6.6; −30: 13 dB / 17.1; −20: 23 dB / 29.7; −10: 33 dB / 42.7; 0: 43 dB / 55.6 (SNR / sum secrecy).
- **New default `P_max_dBm = −30`**: ~13–15 dB mean user SNR, ~4.4 bit/s/Hz per user, sum secrecy **17.1 ± 0.9** (10 seeds). Because
  `P_probe` and `gamma_sens` were absolute 0 dBm, they now default to `None` = P_max (relative; the radar model already normalises by P_max),
  which reproduces the old numbers at 0 dBm.
- **Dip around instants 20–24 (seed 0; all frameworks including the genie):** not a bug. Users 2 and 3 cross in angle:
  sin θ = 0.444 vs 0.441 at t = 22 (min separation 0.0023 at t = 23), against a beamwidth 2/Nt = 0.031. A LoS-only ULA cannot
  separate two users at the same angle (collinear steering vectors, MRT and ZF both fail), so with equal power split each gets
  SINR ≈ 1 → R ≈ 1.02 bit/s/Hz (the logged R_users of users 2, 3) and the sum falls from ~57 to ~26 (at 0 dBm; at −30 dBm 9.5 vs 16.8).
  Other seeds have the same event at other times (seed 1: t = 7–11, seed 2: t = 12–20; seed 3: none), so it averages out in the 10-seed mean.
  Possible mitigation (not implemented): unequal power allocation — `grid_split` uses equal user powers.

## 6. Experiments (`run_demo.py`)
- 10 seeds (0–9), mean ± std (ddof = 1) of per-seed means; error bars/bands on all figures. Added: sum secrecy vs P_max, vs Nt, OTFS-ISAC vs baselines,
  uncertainty region vs time. Parallelised. `summary.csv` now has `<metric>_mean/_std` columns (plus `per_seed.csv`).
- The "DRAFT MODELS: crude AN + crude CRB" stamp is gone; a figure is stamped only if a run behind it touched a registered placeholder
  (`FEEDBACK`, new `OFDM-SENSING`), so the feedback/OFDM comparisons stay stamped.
- Headline (P_max = −30 dBm, 100 frames, 10 seeds): OTFS-ISAC 17.08 ± 0.91, OFDM-ISAC 15.82 ± 0.84, OTFS feedback 14.10 ± 0.64, genie 17.08 ± 0.91;
  no AN 16.30 ± 0.90, isotropic AN 14.86 ± 0.90 (same seeds). The OTFS–OFDM gap grows with P_max (ICI): 41.7 vs 56.0 at 0 dBm.

## 7. Solvers
- `grid_split` is the main solver; `slsqp` is documented/tested as refinement only (warm start from `grid_split`; mean 17.14 vs 17.08).
  Caveat: its "better of the two" rule orders by (constraint violation, objective), so when the problem is infeasible (e.g. AN budget 0: C4/C5 cannot be met)
  SLSQP may pick a less-violating, lower-objective point (12.4 vs 17.1 at budget 0 in `an_budget`).
- `sca_sdr` is still a STUB (optional; cvxpy not installed, not attempted). `drl` stub unchanged.

## Housekeeping
- Added `.gitignore`, removed tracked `__pycache__/*.pyc` from the index, `pytest` in requirements, `scripts/`.
- 18 tests in `tests/test_core.py`; old tests updated for renamed APIs (`crude_*`).

## Still placeholders / open
- Feedback benchmark (stylised), OFDM-ISAC sensing (reuses the OTFS CRB), `sca_sdr`, `drl`.
- Radar SNR normalisation (Gm = MN as an *amplitude* gain, σ² = 1, illumination/P_max) gives ~30–50 dB per-element SNR — [ASSUMED]; it is why E_j is only ~1 cm² and Eves are almost point-localised.
- Eves treat other users' streams as noise (no SIC); h_k convention (above); Sec. VII-C (problem formulation) is still TODO in the report.
