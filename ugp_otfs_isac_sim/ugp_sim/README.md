# OTFS-ISAC sum-secrecy simulation environment (UGP)

Simulation environment for *Sum-Secrecy Rate Maximization in OTFS-Assisted ISAC*
(latest Google Doc draft). Parameters follow Yuan et al., *ISAC-assisted OTFS
Transmission for Vehicular Networks*, arXiv:2105.03125, Sec. IV.

```
pip install numpy scipy matplotlib          # cvxpy only needed for the sca_sdr stub
python tests/test_core.py                   # 8 sanity checks
python run_demo.py --frames 100 --seeds 0   # ~2 min, writes results/
```

> **DRAFT components.** The AN matrix formulation and the CRB matrix are crude
> placeholders. They are marked with a `DRAFT / PLACEHOLDER` banner in the code,
> raise a `DraftWarning` on first use, and every run ends with a report of the
> placeholders it touched. All figures carry a "DRAFT MODELS" stamp. Search the
> code for `DRAFT` and `TODO`.

## Layout

| Module | Contents | Status |
|---|---|---|
| `config.py` | `SimParams`; every value tagged Yuan21 / Draft / ASSUMED (`PARAM_SOURCES`) | ready |
| `geometry.py` | road, kinematics, ULA steering, path loss (Friis or draft formula) | ready |
| `waveform.py` | ISFFT/SFFT, Heisenberg/Wigner, DD channel, OTFS + OFDM link level; system-level Doppler/ICI and CP-overhead abstraction | ready |
| `sensing.py` | echo model, ML angle search, localisation, speed, prediction; feedback benchmark | DD estimator is a shortcut (DRAFT) |
| `crb.py` | crude diagonal FIM/CRB, ellipsoid E_j sampling, CRB → min-illumination | **DRAFT** |
| `an.py` | null-space projector, AN basis, S, AN power, heuristic allocations | **DRAFT** |
| `metrics.py` | true-geometry rates, Eve rates, secrecy, beampattern | ready |
| `solvers.py` | `Problem`, linear model, constraint checks, solver registry | see below |
| `frameworks.py` | OTFS-ISAC, OFDM-ISAC, OTFS-feedback, genie | ready (OFDM sensing reuses crude CRB) |
| `simulator.py` | time loop: predict → design → evaluate on truth → sense | ready |
| `draft.py` | registry/warnings for placeholders | — |

## Problem implemented

max Σ_k [R_k − max_j max_{η∈E_j} R_{j,k}^Eve]^+ over F, q ≥ 0, subject to

- C1 Σ‖f_k‖² + tr(S) ≤ P_max
- **C2 tr(S) = Σ q_{j,m}‖ã_{j,m}‖² ≤ P_AN,max** — new AN budget (`P_an_max_frac`, default 30 % of P_max)
- C3 R_k ≥ R_min
- C4 aᴴ(θ_i)(Σ f fᴴ + S)a(θ_i) ≥ Γ_sens, ∀i ∈ K ∪ J
- C5 I⁻¹(η_j) ≼ C_j^req — with the crude diagonal FIM this becomes a minimum
  illumination toward each Eve (linear in the powers)

All solvers use f_k = √p_k w_k (w_k = MRT, as in the draft, or ZF) and S over the
fixed AN basis, which keeps C1, C2, C4, C5 linear in (p, q).

| Solver | Notes |
|---|---|
| `mrt_no_an`, `mrt_crude_an`, `zf_crude_an`, `isotropic_an` | baselines |
| `grid_split` | 1-D search over AN fraction × {MRT, ZF} |
| `slsqp` | SciPy SLSQP over (p, q), smoothed objective, C1–C5 imposed, warm start from grid |
| `sca_sdr`, `drl` | **stubs** — interface and plan in docstrings |

Add a solver: write `def solve_x(prob) -> Solution` and decorate with `@register("x")`.

## Deviations from the draft (deliberate — check before writing them up)

1. **ΔT clash.** Yuan21's ΔT = 0.02 s is the tracking step, not the frame NT = 5 ms. Kept separate (`dT` vs `frame_duration`).
2. **Speed from Doppler.** The draft's s = cω/(f_c cos θ) uses the other angle convention and omits the round-trip factor 2. Code uses v = −cω/(2 f_c sin θ), consistent with sin θ = x/d.
3. **Angle grid.** The draft's Θ = {π/N_t, …, π} is too coarse and lies in [0, π]. A fine local grid around θ̄ is used.
4. **OTFS DD relation.** With rectangular pulses, y[k,l] = h·x[k−k_ν, l] carries an extra known phase e^{j2πl k_ν/(MN)}. It is power-neutral and compensable, but the draft equation omits it (verified in tests).
5. **User SINR** includes AN leakage h_k² aᴴ(θ_k)S a(θ_k). This is zero only when the predicted and true angles coincide; the draft drops the term.
6. **Path loss.** The draft/Yuan21 h² = c/(4πf_c d²) is not Friis; `pathloss_model="draft"` reproduces it, and the default is Friis.
7. **Radar normalisation.** Echo SNR uses Yuan21's normalised model (σ² = 1) with illumination divided by P_max.

## Results from the default run (100 instants, seed 0, provisional)

| Framework (grid_split) | mean sum secrecy (bit/s/Hz) |
|---|---|
| OTFS-ISAC | 54.7 |
| OFDM-ISAC | 39.5 |
| OTFS pilot feedback | 37.5 |
| OTFS genie | 55.0 |

Absolute rates are high because the link SNR is high at the assumed P_max = 0 dBm.
The main knobs are `P_max_dBm` and `noise_figure_dB`.
