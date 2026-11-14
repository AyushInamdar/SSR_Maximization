# OTFS-ISAC sum-secrecy simulation environment (UGP)

Simulation environment for *Sum-Secrecy Rate Maximization in OTFS-Assisted ISAC*
(latest Google Doc draft). Parameters follow Yuan et al., *ISAC-assisted OTFS
Transmission for Vehicular Networks*, arXiv:2105.03125, Sec. IV.

```
pip install numpy scipy matplotlib pytest   # cvxpy only needed for the sca_sdr stub
python -m pytest tests -q                    # or: python tests/test_core.py
python run_demo.py                           # 10 seeds, writes results/ (~15 min on 12 cores)
python scripts/sanity_check.py               # per-user SNR/rate, P_max sweep, angular-crossing diagnostic
```

Every model change, with before/after numbers, is listed in [CHANGELOG.md](CHANGELOG.md);
the experiments and the exact reproduction command are in [results/README.md](results/README.md).

> **Placeholders.** Remaining placeholders (stylised feedback benchmark, OFDM sensing reusing the
> OTFS CRB, `sca_sdr`/`drl` stubs) are registered in `draft.py`; a figure is stamped "DRAFT" only if
> a run behind it used one. Search the code for `DRAFT` and `TODO`.

## Layout

| Module | Contents | Status |
|---|---|---|
| `config.py` | `SimParams`; every value tagged Yuan21 / Draft / ASSUMED (`PARAM_SOURCES`) | ready |
| `geometry.py` | road, kinematics, ULA steering, path loss (Friis or draft formula) | ready |
| `waveform.py` | ISFFT/SFFT, Heisenberg/Wigner, DD channel, OTFS + OFDM link level; system-level Doppler/ICI and CP-overhead abstraction | ready |
| `sensing.py` | echo model, ML angle search, localisation, speed, prediction; feedback benchmark | ML angle (parabolic refinement), information-filter tracking; DD estimate drawn from N(truth, CRB) (efficient-estimator model, validated by test) |
| `crb.py` | CRB of Report App. A Eq. (66) from the actual OTFS waveform moments, exact 5×5 FIM check, ellipsoid E_j, CRB → min-illumination | ready |
| `an.py` | AN per Su et al. (2210.08286): S = V R_n̄ Vᴴ, V = null-space projector of the users; atoms, allocation | ready |
| `metrics.py` | Eqs. (27)–(29): SINR_k, N_{j,k}, D_{j,k} (derived, not in the report), secrecy | ready (derived) |
| `solvers.py` | `Problem`, linear model, constraint checks, solver registry | see below |
| `frameworks.py` | OTFS-ISAC, OFDM-ISAC, OTFS-feedback, genie | OFDM sensing reuses the OTFS CRB (DRAFT); feedback benchmark stylised (DRAFT) |
| `simulator.py` | time loop: predict → design → evaluate on truth → sense | ready |
| `draft.py` | registry/warnings for placeholders | — |

## Problem implemented

max Σ_k [R_k − max_j max_{η∈E_j} R_{j,k}^Eve]^+ over F, q ≥ 0, subject to

- C1 Σ‖f_k‖² + tr(S) ≤ P_max
- **C2 tr(S) = Σ q_{j,m}‖ã_{j,m}‖² ≤ P_AN,max** — new AN budget (`P_an_max_frac`, default 30 % of P_max)
- C3 R_k ≥ R_min
- C4 aᴴ(θ_i)(Σ f fᴴ + S)a(θ_i) ≥ Γ_sens, ∀i ∈ K ∪ J
- C5 I⁻¹(η_j) ≼ C_j^req — with the diagonal CRB of Eq. (66), every entry ∝ 1/illumination, so this is a minimum
  illumination toward each Eve (linear in the powers)

All solvers use f_k = √p_k w_k (w_k = MRT, as in the draft, or ZF) and S over the
fixed AN basis, which keeps C1, C2, C4, C5 linear in (p, q).

| Solver | Notes |
|---|---|
| `mrt_no_an`, `mrt_an`, `zf_an`, `isotropic_an` | baselines |
| `grid_split` | **main solver**: search over AN fraction × isotropic share × {MRT, ZF} |
| `slsqp` | refinement only: SciPy SLSQP over (p, q), warm-started from `grid_split`, never worse than it |
| `sca_sdr`, `drl` | **stubs** — interface and plan in docstrings |

Add a solver: write `def solve_x(prob) -> Solution` and decorate with `@register("x")`.

## Deviations from the draft (deliberate — check before writing them up)

1. **ΔT clash.** Yuan21's ΔT = 0.02 s is the tracking step, not the frame NT = 5 ms. Kept separate (`dT` vs `frame_duration`).
2. **Speed from Doppler.** The draft's s = cω/(f_c cos θ) uses the other angle convention and omits the round-trip factor 2. Code uses v = −cω/(2 f_c sin θ), consistent with sin θ = x/d.
3. **Angle grid.** The draft's Θ = {π/N_t, …, π} is too coarse and lies in [0, π]. A fine local grid around θ̄ is used.
4. **OTFS DD relation.** With rectangular pulses, y[k,l] = h·x[k−k_ν, l] carries an extra known phase e^{j2πl k_ν/(MN)}. It is power-neutral and compensable, but the draft equation omits it (verified in tests).
5. **User SINR** includes AN leakage h_k² aᴴ(θ_k)S a(θ_k). This is zero only when the predicted and true angles coincide; the draft drops the term.
6. **Path loss.** The draft/Yuan21 h² = c/(4πf_c d²) is not Friis; `pathloss_model="draft"` reproduces it, and the default is Friis.
7. **Radar normalisation.** Echo SNR uses Yuan21's normalised model (σ² = 1) with illumination divided by P_max; the resulting per-element radar SNR is very high (~30–50 dB), [ASSUMED] and worth revisiting.
8. **Report Eq. (12)** writes h_k = (c/4πf_c d_k)², which is used here as the *power* gain |h_k|² (Friis); using it as an amplitude would square the path loss twice.

## Results

See [results/README.md](results/README.md) (10 seeds, mean ± std) and [CHANGELOG.md](CHANGELOG.md).
