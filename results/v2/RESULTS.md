# Results, version 2

Produced by `scripts/evaluate_v2.py` and `scripts/mateus_check.py` under PREREGISTRATION_v2.md (hash `5dc7a927085f8e46`).

## A. Blind test: Dynasore, Doorn et al. (2024)

| Quantity | Value |
|---|---|
| Wells scored | 10 |
| Top-1 (u_rel or tau_d) | **2/10 = 0.20**, 95% CI 0.06 to 0.51 |
| Chance | 0.20 |
| P under chance | 0.62 |
| Pre-registered success | at least 5 of 10: **not met** |
| Mechanism class (adaptation and short-term plasticity) | 0.80 (chance 0.30) |
| Top-2 | 0.40 |
| Direction up, among correct calls | 1.00 (2 wells) |
| Wells with a call above 0.5 | 0.90 |

Per well: the named mechanism, its probability, and the effect if it moved.

| Well | Top-1 | p | Effect of top-1 (transformed units) | u_rel p | tau_d p | Guard |
|---|---|---|---|---|---|---|
| FB2_B6 | `g_ampa` | 0.94 | +0.63 [+0.14, +1.11] | 0.43 | 0.32 | outside |
| FB2_C6 | `tau_d` | 0.71 | +1.38 [+1.13, +1.57] | 0.10 | 0.71 | outside |
| FB2_D6 | `g_ahp` | 0.95 | +1.19 [+0.86, +1.55] | 0.16 | 0.26 | outside |
| FB2t_A2 | `tau_d` | 0.92 | +1.54 [+1.26, +1.74] | 0.08 | 0.92 | outside |
| FB2t_A3 | `g_ahp` | 0.93 | +0.08 [+0.01, +0.48] | 0.57 | 0.24 | inside |
| FB2t_B1 | `g_ahp` | 0.83 | +0.55 [+0.26, +0.86] | 0.74 | 0.23 | inside |
| FB2t_C1 | `g_ahp` | 0.96 | +0.54 [+0.16, +0.91] | 0.48 | 0.29 | outside |
| FB3t_A3 | `g_ahp` | 0.96 | +1.13 [+0.74, +1.50] | 0.05 | 0.02 | outside |
| FB3t_C3 | `g_ahp` | 0.80 | +0.65 [+0.21, +1.21] | 0.04 | 0.32 | inside |
| FB3t_D1 | `g_gaba` | 0.41 | +1.43 [+0.33, +3.36] | 0.09 | 0.04 | outside |

| Method | Top-1 |
|---|---|
| Hodgkin's Razor | 0.20 |
| Same twin, pairing removed | 0.80 |
| Doorn et al. 2025 estimator (U or tau_D accepted) | 0.80 |
| Hodgkin's Razor, wells the guard passes (3) | 0.00 |

## B. Chip readout prediction

Simulated chips: recovery r and 90% interval coverage of each chip parameter, by readout (held-out chips).

| Readout | `p_cross` r (cov.) | `direction_sel` r (cov.) | `g_cross` r (cov.) | `tgt_autonomy` r (cov.) |
|---|---|---|---|---|
| compartment | 0.17 (0.93) | -0.01 (0.95) | 0.26 (0.91) | 0.39 (0.90) |
| calcium_2hz | 0.16 (0.89) | 0.02 (0.92) | 0.26 (0.90) | 0.33 (0.89) |
| channel | 0.59 (0.89) | 0.37 (0.88) | 0.03 (0.89) | 0.14 (0.89) |
| perfusion | 0.23 (0.88) | 0.11 (0.96) | 0.35 (0.90) | 0.40 (0.91) |
| compartment+channel | 0.73 (0.90) | 0.45 (0.91) | 0.28 (0.93) | 0.41 (0.92) |

Simulated AUROC, strong diode against symmetric channel: channel statistic 0.78, chamber statistic 0.52.

NMDA reduction on 191 source-driven chips: target calcium event frequency lower in 0.30, target synchrony lower in 0.32 of chips (Lassus et al.: both lower).

Recorded chips (Mateus et al. 2024), diode (Rams, Arrows) against straight channels:

| Statistic | AUROC | 95% CI (chips resampled) | Pre-registered | Outcome |
|---|---|---|---|---|
| Channel electrodes: dominant share of propagation | 0.62 | 0.28 to 0.88 | separates (>= 0.75) | **not met** |
| Chamber electrodes: cross-correlation asymmetry | 0.41 | 0.20 to 0.60 | does not separate (< 0.7) | **met** |

33 recordings from 17 chips scored.

| Design | Recordings | Median dominant share | Median chamber asymmetry |
|---|---|---|---|
| arrows | 13 | 0.84 | 0.16 |
| control | 15 | 0.73 | 0.21 |
| rams | 11 | 0.91 | 0.25 |
| tesla | 8 | 0.79 | 0.17 |
| tesla_v2 | 10 | 0.84 | 0.16 |

## C. Development set: Tampere (fifth scoring, not blind)

| Quantity | Version 1 key | Version 2 key |
|---|---|---|
| Top-1 | **0.24** (16/66) | **0.24** (16/66) |
| Chance | 0.10 | 0.12 |
| Top-2 | 0.35 | 0.35 |
| Mechanism class | 0.33 | 0.33 |
| Control false-mechanism rate (ceiling 0.20) | 0.00 | - |
| Treated wells with a call | 0.32 | - |
| Accuracy of those calls | 0.52 | 0.52 |
| Detection AUROC, treated against control | 0.89 | - |

| Compound | Wells | Top-1 (v1 key) | Rat | Human |
|---|---|---|---|---|
| CNQX | 11 | 4/11 | 4/7 | 0/4 |
| D-AP5 | 11 | 1/11 | 1/7 | 0/4 |
| GABA | 11 | 0/11 | 0/7 | 0/4 |
| Gabazine | 11 | 2/11 | 2/7 | 0/4 |
| Kainic acid | 11 | 5/11 | 5/7 | 0/4 |
| TTX | 11 | 4/11 | 4/7 | 0/4 |

| Method | Sees labels | Top-1 (v1 key) |
|---|---|---|
| Hodgkin's Razor v2 | no | 0.24 |
| Same twin, pairing removed | no | 0.09 |
| Doorn et al. 2025 estimator | no | 0.15 |
| Supervised nearest centroid, leave one well out | yes | 0.52 |

Wells the guard passes: 39; top-1 on them (v2 key) 0.39.

## D. Transfer across species (Tampere)

| Representation | Rat to human | Human to rat |
|---|---|---|
| Raw feature change | 0.25 (n 24, chance 0.17) | 0.26 (n 42, chance 0.17) |
| Twin reading (presence, effect) | 0.38 (n 24, chance 0.17) | 0.14 (n 42, chance 0.17) |

## E. Simulations

**grid16**

| Case | n | Top-1 | Top-2 | Class |
|---|---|---|---|---|
| all single mechanism | 3356 | 0.57 | 0.75 | 0.65 |
| saturating | 536 | 0.77 | 0.90 | 0.80 |
| chance | | 0.10 | | 0.25 |

| Mechanism | Presence AUROC | ECE | Coverage 50/80/90 | Effect r (given active) |
|---|---|---|---|---|
| `g_na` | 0.90 | 0.012 | 0.41 / 0.75 / 0.88 | 0.84 (128) |
| `g_kdr` | 0.56 | 0.019 | 0.49 / 0.80 / 0.87 | 0.30 (112) |
| `g_ahp` | 0.77 | 0.004 | 0.41 / 0.75 / 0.86 | 0.71 (116) |
| `g_ampa` | 0.95 | 0.010 | 0.44 / 0.75 / 0.89 | 0.50 (120) |
| `g_nmda` | 0.85 | 0.008 | 0.43 / 0.77 / 0.88 | 0.78 (115) |
| `g_gaba` | 0.86 | 0.010 | 0.38 / 0.73 / 0.86 | 0.73 (150) |
| `g_tonic_inh` | 0.72 | 0.009 | 0.44 / 0.77 / 0.89 | 0.68 (117) |
| `tau_d` | 0.83 | 0.006 | 0.45 / 0.76 / 0.86 | 0.68 (103) |
| `u_rel` | 0.77 | 0.010 | 0.46 / 0.77 / 0.89 | 0.71 (98) |
| `i_drive` | 0.66 | 0.011 | 0.45 / 0.79 / 0.89 | 0.57 (95) |

**grid12**

| Case | n | Top-1 | Top-2 | Class |
|---|---|---|---|---|
| all single mechanism | 3161 | 0.75 | 0.88 | 0.82 |
| saturating | 614 | 0.77 | 0.90 | 0.82 |
| chance | | 0.10 | | 0.25 |

| Mechanism | Presence AUROC | ECE | Coverage 50/80/90 | Effect r (given active) |
|---|---|---|---|---|
| `g_na` | 0.95 | 0.008 | 0.42 / 0.74 / 0.89 | 0.93 (120) |
| `g_kdr` | 0.66 | 0.016 | 0.44 / 0.76 / 0.86 | 0.41 (124) |
| `g_ahp` | 0.95 | 0.003 | 0.43 / 0.76 / 0.88 | 0.73 (77) |
| `g_ampa` | 0.96 | 0.006 | 0.45 / 0.76 / 0.89 | 0.74 (109) |
| `g_nmda` | 0.92 | 0.011 | 0.44 / 0.76 / 0.89 | 0.74 (84) |
| `g_gaba` | 0.88 | 0.007 | 0.43 / 0.78 / 0.89 | 0.76 (120) |
| `g_tonic_inh` | 0.70 | 0.009 | 0.42 / 0.74 / 0.86 | 0.48 (100) |
| `tau_d` | 0.94 | 0.008 | 0.49 / 0.78 / 0.87 | 0.84 (130) |
| `u_rel` | 0.91 | 0.008 | 0.45 / 0.78 / 0.89 | 0.86 (111) |
| `i_drive` | 0.74 | 0.008 | 0.45 / 0.79 / 0.89 | 0.70 (96) |


## E. The guard

| System | Case | Must | n | Fired | Typicality fired | Check fired | Outcome |
|---|---|---|---|---|---|---|---|
| grid16 | bank holdout | pass, at most 0.10 | 60 | 0.03 | 0.00 | 0.03 | **met** |
| grid16 | variant kinetics | fire, at least 0.80 | 60 | 0.53 | 0.30 | 0.43 | **not met** |
| grid16 | shuffled real | fire, at least 0.80 | 60 | 0.45 | 0.20 | 0.32 | **not met** |
| grid12 | bank holdout | pass, at most 0.10 | 60 | 0.02 | 0.02 | 0.02 | **met** |
| grid12 | variant kinetics | fire, at least 0.80 | 60 | 0.95 | 0.92 | 0.83 | **met** |
| grid12 | shuffled real | fire, at least 0.80 | 50 | 1.00 | 1.00 | 1.00 | **met** |

Thresholds: grid16: typicality 10.04, predictive check 5.03; grid12: typicality 9.54, predictive check 4.03.
