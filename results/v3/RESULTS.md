# Results, version 3

Produced by `scripts/evaluate_v3.py` under PREREGISTRATION_v3.md (hash `2e1e630e6bffbecf`), run through `scripts/evaluate_v3_lowmem.py`, which feeds the frozen code one sister pair at a time.

## Stop rule (committed before the twin was trained)

| Attempt | Twin digest | `g_nmda` alone (bar 0.80) | With a co-shift (bar 0.70) | Passed |
|---|---|---|---|---|
| 1 | `c4096c60149d` | 0.792 | 0.747 | no |
| 2 | `a4f079b64bb4` | 0.810 | 0.755 | yes |

## A. Blind test: chronic APV on sister cultures (Charlesworth et al. 2015)

| Pre-registered outcome | Bar | Result | Outcome |
|---|---|---|---|
| **Primary**: AUROC of p(`g_nmda`), treated against genotype-matched null preparations, 10–14 days | ≥ 0.70, lower 95% bound > 0.50 | **0.864** [0.74, 0.96] | **met** |
| **Co-primary**: share with `g_nmda` top-1, treated against null | higher, one-sided Fisher p < 0.05 | 2/29 against 1/23, p = 0.59 | **not met** |

| Contrast | Treated / null preparations | AUROC of p(`g_nmda`) [95% CI] | Top-1 `g_nmda` | Median p(`g_nmda`) | Detection AUROC |
|---|---|---|---|---|---|
| Primary ages, genotype-matched null | 29 / 23 | **0.86** [0.74, 0.96] | 2/29 vs 1/23 (p 0.59) | 0.12 vs 0.10 | 0.62 |
| Primary ages, pooled null (all genotypes) | 29 / 78 | **0.79** [0.70, 0.87] | 2/29 vs 4/78 (p 0.52) | 0.12 vs 0.10 | 0.60 |
| Primary ages, GluR1 only | 13 / 6 | **0.71** [0.45, 0.92] | 1/13 vs 0/6 (p 0.68) | 0.12 vs 0.11 | 0.85 |
| Primary ages, WT only | 16 / 17 | **0.92** [0.78, 1.00] | 1/16 vs 1/17 (p 0.74) | 0.14 vs 0.10 | 0.53 |
| Late (15 days and after), genotype-matched null | 44 / 24 | **0.48** [0.33, 0.63] | 1/44 vs 0/24 (p 0.65) | 0.10 vs 0.10 | 0.50 |
| Late, pooled null | 44 / 79 | **0.49** [0.38, 0.60] | 1/44 vs 0/79 (p 0.36) | 0.10 vs 0.10 | 0.63 |

**Canalization** (pre-registered secondary): in the 29 treated preparations recorded at both ages, the median presence of `g_nmda` is 0.125 at 10–14 days and 0.099 at 15 days and after; one-sided Wilcoxon signed-rank p = 3e-05.

Calls above 0.5: 0.00 of treated and 0.00 of null preparations. The twin ranks treated preparations above untreated ones on `g_nmda`, but no preparation's presence probability reaches 0.5: the reading is a ranking against untreated sisters, not a call on one culture.
Direction among top-1 hits: 1.00 down (2 preparations).
Windows scored: 14607.

**Exploratory: every mechanism, treated against matched null** (no bar). AUROC of each mechanism's presence probability:

| Mechanism | AUROC | Median treated | Median null |
|---|---|---|---|
| `g_nmda` | 0.86 | 0.125 | 0.103 |
| `tau_d` | 0.79 | 0.127 | 0.089 |
| `g_na` | 0.70 | 0.095 | 0.087 |
| `g_ahp` | 0.55 | 0.108 | 0.093 |
| `g_ampa` | 0.55 | 0.052 | 0.045 |
| `u_rel` | 0.52 | 0.095 | 0.097 |
| `g_gaba` | 0.48 | 0.136 | 0.136 |
| `i_drive` | 0.44 | 0.129 | 0.132 |
| `g_kdr` | 0.37 | 0.148 | 0.152 |
| `g_tonic_inh` | 0.18 | 0.123 | 0.132 |

The top-1 counts show why the co-primary failed while the primary passed: the most probable mechanism is usually another one in both groups, and `g_nmda` rises relative to untreated preparations without becoming the largest.

| Group | Top-1 counts |
|---|---|
| Treated | `g_gaba` 10, `g_kdr` 7, `tau_d` 7, `g_na` 2, `g_nmda` 2, `g_ampa` 1 |
| Null | `g_kdr` 13, `g_gaba` 5, `u_rel` 2, `tau_d` 1, `g_nmda` 1, `g_ahp` 1 |

## B. Comparators on the same preparations

| Contrast | Treated / null preparations | AUROC of p(`g_nmda`) [95% CI] | Top-1 `g_nmda` | Median p(`g_nmda`) | Detection AUROC |
|---|---|---|---|---|---|
| Hodgkin's Razor (paired twin) | 29 / 23 | **0.86** [0.74, 0.96] | 2/29 vs 1/23 (p 0.59) | 0.12 vs 0.10 | 0.62 |
| Same twin, pairing removed (unpaired) | 29 / 23 | **0.66** [0.51, 0.81] | 1/29 vs 5/23 (p 1) | 0.31 vs 0.26 | 0.72 |
| Doorn et al. 2025 estimator (score: standardised g_NMDA shift, downward) | 29 / 23 | **0.49** [0.33, 0.65] | 2/29 vs 5/23 (p 0.98) | -0.10 vs -0.01 | 0.54 |

For the unpaired twin and the Doorn estimator, the "median p" column is their own score, not a probability.

## C. The guard

| Test | Bar | Fire rate | Outcome |
|---|---|---|---|
| Held-out simulated sister pairs | ≤ 0.10 | 0.02 (n 60) | met |
| Simulated pairs with unmodelled receptor kinetics | ≥ 0.80 | 0.58 (n 60) | not met |
| Recorded pairs, electrodes circularly shifted | ≥ 0.80 | 0.63 (n 60) | not met |

On the recorded windows it fires on 0.35 of treated and 0.46 of null windows.
Preparations outside the model (most windows fire): 37 of 107.
Primary contrast restricted to preparations the guard passes: AUROC 0.89 [0.75, 0.98] (24 treated, 16 null).

## D. Simulations

| Held-out simulated sister pairs | n | Top-1 | Top-2 | Class |
|---|---|---|---|---|
| all single mechanism | 7858 | 0.31 | 0.45 | 0.45 |
| saturating | 1730 | 0.43 | 0.58 | 0.57 |

| Mechanism | Presence AUROC | 90% coverage | Effect r |
|---|---|---|---|
| `g_na` | 0.71 | 0.89 | 0.76 |
| `g_kdr` | 0.54 | 0.85 | 0.21 |
| `g_ahp` | 0.71 | 0.88 | 0.63 |
| `g_ampa` | 0.80 | 0.89 | 0.73 |
| `g_nmda` | 0.67 | 0.88 | 0.68 |
| `g_gaba` | 0.60 | 0.87 | 0.53 |
| `g_tonic_inh` | 0.58 | 0.87 | 0.44 |
| `tau_d` | 0.74 | 0.89 | 0.54 |
| `u_rel` | 0.72 | 0.89 | 0.74 |
| `i_drive` | 0.56 | 0.88 | 0.41 |

## Version 2 blind test, re-read with null pairs (post hoc)

| Method | Dynasore pairs: top-1 in {`u_rel`, `tau_d`} | Pre-drug null pairs: same | Null pairs with a call |
|---|---|---|---|
| Hodgkin's Razor (paired twin) | 2/10 | 2/10 | 0/10 |
| Same twin, pairing removed | 8/10 | 6/10 | 1/10 |
| Doorn et al. estimator | 8/10 | 2/10 ignoring direction; 0/10 in the answer's direction (up) | - |

## Cortico-striatal chip (Lassus et al. 2018)

**First run** (prediction `docs/LASSUS_PREDICTION.md`, committed before the run). 109 chips scored. Striatal rate with the cortex silent: 0.223 Hz, with it driving: 0.244 Hz.

| Readout, NMDA x0.3 on the striatum | Published | Share of chips lower | Median change | Wilcoxon p (lower) | Criterion |
|---|---|---|---|---|---|
| striatal calcium-event frequency | lower | 0.50 | -0.2% | 0.46 | not met |
| striato-striatal synchrony | lower | 0.47 | +0.2% | 0.55 | not met |
| cortico-striatal synchrony | lower | 0.52 | -0.4% | 0.29 | not met |

**Second run** (prediction `docs/LASSUS_PREDICTION_2.md`, committed before the run). 34 chips scored, down state -6 pA, chips kept only if the cortex drives the striatum. Striatal rate with the cortex silent: 0.000 Hz, with it driving: 0.148 Hz.

| Readout, NMDA x0.3 on the striatum | Published | Share of chips lower | Median change | Wilcoxon p (lower) | Criterion |
|---|---|---|---|---|---|
| striatal calcium-event frequency | lower | 0.59 | -3.3% | 0.49 | not met |
| striato-striatal synchrony | lower | 0.76 | -9.0% | 0.0023 | met |
| cortico-striatal synchrony | lower | 0.62 | -3.6% | 0.0077 | met |


## Chip sample size

| Chips per design | Power if the twin is right | Power at the recorded separation (0.62) |
|---|---|---|
| 6 | 0.49 | 0.20 |
| 8 | 0.50 | 0.22 |
| 10 | 0.59 | 0.23 |
| 15 | 0.66 | 0.25 |
| 20 | 0.83 | 0.29 |
| 30 | 0.90 | 0.41 |
| 50 | 1.00 | 0.59 |
| 100 | 1.00 | 0.86 |

Channel statistic, simulated AUROC 0.77. Chips per design for 80% power: **20** if the twin is right, **100** at the recorded separation. The recorded test (17 chips, about 8 per design) had power 0.48 even if the twin is right.

## Bath GABA on simulations

| Simulated condition | Rate after / before | Top-1 correct | Top-1 `g_na` | Top-1 inhibition |
|---|---|---|---|---|
| bath GABA, tonic +0.01 | 0.86 | 0.15 | 0.00 | 0.19 |
| bath GABA, tonic +0.02 | 0.70 | 0.24 | 0.03 | 0.25 |
| bath GABA, tonic +0.04 | 0.39 | 0.19 | 0.23 | 0.21 |
| bath GABA, saturating (0.25) | 0.00 | 0.33 | 0.64 | 0.33 |
| synaptic GABA-A x3 | 0.66 | 0.41 | 0.01 | 0.44 |
| sodium block x0.4 | 0.14 | 0.59 | 0.59 | 0.08 |

## Next-experiment recommendation, on simulations

| Follow-up policy | Ties resolved | 95% CI |
|---|---|---|
| Recommended by the twin | 65/120 = 0.54 | [0.45, 0.63] |
| Fixed best compound (cross-fitted) | 56/120 = 0.47 | [0.38, 0.56] |
| Random compound | 59/120 = 0.49 | [0.40, 0.58] |
| Record the same well again | 51/120 = 0.42 | [0.34, 0.51] |
| Oracle (best compound per case, known only in hindsight) | 120/120 = 1.00 | [0.97, 1.00] |
| Mean over every compound | 0.53 | - |

Recommended against fixed best: 26 ties only the recommendation resolved, 17 only the other; one-sided exact p = 0.11.

Recommended against random: 30 ties only the recommendation resolved, 24 only the other; one-sided exact p = 0.25.
