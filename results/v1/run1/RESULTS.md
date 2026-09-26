# Results

Produced by `scripts/evaluate.py`. Pre-registration hash `c40708bb19668164`, verified before the run.

Bank: 110000 simulated pairs, 6600 held out for validation and 6600 for calibration. Windows of 60 s.

## Primary metric

| Quantity | Value | Pre-registered | Outcome |
|---|---|---|---|
| Top-1 mechanism accuracy | **0.091** (66 wells) | at least 0.50 | **not met** |
| Chance | 0.111 | | |
| Control false-mechanism rate | **0.000** (11 wells) | at most 0.20 | met |
| Top-2 accuracy | 0.197 | | |
| Direction agreement | 1.000 (6 wells) | | |

## Against the baselines

| Method | Sees compound labels | Top-1 accuracy |
|---|---|---|
| Hodgkin's Razor | no | **0.091** |
| Same twin, paired design removed | no | 0.197 |
| Supervised nearest centroid, leave-one-well-out | yes | 0.515 |

## Per compound

| Compound | Target | Wells | Top-1 | Top-2 | Rat | Human |
|---|---|---|---|---|---|---|
| CNQX | `g_ampa` | 11 | 0/11 | 0/11 | 0/7 | 0/4 |
| D-AP5 | `g_nmda` | 11 | 1/11 | 1/11 | 0/7 | 1/4 |
| GABA | `g_gaba` | 11 | 0/11 | 0/11 | 0/7 | 0/4 |
| Gabazine | `g_gaba` | 11 | 4/11 | 6/11 | 0/7 | 4/4 |
| Kainic acid | `g_ampa` | 11 | 0/11 | 0/11 | 0/7 | 0/4 |
| TTX | `g_na` | 11 | 1/11 | 6/11 | 1/7 | 0/4 |

## Per species

| Species | Wells | Top-1 accuracy |
|---|---|---|
| hPSC | 24 | 0.208 |
| rat | 42 | 0.024 |

Windows scored: 231. Excluded as unreadable before any model ran: 0.

## Confusion

Rows are the mechanism the compound acts on, columns the one the twin named.

| acts on \ named | g_na | g_kdr | g_ahp | g_ampa | g_nmda | g_gaba | tau_d | u_rel | i_drive |
|---|---|---|---|---|---|---|---|---|---|
| `g_na` | 1 | . | . | 1 | 1 | . | 1 | 1 | 6 |
| `g_ampa` | 10 | . | 3 | . | 4 | 1 | . | 1 | 3 |
| `g_nmda` | . | . | 5 | . | 1 | . | . | 1 | 4 |
| `g_gaba` | 1 | . | 3 | 4 | 1 | 4 | 2 | . | 7 |

## Calibration on held-out simulations

1500 records the model never saw.

| Parameter | 50% | 80% | 90% | rank KS | recovery r (active) |
|---|---|---|---|---|---|
| `g_na` | 0.45 | 0.77 | 0.88 | 0.034 | 0.558 (216) |
| `g_kdr` | 0.56 | 0.75 | 0.83 | 0.038 | 0.311 (219) |
| `g_ahp` | 0.45 | 0.75 | 0.85 | 0.061 | 0.490 (235) |
| `g_ampa` | 0.49 | 0.76 | 0.85 | 0.037 | 0.285 (226) |
| `g_nmda` | 0.44 | 0.74 | 0.85 | 0.052 | 0.618 (234) |
| `g_gaba` | 0.49 | 0.77 | 0.85 | 0.029 | 0.311 (215) |
| `tau_d` | 0.45 | 0.73 | 0.83 | 0.039 | 0.387 (252) |
| `u_rel` | 0.45 | 0.75 | 0.85 | 0.040 | 0.537 (249) |
| `i_drive` | 0.46 | 0.74 | 0.85 | 0.039 | 0.466 (219) |

Nominal coverage is 0.50, 0.80 and 0.90. The rank statistic is the Kolmogorov-Smirnov distance from uniform; 0 is exact.

## Presence probability

| Mechanism | AUROC | Expected calibration error |
|---|---|---|
| `g_na` | 0.848 | 0.012 |
| `g_kdr` | 0.550 | 0.005 |
| `g_ahp` | 0.830 | 0.013 |
| `g_ampa` | 0.712 | 0.013 |
| `g_nmda` | 0.886 | 0.012 |
| `g_gaba` | 0.694 | 0.014 |
| `tau_d` | 0.763 | 0.014 |
| `u_rel` | 0.810 | 0.011 |
| `i_drive` | 0.769 | 0.012 |

## The guard

Threshold 4.25, the 95% percentile of the discrepancy over 100 held-out records, fixed before any recording was scored.

| Case | Must | n | Fired | Median discrepancy | Outcome |
|---|---|---|---|---|---|
| held-out simulations | pass (at most 0.10) | 40 | 0.12 | 0.6 | **not met** |
| changed receptor kinetics | fire (at least 0.80) | 26 | 0.12 | 1.0 | **not met** |
| real recordings, structure destroyed | fire (at least 0.80) | 40 | 0.17 | 1.7 | **not met** |
| real recordings | reported (-) | 40 | 0.05 | 1.3 | - |

## By mechanism class

Secondary and not pre-registered. Classes are excitatory transmission, inhibitory transmission, intrinsic excitability, and adaptation with short-term plasticity.

| Quantity | Value |
|---|---|
| Class accuracy | **0.227** (66 wells) |
| Chance | 0.250 |

| Compound | Class correct |
|---|---|
| CNQX | 1/11 |
| D-AP5 | 0/11 |
| GABA | 0/11 |
| Gabazine | 4/11 |
| Kainic acid | 3/11 |
| TTX | 7/11 |

## Restricted to wells the twin reproduces

Secondary and not pre-registered. It is here so that a refusal cannot be read as an error.

| Quantity | Value |
|---|---|
| Wells passing the predictive check | 70 of 77 (91%) |
| Top-1 accuracy on those wells | 0.082 (61 scored) |

## Figures

`results/figures/` holds the confusion matrix, the per-compound accuracy, interval coverage, presence reliability, the guard, the recovered effect sizes, the parameter sweep and a measured raster beside the twin's.
