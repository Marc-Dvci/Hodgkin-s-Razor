# Results

Produced by `scripts/evaluate.py`. Pre-registration hash `c40708bb19668164`, verified before the run.

Bank: 27314 simulated pairs, 1638 held out for validation and 1638 for calibration. Windows of 60 s.

## Primary metric

| Quantity | Value | Pre-registered | Outcome |
|---|---|---|---|
| Top-1 mechanism accuracy | **0.303** (66 wells) | at least 0.50 | **not met** |
| Chance | 0.100 | | |
| Control false-mechanism rate | **0.000** (11 wells) | at most 0.20 | met |
| Top-2 accuracy | 0.364 | | |
| Direction agreement | 0.650 (20 wells) | | |

## Against the baselines

| Method | Sees compound labels | Top-1 accuracy |
|---|---|---|
| Hodgkin's Razor | no | **0.303** |
| Same twin, paired design removed | no | 0.091 |
| Supervised nearest centroid, leave-one-well-out | yes | 0.515 |

## Per compound

| Compound | Target | Wells | Top-1 | Top-2 | Rat | Human |
|---|---|---|---|---|---|---|
| CNQX | `g_ampa` | 11 | 3/11 | 6/11 | 1/7 | 2/4 |
| D-AP5 | `g_nmda` | 11 | 2/11 | 2/11 | 0/7 | 2/4 |
| GABA | `g_gaba` | 11 | 0/11 | 0/11 | 0/7 | 0/4 |
| Gabazine | `g_gaba` | 11 | 4/11 | 4/11 | 4/7 | 0/4 |
| Kainic acid | `g_ampa` | 11 | 7/11 | 7/11 | 3/7 | 4/4 |
| TTX | `g_na` | 11 | 4/11 | 5/11 | 4/7 | 0/4 |

## Per species

| Species | Wells | Top-1 accuracy |
|---|---|---|
| hPSC | 24 | 0.333 |
| rat | 42 | 0.286 |

Windows scored: 231. Excluded as unreadable before any model ran: 0.

## Confusion

Rows are the mechanism the compound acts on, columns the one the twin named.

| acts on \ named | g_na | g_kdr | g_ahp | g_ampa | g_nmda | g_gaba | g_tonic_inh | tau_d | u_rel | i_drive |
|---|---|---|---|---|---|---|---|---|---|---|
| `g_na` | 4 | . | . | 1 | 1 | . | . | 3 | . | 2 |
| `g_ampa` | 4 | 1 | 1 | 10 | 3 | 1 | . | 1 | . | 1 |
| `g_nmda` | . | . | 1 | 1 | 2 | . | . | 1 | . | 6 |
| `g_gaba` | 5 | 1 | 1 | 3 | . | 4 | . | 5 | 1 | 2 |

## Calibration on held-out simulations

1200 records the model never saw.

| Parameter | 50% | 80% | 90% | rank KS | recovery r (active) |
|---|---|---|---|---|---|
| `g_na` | 0.46 | 0.75 | 0.85 | 0.046 | 0.435 (184) |
| `g_kdr` | 0.54 | 0.74 | 0.82 | 0.046 | 0.059 (181) |
| `g_ahp` | 0.51 | 0.76 | 0.83 | 0.039 | 0.512 (165) |
| `g_ampa` | 0.49 | 0.76 | 0.86 | 0.047 | 0.341 (180) |
| `g_nmda` | 0.49 | 0.78 | 0.87 | 0.027 | 0.369 (176) |
| `g_gaba` | 0.52 | 0.77 | 0.85 | 0.040 | 0.238 (186) |
| `g_tonic_inh` | 0.43 | 0.73 | 0.83 | 0.049 | 0.651 (146) |
| `tau_d` | 0.51 | 0.75 | 0.82 | 0.045 | 0.454 (184) |
| `u_rel` | 0.50 | 0.75 | 0.83 | 0.036 | 0.557 (181) |
| `i_drive` | 0.49 | 0.75 | 0.84 | 0.043 | 0.299 (165) |

Nominal coverage is 0.50, 0.80 and 0.90. The rank statistic is the Kolmogorov-Smirnov distance from uniform; 0 is exact.

## Presence probability

| Mechanism | AUROC | Expected calibration error |
|---|---|---|
| `g_na` | 0.742 | 0.012 |
| `g_kdr` | 0.499 | 0.036 |
| `g_ahp` | 0.633 | 0.036 |
| `g_ampa` | 0.771 | 0.018 |
| `g_nmda` | 0.757 | 0.011 |
| `g_gaba` | 0.725 | 0.006 |
| `g_tonic_inh` | 0.915 | 0.025 |
| `tau_d` | 0.590 | 0.019 |
| `u_rel` | 0.638 | 0.015 |
| `i_drive` | 0.688 | 0.010 |

## The guard

Threshold 3.05, the 95% percentile of the discrepancy over 100 held-out records, fixed before any recording was scored.

| Case | Must | n | Fired | Median discrepancy | Outcome |
|---|---|---|---|---|---|
| held-out simulations | pass (at most 0.10) | 40 | 0.07 | 1.0 | met |
| changed receptor kinetics | fire (at least 0.80) | 8 | 0.50 | 2.5 | **not met** |
| real recordings, structure destroyed | fire (at least 0.80) | 40 | 0.30 | 3.0 | **not met** |
| real recordings | reported (-) | 40 | 0.25 | 2.0 | - |

## The same metric on simulations

Held-out simulated experiments, where the answer is known. The second row is the regime a saturating concentration produces, which is where the recorded compounds sit.

| Case | n | Top-1 | Top-2 | Class |
|---|---|---|---|---|
| every single-mechanism pair | 716 | 0.369 | 0.514 | 0.529 |
| strong effect and a large observable change | 85 | 0.576 | 0.718 | 0.635 |
| chance | | 0.100 | | 0.250 |

## By mechanism class

Secondary and not pre-registered. Classes are excitatory transmission, inhibitory transmission, intrinsic excitability, and adaptation with short-term plasticity.

| Quantity | Value |
|---|---|
| Class accuracy | **0.394** (66 wells) |
| Chance | 0.250 |

| Compound | Class correct |
|---|---|
| CNQX | 6/11 |
| D-AP5 | 3/11 |
| GABA | 0/11 |
| Gabazine | 4/11 |
| Kainic acid | 7/11 |
| TTX | 6/11 |

## Restricted to wells the twin reproduces

Secondary and not pre-registered. It is here so that a refusal cannot be read as an error.

| Quantity | Value |
|---|---|
| Wells passing the predictive check | 35 of 77 (45%) |
| Top-1 accuracy on those wells | 0.429 (28 scored) |

## Figures

`results/figures/` holds the confusion matrix, the per-compound accuracy, interval coverage, presence reliability, the guard, the recovered effect sizes, the parameter sweep and a measured raster beside the twin's.
