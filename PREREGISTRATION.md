# Pre-registration

Written before any trained model was scored against a recorded experiment. The
file is hashed into `PREREGISTRATION.sha256`; every number reported later comes
from `scripts/evaluate.py`, which reads this file's answer key and nothing else.

Author: Marc Donovici. Date: 26 September 2026.

## 1. What is being claimed

Given two recordings of the same well, before and after a compound was washed
on, the twin names which molecular mechanism the compound moved. The twin is
trained only on simulations. No recorded experiment, and no compound label,
enters training, model selection, feature selection, thresholds, or any
hyperparameter.

## 2. Answer key

Fixed here, from the published pharmacology of each compound. `parameter` is
the model parameter scored as correct; `direction` is the expected sign.

| Compound | Concentration | Parameter | Direction | Note |
|---|---|---|---|---|
| CNQX | 50 uM | `g_ampa` | down | AMPA/kainate antagonist |
| D-AP5 | 50 uM | `g_nmda` | down | NMDA antagonist |
| GABA | 10 uM | `g_gaba` | up | GABA-A agonist |
| Gabazine | 30 uM | `g_gaba` | down | GABA-A antagonist |
| Kainic acid | 5 uM | `g_ampa` | up | AMPA/kainate agonist |
| TTX | 100 nM | `g_na` | down | Na channel blocker |
| Control | vehicle | none | none | no mechanism may be called |

Kainic acid is scored on `g_ampa` whatever the sign of the recovered shift,
because at 5 uM a rate decrease through receptor desensitisation and
depolarisation block is the expected outcome of raising that conductance. Its
direction is reported separately and is not part of the primary metric.

## 3. Datasets

Tampere comparative MEA dataset, CC BY 4.0
(gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset). Two pharmacology
plates: rat cortical at DIV 22, human pluripotent-stem-cell derived at DIV 29.
Electrodes the authors marked noisy are dropped. Windows are three consecutive
60 s blocks starting 300 s into each recording, the same window on both sides
of a pair.

## 4. Primary metric

**Top-1 mechanism accuracy** over the five compounds present on both plates
(CNQX, D-AP5, GABA, Gabazine, kainic acid), scored per well. A well is correct
when the parameter with the highest presence probability, averaged over that
well's windows, is the parameter in the answer key.

The sample size is the number of wells, not the number of windows: 7 wells per
compound on the rat plate and 4 on the human plate. Windows are replicates of
one well and are not independent.

**Success is declared** if top-1 accuracy over all scored wells is at least
0.50, against a chance rate of 1/9 = 0.111 over the nine shiftable parameters.

## 5. Secondary metrics

1. Top-2 accuracy, same unit.
2. Direction agreement on wells whose top-1 is correct.
3. **False-mechanism rate on controls**: the fraction of control wells where
   any mechanism exceeds presence probability 0.5. Pre-registered ceiling: 0.20.
4. Per-compound accuracy, reported for every compound including failures.
5. Species split: rat and human reported separately, never pooled only.
6. Confusion matrix over all nine parameters, so a confusable pair is visible
   rather than hidden inside an aggregate.

## 6. Calibration and recovery on simulations

Reported on bank records held out from training and from calibration:

1. Coverage of the central 50, 80 and 90 percent credible intervals for each
   parameter and each shift.
2. Rank statistics of the true value within the posterior, per parameter,
   which should be uniform.
3. Presence-probability reliability, per mechanism, after the stored
   calibration map.
4. Parameter recovery, as the correlation between the true and the posterior
   median shift on active parameters.

## 7. The guard

A recording whose posterior cannot reproduce it is reported as outside the
model and no mechanism is named. The threshold is the 95th percentile of the
discrepancy on held-out bank records, fixed before any recorded experiment is
scored.

The guard is tested on cases it must catch:

1. Simulations from a variant simulator whose receptor kinetics are changed
   (AMPA decay 2 ms to 20 ms, GABA decay 6 ms to 40 ms). No parameter of the
   twin can produce this, so the guard must fire.
2. Real recordings whose spike times have been circularly shifted per
   electrode, which destroys network structure while preserving every
   per-electrode rate. The guard must fire.

And on cases it must pass: held-out bank records, and the real Tampere pairs.
Pre-registered expectation: fires on at least 0.80 of case 1 and case 2, and on
no more than 0.10 of held-out bank records.

## 8. Baselines

1. **Unpaired twin.** The same simulator, features and bank, with the paired
   sparse-shift design removed: parameters are inferred separately for the two
   recordings and the shift is the difference of posterior medians. This
   isolates the paired design.
2. **Supervised feature classifier.** A nearest-centroid classifier on the
   feature-difference vector, trained on the real compound labels under
   leave-one-well-out cross-validation. This baseline is given the labels that
   the twin never sees, and is reported whether or not it wins.
3. **Prior art**, best effort: the published estimator of Doorn et al. (2025)
   applied to the same recordings through their own feature code.

## 9. What would count as a failure

- Top-1 accuracy below 0.50.
- Control false-mechanism rate above 0.20.
- Guard firing on more than 0.10 of held-out bank records.
- Interval coverage off nominal by more than 0.15 for a parameter whose shift
  is reported as credible.

Every one of these is reported whatever it comes out at, including the
per-compound table.

## 10. Analysis decisions fixed in advance

- 4000 posterior samples per recording; 48 predictive draws per check.
- Presence threshold for calling a mechanism: 0.5 after calibration.
- Credible intervals at 90 percent.
- Wells whose baseline window has fewer than 50 detected events on fewer than
  3 electrodes are excluded as unreadable, before any model is run, and the
  count of exclusions is reported.
- No compound is dropped after seeing a result.
