# Pre-registration, version 2

Written and hashed before any version 2 model output was computed on the Doorn
et al. Dynasore recordings, and before any statistic was computed on the Mateus
et al. microchannel chips. `scripts/freeze_v2.py` fills in the model digests and
the chip twin's simulated predictions, writes this file and its SHA-256, and
nothing is scored until both exist. `hodgkins_razor/doorn.py` refuses to return a
treated Dynasore window, and `scripts/evaluate_v2.py` and `scripts/mateus_check.py`
refuse to run, unless this file matches its recorded hash.

Author: Marc Donovici. Date: {DATE}.

## 1. Why there is a second pre-registration

The first (`PREREGISTRATION.md`, hash `c40708bb19668164`) was scored on the
Tampere plates. The pre-registered model reached a top-1 of 0.091, which is
chance. Three label-free corrections followed, and the recorded set was scored
four times in all. The Tampere plates are therefore no longer a blind test.
From here on they are the **development set**: they are reported, and no claim
of blind performance rests on them.

Version 2 is tested on data the project has not scored:

* **A.** The Dynasore recordings of Doorn et al. (2024), from another
  laboratory, another cell source (human iPSC-derived Ngn2 neurons with rat
  astrocytes), another recording system (MCS 24-well, 12 electrodes per well)
  and another compound.
* **B.** The twin's prediction of which chip readout resolves directionality,
  on the microchannel chips of Mateus et al. (2024).

## 2. What changed from version 1, and on what evidence

Every change was chosen on simulations, on the published pharmacology, or on
untreated baseline recordings. None used a compound label or a treated
recording from a dataset scored here.

1. A defect in the spike-time tiling coefficient: the coincidence test looked
   only at the next spike of the other train. Fixed and checked against a
   brute-force reference.
2. Label noise in the shift prior: 28 percent of active tonic-inhibition labels
   and 18 percent of active AMPA labels described shifts that clipping at the
   prior bounds had removed. Directions are now drawn where the baseline leaves
   room, and a clipped shift is labelled inactive.
3. Linear parameters get a log-uniform active magnitude, so a partial agonist
   effect has support.
4. Validation and calibration are split by simulated culture, not by record.
5. The flow is conditioned on the active mechanism set, so the effect of each
   mechanism is reported given that it moved.
6. Five presence heads are averaged.
7. Recording systems are explicit views: electrode layout and detection dead
   time (Axion 48-well: 16 electrodes, 2 ms; MCS 24-well: 12 electrodes, 0.3 ms).
8. Each system's domain is the range its own untreated baselines span, widened,
   plus the version 1 criterion for a living network-driven culture. The Doorn
   domain was read from the 50 Doorn baseline windows, which are untreated.
9. A second guard test, typicality, alongside the predictive check.
10. Asynchronous release, from the model of Doorn et al., because the
    published model of the blind-test cultures carries it (a culture
    property, not shiftable).
11. The tonic GABA-A conductance is log-scaled over 0.001 to 0.5 nS. A graded
    scan on simulated cultures showed firing halving near +0.04 nS and silence
    beyond +0.1 nS, so the version 1 range of 0 to 6 nS could only represent a
    bath agonist that silences everything.

Selection evidence, all on held-out simulated cultures: top-1 on single-
mechanism pairs rose from 0.369 (version 1) to the values in
`results/v2/sim_grid16.json` and `results/v2/sim_grid12.json`; five presence
heads and longer training beat one head on a pilot bank.

## 3. What was known about the blind data before this file was hashed

* Doorn et al. publish summary statistics for these wells (their Table S2):
  after Dynasore, network burst rate rose, burst duration and the percentage of
  spikes inside bursts fell. They model Dynasore as an increase of the
  short-term-depression parameter U.
* The per-minute firing rate of each Dynasore file was plotted once, to find the
  application time, before their Table S2 windows were located and used instead.
* The Doorn baseline windows were read to set the domain (item 8).
* For the chips: the file layout (electrode rows 7 to 11 are the channels) and
  the per-electrode spike counts of one Rams recording. The published result of
  Mateus et al.: Rams chips send about 94 percent of propagating events forward,
  Arrows are significantly forward-biased, Tesla designs do not differ from
  straight controls.

## 4. The frozen twin

The models, the bank and the operating points are listed in the JSON block
(section 9), with a digest of every model directory. `evaluate_v2.py` checks
them before it runs.

## 5. A. Blind test: Doorn et al. Dynasore wells

**Wells and windows.** The ten wells of their Table S2, with their windows:
FB2 B6, C6, D6 (baseline 0 to 300 s, Dynasore 1550 to 1850 s); FB2t A2, A3, B1,
C1 (0 to 300 s, 950 to 1250 s); FB3t A3, C3, D1 (0 to 300 s, 1520 to 1820 s).
Each side is cut into five 60 s windows, matched by position. A window is
unreadable if its baseline has fewer than 50 events or fewer than 3 active
electrodes. The unit is the well; presence is averaged over its windows.

**Answer key.** Dynasore inhibits dynamin and slows vesicle recycling, which
strengthens short-term depression. Doorn et al. model it as an increase of U.
Both `u_rel` (release fraction, their U) and `tau_d` (vesicle recovery time) are
accepted, direction up.

**Primary metric.** Top-1 accuracy over the ten wells. Chance is 2 of 10
mechanisms, 0.20.

**Success** is declared at 5 or more of 10 wells (P = 0.033 under chance).

**Secondary.** Mechanism-class accuracy (adaptation and short-term plasticity,
chance 0.30); direction agreement; the fraction of wells with a call above
0.5; the effect size given active; the guard's verdict per well; the same
metric for the unpaired twin and for the published estimator of Doorn et al.,
whose accepted parameters are U and tau_D.

## 6. B. Chip readout prediction: Mateus et al. chips

**Prediction** (from simulation, `results/chip_study.json`, values in section
9): the share of propagating events running a channel's dominant way, read
from electrodes inside the channels, separates diode channels from straight
channels. The unsigned asymmetry of the two chambers' rate cross-correlation,
read from electrodes under the chambers, does not. The simulated values are
computed on chips whose two chambers are both self-active, because Mateus et
al. seed both chambers with the same neurons at the same density (their
methods); on all simulated chips the channel statistic is weaker, because a
quiet target makes forward events dominate whatever the channel does. Both
values are reported. A statistic is predicted to separate when its simulated
AUROC is at least 0.70; the recorded bar is then 0.75, and otherwise the
recorded AUROC must stay below 0.70.

**Recorded test.** Every recording of the dataset. Diode designs: Rams and
Arrows (the designs the authors found forward-biased). Straight designs: the
straight-channel controls. Tesla and Tesla v2 are reported, not scored.
Recordings with fewer than the listed number of propagating events are excluded
from the channel statistic, and the count is reported.

**Scored** as the AUROC of each statistic for diode against straight, with a
95 percent interval that resamples chips. The prediction holds for the channel
statistic if its AUROC is at least the listed threshold, and for the chamber
statistic if its AUROC is below the listed threshold.

## 7. C. Development set: Tampere, fifth scoring

Scored with the version 1 answer key unchanged, and with a version 2 key that
also accepts `g_tonic_inh` for bath GABA, because version 2 represents a bath
agonist through that conductance (section 2 of the report explains why). Both
are reported. Also reported: top-2, class accuracy, control false-mechanism rate
(ceiling 0.20), detection AUROC (the largest presence probability, treated
against control), the unpaired twin, the supervised baseline under
leave-one-well-out, and the published estimator of Doorn et al. No success is
claimed on this set.

## 8. D. Transfer, and E. simulations and the guard

**Transfer.** On the Tampere wells: a nearest-centroid classifier over compounds
is fitted on one species and scored on the other, in two representations (the
raw feature change, and the twin's presence logits with its effects given
active), both directions. Reported whatever it shows.

**Simulations.** On held-out cultures of each system: presence AUROC and
calibration error, top-1 on single-mechanism pairs and on the saturating
regime, interval coverage, and recovery of the effect given active.

**Guard.** Typicality threshold: the 97.5th percentile on validation cultures.
Predictive-check threshold: the 97.5th percentile over held-out records. A pair
is outside the model when either fires. Must pass: held-out simulations, at
most 0.10 firing. Must fire: simulations with receptor kinetics the twin cannot
represent (AMPA decay 2 to 20 ms, GABA decay 6 to 40 ms), and recorded pairs
with each electrode circularly shifted, at least 0.80 each. The same bars as
version 1.

## 9. Machine-readable specification

```json
{SPEC}
```
