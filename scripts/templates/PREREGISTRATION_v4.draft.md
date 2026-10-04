# Pre-registration, version 4: the chip twin on a four-compartment chip

Written and hashed before any spike of Lassers et al. (2023) was counted.
`scripts/freeze_v4.py` fills in the twin's predictions from
`results/v4/brewer_prediction.json` and the digests of the data files, writes
this file and its SHA-256, and nothing is scored until both exist.
`hodgkins_razor/brewer.py` refuses to return a spike time, and
`scripts/evaluate_v4.py` refuses to run, unless this file matches its recorded
hash.

Author: Marc Donovici. Date: {{DATE}}.

## 1. Why a fourth test

The chip twin says, from simulation alone, which readout of a
compartmentalised chip resolves which property of the device
(`scripts/chip_study.py`). Its central claim is that electrodes under the
compartments cannot tell which way axons run between them, and electrodes in
the channels can. The second pre-registration tested that on recorded chips
with designed diodes (Mateus et al. 2024): the compartment statistic did not
separate the designs, as predicted, and the channel statistic did not reach its
bar (AUROC 0.62, 17 chips, inconclusive at power 0.48).

Lassers, Vakilna, Tang and Brewer (2023) recorded the other kind of device:
hippocampal subregions (EC, DG, CA3, CA1) grown in four compartments of one
PDMS device on a 120-electrode array, joined in a loop by tunnels 400 um long,
with 19 electrodes under each compartment and an electrode pair spanning five
tunnels of each of the four boundaries. The pair's delay gives every axon's
direction. It is the one public recording of a neural chip in which both
readouts the twin compares sit on the same device. Nothing in it was designed
to be directional, so it tests the twin's account of how compartments and
axons relate, not a diode.

## 2. What was known about the data before this file was hashed

Read: the dataset's README and `set_parameters.m`; the column names of the
tables in `*SortedAxons.mat` and `*WellSpikes.mat`; the labels in every
culture (each tunnel's boundary and electrode pair, each well electrode's
subregion and name), which show 9 unstimulated cultures with 20 monitored
tunnels and 76 well electrodes each, identical in every culture up to the
array's orientation. To print those labels the files were loaded whole into
memory; no spike train, count, rate, direction, conduction time or axon count
was computed or printed. The file checksums were compared with Zenodo's.

Not read: anything computed from a spike train. The secondary conditions
(HFS 5, HFS 40) were not opened beyond their variable names.

The twin was not fitted to this dataset. Its cultures come from the domain
fitted on rat cortical baselines (Tampere), not hippocampal ones.

## 3. Units and statistics

All statistics are computed on the full 300 s recording, by
`scripts/evaluate_v4.py`, with the functions `scripts/brewer_prediction.py`
used on simulated chips (`chip.rate_xcorr`: population rates in 5 ms bins,
cross-correlation over +/- 300 ms).

* **Boundary unit**: one culture's boundary (EC-DG, DG-CA3, CA3-CA1,
  CA1-EC). Upstream is the first region along EC > DG > CA3 > CA1 > EC. Each
  compartment's spike times are its 19 electrodes pooled. An axon is one spike
  train of a tunnel table cell, by direction (feed-forward or feedback), read
  at the tunnel's upstream electrode, as the dataset's authors do. An axon with
  no spike at that electrode is not counted.
  * Included if both compartments have at least {{MIN_WELL}} spikes, at least
    {{MIN_AXONS}} axons are counted, and those axons carry at least
    {{MIN_AXON_SPIKES}} spikes.
  * `asym`: the asymmetry of the cross-correlation of the two compartments'
    population rates, positive when upstream leads.
  * `traffic_ff`: feed-forward share of the counted axons' spikes.
  * `axons_ff`: feed-forward share of the counted axons.
* **Home unit**: one boundary's axons of one direction, pooled, if they carry
  at least {{MIN_AXON_SPIKES}} spikes and both compartments pass the well rule.
  * `home`: peak correlation of the pooled axon spikes with the population
    rate of the compartment the axons grow from (upstream for feed-forward,
    downstream for feedback), minus the same with the compartment they grow
    into.

## 4. The twin's predictions

The twin simulated {{N_SIM}} boundaries as two-chamber chips: both chambers
self-active cultures (target autonomy 0.7 to 1.0), axons in the channels drawn
from the chip prior in either direction, five of them monitored, 300 s each.
Which chamber is upstream is random, so feed-forward shares span 0 to 1.
Predictive intervals come from resampling simulated units at the number the
real data yield; the table below quotes them at the largest possible n, and
`scripts/evaluate_v4.py` reads the interval at the scored n from the
prediction file (sha256 `{{PRED_SHA_SHORT}}`).

| | twin, whole bank | 90% predictive interval at n = {{N_BND}} boundaries / {{N_HOME}} home units |
|---|---|---|
| Spearman(`asym`, `traffic_ff`) | {{RHO_TRAFFIC}} | {{PI_TRAFFIC}} |
| Spearman(`asym`, `axons_ff`) | {{RHO_AXONS}} | {{PI_AXONS}} |
| share of home units with `home` > 0 | {{HOME_SHARE}} | {{PI_HOME}} |

In words: the twin expects that **an axon's spikes follow the compartment it
grows from more closely than the one it grows into**, in about {{HOME_PCT}} of
cases, and that **the compartments' own timing says little about which way the
axons between them run**.

## 5. Outcomes, fixed now

Primary, unstimulated recordings (9 cultures):

* **P1.** The share of home units with `home` > 0 is at or above the lower
  bound of the twin's 90% predictive interval at the scored n.
* **P2.** Spearman(`asym`, `traffic_ff`) over boundary units lies inside the
  twin's 90% predictive interval at the scored n.

Each is reported met or not met on its own. Reported with them: a sign test of
P1's share against 0.5, a permutation p for P2, and 95% intervals resampling
cultures (units within a culture are not independent).

Secondary: Spearman(`asym`, `axons_ff`) against its interval; the same three
statistics on the HFS 5 and HFS 40 recordings (6 cultures each), against the
same intervals. Anything else is labelled exploratory.

How each result would be read, written before scoring:

* P1 near 0.5: in this device the compartments fire so tightly together that
  an axon's spikes do not tell its home from its target. The twin's coupling
  between compartments is then too loose for this chip.
* P2 above the interval: compartment electrodes do carry the direction of
  axonal traffic on this device, and the chip planner's advice to place
  electrodes in the channels is overstated.
* P2 below the interval: compartment timing runs against axonal traffic, which
  the twin has no mechanism for.

Known mismatches, none tuned: the device is a loop of four compartments, read
here one boundary at a time as two-chamber chips; tunnels are 400 um, the
twin's channels 500 um; the twin's cultures are cortical, these hippocampal;
the twin applies a 2 ms detection dead time, the dataset's own spike detection
differs.

## 6. Machine-readable specification

```json
{{SPEC}}
```
