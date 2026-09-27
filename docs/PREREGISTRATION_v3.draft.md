# Pre-registration, version 3

Written and hashed before any version 3 model output was computed on a scored
recording of Charlesworth et al. (2015), and before any statistic was computed
on the second sister of any pair recorded after the drug was added.
`scripts/freeze_v3.py` fills in the model digests and the simulation evidence,
writes this file and its SHA-256, and nothing is scored until both exist.
`hodgkins_razor/charlesworth.py` refuses to return a second-sister recording
made at 8 days in vitro or later, and `scripts/evaluate_v3.py` refuses to run,
unless this file matches its recorded hash.

Author: Marc Donovici. Date: {DATE}.

## 1. Why there is a third pre-registration

The second pre-registration (`PREREGISTRATION_v2.md`, hash `5dc7a927085f8e46`)
was scored on the Dynasore wells of Doorn et al. The paired twin named the
accepted mechanism in 2 of 10 wells, against a bar of 5. It flagged 9 of 10
treated wells as changed, and 0 of 10 pre-drug pairs from the same wells, and
placed 8 of 10 in the right mechanism class; those were secondary outcomes.

An audit after scoring found that the unpaired comparator's 8 of 10 was not
evidence: on pre-drug pairs from the same wells, where nothing was applied, it
named the accepted mechanisms in 6 of 10. A ranking can land on an answer key
by preference alone. So the chance level of a method is its own hit rate when
nothing was applied, and a test needs a matched null inside its primary
metric. This pre-registration is built around one.

The Dynasore and Tampere recordings are development data from here on. One
development fact bears on this test: on Tampere, acute D-AP5 in rat cultures
gave `g_nmda` as the top-1 mechanism in 1 of 7 wells. That is why a stop rule
(`docs/STOP_RULE_v3.md`, committed before the version 3 twin was trained)
requires the twin to pass this test on its own simulations before this file
may be hashed.

## 2. The test

Charlesworth et al. plated each preparation of mouse hippocampal neurons on
two sister arrays (A and B) and recorded both twice a week. After the recording
at 7 days in vitro, 50 uM APV, a competitive NMDA receptor antagonist, was
added to the medium of the B array of some preparations and kept there. Other
preparations kept both sisters untreated. A pair is one preparation's two
sisters recorded the same day:

* **treated**: the B sister carries APV;
* **null**: neither sister was treated, the preparation never received APV,
  and both sisters share a genotype;
* **pre-drug**: any preparation at 6 or 7 days, before APV was added;
* **A–C**: {N_FOUR} preparations were plated on four arrays and never treated.
  Their A and C arrays are sisters that are never locked. These pairs were
  read before the freeze (section 4), so these preparations are **excluded**
  from the scored null group.

The twin reads the first sister as the baseline and the second as the treated
recording, exactly as it reads one well before and after a compound. For a
null pair the correct answer is that nothing moved.

**Ages.** Primary: 10 to 14 days in vitro. Those are the ages at which treated
pairs were recorded (10, 11, 13 and 14 days), so both groups are read over the
same span. Charlesworth et al. report that the effects of chronic APV were
strongest early and faded as the cultures matured. Late: 15 days and after,
secondary.

**Readout.** Each recording is read as its four 4 x 4 quadrants (twelve
electrodes each, the grid12 layout). Each quadrant is cut into six 60 s
windows from 60 to 420 s, matched by position between sisters. A window is
scored when its first sister has at least 50 events on at least 3 electrodes.
The unit is the **preparation**: presence probabilities are averaged over
every scored window of every primary-age pair of that preparation. Treated and
null preparations are disjoint.

**Groups.** Treated preparations are wild type or Gria1 knockout only. The
null group of the whole dataset also holds six other knockouts, three of them
NMDA-receptor scaffolds (PSD-95, PSD-93, SAP102). The sister design cancels a
genotype's mean effect on a difference. It does not cancel its effect on where
a culture sits in parameter space, which governs how readable `g_nmda` is. The
primary therefore compares **like with like**: {N_TREATED} treated
preparations against the {N_NULL_MATCHED} null preparations of the same two
genotypes. The pooled null group ({N_NULL_POOLED} preparations) is secondary.
Preparations per genotype: `{GROUPS}`.

**Answer key.** APV blocks NMDA receptors: `g_nmda`, direction down.

**Primary outcome.** The area under the ROC curve of the presence probability
of `g_nmda`, treated preparations against genotype-matched null preparations.
**Success:** at least 0.70, and the lower end of its 95 percent interval above
0.50. The interval comes from 4,000 bootstrap resamples of preparations within
each group.

**Co-primary outcome, the metric of versions 1 and 2.** The share of treated
preparations whose top-1 mechanism is `g_nmda`, against the same share among
genotype-matched null preparations. A top-1 hit counts whatever the sign of
the effect; the sign is reported separately. **Success:** higher in treated
preparations, with a one-sided Fisher exact p below 0.05.

**Secondary outcomes.**
* The direction of the `g_nmda` effect among treated top-1 hits.
* The share of preparations with `g_nmda` called above 0.5 in each group.
* Detection: the largest presence probability, treated against null.
* The primary and co-primary against the pooled null group.
* The same contrasts at late ages.
* The contrast within each genotype: treated wild type against null wild
  type, and treated Gria1 against null Gria1.
* Whether the `g_nmda` reading fades with age in treated preparations (paired
  primary against late, one-sided Wilcoxon). This is the canalization that
  Charlesworth et al. describe. It is the secondary outcome of most
  scientific interest.
* The guard's verdicts.
* The same contrasts for the unpaired twin and for the published estimator of
  Doorn et al. That estimator is scored on the first 60 s window of quadrants
  0 and 3 of every primary-age pair. Its score is the size of its standardised
  `g_NMDA` shift in the answer's direction, so it is direction-aware.

Every outcome is reported whatever it shows.

**Exploratory, no bar: compensation.** Days of NMDA blockade can recruit
homeostatic compensation, so the twin may read `g_nmda` down together with a
second mechanism. The presence AUROC of every mechanism (treated against
matched null) is reported. Named in advance as the readings that would count
as compensation rather than as errors: `g_ampa` up, `i_drive` up, `g_gaba`
down and `g_tonic_inh` down. The primary is a presence probability, so a
co-shift does not count against it. The top-1 co-primary can lose to one.

## 3. What changed from version 2, and on what evidence

No compound label and no second-sister recording made after the drug was added
entered any of these choices.

1. **A recording-system view** for the Multi Channel Systems 60-electrode
   array, read as quadrants on the grid12 layout, with a dead time of 1.08 ms:
   the shortest interval on any electrode of the first-sister recordings.
2. **The sister design.** Sisters are the same preparation, not the same
   network. The bank simulates the second sister with its own wiring and
   electrode pickup. It also nudges every culture parameter by a drift that is
   never labelled as a mechanism. Its size, {DRIFT} of each parameter's
   baseline range, was set by `scripts/calibrate_drift.py` so that simulated
   sister differences match those of sister pairs recorded before any drug
   (6 and 7 days). It was then checked at the scored ages on the A–C pairs at
   10 to 14 days (369 windows inside the domain): the same value, 0.12, is
   chosen there (`models/drift_v3_ac.json`).
3. **The domain** is fitted by the version 2 rule from the first sisters of
   the scored pairs at 9 days and later, which never carry a drug.

The simulator, the parameter table, the shift prior, the features, the
conditional flow with five presence heads, the guard and its thresholds are
those of version 2.

## 4. What was known about the blind data before this file was hashed

* The metadata sheet: file names, ages, genotypes and which recordings carry
  APV. It defines the pairs and the groups.
* The published description: 50 uM APV added after the 7-day recording and
  kept in the medium; chronic APV disturbed burst patterns and increased
  asynchronous firing early; the effects diminished as the cultures matured.
* First-sister recordings at every age and both sisters at 6 and 7 days. They
  were read for the recording system's dead time, the domain and the drift.
* The A–C pairs of the four-array preparations at every age. They were read
  for the drift, and the frozen twin was run on them (section 6). Those
  preparations are not scored.
* The frozen twin's scores on held-out simulations, including the two
  contrasts of the stop rule.

## 5. The guard

Typicality threshold at the 97.5th percentile of validation cultures;
predictive-check threshold at the 97.5th percentile over held-out simulated
pairs. Tests, with the bars of versions 1 and 2: held-out simulated sister
pairs must pass (fire at most 0.10); simulated sister pairs with receptor
kinetics the twin cannot represent, and recorded pairs with each electrode
circularly shifted, must fire (at least 0.80 each). On the recorded pairs the
guard reads the first two windows of every quadrant of every primary-age pair; a
preparation is outside the model when most of its guarded windows fire.

## 6. Simulation evidence at the freeze

{SIMULATION}

## 7. Machine-readable specification

```json
{SPEC}
```
