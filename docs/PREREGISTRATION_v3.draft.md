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

The Dynasore and Tampere recordings are development data from here on.

## 2. The test

Charlesworth et al. plated each preparation of mouse hippocampal neurons on
two sister arrays and recorded both twice a week. After the recording at 7
days in vitro, 50 uM APV, a competitive NMDA receptor antagonist, was added to
the medium of one sister (array B, or D in the four-array preparations) of
some preparations and kept there. Other preparations kept both sisters
untreated. A pair is one preparation's two sisters recorded the same day:

* **treated**: the second sister carries APV;
* **null**: neither sister was treated, the preparation never received APV,
  and both sisters share a genotype;
* **pre-drug**: any preparation at 6 or 7 days, before APV was added.

The twin reads the first sister as the baseline and the second as the treated
recording, exactly as it reads one well before and after a compound. For a
null pair the correct answer is that nothing moved.

**Ages.** Early: 9 to 15 days in vitro, the primary range. Charlesworth et al.
report that the effects of chronic APV were strongest early and faded as the
cultures matured. Late: 16 days and after, secondary.

**Readout.** Each recording is read as its four 4 x 4 quadrants (twelve
electrodes each, the grid12 layout), each cut into six 60 s windows from 60 to
420 s, matched by position between sisters. A window is scored when its first
sister has at least 50 events on at least 3 electrodes. The unit is the
**preparation**: presence probabilities are averaged over every scored window
of every early pair of that preparation. Treated and null preparations are
disjoint.

**Answer key.** APV blocks NMDA receptors: `g_nmda`, direction down.

**Primary outcome.** The area under the ROC curve of the presence probability
of `g_nmda`, treated preparations against null preparations. **Success:**
at least 0.70, and the lower end of its 95 percent interval (4,000 bootstrap
resamples of preparations within each group) above 0.50.

**Co-primary outcome, the metric of versions 1 and 2.** The share of treated
preparations whose top-1 mechanism is `g_nmda`, against the same share among
null preparations. **Success:** higher in treated preparations, with a
one-sided Fisher exact p below 0.05.

**Secondary outcomes.** The direction of the `g_nmda` effect among treated
top-1 hits; the share of preparations with `g_nmda` called above 0.5 in each
group; detection (the largest presence probability, treated against null);
the same contrasts at late ages; the contrast by genetic background (wild
type, Gria1 knockout); whether the `g_nmda` reading fades with age in treated
preparations (paired early against late, one-sided Wilcoxon); the guard's
verdicts; the same contrasts for the unpaired twin and for the published
estimator of Doorn et al., scored on the first 60 s window of quadrants 0 and
3 of every early pair, with the size of its standardised `g_NMDA` shift in the
answer's direction as its score. Every outcome is reported whatever it shows.

## 3. What changed from version 2, and on what evidence

No compound label and no second-sister recording made after the drug was added
entered any of these choices.

1. **A recording-system view** for the Multi Channel Systems 60-electrode
   array, read as quadrants on the grid12 layout, with a dead time of 1.08 ms:
   the shortest interval on any electrode of the first-sister recordings.
2. **The sister design.** Sisters are the same preparation, not the same
   network. The bank simulates the second sister with its own wiring and
   electrode pickup, and nudges every culture parameter by a drift that is
   never labelled as a mechanism. Its size, {DRIFT} of each parameter's
   baseline range, was set by `scripts/calibrate_drift.py` so that simulated
   sister differences match those of sister pairs recorded before any drug.
3. **The domain** is fitted by the version 2 rule from the first sisters of
   the scored pairs at 9 days and later, which never carry a drug.

The simulator, the parameter table, the shift prior, the features, the
conditional flow with five presence heads, the guard and its thresholds are
those of version 2.

## 4. What was known about the blind data before this file was hashed

* The metadata sheet: file names, ages, genotypes and which recordings carry
  APV. It defines the pairs.
* The published description: 50 uM APV added after the 7-day recording and
  kept in the medium; chronic APV disturbed burst patterns and increased
  asynchronous firing early; the effects diminished as the cultures matured.
* First-sister recordings at every age and both sisters at 6 and 7 days:
  read for the recording system's dead time, the domain and the drift.

## 5. The guard

Typicality threshold at the 97.5th percentile of validation cultures;
predictive-check threshold at the 97.5th percentile over held-out simulated
pairs. Tests, with the bars of versions 1 and 2: held-out simulated sister
pairs must pass (fire at most 0.10); simulated sister pairs with receptor
kinetics the twin cannot represent, and recorded pairs with each electrode
circularly shifted, must fire (at least 0.80 each). On the recorded pairs the
guard reads the first two windows of every quadrant of every early pair; a
preparation is outside the model when most of its guarded windows fire.

## 6. Simulation evidence at the freeze

{SIMULATION}

## 7. Machine-readable specification

```json
{SPEC}
```
