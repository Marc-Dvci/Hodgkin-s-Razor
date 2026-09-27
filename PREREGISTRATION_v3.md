# Pre-registration, version 3

Written and hashed before any version 3 model output was computed on a scored
recording of Charlesworth et al. (2015), and before any statistic was computed
on the second sister of any pair recorded after the drug was added.
`scripts/freeze_v3.py` fills in the model digests and the simulation evidence,
writes this file and its SHA-256, and nothing is scored until both exist.
`hodgkins_razor/charlesworth.py` refuses to return a second-sister recording
made at 8 days in vitro or later, and `scripts/evaluate_v3.py` refuses to run,
unless this file matches its recorded hash.

Author: Marc Donovici. Date: 27 September 2026.

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
* **A–C**: 21 preparations were plated on four arrays and never treated.
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
primary therefore compares **like with like**: 29 treated
preparations against the 23 null preparations of the same two
genotypes. The pooled null group (78 preparations) is secondary.
Preparations per genotype: `{"treated": {"GluR1": 13, "WT": 16}, "null": {"GNB1": 3, "GRIT": 3, "GluR1": 6, "PSD93": 6, "PSD95": 9, "SAP102": 17, "SPA": 17, "WT": 17}}`.

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
   never labelled as a mechanism. Its size, 0.12 of each parameter's
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

Held-out simulated sister pairs of this recording system (`results\v3\sim_mcs60q.json`):

| Case | n | Top-1 | Top-2 | Class |
|---|---|---|---|---|
| all single mechanism | 7858 | 0.31 | 0.45 | 0.45 |
| saturating | 1730 | 0.43 | 0.58 | 0.57 |

Presence AUROC per mechanism: `g_na` 0.71, `g_kdr` 0.54, `g_ahp` 0.71, `g_ampa` 0.80, `g_nmda` 0.67, `g_gaba` 0.60, `g_tonic_inh` 0.58, `tau_d` 0.74, `u_rel` 0.72, `i_drive` 0.56.

The test's contrast on simulations, one window per pair: `g_nmda` blocked at least five-fold alone (243 pairs) against no mechanism (2801 pairs): AUROC 0.81; top-1 `g_nmda` 0.33 against 0.08; called above 0.5 0.20 against 0.01.

With a co-shift allowed (`g_nmda` down at least two-fold, at most one other mechanism moving; 792 pairs): AUROC 0.75. Both clear the stop rule (`docs/STOP_RULE_v3.md`, bars 0.80 and 0.70); every attempt is in `results/v3/stop_rule_attempts.json`:

| Attempt | Twin | Alone | Co-shift | Passed |
|---|---|---|---|---|
| 1 | `c4096c60149d` | 0.792 | 0.747 | no |
| 2 | `a4f079b64bb4` | 0.810 | 0.755 | yes |

The first twin, trained on 144,000 simulated sister pairs, missed the first bar. Its presence heads stopped improving after about 16 epochs, so a second bank of 144,000 pairs (new seed, same design and drift) was simulated and the twin retrained on both. The bars were not moved. The held-out simulations grow with the bank, so the two attempts are scored on different held-out sets.

The twin on the real untreated A-C sister pairs at 10 to 14 days (21 preparations, 972 windows, excluded from scoring): median presence of `g_nmda` 0.11; called above 0.5 in 0.00 of preparations. Top-1 counts: `g_ahp` 1, `g_gaba` 3, `g_kdr` 11, `g_nmda` 3, `g_tonic_inh` 1, `tau_d` 1, `u_rel` 1.

The recorded test has more between-preparation variation than the simulations, the drug acted for days rather than minutes, and a preparation may compensate, so the recorded AUROC is expected below the simulated one; the bar of 0.70 sits under it for that reason. With 29 treated and 23 genotype-matched null preparations, an observed AUROC of 0.70 carries a 95 percent interval of roughly 0.56 to 0.84, so the bar is testable with this sample.

## 7. Machine-readable specification

```json
{
 "view": "mcs60q",
 "twin": "models/twin_v3_mcs60q",
 "unpaired": "models/twin_v3_unpaired_mcs60q",
 "bank": "data/bank_v3,data/bank_v3b",
 "domain_file": "models/domain_v3.json",
 "drift": 0.12,
 "frozen_models": {
  "models/twin_v3_mcs60q": "a4f079b64bb4ab5c14de93bb1abb5e3af1b499c2fd965d0f656cc2eb720d2f50",
  "models/twin_v3_unpaired_mcs60q": "ed65cc5fffad00109fced491b72ca43c6f2ab9614d85af4aa1794f6efbb21a07"
 },
 "frozen_files": {
  "models/domain_v3.json": "8790c28429a914483763d9396801e460074d36a4353520aae26604cbaad74229",
  "models/drift_v3.json": "9412b68053e867a0261345de7969da699fb7b5c8eb119f5c7ea6b6ea60d28293",
  "models/drift_v3_ac.json": "10463fbfc9b8d9ec8af6067a8ca8d5183f356c328350e887a36e1110983cb41d",
  "hodgkins_razor/charlesworth.py": "88304cea468c7f1157a97bd25804f1a0853893a3a7fc71ce04f24f6f9d7f373f",
  "scripts/evaluate_v3.py": "4088cd04b7541bb73db32ae016f2a416760887260cd1d074e939c9bba511c709",
  "docs/STOP_RULE_v3.md": "7e512fec2883acf778a974972804a8fab714af90c0855429ee4e641ad6b3c420",
  "data/raw/charlesworth2015/g2c-1/00g2cdata.csv": "8611c87277c70e695d2cf70ee7c8f4651afe9975e4485fabce57e73d353ca658"
 },
 "operating_points": {
  "posterior_samples": 1000,
  "window_s": 60.0,
  "windows_per_recording": 6,
  "presence_call": 0.5,
  "predictive_draws": 48,
  "ppc_quantile": 0.975,
  "ppc_calibration_records": 200,
  "min_events": 50,
  "min_electrodes": 3
 },
 "blind_test": {
  "dataset": "Charlesworth et al. 2015, Zenodo 31085",
  "compound": "APV 50 uM, chronic from 7 days in vitro",
  "accept": "g_nmda",
  "direction": "down",
  "early_divs": [
   10.0,
   14.0
  ],
  "unit": "preparation",
  "null_group": "untreated preparations of the treated genotypes; four-array preparations excluded",
  "groups": {
   "n": {
    "treated": 29,
    "null_matched": 23,
    "null_pooled": 78,
    "a_c": 21
   },
   "by_genotype": {
    "treated": {
     "GluR1": 13,
     "WT": 16
    },
    "null": {
     "GNB1": 3,
     "GRIT": 3,
     "GluR1": 6,
     "PSD93": 6,
     "PSD95": 9,
     "SAP102": 17,
     "SPA": 17,
     "WT": 17
    }
   },
   "excluded_four_array_preps": [
    "129_TC194",
    "129_TC195",
    "C57_TC191",
    "C57_TC192",
    "C57_TC193",
    "GNB-1_TC11",
    "GNB-1_TC12",
    "GNB-1_TC13",
    "GNB1_TC21",
    "GNB1_TC22",
    "GRIT_TC61",
    "GRIT_TC62",
    "GRIT_TC71",
    "GRIT_TC72",
    "GRIT_TC73",
    "PSD93_TC51",
    "PSD93_TC52",
    "PSD93_TC53",
    "PSD93_TC61",
    "PSD93_TC62",
    "PSD93_TC63"
   ],
   "treated_divs": [
    10.0,
    11.0,
    13.0,
    14.0
   ]
  },
  "top1_hit_rule": "top-1 mechanism is g_nmda whatever its sign; the sign is a secondary outcome",
  "primary": {
   "metric": "AUROC of p(g_nmda), treated against genotype-matched null preparations",
   "min_auroc": 0.7,
   "ci_lower_above": 0.5,
   "bootstrap": 4000
  },
  "co_primary": {
   "metric": "top-1 g_nmda share, treated against null",
   "test": "one-sided Fisher exact",
   "alpha": 0.05
  }
 },
 "guard": {
  "must_pass_max": 0.1,
  "must_fire_min": 0.8,
  "n_cases": 60,
  "windows_per_quadrant": 2
 },
 "simulated_test": {
  "n_treated": 243,
  "n_null": 2801,
  "auroc_key_window": 0.8099370742077712,
  "top1_treated": 0.3292181069958848,
  "top1_null": 0.0767583006069261,
  "called_key_treated": 0.19753086419753085,
  "called_key_null": 0.011424491253123885
 },
 "simulated_test_coshift": {
  "n_treated": 792,
  "n_null": 2801,
  "auroc_key_window": 0.7549846916144667
 },
 "stop_rule": {
  "alone_min_auroc": 0.8,
  "coshift_min_auroc": 0.7
 },
 "real_a_c_null": {
  "n_preps": 21,
  "n_windows": 972,
  "median_p_key": 0.10856290531378836,
  "called_key": 0.0,
  "top1_counts": {
   "g_ahp": 1,
   "g_gaba": 3,
   "g_kdr": 11,
   "g_nmda": 3,
   "g_tonic_inh": 1,
   "tau_d": 1,
   "u_rel": 1
  }
 }
}
```
