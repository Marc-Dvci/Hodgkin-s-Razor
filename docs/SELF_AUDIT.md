# Self-audit against the published criteria

Written before submission, against the rubric in the challenge brief. Scores
are the author's own and deliberately harsh. Every gap that is still open is
listed at the end.

## Problem importance and potential impact — 30 percent

**What is claimed.** Organ-on-chip electrophysiology produces a description,
not a mechanism, and two compounds with opposite pharmacology produce the same
rate plot. Naming the mechanism changes what the assay is for.

**Supporting it.** Real recordings of five compounds on two species. A
measurable question with a fixed answer key. The supporting organisation's own
stated goal, moving from description to predictive simulation, is the thing
that was built.

**Against it.** Two plates in one dataset. No dose-response. No physical chip
recording from a two-compartment device. The impact argument rests on the
method transferring, and transfer is argued rather than shown.

**Score: 25 of 30.**

## Technical approach and innovation — 30 percent

**What is new.** Inhibition in the network model, so GABAergic compounds are
readable at all. A paired design that infers the shift within a well, so
culture and plate offsets cancel. A separate calibrated presence probability
per mechanism. A guard that refuses, calibrated on held-out simulations and
tested on two classes of recording it must reject. A bit-reproducible GPU
simulator fast enough to build the bank on a desktop. A two-compartment chip
twin that answers what a readout resolves before the experiment is run.

**Against it.** Every component is an application of an existing idea:
simulation-based inference, normalising flows, predictive checks. The
composition is the contribution, not any single part. The simulator is a
simplification of the one it extends, with event-driven synapses and quantised
delays.

**Score: 25 of 30.**

## Results and validation — 20 percent

**What is done.** Pre-registration hashed before scoring, with the answer key,
the metric, the unit of analysis, the operating points and the failure
conditions fixed in advance. Interval coverage and rank statistics on held-out
simulations. Presence reliability with expected calibration error. An ablation
that removes only the paired design. A supervised baseline that is given the
labels the twin never sees. The guard tested on cases it must catch and cases
it must pass. Every compound reported, including the ones that fail.

**What the discipline bought.** The first scored run came out at 0.091 against
a chance rate of 0.111. Rather than tune until the number moved, the cause was
found with a check that uses no recorded label: applying a saturating block at
each mechanism in simulation and comparing against published pharmacology. It
showed that an AMPA block left 90 percent of the firing and that a GABA agonist
had no way to act at all, so two of the six compounds were unrepresentable and a
third was mimicked by sodium. Three parameterisation defects and one missing
conductance were fixed, and the corrections raise separation on simulations from
0.64 to 0.89 among the four receptor and channel mechanisms, again without a
recorded label. Both scored runs are reported.

**Against it.** The pre-registration was scored twice, and only the first
scoring was blind to everything. The second run's model differs, and although
every change was justified on simulation evidence and on published
pharmacology, a reader has to take the ordering on trust beyond the git
history. The guard also failed its own pre-registered test on the first run.

**Score: 16 of 20.**

## Reproducibility and implementation quality — 10 percent

One command runs the demo with no GPU. One command rebuilds everything.
Checksums on every downloaded file. A verification script that re-checks the
pre-registration hash, the data checksums, the tests and the agreement between
the written results and the JSON they came from. Twenty-seven tests, including
one that fails if the simulator stops being bit-reproducible. Docker for CPU
and GPU. A notebook for reviewers without a GPU. A static site that needs no
server.

**Against it.** The bank is not distributed and takes about two hours to
rebuild. The GPU path needs CUDA and is not exercised by the CPU image.

**Score: 9 of 10.**

## Presentation quality — 10 percent

A script cued to measured narration, an architecture diagram, figures produced
from the results file rather than drawn, application screenshots captured from
the running page, subtitles planned in English and Chinese, and hosting on a
platform reachable from the panel's country.

**Against it.** The video is not cut yet.

**Score: 7 of 10, pending the cut.**

---

## Open items

0. **The pre-registration was scored twice.** The first run is in
   `results/run1/`, the second is the headline, and section 6b of the report
   gives the defects, the independent check that found them and the
   corrections. Every change was made on simulation evidence; none used a
   recorded label. The git history carries the order.
1. **The demo video is not produced.** The script, the figures and the captured
   application screens are in the repository; the cut, the narration and the
   upload are not done.
2. **The registration form is not submitted.** It is mandatory for eligibility
   and only the entrant can submit it.
3. **The repository is not published.** Links in the writeup are placeholders
   until it is.
4. **The HESI multi-laboratory dataset is behind a login.** It is the natural
   source of dose-response and cross-laboratory invariance, and both are absent
   because of it.
5. **No recording from a physical two-compartment chip.** One from the
   supporting organisation would be the single most valuable addition.

## What would most improve the score

Not another model. A second dataset. Every weakness above is a data weakness:
one study, one concentration, no chip recording. The method is amortised, so a
new dataset costs a forward pass and nothing else.
