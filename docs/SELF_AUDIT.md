# Self-audit against the published criteria

Written before submission, against the rubric in the challenge brief. The
scores are my own. Every gap still open is listed at the end. Every number is
taken from `results/v3/RESULTS.md` or `results/v2/RESULTS.md`.

## Problem importance and potential impact (30 percent)

**What is claimed.** Organ-on-chip electrophysiology produces a description,
not a mechanism, and two compounds with different pharmacology can produce the
same rate plot. Naming the mechanism, against untreated controls read the same
way, changes what the assay is for. The chip twin says before an experiment
which electrodes and how many chips a claim needs.

**Supporting it.**
- Three public datasets from three laboratories: rat, mouse and human cultures,
  and three recording systems.
- A blind result on chronic NMDA blockade, which tests the mechanism reading on
  a drug the model has an answer for.
- Partial reproduction of the sponsor laboratory's cortico-striatal NMDA result
  (both synchrony readouts), after a failed first prediction that is also
  reported.
- A sample-size output that turns the one inconclusive recorded-chip test into
  a design recommendation.

**Against it.**
- The blind tests are on conventional MEA cultures, not chips.
- The one recorded chip set is 17 chips.
- No dose-response, and no human culture inside the model yet.

**Score: 24 of 30.**

## Technical approach and innovation (30 percent)

**What is new.**
- An inhibitory population and tonic GABA-A, so GABAergic compounds are
  readable at all.
- Paired inference within a well, and between sister cultures with a drift
  measured on untreated sisters.
- A calibrated presence probability per mechanism.
- A guard that refuses.
- Recording systems as explicit views.
- A bit-reproducible GPU simulator.
- A chip twin with a striatal target, readout-resolution analysis, sample size
  and a follow-up recommender.
- Methodologically:
  - null-pair scoring, which exposed a comparator's 8/10 as a preference;
  - pre-registration enforced in code;
  - a stop rule that forbids freezing a test the model fails on its own
    simulations.

**Against it.** Each component applies an existing idea (simulation-based
inference, normalising flows, predictive checks); the composition and the
evaluation design are the contribution. The simulator is a point-neuron
simplification.

**Score: 26 of 30.**

## Results and validation (20 percent)

**What is done.**
- Three pre-registrations, each hashed before its blind data was read.
- A prior-art estimator run unchanged on every test.
- An unpaired ablation.
- Untreated null groups inside every primary.
- Every outcome reported, including the failures.

**What holds up.**
- Version 3 primary met: AUROC 0.86 [0.74, 0.96], against 0.49 for the prior
  art and 0.66 unpaired, on the same preparations.
- The canalization secondary: the reading fades with maturation, p = 3e-5.
- Version 2: detection 9/10 against 0/10 on untreated pairs, and mechanism
  class 8/10 (p = 0.0016).

**Against it.**
- Naming the exact mechanism failed on both blind tests: 2/29 top-1 in
  version 3, and 2/10 in version 2.
- In version 3, no preparation's probability reached 0.5.
- The guard missed both must-fire bars on sister pairs (0.58 and 0.63).
- The stop rule needed two attempts; both are listed.
- The Tampere set is development data, scored five times.

**Score: 15 of 20.**

## Reproducibility and implementation quality (10 percent)

- One command runs the demo without a GPU.
- `run_all.py` rebuilds everything, with every frozen model in the repository.
- Checksums on every downloaded file.
- `verify.py` re-checks every pre-registration hash, every frozen model and
  file, the agreement between the written numbers and the JSON, and 43 tests
  (61 checks pass).
- Docker images for CPU and GPU.

**Against it.** The simulation banks are not distributed; they take about 80
minutes each to rebuild. The v3 evaluation needs the low-memory runner on a
32 GB machine.

**Score: 9 of 10.**

## Presentation quality (10 percent)

- A film rendered frame by frame from the results and captures of the running
  application.
- Narration measured and cued, and checked against the results files.
- English and Chinese subtitles.
- The writeup summary in both languages.

**Against it.** Not yet published on YouTube and Bilibili.

**Score: 8 of 10.**

---

## Open items

1. **Publishing.** The repository, the video (YouTube and Bilibili), the
   Pages site, the Kaggle Notebook and Dataset, and the Writeup. The
   placeholders in `docs/WRITEUP.md` are marked `<...>`.
2. **HESI multi-laboratory dataset.** It needs a free EDAP login. It would add
   acute drugs across several mechanisms, drugs the model cannot represent (a
   real-data guard test), true negatives, dose-response and seven
   laboratories.
3. **A recording from the supporting organisation's own chips.**
4. **A human-only domain.** 27 of 28 human Tampere wells were outside the
   model.
5. **Within-well drift in the version 2 twins.** A grid12 twin retrained with
   the measured drift (0.04) was rescored on the Dynasore wells as development
   evidence:
   - named member 4/10, against 3/10 on untreated pairs, where the frozen twin
     had 2/10 against 2/10;
   - AUROC 0.90, against 0.87 for the frozen twin.

   It has not been frozen or blind-tested; a fresh blind set would be needed.
