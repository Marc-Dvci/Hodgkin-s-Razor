# The cortico-striatal chip, second prediction

Committed on 27 September 2026, after the first run failed
(`results/LASSUS_RESULT.md`) and before this revision was run with any NMDA
condition.

## What changed, and on what evidence

The first run showed that the model's striatal target was about 91% self-driven:
its rate with the cortex held silent was 0.223 Hz, against 0.244 Hz with the
cortex driving it. Lassus et al. state the opposite as a property of their
preparation: striatal neurons with no cortical partner show no spontaneous
oscillations. The revision imposes that precondition, and nothing else. No NMDA
condition entered it.

1. **Down state.** Striatal neurons rest in a hyperpolarised down state and
   fire only when convergent cortical input lifts them. In the chip twin this is
   a bias on every striatal neuron (`hodgkins_razor/chip.py`, `down_state_pa`).
2. **The bias is set from the precondition, with no drug**
   (`scripts/lassus_study.py --calibrate`, `results/lassus_calibration.json`,
   576 chips, 120 s). A chip meets the precondition when its striatum is active
   (above 0.05 Hz) with the cortex driving it, and its rate with the cortex held
   silent is at most 5% of that. The chosen bias, **−6 pA**, leaves the largest
   share of prior chips meeting it (2.8%).
3. **Only chips that meet the precondition are scored.** The chip prior spans
   weak to strong cortical coupling, and most prior chips cannot drive a silent
   striatum. 1,800 chips are drawn so that about 50 qualify. This is rejection
   on a published, drug-free property of the preparation.

The drug model, the readouts, the criteria and the seed are unchanged from the
first prediction.

Command: `python scripts/lassus_study.py --chips 1800 --down-state -6 --precondition --out results/lassus_study_v2.json`

## Prediction and criteria (unchanged)

Headline: the striatal target at NMDA x0.3, on the chips that meet the
precondition. The paired change of each readout must be:

1. negative in more than half of the chips, and
2. lower with a one-sided Wilcoxon signed-rank p below 0.05.

The three readouts are striatal calcium-event frequency, striato-striatal
synchrony and cortico-striatal synchrony. Reproduced means all three pass.
Partly reproduced means one or two pass.

Secondaries, as before:
* the dose order x0.15 ≥ x0.3 ≥ x0.5 in the share of chips with lower target
  frequency;
* the generic, self-active target (no down state, no precondition) is scored
  alongside as the contrast.

Every outcome is reported, and the first run stays reported as failed. This is
the last revision under this line of evidence. A third would need new
independent data about the preparation, not another reading of these results.
