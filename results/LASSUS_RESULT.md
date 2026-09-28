# The cortico-striatal chip: result

Scored against `docs/LASSUS_PREDICTION.md`, which was committed (`b5a9e02`)
before the first run. Output: `results/lassus_study.json` (192 chips, 300 s,
seed 2018). Chips whose striatal target was active with the cortical
cocktail: 109 striatal, 112 generic.

## Headline: not reproduced

Striatal target, NMDA x0.3 on the striatal chamber. Paired change (drug minus
baseline, same chip):

| Readout | Published | Share of chips lower | Median change | Wilcoxon p (lower) | Criterion |
|---|---|---|---|---|---|
| Striatal calcium-event frequency | lower | 0.505 | -0.2% | 0.46 | not met |
| Striato-striatal synchrony | lower | 0.468 | +0.2% | 0.55 | not met |
| Cortico-striatal synchrony | lower | 0.523 | -0.4% | 0.29 | not met |

The striatal spike rate (not a published readout) does fall, in 0.76 of chips,
but by a median of only 3%.

## Secondaries

| Stated in advance | Outcome |
|---|---|
| Striatal identity makes the result: fewer generic targets fall than striatal ones | **Not met**: target frequency lower in 0.518 of generic against 0.505 of striatal chips |
| Dose order for target frequency, x0.15 ≥ x0.3 ≥ x0.5 | **Partly**: 0.615, 0.505, 0.514 (x0.15 is highest, but x0.3 < x0.5) |
| Striatal target not self-active: rate with the cortex silent below rate with it driving | **Met, by a small margin**: median 0.223 Hz silent against 0.244 Hz driven |

## What it shows

The third row explains the first table. In the chip twin, the cortex supplies
about 9% of the striatal target's firing. The rest comes from the target's own
drive. In Lassus et al., unconnected striatal neurons show no spontaneous
oscillations, and the striatum's activity is driven by the cortex. A drug that
acts on NMDA receptors carrying cortical input therefore has almost nothing to
act on in the model. The mismatch is in how strongly the cortex drives the
striatum, not in the drug model.

This is reported as a failed reproduction. No setting was changed after the
run. A revision would give the chip a cortex-driven striatum: coupling set from
independent evidence (for example the published fraction of striatal events
that follow cortical ones), with residual striatal drive near zero. It would be
a new, separately dated prediction.

---

# Second prediction: result

Scored against `docs/LASSUS_PREDICTION_2.md`, committed (`ec6520b`) before this
run. Output: `results/lassus_study_v2.json` (1,800 chips, 300 s, seed 2018,
striatal down state -6 pA). **34 chips** met the precondition: striatum active
with the cortex driving it (median 0.148 Hz), silent without it (median 0.000
Hz). That is fewer than the 40 the calibration aimed for, and the count is
reported as it came.

## Headline: partly reproduced (two of three readouts)

Striatal target, NMDA x0.3 on the striatal chamber, paired change on the same
chip:

| Readout | Published | Share of chips lower | Median change | Wilcoxon p (lower) | Criterion |
|---|---|---|---|---|---|
| Striatal calcium-event frequency | lower | 0.588 | -3.3% | 0.49 | **not met** |
| Striato-striatal synchrony | lower | 0.765 | -9.0% | 0.002 | **met** |
| Cortico-striatal synchrony | lower | 0.618 | -3.6% | 0.008 | **met** |

The striatal spike rate falls in 0.88 of chips, by a median of 22%. The
calcium-event count, as Lassus et al. define it (upward crossings of each
trace's own mean plus one standard deviation), barely moves. That threshold
scales with the trace, so a uniform fall in rate lowers it with the signal.

## Secondaries

| Stated in advance | Outcome |
|---|---|
| Dose order for target frequency, x0.15 ≥ x0.3 ≥ x0.5 | **Partly**: 0.706, 0.588, 0.676 (x0.15 is highest; x0.3 < x0.5) |
| Generic, self-active target as the contrast | Effects about one third the size or less: median changes -1.4% (frequency), -2.0% and -1.1% (synchrony), against -3.3%, -9.0% and -3.6%. The share of chips with lower frequency is 0.559 against 0.588 striatal |

At the strongest block (x0.15), all three readouts fall with p below 0.01
(0.706, 0.794 and 0.794 of chips lower). That is reported as observed. The
headline dose was fixed at x0.3.

## What it shows

Imposing only the paper's own precondition, a striatum that is silent without
the cortex, moved both synchrony readouts in the published direction. The first
run had moved none. Striatal identity matters: the self-active generic target
barely responds. Event frequency is the readout that remains unmatched.
