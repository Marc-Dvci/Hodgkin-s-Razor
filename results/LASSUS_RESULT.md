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
