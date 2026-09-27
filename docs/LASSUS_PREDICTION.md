# The cortico-striatal chip: prediction written before the run

Committed on 27 September 2026, before `scripts/lassus_study.py` was ever run.
The study, the chip model and every setting are fixed at these git blobs:

| File | Blob |
|---|---|
| `scripts/lassus_study.py` | `66e8901701ac3a31b1181ad2a11ade085cc897f0` |
| `hodgkins_razor/chip.py` | `4b1f5a8245adb3b1643a3d325f7cd345b432a88c` |

Command: `python scripts/lassus_study.py` (defaults: 192 chips, 300 s, NMDA
factors 0.3, 0.5 and 0.15 on the striatal chamber, seed 2018).

## What is being reproduced

Lassus et al. (Sci Rep 2018) rebuilt cortico-striatal networks in two-chamber
microfluidic chips joined by axon diodes, and imaged them with Fluo-4 at 2 Hz.
They perfused GluN2B antagonists (ifenprodil, RO256981) into the striatal
chamber only. Striatal calcium-event frequency fell, and so did
striato-striatal and cortico-striatal synchrony.

The twin has no NMDA subunits, so GluN2B is modelled as a reduction of the
NMDA conductance of the striatal chamber's neurons. It has no dopamine
receptors, so the D1 and D2/D3 results of the same paper are outside it and
are not attempted.

## Prediction and criteria

Headline: the striatal target at NMDA x0.3, on chips whose target is active
with the cortical cocktail. The paired change (drug minus baseline, same chip)
of each of the three readouts must be:

1. **negative in more than half of the chips**, and
2. **lower with a one-sided Wilcoxon signed-rank p below 0.05**.

| Readout | Published direction | Criterion |
|---|---|---|
| striatal calcium-event frequency (`tgt_freq`) | lower | 1 and 2 |
| striato-striatal synchrony | lower | 1 and 2 |
| cortico-striatal synchrony | lower | 1 and 2 |

Reproduced means all three pass. Partly reproduced means one or two pass.
Each readout is reported separately.

Secondary, stated in advance:

* **Striatal identity is what makes the result.** With the same chips and a
  generic, partly excitatory target, the share of chips whose target frequency
  falls is lower than with the striatal target. In version 2, the generic
  target fell in 0.30 of chips.
* **Dose order.** For target frequency, the share of chips lower is ordered
  x0.15 ≥ x0.3 ≥ x0.5.
* The striatal target is not self-active: its median rate with the cortex held
  silent is below its median rate with the cortex driving it.

Every outcome is reported whatever it shows. If the headline fails, the chip
model is not tuned and rerun under this file. Any revision is a new,
separately dated prediction.
