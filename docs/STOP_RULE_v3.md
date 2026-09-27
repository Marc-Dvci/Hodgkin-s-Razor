# Stop rule for the version 3 freeze

Written and committed on 27 September 2026, before the version 3 twin was
trained. No version 3 model output existed then, on simulations or on
recordings.

The blind test of version 3 rests on one mechanism, `g_nmda`. On development
data the version 2 twin named it for acute D-AP5 in 1 of 7 rat Tampere wells.
A test the twin cannot pass on its own simulations is not worth freezing, so
the version 3 pre-registration is hashed only if these hold. Both are read by
`scripts/freeze_v3.py` and reported in its section 6 whatever they show.

1. **Simulated contrast.** On held-out simulated sister pairs of the scored
   recording system, one 60 s quadrant window per pair: `g_nmda` blocked at
   least five-fold alone, against pairs with no mechanism (drift only). The
   AUROC of the presence probability of `g_nmda` must be at least **0.80**.
2. **Simulated contrast with a co-shift.** The same, with `g_nmda` blocked at
   least two-fold and any second mechanism allowed to move with it, as a
   compensating network would. The AUROC must be at least **0.70**.

If either fails, the twin is revised on simulations only, and the rule is
applied again to the revised twin. Each failed attempt is reported in the
pre-registration. The bars are not moved.

Reported beside the rule, with no bar: the twin's reading of the real
untreated A–C sister pairs of the four-array preparations at 10 to 14 days
(the median presence of `g_nmda` and the share called above 0.5). Those
preparations are excluded from the scored null group.
