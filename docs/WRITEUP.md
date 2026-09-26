# Hodgkin's Razor

## Category: End-to-End System

---

## Demo video

TO BE ADDED (YouTube and Bilibili)

## Code repository

https://github.com/Marc-Dvci/hodgkins-razor

## Live demo

TO BE ADDED

---

## Project summary

A microelectrode array on a neural organ-on-chip records every spike from a
living network. Standard analysis turns that into a description: firing fell by
seventy-one percent, bursting stopped, synchrony dropped. It does not say which
molecular mechanism the compound moved, and that is the question the experiment
was run to answer. Blocking AMPA receptors, opening chloride channels and
shutting down sodium channels all look the same in a rate plot, and they mean
different things for a drug programme.

Hodgkin's Razor fits a conductance-based network model to the recording itself.
It reads a baseline and a treated recording of the same well and returns the
parameter the compound moved, with an effect size, a credible interval and a
calibrated probability. The razor is the prior: among the mechanisms that could
explain the change, prefer the fewest.

On 66 wells of rat cortical and human iPSC-derived networks, across
five compounds and vehicle controls, it named the exact conductance in
**30%** of wells against a chance rate of 0.100, and the mechanism
class in **39%** against 0.25. It raised a mechanism on
0% of vehicle controls. How hard the exact question is was
measured first: with the answer handed to a supervised classifier, the nine-way
question tops out at 0.39 on simulations in the same regime. It has never seen a compound
label: it is trained only on simulations, from a GPU network simulator that
runs about 35 networks per second on one desktop card and is
bit-reproducible.

Every posterior is checked by re-simulating the twin and comparing it to what
was measured. A recording the model cannot reproduce is reported as outside the
model and no mechanism is named. That guard is tested on two classes of
recording it must reject, and both results are published whatever they came out
at. The answer key, the metric, the operating points and the failure conditions
were written and hashed before the first recording was scored.

The same simulator runs a two-compartment chip with one-way microchannels and a
2 Hz calcium readout, and answers what each readout can resolve before the
experiment is run.

---

## Technical report

The full report is in the repository at
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

### The problem

Organ-on-chip platforms exist to replace animal experiments. The readout that
carries the most information per unit cost is extracellular electrophysiology.
The analysis stops at description, and description does not separate mechanisms
that produce the same rate change.

### Method

**Simulator.** 256 conductance-based neurons on a 16 x 16 grid read by 16
electrodes. Excitatory and inhibitory populations, AMPA, NMDA and GABA-A
synapses with the magnesium block on NMDA, short-term depression, slow
after-hyperpolarisation, distance-dependent delays. One CUDA block per network
with the entire time loop in shared memory. Bit-reproducible: spiking neurons
are collected with a warp ballot and read back in neuron order, so
floating-point sums never reorder. An atomic counter, the obvious
implementation, does reorder; the determinism test exists because that version
failed it.

**Priors set by measurement.** `scripts/sensitivity.py` sweeps every parameter
across its range and reports whether the recording responds. That sweep changed
the model: the sodium range originally sat entirely above the region where a
channel block happens, so TTX would have been unrepresentable. It is now
log-scaled from 0.08.

**The paired design.** Wiring, inhibitory fraction, detection threshold and
electrode pickup describe the well and the amplifier, not the drug, so they are
held fixed across a pair. The shift is inferred within a well, which is what
cancels the culture and plate offsets that otherwise dominate any comparison
between two recordings.

**Two heads.** A conditional masked autoregressive flow gives the joint
posterior over baseline parameters and shift, which supplies effect sizes and
lets the twin be re-simulated. A separate calibrated head gives, per mechanism,
the probability it moved at all. Size and presence are different questions.

**The guard.** The posterior is re-simulated and the measured features scored
against the predictive spread. The threshold is the 95th percentile over
held-out simulation records, fixed in advance.

### Results

| Quantity | Value | Pre-registered | Outcome |
|---|---|---|---|
| Top-1 mechanism accuracy | **0.303** (66 wells) | at least 0.50 | **not met** |
| Chance | 0.100 | | |
| Control false-mechanism rate | **0.000** (11 wells) | at most 0.20 | met |
| Top-2 accuracy | 0.364 | | |
| Direction agreement | 0.650 (20 wells) | | |

| Method | Sees compound labels | Top-1 accuracy |
|---|---|---|
| Hodgkin's Razor | no | **0.303** |
| Same twin, paired design removed | no | 0.091 |
| Supervised nearest centroid, leave-one-well-out | yes | 0.515 |

| Compound | Target | Wells | Top-1 | Top-2 | Rat | Human |
|---|---|---|---|---|---|---|
| CNQX | `g_ampa` | 11 | 3/11 | 6/11 | 1/7 | 2/4 |
| D-AP5 | `g_nmda` | 11 | 2/11 | 2/11 | 0/7 | 2/4 |
| GABA | `g_gaba` | 11 | 0/11 | 0/11 | 0/7 | 0/4 |
| Gabazine | `g_gaba` | 11 | 4/11 | 4/11 | 4/7 | 0/4 |
| Kainic acid | `g_ampa` | 11 | 7/11 | 7/11 | 3/7 | 4/4 |
| TTX | `g_na` | 11 | 4/11 | 5/11 | 4/7 | 0/4 |

Secondary and not pre-registered. Classes are excitatory transmission, inhibitory transmission, intrinsic excitability, and adaptation with short-term plasticity.

| Quantity | Value |
|---|---|
| Class accuracy | **0.394** (66 wells) |
| Chance | 0.250 |

| Compound | Class correct |
|---|---|
| CNQX | 6/11 |
| D-AP5 | 3/11 |
| GABA | 0/11 |
| Gabazine | 4/11 |
| Kainic acid | 7/11 |
| TTX | 6/11 |

Threshold 3.05, the 95% percentile of the discrepancy over 100 held-out records, fixed before any recording was scored.

| Case | Must | n | Fired | Median discrepancy | Outcome |
|---|---|---|---|---|---|
| held-out simulations | pass (at most 0.10) | 40 | 0.07 | 1.0 | met |
| changed receptor kinetics | fire (at least 0.80) | 8 | 0.50 | 2.5 | **not met** |
| real recordings, structure destroyed | fire (at least 0.80) | 40 | 0.30 | 3.0 | **not met** |
| real recordings | reported (-) | 40 | 0.25 | 2.0 | - |

### Data

Tampere comparative MEA dataset (CC BY 4.0): rat cortical DIV 22 and human
iPSC-derived DIV 29 pharmacology plates, 16 electrodes per well, matched
baseline, treated and TTX recordings. Prior art and comparison: the Doorn et
al. 2025 SBI repository (Apache-2.0). No restricted, clinical or personal data
is used. No pretrained model, commercial API or cloud service is part of the
system.

### Reproduction

```bash
pip install -r requirements.txt
python demo.py                 # bundled recordings, no GPU needed
python -m pytest tests/ -q
```

Full rebuild on a CUDA device is nine commands, listed in the README. Hardware
used: one RTX 4070, 32 GB RAM, six-core CPU.

### Limitations

Every recorded result comes from two plates in one published dataset; two
species and two cultures are not a multi-laboratory validation. One
concentration per compound, so no dose-response in parameter space. The twin
carries the mechanisms its equations carry, and anything else is reported as
outside the model, which is correct but uninformative. The chip twin has not
been fitted to a recording from a physical two-compartment device, because none
is public.

---

## Team

Marc Donovici, solo entrant. Background in audit and applied machine learning;
previous work includes a third place of 850 teams in the Google MedGemma Impact
Challenge and finalist at the European Patent Office CodeFest.

This entry does not claim the cross-disciplinary bonus.
