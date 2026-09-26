# Hodgkin's Razor

## Category: End-to-End System

---

## Demo video

{{video_link}}

## Code repository

{{repo_link}}

## Live demo

{{demo_link}}

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

On {{n_wells}} wells of rat cortical and human iPSC-derived networks, across
five compounds and vehicle controls, it named the exact conductance in
**{{top1}}** of wells against a chance rate of {{chance}}, and the mechanism
class in **{{class_top1}}** against {{class_chance}}. It raised a mechanism on
{{control_rate}} of vehicle controls. How hard the exact question is was
measured first: with the answer handed to a supervised classifier, the nine-way
question tops out at 0.39 on simulations in the same regime. It has never seen a compound
label: it is trained only on simulations, from a GPU network simulator that
runs about {{sims_per_s}} networks per second on one desktop card and is
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
[`docs/TECHNICAL_REPORT.md`]({{report_link}}).

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

{{results_primary}}

{{results_baselines}}

{{results_per_compound}}

{{results_class}}

{{results_guard}}

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
