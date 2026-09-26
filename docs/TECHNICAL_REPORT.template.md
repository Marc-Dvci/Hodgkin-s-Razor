# Hodgkin's Razor: mechanism inference for neural organ-on-chip recordings

**AI4S Open Innovation: AI for Life Science**, 5th Pazhou Algorithm Competition
**Category: End-to-End System**
Author: Marc Donovici (solo entrant). Apache-2.0.
Code: `{{repo}}` · Pre-registration hash `{{prereg}}`

---

## Summary

A microelectrode array on a neural organ-on-chip records every spike from a
living network for half an hour. Standard analysis turns that into a table of
descriptive statistics: firing rate fell, bursts stopped, synchrony dropped. It
does not say which molecular mechanism the compound moved, and that is the
question a drug programme, a safety assessment and a disease model all ask.

Hodgkin's Razor fits a conductance-based network model to the recording. It
reads a baseline and a treated recording of the same well and returns the
mechanism the compound moved, with an effect size, a credible interval and a
calibrated probability. When the fitted model cannot reproduce the recording it
says so and names nothing.

On {{n_wells}} wells of rat cortical and human iPSC-derived networks it named
the exact conductance in **{{top1}}** of wells against a chance rate of
{{chance}}, and the mechanism class in **{{class_top1}}** against a chance rate
of {{class_chance}}. On vehicle controls it raised a mechanism in
{{control_rate}} of wells. The model is trained only on simulations and has
never seen a compound label.

How hard the exact question is was measured before the recordings were scored.
With the answer given directly to a supervised classifier, on simulations in
the regime the twin operates in, the nine-way question tops out at 0.39 and the
four-way question among the receptor and channel mechanisms at 0.64. Those are
limits of a sixty-second paired recording read through these statistics, not of
one estimator, and they are why every call carries a probability and why the
mechanism class is reported beside the conductance.

---

## 1. Problem

Organ-on-chip platforms are built to replace animal experiments in
neurotoxicity and neuropharmacology. The readout that carries the most
information per unit of cost is extracellular electrophysiology: a multi-well
MEA plate records network activity non-invasively for weeks.

The analysis, however, stops at description. Published MEA pipelines report
firing rate, burst rate, burst duration, percentage of spikes in bursts and
pairwise synchrony. These are the right descriptors, and they do not answer the
question the experiment was run to answer. Two compounds with opposite
mechanisms can produce the same drop in firing rate: blocking AMPA receptors
removes excitatory drive, opening GABA-A channels shunts the membrane, and both
show up as "firing fell by 70 percent, bursting stopped".

The distinction matters. In safety pharmacology, a compound that silences a
network by blocking sodium channels carries a different liability from one that
does it by potentiating inhibition. In disease modelling, a patient line whose
network is quiet because of reduced excitatory conductance points somewhere
different from one that is quiet because of raised adaptation current.

The supporting organisation for this challenge, CellShells, states the goal
directly: to build AI-driven organ-on-chip digital twins from neural chip data,
moving the field from experimental description toward predictive simulation.
This project is an attempt at exactly that step.

## 2. Related work and what is new

Simulation-based inference has been applied to this problem once, well. Doorn,
van Putten and Frega (*Communications Biology*, 2025) trained a neural density
estimator on 300,000 simulations of a 100-neuron Hodgkin-Huxley network and
recovered ten parameters from MEA features of human iPSC-derived networks,
identifying mechanisms behind SCN1A and CACNA1A patient phenotypes. Their code,
their simulation bank and their trained estimator are public, and they are the
baseline here.

They state five limitations. This project addresses each:

| Their limitation | This work |
|---|---|
| The model has only excitatory neurons, so GABAergic compounds are out of reach | an inhibitory population and a GABA-A conductance; gabazine and GABA are in scope and are scored |
| Posteriors are comparable only within one MEA experiment, because of batch effects | the shift is inferred within a single well, with wiring and recording nuisances held fixed across the pair, so culture and plate offsets cancel |
| Model misspecification is noted as a risk and not handled | a calibrated predictive check that refuses to name a mechanism, tested on two classes of recording it must reject |
| A single population, with no device geometry | a two-compartment chip twin with directional microchannels and a population calcium readout |
| Effect sizes only | a separate calibrated probability, per mechanism, that it moved at all |

Three further contributions are independent of that comparison: a GPU simulator
fast enough to build the bank on a desktop and bit-reproducible while doing it;
a prior whose ranges were set by measuring the model's response rather than
assumed; and an evaluation whose answer key, metric and failure conditions were
hashed before the first recording was scored.

## 3. Data

| Dataset | Role | Licence |
|---|---|---|
| Tampere comparative MEA dataset | every recorded result | CC BY 4.0 |
| Doorn et al. 2025 SBI repository | prior art and comparison | Apache-2.0 |

The Tampere dataset (Hyvärinen and colleagues, *Scientific Data* 9:118, 2022)
provides two pharmacology plates recorded on 48-well MEAs at 16 electrodes per
well: rat cortical neurons at DIV 22 and human pluripotent-stem-cell-derived
neurons at DIV 29. Each plate holds three 30-minute recordings of the same
wells, a baseline, a recording after compound wash-on, and a recording after
TTX. Compounds are CNQX 50 uM, D-AP5 50 uM, GABA 10 uM, gabazine 30 uM, kainic
acid 5 uM, and vehicle controls, with 7 wells per condition on the rat plate
and 4 on the human plate.

Electrodes the original authors marked as noisy are dropped. Analysis windows
are three consecutive 60-second blocks starting 300 seconds into each
recording, the same window on both sides of every pair.

No restricted, clinical or personal data is used. No data was purchased, and
nothing in this project requires a paid service.

## 4. The model

### 4.1 Network

Each network is 256 conductance-based neurons on a 16 x 16 grid at 45 um
spacing, read by 16 electrodes in a 4 x 4 arrangement. The membrane follows
Traub-Miles kinetics, with sodium, delayed-rectifier potassium and leak
currents, a slow calcium-dependent after-hyperpolarisation, and a noisy
membrane drive.

Synapses are conductance-based: AMPA and NMDA for excitatory connections, with
the standard magnesium block on the NMDA component, and GABA-A with a chloride
reversal for inhibitory ones. A fraction of neurons is inhibitory. Every
presynaptic terminal carries short-term depression, and conduction delays scale
with the distance between neurons.

The fourteen parameters, and whether a wash-on may move them:

| Parameter | Range | Scale | Shiftable |
|---|---|---|---|
{{param_table}}

Wiring, the inhibitory fraction and the two recording nuisances describe the
well and the amplifier, not the drug. Holding them fixed across a pair is the
mechanism by which culture-to-culture and plate-to-plate offsets cancel.

### 4.2 Observation model

An electrode reports an event when a neuron within its pickup radius fires,
subject to a 2 ms dead time, which is what makes a real electrode report
multi-unit activity rather than single spikes. Two nuisance parameters complete
the model: the fraction of candidate events that clear the detection threshold,
and the spread of pickup across electrodes within a well. Both are properties of
the plating and the amplifier, so both are held fixed across a pair.

These two parameters are what let one simulator match datasets recorded on
different systems. The Tampere plates report 0.05 to 7 events per second per
electrode; the recordings used by Doorn et al. report 2.6 to 76. The dynamics
are the same; the detection threshold is not.

### 4.3 Implementation

One CUDA block simulates one network and one thread integrates one neuron, with
all neuron state held in shared memory so the time loop never leaves the
streaming multiprocessor. Synaptic arrivals pass through a shared ring buffer,
which quantises conduction delays to 0.8 ms. Gating variables and the membrane
potential are advanced by exponential Euler at a 0.1 ms step.

Throughput is about {{sims_per_s}} networks per second of 65 simulated seconds
on one RTX 4070, so the {{bank_pairs}}-pair bank used here was built in under
two hours on a desktop.

The simulation is bit-reproducible. Spiking neurons are collected with a warp
ballot and read back in neuron order, so the floating-point accumulation of
synaptic input never reorders between runs. An atomic counter, the obvious
implementation, does reorder and produced runs that diverged; the test
`test_simulator_is_deterministic` exists because that version failed it.

### 4.4 The domain the twin covers

Sampling the prior uniformly produces mostly networks that cannot answer the
question. Measured over 9 600 screening simulations, **9.9 percent** of
parameter sets give a living, network-driven culture. In the rest, activity
comes from membrane noise rather than from the network's own synapses, so
removing a synaptic conductance changes nothing that a recording can show, and
no method could identify what a compound did.

That is not a nuisance to be trained through. It is a statement about which
experiments carry information. The criterion is measured on the recording, not
on the parameters:

| | |
|---|---|
| firing rate | 0.3 to 40 events per second per electrode |
| active electrodes | at least 60 percent |
| network bursts | at least 1 per minute |
| spikes inside bursts | at least 5 percent |

`scripts/fit_regime.py` screens the prior once, fits a classifier over
parameters, and `scripts/make_bank.py` uses it as a proposal and then confirms
every accepted parameter set by simulation. Acceptance rises from 9.9 percent
to about 38 percent, which is what makes the bank affordable.

Applying the same criterion to the recorded baselines puts **70 percent of the
rat windows and 99 percent of the human windows** inside the twin's domain,
spread evenly across compounds. Recordings outside it are not excluded from the
evaluation: they are scored like any other, and the guard is what reports them.

The measured identifiability, on simulations where the answer is known and the
baseline is a living culture, is a 9-way top-1 of 0.39 for a classifier given
the labels directly. Among the four receptor and channel mechanisms that the
recorded compounds act on it is 0.64, against a chance rate of 0.25. Those are
ceilings for any method reading the same features, and they are the reason the
report gives a probability per mechanism rather than a single name.

### 4.5 Choosing the features by measurement

The same discipline applies to the features. Removing excitatory drive and
adding inhibition both lower the firing rate, and the descriptors in common use
do not separate them: with the twenty-two standard statistics, a classifier
given the labels directly reaches 0.57 on the four receptor and channel
mechanisms. Eighteen further statistics were added for that job, chosen for
what they measure rather than for what they scored: spike-time tiling, the rate
that survives between bursts, burst participation and onset jitter, burst shape,
Fano factors and population autocorrelation at three timescales each, and
rate-robust interval statistics. They raise that number to 0.64. Both figures
are measured on simulations, before any recording was scored.

`scripts/sensitivity.py` sweeps each parameter across its prior range against
six random backgrounds and reports whether the recording responds. A parameter
whose range sits in a flat region cannot be recovered and cannot represent a
compound acting on it.

That sweep changed the model. The sodium conductance was initially given a
linear range from 0.5 to 2.0 times the nominal value. Sweeping it across that
range, the lowest firing rate the model could reach was **13.5 events per
second per electrode**: the entire prior sat above the region where a channel
block happens, so TTX was unrepresentable and the twin would have explained it
with whatever else fit. Log-scaled from 0.08, the same sweep reaches **0.27**,
and the rank correlation with firing rate rises from 0.83 to 0.98. Both sweeps
are kept, in `results/sensitivity_before_sodium_fix.json` and
`results/sensitivity.json`.

### 4.6 The razor

A compound acts on one or two targets. The prior over the shift is sparse: for
each shiftable parameter, a narrow component of width {{inactive_scale}} in
transformed units, and for a small active set, a wide component spanning fold
changes from {{min_fold}} to 25 in either direction. The lower bound is set
where it is because a smaller change leaves no trace in a 60 second recording,
and labelling such a case active would ask the model to detect something that
is not there. The number of active mechanisms is
drawn from {0: 0.15, 1: 0.45, 2: 0.25, 3: 0.15}. The zero case is what teaches
the model what a vehicle control looks like.

Shifts are applied in the transformed space and clipped to the prior box, and
the label stored is the shift that was realised after clipping, so a training
label always describes the simulation that was actually run.

### 4.7 Inference

Two heads read the same paired recording, because effect size and presence are
different questions.

A conditional masked autoregressive flow models the joint posterior over the
fourteen baseline parameters and the nine shifts, conditioned on the
transformed features of both recordings and their difference. The shift is
modelled on an inverse-hyperbolic-sine scale so that the narrow prior component
occupies order-one width, which keeps the target well conditioned.

A presence head outputs, per mechanism, the probability that it was in the
active set. It is trained with the same bank and calibrated on a split that the
flow never trained on.

Reading presence off the size posterior would have been possible and wrong: the
sparse prior is so sharp that a credible interval containing zero conflates "no
effect" with "no information".

### 4.8 The guard

A posterior is worth reading only if the twin, run at those parameters,
reproduces the recording. The check re-simulates from posterior draws and scores
the measured features against the predictive spread as a mean squared robust
z-score over both recordings.

The threshold is the 95th percentile of that statistic over held-out bank
records, where the model generated the data and is correct by construction. It
is fixed before any recording is scored.

The guard is tested on cases it must catch: simulations from a variant
simulator whose AMPA and GABA decay constants were changed, which no parameter
of the twin can produce; and real recordings whose spike times were circularly
shifted per electrode, which preserves every per-electrode rate and interval
distribution while destroying the network structure between electrodes.

## 5. Evaluation

Every threshold, metric, answer key and failure condition is in
`PREREGISTRATION.md`, hashed as `{{prereg}}`. `scripts/evaluate.py` verifies the
hash before running and refuses to proceed if the file changed.

The unit is the well, not the window: windows from one well are replicates.

### 5.1 Primary result

{{results_primary}}

This is the second scored run. The first is in `results/run1/`, and section 6b
gives the defects it exposed, the independent check that found them and the
corrections made. Both runs are reported in full.

The answer key, the primary metric, the unit of analysis and the success
threshold are the ones hashed in the pre-registration and are unchanged. One
number in it moved for a stated reason: adding a tonic inhibitory conductance
gives the model ten shiftable mechanisms rather than nine, so the chance rate
for the primary metric is 0.100 rather than the 0.111 written there.

### 5.2 Per compound

{{results_per_compound}}

### 5.3 Against baselines

{{results_baselines}}

The unpaired baseline is the same simulator, features, bank and flow size with
only the paired design removed: parameters are inferred separately for the two
recordings and the shift is the difference of posterior medians. The supervised
classifier is given the compound labels the twin never sees, under
leave-one-well-out cross-validation.

### 5.4 By mechanism class

{{results_class}}

A compound identified as acting on inhibition rather than on excitatory
transmission is a useful answer even when the exact conductance is not
resolved, and it is the level at which the measured identifiability is high.
This analysis is secondary and was not pre-registered; it is reported alongside
the primary metric, not in place of it.

### 5.5 Calibration and recovery

{{results_calibration}}

### 5.6 The guard

{{results_guard}}

## 6. The chip twin

The geometry that CellShells works in is two chambers joined by asymmetric
microchannels that let axons grow one way, the "axon diode" of Peyrin et al.
(*Lab on a Chip*, 2011), which reaches about 97 percent directional
selectivity. Lassus et al. (*Scientific Reports*, 2018) read such
cortico-striatal chips with Fluo-4 calcium imaging at 2 Hz rather than with
electrodes.

The same CUDA kernel runs that device, because a chip is a wiring matrix, a
delay table and an electrode map. Four parameters describe the device: the
cross-channel connection probability, the directional selectivity, the strength
of cross-channel synapses, and the autonomy of the target chamber, which scales
both its tonic drive and its own recurrent weights so that at the low end the
target only fires when the source drives it.

A population calcium observation model convolves the spike train of each
chamber with a double-exponential indicator kernel and samples it at 2 Hz.

`scripts/chip_study.py` asks what each readout resolves.

{{results_chip}}

This is the question a laboratory faces before it runs the experiment. The
answer is a property of the readout and the question together, and it is
computable in advance.

## 6b. Does the twin reproduce known pharmacology?

Accuracy on a scored set says nothing about whether a model can represent the
compound it is being asked about. `scripts/pharmacology_check.py` applies a
saturating block or agonist at each mechanism to living simulated cultures and
compares the result against the published direction and rough magnitude for the
matching compound. Those expectations come from the neuropharmacology
literature and from no dataset scored here, so the check is independent of the
evaluation.

{{results_pharmacology}}

### What the check found, and what it changed

The first scored run of this project failed, at a top-1 of 0.091 against a
chance rate of 0.111, and the confusion matrix said why: of 22 wells treated
with an AMPA-acting compound, 10 were attributed to sodium channels and none to
AMPA. The pharmacology check, run afterwards on the same model, found the cause
without reference to any recording. Blocking AMPA in that simulator left 90
percent of the firing, and raising inhibition left 85 percent. A twin in which
those two compounds do almost nothing cannot attribute a recording to them, and
its accuracy on a scored set was never going to reveal which of the fifteen
parameters was at fault.

Three defects, all of the same kind:

1. **NMDA substituted for AMPA.** At this model family's resting potential of
   -39.2 mV the magnesium block leaves about a quarter of the NMDA conductance
   open, and NMDA decays fifty times more slowly than AMPA, so equal
   conductances give NMDA twelve times the synaptic charge. With both priors
   spanning the same range, NMDA carried fast transmission and an AMPA block
   was compensated. The NMDA range was scaled down accordingly.
2. **A bath-applied agonist had no way to act.** The model had only synaptic
   GABA-A, released by inhibitory neurons. GABA and muscimol open
   extrasynaptic receptors on every cell, so a tonic inhibitory conductance was
   added as a parameter in its own right.
3. **The receptor ranges did not reach the blocked extreme.** The same defect
   already caught for sodium, where the original prior could not fall below a
   firing rate of 13.5 events per second per electrode. Every receptor range
   now reaches the value a saturating antagonist produces. Because that leaves
   most of the range dead, the prior a baseline is drawn from was separated
   from the support a compound can reach: a healthy culture is never at the
   blocked extreme, but a drug must be able to take it there.

A fourth followed from the same reasoning. A cortical culture is defined
pharmacologically by its activity depending on fast excitatory transmission, so
the criterion for admitting a simulated culture to the bank now requires that
blocking AMPA collapses it. A network that keeps firing through an AMPA block
is not the preparation these compounds were applied to.

Measured on simulations, with the answer handed to a supervised classifier, the
corrections move separation among the four receptor and channel mechanisms from
0.64 to 0.89, the four mechanism classes from 0.62 to 0.76, and the full
mechanism question from 0.39 to 0.63. None of those numbers uses a recorded
label.

### The guard also failed, and why

The first run's guard fired on 12 percent of recordings from a simulator whose
receptor kinetics the twin has no parameter for, where the pre-registered
requirement was 80 percent. The statistic averaged a robust z-score over all
eighty numbers, and a recording the model cannot produce usually fails on a few
statistics rather than drifting on all of them, so the average buried it. It
now takes the worst eight. The safety net that should have caught the
misspecification was itself too blunt to see it, which is the more useful half
of that finding.

## 7. Reliability and limitations

**What the evaluation does not establish.** Every recorded result comes from
two plates in one published dataset. Two species and two independent cultures
are not a multi-laboratory validation, and the wells within a plate share a
culture, a plating day and an amplifier.

**One concentration per compound.** The Tampere plates use a single
concentration, so no dose-response in parameter space is reported. The
multi-laboratory HESI dataset that would supply seven laboratories and five
concentrations per compound is registered as public but served behind a login,
so it is listed here as an open item rather than a result.

**The twin carries the mechanisms its equations carry.** A compound acting
through a target the model does not represent is reported as outside the model.
That is the correct answer and not an informative one, and it is the reason the
guard exists rather than a defence of it.

**Kainic acid is a hard case by construction.** At 5 uM the recorded effect is
a fall in firing, through receptor desensitisation and depolarisation block,
while the answer key scores the compound on raising AMPA conductance. The
pre-registration states that it is scored on the parameter regardless of sign,
and its direction is reported separately.

**The chip twin has not been fitted to a physical chip.** It reproduces
published directions and answers the readout question. No recording from a
two-compartment device is public, and one from the supporting organisation
would be the single most valuable addition to this work.

## 8. Impact

A working mechanism readout changes what an MEA experiment is for. Three
consequences follow directly from what is demonstrated here.

**Screening gains a mechanistic axis.** A developmental-neurotoxicity or
seizure-liability screen currently ranks compounds by how much they perturb
activity. With a mechanism attached, compounds group by target, and a hit whose
mechanism is inconsistent with its intended pharmacology is visible
immediately.

**Patient-derived lines can be compared across sites.** Because the shift is
estimated within a well, the culture and plate offsets that normally prevent
pooling across laboratories cancel. That is what the paired design buys, and it
is measured here against the unpaired ablation.

**Experiments can be designed before they are run.** The chip study shows the
readout question answered in advance: which device parameters an electrode
array resolves, which a 2 Hz calcium movie resolves, and which neither does.

The refusal path matters as much as the answer. A system that names a mechanism
for every recording is useless in a laboratory, because the recordings that
matter most are the surprising ones. The guard is what makes the output
safe to act on.

## 9. Reproduction

```bash
pip install -r requirements.txt
python demo.py                      # bundled recordings, no GPU needed
python -m pytest tests/ -q          # {{n_tests}} tests
```

Full rebuild on a CUDA device:

```bash
pip install -r requirements-gpu.txt
python scripts/fetch_tampere.py
python scripts/sensitivity.py
python scripts/make_bank.py --pairs {{bank_pairs}}
python scripts/train.py  --bank data/bank --out models/twin
python scripts/train.py  --bank data/bank --out models/twin_unpaired --unpaired
python scripts/evaluate.py
python scripts/render_results.py
python scripts/chip_study.py
python scripts/figures.py
```

Hardware used: one RTX 4070 12 GB, 32 GB RAM, six-core CPU. No cloud service,
paid API or proprietary model is required at any point.

## 10. Sources and licences

**Data.** Tampere comparative MEA dataset, CC BY 4.0. Doorn et al. 2025 SBI
repository, Apache-2.0.

**Software.** Python, NumPy, SciPy, pandas, PyTorch, zuko, CuPy, scikit-learn,
FastAPI, uvicorn, Matplotlib, pytest. All open source under permissive
licences.

**References.**

1. Doorn N., van Putten M., Frega M. Automated inference of disease mechanisms
   in patient-hiPSC-derived neuronal networks. *Communications Biology*, 2025.
2. Hyvärinen T. et al. Comparative microelectrode array data of the functional
   development of hPSC-derived and rat neuronal networks. *Scientific Data*
   9:118, 2022.
3. Peyrin J.-M. et al. Axon diodes for the reconstruction of oriented neuronal
   networks in microfluidic chambers. *Lab on a Chip* 11:3663, 2011.
4. Lassus B. et al. Glutamatergic and dopaminergic modulation of cortico-striatal
   circuits probed by dynamic calcium imaging of networks reconstructed in
   microfluidic chips. *Scientific Reports* 8:17461, 2018.
5. Papamakarios G., Pavlakou T., Murray I. Masked autoregressive flow for
   density estimation. *NeurIPS*, 2017.
6. Cranmer K., Brehmer J., Louppe G. The frontier of simulation-based
   inference. *PNAS* 117:30055, 2020.
7. Traub R., Miles R. *Neuronal Networks of the Hippocampus*. Cambridge, 1991.
8. Tsodyks M., Markram H. The neural code between neocortical pyramidal neurons
   depends on neurotransmitter release probability. *PNAS* 94:719, 1997.

**External models and services.** None. No pretrained model, foundation model,
commercial API or cloud service is part of the system. Every model in this
repository was trained here from simulations generated here, and every number
in this report is produced by a script in the repository from public data.

---

*Generated {{generated}} from `results/results.json`.*
