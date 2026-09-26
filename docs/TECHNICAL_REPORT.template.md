# Hodgkin's Razor: a mechanistic digital twin for neural organ-on-chip recordings

**AI4S Open Innovation: AI for Life Science**, 5th Pazhou Algorithm Competition
**Category: End-to-End System**
Marc Donovici, solo entrant · Apache-2.0 · `github.com/Marc-Dvci/hodgkins-razor`
Pre-registrations: version 1 `c40708bb19668164` (Tampere), version 2 `{{PREREG2}}` (blind tests)

---

## Summary

A microelectrode array under a neural organ-on-chip records every network burst
of a living culture, and standard analysis turns it into a description: firing
fell, bursts shortened, synchrony dropped. It does not say which molecular
mechanism a compound moved. Blocking AMPA receptors, opening chloride channels
and shutting sodium channels can all look the same in a rate plot.

Hodgkin's Razor fits a biophysical network model to the recording itself. From
a baseline and a treated recording of the same well it returns the probability
that each of ten mechanisms moved, the size of the shift if it did, and an
interval. When no fitted model can reproduce the recording it says so and names
nothing. It is trained only on simulations from a GPU network simulator and has
never seen a compound label.

{{SUMMARY_RESULTS}}

The same simulator runs a two-compartment chip with directional microchannels,
the geometry of the supporting organisation, and computes before an experiment
is run which readout can resolve which property of a device. {{SUMMARY_CHIP}}

Everything runs from public data on one desktop GPU; the demo, the web
application and the notebook run on a CPU.

---

## 1. The problem

Organ-on-chip platforms exist to replace animal experiments in neurotoxicity
and neuropharmacology, and extracellular electrophysiology is the readout that
carries the most information per unit cost: a multi-well array records network
activity non-invasively for weeks. The analysis stops at description. Published
pipelines report firing rate, burst rate and duration, the fraction of spikes
in bursts and pairwise synchrony. Those are the right numbers, and they do not
answer the question an experiment is run to answer.

Mechanism matters downstream. In safety pharmacology a compound that silences a
network by blocking sodium channels carries a different liability from one that
potentiates inhibition. In disease modelling a patient line that is quiet
because of reduced excitatory conductance points somewhere different from one
that is quiet because of raised adaptation. The supporting organisation of this
challenge, CellShells, states the goal directly: AI-driven organ-on-chip
digital twins built on neural chip data, moving the field from experimental
description toward predictive simulation. This project is that step, for the
two questions a laboratory asks: what did the compound do, and what should be
measured next.

## 2. What the system is

An end-to-end system from a recording to a decision (Figure 1):

1. **Input.** Two spike tables of one well, before and after wash-on, from any
   array (Axion, Multi Channel Systems, or a two-column CSV).
2. **Recording system.** The electrode layout and detection dead time of the
   array are applied identically to the recording and to every simulation.
3. **Statistics.** Forty summary statistics from one function used for recorded
   and simulated events alike.
4. **Posterior.** Five presence heads give the probability that each mechanism
   moved; a conditional normalising flow gives the culture's parameters and the
   size of each shift given that it moved.
5. **Guard.** Typicality (does any simulation resemble this pair?) and a
   predictive check (does the twin, re-simulated at the posterior, reproduce
   it?). If either fires, no mechanism is named.
6. **Report.** Called mechanisms with direction, effect and interval, the
   mechanism class, the rivals the recording cannot separate, the culture's
   parameters, a content hash; as JSON, Markdown, a web page or a notebook cell.
7. **Chip twin and experiment design.** The same kernel simulates a
   two-compartment device and ranks readouts by what they resolve.

![Figure 1. Architecture.](../results/v2/figures/architecture.png)

## 3. Related work and what is new

Simulation-based inference has been applied to MEA recordings once, well: Doorn,
van Putten and Frega (*Communications Biology*, 2025) trained a masked
autoregressive flow on 300,000 simulations of a 100-neuron Hodgkin-Huxley
network and recovered ten parameters of human iPSC-derived networks. Their code
and trained estimator are public, and both are used here: their simulator is
the starting point of this one, and their estimator is scored beside it.

| Their stated limitation | Here |
|---|---|
| Excitatory neurons only, so GABAergic compounds are out of reach | inhibitory population, synaptic and tonic GABA-A; gabazine and GABA are scored |
| Posteriors comparable only within one MEA experiment | the shift is inferred within a well, from a paired design that holds wiring and recording nuisances fixed |
| Misspecification noted, not handled | a guard with two tests that refuses to name a mechanism, with thresholds fixed on held-out simulations |
| One population, no device geometry | a two-compartment chip twin with directional channels and four readouts |
| One recording system | recording systems as explicit views; one twin per system, trained on the domain its own untreated baselines span |

Other entries to this challenge that work on neural MEA data describe
recordings (quality control, feature fingerprints, forecasts); none infers a
mechanism with a model that can be re-simulated.

## 4. Data

| Dataset | What it is | Role | Licence |
|---|---|---|---|
| Tampere comparative MEA (Hyvärinen et al., *Sci Data* 2022) | rat cortical DIV 22 and hPSC-derived DIV 29 plates; CNQX, D-AP5, GABA, gabazine, kainic acid, TTX, vehicle; 16 electrodes | development set (scored in version 1) | CC BY 4.0 |
| Doorn et al., *Stem Cell Rep.* 2024 | human iPSC Ngn2 neurons with rat astrocytes, 10 uM Dynasore, 10 wells; MCS 24-well, 12 electrodes | **blind test** of version 2 | Apache-2.0 repository |
| Mateus et al., bioRxiv 2024 | rat hippocampal neurons in two-compartment chips with straight, Tesla, Tesla v2, Rams and Arrows microchannels; MCS 256-electrode | **blind test** of the chip readout prediction | CC BY-NC-ND (research use; read in place, not redistributed) |
| Doorn et al., *Commun Biol* 2025 | trained estimator and feature code | prior art, scored beside the twin | Apache-2.0 |

No restricted, clinical or personal data is used. Every file the results depend
on is checked against its SHA-256 by `scripts/fetch_tampere.py` and
`scripts/fetch_external.py`.

## 5. The model

### 5.1 The network

256 conductance-based neurons on a 16 x 16 grid at 45 um, with Traub-Miles
sodium and potassium currents, a slow calcium-dependent after-hyperpolarisation
and membrane noise. A fraction of neurons is inhibitory. Synapses are AMPA,
NMDA with its magnesium block, and GABA-A; bath-applied GABA acts through a
tonic GABA-A conductance on every cell. Every terminal carries short-term
depression and, after Doorn et al., asynchronous release: each spike raises a
release rate that decays over 700 ms, and each asynchronous release transmits
and depletes the vesicle pool without a somatic spike. Conduction delays scale
with distance.

Sixteen parameters, ten of which a compound may move:

{{PARAM_TABLE}}

### 5.2 Implementation

One CUDA block simulates one network and one thread one neuron, with the whole
time loop in shared memory. Spiking neurons are collected with a warp ballot and
read back in neuron order, so floating-point sums never reorder and a run is
bit-reproducible; `test_simulator_is_deterministic` holds that. On one RTX 4070
the kernel simulates about {{NET_PER_S}} networks of 65 simulated seconds per
second.

### 5.3 Recording systems

A real array reports what its electrodes see after its own detection. The
kernel records every electrode at 0.2 ms; each recording system is a view that
applies its electrode layout and dead time to recorded and simulated events
alike: the Axion 48-well plate of the Tampere data (16 electrodes, 2 ms) and the
MCS 24-well plate of the Doorn data (a 4 x 4 grid whose four corners are
reference electrodes, 0.3 ms, the shortest interval in their peak trains).

### 5.4 The domain each twin covers

Most of the prior produces cultures that are silent, saturated or driven by
membrane noise, in which blocking a synapse changes nothing. Each recording
system's domain is read from its own untreated baselines: the 2nd to 98th
percentile of eight detection-robust statistics (rate, burst rate and duration,
spikes in bursts, correlation, spike-time tiling, burst participation,
inter-burst rate), widened, plus a living, network-driven culture and the
pharmacological definition of a cortical culture, that blocking AMPA collapses
its activity below 35 percent. A population search, then a nearest-neighbour
step that centres the proposal on admissible simulated cultures closest to each
recorded baseline window, finds where the simulator produces that domain. No
treated recording and no compound label enters it (Figure 2).

![Figure 2. Recorded baselines against the simulated cultures admitted to each domain.](../results/v2/figures/domain.png)

### 5.5 The sparse prior over what a compound does

A compound moves one to three mechanisms, rarely all. For each shiftable
parameter the prior is a narrow component near zero, and for a small active set
a wide component: fold changes from 1.5 to 25 for conductances, log-uniform
magnitudes for the linear parameters. The direction of each active shift is
drawn among the directions the baseline leaves room for, and a shift that the
prior bounds would clip away is labelled inactive: in version 1 that label noise
covered 28 percent of active tonic-inhibition labels and 18 percent of active
AMPA labels.

### 5.6 Inference

The bank holds {{BANK_PAIRS}} simulated pairs, each a culture simulated before
and after a shift with its wiring and recording nuisances held fixed. Five
presence heads read both recordings and their difference and give the
probability each mechanism moved; they are averaged and calibrated on held-out
cultures. A masked autoregressive flow, conditioned on the recordings and on the
active set, gives the culture's parameters and the shift: sampling the active
set from the presence probabilities gives the joint posterior, and fixing it to
one mechanism gives the effect of that mechanism if it is the one that moved.
Training, validation and calibration are split by simulated culture.

### 5.7 The guard

A posterior is worth reading only if the twin can produce the recording.
Typicality is the mean distance of the pair, in the twin's standardised
statistics, to its ten nearest training simulations. The predictive check
re-simulates 48 posterior draws and counts the statistics that fall outside
their predictive band. Each threshold is the 97.5th percentile over held-out
simulations, where the model is right by construction; either firing means the
recording is outside the model and no mechanism is named.

## 6. How it was evaluated, and what was scored more than once

Two pre-registrations. The first fixed the Tampere answer key, metric and
failure conditions; the pre-registered model reached 0.091, which is chance.
Three corrections driven by label-free checks followed, and the Tampere plates
were scored four times in version 1. They are therefore reported here as a
development set, scored a fifth time by version 2, and no claim of blind
performance rests on them.

The second pre-registration was written and hashed after version 2 was frozen
on simulation evidence, and before any version 2 output on the two blind
datasets existed. The code enforces the order: the Doorn reader refuses to
return a treated window, and the evaluation refuses to run, unless the second
pre-registration matches its recorded hash and every frozen model matches the
digest it lists. What was known about the blind data beforehand is listed in
it: the published summary statistics, and the untreated Doorn baselines that
set the domain.

## 7. Results

### 7.1 Blind test: Dynasore on human iPSC networks from another laboratory

Dynasore inhibits dynamin, slows vesicle recycling and strengthens short-term
depression; Doorn et al. model it as an increase of the depression parameter U.
Both `u_rel` (U) and `tau_d` (vesicle recovery) are accepted, direction up.
Chance is 2 of 10.

{{DOORN}}

![Figure 3. Blind test: presence probability per well and mechanism.](../results/v2/figures/doorn_wells.png)

### 7.2 The chip readout prediction, on recorded microchannel chips

{{CHIPS}}

![Figure 4. What each chip readout recovers, in simulation.](../results/v2/figures/chip_readouts.png)

![Figure 5. The prediction tested on recorded chips.](../results/v2/figures/mateus_chips.png)

### 7.3 Development set: Tampere, fifth scoring

{{TAMPERE}}

![Figure 6. Tampere development set.](../results/v2/figures/tampere_confusion.png)

### 7.4 Does the mechanistic reading transfer across species?

A supervised classifier given the labels will fit one plate. The question that
separates a mechanistic reading from a fingerprint is whether it transfers: a
nearest-centroid classifier over compounds is fitted on every well of one
species and scored on the other, once on the raw feature change and once on the
twin's reading of it.

{{TRANSFER}}

![Figure 7. Transfer across species.](../results/v2/figures/transfer.png)

### 7.5 Simulations and calibration

{{SIMULATION}}

![Figure 8. Held-out simulations.](../results/v2/figures/simulation.png)

### 7.6 The guard

{{GUARD}}

![Figure 9. The guard.](../results/v2/figures/guard.png)

### 7.7 Does the twin reproduce known pharmacology?

A twin that cannot represent a compound cannot attribute a recording to it.
Saturating blocks and agonists are applied to cultures the bank admitted, and
the change in firing is compared with the published direction and rough size.

{{PHARMACOLOGY}}

![Figure 10. Pharmacology check.](../results/v2/figures/pharmacology.png)

## 8. The chip twin as a planning tool

{{CHIP_DESIGN}}

## 9. Reliability and limitations

{{LIMITATIONS}}

## 10. Impact

{{IMPACT}}

## 11. Reproduction

```bash
pip install -r requirements.txt
python demo.py                       # bundled pairs, CPU only
python demo.py --serve               # web application on 127.0.0.1:8000
python -m pytest tests/ -q           # {{N_TESTS}} tests
python scripts/verify.py             # hashes, checksums, tests, written numbers
```

Full rebuild on a CUDA device, each step skipped when its output exists:

```bash
pip install -r requirements-gpu.txt
python scripts/fetch_tampere.py && python scripts/fetch_external.py
python scripts/run_all.py
```

`run_all.py` fits the domains, builds the bank, trains the twins, runs the
pharmacology check and the chip study, and, once the pre-registration exists,
the evaluation, the chip check, the figures and this report. The prior-art
estimator runs in its own environment (`requirements-doorn.txt`). Hardware used:
one RTX 4070 12 GB, 32 GB RAM, six-core CPU. No cloud service, paid API or
proprietary model is required.

## 12. Team, sources and licences

Marc Donovici, solo entrant, background in audit and applied machine learning.
The entry does not claim the cross-disciplinary bonus.

**Data.** Tampere comparative MEA dataset, CC BY 4.0. Doorn et al. peak trains
and estimator, Apache-2.0. Mateus et al. chip recordings, CC BY-NC-ND, used for
research, read in place and not redistributed.

**Software.** Python, NumPy, SciPy, pandas, PyTorch, zuko, CuPy, numba, h5py,
scikit-learn, FastAPI, Matplotlib, Playwright, pytest; sbi 0.21 and brian2 for
the prior-art estimator only. All open source under permissive licences.

**Models.** Every model in the repository was trained here, from simulations
generated here. No pretrained model, foundation model, commercial API or cloud
service is part of the system.

**References.**
1. Doorn N., van Putten M., Frega M. Automated inference of disease mechanisms in patient-hiPSC-derived neuronal networks. *Commun Biol*, 2025.
2. Doorn N., Voogd E., Levers M., van Putten M., Frega M. Breaking the burst: unveiling mechanisms behind fragmented network bursts in patient-derived neurons. *Stem Cell Rep* 19:1583, 2024.
3. Hyvärinen T. et al. Comparative microelectrode array data of the functional development of hPSC-derived and rat neuronal networks. *Sci Data* 9:118, 2022.
4. Mateus J., Melo P., Aroso M., Charlot B., Aguiar P. Influence of asymmetric microchannels in the structure and function of engineered neuronal circuits. bioRxiv 10.1101/2024.07.09.602729, 2024.
5. Peyrin J.-M. et al. Axon diodes for the reconstruction of oriented neuronal networks in microfluidic chambers. *Lab Chip* 11:3663, 2011.
6. Lassus B. et al. Glutamatergic and dopaminergic modulation of cortico-striatal circuits probed by dynamic calcium imaging of networks reconstructed in microfluidic chips. *Sci Rep* 8:17461, 2018.
7. Papamakarios G., Pavlakou T., Murray I. Masked autoregressive flow for density estimation. *NeurIPS*, 2017.
8. Cranmer K., Brehmer J., Louppe G. The frontier of simulation-based inference. *PNAS* 117:30055, 2020.
9. Tsodyks M., Markram H. The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS* 94:719, 1997.
10. Cutts C., Eglen S. Detecting pairwise correlations in spike trains: an objective comparison of methods. *J Neurosci* 34:14288, 2014.
