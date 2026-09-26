# Hodgkin's Razor

**Name the molecular mechanism a compound moved, from two recordings of one well.**

AI4S Open Innovation: AI for Life Science, 5th Pazhou Algorithm Competition.
Category: **End-to-End System**. Apache-2.0.

A microelectrode array on a neural organ-on-chip produces a rich recording and
a thin answer. Standard analysis reports that firing fell by 71 percent and
bursting stopped. It does not say whether the compound blocked AMPA receptors,
opened chloride channels or shut down sodium channels, and those three have
different consequences for a drug programme.

Hodgkin's Razor fits a conductance-based network model to the recording itself.
It reads a baseline and a treated recording of the same well and returns the
parameter the compound moved, with an effect size, a credible interval and a
probability. When the fitted model cannot reproduce the recording, it says so
and names no mechanism.

The razor is the sparse prior: among the mechanisms that could explain the
change, prefer the fewest.

---

## The headline result

| | |
|---|---|
| Top-1 mechanism accuracy | see `results/RESULTS.md` |
| Chance | 0.100, over ten mechanisms |
| Wells scored | rat cortical DIV 22 and human iPSC DIV 29 |
| Compound labels seen in training | **none** |

The model is trained only on simulations. No recording, and no compound label,
enters training, feature selection, thresholds or any hyperparameter.

The recorded set was scored four times: 0.091 as pre-registered, then 0.167,
then 0.303, and 0.121 for a variant that was tried and dropped. Each correction
in between was driven by a check that uses no compound label. Simulation
evidence does not separate the last two models, so **the reported 0.303 carries
selection on the recorded set and the blind number is 0.091**. The whole
sequence, and why each change was made, is in
[section 5.1 and section 6b of the report](docs/TECHNICAL_REPORT.md). The
answer key, the primary metric, the success threshold and the failure
conditions were written and hashed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any trained model was scored
against a recording.

---

## Quick start

```bash
pip install -r requirements.txt          # CPU is enough for the demo
python demo.py                           # bundled recordings, cached analyses
python demo.py --serve                   # web application on 127.0.0.1:8000
```

With a CUDA device, everything re-runs from scratch:

```bash
pip install -r requirements-gpu.txt
python scripts/fetch_tampere.py     # public recordings, CC BY 4.0
python scripts/run_all.py --with-bank
```

`run_all.py` screens the prior, builds the simulation bank, trains the twin and
the unpaired baseline, runs every pre-registered test, studies the chip
readouts, and writes the results, the figures and the report. Each stage is
skipped if its output exists, so an interrupted run resumes. The individual
commands are listed in the technical report.

Docker:

```bash
docker build -t hodgkins-razor .                 # CPU: demo, app, tests
docker run -p 8000:8000 hodgkins-razor
docker build -f Dockerfile.gpu -t hodgkins-razor-gpu .   # simulation, training
```

---

## How it works

```
 baseline + treated recording of one well
              |
   [ 40 summary statistics ]   one function, used for simulation and experiment alike
              |
   [ conditional normalising flow ]    joint posterior over culture parameters
              |                        and the shift the compound caused
              +--> presence head       probability that each mechanism moved
              |
   [ posterior predictive check ]      re-simulate at the fitted parameters
              |
       inside the model? ---- no ----> report no mechanism
              | yes
       mechanism report: target, direction, fold change, interval, probability
```

### The simulator

A conductance-based network of 256 Hodgkin-Huxley neurons on a 16 x 16 grid,
read by 16 electrodes. Excitatory and inhibitory populations, AMPA, NMDA and
GABA-A synapses, short-term depression, slow after-hyperpolarisation and
distance-dependent conduction delays.

It runs one network per CUDA block with the whole time loop inside shared
memory: **about 35 networks per second of 65 simulated seconds on one RTX 4070**,
so a bank of 200,000 recordings is built in under two hours on a desktop rather
than on a cluster.

The simulation is **bit-reproducible**. Spikes are collected with a warp ballot
and read back in neuron order, so the floating-point sums never reorder between
runs. `tests/test_core.py::test_simulator_is_deterministic` asserts it.

### The domain the twin covers

Only about one parameter set in ten produces a living, network-driven culture.
In the rest, activity comes from membrane noise rather than from the network's
own synapses, so blocking a synaptic conductance changes nothing and no method
could recover what a compound did. `scripts/fit_regime.py` measures that rate,
fits a proposal over parameters, and the bank is built on parameter sets whose
simulated baseline passes the criterion:

| | |
|---|---|
| firing rate | 0.3 to 40 events/s/electrode |
| active electrodes | at least 60 percent |
| network bursts | at least 1 per minute |
| spikes inside bursts | at least 5 percent |

The same criterion applied to the recorded baselines puts 70 percent of the rat
windows and 99 percent of the human windows inside the twin's domain, spread
evenly across compounds. Recordings outside it are still scored, and the guard
is what reports them.

### The model parameters

Fourteen parameters, of which nine can be moved by a compound:

| Shiftable | Nuisance, fixed within a pair |
|---|---|
| Na, Kdr, slow-AHP conductance | membrane noise |
| AMPA, NMDA, GABA-A conductance | connection probability |
| vesicle recovery time, release fraction | inhibitory fraction |
| tonic drive | event detection fraction, electrode pickup spread |

Wiring and recording nuisances are properties of the well, so a wash-on cannot
change them. Holding them fixed across the pair is what removes the culture and
plate offsets that otherwise dominate a comparison between two recordings.

Every prior range was set by measuring it: `scripts/sensitivity.py` sweeps each
parameter and reports whether the recording actually responds over that range.
The sodium range was widened after this sweep showed a block was unreachable.

The feature set was chosen the same way. Removing excitatory drive and adding
inhibition both lower the firing rate, and the descriptors in common use do not
separate them. Eighteen further statistics were added for that job: spike-time
tiling, the rate that survives between bursts, burst participation, onset
jitter, burst shape, Fano factors and population autocorrelation at three
timescales each, and rate-robust interval statistics. On simulations where the
answer is known, they raise separation among the four receptor and channel
mechanisms from 0.57 to 0.64.

### The razor

A compound acts on one or two targets. The prior over the shift is sparse: a
narrow component near zero for every mechanism, and a wide component for a
small active set. Two heads read the recording, because size and presence are
different questions:

* a **masked autoregressive flow** over the joint vector of baseline parameters
  and shift, which gives effect sizes with intervals and lets the twin be
  re-simulated;
* a **presence head** giving, per mechanism, the calibrated probability that it
  was in the active set at all.

### The guard

A posterior is only worth reading if the twin, run at those parameters,
produces a recording like the measured one. The discrepancy threshold is the
95th percentile over held-out simulation records, fixed before any recording is
scored. The guard is tested on cases it must catch:

* simulations from a variant simulator whose receptor kinetics were changed, a
  change no parameter of the twin can produce;
* real recordings whose spike times were circularly shifted per electrode,
  which preserves every per-electrode rate and destroys the network structure.

Both are reported in `results/RESULTS.md` whatever they came out at.

### The chip twin

The same kernel runs a two-compartment device with asymmetric microchannels,
the geometry used for directional cortico-striatal circuits, plus a population
calcium observation model at 2 Hz. `scripts/chip_study.py` asks what each
readout can resolve: which chip parameters an electrode array recovers, which
a calcium movie recovers, and which neither does. That is the question a
laboratory faces before it runs the experiment, not after.

---

## Repository layout

```
hodgkins_razor/
  params.py       the parameter table: one definition, read by everything
  kernel.cu       the CUDA network kernel
  simulator.py    batched simulator, paired mode, observation model
  features.py     the 40 summary statistics, one path for simulation and experiment
  shift.py        the sparse prior over what a compound does
  nde.py          flow, presence head, and the unpaired baseline
  ppc.py          posterior predictive check and the guard
  chip.py         two-compartment chip and the calcium readout
  tampere.py      reader for the public recordings
  report.py       the structured mechanism report
scripts/
  fetch_tampere.py  make_bank.py  train.py  evaluate.py
  sensitivity.py    chip_study.py  figures.py  make_examples.py
app/              FastAPI service and the single-page interface
tests/            pytest suite
docs/             technical report
```

Entry points: `demo.py` to see it work, `scripts/evaluate.py` to reproduce
every number, `app/server.py` for the interface.

---

## Data

| Dataset | Use | Licence |
|---|---|---|
| [Tampere comparative MEA dataset](https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset) | every recorded result | CC BY 4.0 |
| [Doorn et al. 2025 SBI repository](https://gitlab.utwente.nl/m7706783/SBI_MEA_model) | prior art, comparison | Apache-2.0 |

The Tampere pharmacology plates give matched baseline, treated and TTX
recordings of the same wells: rat cortical neurons at DIV 22 and human
pluripotent-stem-cell-derived neurons at DIV 29, with CNQX, D-AP5, GABA,
gabazine, kainic acid and vehicle controls. `scripts/fetch_tampere.py`
downloads them and writes a SHA-256 for every file.

No restricted, clinical or personal data is used anywhere in this project.

---

## Prior art, and what is new here

Doorn, van Putten and Frega, *Communications Biology* 2025, applied
simulation-based inference to MEA recordings of human iPSC networks, with code
and a trained estimator released publicly. That work is the closest prior art
and the baseline here. Their stated limitations, and what this project does
about each:

| Their limitation | Here |
|---|---|
| Excitatory neurons only, so GABAergic compounds cannot be read | inhibitory population and a GABA-A conductance, so gabazine and GABA are in scope |
| Posteriors comparable only within one experiment | the shift is inferred within a well, so plate and culture offsets cancel |
| Model misspecification noted, not handled | a calibrated guard that refuses to name a mechanism, tested on cases it must catch |
| One population, no device geometry | two-compartment chip twin with directional channels and a calcium readout |
| Effect size only | a separate calibrated presence probability per mechanism |

The **unpaired baseline** in `scripts/train.py --unpaired` is the same
simulator, features, bank and flow size with only the paired design removed. It
isolates what the pairing is worth. A **supervised classifier** that is given
the compound labels the twin never sees is reported alongside, whether or not
it wins.

---

## Reproducing the results

```bash
python -m pytest tests/ -q          # 27 tests
python scripts/verify.py            # hashes, checksums, tests, written numbers
python scripts/evaluate.py          # writes results/results.json
python scripts/figures.py           # writes results/figures/*.png
```

Full instructions, with timings for each stage, are in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

`scripts/evaluate.py` verifies `PREREGISTRATION.md` against its recorded hash
before it runs and refuses to proceed if the file changed.

Hardware used: one RTX 4070 12 GB, 32 GB RAM, six-core CPU. No cloud service,
no paid API and no proprietary model is required at any point.

---

## Limitations

* Every recorded result comes from two plates in one published dataset. Two
  species and two independent cultures are not a multi-laboratory validation.
* One concentration per compound, so no dose-response in parameter space is
  reported. The multi-laboratory HESI dataset that would supply it sits behind
  a login.
* The twin carries the mechanisms its equations carry. A compound acting
  through anything else is reported as outside the model, which is the correct
  answer but not an informative one.
* The chip twin reproduces the published direction of cortico-striatal
  pharmacology and studies what each readout resolves. It has not been fitted
  to a recording from a physical two-compartment device, because no such
  recording is public.

---

## Documents

| | |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | the answer key, metric and failure conditions, hashed before scoring |
| [`results/RESULTS.md`](results/RESULTS.md) | every pre-registered outcome |
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | the full report |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | how to rebuild everything |
| [`docs/DATA.md`](docs/DATA.md) | sources, licences and compliance |
| [`docs/SELF_AUDIT.md`](docs/SELF_AUDIT.md) | scored against the published criteria, with the open items |
| [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) | the demo script |

Author: Marc Donovici. Built with public data and open-source software only.
