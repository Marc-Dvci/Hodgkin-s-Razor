# Hodgkin's Razor

**A mechanistic digital twin for neural organ-on-chip recordings: from two recordings of one well, the molecular mechanism a compound moved, or a refusal.**

AI4S Open Innovation: AI for Life Science, 5th Pazhou Algorithm Competition.
Category: **End-to-End System**. Apache-2.0.

A microelectrode array under a neural organ-on-chip produces a rich recording
and a thin answer: firing fell, bursts shortened, synchrony dropped. It does not
say whether the compound blocked AMPA receptors, opened chloride channels or
slowed vesicle recycling, and those have different consequences for a drug
programme.

Hodgkin's Razor fits a biophysical network model to the recording itself. It
reads a baseline and a treated recording of the same well and returns, for ten
mechanisms, the probability that each moved, the size of the shift if it did,
and an interval. When no simulation resembles the recording, or the fitted
twin cannot reproduce it, it says so and names nothing. The razor is the
sparse prior: among the mechanisms that could explain a change, prefer the
fewest.

<!-- RESULTS -->

---

## Quick start

```bash
pip install -r requirements.txt          # CPU is enough
python demo.py                           # bundled pairs, cached analyses
python demo.py --serve                   # web application on 127.0.0.1:8000
python -m pytest tests/ -q
python scripts/verify.py                 # hashes, checksums, frozen models, written numbers
```

A static version of the application, with no server, is in `site/` and is
published by `.github/workflows/pages.yml`. A CPU notebook is in
`notebooks/hodgkins_razor_demo.ipynb`.

With a CUDA device, the whole pipeline rebuilds from public data:

```bash
pip install -r requirements-gpu.txt
python scripts/fetch_tampere.py          # Tampere recordings, CC BY 4.0, checksummed
python scripts/fetch_external.py         # Doorn et al. and Mateus et al., checksummed
python scripts/run_all.py
```

`docs/REPRODUCTION.md` lists every stage with its measured time.

---

## How it works

```
 baseline + treated recording of one well (Axion, MCS or a two-column CSV)
              |
   recording system: electrode layout and detection dead time,
   applied to the recording and to every simulation alike
              |
   40 summary statistics        one function for recorded and simulated events
              |
   5 presence heads             probability that each of 10 mechanisms moved
   conditional flow             culture parameters; size of each shift if it moved
              |
   guard: typicality + predictive check   (either firing -> "outside the model")
              |
   mechanism report             called mechanism, direction, effect, interval,
                                class, rivals, culture parameters, hash;
                                next experiment when two mechanisms tie
```

**The simulator.** 256 conductance-based neurons on a 16 x 16 grid: excitatory
and inhibitory populations, AMPA, NMDA and GABA-A synapses, a tonic GABA-A
conductance for bath agonists, short-term depression, asynchronous release
(after Doorn et al.), slow after-hyperpolarisation and distance-dependent
delays. One CUDA block per network, the whole time loop in shared memory, and
bit-reproducible: spikes are collected with a warp ballot and read back in
neuron order, so floating-point sums never reorder.

**Recording systems.** The kernel records at 0.2 ms. Each system is a view with
its own electrode layout and dead time: the Axion 48-well plate (16 electrodes,
2 ms) and the MCS 24-well plate (12 electrodes, 0.3 ms).

**The domain.** A twin is trained on the cultures its own recording system
produces: the range the system's untreated baselines span, a living
network-driven culture, and activity that collapses when AMPA is blocked.
`scripts/fit_domain.py` finds where the simulator produces that domain, from
baselines only.

**The prior.** A compound moves one to three mechanisms. Directions are drawn
where the baseline leaves room, and a shift the bounds would clip is labelled
inactive.

**Inference.** Five presence heads, averaged and calibrated on held-out
cultures, and a masked autoregressive flow conditioned on the recording and on
the active set, so each mechanism's effect is reported given that it moved.

**The guard.** Typicality: distance to the nearest training simulations, which
needs no GPU. Predictive check: re-simulate 48 posterior draws and count the
statistics outside their band. Thresholds at the 97.5th percentile of held-out
simulations.

**The chip twin.** The same kernel simulates two chambers joined by directional
microchannels and reads them four ways: chamber electrodes, 2 Hz calcium,
channel electrodes and a chamber perfusion. For each chip property it says
which readout resolves it, before the experiment is run.

---

## Validation

Two pre-registrations, both hashed before scoring.

* `PREREGISTRATION.md` (version 1) fixed the Tampere answer key. The
  pre-registered model scored at chance, and after three label-free corrections
  the plates had been scored four times. They are the **development set** now.
  The record is in `results/v1/` and the git tag `v1`.
* `PREREGISTRATION_v2.md` was written after version 2 was frozen on simulation
  evidence and before any version 2 output existed on two blind tests: Dynasore
  on human iPSC networks from another laboratory (Doorn et al. 2024), and the
  chip twin's readout prediction on recorded microchannel chips (Mateus et al.
  2024). `hodgkins_razor/doorn.py` refuses to return a treated Dynasore window
  until that file exists and matches its hash, and `scripts/evaluate_v2.py`
  checks it and every frozen model's digest before it runs.

Every outcome, including what failed, is in `results/v2/RESULTS.md`.

---

## Data

| Dataset | Use | Licence |
|---|---|---|
| [Tampere comparative MEA dataset](https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset) | development set; domain of the Axion twin | CC BY 4.0 |
| [Doorn et al. 2024, Dynasore peak trains](https://gitlab.utwente.nl/m7706783/fb_model) | blind test; domain of the MCS twin (baselines only) | Apache-2.0 |
| [Mateus et al. 2024, microchannel chips](https://zenodo.org/records/14525182) | blind test of the chip readout prediction | CC BY-NC-ND, read in place, not redistributed |
| [Doorn et al. 2025, SBI estimator](https://gitlab.utwente.nl/m7706783/SBI_MEA_model) | prior art, scored beside the twin | Apache-2.0 |

No restricted, clinical or personal data. `docs/DATA.md` has the details.

---

## Repository layout

```
hodgkins_razor/
  params.py      the parameter table, read by everything
  kernel.cu      the CUDA network kernel
  simulator.py   batched simulator, paired mode, recording-system views
  features.py    the 40 statistics, one path for simulation and experiment
  shift.py       the sparse prior over what a compound does
  nde.py         presence ensemble, conditional flow, unpaired baseline
  ppc.py         predictive check and typicality
  report.py      the mechanism report
  design.py      the next experiment, when two mechanisms tie
  chip.py        two-compartment chip twin and its four readouts
  tampere.py  doorn.py  mateus.py   readers for the three datasets
scripts/
  fit_domain.py make_bank.py train.py sim_eval.py guard_sim_check.py
  pharmacology_check.py chip_study.py freeze_v2.py
  evaluate_v2.py mateus_check.py prior_art_doorn.py
  render_v2.py figures_v2.py render_report_v2.py make_examples.py
  export_static.py make_notebook.py verify.py run_all.py
app/        FastAPI service and the single-page interface
site/       the same page as a static site
film/       the demo film, rendered from the running page and the results
tests/      pytest suite
docs/       technical report, data, reproduction, self-audit
```

The version 1 scripts (`evaluate.py`, `figures.py`, `render_results.py`,
`render_report.py`, `development.py`, `fit_regime.py`, `sensitivity.py`) are kept
for the record; they describe the version 1 model at the tag `v1`.

---

## Documents

| | |
|---|---|
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | the full report (PDF in `docs/`) |
| [`results/v2/RESULTS.md`](results/v2/RESULTS.md) | every pre-registered outcome |
| [`PREREGISTRATION_v2.md`](PREREGISTRATION_v2.md) | the blind tests, fixed before scoring |
| [`docs/DATA.md`](docs/DATA.md) | sources, licences, compliance |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | how to rebuild everything |
| [`docs/SELF_AUDIT.md`](docs/SELF_AUDIT.md) | scored against the published criteria, with the open items |

Author: Marc Donovici. Built with public data and open-source software only.
