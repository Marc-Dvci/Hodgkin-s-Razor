# Reproduction

Three levels, depending on what you have. Every model the results were scored
with is in `models/`, so the first level needs no training and no GPU.

## 1. No GPU

```bash
pip install -r requirements.txt
python demo.py                  # the pre-registered evidence, well by well
python demo.py --windows        # single windows of the bundled recordings
python demo.py --serve          # the web application on 127.0.0.1:8000
python -m pytest tests/ -q      # the test suite
python scripts/verify.py        # hashes, frozen models, written numbers, tests
```

`demo.py` reads the frozen evaluation outputs in `results/` and prints them at
the level they were scored: preparations and wells, with the untreated
comparison beside each treated one. `verify.py` checks that:
- each pre-registration matches its recorded hash;
- each frozen model and file matches the digest the pre-registration lists;
- the numbers written in `results/*/RESULTS.md` and the writeup match the JSON
  they came from;
- the tests pass.

A notebook covering the same ground is in `notebooks/hodgkins_razor_demo.ipynb`.
A static site that needs no Python at all is produced by
`python scripts/export_static.py`.

## 2. A CUDA device

```bash
pip install -r requirements.txt -r requirements-gpu.txt
python scripts/fetch_tampere.py      # Tampere recordings, CC BY 4.0, checksummed
python scripts/fetch_external.py     # Doorn, Mateus and Charlesworth et al., checksummed
python scripts/run_all.py
```

Each stage is skipped when its output exists, so an interrupted run resumes and
a fresh clone only rebuilds what is missing: the simulation banks, which are
not in the repository. The pipeline stops before scoring if a pre-registration
is missing. The prior-art estimator of Doorn et al. runs in its own
environment:

```bash
python -m venv .venv-doorn
.venv-doorn/Scripts/pip install -r requirements-doorn.txt   # sbi 0.21, brian2, torch 1.13 CPU
```

Stages measured on one RTX 4070 12 GB, 32 GB RAM, 12-thread CPU:

| Stage | Output | Measured |
|---|---|---|
| sister bank, 144,000 pairs (each of two) | `data/bank_v3`, `data/bank_v3b` | about 80 minutes each, alone on the machine |
| v3 twin, 288,000 pairs, five presence heads | `models/twin_v3_mcs60q` | 4 min 39 s (63 epochs) |
| v3 evaluation, section A (14,607 scored windows) | `results/v3/results.json` | 2 h 38 min |
| v3 evaluation, sections C and D (guard, simulations) | same file | 1 h 17 min |
| v3 evaluation, section B (unpaired twin, prior art) | same file | 26 min |
| Doorn et al. estimator on 362 sister pairs, CPU | `results/prior_art_doorn_charlesworth.json` | about 40 minutes |

Run the evaluation through `scripts/evaluate_v3_lowmem.py`. It calls the frozen
`scripts/evaluate_v3.py` with one sister pair in memory at a time. The frozen
script loads all 14,808 windows at once and ran a 32 GB machine out of memory.
The wrapper yields the same pairs in the same order (checked window by window).

## 3. Docker

```bash
docker build -t hodgkins-razor .
docker run -p 8000:8000 hodgkins-razor            # the application, CPU only

docker build -f Dockerfile.gpu -t hodgkins-razor-gpu .
docker run --gpus all hodgkins-razor-gpu python3 scripts/run_all.py
```

## What is and is not in the repository

**Present:**
- the simulator and the inference code;
- every trained model the results were scored with, including the first
  version 3 twin, which failed the stop rule;
- the bundled recording extracts with their analyses;
- every script, the tests, the figures and the written results.

**Absent by design:**
- the simulation banks, which are rebuilt by `run_all.py`. The simulator is
  bit-reproducible, so the same seed gives the same bank;
- the downloaded recordings, which are fetched and checksummed by the two
  fetch scripts. The Mateus et al. recordings may not be redistributed.

## If a number disagrees

`scripts/verify.py` fails the run on any mismatch between a written number and
the JSON it came from, and on any change to a frozen model or file. It does not
pass quietly.
