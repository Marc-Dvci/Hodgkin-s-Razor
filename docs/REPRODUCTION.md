# Reproduction

Three levels, depending on what you have.

## 1. No GPU, five minutes

```bash
pip install -r requirements.txt
python demo.py
```

Prints, for every bundled recording pair, the mechanism the twin named and
whether it matches the published pharmacology. The analyses were computed by
the twin in this repository and cached into `data/examples/*.json`.

```bash
python demo.py --serve          # the web application on 127.0.0.1:8000
python -m pytest tests/ -q      # the test suite
python scripts/verify.py        # checks hashes, checksums, tests and figures
```

A notebook covering the same ground is in
`notebooks/hodgkins_razor_demo.ipynb`, and a static site that needs no Python
at all is produced by `python scripts/export_static.py`.

## 2. A CUDA device, about three hours

```bash
pip install -r requirements.txt -r requirements-gpu.txt
python scripts/fetch_tampere.py      # public recordings, CC BY 4.0, checksummed
python scripts/run_all.py --with-bank
```

Stages, in order, each skipped when its output already exists:

| Stage | What it does | Time |
|---|---|---|
| `fit_regime.py` | screens the prior, fits the proposal over parameters | 10 min |
| `make_bank.py` | 110 000 paired simulations | 100 min |
| `train.py` | the flow and the presence head | 2 min |
| `train.py --unpaired` | the ablation baseline | 3 min |
| `evaluate.py` | every pre-registered test | 40 min |
| `render_results.py` | `results/RESULTS.md` | seconds |
| `chip_study.py` | what each chip readout resolves | 30 min |
| `make_examples.py --cache` | bundled pairs with their analyses | 5 min |
| `figures.py` | every figure in the report | 2 min |
| `render_report.py` | the technical report and the writeup | seconds |
| `export_static.py` | the server-free site | seconds |
| `verify.py` | re-checks everything above | 1 min |

Measured on one RTX 4070 12 GB with a six-core CPU.

## 3. Docker

```bash
docker build -t hodgkins-razor .
docker run -p 8000:8000 hodgkins-razor            # the application, CPU only

docker build -f Dockerfile.gpu -t hodgkins-razor-gpu .
docker run --gpus all hodgkins-razor-gpu python3 scripts/run_all.py --with-bank
```

## What is and is not in the repository

Present: the simulator, the inference code, the trained models, the bundled
recording extracts with their analyses, every script, the tests, the figures
and the written results.

Absent by design: the simulation bank, which is 110 000 pairs and is rebuilt by
one command. The simulation is bit-reproducible, so rebuilding from the same
seed gives the same bank.

## If a number disagrees

`scripts/verify.py` compares the headline numbers written in
`results/RESULTS.md` against `results/results.json`, and re-checks the
pre-registration against its recorded hash. A mismatch fails the run rather
than passing quietly.
