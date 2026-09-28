# Hodgkin's Razor: a mechanistic digital twin for neural organ-on-chip recordings

## Category: End-to-End System

## Demo video

<VIDEO_URL_YOUTUBE> · <VIDEO_URL_BILIBILI> (5 minutes, English and Chinese subtitles)

## Code repository

https://github.com/Marc-Dvci/Hodgkin-s-Razor (Apache-2.0)

## Live demo

<PAGES_URL> (static site, no login) · Kaggle Notebook: <KAGGLE_NOTEBOOK_URL>

## Project summary

A microelectrode array under a neural organ-on-chip records every network burst
of a living culture. Analysis then reduces it to a description: firing fell,
bursts shortened. It does not say which molecular mechanism a compound moved.
Hodgkin's Razor fits a GPU-simulated biophysical network to the recording. For
ten mechanisms it returns the probability that each moved, the size of the
shift, and an interval, or it refuses when no fitted model reproduces the data.
It is trained only on simulations and has never seen a compound label.

Every test is scored against untreated recordings read the same way, because a
method's chance level is its own hit rate when nothing was applied. That rule
exposed a comparator whose 8/10 blind score was a preference: it gave the same
answer on 6 of 10 untreated pairs.

On a pre-registered blind test from another laboratory (Charlesworth et al.
2015), the twin's probability that NMDA moved separates cultures kept on an
NMDA blocker from untreated sister cultures of the same genotypes: AUROC 0.86
[0.74, 0.96], against a bar of 0.70. The published estimator it builds on
scores 0.49 on the same cultures. The reading fades as the cultures compensate,
as the authors reported (p = 3e-5). Naming NMDA as the single top mechanism
failed (2/29), and so did the exact-mechanism primary of an earlier Dynasore
test (2/10). There, the twin still detected the drug in 9/10 wells, with 0/10
false calls, and got the mechanism class in 8/10.

For chips, it says before the experiment which electrodes resolve
directionality and how many chips a claim needs: about 20 per design, against
8 in the only public test.

### 项目摘要（中文）

神经器官芯片下的微电极阵列记录下活体培养物的每一次网络爆发，但常规分析只能给出描述：放电减少、爆发变短。它无法说明化合物作用于哪一种分子机制。Hodgkin's Razor 将 GPU 模拟的生物物理神经网络拟合到记录本身：对十种机制，给出每种机制发生变化的概率、变化幅度及区间；当没有任何拟合模型能再现数据时，它拒绝给出结论。模型只用模拟数据训练，从未见过任何化合物标签。

每项测试都与以同样方式读取的未处理记录对照评分，因为一种方法的机会水平就是在未施药时它自身的命中率。这一规则揭示了一个对照方法的 8/10 盲测成绩其实是一种偏好：它在 10 对未处理样本中有 6 对给出了同样的答案。

在一项来自另一实验室的预注册盲测中（Charlesworth 等，2015），孪生模型对"NMDA 发生变化"给出的概率，能够把长期施加 NMDA 阻断剂的培养物与同基因型、未处理的姊妹培养物区分开：AUROC 0.86 [0.74, 0.96]，预设门槛为 0.70。它所基于的已发表估计器在同一批培养物上仅得 0.49。随着培养物发生代偿，这一读数逐渐减弱，与原作者的报告一致（p = 3e-5）。将 NMDA 列为唯一首位机制未达标（2/29），此前 Dynasore 盲测的精确机制主终点也未达标（2/10）。但在那项测试中，模型在 10 个孔中检出 9 个药物效应，10 对未处理样本中误报为 0，机制类别正确 8/10。

对于芯片，它在实验之前就能指出哪些电极可以分辨方向性，以及一个结论需要多少块芯片：每种设计约 20 块，而唯一的公开测试只有 8 块。

## Technical report

Full report (PDF): <REPORT_PDF_URL> · Markdown: `docs/TECHNICAL_REPORT.md` in the repository. Every number below is read from `results/v3/RESULTS.md` and `results/v2/RESULTS.md`.

### Problem and application

Neural organ-on-chip platforms are meant to replace animal experiments in
neurotoxicity and neuropharmacology. Their richest readout is extracellular
electrophysiology, and its analysis stops at description. In safety
pharmacology, a compound that silences a network through sodium channels
carries a different liability from one that acts on excitatory transmission,
and a rate plot cannot tell them apart. The supporting organisation's stated
aim is organ-on-chip digital twins that move the field from experimental
description toward predictive simulation. This project builds that for the two
questions a laboratory asks: what did the compound do, and what should be
measured next?

### Method

1. **Simulator.** 256 conductance-based neurons with AMPA, NMDA and GABA-A
   synapses, tonic inhibition, short-term depression, asynchronous release and
   slow after-hyperpolarisation. It runs on one CUDA block per network, is
   bit-reproducible, and has sixteen parameters, ten of which a compound may
   move.
2. **Recording systems as views.** Each array's electrode layout and detection
   dead time are applied to simulated and recorded events alike: Axion 48-well,
   MCS 24-well, and the MCS 60-electrode array read as quadrants.
3. **Domain.** Each twin is trained on the cultures its own recording system
   produces, read from untreated baselines only.
4. **Sparse prior.** A compound moves one to three mechanisms. Between two
   recordings, a no-compound drift is added, measured on untreated pairs.
5. **Inference.** Five presence heads give the probability that each mechanism
   moved. A conditional normalising flow gives the culture's parameters and the
   size of each shift if it moved. The bank holds 288,000 simulated sister
   pairs for version 3.
6. **Guard.** Typicality and a posterior predictive check. If either fires, no
   mechanism is named.
7. **Chip twin.** Two chambers joined by directional channels, four readouts,
   and a striatal target. It answers which readout resolves which property,
   how many chips are needed, and which follow-up compound resolves a tie.

**Evaluation discipline.** There were three pre-registrations, each hashed
before its blind data was read, and the readers refuse to return blind
recordings until the hash matches. A stop rule was committed before training:
the twin must pass the test on its own simulations before the test may be
frozen. The first twin failed it (0.792 against 0.80); after more simulations,
the second passed (0.810), and both attempts are published. Every test has an
untreated null group.

### Results

**Version 3 blind test (Charlesworth et al. 2015, chronic APV on sister arrays, 10 to 14 days in vitro).**

| Method, same preparations | AUROC of the NMDA reading, 29 treated vs 23 untreated (same genotypes) |
|---|---|
| **Hodgkin's Razor** | **0.86** [0.74, 0.96] (pre-registered bar 0.70: met) |
| Same twin, pairing removed | 0.66 [0.51, 0.81] |
| Doorn et al. 2025 estimator, run unchanged | 0.49 [0.33, 0.65] |

- Pooled over all genotypes (78 untreated preparations): 0.79. Wild type
  alone: 0.92.
- Of the ten mechanisms, NMDA separates treated from untreated best; `tau_d`
  is next, at 0.79.
- At 15 days and later the contrast vanishes (0.48), and the reading falls
  from early to late within the same treated preparations (p = 3e-5). This
  matches the canalization the original authors describe.
- **Not met:** NMDA as the single top mechanism (2/29 against 1/23), and no
  preparation reached p > 0.5.
- The guard passed held-out simulations (0.02 fire rate). It missed its two
  must-fire bars on sister pairs (0.58 and 0.63, against 0.80).

**Version 2 blind test (Doorn et al. 2024, Dynasore, 10 wells).**

- Primary, exact mechanism: 2/10, against a bar of 5. **Not met.**
- Drug called in 9/10 treated wells and in 0/10 untreated pairs of the same
  wells.
- Mechanism class correct in 8/10, against a chance rate of 0.30 (p = 0.0016).
- The unpaired comparator's 8/10 turned out to be a preference: 6/10 on
  untreated pairs.
- Diagnosis: the v2 bank assumed no drift between recordings. Measured drift
  is 0.03–0.04 of each parameter's range, and with none, 23% of recorded
  untreated differences fall outside the simulated band.

**Development set (Tampere, scored five times, not blind).**

- 0/11 vehicle wells called; treated-against-vehicle detection AUROC 0.89.
- Rat: 16/42 named.
- Human: 27/28 wells flagged as outside the model.
- Bath GABA is read as a sodium block. Simulations show why: both silence the
  culture, and a silent recording carries no signature.

**Chips.**
- Recorded microchannel chips (Mateus et al. 2024): the channel statistic
  scored AUROC 0.62 [0.28, 0.88]. That is inconclusive at 17 chips (power
  0.48); the twin says about 20 per design are needed.
- The sponsor laboratory's cortico-striatal NMDA result (Lassus et al. 2018):
  - The first committed prediction failed; the model's striatum was 91%
    self-driven.
  - The second changed one thing, imposing the paper's own statement that an
    isolated striatum is silent. Both synchrony readouts then fell in the
    published direction across 34 chips (p = 0.002 and 0.008).
  - Calcium-event frequency did not follow (p = 0.49).

### Data and compliance

| Dataset | Role | Licence |
|---|---|---|
| Charlesworth et al. 2015, Zenodo 31085 | blind test v3 | CC0 1.0 |
| Doorn et al. 2024, Dynasore peak trains | blind test v2 | Apache-2.0 |
| Mateus et al. 2024, Zenodo 14525182 | chip test | CC BY-NC-ND: read in place, not redistributed |
| Tampere comparative MEA (Sci Data 2022) | development set | CC BY 4.0 |
| Doorn et al. 2025 estimator | prior art | Apache-2.0 |

No clinical, personal or restricted data. Every input file is checksummed.

### Reproduction

```bash
pip install -r requirements.txt
python demo.py              # the pre-registered evidence, well by well, CPU
python demo.py --serve      # web application
python -m pytest tests/ -q
python scripts/verify.py
```

A full rebuild (`scripts/run_all.py`) needs one CUDA GPU and public data only.
It uses no paid API, cloud service or proprietary model.

### Limitations

- Naming the exact mechanism is the weak link. What holds up blind is
  detection, mechanism class, and one mechanism's ranking against untreated
  cultures.
- The guard is weaker on sister pairs than within a well.
- Human cultures are outside the current domain.
- A silenced culture cannot be read.
- The blind tests are on conventional MEA cultures. The one recorded chip set
  is underpowered for the question asked of it.

## Team

Marc Donovici, solo entrant: audit, and applied machine learning, including
earlier competition work on brain-imaging and brain-decoding data.
