# Hodgkin's Razor: a mechanistic digital twin for neural organ-on-chip recordings

## Category: End-to-End System

## Demo video

<VIDEO_URL_YOUTUBE> · <VIDEO_URL_BILIBILI> (5 minutes, English and Chinese subtitles)

## Code repository

https://github.com/Marc-Dvci/Hodgkin-s-Razor (Apache-2.0)

## Live demo

https://marc-dvci.github.io/Hodgkin-s-Razor/ (static site, no login, works on a phone) · Notebook: [open in Colab](https://colab.research.google.com/github/Marc-Dvci/Hodgkin-s-Razor/blob/main/notebooks/hodgkins_razor_demo.ipynb) (CPU only; it clones the repository and re-checks the pre-registration hashes and scored results)

The page has three views: **Analyse a recording pair** (the full mechanism
report on bundled recordings, with the measured raster beside the re-simulated
twin), **Blind test** (every preparation of the pre-registered test, with the
method, the untreated group and the mechanism switchable, each AUROC recomputed
in the browser from the scored values) and **Chip planner** (which readout
resolves which chip property, and the power for a given number of chips).

![Blind test: the twin's NMDA reading against untreated sisters, and the same preparations read by two comparators](https://raw.githubusercontent.com/Marc-Dvci/Hodgkin-s-Razor/main/results/v3/figures/charlesworth_primary.png)

## Project summary

Neural organ-on-chip platforms record every network burst of a living culture
on a microelectrode array, but the analysis ends in a description: firing
fell, bursts shortened. It cannot tell an AMPA-receptor blocker from a
sodium-channel blocker. Hodgkin's Razor turns the recording into a mechanism
hypothesis. It fits a GPU-simulated biophysical network to a baseline and a
treated recording and returns, for ten molecular mechanisms, the probability
that each moved, the size of the shift and an interval. When no fitted model
reproduces the data, it refuses. It is trained only on simulated culture pairs
and has never seen a compound label.

Every test was pre-registered, its hash committed before the blind data was
read, and scored against untreated recordings read the same way.

On a blind test from another laboratory (Charlesworth et al. 2015: mouse
cultures kept on the NMDA blocker APV, against untreated sisters), the twin's
NMDA probability separates 29 treated from 23 untreated preparations: AUROC
0.86 [0.74, 0.96], against a pre-registered bar of 0.70. The published
estimator it builds on scores 0.49 on the same preparations. Of the ten
mechanisms, NMDA separates best, and the reading fades as the cultures
compensate, as the authors reported (p = 3e-5). On an earlier blind test
(Dynasore, human iPSC networks), it detected the drug in 9 of 10 wells with no
false call on the untreated pairs, and named the right mechanism class in 8.

Naming the single exact mechanism is the weak link (2/29, 2/10); both failures
are reported in full.

For chip design, the same simulator says before the experiment which readout
resolves which chip property and how many chips a claim needs: about 20 per
design, against 8 in the only public test.

### 项目摘要（中文）

神经器官芯片通过微电极阵列记录活体培养物的每一次网络爆发，但分析最终只停留在描述：放电减少、爆发变短。它无法区分 AMPA 受体阻断剂与钠通道阻断剂。Hodgkin's Razor 把记录变成机制假设：它将 GPU 模拟的生物物理神经网络拟合到一段基线记录和一段处理后记录上，对十种分子机制给出每种机制发生变化的概率、变化幅度及区间；当没有任何拟合模型能再现数据时，它拒绝给出结论。模型只用模拟的培养物配对训练，从未见过任何化合物标签。

每项测试都经过预注册，在读取盲测数据之前提交哈希，并与以同样方式读取的未处理记录对照评分。

在一项来自另一实验室的盲测中（Charlesworth 等，2015：长期施加 NMDA 阻断剂 APV 的小鼠培养物与未处理的姊妹培养物），孪生模型的 NMDA 概率把 29 份处理过的制备与 23 份未处理制备区分开：AUROC 0.86 [0.74, 0.96]，预设门槛为 0.70。它所基于的已发表估计器在同一批制备上仅得 0.49。在十种机制中，NMDA 的区分能力最强；随着培养物发生代偿，这一读数逐渐减弱，与原作者的报告一致（p = 3e-5）。在此前的盲测中（Dynasore，人源 iPSC 网络），模型在 10 个孔中检出 9 个药物效应，未处理配对中没有误报，机制类别正确 8 个。

精确命名单一机制是薄弱环节（2/29，2/10），两项失败均完整报告。

在芯片设计方面，同一个模拟器在实验之前就能指出哪种读数可以分辨哪种芯片性质，以及一个结论需要多少块芯片：每种设计约 20 块，而唯一的公开测试只有 8 块。

## Technical report

Full report (PDF): https://github.com/Marc-Dvci/Hodgkin-s-Razor/blob/main/docs/TECHNICAL_REPORT.pdf · Markdown: `docs/TECHNICAL_REPORT.md` in the repository. Every number below is read from `results/v3/RESULTS.md` and `results/v2/RESULTS.md`.

### Problem and application

Neural organ-on-chip platforms are meant to replace animal experiments in
neurotoxicity and neuropharmacology. Their richest readout is extracellular
electrophysiology, and its analysis stops at description. In safety
pharmacology, a compound that silences a network through sodium channels
carries a different liability from one that acts on excitatory transmission,
and a rate plot cannot tell them apart.

The regulatory door is open. The US FDA Modernization Act 2.0 (December 2022)
removed the statutory requirement for animal testing before human trials and
named cell-based assays, microphysiological systems and computer models among
the alternatives. The OECD in vitro battery for developmental neurotoxicity
already includes a microelectrode-array network-formation assay. What such an
assay still lacks is a readout a reviewer can weigh: a named mechanism, with
the rate at which the same reading appears when nothing was applied.

In practice, for a laboratory screening compounds on neural chips:

1. **Before plating:** the chip planner says which electrodes resolve the
   property under test and how many chips per design reach 80% power.
2. **After recording:** each compound gets a ranked mechanism profile against
   the plate's own untreated chips, or a refusal that flags it for follow-up.
3. **When two mechanisms tie:** the twin ranks follow-up recordings by
   simulation (tested, and reported as not yet better than a fixed protocol).

The supporting organisation's stated aim is organ-on-chip digital twins that
move the field from experimental description toward predictive simulation.
This project builds that for the two questions a laboratory asks: what did the
compound do, and what should be measured next?

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
   and how many chips are needed. A follow-up recommender, for when two
   mechanisms tie, is tested on simulations: it beats repeating the recording
   (0.54 against 0.42 of ties, p = 0.04), but not a fixed protocol (0.47,
   p = 0.11).

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
- Post hoc (these wells are no longer blind), read with the v3 primary: the
  accepted mechanisms' probability separates treated wells from their own
  untreated pairs with AUROC 0.87. A twin retrained with the measured drift
  scores 0.90.

**Development set (Tampere, scored five times, not blind).**

- 0/11 vehicle wells called; treated-against-vehicle detection AUROC 0.89.
- Rat: 16/42 named.
- Human: 27/28 wells flagged as outside the model. Post hoc, a twin trained
  on a human-only domain names 6/24 (the frozen twin names 0/24; p = 0.06
  against chance). Its guard still flags 27/28.
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
- The next-experiment recommender does not yet beat a fixed follow-up protocol.
- The blind tests are on conventional MEA cultures. The one recorded chip set
  is underpowered for the question asked of it.

## Team

Marc Donovici, solo entrant: audit, and applied machine learning, including
earlier competition work on brain-imaging and brain-decoding data.
