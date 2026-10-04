**Category: End-to-End System**

## Demo video

<VIDEO_URL_YOUTUBE> · <VIDEO_URL_BILIBILI> (4:32, narrated in English, English and Chinese subtitles)

## Code repository

https://github.com/Marc-Dvci/Hodgkin-s-Razor (Apache-2.0) · 中文说明: [README.zh.md](https://github.com/Marc-Dvci/Hodgkin-s-Razor/blob/main/README.zh.md)

## Live demo

https://marc-dvci.github.io/Hodgkin-s-Razor/ (static site, no login, works on a phone) · Notebook: [open in Colab](https://colab.research.google.com/github/Marc-Dvci/Hodgkin-s-Razor/blob/main/notebooks/hodgkins_razor_demo.ipynb) (CPU only; it clones the repository and re-checks the pre-registration hashes and scored results)

The page has three views: **Analyse a recording pair** (the full mechanism
report on bundled development recordings, each labelled with its compound's
target and what the twin returned, with the measured raster beside the
re-simulated twin), **Blind test** (every preparation of the pre-registered
test, with the method, the untreated group and the mechanism switchable, each
AUROC recomputed in the browser from the scored values) and **Chip planner**
(which readout resolves which chip property, and the power for a given number
of chips).

![Blind test: the twin's NMDA reading against untreated sisters, and the same preparations read by two comparators](https://raw.githubusercontent.com/Marc-Dvci/Hodgkin-s-Razor/main/results/v3/figures/charlesworth_primary.png)

## Project summary

Neural organ-on-chip platforms record every network burst of a living culture,
but the analysis ends in a description: firing fell, bursts shortened. That
cannot tell an AMPA-receptor blocker from a sodium-channel blocker, and a
laboratory screening compounds needs to know which. Hodgkin's Razor turns the
recording into a mechanism hypothesis. It fits a GPU-simulated biophysical
network to a baseline and a treated recording and returns, for ten molecular
mechanisms, the probability that each moved, the size of the shift and an
interval. When no fitted model reproduces the data, it refuses. It is trained
only on simulations and has never seen a compound label.

The question is harder than detecting that a compound acted: it asks which of
ten mechanisms moved. Every test was pre-registered, its hash committed before
the data were read, and scored against untreated recordings read the same way.

On a blind test from another laboratory (Charlesworth et al. 2015: mouse
cultures kept on the NMDA blocker APV, against untreated sisters), the twin's
NMDA probability separates 29 treated from 23 untreated preparations: AUROC
0.86 [0.74, 0.96], against a bar of 0.70, and 0.89 on the preparations its
guard accepts. The published estimator it builds on scores 0.49. The reading
is a ranking, not a call: no preparation passed 0.5.

For chip design, the same simulator says before an experiment which electrodes
resolve which chip property, and how many chips a claim needs: about 20 per
design, against 8 in the only public test. On a recorded four-compartment
hippocampal chip, its pre-registered prediction that axons follow their home
compartment held (49 of 60 axon pools); its prediction that compartment
timing barely tracks axonal traffic did not (-0.30, outside its interval).

Naming the single exact mechanism is the weak link (2/29, 2/10). Every failure
is reported in full.

### 项目摘要（中文）

神经器官芯片能够记录培养神经网络的每一次爆发放电，但现有分析止步于描述：放电减少了，爆发变短了。这类描述无法区分 AMPA 受体阻断剂和钠通道阻断剂，而筛选化合物的实验室恰恰需要知道是哪一种。Hodgkin's Razor 把记录转化为机制假设：将一个在 GPU 上模拟的生物物理神经网络拟合到给药前后两段记录上，针对十种分子机制，分别给出该机制发生改变的概率、改变幅度及其区间；如果没有任何拟合模型能复现数据，系统会拒绝下结论。模型完全基于模拟数据训练，从未接触过任何化合物标签。

这比"判断化合物是否起效"更难：它要回答的是十种机制中究竟哪一种发生了改变。所有测试均事先预注册，在读取数据之前提交哈希值，并以同样方式读取的未处理记录作为对照来评分。

在一项来自其他实验室的盲测中（Charlesworth 等，2015：小鼠海马培养持续接触 NMDA 受体阻断剂 APV，以同批次未处理的姊妹培养为对照），数字孪生给出的 NMDA 概率能够区分 29 份处理样本与 23 份未处理样本：AUROC 为 0.86（95% 置信区间 0.74–0.96），预注册阈值为 0.70；在守卫模块判定为可解读的样本上为 0.89。本项目所依据的已发表估计方法在同一批样本上仅为 0.49。这一读数是排序，而非判定：没有任何样本的概率超过 0.5。

在芯片设计方面，同一个模拟器能在实验之前指出哪种电极读出可以分辨芯片的哪项特性，以及一个结论需要多少块芯片：每种设计约需 20 块，而唯一的公开测试只用了 8 块。在一块实测的四腔室海马芯片上，预注册的两项预测中，"轴突放电跟随其起源腔室"得到验证（60 组轴突中 49 组符合）；"腔室放电时序与轴突传导方向基本无关"则未通过（实测 −0.30，落在预测区间之外）。

精确指认单一机制仍是短板（2/29、2/10）。所有失败结果均完整公开。

## Technical report

Full report (PDF): https://github.com/Marc-Dvci/Hodgkin-s-Razor/blob/main/docs/TECHNICAL_REPORT.pdf · Markdown: `docs/TECHNICAL_REPORT.md` in the repository. Every number below is read from `results/v4/RESULTS.md`, `results/v3/RESULTS.md` and `results/v2/RESULTS.md`.

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
7. **Chip twin.** Compartments joined by microchannels, read four ways
   (compartment electrodes, 2 Hz calcium, channel electrodes, perfusion). It
   answers which readout resolves which property, and how many chips are
   needed. A follow-up recommender, for when two mechanisms tie, is tested on
   simulations: it beats repeating the recording (0.54 against 0.42 of ties,
   p = 0.04), but not a fixed protocol (0.47, p = 0.11).

**Evaluation discipline.** There were four pre-registrations, each hashed
before its data were read, and the readers refuse to return the recordings
until the hash matches. A stop rule was committed before training: the twin
must pass the test on its own simulations before the test may be frozen. The
first twin failed it (0.792 against 0.80); after more simulations, the second
passed (0.810), and both attempts are published. Every test has an untreated
null group.

### Results

**Version 3 blind test (Charlesworth et al. 2015, chronic APV on sister arrays, 10 to 14 days in vitro).**

| Method, same preparations | AUROC of the NMDA reading, 29 treated vs 23 untreated (same genotypes) |
|---|---|
| **Hodgkin's Razor** | **0.86** [0.74, 0.96] (pre-registered bar 0.70: met) |
| Same twin, preparations the guard accepts (24 vs 16) | 0.89 [0.75, 0.98] |
| Same twin, pairing removed | 0.66 [0.51, 0.81] |
| Doorn et al. 2025 estimator, run unchanged | 0.49 [0.33, 0.65] |

- Pooled over all genotypes (78 untreated preparations): 0.79. Wild type
  alone: 0.92.
- Of the ten mechanisms, NMDA separates treated from untreated best; `tau_d`
  is next, at 0.79.
- At 15 days and later the contrast vanishes (0.48), and the reading falls
  from early to late within the same treated preparations (p = 3e-5). This
  matches the canalization the original authors describe.
- **Not met:** NMDA as the single top mechanism (2/29 against 1/23). No
  preparation reached p > 0.5: the median probability is 0.125 treated and
  0.103 untreated, where NMDA moved in 0.14 of the training simulations. The
  twin ranks; it does not call.
- The guard passed held-out simulations (0.02 fire rate). It missed its two
  must-fire bars on sister pairs (0.58 and 0.63, against 0.80).

**Version 4: a recorded four-compartment chip (Lassers et al. 2023).**
Hippocampal EC, DG, CA3 and CA1 grown in four compartments joined by axon
tunnels, with electrodes under each compartment and electrode pairs that give
each axon's direction. The twin had never seen it; it simulated 3,077
boundaries and its predictions were hashed and pushed before a spike was
counted.

| Prediction (unstimulated, 9 cultures) | Twin, fixed in advance | Recorded |
|---|---|---|
| An axon's spikes follow the compartment it grows from more than the one it grows into | at least 0.73 of axon pools | **0.82** (49/60): **met**; 0.79 and 0.86 after two stimulation patterns |
| Compartment timing barely tracks the direction of axonal traffic | Spearman inside [-0.16, 0.41] | **-0.30**: **not met**, opposite sign (permutation p = 0.09); inside the interval after stimulation |

Neither in the twin nor on the device does compartment timing give the
direction of axonal traffic; electrodes in the tunnels do. That is the chip
planner's advice.

**Version 2 blind test (Doorn et al. 2024, Dynasore, 10 wells).**

- Primary, exact mechanism: 2/10, against a bar of 5. **Not met.**
- Drug called in 9/10 treated wells and in 0/10 untreated pairs of the same
  wells; mechanism class correct in 8/10, against a chance rate of 0.30
  (p = 0.0016). These count every well: the guard put 7 of the 10 outside
  the model, where the product names nothing.
- The unpaired comparator's 8/10 turned out to be a preference: 6/10 on
  untreated pairs.
- Diagnosis: the v2 bank assumed no drift between recordings. Measured drift
  is 0.03–0.04 of each parameter's range, and with none, 23% of recorded
  untreated differences fall outside the simulated band.

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
  the first committed prediction failed (the model's striatum was 91%
  self-driven). The second changed one thing, imposing the paper's own
  statement that an isolated striatum is silent. Both synchrony readouts then
  fell in the published direction across 34 chips (p = 0.002 and 0.008);
  calcium-event frequency did not follow (p = 0.49).

### Data and compliance

| Dataset | Role | Licence |
|---|---|---|
| Charlesworth et al. 2015, Zenodo 31085 | blind test v3 | CC0 1.0 |
| Lassers et al. 2023, Zenodo 10257483 | four-compartment chip test v4 | CC0 1.0 |
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
  cultures. The probabilities rank preparations; they do not call them.
- The guard is weaker on sister pairs than within a well.
- Human cultures are outside the current domain.
- A silenced culture cannot be read.
- The next-experiment recommender does not yet beat a fixed follow-up protocol.
- The mechanism tests are on conventional MEA cultures. The chip twin is tested
  on two recorded chip sets: one underpowered, and one where one of its two
  predictions failed with the opposite sign.

## Team

Marc Donovici, solo entrant: audit, and applied machine learning, including
earlier competition work on brain-imaging and brain-decoding data.
