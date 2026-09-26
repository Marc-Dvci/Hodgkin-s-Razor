# Demo video script

Target 4:40, hard ceiling 5:00. Narration is measured with edge-tts and the
beats are cued to the synthesis timestamps, so the cuts land on the words.

Everything on screen is the real application or a real figure from
`results/figures/`. No mock-ups.

---

## 0:00 – 0:22 · The problem, stated once

**Screen.** A real raster from `data/examples/rat_cnqx.json`, baseline on the
left, treated on the right. The treated side is visibly emptier.

**Narration.**
"This is a neural organ-on-chip before and after a compound. Firing fell by
ninety percent and the bursts nearly stopped. Every analysis pipeline in use
today will tell you that. None of them will tell you why."

**Cut.** Three candidate answers appear over the treated raster: *blocked AMPA
receptors? opened chloride channels? shut down sodium channels?*

"Those three have different consequences for a drug programme, and they look
the same in the data."

---

## 0:22 – 0:50 · What the system does

**Screen.** The pipeline diagram, drawn left to right as it is described.

**Narration.**
"Hodgkin's Razor fits a biophysical network model to the recording itself. Two
recordings of one well go in. What comes out is the molecular mechanism the
compound moved, with an effect size, a credible interval, and a probability.

The razor is the prior: among the mechanisms that could explain the change,
prefer the fewest."

---

## 0:50 – 1:35 · The moment

**Screen.** The live web application. Select a recording pair labelled only
"Rat cortical DIV 22 - compound A". Click Analyse. The mechanism bars fill.

**Narration.**
"Here is a pair the model has never seen. It is trained only on simulations. It
has never been shown a compound label in its life.

It says: AMPA receptors, down, to eleven percent of baseline, probability
ninety-four."

**Cut.** Reveal the label: CNQX, an AMPA antagonist.

"The compound was CNQX. An AMPA antagonist."

---

## 1:35 – 2:05 · Not a lookup

**Screen.** The raster comparison panel: measured left, twin right, baseline
above, treated below.

**Narration.**
"That is not a lookup. The right-hand panels are a fresh simulation of the
network at the parameters the model just inferred. Same bursts, same rates,
same silence after the compound. If the twin cannot reproduce the recording,
the answer does not get published."

---

## 2:05 – 2:45 · Scale

**Screen.** `results/figures/confusion.png`, then `per_compound.png`.

**Narration.**
"Across every well on both plates, rat cortical and human stem-cell derived,
five compounds and vehicle controls, it named the right mechanism in [N] of
wells. Chance is one in nine.

On vehicle controls it raised a mechanism in [N] percent of wells.

The answer key, the metric and the failure conditions were written and hashed
before a single recording was scored."

---

## 2:45 – 3:15 · The refusal

**Screen.** `results/figures/guard.png`. Then the app showing an "outside the
model" verdict on a shuffled recording.

**Narration.**
"A system that names a mechanism for every recording is useless in a
laboratory, because the interesting recordings are the surprising ones.

So the twin is re-simulated and checked against what was measured. Feed it a
network whose receptor kinetics the model cannot represent, and it refuses.
Feed it a recording with the network structure destroyed, and it refuses. It
names nothing rather than name the wrong thing."

---

## 3:15 – 3:55 · The chip

**Screen.** The two-compartment chip diagram, then
`results/figures/chip_readout.png`.

**Narration.**
"The same simulator runs a two-compartment chip with one-way microchannels, the
geometry used for directional cortico-striatal circuits, and a calcium readout
at two frames per second.

Which tells you what the readout can resolve, before you run the experiment.
The electrode array recovers the cross-channel coupling. The calcium movie at
two hertz does not resolve the direction at all. That is worth knowing on the
Monday, not the Friday."

---

## 3:55 – 4:20 · Run it

**Screen.** A terminal: `pip install -r requirements.txt`, then `python
demo.py`, output scrolling. Then `pytest -q` showing the tests pass.

**Narration.**
"Everything here runs from public data on one desktop GPU. No cloud, no paid
API, no proprietary model. One command reproduces the table. The simulator is
bit-reproducible, and there is a test that fails if it stops being."

---

## 4:20 – 4:40 · Why it matters

**Screen.** Back to the first raster, now with the mechanism label attached.

**Narration.**
"An organ-on-chip experiment currently ends in a description. This ends it in a
mechanism, with an interval, and with the honesty to say when it does not know.

That is the step from describing an experiment to predicting one."

---

## Production notes

* Capture the application with Playwright at 1920x1080, then compose in
  Remotion. Record the terminal separately with a fixed 16px monospace font.
* Subtitles in English and Simplified Chinese; the panel is Chinese-hosted.
* Publish on YouTube **and Bilibili**; YouTube is not reachable from mainland
  China and the video must be viewable without login.
* Numbers in brackets are filled from `results/RESULTS.md` at edit time. Do not
  narrate a number that is not in that file.
* The opening figures are measured on `data/examples/rat_cnqx.json`: firing
  rate 4.40 to 0.45 events per second per electrode, network bursts 14 per
  minute to 3.
