# Data sources, licences and compliance

## What is used

| Dataset | What it provides | Licence | Where |
|---|---|---|---|
| Tampere comparative MEA dataset | development set; the domain of the Axion twin | CC BY 4.0 | gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset |
| Doorn et al. 2024, Dynasore peak trains | blind test of version 2; the untreated baselines set the domain of the MCS twin | Apache-2.0 | gitlab.utwente.nl/m7706783/fb_model (commit a88e15d) |
| Mateus et al. 2024, microchannel chip recordings | blind test of the chip readout prediction | CC BY-NC-ND (dataset README) | zenodo.org/records/14525182 |
| Doorn et al. 2025, SBI repository | prior art: their trained estimator and feature code, scored beside the twin | Apache-2.0 | gitlab.utwente.nl/m7706783/SBI_MEA_model (commit d7f3615) |

No purchased data, no restricted repository, no data behind a login, and no
clinical or personal data. Every file the results depend on is checked against
the SHA-256 it had when the results were produced: `scripts/fetch_tampere.py`
writes and `scripts/verify.py` re-checks the Tampere checksums, and
`scripts/fetch_external.py` checks the others.

## Tampere comparative MEA dataset

Hyvärinen T., Hyysalo A., Kapucu F.E., Aarnos L., Vinogradov A., Eglen S.J.,
Ylä-Outinen L., Narkilahti S. *Comparative microelectrode array data of the
functional development of hPSC-derived and rat neuronal networks.* Scientific
Data 9:118 (2022).

Two pharmacology plates, Axion 48-well MEAs with 16 electrodes per well: rat
cortical neurons at DIV 22 and human pluripotent-stem-cell-derived neurons at
DIV 29, each recorded at baseline, after compound wash-on and after TTX.
Compounds: CNQX 50 uM, D-AP5 50 uM, GABA 10 uM, gabazine 30 uM, kainic acid
5 uM, vehicle; seven wells per condition on the rat plate, four on the human
plate. Electrodes the authors marked noisy are dropped. Windows are three
consecutive 60 s blocks from 300 s, matched on both sides of a pair.

These plates were scored four times under version 1 and are the development
set of version 2: reported, never claimed as a blind result.

## Doorn et al. 2024, Dynasore

Doorn N., Voogd E.J.H.F., Levers M.R., van Putten M.J.A.M., Frega M.
*Breaking the burst: unveiling mechanisms behind fragmented network bursts in
patient-derived neurons.* Stem Cell Reports 19:1583 (2024).

Human iPSC-derived excitatory neurons (Ngn2) with rat astrocytes, Multi Channel
Systems 24-well MEA (12 recording electrodes per well, 10 kHz), DIV 35, 10 uM
Dynasore added after a baseline recording. The ten wells and the analysis
windows are the ones in the authors' Table S2 (Europe PMC supplement
`mmc2.xlsx`): FB2 B6, C6, D6 (baseline 0 to 300 s, Dynasore 1550 to 1850 s);
FB2t A2, A3, B1, C1 (0 to 300 s, 950 to 1250 s); FB3t A3, C3, D1 (0 to 300 s,
1520 to 1820 s). Each side is cut into five 60 s windows. The peak-train file
of well FB2t A1 is in the repository but not in their table and is not used.

`hodgkins_razor/doorn.py` returns the treated windows only when
`PREREGISTRATION_v2.md` exists and matches its hash.

## Mateus et al. 2024, microchannel chips

Mateus J., Melo P., Aroso M., Charlot B., Aguiar P. *Influence of asymmetric
microchannels in the structure and function of engineered neuronal circuits.*
bioRxiv 10.1101/2024.07.09.602729 (2024).

Rat hippocampal neurons (E18) seeded at the same density in both chambers of a
two-compartment device, recorded for 10 minutes at DIV 12 to 24 on a Multi
Channel Systems MEA2100-256. Sixteen microchannels lie along the sixteen
electrode columns and each covers five electrodes (rows 7 to 11); the chambers
are read by the rows above and below. 57 recordings from 28 chips: straight
controls (15 recordings), Tesla (8), Tesla v2 (10), Rams (11), Arrows (13).

**Licence.** The Zenodo record lists CC BY 4.0; the dataset's own README states
CC BY-NC-ND (attribution, non-commercial, no derivatives). This project follows
the stricter terms: the data are downloaded by a script, read in place, never
redistributed or altered, and only derived summary statistics are published,
with credit to the authors.

## Doorn et al. 2025, prior art

Doorn N., van Putten M.J.A.M., Frega M. *Automated inference of disease
mechanisms in patient-hiPSC-derived neuronal networks.* Communications Biology
(2025). Their trained estimator (`TrainedNDE`, an sbi 0.21 object) and feature
code (`FeatureExtraction.py`) are run unchanged by `scripts/prior_art_doorn.py`
in a separate environment (`requirements-doorn.txt`). One adaptation, stated in
that script: their feature code needs every electrode to carry a spike, so a
silent electrode is given one spike at the last sample of the window.

## The domain each twin covers

A simulated culture is admitted to the bank of a recording system only when
its simulated baseline, read through that system, falls inside the range the
system's untreated baselines span (2nd to 98th percentile of eight
detection-robust statistics, widened), is a living network-driven culture, and
collapses below 35 percent of its activity when AMPA is blocked. Only baseline
windows enter this: 198 Tampere windows and 50 Doorn windows. No treated
recording and no compound label.
