# Data sources, licences and compliance

## What is used

| Dataset | What it provides | Licence | Where |
|---|---|---|---|
| Tampere comparative MEA dataset | development set; the domain of the Axion twin | CC BY 4.0 | gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset |
| Doorn et al. 2024, Dynasore peak trains | blind test of version 2; the untreated baselines set the domain of the MCS twin | Apache-2.0 | gitlab.utwente.nl/m7706783/fb_model (commit a88e15d) |
| Mateus et al. 2024, microchannel chip recordings | blind test of the chip readout prediction | CC BY-NC-ND (dataset README) | zenodo.org/records/14525182 |
| Charlesworth et al. 2015, sister-array recordings | blind test of version 3 (chronic APV); the untreated first sisters set the domain of the MCS 60-electrode twin, and untreated sister pairs set the drift | CC0 1.0 (public domain) | zenodo.org/records/31085 |
| Lassers et al. 2023, four-compartment hippocampal chips | test of the chip twin, version 4 | CC0 1.0 (public domain) | zenodo.org/records/10257483 (Dryad 10.5061/dryad.7h44j1013) |
| Lassus et al. 2018, cortico-striatal chips | published directions of the GluN2B result only (no data) | cited | Sci Rep 8:17461 |
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

## Charlesworth et al. 2015, sister arrays

Charlesworth P., Morton A., Eglen S.J., Komiyama N.H., Grant S.G.N.
*Canalization of genetic and pharmacological perturbations in developing
primary neuronal activity patterns.* Neuropharmacology 100:47-55 (2015).

Mouse hippocampal neurons on Multi Channel Systems 60-electrode arrays,
recorded twice a week from 6 to about 30 days in vitro. Each preparation was
plated on two sister arrays (A and B; 21 preparations on four, A to D). After
the recording at 7 days, 50 uM APV was added to the B array of some
preparations and kept in the medium. Wild type and eight knockout lines
(Gria1, PSD-95, PSD-93, SAP102, SPA, GNB1, GRIT). Spike times as released by
the authors (detection at -20 uV); each 8 x 8 recording is read as four
4 x 4 quadrants.

**Use.** The blind test of `PREREGISTRATION_v3.md`. `hodgkins_razor/charlesworth.py`
returns a B or D recording made at 8 days or later only when that file exists
and matches its hash. Before the freeze, only the metadata sheet, the A and C
arrays and the recordings at 6 and 7 days were read. The four-array
preparations' A–C pairs (never treated) set the drift at the scored ages and
are excluded from scoring.

**Licence.** CC0 1.0 on the Zenodo record: no restriction. The data are
downloaded by `scripts/fetch_external.py` and not redistributed in the
repository.

## Lassers et al. 2023, four-compartment hippocampal chips

Lassers S., Vakilna Y.S., Tang W., Brewer G.J. *The flow of axonal information
among hippocampal subregions: 2. Patterned stimulation sharpens routing of
information transmission.* Dryad, doi:10.5061/dryad.7h44j1013; Zenodo record
10257483. Licence CC0 1.0. Rat hippocampal EC, DG, CA3 and CA1 cultured in four
compartments of one PDMS device on an MCS 120-electrode array; 19 electrodes
per compartment and electrode pairs across five tunnels per boundary; 300 s at
25 kHz; nine unstimulated cultures, six of them recorded again after each of
two high-frequency stimulation patterns. Only the processed spike tables are
downloaded (`*SortedAxons.mat`, `*WellSpikes.mat` and the spike-dynamics
tables, 130 MB); `hodgkins_razor/brewer.py` reads them with `mat-io` and
returns no spike time until `PREREGISTRATION_v4.md` matches its hash. Animal
cultures from a published study; no personal data.

## Lassus et al. 2018, cortico-striatal chips

Lassus B., Naudé J., Faure P., Guedin D., Von Boxberg Y., Mannoury la Cour C.,
Millan M.J., Peyrin J.-M. *Glutamatergic and dopaminergic modulation of
cortico-striatal circuits probed by dynamic calcium imaging of networks
reconstructed in microfluidic chips.* Sci Rep 8:17461 (2018). Only the
published directions of the GluN2B result and the protocol description are
used (`docs/LASSUS_PREDICTION.md`); no data from the paper.

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
windows enter this: 198 Tampere windows, 50 Doorn windows and the first-sister
quadrant windows of the Charlesworth pairs at 9 days and later. No treated
recording and no compound label.
