"""Write notebooks/hodgkins_razor_demo.ipynb, the CPU notebook for reviewers.

    python scripts/make_notebook.py

The notebook reads every number it prints from the results files at run time,
so it cannot disagree with them. It runs on a CPU: in a checkout, or on Kaggle
after cloning the repository (the first cell does that when the package is not
found).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Hodgkin's Razor

**Which molecular mechanism did a compound move, and can the twin say so on data it never saw?**

A biophysical digital twin of neural organ-on-chip networks, trained only on
simulations. From a baseline and a treated recording of one well it returns the
probability that each of ten mechanisms moved, the size of the shift if it did,
and a guard that refuses when no fitted model reproduces the recording.

Every result is scored against untreated recordings read the same way: a
method's chance level is its own hit rate when nothing was applied.

This notebook runs on a CPU. It uses the trained twins in `models/`, the
recording pairs bundled in `data/examples/`, and the pre-registered results in
`results/v3/` and `results/v2/`. Rebuilding the simulation banks needs a CUDA
device.
"""),
    code("""
import sys, json, pathlib, subprocess
ROOT = pathlib.Path.cwd()
for cand in (ROOT, ROOT.parent, pathlib.Path('/kaggle/working/hodgkins-razor')):
    if (cand / 'hodgkins_razor').exists():
        ROOT = cand
        break
else:
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/Marc-Dvci/Hodgkin-s-Razor', '/kaggle/working/hodgkins-razor'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'zuko', 'numba'], check=True)
    ROOT = pathlib.Path('/kaggle/working/hodgkins-razor')
sys.path.insert(0, str(ROOT))
import numpy as np
from hodgkins_razor import features as F, nde, params as P, ppc, report
print('statistics:', F.N_FEATURE, ' parameters:', P.N_PARAM, ' mechanisms a compound can move:', P.N_SHIFT)
"""),
    md("""
## 1. Blind test, version 3: chronic NMDA blockade on sister cultures

Charlesworth et al. (2015) plated each preparation of mouse hippocampal neurons
on two sister arrays and kept an NMDA antagonist (APV) on one of them. The twin
reads the untreated sister as baseline and the other as treated, and reads
untreated sister pairs of the same genotypes the same way. The primary: the
AUROC of the twin's probability that NMDA moved, treated against untreated
preparations, with a pre-registered bar of 0.70.
"""),
    code("""
import hashlib
md3 = (ROOT / 'PREREGISTRATION_v3.md').read_bytes()
print('pre-registration v3 hash matches:',
      hashlib.sha256(md3).hexdigest() == (ROOT / 'PREREGISTRATION_v3.sha256').read_text().split()[0])
r3 = json.loads((ROOT / 'results' / 'v3' / 'results.json').read_text())
A = r3['A_blind']; m = A['early']['metrics']
print(f"p(g_nmda), {m['n_treated']} treated vs {m['n_null']} untreated preparations: "
      f"AUROC {m['auroc_key']:.3f} [{m['auroc_key_ci95'][0]:.2f}, {m['auroc_key_ci95'][1]:.2f}]")
for name, key in (('unpaired twin', 'unpaired'), ('Doorn et al. estimator', 'prior_art')):
    mb = r3['B_comparators'][key]['metrics']
    print(f"  {name:24s} AUROC {mb['auroc_key']:.3f}")
print(f"co-primary, g_nmda top-1: {m['top1_treated_hits']}/{m['n_treated']} vs "
      f"{m['top1_null_hits']}/{m['n_null']} (Fisher p {m['top1_fisher_p']:.2g})")
c = A['canalization']
print(f"canalization: median p(g_nmda) {c['median_early']:.3f} early -> {c['median_late']:.3f} late, "
      f"Wilcoxon p {c['wilcoxon_p_early_gt_late']:.1g}")
"""),
    md("""
## 2. Blind test, version 2: Dynasore

Version 2 was frozen on simulation evidence, its answer key and success bar were
hashed in `PREREGISTRATION_v2.md`, and only then was it run on the Dynasore
recordings of Doorn et al. (2024): another laboratory, human iPSC-derived
neurons, another recording system. Dynasore strengthens short-term depression;
the key accepts `u_rel` or `tau_d`, and chance is 0.2.
"""),
    code("""
import hashlib, re
md_ = (ROOT / 'PREREGISTRATION_v2.md').read_bytes()
print('pre-registration hash matches:',
      hashlib.sha256(md_).hexdigest() == (ROOT / 'PREREGISTRATION_v2.sha256').read_text().split()[0])
r = json.loads((ROOT / 'results' / 'v2' / 'results.json').read_text())
a = r['A_doorn']['metrics']
print(f"blind test: {a['top1_hits']}/{a['n_wells']} wells named correctly "
      f"(chance {a['chance']:.2f}, P under chance {a['p_vs_chance']:.2g})")
for w in sorted(r['A_doorn']['wells'], key=lambda w: w['well']):
    print(f"  {w['well']:9s} top call {w['top1']:12s} p={max(w['p_active']):.2f}")
"""),
    md("""
## 3. One pair, end to end

A rat culture before and after TTX, a sodium-channel blocker (any bundled pair
works; change `name`). The same function computes the statistics of a recording
and of a simulation; the recording system (electrode layout, detection dead
time) decides which twin reads it.
"""),
    code("""
name = 'rat_ttx'
d = json.loads((ROOT / 'data' / 'examples' / f'{name}.json').read_text())
n_elec = int(d.get('n_elec', 16)); view = 'grid12' if n_elec == 12 else 'grid16'
base = np.array(d['baseline']).reshape(-1, 2); treat = np.array(d['treated']).reshape(-1, 2)
dur = float(d['duration'])
print(d['label'], '|', d.get('system', ''), '|', len(base), '->', len(treat), 'events in', dur, 's')
x_base, x_treat = F.compute(base, n_elec, dur), F.compute(treat, n_elec, dur)
for n, a_, b_ in list(zip(F.NAMES, x_base, x_treat))[:8]:
    print(f'{n:22s} {a_:9.3f} -> {b_:9.3f}')
"""),
    code("""
twin = nde.Twin.load(ROOT / 'models' / f'twin_v2_{view}', device='cpu')
post = twin.posterior(x_base, x_treat, n_samples=4000)
typ = ppc.Typicality.for_twin(twin)
t = typ.of_pair(twin, x_base, x_treat)
# The predictive check re-simulates the twin, which needs a CUDA device. On a
# CPU its verdict is read from the analysis cached by the GPU run of this pair.
cached = d['analysis']['report']['guard']
guard = {'typicality': t, 'typicality_threshold': typ.threshold,
         'discrepancy': cached['discrepancy'], 'threshold': cached['threshold'],
         'inside_model': bool(t <= typ.threshold and cached['discrepancy'] <= cached['threshold'])}
print(f"typicality {t:.2f} (threshold {typ.threshold:.2f}); predictive check, cached from the GPU run: "
      f"discrepancy {cached['discrepancy']:.1f} (threshold {cached['threshold']:.1f})")
body = report.build(post, guard, x_base, x_treat, meta={'duration_s': dur, 'recording_system': view})
print(body['sentence'])
print()
print(report.to_markdown(body).split('## Mechanism class')[0])
"""),
    md("""
The predictive check, the second guard test, re-simulates the twin and needs a
CUDA device; `python demo.py --live` runs it live. Typicality needs no simulation.
If either test fires, no mechanism is named.

## 4. Every bundled pair

One 60 s window of each pair, read by the version 2 twins. These are
development recordings: their answers were known while the twin was built. The
last column is the guard's verdict from the cached GPU run; a pair outside the
model gets no mechanism in the product, whatever its top-ranked parameter. A
single window is noisy: the rat vehicle window below is flagged, while the
vehicle wells scored on all their windows are not (0 of 11,
`results/v2/RESULTS.md`).
"""),
    code("""
KEY = {'CNQX': {'g_ampa'}, 'D-AP5': {'g_nmda'}, 'GABA': {'g_gaba', 'g_tonic_inh'},
       'gabazine': {'g_gaba'}, 'kainic acid': {'g_ampa'}, 'TTX': {'g_na'},
       'Dynasore': {'u_rel', 'tau_d'}, 'vehicle control': None}
twins = {v: nde.Twin.load(ROOT / 'models' / f'twin_v2_{v}', device='cpu') for v in ('grid16', 'grid12')}
for p in sorted((ROOT / 'data' / 'examples').glob('*.json')):
    e = json.loads(p.read_text()); n = int(e.get('n_elec', 16)); v = 'grid12' if n == 12 else 'grid16'
    xb = F.compute(np.array(e['baseline']).reshape(-1, 2), n, e['duration'])
    xt = F.compute(np.array(e['treated']).reshape(-1, 2), n, e['duration'])
    top = nde.summarise(twins[v].posterior(xb, xt, n_samples=1500))[0]
    want = KEY.get(e['compound'])
    called = top['key'] if top['p_active'] >= 0.5 else '-'
    mark = ('ok' if called == '-' else 'flag') if want is None else ('ok' if top['key'] in want else '.')
    g = 'inside' if e['analysis']['report']['inside_model'] else 'OUTSIDE: nothing named'
    print(f"{e['label']:52s} top {top['key']:12s} p={top['p_active']:.2f}  {mark:4s} guard {g}")
"""),
    md("""
## 5. The chip twin

The same kernel simulates a two-compartment chip with directional microchannels.
In simulation, electrodes under the chambers cannot resolve a chip's
directionality and electrodes inside the channels can. That prediction was
pre-registered and then tested on recorded chips (Mateus et al. 2024).
"""),
    code("""
chip = json.loads((ROOT / 'results' / 'chip_study.json').read_text())
for readout, res in chip['readouts'].items():
    print(f"{readout:22s} direction recovery r = {res['direction_sel']['pearson_r']:.2f}")
b = r.get('B_chips')
if b:
    for stat in ('dominant_share', 'chamber_asymmetry'):
        s = b[stat]
        print(f"recorded chips, {stat}: AUROC {s['auroc']:.2f} "
              f"(predicted: {s['prediction']}) -> {'met' if s['met'] else 'not met'}")
cp = json.loads((ROOT / 'results' / 'chip_power.json').read_text())['readouts']['channel_dominant_share']
print(f"chips per design for 80% power: {cp['twin']['chips_needed']} if the twin is right, "
      f"{cp['observed']['chips_needed']} at the recorded separation")
l2 = ROOT / 'results' / 'lassus_study_v2.json'
if l2.exists():
    h = json.loads(l2.read_text())['striatal_target']['nmda']['0.3']
    for k in ('tgt_freq', 'striato_striatal_sync', 'cortico_striatal_sync'):
        print(f"Lassus, second prediction, {k:22s} lower in {h[k]['fraction_lower']:.2f} of chips, "
              f"p {h[k]['wilcoxon_p_lower']:.2g}")
"""),
    md("""
## 6. The chip twin on a recorded four-compartment chip (version 4)

Lassers et al. (2023) grew hippocampal EC, DG, CA3 and CA1 in four compartments
of one device, joined by axon tunnels, with electrodes under each compartment
and electrode pairs that give each axon's direction. The twin's two predictions
were simulated (`scripts/brewer_prediction.py`), hashed in
`PREREGISTRATION_v4.md` and pushed before a spike of the dataset was counted.
This cell re-checks the hash and prints both outcomes, including the one that
failed.
"""),
    code("""
import hashlib
md4 = (ROOT / 'PREREGISTRATION_v4.md').read_bytes()
print('pre-registration v4 hash matches:',
      hashlib.sha256(md4).hexdigest() == (ROOT / 'PREREGISTRATION_v4.sha256').read_text().split()[0])
r4 = json.loads((ROOT / 'results' / 'v4' / 'results.json').read_text())
for name, v in r4['verdicts'].items():
    print(f"{name:20s} observed {v['observed']:+.2f}  bar {v['bar']:45s} {'MET' if v['met'] else 'NOT MET'}")
for c in ('hfs5', 'hfs40'):
    s = r4['secondary'][c]
    print(f"after {c}: home share {s['home']['share_positive']:.2f}, "
          f"timing vs traffic {s['asym_vs_traffic_ff']['spearman']:+.2f}")
"""),
    md("""
## 7. Where to go next

* `results/v4/RESULTS.md`, `results/v3/RESULTS.md` and `results/v2/RESULTS.md`: every pre-registered number, including what failed.
* `docs/TECHNICAL_REPORT.md`: the full report.
* `python demo.py --serve`: the web application.
* `python scripts/run_all.py`: the whole pipeline on a CUDA device.
"""),
]


def main() -> None:
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "hodgkins_razor_demo.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
