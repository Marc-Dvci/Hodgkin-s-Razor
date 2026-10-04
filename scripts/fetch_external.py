"""Download the external datasets and the prior-art estimator, and check them.

    python scripts/fetch_external.py

* Doorn et al. (2024) Dynasore peak trains and (2025) trained estimator, two
  public GitLab repositories (Apache-2.0), cloned at the commits used here.
* Mateus et al. (2024) microchannel chip recordings, Zenodo record 14525182.
  The dataset README states CC BY-NC-ND: research use with credit, no altered
  redistribution and no commercial use. It is downloaded to data/raw and read
  in place; nothing derived from it is written except summary statistics.
* Charlesworth et al. (2015) sister-culture recordings, Zenodo record 31085,
  CC0 (public domain). The blind test of the third pre-registration.
* Lassers et al. (2023) four-compartment hippocampal chips, Zenodo record
  10257483 (Dryad 10.5061/dryad.7h44j1013), CC0. The processed spike tables
  only (130 MB); the raw tunnel archives are not needed. The test of the
  fourth pre-registration.

Every file the results depend on is checked against the SHA-256 it had when
the results were produced, so a changed upstream file is reported instead of
silently producing different numbers.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

REPOS = {
    "doorn_fb_model": ("https://gitlab.utwente.nl/m7706783/fb_model.git",
                       "a88e15d2e98d85c394c1aaa09fa598c4f68371f7"),
    "doorn_sbi": ("https://gitlab.utwente.nl/m7706783/SBI_MEA_model.git",
                  "d7f3615762d6c72d5075dc509262d6accbc91406"),
}
ARCHIVES = {
    "mateus2024/1_MEA_Recordings.zip":
        "https://zenodo.org/records/14525182/files/1_MEA_Recordings.zip?download=1",
    "charlesworth2015/g2c-1.zip":
        "https://zenodo.org/records/31085/files/g2c-1.zip?download=1",
}

FILES = {f"brewer2023/{name}": f"https://zenodo.org/api/records/10257483/files/{name}/content"
         for name in ("NoStimSortedAxons.mat", "NoStimWellSpikes.mat",
                      "HFS5SortedAxons.mat", "HFS5WellSpikes.mat",
                      "HFS40SortedAxons.mat", "HFS40WellSpikes.mat",
                      "NoStimTunnelSpikeDynamics.mat", "NoStimWellSpikeDynamics.mat",
                      "HFS5TunnelSpikeDynamics.mat", "HFS5WellSpikeDynamics.mat",
                      "HFS40TunnelSpikeDynamics.mat", "HFS40WellSpikeDynamics.mat",
                      "README.md")}

SHA256 = {
    "doorn_fb_model/Experimental_peaktrains/APS_FB2_B6.mat": "020a0bb40403c262282cc204b43c7aab4dae454b430633e659e08d489c799996",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2_C6.mat": "5375269e65a99605270d451537061867356d564e3848744d185f60d69a5a2891",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2_D6.mat": "2b82284ef13191bd35de400c91b735720532b55b8cc6e1bbc0badd2ad1dad020",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2t_Dyn_tot_A2.mat": "6f44a2a53b5ffb55660204cafb57e4bceeebda78f15c726dc72d525f01e16808",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2t_Dyn_tot_A3.mat": "f953e59f0a8d92f7b57d29f8c695417bf8d7b7c1599c43591698518c1f424d66",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2t_Dyn_tot_B1.mat": "bbce49c173d9fab6ed850cbbae6de3b72376fe5b4cb9b39973175163afdd44a8",
    "doorn_fb_model/Experimental_peaktrains/APS_FB2t_Dyn_tot_C1.mat": "4615c413f359a7f0723a8f04e03e8c5724cb650a871d0f8e9add767b035007dd",
    "doorn_fb_model/Experimental_peaktrains/APS_FB3t_Dyn_A3.mat": "c2346a149e2db43e1b2590eaebc401c7b177ad4bede1a29dacc7b84740086107",
    "doorn_fb_model/Experimental_peaktrains/APS_FB3t_Dyn_C3.mat": "c4c50149a49a9b8c2f650f3154bfaf82632b93b4e0d4a6307e25dd6c631db916",
    "doorn_fb_model/Experimental_peaktrains/APS_FB3t_Dyn_tot_D1.mat": "6a1988ba6dd65ebe3784b04069fa9a84003ed104164faeb84492cd69ac555033",
    "doorn_sbi/TrainedNDE": "5655060603b08e18804437be83526525e88b7264cc6987ef44731c62228d7999",
    "doorn_sbi/FeatureExtraction.py": "d7c367ae6aa0056bf50245d559c6d4b3cdd1b2bd7e63913729917ef2752e1a68",
    "mateus2024/1_MEA_Recordings.zip": "f70d42656a9e0d4bd7c65b33b9b535b90401139b91def8262098d0449bb71371",
    "charlesworth2015/g2c-1.zip": "bbac9851229c7768d2099594a241433fc00b0edb385fed860c0bbb464d946239",
    "brewer2023/HFS40SortedAxons.mat": "69712f88dd7343315f35aafa5b0138f10fc3c3320498e992da6bd7404931d5d5",
    "brewer2023/HFS40TunnelSpikeDynamics.mat": "e89929ce39007ffe56ea7c0281fd5d8617da590b7a3fb0061c9224a119ea41b8",
    "brewer2023/HFS40WellSpikeDynamics.mat": "f1f1769d2753150d8bea788c97ab57bca5f815fed473e815eb438b63d1bacd36",
    "brewer2023/HFS40WellSpikes.mat": "7455cd60f6e09e3140c0b769acef625b40035dfa6676fb588cdc6917f7a5dbe0",
    "brewer2023/HFS5SortedAxons.mat": "293005fb831956a19733a548c51e0e24d0e89f7fce0b7625205b4db74fbaca23",
    "brewer2023/HFS5TunnelSpikeDynamics.mat": "c50ca05e9d5ad9b636bc6ba511f6542597abadd57e5c76bd5ab26db3cf2d1edf",
    "brewer2023/HFS5WellSpikeDynamics.mat": "a005bb8485c9f7d4df11c70906fd7bacfd1325d8cb3ea3f504bf753f3d870d9e",
    "brewer2023/HFS5WellSpikes.mat": "45e667739b8b280d727bce5779a55b65118941271b5ee22c1cd698a2f56bac83",
    "brewer2023/NoStimSortedAxons.mat": "0a946cf542b58e2d9c3815c8998ee17e8a5f8a5652279a99681f7edbc257888f",
    "brewer2023/NoStimTunnelSpikeDynamics.mat": "4d38e0f75430575ba704aa014e0a5bc6fd578157a378d9934d5d431418b20a9b",
    "brewer2023/NoStimWellSpikeDynamics.mat": "5e9496778758fe6a865eff04d886d801208ac71618948cc2d50264e7e95bc1c5",
    "brewer2023/NoStimWellSpikes.mat": "14d1d83824e50d619050143c6d138d939a2df3585dcc7143c0188baedbd19cbe",
}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        dest = RAW / name
        if not dest.exists():
            subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
        print(f"{name} at {commit[:10]}")
    for rel, url in ARCHIVES.items():
        out = RAW / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=900) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(out) as z:
            top = z.namelist()[0].split("/")[0]
            if not (out.parent / top).exists():
                z.extractall(out.parent)
    for rel, url in FILES.items():
        out = RAW / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=900) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
    bad = []
    for rel, want in SHA256.items():
        p = RAW / rel
        got = sha(p) if p.exists() else "missing"
        if got != want:
            bad.append(rel)
        print(f"  {'ok ' if got == want else 'BAD'} {rel}")
    if bad:
        sys.exit(f"{len(bad)} files differ from the versions the results used")
    print("all external files match")


if __name__ == "__main__":
    main()
