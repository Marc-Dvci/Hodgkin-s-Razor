"""Download the external datasets and the prior-art estimator, and check them.

    python scripts/fetch_external.py

* Doorn et al. (2024) Dynasore peak trains and (2025) trained estimator, two
  public GitLab repositories (Apache-2.0), cloned at the commits used here.
* Mateus et al. (2024) microchannel chip recordings, Zenodo record 14525182.
  The dataset README states CC BY-NC-ND: research use with credit, no altered
  redistribution and no commercial use. It is downloaded to data/raw and read
  in place; nothing derived from it is written except summary statistics.

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
MATEUS = ("https://zenodo.org/records/14525182/files/1_MEA_Recordings.zip?download=1",
          "mateus2024/1_MEA_Recordings.zip")

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
    url, rel = MATEUS
    out = RAW / rel
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=900) as r, open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if not (out.parent / "1_MEA_Recordings").exists():
        with zipfile.ZipFile(out) as z:
            z.extractall(out.parent)
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
