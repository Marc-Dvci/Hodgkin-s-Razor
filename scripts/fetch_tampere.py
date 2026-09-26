"""Download the Tampere Comparative MEA dataset (CC BY 4.0) spike and metadata files.

Raw voltage HDF5 files are skipped unless --raw is passed.
"""
import argparse
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

REPO = "NeuroGroup_TUNI/Comparative_MEA_dataset"
RAW = f"https://gin.g-node.org/{REPO}/raw/master/"
ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "tampere_manifest.txt"
DEST = ROOT / "data" / "raw" / "tampere"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="also fetch raw HDF5 voltage files")
    args = ap.parse_args()
    files = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip()]
    if not args.raw:
        files = [f for f in files if not f.endswith(".h5")]
    sums = {}
    for rel in files:
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_size == 0:
            url = RAW + urllib.parse.quote(rel)
            with urllib.request.urlopen(url, timeout=600) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            print("got", rel, out.stat().st_size)
        sums[rel] = hashlib.sha256(out.read_bytes()).hexdigest()
    (DEST / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1))
    print(len(sums), "files")


if __name__ == "__main__":
    main()
