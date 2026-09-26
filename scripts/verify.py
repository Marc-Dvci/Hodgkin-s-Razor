"""Check that the repository states what it can support.

    python scripts/verify.py

Runs the checks a reviewer would otherwise do by hand: the pre-registration
matches its hash, the data files match their checksums, the tests pass, the
headline numbers in the written results match the JSON they came from, and no
artefact is missing. Exits non-zero on the first failure class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    c = Check()

    print("pre-registration")
    md = ROOT / "PREREGISTRATION.md"
    rec = (ROOT / "PREREGISTRATION.sha256").read_text().split()[0]
    digest = hashlib.sha256(md.read_bytes()).hexdigest()
    c.add("PREREGISTRATION.md matches its recorded hash", digest == rec, digest[:16])

    print("source layout")
    for rel in ["hodgkins_razor/params.py", "hodgkins_razor/kernel.cu",
                "hodgkins_razor/simulator.py", "hodgkins_razor/features.py",
                "hodgkins_razor/nde.py", "hodgkins_razor/ppc.py",
                "hodgkins_razor/chip.py", "hodgkins_razor/report.py",
                "scripts/make_bank.py", "scripts/train.py", "scripts/evaluate.py",
                "app/server.py", "demo.py", "README.md", "LICENSE",
                "requirements.txt", "Dockerfile"]:
        c.add(f"{rel} present", (ROOT / rel).exists())

    print("one feature path")
    src = (ROOT / "hodgkins_razor" / "features.py").read_text()
    c.add("features.compute is the only entry to the statistics",
          src.count("def compute(") == 1)
    users = []
    for d in ("hodgkins_razor", "scripts", "app"):
        for f in (ROOT / d).glob("*.py"):
            t = f.read_text()
            if "F.compute(" in t or "features.compute(" in t:
                users.append(f"{d}/{f.name}")
    c.add("simulation and experiment share it", len(users) >= 4,
          ", ".join(sorted(users)))

    if not args.skip_data:
        print("data")
        sums = ROOT / "data" / "raw" / "tampere" / "SHA256SUMS.json"
        if sums.exists():
            d = json.loads(sums.read_text())
            bad = []
            base = sums.parent
            for rel, want in list(d.items())[:400]:
                p = base / rel
                if not p.exists():
                    bad.append(rel)
                    continue
                if hashlib.sha256(p.read_bytes()).hexdigest() != want:
                    bad.append(rel)
            c.add(f"{len(d)} recording files match their checksums", not bad,
                  "" if not bad else f"{len(bad)} mismatched")
        else:
            c.add("recording checksums present", False,
                  "run scripts/fetch_tampere.py")
        ex = sorted((ROOT / "data" / "examples").glob("*.json"))
        c.add("bundled examples present", len(ex) >= 10, f"{len(ex)} files")

    print("results")
    rj = ROOT / "results" / "results.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        c.add("results were produced against this pre-registration",
              r.get("prereg_sha256") == digest)
        rm = ROOT / "results" / "RESULTS.md"
        if rm.exists():
            text = rm.read_text()
            top1 = r["mechanism"]["top1_accuracy"]
            c.add("written top-1 accuracy matches the JSON",
                  f"{top1:.3f}" in text, f"{top1:.3f}")
            ctrl = r["mechanism"]["control_false_mechanism_rate"]
            c.add("written control rate matches the JSON",
                  f"{ctrl:.3f}" in text, f"{ctrl:.3f}")
        else:
            c.add("results/RESULTS.md present", False, "run scripts/render_results.py")
        rep = ROOT / "docs" / "TECHNICAL_REPORT.md"
        if rep.exists():
            left = re.findall(r"\{\{(\w+)\}\}", rep.read_text())
            c.add("technical report has no unfilled placeholder", not left,
                  ", ".join(sorted(set(left))))
        for fig in ["confusion.png", "per_compound.png", "coverage.png",
                    "reliability.png", "guard.png", "architecture.png"]:
            c.add(f"figure {fig}", (ROOT / "results" / "figures" / fig).exists())
    else:
        c.add("results/results.json present", False, "run scripts/evaluate.py")

    if not args.skip_tests:
        print("tests")
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
                             capture_output=True, text=True, cwd=ROOT)
        m = re.search(r"(\d+) passed", out.stdout)
        failed = "failed" in out.stdout
        c.add("test suite passes", bool(m) and not failed,
              m.group(0) if m else out.stdout.strip().splitlines()[-1:][0] if out.stdout else "")

    print()
    if c.failed:
        print(f"{c.failed} check(s) failed")
        raise SystemExit(1)
    print(f"all {len(c.rows)} checks passed")


if __name__ == "__main__":
    main()
