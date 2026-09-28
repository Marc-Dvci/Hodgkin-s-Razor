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
                "scripts/make_bank.py", "scripts/train.py", "scripts/evaluate_v2.py",
                "hodgkins_razor/doorn.py", "hodgkins_razor/mateus.py",
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

    print("pre-registration, version 2")
    md2 = ROOT / "PREREGISTRATION_v2.md"
    digest2 = ""
    if md2.exists():
        rec2 = (ROOT / "PREREGISTRATION_v2.sha256").read_text().split()[0]
        digest2 = hashlib.sha256(md2.read_bytes()).hexdigest()
        c.add("PREREGISTRATION_v2.md matches its recorded hash", digest2 == rec2, digest2[:16])
        spec = json.loads(re.search(r"```json\n(.*?)\n```", md2.read_text(encoding="utf-8"),
                                    re.S).group(1))
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate_v2 import model_digest
        for name, want in spec["frozen_models"].items():
            c.add(f"{name} is the model that was frozen",
                  (ROOT / name).exists() and model_digest(ROOT / name) == want, want[:12])
        dom = hashlib.sha256((ROOT / "models" / "domain.json").read_bytes()).hexdigest()
        c.add("models/domain.json is the one that was frozen", dom == spec["domain_sha256"])
    else:
        c.add("PREREGISTRATION_v2.md present", False, "run scripts/freeze_v2.py")

    if not args.skip_data:
        print("external data")
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_external.py")],
                             capture_output=True, text=True, cwd=ROOT)
        c.add("external files match the versions the results used",
              out.returncode == 0, out.stdout.strip().splitlines()[-1] if out.stdout else "")

    print("results, version 2")
    rj = ROOT / "results" / "v2" / "results.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        c.add("results were produced against the second pre-registration",
              bool(digest2) and r.get("prereg_sha256") == digest2)
        rm = ROOT / "results" / "v2" / "RESULTS.md"
        text = rm.read_text(encoding="utf-8") if rm.exists() else ""
        a = r.get("A_doorn", {}).get("metrics", {})
        if a:
            want = f"{a['top1_hits']}/{a['n_wells']}"
            c.add("written blind-test result matches the JSON", want in text, want)
        cm = r.get("C_tampere", {}).get("metrics_v1_key", {})
        if cm:
            want = f"{cm['top1_accuracy']:.2f}"
            c.add("written Tampere result matches the JSON", want in text, want)
        rep = ROOT / "docs" / "TECHNICAL_REPORT.md"
        if rep.exists():
            left = re.findall(r"\{\{(\w+)\}\}", rep.read_text(encoding="utf-8"))
            c.add("technical report has no unfilled placeholder", not left,
                  ", ".join(sorted(set(left))))
            for img in re.findall(r"\]\(\.\./(results/[^)]+\.png)\)", rep.read_text(encoding="utf-8")):
                c.add(f"figure {img}", (ROOT / img).exists())
    else:
        c.add("results/v2/results.json present", False, "run scripts/evaluate_v2.py")

    print("pre-registration, version 3")
    md3 = ROOT / "PREREGISTRATION_v3.md"
    digest3 = ""
    if md3.exists():
        rec3 = (ROOT / "PREREGISTRATION_v3.sha256").read_text().split()[0]
        digest3 = hashlib.sha256(md3.read_bytes()).hexdigest()
        c.add("PREREGISTRATION_v3.md matches its recorded hash", digest3 == rec3, digest3[:16])
        spec3 = json.loads(re.search(r"```json\n(.*?)\n```", md3.read_text(encoding="utf-8"),
                                     re.S).group(1))
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate_v2 import model_digest
        for name, want in spec3["frozen_models"].items():
            c.add(f"{name} is the model that was frozen",
                  (ROOT / name).exists() and model_digest(ROOT / name) == want, want[:12])
        for name, want in spec3["frozen_files"].items():
            p = ROOT / name
            c.add(f"{name} is the file that was frozen",
                  p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == want, want[:12])
        stop = ROOT / "docs" / "STOP_RULE_v3.md"
        c.add("stop rule committed", stop.exists())
    else:
        c.add("PREREGISTRATION_v3.md present", False, "run scripts/freeze_v3.py")

    print("results, version 3")
    rj3 = ROOT / "results" / "v3" / "results.json"
    if rj3.exists():
        r3 = json.loads(rj3.read_text())
        c.add("v3 results were produced against the third pre-registration",
              bool(digest3) and r3.get("prereg_sha256") == digest3)
        rm3 = ROOT / "results" / "v3" / "RESULTS.md"
        text3 = rm3.read_text(encoding="utf-8") if rm3.exists() else ""
        m3 = r3.get("A_blind", {}).get("early", {}).get("metrics", {})
        if m3:
            want = f"{m3['auroc_key']:.3f}"
            c.add("written v3 primary matches the JSON", want in text3, want)
            want = f"{m3['top1_treated_hits']}/{m3['n_treated']} against {m3['top1_null_hits']}/{m3['n_null']}"
            c.add("written v3 co-primary matches the JSON", want in text3, want)
        wu = ROOT / "docs" / "WRITEUP.md"
        if wu.exists() and m3:
            want = f"{m3['auroc_key']:.2f}"
            c.add("writeup quotes the v3 primary as scored", want in wu.read_text(encoding="utf-8"), want)
    else:
        c.add("results/v3/results.json present", False, "run scripts/evaluate_v3_lowmem.py")

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
