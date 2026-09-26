"""Synthesise the narration and write word timings for the edit.

    python scripts/narrate.py --out results/narration

Each beat of `docs/VIDEO_SCRIPT.md` becomes one audio file and one JSON file of
word boundaries, so cuts are placed on measured speech rather than guessed
durations. Numbers are checked against `results/RESULTS.md` before synthesis:
a figure that is not in the results file is refused rather than narrated.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOICE = "en-GB-RyanNeural"


def beats(script: str) -> list[dict]:
    """Pull the narration blocks out of the script, in order."""
    out = []
    section = None
    for block in script.split("\n## "):
        head = block.splitlines()[0].strip()
        if "·" not in head:
            continue
        section = head
        for m in re.finditer(r'"([^"]+)"', block):
            text = " ".join(m.group(1).split())
            if len(text) > 20:
                out.append({"section": section, "text": text})
    return out


def unverified_numbers(text: str, allowed: set[str]) -> list[str]:
    """Spelled-out or digit figures that do not appear in the results file."""
    bad = []
    for tok in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        if tok not in allowed:
            bad.append(tok)
    return bad


def mp3_duration(path: pathlib.Path) -> float:
    """Length of an MP3 in seconds, from its frame headers.

    edge-tts does not always emit word boundaries, so the duration is measured
    from the audio rather than assumed from the marks.
    """
    RATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
             2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
    SR = {0: [44100, 48000, 32000], 1: [22050, 24000, 16000],
          2: [11025, 12000, 8000]}
    data = path.read_bytes()
    i, total = 0, 0.0
    while i < len(data) - 4:
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        ver = (data[i + 1] >> 3) & 3          # 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
        layer = (data[i + 1] >> 1) & 3
        bi = (data[i + 2] >> 4) & 0xF
        si = (data[i + 2] >> 2) & 3
        pad = (data[i + 2] >> 1) & 1
        if layer != 1 or bi in (0, 15) or si == 3:
            i += 1
            continue
        kbps = RATES[1 if ver == 3 else 2][bi]
        rate = SR[{3: 0, 2: 1, 0: 2}.get(ver, 1)][si]
        spf = 1152 if ver == 3 else 576
        size = int((spf // 8) * 1000 * kbps / rate) + pad
        if size <= 4:
            i += 1
            continue
        total += spf / rate
        i += size
    return total


async def synth(items: list[dict], out: pathlib.Path, voice: str) -> None:
    import edge_tts
    for i, item in enumerate(items):
        stem = out / f"{i:02d}"
        comm = edge_tts.Communicate(item["text"], voice)
        marks = []
        with open(stem.with_suffix(".mp3"), "wb") as f:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    marks.append({"word": chunk["text"],
                                  "offset_s": chunk["offset"] / 1e7,
                                  "duration_s": chunk["duration"] / 1e7})
        dur = ((marks[-1]["offset_s"] + marks[-1]["duration_s"]) if marks
               else mp3_duration(stem.with_suffix(".mp3")))
        stem.with_suffix(".json").write_text(json.dumps(
            {"section": item["section"], "text": item["text"],
             "duration_s": round(dur, 3), "words": marks,
             "timing_from": "word boundaries" if marks else "audio frames"},
            indent=1))
        print(f"  {i:02d}  {dur:5.1f}s  {item['section'][:46]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="docs/VIDEO_SCRIPT.md")
    ap.add_argument("--out", default="results/narration")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    script = (ROOT / args.script).read_text(encoding="utf-8")
    items = beats(script)
    if not items:
        raise SystemExit("no narration found in the script")

    results = ROOT / "results" / "RESULTS.md"
    allowed: set[str] = set()
    if results.exists():
        allowed = set(re.findall(r"\b\d+(?:\.\d+)?\b", results.read_text()))
    allowed |= set(re.findall(r"\b\d+(?:\.\d+)?\b",
                              (ROOT / "README.md").read_text()))

    problems = []
    for item in items:
        bad = unverified_numbers(item["text"], allowed)
        if bad:
            problems.append((item["section"], bad, item["text"][:70]))
    if problems and not args.allow_unverified:
        print("figures not found in the results file:")
        for sec, bad, text in problems:
            print(f"  {sec}: {bad}  in  \"{text}...\"")
        raise SystemExit("refusing to narrate a figure that is not in the results")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} beats, voice {args.voice}")
    asyncio.run(synth(items, out, args.voice))
    total = sum(json.loads(p.read_text())["duration_s"]
                for p in sorted(out.glob("*.json")))
    print(f"total narration {total / 60:.1f} min "
          f"({'inside' if total <= 300 else 'OVER'} the five-minute ceiling)")


if __name__ == "__main__":
    main()
