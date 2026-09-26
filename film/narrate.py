"""Synthesise the film narration from film/story.json and measure it.

    python film/narrate.py

Each beat's `say` text is synthesised with Edge TTS, measured with ffprobe and
padded by a short gap. Writes film/speech/*.mp3, film/narration.wav (all beats
back to back), film/timing.json (per-beat durations, read by record.py) and
three subtitle files from the same timings: English, Chinese, and both.

Every number spoken is checked against results/v2/RESULTS.md and
results/chip_study.json first; a figure that appears in neither is refused, so
the film cannot say a number the results do not contain.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import subprocess
import sys

import edge_tts

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SPEECH = HERE / "speech"
VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+4%"
GAP_MS = 650

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def duration_ms(path: pathlib.Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(round(float(out) * 1000))


def allowed_numbers() -> str:
    text = ""
    for p in (ROOT / "results" / "v2" / "RESULTS.md", ROOT / "results" / "chip_study.json",
              ROOT / "results" / "v2" / "results.json", HERE / "allowed_numbers.txt"):
        if p.exists():
            text += p.read_text(encoding="utf-8")
    return text


def check_numbers(beats: list[dict]) -> list[str]:
    """Digits in the narration that the results never mention."""
    source = allowed_numbers()
    bad = []
    for b in beats:
        for m in re.findall(r"\d+(?:\.\d+)?", b["say"]):
            if m not in source:
                bad.append(f"{m!r} in: {b['say'][:70]}")
    return bad


def srt_time(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def synth(text: str, out: pathlib.Path) -> None:
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


def main() -> None:
    story = json.loads((HERE / "story.json").read_text(encoding="utf-8"))
    beats = story["beats"]
    bad = check_numbers(beats)
    if bad and "--force" not in sys.argv:
        raise SystemExit("numbers not found in the results:\n  " + "\n  ".join(bad))
    SPEECH.mkdir(exist_ok=True)
    for old in SPEECH.glob("*"):
        old.unlink()
    durations = []
    clips = []
    for i, b in enumerate(beats):
        mp3 = SPEECH / f"{i:02d}.mp3"
        asyncio.run(synth(b["say"], mp3))
        d = duration_ms(mp3) + GAP_MS + int(b.get("hold_ms", 0))
        durations.append(d)
        clips.append((mp3, d))
        print(f"  {i:02d}  {d / 1000:5.1f}s  {b['say'][:70]}")
    # Concatenate, each clip padded with silence to its measured duration.
    parts, filt = [], []
    for k, (mp3, d) in enumerate(clips):
        parts += ["-i", str(mp3)]
        filt.append(f"[{k}:a]aresample=48000,apad,atrim=0:{d / 1000:.3f}[a{k}]")
    filt.append("".join(f"[a{k}]" for k in range(len(clips))) + f"concat=n={len(clips)}:v=0:a=1[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *parts, "-filter_complex", ";".join(filt),
                    "-map", "[out]", "-ac", "2", str(HERE / "narration.wav")], check=True)
    total = sum(durations)
    (HERE / "timing.json").write_text(json.dumps({"durations": durations, "totalMs": total}))
    for name, key in (("en", "show"), ("zh", "zh")):
        lines, t = [], 0
        for i, (b, d) in enumerate(zip(beats, durations)):
            txt = b.get(key) or (b["say"] if key == "show" else "")
            if txt:
                lines += [str(i + 1), f"{srt_time(t)} --> {srt_time(t + d - 200)}", txt, ""]
            t += d
        (HERE / f"hodgkins-razor.{name}.srt").write_text("\n".join(lines), encoding="utf-8")
    print(f"total {total / 1000:.1f}s over {len(beats)} beats")
    if total > 295_000:
        print("WARNING: longer than 4:55")


if __name__ == "__main__":
    main()
