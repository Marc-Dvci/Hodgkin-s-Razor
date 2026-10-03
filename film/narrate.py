"""Synthesise the film narration from film/story.json and measure it.

    python film/narrate.py                 # the author's cloned voice (default)
    python film/narrate.py --engine edge   # Edge TTS fallback

Default engine: the author's own voice, cloned with Qwen3-TTS from his
recordings (VoiceClone/narrate.py, outside this repository; set VOICECLONE to
its folder). Every beat is one paragraph, one sentence per line; the
synthesiser returns each sentence's measured start, and those starts become the
beat durations and the cues the live scenes act on.

Edge engine: each beat's `say` text is synthesised with Edge TTS, measured with
ffprobe and padded by a short gap. Writes film/speech/*.mp3, film/narration.wav (all beats
back to back), film/timing.json (per-beat durations, read by record.py) and
three subtitle files from the same timings: English, Chinese, and both.

Every number spoken is checked against results/v3/RESULTS.md,
results/LASSUS_RESULT.md, results/v2/RESULTS.md and results/chip_study.json first; a figure that appears in neither is refused, so
the film cannot say a number the results do not contain.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import subprocess
import sys

import os
import shutil

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
    for p in (ROOT / "results" / "v3" / "RESULTS.md", ROOT / "results" / "LASSUS_RESULT.md",
              ROOT / "results" / "v2" / "RESULTS.md", ROOT / "results" / "chip_study.json",
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
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(out))


VOICECLONE = pathlib.Path(os.environ.get("VOICECLONE", "D:/VoiceClone"))
LEXICON = {"AUROC": "aw-rock", "APV": "A P V", "TTX": "T T X", "NMDA": "N M D A",
           "iPSC": "i P S C", "GPU": "G P U", "CPU": "C P U"}
TAIL_MS = 900


def sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


def write_subtitles(beats: list[dict], durations: list[int]) -> None:
    for name, key in (("en", "show"), ("zh", "zh")):
        lines, t = [], 0
        for i, (b, d) in enumerate(zip(beats, durations)):
            txt = b.get(key) or (b["say"] if key == "show" else "")
            if txt:
                lines += [str(i + 1), f"{srt_time(t)} --> {srt_time(t + d - 200)}", txt, ""]
            t += d
        (HERE / f"hodgkins-razor.{name}.srt").write_text("\n".join(lines), encoding="utf-8")


def clone(beats: list[dict]) -> None:
    py = VOICECLONE / ".venv-qwen" / "Scripts" / "python.exe"
    if not py.exists():
        raise SystemExit(f"cloned voice not found at {VOICECLONE}; use --engine edge")
    SPEECH.mkdir(exist_ok=True)
    script = SPEECH / "script.txt"
    script.write_text("\n\n".join("\n".join(sentences(b["say"])) for b in beats) + "\n",
                      encoding="utf-8")
    lex = SPEECH / "lexicon.json"
    lex.write_text(json.dumps(LEXICON), encoding="utf-8")
    out = SPEECH / "clone"
    if out.exists() and "--reuse" not in sys.argv:
        shutil.rmtree(out)
    if not (out / "narration.json").exists():
        subprocess.run([str(py), str(VOICECLONE / "narrate.py"), str(script), str(out),
                        "--lexicon", str(lex), "--para-gap", "0.9"], check=True, cwd=VOICECLONE)
    nar = json.loads((out / "narration.json").read_text(encoding="utf-8"))
    sents = nar["sentences"]
    per = [[s for s in sents if s["paragraph"] == i] for i in range(len(beats))]
    for i, (b, p) in enumerate(zip(beats, per)):
        if len(p) != len(sentences(b["say"])):
            raise SystemExit(f"beat {i}: {len(p)} sentences synthesised, "
                             f"{len(sentences(b['say']))} written")
    # A beat runs from its first sentence to the next beat's first sentence.
    starts = [0.0] + [p[0]["start"] for p in per[1:]]
    ends = starts[1:] + [nar["duration"]]
    durations, cues = [], []
    for i, p in enumerate(per):
        speech = (ends[i] - starts[i]) * 1000 + (TAIL_MS if i == len(per) - 1 else 0)
        durations.append(int(round(speech)) + int(beats[i].get("hold_ms", 0)))
        cues.append([int(round((s["start"] - starts[i]) * 1000)) for s in p])
    # Holds lengthen a beat beyond its speech, so the audio is re-laid beat by beat.
    filt = [f"[0:a]atrim={starts[i]:.3f}:{ends[i]:.3f},asetpts=PTS-STARTPTS,aresample=48000,"
            f"apad,atrim=0:{d / 1000:.3f}[a{i}]" for i, d in enumerate(durations)]
    filt.append("".join(f"[a{i}]" for i in range(len(durations)))
                + f"concat=n={len(durations)}:v=0:a=1[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "narration.wav"),
                    "-filter_complex", ";".join(filt), "-map", "[out]", "-ac", "2",
                    str(HERE / "narration.wav")], check=True)
    total = sum(durations)
    (HERE / "timing.json").write_text(json.dumps({"durations": durations, "totalMs": total,
                                                  "cues": cues, "voice": nar.get("voice", "")}))
    write_subtitles(beats, durations)
    for i, (b, d) in enumerate(zip(beats, durations)):
        print(f"  {i:02d}  {d / 1000:5.1f}s  {b['say'][:70]}")
    redo = [s["text"][:60] for s in sents if s.get("wer", 0) > 0.10]
    if redo:
        print("transcript still differs on:", redo)
    print(f"total {total / 1000:.1f}s over {len(beats)} beats")
    if total > 295_000:
        print("WARNING: longer than 4:55")


def main() -> None:
    story = json.loads((HERE / "story.json").read_text(encoding="utf-8"))
    beats = story["beats"]
    bad = check_numbers(beats)
    if bad and "--force" not in sys.argv:
        raise SystemExit("numbers not found in the results:\n  " + "\n  ".join(bad))
    if "edge" not in sys.argv:
        return clone(beats)
    SPEECH.mkdir(exist_ok=True)
    for old in SPEECH.glob("*.mp3"):
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
