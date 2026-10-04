"""Rebuild the cloned-voice narration after editing a few sentences.

    python film/resplice.py film/speech/patch [--take "sentence text=dir:index" ...]

Synthesising the whole film again costs a GPU run and re-rolls every take.
This keeps every sentence whose text is unchanged (film/speech/clone_prev,
the previous run), takes new or re-rolled sentences from the patch runs given
(each a VoiceClone/narrate.py output folder), and lays them out with the same
gaps narrate.py uses. It writes film/speech/clone/narration.json and
narration.wav, which `python film/narrate.py --reuse` then measures.

A sentence listed in REDO is never reused from the previous run.
`--take` picks one take explicitly when a patch run holds several.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np
from scipy.io import wavfile

HERE = pathlib.Path(__file__).resolve().parent
SPEECH = HERE / "speech"
GAP, PARA_GAP = 0.35, 0.9
REDO = {"Remove the pairing, and it falls to 0.66."}   # the old take repeats a syllable


def sentences(text: str) -> list[str]:
    sys.path.insert(0, str(HERE))
    from narrate import sentences as split
    return split(text)


def load_run(folder: pathlib.Path) -> list[dict]:
    nar = json.loads((folder / "narration.json").read_text(encoding="utf-8"))
    for s in nar["sentences"]:
        s["_dir"] = folder
    return nar["sentences"]


def wav(s: dict) -> tuple[int, np.ndarray]:
    path = s["_dir"] / s["file"] if not pathlib.Path(s["file"]).is_absolute() else pathlib.Path(s["file"])
    if not path.exists():
        path = s["_dir"] / "lines" / pathlib.Path(s["file"]).name
    rate, x = wavfile.read(path)
    return rate, x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)


def main() -> None:
    args = sys.argv[1:]
    takes: dict[str, tuple[str, int]] = {}
    patch_dirs = []
    i = 0
    while i < len(args):
        if args[i] == "--take":
            text, ref = args[i + 1].rsplit("=", 1)
            d, k = ref.rsplit(":", 1)
            takes[text] = (str(pathlib.Path(d).resolve()), int(k))
            i += 2
        else:
            patch_dirs.append(pathlib.Path(args[i]).resolve())
            i += 1
    prev = SPEECH / "clone_prev"
    cur = SPEECH / "clone"
    if not prev.exists():
        shutil.copytree(cur, prev)
    old = load_run(prev)
    patches = [s for d in patch_dirs for s in load_run(d)]
    beats = json.loads((HERE / "story.json").read_text(encoding="utf-8"))["beats"]

    chosen = []
    for p, b in enumerate(beats):
        for text in sentences(b["say"]):
            if text in takes:
                d, k = takes[text]
                s = next(x for x in patches if str(x["_dir"]) == d and x["index"] == k)
            elif text not in REDO and any(x["text"] == text for x in old):
                s = next(x for x in old if x["text"] == text)
            else:
                cands = [x for x in patches if x["text"] == text]
                if not cands:
                    raise SystemExit(f"no take for: {text}")
                s = min(cands, key=lambda x: (x.get("wer", 0), x["duration"]))
            chosen.append((p, text, s))

    rate = None
    out, rows, t = [], [], 0.0
    for j, (p, text, s) in enumerate(chosen):
        r, x = wav(s)
        rate = rate or r
        if r != rate:
            raise SystemExit("sample rates differ between takes")
        if j:
            gap = PARA_GAP if p != chosen[j - 1][0] else GAP
            out.append(np.zeros(int(round(gap * rate)), dtype=np.float32))
            t += gap
        dur = x.shape[0] / rate
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        shift = t - s["start"]
        row.update({"index": j + 1, "paragraph": p, "start": round(t, 3),
                    "end": round(t + dur, 3), "duration": round(dur, 3),
                    "file": str((s["_dir"] / "lines" / pathlib.Path(s["file"]).name)),
                    "source": str(s["_dir"].name)})
        if "words" in s:
            row["words"] = [{**w, "start": w["start"] + shift, "end": w["end"] + shift}
                            if "start" in w else w for w in s["words"]]
        rows.append(row)
        out.append(x)
        t += dur
    audio = np.concatenate(out)
    cur.mkdir(exist_ok=True)
    wavfile.write(cur / "narration.wav", rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    meta = json.loads((prev / "narration.json").read_text(encoding="utf-8"))
    meta.update({"sentences": rows, "duration": round(t, 3)})
    (cur / "narration.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    for row in rows:
        print(f"  {row['index']:02d} p{row['paragraph']:02d} {row['source']:10s} "
              f"{row['duration']:5.2f}s  {row['text'][:60]}")
    print(f"total {t:.1f}s")


if __name__ == "__main__":
    main()
