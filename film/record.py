"""Render the film to an MP4, one frame at a time, on a virtual clock.

    python film/narrate.py      # voice, timing.json, subtitles
    python film/record.py       # frames + narration -> film/hodgkins-razor.mp4

The film page is served locally and rendered with Chromium's BeginFrame
control: every CSS transition and timer advances exactly one frame interval per
frame, and each frame is returned as a lossless screenshot and piped to ffmpeg.
No screen recorder sits between the page and the file, so there are no smeared
keyframes and no dropped frames, and the picture does not depend on the speed
of the machine.

Every image on screen is either a capture of the running application or a
figure written by scripts/figures_v2.py from the results.
"""
from __future__ import annotations

import base64
import http.server
import json
import pathlib
import socketserver
import subprocess
import sys
import threading
import time

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
VIEWPORT = (1600, 900)
FRAME = (1920, 1080)
FPS = 30
TAIL = 1.5
CHROMIUM_ARGS = ["--enable-begin-frame-control", "--run-all-compositor-stages-before-draw",
                 "--disable-new-content-rendering-timeout", "--disable-threaded-animation",
                 "--disable-threaded-scrolling", "--disable-checker-imaging",
                 "--force-color-profile=srgb", "--hide-scrollbars"]
PREVIEW = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else None


def serve(port: int):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(ROOT), **k)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def encoder(out: pathlib.Path, seconds: float, total_ms: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
         "-i", str(HERE / "narration.wav"),
         "-filter_complex",
         f"[0:v]scale={FRAME[0]}:{FRAME[1]}:flags=lanczos,fade=t=in:st=0:d=0.5,"
         f"fade=t=out:st={seconds - 0.8:.2f}:d=0.8[v];"
         f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=0:{seconds:.3f}[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{seconds:.2f}",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    timing = json.loads((HERE / "timing.json").read_text())
    durations, total = timing["durations"], timing["totalMs"]
    seconds = min(total / 1000 + TAIL, PREVIEW or 1e9)
    out = HERE / ("preview.mp4" if PREVIEW else "hodgkins-razor.mp4")
    httpd = serve(8766)
    ff = encoder(out, seconds, total)
    interval = 1000 / FPS
    frames = int(round(seconds * FPS))
    starts = [sum(durations[:i]) for i in range(len(durations))]
    stills = {int(round((s + min(2500, d * 0.6)) / interval)): i for i, (s, d) in enumerate(zip(starts, durations))}
    (HERE / "stills").mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        ctx = browser.new_context(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
        page = ctx.new_page()
        page.on("pageerror", lambda e: print("page error:", e))
        page.goto("http://127.0.0.1:8766/film/index.html", wait_until="load")
        page.wait_for_function("() => window.Film && window.Film.ready", polling=100, timeout=60000)
        beats = page.evaluate("() => window.Film.beats")
        if beats != len(durations):
            raise SystemExit(f"timing has {len(durations)} beats, the story has {beats}")
        page.wait_for_timeout(400)
        cdp = ctx.new_cdp_session(page)
        expired = threading.Event()
        cdp.on("Emulation.virtualTimeBudgetExpired", lambda _e: expired.set())
        ticks = float(cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})["virtualTimeTicksBase"])
        page.evaluate("(a) => { window.__FILM_TIMING = a[0]; window.__FILM_CUES = a[1]; window.Film.start(); }",
                      [durations, timing.get("cues")])
        last = None
        t0 = time.time()
        for n in range(frames):
            expired.clear()
            cdp.send("Emulation.setVirtualTimePolicy", {"policy": "advance", "budget": interval})
            shot = cdp.send("HeadlessExperimental.beginFrame",
                            {"frameTimeTicks": ticks, "interval": interval, "noDisplayUpdates": False,
                             "screenshot": {"format": "png"}})
            ticks += interval
            deadline = time.time() + 20
            while not expired.is_set() and time.time() < deadline:
                page.wait_for_timeout(2)
            if "screenshotData" in shot:
                last = base64.b64decode(shot["screenshotData"])
            if last is None:
                continue
            ff.stdin.write(last)
            if n in stills:
                (HERE / "stills" / f"beat{stills[n]:02d}.png").write_bytes(last)
            if n % 600 == 0:
                print(f"  {n / FPS:6.1f}s  {(time.time() - t0) / max(n, 1) * 1000:4.0f} ms/frame", flush=True)
        missed = page.evaluate("() => window.Film.missed")
        browser.close()
    ff.stdin.close()
    if missed:
        print(f"WARNING: {len(missed)} live-scene selectors matched nothing: {sorted(set(missed))}")
    ff.wait()
    httpd.shutdown()
    if ff.returncode:
        raise SystemExit(ff.stderr.read().decode(errors="replace")[-3000:])
    print(f"wrote {out} ({seconds:.1f}s)")


if __name__ == "__main__":
    main()
