"""Screenshots and a screen recording of the running application.

    python scripts/capture_app.py --out results/capture

Starts the static site, drives the real page, and captures what a reviewer
would see. Everything here is the product; nothing is drawn for the camera.
"""
from __future__ import annotations

import argparse
import http.server
import pathlib
import socketserver
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

SHOTS = [
    ("rat_gabazine", "gabazine", "GABA-A antagonist; the twin names it at 0.81"),
    ("rat_ttx", "ttx", "sodium channel block; named correctly"),
    ("human_kainicacid", "kainate", "AMPA agonist; named correctly"),
    ("rat_control", "control", "vehicle control; nothing is named"),
    ("rat_cnqx", "cnqx", "AMPA antagonist; this window is called wrongly"),
    ("human_gaba", "gaba_human", "GABA-A agonist on human neurons"),
]


def serve(directory: pathlib.Path, port: int):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--out", default="results/capture")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--video", action="store_true", help="also record a walkthrough")
    args = ap.parse_args()

    site = ROOT / args.site
    if not site.exists():
        print("no static site; run scripts/export_static.py first")
        raise SystemExit(1)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright
    httpd = serve(site, args.port)
    url = f"http://127.0.0.1:{args.port}/index.html"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            ctx_args = {"viewport": {"width": 1440, "height": 950},
                        "device_scale_factor": 2}
            if args.video:
                ctx_args["record_video_dir"] = str(out / "video")
                ctx_args["record_video_size"] = {"width": 1440, "height": 950}
            ctx = browser.new_context(**ctx_args)
            page = ctx.new_page()
            page.goto(url)
            page.wait_for_function("document.querySelectorAll('#example option').length > 0")
            page.screenshot(path=str(out / "00_landing.png"))

            for name, slug, note in SHOTS:
                if page.locator(f"#example option[value='{name}']").count() == 0:
                    print("  skip", name)
                    continue
                page.select_option("#example", name)
                page.click("#run")
                page.wait_for_selector("#mechpanel:not([hidden])", timeout=60000)
                page.wait_for_timeout(700)
                page.screenshot(path=str(out / f"10_{slug}_full.png"), full_page=True)
                page.locator("#out").screenshot(path=str(out / f"11_{slug}_verdict.png"))
                page.locator("#mechpanel").screenshot(path=str(out / f"12_{slug}_mech.png"))
                if page.locator("#rasterpanel:not([hidden])").count():
                    page.locator("#rasterpanel").screenshot(
                        path=str(out / f"13_{slug}_raster.png"))
                if page.locator("#classpanel:not([hidden])").count():
                    page.locator("#classpanel").screenshot(
                        path=str(out / f"14_{slug}_class.png"))
                print("  captured", name, "-", note)

            ctx.close()
            browser.close()
    finally:
        httpd.shutdown()

    shots = sorted(out.glob("*.png"))
    print(f"wrote {len(shots)} images to {out}")
    if args.video:
        vids = sorted((out / "video").glob("*.webm"))
        print(f"wrote {len(vids)} recording(s) to {out / 'video'}")


if __name__ == "__main__":
    main()
