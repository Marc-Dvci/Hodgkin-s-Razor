"""Web service for the network twin.

    python app/server.py --twin models/twin

Serves a single page that takes a matched pair of recordings, returns the
mechanism report, and draws the measured raster beside a raster the twin
produced at the fitted parameters. Bundled examples are read from
`data/examples`, so the page works with no upload and, with `--no-gpu`, with no
CUDA device either.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hodgkins_razor import features as F, nde, ppc, report
from hodgkins_razor import simulator as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = pathlib.Path(__file__).resolve().parent / "static"
EXAMPLES = ROOT / "data" / "examples"

app = FastAPI(title="Hodgkin's Razor")
STATE: dict = {}


def parse_events(raw: bytes, n_elec: int, duration: float) -> np.ndarray:
    """Read a two-column spike table: electrode index (or name) and time.

    Accepts the Tampere `Channel,Time` layout and a plain `electrode,time`
    layout, with or without a header.
    """
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return np.zeros((0, 2))
    rows = []
    names: dict[str, int] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",")]
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        try:
            t = float(b)
        except ValueError:
            continue
        try:
            e = int(float(a))
        except ValueError:
            key = a.split("_")[-1] if "_" in a else a
            e = names.setdefault(key, len(names))
        rows.append((e, t))
    if not rows:
        return np.zeros((0, 2))
    ev = np.array(rows, dtype=float)
    ev = ev[(ev[:, 1] >= 0) & (ev[:, 1] < duration)]
    ev = ev[ev[:, 0] < n_elec]
    return ev


def raster_payload(events: np.ndarray, duration: float, cap: int = 6000) -> dict:
    if events.size == 0:
        return {"e": [], "t": [], "n": 0}
    ev = events
    if ev.shape[0] > cap:
        ev = ev[np.linspace(0, ev.shape[0] - 1, cap).astype(int)]
    return {"e": ev[:, 0].astype(int).tolist(),
            "t": np.round(ev[:, 1], 4).tolist(), "n": int(events.shape[0])}


def rate_payload(events: np.ndarray, duration: float, bin_s: float = 0.05) -> list:
    n = max(int(duration / bin_s), 1)
    if events.size == 0:
        return [0] * n
    h, _ = np.histogram(events[:, 1], bins=np.linspace(0, duration, n + 1))
    return h.astype(int).tolist()


def analyse(base: np.ndarray, treat: np.ndarray, duration: float,
            label: str = "") -> dict:
    twin = STATE["twin"]
    n_elec = S.NELEC
    xb = F.compute(base, n_elec, duration)
    xt = F.compute(treat, n_elec, duration)
    post = twin.posterior(xb, xt, n_samples=STATE["n_samples"])

    guard = None
    twin_raster = None
    if STATE["sim"] is not None:
        g = ppc.check(twin, STATE["sim"], xb, xt, STATE["threshold"], duration,
                      STATE["transient"], n_draws=STATE["n_draws"], post=post)
        guard = {"inside_model": g["inside_model"],
                 "discrepancy": g["discrepancy"], "threshold": g["threshold"]}
        res = g["result"]
        twin_raster = {"base": raster_payload(res.as_events(0), duration),
                       "treat": raster_payload(res.as_events(1), duration),
                       "rate_base": rate_payload(res.as_events(0), duration),
                       "rate_treat": rate_payload(res.as_events(1), duration)}

    body = report.build(post, g if STATE["sim"] is not None else None, xb, xt,
                        meta={"label": label, "duration_s": duration})
    return {"report": body, "guard": guard,
            "features": {"names": list(F.NAMES),
                         "base": xb.tolist(), "treat": xt.tolist()},
            "observed": {"base": raster_payload(base, duration),
                         "treat": raster_payload(treat, duration),
                         "rate_base": rate_payload(base, duration),
                         "rate_treat": rate_payload(treat, duration)},
            "twin": twin_raster,
            "markdown": report.to_markdown(body)}


@app.get("/api/examples")
def examples() -> JSONResponse:
    items = []
    for p in sorted(EXAMPLES.glob("*.json")):
        d = json.loads(p.read_text())
        items.append({"id": p.stem, "label": d.get("label", p.stem),
                      "compound": d.get("compound", ""),
                      "species": d.get("species", ""),
                      "duration": d.get("duration", 60.0),
                      "cached": "analysis" in d})
    return JSONResponse({"examples": items, "gpu": STATE["sim"] is not None})


@app.get("/api/example/{name}")
def example(name: str) -> JSONResponse:
    path = EXAMPLES / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "no such example")
    d = json.loads(path.read_text())
    if "analysis" in d and STATE["prefer_cache"]:
        return JSONResponse(d["analysis"])
    base = np.array(d["baseline"], dtype=float).reshape(-1, 2)
    treat = np.array(d["treated"], dtype=float).reshape(-1, 2)
    return JSONResponse(analyse(base, treat, float(d.get("duration", 60.0)),
                                label=d.get("label", name)))


@app.post("/api/analyse")
async def analyse_upload(baseline: UploadFile = File(...),
                         treated: UploadFile = File(...),
                         duration: float = Form(60.0)) -> JSONResponse:
    b = parse_events(await baseline.read(), S.NELEC, duration)
    t = parse_events(await treated.read(), S.NELEC, duration)
    if b.shape[0] < 20 or t.shape[0] < 20:
        raise HTTPException(400, "each recording needs at least 20 events "
                                 "inside the window")
    return JSONResponse(analyse(b, t, duration, label=baseline.filename or ""))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default="models/twin")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-gpu", action="store_true",
                    help="serve cached analyses only, no simulation")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--transient", type=float, default=5.0)
    args = ap.parse_args()

    device = "cpu" if args.no_gpu else ("cuda" if S.available() else "cpu")
    STATE["twin"] = nde.Twin.load(args.twin, device=device)
    STATE["n_samples"] = args.n_samples
    STATE["n_draws"] = args.n_draws
    STATE["transient"] = args.transient
    STATE["prefer_cache"] = args.no_gpu
    STATE["sim"] = None if args.no_gpu or not S.available() else S.Simulator()

    threshold = args.threshold
    if threshold is None:
        res = ROOT / "results" / "results.json"
        if res.exists():
            d = json.loads(res.read_text())
            threshold = d.get("guard_calibration", {}).get("threshold")
    STATE["threshold"] = float(threshold) if threshold else float("inf")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    import uvicorn
    print(f"twin on {device}; simulation "
          f"{'on' if STATE['sim'] else 'off'}; guard threshold {STATE['threshold']:.2f}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
