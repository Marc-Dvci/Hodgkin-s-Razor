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

from hodgkins_razor import design, examples_key, features as F, nde, ppc, report
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


def view_for(n_elec: int) -> str:
    """The recording system whose layout matches the upload."""
    for v in S.VIEWS:
        if S.n_electrodes(v) == n_elec and v in STATE["twins"]:
            return v
    return next(iter(STATE["twins"]))


def analyse(base: np.ndarray, treat: np.ndarray, duration: float,
            label: str = "", n_elec: int = 16) -> dict:
    view = view_for(n_elec)
    twin = STATE["twins"][view]
    n_elec = S.n_electrodes(view)
    xb = F.compute(base, n_elec, duration)
    xt = F.compute(treat, n_elec, duration)
    post = twin.posterior(xb, xt, n_samples=STATE["n_samples"])

    # Typicality needs no simulation, so every analysis carries it; the
    # predictive check is added when a CUDA device is present.
    guard = {"inside_model": True}
    typ = ppc.Typicality.for_twin(twin)
    if typ is not None and typ.threshold is not None:
        t = typ.of_pair(twin, xb, xt)
        guard.update({"typicality": t, "typicality_threshold": typ.threshold})
        guard["inside_model"] = t <= typ.threshold
    twin_raster = None
    if STATE["sim"] is not None:
        thr = STATE["threshold"].get(view, float("inf"))
        g = ppc.check(twin, STATE["sim"], xb, xt, thr, duration,
                      STATE["transient"], n_draws=STATE["n_draws"], post=post)
        guard.update({"discrepancy": g["discrepancy"], "threshold": g["threshold"]})
        guard["inside_model"] = guard["inside_model"] and g["inside_model"]
        res = g["result"]
        k = 2 * g.get("closest_draw", 0)
        eb = S.view_events(res.raw_events(k), view)[0]
        et = S.view_events(res.raw_events(k + 1), view)[0]
        twin_raster = {"base": raster_payload(eb, duration),
                       "treat": raster_payload(et, duration),
                       "rate_base": rate_payload(eb, duration),
                       "rate_treat": rate_payload(et, duration),
                       "draw": g.get("closest_draw", 0),
                       "n_draws": g.get("n_draws", 1)}

    body = report.build(post, guard if len(guard) > 1 else None, xb, xt,
                        meta={"label": label, "duration_s": duration,
                              "recording_system": view})
    # When the top two mechanisms are not separated and the recording is
    # inside the model, rank the follow-up tool compounds by how far apart the
    # two hypotheses predict their recordings. This is the twin's prediction,
    # computed by simulation; it has not been tested on recordings.
    if STATE["sim"] is not None and body["verdict"] != "outside_model":
        body["next_experiment"] = design.recommend(
            STATE["sim"], post, body["mechanisms"], duration, STATE["transient"],
            view=view, n_draws=16)
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
                      "cached": "analysis" in d,
                      "default": p.stem == examples_key.DEFAULT,
                      **examples_key.describe(d.get("compound", ""), d.get("analysis"))})
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
                                label=d.get("label", name),
                                n_elec=int(d.get("n_elec", 16))))


@app.post("/api/analyse")
async def analyse_upload(baseline: UploadFile = File(...),
                         treated: UploadFile = File(...),
                         duration: float = Form(60.0),
                         n_elec: int = Form(16)) -> JSONResponse:
    b = parse_events(await baseline.read(), n_elec, duration)
    t = parse_events(await treated.read(), n_elec, duration)
    if b.shape[0] < 20 or t.shape[0] < 20:
        raise HTTPException(400, "each recording needs at least 20 events "
                                 "inside the window")
    return JSONResponse(analyse(b, t, duration, label=baseline.filename or "",
                                n_elec=n_elec))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", default="models/twin_v2_grid16,models/twin_v2_grid12",
                    help="one twin per recording system, comma separated")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-gpu", action="store_true",
                    help="serve cached analyses only, no simulation")

    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--transient", type=float, default=5.0)
    args = ap.parse_args()

    device = "cpu" if args.no_gpu else ("cuda" if S.available() else "cpu")
    STATE["twins"] = {}
    for path in args.twins.split(","):
        t = nde.Twin.load(ROOT / path, device=device)
        STATE["twins"][t.meta.get("view", "grid16")] = t
    STATE["n_samples"] = args.n_samples
    STATE["n_draws"] = args.n_draws
    STATE["transient"] = args.transient
    STATE["prefer_cache"] = args.no_gpu
    STATE["sim"] = None if args.no_gpu or not S.available() else S.Simulator()
    # Predictive-check thresholds as calibrated by the pre-registered evaluation.
    STATE["threshold"] = {}
    res = ROOT / "results" / "v2" / "results.json"
    if res.exists():
        d = json.loads(res.read_text())
        STATE["threshold"] = {v: g["ppc"] for v, g in d.get("guard_thresholds", {}).items()}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    import uvicorn
    print(f"twins {list(STATE['twins'])} on {device}; simulation "
          f"{'on' if STATE['sim'] else 'off'}; predictive-check thresholds {STATE['threshold']}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
