"""Export the application as a static site.

    python scripts/export_static.py --out site

Writes a folder that serves from any static host with no server and no
installation: the same page, reading analyses that were computed here and
cached into the example files. Use it when the reviewer needs a link rather
than a checkout.

Requires the examples to have been built with `make_examples.py --cache`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
EXAMPLES = ROOT / "data" / "examples"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--title", default="Hodgkin's Razor")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)

    items = []
    n_cached = 0
    for p in sorted(EXAMPLES.glob("*.json")):
        d = json.loads(p.read_text())
        if "analysis" not in d:
            continue
        # Only the analysis is needed; the raw events are already inside it.
        (out / "data" / p.name).write_text(json.dumps(d["analysis"]))
        items.append({"id": p.stem, "label": d.get("label", p.stem),
                      "compound": d.get("compound", ""),
                      "species": d.get("species", ""),
                      "duration": d.get("duration", 60.0), "cached": True})
        n_cached += 1
    if not items:
        raise SystemExit("no cached analyses; run "
                         "scripts/make_examples.py --cache first")
    (out / "data" / "index.json").write_text(
        json.dumps({"examples": items, "gpu": False}))

    html = (STATIC / "index.html").read_text()
    html = html.replace("/static/app.js", "app.js").replace("/static/evidence.js", "evidence.js")
    # The upload path needs a server, so it is removed rather than left broken.
    start = html.find('    <details style="margin-top:14px">')
    end = html.find("</details>", start)
    if start != -1 and end != -1:
        html = html[:start] + html[end + len("</details>\n"):]
    html = html.replace(
        '<span class="sub" id="mode"></span>',
        '<span class="sub" id="mode"></span>'
        '<span class="sub">analyses computed ahead of time; '
        'the repository runs the twin live</span>')
    (out / "index.html").write_text(html)

    js = (STATIC / "app.js").read_text()
    js = js.replace('await fetch("/api/examples")', 'await fetch("data/index.json")')
    js = js.replace('await fetch(`/api/example/${id}`)', 'await fetch(`data/${id}.json`)')
    js = js.replace('$("upload").addEventListener("click", runUpload);', "")
    js = js.replace('const fb = $("fb").files[0], ft = $("ft").files[0];',
                    'const fb = null, ft = null;')
    (out / "app.js").write_text(js)
    # The evidence views fetch evidence.json relative to their own script.
    for name in ("evidence.js", "evidence.json"):
        if not (STATIC / name).exists():
            raise SystemExit(f"app/static/{name} missing; run scripts/export_evidence.py")
        shutil.copy(STATIC / name, out / name)
    (out / ".nojekyll").write_text("")

    readme = [
        "# Hodgkin's Razor, static demo", "",
        "A page that runs with no server. Every analysis here was computed by",
        "the twin in the repository and cached; the page displays it.", "",
        "To serve locally:", "", "    python -m http.server --directory site 8080",
        "", "The live version, which re-runs the twin on a CUDA device, is",
        "`python demo.py --serve`.", ""]
    (out / "README.md").write_text("\n".join(readme))
    print(f"wrote {out} with {n_cached} analyses")
    print("serve with: python -m http.server --directory", args.out, "8080")


if __name__ == "__main__":
    main()
