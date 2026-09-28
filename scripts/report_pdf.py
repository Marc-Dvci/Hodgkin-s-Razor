"""Print docs/TECHNICAL_REPORT.md to docs/TECHNICAL_REPORT.pdf.

    python scripts/report_pdf.py

Markdown to HTML with python-markdown, then Chromium (Playwright) prints it on
A4 with page numbers. Figures are the PNGs the report links, embedded as files.
"""
from __future__ import annotations

import pathlib

import markdown
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "TECHNICAL_REPORT.md"
OUT = ROOT / "docs" / "TECHNICAL_REPORT.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.6pt;
        break-inside: auto; }
th, td { border-bottom: 1px solid #e3e3e3; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f5f5f3; }
tr { break-inside: avoid; }
code { font-family: Consolas, ui-monospace, monospace; font-size: 8.8pt; background: #f4f4f2;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f4f4f2; padding: 6pt 8pt; border-radius: 4pt; font-size: 8.4pt;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 92mm; display: block; margin: 8pt auto 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; text-align: center; }
hr { border: 0; border-top: 1px solid #ddd; margin: 12pt 0; }
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    base = (ROOT / "docs").as_uri() + "/"
    html = (f"<!doctype html><html><head><meta charset='utf-8'><base href='{base}'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    tmp = ROOT / "docs" / "_report_print.html"
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(tmp.as_uri(), wait_until="load")
        page.pdf(path=str(OUT), format="A4", print_background=True,
                 display_header_footer=True,
                 header_template="<div></div>",
                 footer_template="<div style='font-size:7pt;color:#888;width:100%;text-align:center'>"
                                 "Hodgkin's Razor · technical report · <span class='pageNumber'></span> / "
                                 "<span class='totalPages'></span></div>",
                 margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        browser.close()
    tmp.unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
