# Hodgkin's Razor, static demo

A page that runs with no server. Every analysis here was computed by
the twin in the repository and cached; the page displays it.

To serve locally:

    python -m http.server --directory site 8080

The live version, which re-runs the twin on a CUDA device, is
`python demo.py --serve`.
