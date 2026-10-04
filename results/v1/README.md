# Version 1 results, kept for the record

These are the outputs of version 1, scored under `PREREGISTRATION.md` (hash
`c40708bb19668164`) on the Tampere plates. The code that produced them is the
git tag `v1` (commit `9e554cf`). The model they describe no longer runs on the
current code: version 2 changed the simulator (asynchronous release, per-system
dead time), the parameter table and the inference.

The recorded set was scored four times in version 1:

| Scoring | Model | Top-1 |
|---|---|---|
| 1 | as pre-registered | 0.091 |
| 2 | corrected pharmacology | 0.167 |
| 3 | restricted to the recorded baseline domain | 0.303 |
| 4 | domain reweighted rather than restricted | 0.121 |

That is why version 2 treats the Tampere plates as a development set and was
tested blind on other data (`PREREGISTRATION_v2.md`, `results/v2/`).
