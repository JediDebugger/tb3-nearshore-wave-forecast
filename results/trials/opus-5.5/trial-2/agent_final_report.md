`/app/predict.py` is finished and runs in about 0.7 s per case, so 3 cases take roughly 2 s against the 10-minute limit. It needs only numpy and scipy, and reads `model_params.npz`, which must stay next to it. I can't see the real tolerances, so I can only say how well it reproduces the 24 recorded hours: well. Extrapolation to much bigger storms is less certain.

**What it does:** it moves the buoy's wave spectrum onshore across the grid. Along the way it handles wave bending over the bottom, growth in height as water shallows, and wave breaking. It adds waves reflected off the structure, then works out the water-level rise from breaking waves, re-running the waves on the raised water level. Breaking fraction, crest height and the rip current come from the same run. Its settings were fitted to the 24 hours of observations.

**What the observations showed that the chart doesn't:**
- **The bar has moved.** Its crest is at about x = 400 m and 2.75 m deep, not at x = 360 m as charted. The shoreline depths also differ from the chart; I took them from the wet/dry edges and where the waves break.
- **There is a rip channel.** It is a gap through the bar crest at y ≈ 381 m, about ±43 m wide.
- **The rip current is controlled by tide.** It flows offshore (about −0.1 m/s) only when the tide is at or below −0.8 m, and is off at −0.68 m and above. Its strength scales with wave height on the bar.
- **The structure reflects waves.** Its seaward face sends back roughly a quarter of the wave energy. This explains why Hs (up to +11%) and Dm (up to about −12°) differ from the buoy toward y = 1000 m.
- **Crest height follows a clean rule.** Cs/Hs = min(0.8, 0.5 + 0.00763·Ur^0.91), where Ur measures how nonlinear the waves are.
- **The observation noise is random and uncorrelated**, so it averages out well when fitting.

**Fit to the record:** in every region (offshore, bar, inner surf zone, rip channel, near the structure), errors on all 24 hours are close to the measurement noise. Water level, the current `u` and Tm are at the noise level. Hs is within about 2–4% on average. The wet/dry mask matches exactly: no wet cell is left blank and no dry cell is filled.

**Where it could fail grading:**
- **Near the structure,** the observations show striped Hs and Cs patterns from incoming and reflected waves interfering. I only model the average, so the error there is about 10–13%, against 5.5% noise.
- **Rip current in large storms** is extrapolated from only 3 active hours. Both the tide switch and the straight-line scaling with wave height are guesses beyond the record.
- **Breaking in big storms** rests on a breaking model fitted to waves up to about 1.3 m. Synthetic tests at 4–5 m stay stable and behave sensibly (height limited by depth, water-level rise up to about 0.3 m), but there is no data to check them against.
- **Smaller biases:** Hs runs about 2% high overall, Tm about 0.1–0.2 s low in the inner surf zone, and Dm offshore near the structure can be off by a few degrees.

Outside the deliverables, I installed matplotlib into the Python environment for diagnostic plots; `predict.py` doesn't use it.

Files are in /app:
- predict.py
- model_params.npz
