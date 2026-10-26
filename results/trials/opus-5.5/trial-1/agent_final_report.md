I've built `/app/predict.py`. Against the 24 observed hours it fits well; for the bigger storms and new directions it will be graded on, I'd put my confidence at moderate to low.

**What it does.** `predict.py` loads `wavemodel.py` and `params.json` from its own folder. It runs 3 cases in about 30 s and leaves NaN only on dry cells.

- **Bathymetry.** The chart turned out to be a smooth beach profile plus a bar. The observations put the bar about 1.5 m high at x≈390 m, roughly 40 m further onshore than the chart shows. They also show a rip channel through the bar at y≈380 m that isn't on the chart. The structure is treated as land.
- **Waves.** A spectral model steps the buoy spectrum shoreward from x=0 (12 frequencies, 5° direction bins). It includes refraction, shoaling and depth-limited breaking, with the breaking settings fitted to all 24 hours. Mean water level comes from the wave forcing, recomputed together with the waves.
- **Fitted add-ons:**
  - **Crest height:** a ratio to Hs that rises from 0.5 offshore to a cap of 0.80 in shallow water (fit error about 0.008).
  - **Rip current `u`:** a fixed jet shape whose strength switches on when Hs/depth over the bar passes about 0.5. In the data it only appears at low tide.
  - **Reflection:** the structure reflects some energy back offshore, which raises Hs and turns Dm in front of it.

**Check against observations.** Error here means average difference per region and hour, so the random measurement noise cancels:

| Field | Typical error | Worst case (region) |
|---|---|---|
| Hs | ~0.01–0.02 m | 0.045 m (rip) |
| Dm | ~1–2° | 4° (structure) |
| Qb | ≤0.02 | 0.05 |
| mwl | ~0.002 m | 0.005 m |
| Cs | ~0.01 m | 0.08 m (structure) |

There are no cells where my output is NaN but the observations are wet. I only re-checked 12 of the 24 hours after the last change. A 3-case test with waves 2–4× bigger than anything recorded (Hs 2.6–4.9 m), at new tides and angles, gave values in sensible ranges with no NaN gaps.

**Why confidence is only moderate to low:**
- **Unknown tolerances.** Every region of every case must pass, and the worst ones are near the structure and in the rip channel.
- **Rip current.** It's calibrated on only 3 low-tide hours with a strong rip. I hand-picked the channel's shape instead of fitting it, so its strength in a big storm is a guess.
- **Offshore edge in strong wind.** In windy hours the observed Hs at x=0 is up to about 5% below the buoy value, and Dm differs by up to 4°. I didn't find the reason.
- **Not modelled:** the standing-wave stripes just in front of the structure, and Hs in the last one or two wet cells at the shoreline.
- **Extrapolation.** Breaking was tuned on waves of 0.5–1.2 m, so large-storm surf-zone values rest on that tuning carrying over.
