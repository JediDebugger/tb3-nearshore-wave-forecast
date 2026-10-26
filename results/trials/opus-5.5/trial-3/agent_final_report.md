`/Users/dev/ocean-r3-probe/app/predict.py` is finished. It finds its data files relative to its own location and runs 3 cases in about 2 seconds, well inside the 10-minute limit. It writes `case_<case>.npz` with `Hs, Tm, Dm, Qb, mwl, Cs, u` on the 61×101 grid, and NaN appears only on dry cells and the structure.

**What it is:** a physics model rather than a statistical fit, so it should hold up for storms bigger than anything in the record. It tracks wave energy by frequency and direction from the buoy line to the shore. It includes:
- waves bending over the seabed and over the rip current,
- waves steepening in shallow water,
- depth-limited breaking (the standard Battjes–Janssen formula, which also gives `Qb`),
- partial reflection of waves off the structure,
- the rise in mean water level from breaking (`mwl`),
- a rip-current jet whose waves interact with it, and
- crest height (`Cs`) from a fitted rule based on wave nonlinearity.

**What the observations showed that the chart doesn't:**
- **The bar has moved.** Its crest is now about 2.34 m deep at x≈400 m; the chart has 3.1 m at x=360 m. The beach near the waterline is also a little deeper than charted.
- **There is a rip channel** through the bar at y≈380 m. The rip current only switches on at low tide (tide ≤ −0.8 m), when waves break on the bar, at about 0.1–0.13 m/s. Waves in the channel then grow by roughly 7%.
- **The structure reflects waves.** At high tide the reflected waves raise Hs by about 10% and turn the wave direction by about −10° across the whole northern part of the site, out to the buoy line.
- **Crest height** follows Cs/Hs = min(0.80, 0.508 + 0.006·Ur^0.965), where Ur is a standard nonlinearity measure (wave height × wavelength² ÷ depth³).

**How well it fits:** I re-ran all 24 observed hours through `predict.py`. Averaged over each region (offshore, bar, trough, inner surf, rip, north), the mean errors are:

| Field | Mean error |
|---|---|
| Hs | 0.01 m or less |
| Tm | under 0.03 s |
| Dm | under 1° |
| mwl | a few mm |
| Cs | about 0.01 m |
| u | about 0.005 m/s |

The rip-zone and north-zone wave-height ratios match the observations to within about 1–2%. One caveat on this check: the same 24 hours were used for calibration, so it shows consistency, not true out-of-sample skill.

**How confident I am:** moderate. Offshore, the bar, the trough and the reflection zone should be within a reasonable tolerance. The main risks, roughly in order:
1. **Rip current (`u`) for new conditions.** It only appeared in 3 of 24 hours. The on/off trigger (how hard waves break on the bar) separated those hours by a hair. I scaled its speed linearly with wave height on the bar, which gives about 0.2–0.4 m/s in storms. The truth could be quite different in size or in how it depends on the tide.
2. **Breaking strength in big storms.** All the breaking data comes from waves under about 1.2 m, so the breaking settings are being extrapolated.
3. **The swash edge.** Model breaking (`Qb`) runs about 0.05 too high in the inner surf zone. Where the shoreline falls at high tide is approximate: the model may give values in one row that the truth treats as dry. I chose that over leaving NaN where the truth is wet.
4. **Not modelled:** wind input and reflection off the structure's side face, which only matters for waves coming from the positive (+y) side.
