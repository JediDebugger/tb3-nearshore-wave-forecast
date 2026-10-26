# Data sources and licence status

**I simulated all the data. I used no third-party data; see [NOTICE.md](NOTICE.md).**

| Data | Where | How I made it |
|---|---|---|
| 24 level-2 snapshots across ten days (`Hs, Tm, Dm, Qb, mwl, Cs, u, v` on a 10 m grid) | `environment/data/obs/` | I simulated fields for 24 offshore conditions and added measurement noise (wave height 5 % plus 2 cm, crest height 5 %, period 0.25 s, direction 3 degrees, breaking fraction 0.015, water level 0.015 m, current 0.04 m/s) |
| Offshore buoy spectra, wind and tide per snapshot | `environment/data/given_conditions.json` | I drew swell and wind-sea conditions over ten days of tide and stored frequency spectra, mean direction and spreading per bin |
| Hydrographic chart | `environment/data/chart.npz` | I simulated an older survey with a different beach profile, the bar at its old position, no rip channel, depths rounded to 0.1 m on a 20 m grid, and the structure's footprint |
| Example conditions | `environment/data/example_new_conditions.json` | I saved one condition in the input format |
| Unseen conditions and their true fields | `tests/unseen_conditions.json`, `tests/truth/` | I selected five conditions with bigger waves and longer periods than the record, mostly from the other side of shore-normal, at tides within the observed range, and simulated their fields without noise |

I used [tools/nearshore-wave-forecast/generation/make_data.py](tools/nearshore-wave-forecast/generation/make_data.py) (c0-c2) and [tools/nearshore-wave-forecast/hardening/add_conditions.py](tools/nearshore-wave-forecast/hardening/add_conditions.py) (c3, c4), both with the simulator in [solution/sim.py](tasks/nearshore-wave-forecast/solution/sim.py).
