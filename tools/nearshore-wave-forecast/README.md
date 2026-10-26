# Generation, calibration and hardening records

I used these scripts to produce the calibration figures in the top-level README and saved their logs. In my working folder, `hidden/` held `sim.py` (identical to `tasks/nearshore-wave-forecast/solution/sim.py` apart from the canary line), `grade.py` (identical to `tests/grade.py` apart from the canary line and the grid import), the unseen conditions and their truth, and `solver/` held the agent's files. I kept the absolute paths from that working folder in these records. The logs (`*.log`) were produced in that working folder during calibration; `calibration/verify_saved_fits.py` and `calibration/recheck_current_grader.py` run from this repository and re-grade the saved fits and every candidate with the task's current grader.

| Path | What |
|---|---|
| `generation/make_data.py` | Generates the 24 observation snapshots (truth plus measurement noise), the outdated chart, the example conditions, and the unseen conditions with their truth |
| `generation/package_verifier_data.py` | I use this to reproduce `tests/fixed-data.tar.gz`, the copy of the fixed inputs baked into the verifier image for local re-grades of saved submissions |
| `calibration/verify_saved_fits.py` | I re-grade the four saved fitted models against the current grader without refitting or calling an LLM |
| `calibration/recheck_current_grader.py` | I re-grade every calibration candidate (reference, ±2 % draws, all saved refits, the alternative breaking-law refits, shortcut models, the rip-speed coefficient scaled ×0.6 to ×1.5) against the current grader without refitting; output in `results/checks/validation/calibration-recheck.json` |
| `calibration/fit2.py`, `run_fit2.sh` | Staged truth-free refits on the 24 observation snapshots: profile, bar and breaking on the strip y 100-250 m; channel and rip on y 250-520 m; wall reflection on y 700-1000 m; then a check on all hours. Four starts (`full_s0`-`full_s3`) and five refits with one process left out |
| `calibration/fits2/*.json` | The fitted parameters of each refit |
| `calibration/calib.py` | Calibration driver: reference, ±2 % parameter draws, refits, shortcut models |
| `calibration/calib_hardened.log` | Its output on the hardened unseen set (c0-c4): `a` reference, `b` ten ±2 % draws, `fit2` refits (worst error/tolerance, failing features), `e` shortcut models |
| `audit/sim_alt.py`, `fit_gd.py`, `fit_gd.json`, `fit_miche.json` | The simulator with a switch for the breaking law (`hmax_form = "gd"`: `Hmax = gamma d`; otherwise the reference's Miche-type law), the refit of `gamma` and `alpha` on the strip y 100-250 m, and the two fitted results; `hardening/variants.py` and `fairness_strict.py` import these |
| `hardening/cand.py`, `cand2.py`, `*_meta.json` | The 123 candidate conditions screened for the two added unseen conditions (75 in a broad round, 48 in a focused round) |
| `hardening/variants.py`, `run_subs.py`, `analyze.py`, `hcommon.py` | Truth and model variants (refits, ±2 % draws, breaking-law refits) on the candidates, the earlier submissions on the candidates, and the analysis |
| `hardening/add_conditions.py` | Appends the two chosen conditions to the unseen set and checks that c0-c2 and their truth are unchanged bit for bit |
| `hardening/fairness_strict.py`, `.log`, `.json` | Margins of the refits and model variants on the hardened set, graded exactly as a submission would be |
| `FAIRNESS.md` | The fairness audit of the features the agents failed |
