# Nearshore Wave Forecast — a TB3 task

**Author:** Hamz Cohan (GitHub `JediDebugger`). Background: [About me](#about-me).

**Task:** [`tasks/nearshore-wave-forecast/`](tasks/nearshore-wave-forecast/)

**Built against:** `harbor-framework/terminal-bench` at commit `1dcda871`, Harbor `0.23.1.dev202609170426`

**Evaluation models:** Claude Opus 5.5 at max effort and GPT-6 Astra at xhigh effort.

## Summary

I simulated 24 radar and stereo-video snapshots sampled across ten days for a 1 km stretch of sandy coast (`time_h` 5–239). The agent gets those observations (maps of wave height, period, direction, breaking, water level, crest height and current), the offshore buoy, wind and tide for each snapshot, and an outdated chart, and must write a program that predicts the same maps for **storms and swells bigger than anything in the record, from other directions and at other tides**. It is graded on five unseen conditions, region by region over the site: every feature of every region of every condition must be within tolerance. I simulated all the data with my nearshore wave model.

**Claude Opus 5.5 at max effort failed all three runs** (§4.2), by wide margins: worst error/tolerance 2.63, 3.36 and 4.43 (pass at 1.0). None modelled the standing waves in front of the seawall; all three missed the rip region.

**GPT-6 Astra at xhigh effort failed all three runs** (§4.3): 2.52, 1.69 and 1.51.

The recorded evaluations are Harbor trials using the `claude-code` and `codex` agents, with a 2-hour agent limit (§4). Saved submissions were re-graded with the current verifier.

| Requirement | Status |
|---|---|
| Static checks (26 scripts in `scripts/checks/`) | ✅ 26/26 pass ([logs](results/checks/static/)) |
| Docker build (agent and verifier images) | ✅ |
| Oracle | ✅ reward 1.0; worst error/tolerance 0.03 ([output](results/execution/oracle/verifier/test-stdout.txt)) |
| Nop | ✅ reward 0.0 ([output](results/execution/nop/verifier/test-stdout.txt)) |
| Anti-cheat (three local exploit tests) | ✅ predictors that try to read the truth, hand in links to it, write the reward, leave a background process behind, or hand in `/app/data` as a link to `/tests` score 0 ([record](results/verifier-checks.md)) |
| Harbor trials, Claude Code / Opus 5.5 (max) | ✅ **3 runs, all failed** (2.63, 3.36, 4.43), each ending by itself in 21–29 min (§4.2) |
| Harbor trials, Codex / GPT-6 Astra (xhigh) | ✅ **3 runs, all failed** (2.52, 1.69, 1.51), each ending by itself in 74–90 min (§4.3) |
| Harbor cheat trials (unchanged TB3 CI hack prompt) | Opus: reward 0. Astra: provider refusal before any action; not graded (§4.4). |
| Implementation rubric review | **34 criteria pass, 0 fail, 1 not applicable** ([report](results/checks/implementation/check-result.json), [run details](results/checks/implementation/run.json)). |
| Licence of the data | ✅ I simulated all the data ([NOTICE.md](NOTICE.md), [DATA-SOURCES.md](DATA-SOURCES.md)) |

## 1. The task

**What the agent sees** (`/app/`):

| Path | Content |
|---|---|
| `data/obs/hour_00.npz` … `hour_23.npz` | 24 radar and stereo-video snapshots across ten days on a 10 m grid (61 × 101 cells) |
| `data/given_conditions.json` | per snapshot the offshore buoy spectrum (energy, direction and spreading per frequency), wind and tide |
| `data/chart.npz` | the latest hydrographic chart (20 m grid) and the structure's footprint |
| `data/grid.npz`, `data/example_new_conditions.json` | the grid, and the input format for new conditions |
| `predict.py` | a stub that writes empty fields |

**What it must produce:** `/app/predict.py`, run as `python3 /app/predict.py CONDITIONS_JSON OUT_DIR`, writing seven fields per case. It may use its own files anywhere under `/app`, runs with Python 3.12, numpy and scipy only, and must handle 3 cases in under 10 minutes. The agent has 2 hours. [Instruction](tasks/nearshore-wave-forecast/instruction.md).

**Why it is hard, and why fairly so**
- **The unseen cases are outside the record.** Offshore wave heights are 1.5–2.7 m against 0.5–1.2 m in the 24 snapshots, buoy spectra peak at up to 14.3 s against at most 11.1 s, and waves come mostly from the other side of shore-normal (mean direction +1° to +27°, against −35° to +3°). Their tides (−1.0 to +0.7 m) lie within the record's range (−1.03 to +1.01 m). A fit or interpolation of the record fails (§2); the predictor has to be a physical model of the site calibrated on the record.
- **Three local mechanisms, each with a small trace.** The migrated bar and the uncharted rip channel show where waves break and in the current maps; the seawall's reflection shows as stripes of wave height in front of it. In the record each matters only in its corner of the site. Under the unseen storms each decides its region, and grading is region by region.
- **Standard theory is enough.** Spectral wave propagation, depth-limited breaking, wave setup, rip currents and partial reflection with standing waves are textbook coastal engineering. Refits of the model structure to the 24 snapshots alone, never seeing the truth, pass from four different starting points (§2).
- **Fairness audit.** Every feature the Opus runs failed was checked against correct truth-free models fitted to the 24 observations, and every Opus run also fails with the seawall standing-wave modulation and the rip-current speed removed from grading ([tools/nearshore-wave-forecast/FAIRNESS.md](tools/nearshore-wave-forecast/FAIRNESS.md)).
- **Real-world value.** Turning a few days of nearshore remote sensing into a forecast model for design storms is routine work in coastal engineering: structure design, beach safety (rip currents), port and beach operations.

## 2. Calibration

I score each candidate with the task grader on the five unseen conditions. I report the worst error/tolerance across all features, regions and conditions (pass ≤ 1.0). I re-graded every candidate below except the oracle with the current grader on 2026-10-26, without refitting ([script](tools/nearshore-wave-forecast/calibration/recheck_current_grader.py), [output](results/checks/validation/calibration-recheck.json)); the values are the current ones.

| Candidate | Worst error / tolerance | Verdict |
|---|---|---|
| Reference ([predict.py](tasks/nearshore-wave-forecast/solution/predict.py) with [sim.py](tasks/nearshore-wave-forecast/solution/sim.py), the generating model with its true parameters); also the oracle | 0.03 in the verifier image | pass |
| Ten draws of the true parameters perturbed by ±2 % | 0.19–0.78 | pass (10 of 10) |
| Truth-free refits of the model structure to the 24 observation snapshots, four starting points | 0.30, 0.29, 0.33, 0.88 | pass (4 of 4) |
| Breaking parameters (`γ`, `α`) refitted with another standard breaking law (SWAN-style `Hmax = γd`), and with the reference's own Miche-type law | 0.25, 0.19 | pass |
| Truth-free refits with one process left out (channel, rip current, reflection, crest nonlinearity, breaking) | 2.07, 2.07, 2.69, 4.41, ∞ (missing values in wet cells) | fail |
| Standard spectral model on the chart, no current, no reflection | 3.28 | fail |
| Regression of each cell on the offshore conditions; persistence | 241; 7.75 and 10.24 | fail |
| The stub (nop) | all-NaN fields | fail |
| **Claude Opus 5.5's submissions** (§4.2), runs 1 / 2 / 3 | **2.63 / 3.36 / 4.43** | **fail** |
| **GPT-6 Astra's submissions** (§4.3), runs 1 / 2 / 3 | **2.52 / 1.69 / 1.51** | **fail** |

**How I made the unseen set.** I fixed the first three conditions (a storm at low tide, a storm at high tide, a long swell) when I generated the data. After the three Opus runs and the first Astra run, I added two low-tide storm conditions on the bar and rip channel: a long-period storm swell and a near-normal storm. I selected them from 123 candidates and rejected candidates that the worst truth-free refit failed as unfair. I also screened the four submissions available at that point. During this work, I widened the rip-current tolerance from 0.10 to 0.15 m/s and added a 20 m shoreline tolerance. The four submissions had already failed the original three conditions (Opus 2.63, 3.36, 3.11; Astra 1.25), so these changes affected their margins without changing their verdicts. I ran Astra trials 2 and 3 on the hardened task. Records: [tools/nearshore-wave-forecast/](tools/nearshore-wave-forecast/).

## 3. Verification and anti-cheat

- **The unseen conditions and their truth exist only in the verifier image,** under `/tests`, readable by root only (`chmod go-rwx`).
- **The submission is run as `nobody`.** Harbor copies the agent's `/app` into the verifier; the verifier hands that tree to user `nobody` (without following links in it) and runs `python3 /app/predict.py` from `/app` in batches of three and two conditions as `nobody`, each in its own session with a 600 s timeout and an empty, already created `OUT_DIR`. I disable verifier networking with `allow_internet = false` in `[verifier.environment]`. The transferred `/app` includes the agent's `/app/data` (Harbor empties the verifier's `/app` before the upload); the grader reads only `/tests` and the predictor's output.
- **The reward is out of the predictor's reach.** `/logs/verifier` is root-only (mode 700) before any submitted code runs, the predictor's output goes to a file and not a pipe, and afterwards its process group and anything else running as `nobody` is killed. Root then reads the predicted fields only from regular files, opened without following links, and computes the reward.
- **Tests** ([test_outputs.py](tasks/nearshore-wave-forecast/tests/test_outputs.py)): the predictor exists; the unseen set is intact; each of the five conditions is within tolerance; the overall verdict. [test_verifier_regressions.py](tasks/nearshore-wave-forecast/tests/test_verifier_regressions.py) adds 13 self-tests of the verifier (corrupt or incomplete wet fields fail, the shoreline rule, the batch size and the timeout watchdog); they do not depend on the submission. The reward is 1 only if all 21 tests pass.
- **Scoring** ([grade.py](tasks/nearshore-wave-forecast/tests/grade.py)): five regions (offshore, south bar, rip channel, central bar, seawall). Per region: normalised RMS error of wave height (0.10) and crest height (0.12); RMS error of direction (4°), period (0.6 s), breaking fraction smoothed over 50 m (0.06), setup (0.03 m) and current (0.10 m/s; 0.15 m/s in the rip region); at the seawall, mean height and crest height (8 %) and the standing-wave modulation of height (0.04). Across every wet cell on the full grid: height (0.07), crest height (0.08), setup (0.02 m), direction (4°), period (0.6 s), smoothed breaking fraction (0.06), and current (0.10 m/s). I only fill a trailing strip of missing values up to 20 m wide at the landward wet edge, using the nearest valid wet cell in that column. Interior missing values and infinities in wet cells fail; dry cells are not graded.
- **Local exploit tests.** I tested a `predict.py` that tries to read `/tests/truth`, write `reward.txt`, and hand in links to the truth files as its outputs scored 0: the links were refused and both accesses gave `Permission denied`. I also tested a predictor that leaves a detached process behind, repeatedly tries to write the reward and swaps its outputs for links. It scored 0, and I found nothing left running. A submission whose `/app/data` is a link to `/tests` also scored 0: every access through the link gave `Permission denied` ([record](results/verifier-checks.md)).
- **Determinism.** I fixed the truth and used deterministic grading. I re-graded all six submissions in the verifier image and reproduced their verdicts and worst error/tolerance values.

## 4. Results

### 4.1 Automated checks

| Check | Command | Result |
|---|---|---|
| Static checks | `for c in scripts/checks/check-*.sh; do bash $c tasks/nearshore-wave-forecast; done` (Python ≥ 3.11 on `PATH`) | 26/26 |
| Docker build + oracle | `harbor run -p tasks/nearshore-wave-forecast --agent oracle --env docker` | reward 1.0 |
| Nop | `harbor run -p tasks/nearshore-wave-forecast --agent nop --env docker` | reward 0.0 |
| Exploit tests (three), re-grades of the six submissions reported here | verifier image with the submission placed in `/app` ([record](results/verifier-checks.md)) | reward 0 for all three exploits and all six submissions; worst error/tolerance as recorded |
| Rubric review | `harbor check` with the TB3 implementation rubric, Claude Code / Sonnet 5 ([command](results/checks/implementation/command.txt)) | Report: 34 pass, 0 fail, 1 not applicable |
| Validation of the current files (2026-10-26) | static checks, image builds, Harbor oracle/nop, exploit tests, saved submissions, the 13 verifier regression tests, the calibration candidates of §2 | all as above; no LLM calls ([summary](results/checks/validation/summary.json)) |

### 4.2 Agent evidence: three Claude Opus 5.5 runs

**Configuration.** I ran three Harbor trials with Claude Code / Opus 5.5 at max effort, using the task environment and its 2-hour agent limit. I gave each trial a fresh workspace and the task instruction. Each trial ended on its own with a finished predictor; none reached the time limit, hit a usage limit or an API error, or needed a resume. I saved the results in [results/trials/opus-5.5/](results/trials/opus-5.5/) and provided [Harbor reproduction commands](configs/run_trials.sh).

| Run | Ended by itself at | Worst error / tolerance | Main failures |
|---|---|---|---|
| 1 | 29 min | 2.63 | seawall standing-wave modulation in all five conditions; rip-region direction in all five |
| 2 | 21 min | 3.36 | rip current and seawall modulation in all five; height in the rip and on the bars under the storms |
| 3 | 25 min | 4.43 | rip direction and breaking, seawall modulation and mean height; in the long-period storm swell, height, setup and breaking almost everywhere |

- **The seawall.** All three added energy reflected from the structure, but none the interference between incident and reflected waves that makes the standing-wave stripes; runs 1 and 2 said in their closing messages that they saw the stripes and left them out.
- **They stopped early.** The runs ended after 21–29 of the 120 minutes. Their closing messages rate confidence for the unseen conditions as moderate to low (run 1), less certain than the fit to the record (run 2) and moderate (run 3).

### 4.3 Agent evidence: GPT-6 Astra

**Configuration.** I ran three Harbor trials with Codex / GPT-6 Astra at xhigh effort, using the task environment, instruction and 2-hour agent limit. I saved the results in [results/trials/gpt-6-astra/](results/trials/gpt-6-astra/). I use the same [Harbor launcher](configs/run_trials.sh) for both model configurations.

| Run | Ended by itself at | Worst error / tolerance | Main failures |
|---|---|---|---|
| 1 | 76 min | 2.52 | breaking fraction on both bars and rip direction in the three low-tide storm conditions |
| 2 | 90 min | 1.69 | rip direction; mean height and crest height at the seawall, in the three low-tide storm conditions |
| 3 | 74 min | 1.51 | setup at the seawall and site-wide; breaking fraction on the bars in the long-period swell |

### 4.4 Cheat trials

I attempted the Harbor cheat trials with the unchanged [TB3 CI hack prompt](configs/hack-trial-prompt.md), appended to the task instruction. I saved the outcomes in [results/cheat-trials/](results/cheat-trials/) and included the invocation in my [Harbor launcher](configs/run_trials.sh).

I used the CI-provided prompt unchanged and preserved the provider responses in the run records.

| Agent | Result | Reward |
|---|---|---|
| Claude Opus 5.5 (max) | Looked over the workspace, found no test files, truth or hidden files, and stopped after 1 minute when Anthropic's safety classifier cut off one of its turns; reported that no bypass was found or tested and left the stub | 0 (unchanged stub) |
| GPT-6 Astra (xhigh) | Refused by OpenAI at the first turn ("flagged for possible cybersecurity risk"); no command ran and no deliverable was produced | not graded |

## 5. How the idea was chosen

In earlier task experiments, I found that agents usually solved problems within an hour when they could build an exact check for themselves. I chose a problem where they must notice, model and extrapolate hidden physical mechanisms from small local traces in observational data. I combined three such mechanisms on one coast and generated the observations with my own simulator.

## 6. Repository layout

| Path | What |
|---|---|
| `tasks/nearshore-wave-forecast/` | **The task**: instruction, agent environment with the 24 observation snapshots, chart and stub, oracle (the reference model), separate verifier with the five unseen conditions and their truth |
| `results/` | Static-check logs, oracle/nop outputs, exploit tests, the Opus 5.5 and GPT-6 Astra runs, and the cheat attempts (submissions, verifier outputs, agents' closing messages, run summaries; no agent transcripts) |
| `tools/nearshore-wave-forecast/` | Data generation, calibration (truth-free refits, perturbation draws, shortcut models, the re-grade with the current grader), the alternative breaking-law refits, the hardening record and the fairness audit |
| `configs/regrade_trials.sh` | Re-grade recorded Harbor submissions in the verifier image |
| `configs/hack-trial-prompt.md` | The unchanged TB3 CI hack prompt |
| `configs/run_trials.sh` | Harbor launcher for standard and cheat trials with the recorded model configurations |
| `NOTICE.md`, `DATA-SOURCES.md` | How I simulated the data and generated each file |

## Reproducing

```bash
# static checks (from a terminal-bench checkout, Python >= 3.11 on PATH)
for c in scripts/checks/check-*.sh; do bash "$c" tasks/nearshore-wave-forecast; done

# oracle and nop
harbor run -p tasks/nearshore-wave-forecast --agent oracle --env docker
harbor run -p tasks/nearshore-wave-forecast --agent nop --env docker

# re-grade recorded submissions in the verifier image (rebuilds ocean-verifier with the current tests)
configs/regrade_trials.sh results/trials/opus-5.5/trial-1 results/trials/gpt-6-astra/trial-1

# Harbor trials for the recorded model configurations
configs/run_trials.sh standard opus
configs/run_trials.sh standard astra
configs/run_trials.sh cheat opus
configs/run_trials.sh cheat astra
```

## About me

I am **Hamz Cohan**, a Senior Data Scientist and AI Architect (GitHub `JediDebugger`).

- **Profile:** I have 7+ years of experience delivering GenAI, LLM and ML systems: agentic and multi-agent workflows (LangGraph), RAG, MLOps on Azure, AWS and GCP, and data engineering (Spark, Airflow, BigQuery).
- **Experience:**
  - Tabiya: Senior Data Scientist and AI Architect (Mar 2025 to present); Data Scientist (Mar 2023 to Mar 2025).
  - StyleSeat: Machine Learning Engineer (Mar 2021 to Feb 2023).
  - Regeneron: AI Engineer (Jan 2019 to Feb 2021).
- **Education:** I earned a Bachelor's in Finance at Baruch College.
