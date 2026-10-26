# Claude Opus 5.5 runs (three, all failed)

I ran three Harbor trials with Claude Code / Opus 5.5 at max effort. I used a fresh task workspace for each trial and the 2-hour agent limit from `task.toml`. I saved the submissions, verifier outputs and run summaries here. My [Harbor launcher](../../../configs/run_trials.sh) includes the reproduction command.

Each trial ended on its own with a finished `predict.py` and a closing report. I did not count timeouts, usage-limit errors or API errors as model failures, and I did not resume any of these trials. I ran trials 2 and 3 in parallel.

I ran these trials before adding two unseen conditions, widening the rip-current tolerance and adding a shoreline tolerance. I kept the agent inputs unchanged and re-graded their submissions on the hardened set. They had also failed the original three conditions, with worst error/tolerance values of 2.63, 3.36 and 3.11.

| Trial | Start (UTC) | Duration | Verdict | Worst error / tolerance | Main failures (full list in `run.json`) |
|---|---|---|---|---|---|
| 1 | 2026-10-22 00:02 | 29 min | FAIL | 2.63 | standing-wave modulation at the seawall in all five conditions; rip-region direction in all five, breaking fraction in three; c3 breaking fraction on both bars |
| 2 | 2026-10-22 00:33 | 21 min | FAIL | 3.36 | rip current speed and seawall standing-wave modulation in all five; height, crest height and direction in the rip and height on both bars in c0, c3, c4; breaking fraction on the bars in c3, c4; site-wide height in c0, c3 |
| 3 | 2026-10-22 00:33 | 25 min | FAIL | 4.43 | rip direction and breaking fraction in four conditions; seawall modulation in four; mean height at the wall in three; in c3 height, setup and breaking fraction on both bars, in the rip and site-wide |

None modelled the standing waves in front of the seawall: all three added energy reflected from the structure but not the interference between incident and reflected waves, and runs 1 and 2 said in their closing messages that they saw the stripes and left them out.

Per trial:

- `submission/`: the top-level files of the final workspace, unmodified: `predict.py` and the files it loads (run 1: `wavemodel.py`, `params.json`; run 2: `model_params.npz`). The task's `data/` was not changed by any run and is not repeated.
- `verifier/test-stdout.txt`, `verifier/reward.txt`: I re-graded the submission in this repository's verifier image with `configs/regrade_trials.sh` and reproduced the saved verdict and worst error/tolerance.
- `agent_final_report.md`: the agent's closing message, verbatim.
- `run.json`: model, times, tool-call and token counts from the session log, and the grading result.

I omitted agent transcripts. I audited every tool call in each trial and found no access to answer material.
