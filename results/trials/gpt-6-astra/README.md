# GPT-6 Astra runs (three, all failed; one further attempt invalid)

I ran three Harbor trials with Codex / GPT-6 Astra at xhigh effort. I used the same task environment, instruction and 2-hour agent limit as for Opus. I saved the submissions, verifier outputs and run summaries here. My [Harbor launcher](../../../configs/run_trials.sh) includes the reproduction command.

I ran the three valid trials one after another. Each ended on its own with a finished `predict.py` and a closing report.

| Trial | Start (UTC) | Duration | Verdict | Worst error / tolerance | Main failures (full list in `run.json`) |
|---|---|---|---|---|---|
| 1 | 2026-10-22 14:01 | 76 min | FAIL | 2.52 | breaking fraction on both bars in c0, c3, c4 (up to 0.15 against 0.06); rip direction in c0, c3, c4; c3 breaking fraction in the rip and at the wall |
| 2 | 2026-10-22 19:08 | 90 min | FAIL | 1.69 | rip direction in c0, c3, c4; mean height and crest height at the seawall in c0, c3, c4 |
| 3 | 2026-10-22 20:39 | 74 min | FAIL | 1.51 | setup at the seawall in c0, c3, c4 and site-wide in c0, c3; c3 breaking fraction on both bars and at the wall |

I ran trial 1 before adding two unseen conditions, widening the rip-current tolerance and adding a shoreline tolerance. I kept its inputs unchanged and re-graded its submission on the hardened set. It had also failed the original three conditions, with a worst error/tolerance of 1.25. I ran trials 2 and 3 on the hardened task.

All three modelled the seawall's standing waves correctly, the part every Opus run missed. All three closing messages rate confidence as high for execution and moderate for the unseen conditions.

**Invalid attempt (not counted).** I excluded an earlier attempt on 2026-10-22 at 13:17 UTC because it stopped after 43 minutes with an OpenAI service error ("Selected model is at capacity", `turn.failed`) before replacing the stub `predict.py`. It produced no deliverable. I repeated it as trial 1.

For each trial, I saved `submission/` (the final workspace's top-level files, unmodified), `verifier/` (the output of re-grading in this repository's verifier image), `agent_final_report.md` (the closing message, verbatim) and `run.json` (model, timing, command and token counts, and grading result). Trial 2 also left `validation.json` with its own notes. I reproduced each saved verdict and worst error/tolerance, and omitted agent transcripts and sign-in files. I audited each trial's commands and found no access to answer material; trial 1's only command outside its folder was Codex's lookup of `AGENTS.md` files in parent folders.
