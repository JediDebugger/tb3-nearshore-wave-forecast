# Verifier checks run locally (no agent involved)

I ran every check below on 2026-10-26 with the current task files: the `/app` artifact excludes only `__pycache__` and `*.pyc`, so the agent's `/app/data` is transferred with the submission; the verifier has no network (`allow_internet = false` in `[verifier.environment]`, with 2 CPUs and 4096 MB as in the agent's environment); the predictor runs as `nobody` from `/app`, with `OUT_DIR` already created, in batches of at most three cases with 600 s per batch and no memory limit beyond the container's. I used macOS arm64, Docker 29.8 on Colima (8 CPU, 12 GiB), and Harbor 0.23.1.dev202609170426. These runs made no LLM calls. Summary: [checks/validation/summary.json](checks/validation/summary.json).

| Check | Command | Result |
|---|---|---|
| Static checks | `for c in scripts/checks/check-*.sh; do bash $c tasks/nearshore-wave-forecast; done` (Python 3.13) | 26/26 pass ([logs](checks/static/)) |
| Docker build | `docker build` of `environment/` and `tests/` | both build |
| Oracle | `harbor run -p tasks/nearshore-wave-forecast --agent oracle --env docker` | reward 1.0; all 21 tests pass, worst error/tolerance 0.03 ([output](execution/oracle/verifier/test-stdout.txt)) |
| Nop | `harbor run -p tasks/nearshore-wave-forecast --agent nop --env docker` | reward 0.0; the stub `predict.py` writes all-NaN fields, 6 of 21 tests fail ([output](execution/nop/verifier/test-stdout.txt)) |
| Transferred `/app/data` | the oracle and nop artifacts | byte-identical to `environment/data` in both runs. Harbor empties `/app` in the verifier before uploading the transferred `/app`, so grading sees the agent's copy, not the one baked into the verifier image; the grader reads only `/tests` and the predictor's output directory |
| Sandbox: cheating predictor | verifier image, a `predict.py` that tries to read `/tests/truth/*.npz` and to write `/logs/verifier/reward.txt`, and hands in links to the truth files as its outputs ([script](exploit-test/submission/predict.py), [output](exploit-test/output.txt)) | reward 0; links refused; `/tests` and the reward file `Permission denied` as `nobody` |
| Sandbox: cheating predictor that leaves a background process | verifier image, a `predict.py` that detaches a process which keeps trying to write the reward and to swap its outputs for links to the truth ([script](exploit-test/daemon/submission/predict.py), [output](exploit-test/daemon/output.txt)) | reward 0; nothing left running as `nobody` after the run |
| Sandbox: `/app/data` handed in as a link to `/tests` | verifier image, `/app/data` replaced by the submission's link to `/tests`; its `predict.py` tries to list `/tests` and read the truth and the unseen conditions through the link ([script](exploit-test/data-link/submission/predict.py), [output](exploit-test/data-link/output.txt)) | reward 0; every access `Permission denied`; `/tests` and `/tests/truth` still owned by root with mode 700 afterwards |
| The three Opus 5.5 submissions | `configs/regrade_trials.sh results/trials/opus-5.5/trial-*` | reward 0 each; worst error/tolerance 2.63, 3.36, 4.43, as recorded ([outputs](trials/opus-5.5/)) |
| The three GPT-6 Astra submissions | the same | reward 0 each; 2.52, 1.69, 1.51, as recorded ([outputs](trials/gpt-6-astra/)) |
| The four saved truth-free calibration fits | `python tools/nearshore-wave-forecast/calibration/verify_saved_fits.py OUT.json` (Python 3.9, numpy 2.0.2, scipy 1.13.1) | all pass; 0.30, 0.29, 0.33, 0.88 ([output](checks/validation/saved-fits.json)) |
| All calibration candidates of the top-level README §2 | `python tools/nearshore-wave-forecast/calibration/recheck_current_grader.py OUT.json` (Python 3.9, numpy 2.0.2, scipy 1.13.1) | reference, ten ±2 % draws (0.19–0.78), four refits (0.29–0.88) and both breaking-law refits (0.25, 0.19) pass; the five refits with a process left out and the four shortcut models fail; the reference with its rip-speed coefficient scaled ×0.6 to ×1.5 passes ([output](checks/validation/calibration-recheck.json)) |

For the sandbox checks I put the submission into `/app` and ran `/tests/test.sh` (from the repo root, after `docker build -t ocean-verifier tasks/nearshore-wave-forecast/tests`). The first two keep the image's copy of the data; the third removes it first, as Harbor's transfer would:

```bash
docker run --rm --network none -v "$PWD/<dir-with-predict.py>":/sub:ro \
  ocean-verifier \
  bash -c 'cp -a /sub/. /app/ && /tests/test.sh; cat /logs/verifier/reward.txt'
```

`configs/regrade_trials.sh` uses the same command. The saved submissions do not include `data/`, which no run changed, so the copy in the verifier image stands in for it.
