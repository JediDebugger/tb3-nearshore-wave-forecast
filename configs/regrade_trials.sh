#!/usr/bin/env bash
# I use this helper to re-grade saved Harbor submissions in the verifier image:
# I place the submission and fixed data in /app, then run /tests/test.sh.
#
#   configs/regrade_trials.sh results/trials/opus-5.5/trial-1 [more trial dirs...]
#
# Writes <trial>/verifier/test-stdout.txt and <trial>/verifier/reward.txt and adds the
# verdict to <trial>/run.json. At most 4 trials run at a time (2 CPUs, 4 GB each, as in task.toml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK="$ROOT/tasks/nearshore-wave-forecast"
IMAGE="${IMAGE:-ocean-verifier}"
docker build -q -t "$IMAGE" "$TASK/tests" >/dev/null

grade_one() {
  local t; t="$(cd "$1" && pwd)"
  mkdir -p "$t/verifier"
  docker run --rm --network none --cpus 2 --memory 4g \
    -v "$t/submission":/sub:ro "$IMAGE" \
    bash -c 'cp -a /sub/. /app/ && /tests/test.sh; echo "reward: $(cat /logs/verifier/reward.txt)"' \
    > "$t/verifier/test-stdout.txt" 2>&1 || true
  sed -n 's/^reward: //p' "$t/verifier/test-stdout.txt" | tail -1 > "$t/verifier/reward.txt"
  python3 - "$t" <<'PY'
import json, pathlib, re, sys
t = pathlib.Path(sys.argv[1])
out = (t / "verifier" / "test-stdout.txt").read_text()
m = re.search(r"^(PASS|FAIL) worst err/tol = ([0-9.]+|inf)$", out, re.M)
fails = re.findall(r"^   (c\d:\S+)$", out, re.M)
run = json.loads((t / "run.json").read_text())
run["graded_on"] = "hardened unseen set c0-c4, this repository's verifier image"
run["verdict"] = m.group(1) if m else "ERROR"
run["worst_err_over_tol"] = float(m.group(2)) if m else None
run["failures"] = list(dict.fromkeys(fails))
run["reward"] = int((t / "verifier" / "reward.txt").read_text().strip() or 0)
(t / "run.json").write_text(json.dumps(run, indent=1) + "\n")
print(t, run["verdict"], run["worst_err_over_tol"], "reward", run["reward"])
PY
}
export -f grade_one
export TASK IMAGE
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'grade_one "$@"' _ {}
