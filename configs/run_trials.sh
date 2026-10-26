#!/usr/bin/env bash
# I use Harbor for both standard trials and the unchanged CI cheat prompt.
# Usage: configs/run_trials.sh standard|cheat opus|astra [harbor run options...]
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 standard|cheat opus|astra [harbor run options...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=$1
CONFIGURATION=$2
shift 2
EXTRA=()

case "$MODE" in
  standard) ATTEMPTS=3 ;;
  cheat)
    ATTEMPTS=1
    EXTRA+=(--extra-instruction-path "$ROOT/configs/hack-trial-prompt.md")
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac

case "$CONFIGURATION" in
  opus)
    AGENT=claude-code
    MODEL=anthropic/claude-opus-5-5
    EFFORT=max
    EXTRA+=(--ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000)
    ;;
  astra)
    AGENT=codex
    MODEL=openai/gpt-6-astra
    EFFORT=xhigh
    ;;
  *) echo "Unknown configuration: $CONFIGURATION" >&2; exit 2 ;;
esac

exec harbor run --path "$ROOT/tasks/nearshore-wave-forecast" \
  --agent "$AGENT" --model "$MODEL" --ak "reasoning_effort=$EFFORT" \
  --n-attempts "$ATTEMPTS" --n-concurrent 1 --env docker \
  --jobs-dir "$ROOT/jobs" "${EXTRA[@]}" "$@"
