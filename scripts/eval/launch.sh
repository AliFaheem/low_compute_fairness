#!/usr/bin/env bash
# Start experiments in a detached tmux session, so they survive SSH/VS Code disconnects.
#
#   scripts/eval/launch.sh --machine vast-4090              # every run assigned to this machine
#   scripts/eval/launch.sh --model llama-3.1-8b --variant bf16 --lang en ur --limit 500   # any run.py args
#
# Watch progress:
#   tail -f logs/<session>.log          live log on this machine
#   tmux attach -t <session>            attach (detach again with Ctrl-b d)
#   uv run python scripts/eval/status.py --watch   whole grid, from any machine (reads HF_RESULTS_REPO)
set -euo pipefail
cd "$(dirname "$0")/../.."

[ $# -gt 0 ] || { sed -n '2,11p' "$0"; exit 2; }
command -v tmux >/dev/null || { echo "Installing tmux"; (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null; }

# Session name: the --machine value, else model__variant.
name=""; prev=""
for a in "$@"; do
  [ "$prev" = "--machine" ] && name="$a"
  [ "$prev" = "--model" ] && name="${name:-$a}"
  [ "$prev" = "--variant" ] && [ -n "$name" ] && name="${name}__$a"
  prev="$a"
done
name="${name:-exp}"; name="${name//./-}"   # tmux session names cannot contain dots

if tmux has-session -t "$name" 2>/dev/null; then
  echo "Session '$name' is already running: tmux attach -t $name"; exit 1
fi

mkdir -p logs
log="logs/$name.log"
printf '\n=== %s start: run.py %s\n' "$(date '+%F %T')" "$*" >> "$log"
tmux new-session -d -s "$name" \
  "uv run python -u scripts/eval/run.py $(printf '%q ' "$@") 2>&1 | tee -a $log; echo \"=== \$(date '+%F %T') exit \${PIPESTATUS[0]}\" | tee -a $log"
echo "Started tmux session '$name'"
echo "  live log:  tail -f $log"
echo "  attach:    tmux attach -t $name"
echo "  grid:      uv run python scripts/eval/status.py --watch"
