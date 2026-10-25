#!/usr/bin/env bash
# Stop this Vast instance (halts GPU billing; disk and files are kept, storage still billed) once the given tmux
# sessions have all ended and no scoring/compression process is left. Results are already on HF by then: run.py
# uploads before it exits. Restart the instance from the Vast console to resume.
#
#   SESSIONS="llama_q awq" tmux new-session -d -s autostop "bash scripts/eval/queues/autostop.sh 2>&1 | tee -a logs/autostop.log"
cd /root/low_compute_fairness
SESSIONS="${SESSIONS:?set SESSIONS, the tmux sessions to wait for}"
GRACE="${GRACE:-300}"   # seconds to wait after the last session ends (lets late uploads finish)
busy() {
  for s in $SESSIONS; do tmux has-session -t "=$s" 2>/dev/null && return 0; done   # "=": exact name, not prefix
  pgrep -f "scripts/(eval/run|compression/(wanda_prune|awq_quant)|model_prep/merge_lora)\.py" >/dev/null
}
echo "=== $(date "+%F %T") autostop armed: waiting for sessions [$SESSIONS]"
while busy; do sleep 120; done
echo "=== $(date "+%F %T") all sessions ended; stopping in ${GRACE}s unless new work starts"
sleep "$GRACE"
if busy; then echo "=== $(date "+%F %T") new work started; autostop re-arming"; exec bash "$0"; fi
# The Vast-set credentials live in the container's environment; tmux may not have inherited them.
[ -n "$CONTAINER_ID" ] || export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | xargs)
echo "=== $(date "+%F %T") stopping instance $CONTAINER_ID"
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
