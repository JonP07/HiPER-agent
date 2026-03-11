set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_gigpo_1_5b_seed2"
ENGINE=vllm
SCRIPT_GIGPO_1_5B="run_scripts/gigpo_alfworld_1_5b.sh"

# Run C
SEED_C=2
GPUS_C="4,5,6,7"
CMD="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GIGPO_1_5B} ${ENGINE} ${SEED_C}"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "bash -lc 'while true; do ${CMD}; sleep 5; done'" C-m

echo "Launched tmux session: $SESSION"
# echo "Attach with: tmux attach -t $SESSION"