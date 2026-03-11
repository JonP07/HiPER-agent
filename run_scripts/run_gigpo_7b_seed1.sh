set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_gigpo_7b_seed1"
ENGINE=vllm
SCRIPT_GIGPO_7B="run_scripts/gigpo_alfworld_7b.sh"

# Run C
SEED_C=1
GPUS_C="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GIGPO_7B} ${ENGINE} ${SEED_C}" C-m

# echo "Launched tmux session: $SESSION"
# echo "Attach with: tmux attach -t $SESSION"