set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_1_5b_seed1_detached"
ENGINE=vllm
SCRIPT_HGAE_1_5B="run_scripts/hgae_1_5b/run_hgae_1_5b_detached.sh"

# Run C
SEED_C=1
GPUS_C="0,1,2,3"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_HGAE_1_5B} ${ENGINE} ${SEED_C}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
