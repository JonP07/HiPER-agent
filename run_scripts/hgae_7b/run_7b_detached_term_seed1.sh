set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_7b_seed1_term"
ENGINE=vllm
SCRIPT_HGAE_7B="run_scripts/hgae_7b/run_hgae_7b_term.sh"

# Run C
SEED_C=1
GPUS_C="0,1,2,3"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_C}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
