set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_3b_seed2"
ENGINE=vllm
SCRIPT_HGAE_3B="run_scripts/hgae_3b/run_hgae_3b.sh"

# Run B
SEED_B=2
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_B"
  tmux send-keys -t "$SESSION:$SEED_B" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_HGAE_3B} ${ENGINE} ${SEED_B}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
