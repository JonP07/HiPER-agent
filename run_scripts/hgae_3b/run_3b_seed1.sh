set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_3b_seed1"
ENGINE=vllm
SCRIPT_HGAE_3B="run_scripts/hgae_3b/run_hgae_3b.sh"

# Run A
SEED_A=1
GPUS_A="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_A"
  tmux send-keys -t "$SESSION:$SEED_A" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_3B} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
