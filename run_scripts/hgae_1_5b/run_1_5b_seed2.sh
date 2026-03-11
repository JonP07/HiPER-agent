set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_1_5b_seed2"
ENGINE=vllm
SCRIPT_HGAE="run_scripts/hgae_1_5b/run_hgae_1_5b.sh"

# Run B
SEED_B=2
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "HGAE_seed${SEED_B}"
tmux send-keys -t "$SESSION:HGAE_seed${SEED_B}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_HGAE} ${ENGINE} ${SEED_B}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
