set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_1_5b_seed1"
ENGINE=vllm
SCRIPT_HGAE="run_scripts/hgae_1_5b/run_hgae_1_5b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "HGAE_seed${SEED_A}"
tmux send-keys -t "$SESSION:HGAE_seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
