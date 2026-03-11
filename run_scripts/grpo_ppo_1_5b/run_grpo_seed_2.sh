set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_grpo_1_5b_seed2"
ENGINE=vllm
SCRIPT_GRPO="run_scripts/grpo_ppo_1_5b/qwen_grpo_1_5b.sh"

# Run A
SEED_A=2
GPUS_A="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "seed${SEED_A}"

tmux send-keys -t "$SESSION:seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_GRPO} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"