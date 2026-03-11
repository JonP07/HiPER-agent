set -euo pipefail

cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_grpo_7b_seed1"
ENGINE=vllm
SCRIPT_GRPO="run_scripts/grpo_ppo_7b/qwen_grpo_7b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "GRPO_seed${SEED_A}"

tmux send-keys -t "$SESSION:GRPO_seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_GRPO} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
