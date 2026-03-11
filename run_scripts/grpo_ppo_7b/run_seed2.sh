set -euo pipefail

cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_ppo_grpo_7b"
ENGINE=vllm
SCRIPT_PPO="run_scripts/grpo_ppo_7b/qwen_ppo_7b.sh"
SCRIPT_GRPO="run_scripts/grpo_ppo_7b/qwen_grpo_7b.sh"

# Run A
GPUS_A="0,1,2,3"
GPUS_B="4,5,6,7"
SEED=2

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "PPO_seed${SEED}"

tmux send-keys -t "$SESSION:PPO_seed${SEED}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_PPO} ${ENGINE} ${SEED}" C-m

# second window
tmux new-window -t "$SESSION" -n "GRPO_seed${SEED}"

tmux send-keys -t "$SESSION:GRPO_seed${SEED}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_GRPO} ${ENGINE} ${SEED}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"