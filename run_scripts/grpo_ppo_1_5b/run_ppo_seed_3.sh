set -euo pipefail

cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_ppo_1_5b_seed3"
ENGINE=vllm
SCRIPT_PPO="run_scripts/grpo_ppo_1_5b/qwen_ppo_1_5b.sh"

# Run A
SEED_A=3
GPUS_A="0,1,2,3"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "seed${SEED_A}"

tmux send-keys -t "$SESSION:seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_PPO} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"