set -euo pipefail

cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_grpo_3b_prompt"
ENGINE=vllm
SCRIPT2="run_scripts/grpo_ppo_3b_prompt/qwen_grpo_3b_prompt.sh"

# Run B
SEED_B=2
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "seed${SEED_B}"

tmux send-keys -t "$SESSION:seed${SEED_B}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT2} ${ENGINE} ${SEED_B}" C-m


# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"