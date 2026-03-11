set -euo pipefail

cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_ppo_grpo_7b_prompt"
ENGINE=vllm
SCRIPT="run_scripts/grpo_ppo_7b_prompt/qwen_ppo_7b_prompt.sh"
SCRIPT2="run_scripts/grpo_ppo_7b_prompt/qwen_grpo_7b_prompt.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"

# Run B
SEED_B=2
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "seed${SEED_A}"

tmux send-keys -t "$SESSION:seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT} ${ENGINE} ${SEED_A} && \
   CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT2} ${ENGINE} ${SEED_A}" C-m

# second window
tmux new-window -t "$SESSION" -n "seed${SEED_B}"

tmux send-keys -t "$SESSION:seed${SEED_B}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT} ${ENGINE} ${SEED_B} && \
   CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT2} ${ENGINE} ${SEED_B}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"