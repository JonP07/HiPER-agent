set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="webshop_grpo_prompt_0123"
ENGINE=vllm
SCRIPT_grpo_1_5B="run_scripts/webshop/grpo_prompt_1_5b/run_grpo_prompt_1_5b.sh"
SCRIPT_grpo_7B="run_scripts/webshop/grpo_prompt_7b/run_grpo_prompt_7b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"

# Run B
SEED_B=2
GPUS_B="4,5,6,7"

# Run C
SEED_C=3
GPUS_C="0,1,2,3"

# Run D
SEED_D=4
GPUS_D="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "1_5B"
  tmux send-keys -t "$SESSION:1_5B" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_grpo_1_5B} ${ENGINE} ${SEED_A} ; CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_grpo_7B} ${ENGINE} ${SEED_B} ; CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_grpo_1_5B} ${ENGINE} ${SEED_C} ; CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_grpo_7B} ${ENGINE} ${SEED_D}" C-m


# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
# echo "Attach with: tmux attach -t $SESSION"
