set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_grpo_prompt_7b_pen"
ENGINE=vllm
SCRIPT_GRPO_7B="run_scripts/grpo_prompt_alfworld_7b_pen.sh"
SCRIPT_GIGPO_7B="run_scripts/gigpo_prompt_alfworld_7b_pen.sh"
# Run C
SEED_1=1
SEED_2=2
GPUS_C="4,5,6,7"
CMD_1="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GRPO_7B} ${ENGINE} ${SEED_1}"
CMD_2="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GIGPO_7B} ${ENGINE} ${SEED_1}"
CMD_3="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GRPO_7B} ${ENGINE} ${SEED_2}"
CMD_4="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GIGPO_7B} ${ENGINE} ${SEED_2}"
tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_1"
  tmux send-keys -t "$SESSION:$SEED_1" \
  "bash -lc 'while true; do ${CMD_1}; ${CMD_2}; ${CMD_3}; ${CMD_4}; sleep 5; done'" C-m

echo "Launched tmux session: $SESSION"
# echo "Attach with: tmux attach -t $SESSION"