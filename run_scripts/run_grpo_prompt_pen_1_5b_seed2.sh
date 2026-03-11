set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_grpo_prompt_1_5b_seed2_pen"
ENGINE=vllm
SCRIPT_GRPO_1_5B="run_scripts/grpo_prompt_alfworld_1_5b_pen.sh"

# Run C
SEED_C=2
GPUS_C="0,1,2,3"
CMD="CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_GRPO_1_5B} ${ENGINE} ${SEED_C}"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "bash -lc 'while true; do ${CMD}; sleep 5; done'" C-m

echo "Launched tmux session: $SESSION"
# echo "Attach with: tmux attach -t $SESSION"