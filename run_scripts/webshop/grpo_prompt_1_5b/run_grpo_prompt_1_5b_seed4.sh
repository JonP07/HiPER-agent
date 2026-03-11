set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="webshop_grpo_prompt_1_5b_seed4"
ENGINE=vllm
SCRIPT_HGAE_1_5B="run_scripts/webshop/grpo_prompt_1_5b/run_grpo_prompt_1_5b.sh"

# Run C
SEED_C=4
GPUS_C="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_C"
  tmux send-keys -t "$SESSION:$SEED_C" \
  "CUDA_VISIBLE_DEVICES=${GPUS_C} bash ${SCRIPT_HGAE_1_5B} ${ENGINE} ${SEED_C}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
