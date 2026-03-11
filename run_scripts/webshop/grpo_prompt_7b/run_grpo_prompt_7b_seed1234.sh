set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="webshop_grpo_prompt_7b_seed1234"
ENGINE=vllm
SCRIPT_HGAE_7B="run_scripts/webshop/grpo_prompt_7b/run_grpo_prompt_7b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"
SEED_B=2
SEED_C=3
SEED_D=4

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_A"
  tmux send-keys -t "$SESSION:$SEED_A" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_A}; CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_B}; CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_C}; CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_D}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
