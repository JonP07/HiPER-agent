set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="webshop_ppo_prompt_7b_seed3_4"
ENGINE=vllm
SCRIPT_PPO_7B="run_scripts/webshop/ppo_prompt_7b/run_ppo_prompt_7b.sh"

# Run A
SEED_A=3
GPUS_A="0,1,2,3"

# Run B
SEED_B=4
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_A"
  tmux send-keys -t "$SESSION:$SEED_A" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_PPO_7B} ${ENGINE} ${SEED_A}" C-m

# create second window
tmux new-window -t "$SESSION" -n "$SEED_B"
  tmux send-keys -t "$SESSION:$SEED_B" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_PPO_7B} ${ENGINE} ${SEED_B}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
