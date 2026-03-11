set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_turn_ppo_prompt_1_5b_seed2"
ENGINE=vllm
SCRIPT_TPPO="run_scripts/turn_ppo_1_5b/run_turn_ppo_1_5b_prompt.sh"

# Run A
SEED_A=2
GPUS_A="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "HGAE_seed${SEED_A}"
tmux send-keys -t "$SESSION:HGAE_seed${SEED_A}" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_TPPO} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
