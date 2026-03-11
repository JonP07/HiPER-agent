#!/usr/bin/env bash
set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_1_5b_seed1"
ENGINE=vllm
SCRIPT_HGAE="run_scripts/hgae_1_5b/run_hgae_1_5b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"

# Kill existing screen session if it exists
if screen -list | grep -q "\.${SESSION}"; then
    screen -S "${SESSION}" -X quit
fi

# Create a new detached screen session
screen -dmS "${SESSION}"

# Create a named window and run the command
screen -S "${SESSION}" -X screen -t "HGAE_seed${SEED_A}" bash -c \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE} ${ENGINE} ${SEED_A}"

echo "Launched screen session: ${SESSION}"
echo "Attach with: screen -r ${SESSION}"
