#!/usr/bin/env bash
# Run the same wandb agent command multiple times in parallel.
# Prompts for the wandb command and the number of parallel runs.

echo "Enter the wandb command to run in parallel (e.g. 'wandb agent <sweep-id>'):"
read -r WANDB_CMD

echo "Enter the number of parallel runs:"
read -r N_RUNS

if [ -z "$WANDB_CMD" ]; then
  echo "No command entered. Aborting."
  exit 1
fi

if ! echo "$N_RUNS" | grep -Eq '^[0-9]+$' || [ "$N_RUNS" -lt 1 ]; then
  echo "Parallel runs must be a positive integer."
  exit 1
fi

echo "Starting $N_RUNS parallel copies of: $WANDB_CMD"

for i in $(seq 1 "$N_RUNS"); do
  echo "[run $i/$N_RUNS] starting"
  eval "$WANDB_COMMAND" &
done

echo "Launched $N_RUNS run(s). Waiting for all to finish..."
wait
echo "All runs finished."