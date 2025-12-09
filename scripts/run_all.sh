#!/usr/bin/env bash

set -e

NOISE_TYPES=("emoji" "spelling" "slang" "hashtag" "repetition" "codeswitch")
INTENSITIES=(0.1 0.3 0.5)

echo "Running baseline (clean)..."
python -m src.run_baseline

for noise in "${NOISE_TYPES[@]}"; do
  for intensity in "${INTENSITIES[@]}"; do
    echo "Running noise=${noise}, intensity=${intensity}"
    python -m src.run_perturbed --noise_type "${noise}" --intensity "${intensity}"
  done
done
