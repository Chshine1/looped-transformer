#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKBONE_OUTPUT="outputs/backbone-rerun"

echo "Training the six-epoch FP32 backbone with batch size 200..."
pixi run -e fallback looped-transformer \
  --config-dir conf \
  --override checkpoint.output_dir="${BACKBONE_OUTPUT}" \
  train

mapfile -t backbone_checkpoints < <(
  find "${BACKBONE_OUTPUT}" \
    -type f \
    -path '*/backbone_warmup/checkpoint-epoch-6.pt'
)
if [[ ${#backbone_checkpoints[@]} -ne 1 ]]; then
  echo "Expected exactly one completed backbone artifact, found ${#backbone_checkpoints[@]}." >&2
  exit 1
fi

backbone_stage_dir="$(dirname "${backbone_checkpoints[0]}")"
for epoch in 2 3 6; do
  source_checkpoint="$(realpath "${backbone_stage_dir}/checkpoint-epoch-${epoch}.pt")"
  teacher_output="outputs/controller-teacher-from-backbone-epoch-${epoch}"

  echo "Training controller teacher from backbone epoch ${epoch}..."
  pixi run -e fallback looped-transformer \
    --config-dir conf \
    --override stage=controller_teacher \
    --override training.stage.source_checkpoint="${source_checkpoint}" \
    --override checkpoint.output_dir="${teacher_output}" \
    train
done

echo "Backbone and all three controller teachers completed."
