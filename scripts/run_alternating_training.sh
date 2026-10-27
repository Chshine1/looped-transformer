#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 INITIAL_BACKBONE [CYCLES=1] [OUTPUT_ROOT=outputs/alternating]" >&2
  exit 2
fi

initial_backbone="$1"
cycles="${2:-1}"
output_root="${3:-outputs/alternating}"
if [[ ! -f "${initial_backbone}" ]]; then
  echo "Initial backbone does not exist: ${initial_backbone}" >&2
  exit 2
fi
if ! [[ "${cycles}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CYCLES must be a positive integer." >&2
  exit 2
fi

# Recreate the deterministic split so cycle decisions never touch final holdouts.
pixi run -e fallback python scripts/split_mnli_cycle_data.py
current_backbone="$(realpath "${initial_backbone}")"
mkdir -p "${output_root}"

for cycle in $(seq 1 "${cycles}"); do
  cycle_dir="${output_root}/cycle-${cycle}"
  if [[ -e "${cycle_dir}" ]]; then
    echo "Refusing to mix runs in existing directory: ${cycle_dir}" >&2
    exit 2
  fi
  controller_dir="${cycle_dir}/controller"
  controller_eval_dir="${cycle_dir}/controller-evaluation"
  guided_dir="${cycle_dir}/guided-backbone"
  baseline_eval_dir="${cycle_dir}/baseline-evaluation"
  guided_eval_dir="${cycle_dir}/guided-evaluation"

  echo "Cycle ${cycle}: training a one-epoch smoothed-BCE controller..."
  pixi run -e fallback looped-transformer \
    --config-dir conf \
    --override stage=controller_teacher \
    --override training.stage.source_checkpoint="${current_backbone}" \
    --override checkpoint.output_dir="${controller_dir}" \
    train

  mapfile -t controller_checkpoints < <(
    find "${controller_dir}" -type f \
      -path '*/controller_teacher/checkpoint-epoch-1.pt'
  )
  if [[ ${#controller_checkpoints[@]} -ne 1 ]]; then
    echo "Expected one controller checkpoint, found ${#controller_checkpoints[@]}." >&2
    exit 1
  fi
  controller_checkpoint="$(realpath "${controller_checkpoints[0]}")"

  echo "Cycle ${cycle}: measuring controller signal on sampled train and cycle dev..."
  pixi run -e fallback python scripts/evaluate_backbone_teachers.py \
    --backbone-checkpoint "${current_backbone}" \
    --teacher-checkpoint "${controller_checkpoint}" \
    --pair-epoch "${cycle}" \
    --data-file data/mnli_train_sample.tsv \
    --data-file data/mnli_cycle_dev.tsv \
    --batch-size 64 \
    --result-dir "${controller_eval_dir}"
  pixi run -e fallback python scripts/gate_controller.py \
    "${controller_eval_dir}/analysis.json"

  echo "Cycle ${cycle}: training one softly guided backbone epoch..."
  pixi run -e fallback looped-transformer \
    --config-dir conf \
    --override stage=guided_backbone \
    --override training.stage.source_checkpoint="${current_backbone}" \
    --override training.stage.controller_checkpoint="${controller_checkpoint}" \
    --override checkpoint.output_dir="${guided_dir}" \
    train

  mapfile -t guided_checkpoints < <(
    find "${guided_dir}" -type f \
      -path '*/controller_guided_backbone/checkpoint-epoch-1.pt'
  )
  if [[ ${#guided_checkpoints[@]} -ne 1 ]]; then
    echo "Expected one guided checkpoint, found ${#guided_checkpoints[@]}." >&2
    exit 1
  fi
  guided_checkpoint="$(realpath "${guided_checkpoints[0]}")"

  echo "Cycle ${cycle}: gating backbone changes on cycle dev..."
  pixi run -e fallback looped-transformer --config-dir conf eval \
    --checkpoint "${current_backbone}" \
    --data-file data/mnli_cycle_dev.tsv --batch-size 64 \
    --output-dir "${baseline_eval_dir}"
  pixi run -e fallback looped-transformer --config-dir conf eval \
    --checkpoint "${guided_checkpoint}" \
    --data-file data/mnli_cycle_dev.tsv --batch-size 64 \
    --output-dir "${guided_eval_dir}"
  pixi run -e fallback python scripts/compare_backbone_evaluations.py \
    "${baseline_eval_dir}/summary.json" "${guided_eval_dir}/summary.json"

  current_backbone="${guided_checkpoint}"
  printf '%s\n' "${current_backbone}" > "${output_root}/accepted-backbone.txt"
done

echo "Alternating training completed. Accepted backbone: ${current_backbone}"
