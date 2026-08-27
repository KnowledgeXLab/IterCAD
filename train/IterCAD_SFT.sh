#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MS_SWIFT_DIR="${REPO_ROOT}/train/ms-swift"
SFT_DATASET="${REPO_ROOT}/IterCAD_data/SFT/IterCAD_SFT.jsonl"
USER_NAME="${USER:-$(id -un)}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/IterCAD-SFT}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
SAVE_STEPS="${SAVE_STEPS:-500}"

export SKIP_MULTIMODAL_MTP_VALIDATION="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/tmp/${USER_NAME}/triton_cache_itercad_sft"
export MASTER_PORT="29517"
export MAX_PIXELS="1003520"
export VIDEO_MAX_PIXELS="50176"
export FPS_MAX_FRAMES="12"
export NPROC_PER_NODE

mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${SFT_DATASET}" ]]; then
  echo "Missing SFT dataset: ${SFT_DATASET}" >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI not found. Install ms-swift first:" >&2
  echo "  cd ${MS_SWIFT_DIR} && pip install -e ." >&2
  exit 1
fi

echo "=== IterCAD SFT ==="
echo "Dataset : ${SFT_DATASET}"
echo "Model   : ${MODEL_PATH}"
echo "Output  : ${OUTPUT_DIR}"
echo "GPUs    : ${NPROC_PER_NODE}"
echo "================="

# Run from repo root so JSONL image paths (IterCAD_data/SFT/images/...) resolve correctly.
cd "${REPO_ROOT}"

swift sft \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --tuner_type full \
  --torch_dtype bfloat16  \
  --dataset "${SFT_DATASET}" \
  --load_from_cache_file true \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size 1 \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --max_length "${MAX_LENGTH}" \
  --dataloader_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 100 \
  --save_only_model true \
  --deepspeed zero2 \
  --use_logits_to_keep true \
  --padding_free true \
  --attn_impl flash_attn \
  --report_to tensorboard
