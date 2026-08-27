#!/usr/bin/env bash
# IterCAD RL (GRPO) training.
# Start train/IterCAD_Reward_Server.sh in the cadquery environment first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MS_SWIFT_DIR="${REPO_ROOT}/train/ms-swift"
RL_DATASET="${REPO_ROOT}/IterCAD_data/RL/IterCAD_RL.jsonl"
PLUGIN_PATH="${MS_SWIFT_DIR}/examples/train/grpo/plugin/cad_grpo_plugin.py"
USER_NAME="${USER:-$(id -un)}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/model/IterCAD-RL}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/IterCAD-RL}"
CAD_REWARD_API_URL="${CAD_REWARD_API_URL:-http://127.0.0.1:8765}"

export CAD_MASK_TRAJ="1"
export CAD_MASK_TRAJ_K="2"
export CAD_MASK_TRAJ_STAGNANT_CD="0.01"
export CAD_REWARD_API_URL
export CAD_SAMPLE_POINTS="8192"
export CAD_PROGRESS_CD_THRESHOLD="0.001"
export CAD_PROGRESS_BONUS="0.2"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

export SKIP_MULTIMODAL_MTP_VALIDATION="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export TORCH_COMPILE_DISABLE="1"
export TRITON_CACHE_DIR="/tmp/${USER_NAME}/triton_cache_itercad_rl"
export TRITON_HOME="/tmp/${USER_NAME}/triton_home_itercad_rl"
export MASTER_PORT="29518"
export MAX_PIXELS="1003520"
export VIDEO_MAX_PIXELS="50176"
export FPS_MAX_FRAMES="12"
export NPROC_PER_NODE

mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${OUTPUT_DIR}"

if [[ ! -f "${RL_DATASET}" ]]; then
  echo "Missing RL dataset: ${RL_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${PLUGIN_PATH}" ]]; then
  echo "Missing GRPO plugin: ${PLUGIN_PATH}" >&2
  exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI not found. Install ms-swift first:" >&2
  echo "  cd ${MS_SWIFT_DIR} && pip install -e ." >&2
  exit 1
fi

if ! python - "${CAD_REWARD_API_URL}/health" <<'PY'
import sys
import urllib.request

try:
    urllib.request.urlopen(sys.argv[1], timeout=3).read()
    raise SystemExit(0)
except Exception as exc:
    print(f"CAD reward server is unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "Start it first in the cadquery environment:" >&2
  echo "  conda activate cadquery && bash ${SCRIPT_DIR}/IterCAD_Reward_Server.sh" >&2
  exit 1
fi

echo "=== IterCAD RL (GRPO) ==="
echo "Dataset : ${RL_DATASET}"
echo "Model   : ${MODEL_PATH}"
echo "Plugin  : ${PLUGIN_PATH}"
echo "Output  : ${OUTPUT_DIR}"
echo "GPUs    : ${NPROC_PER_NODE}"
echo "========================"

cd "${REPO_ROOT}"

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --external_plugins "${PLUGIN_PATH}" \
  --reward_funcs cad_chamfer cad_format cad_progress cad_cd_value cad_invalid \
  --reward_weights 1.0 0.5 0.2 0.0 0.0 \
  --multi_turn_scheduler cad_scheduler \
  --max_turns 5 \
  --dataset "${RL_DATASET}" \
  --load_from_cache_file true \
  --enable_thinking false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.5 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 32768 \
  --sleep_level 1 \
  --train_type full \
  --importance_sampling_level sequence \
  --torch_dtype bfloat16 \
  --max_length 32768 \
  --max_completion_length 8192 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --temperature 1.0 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --offload_model true \
  --offload_optimizer true \
  --epsilon 3e-4 \
  --epsilon_high 4e-4 \
  --beta 0.0 \
  --steps_per_generation 4 \
  --scale_rewards none \
  --deepspeed zero3 \
  --dataloader_num_workers 8 \
  --save_steps 10 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --log_completions true \
  --vllm_mm_processor_cache_gb 0 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard

# Example:
# srun -N 1 --gres=gpu:8 bash train/IterCAD_RL.sh
