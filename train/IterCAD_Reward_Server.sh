#!/usr/bin/env bash
# Run this script in the cadquery environment, in a separate terminal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/eval"
REWARD_SERVER="${SCRIPT_DIR}/cad_reward_server.py"

REWARD_HOST="${REWARD_HOST:-127.0.0.1}"
REWARD_PORT="${REWARD_PORT:-8765}"

if [[ ! -f "${EVAL_DIR}/code_executor.py" ]]; then
  echo "Missing CAD executor: ${EVAL_DIR}/code_executor.py" >&2
  exit 1
fi

if [[ ! -f "${REWARD_SERVER}" ]]; then
  echo "Missing reward server: ${REWARD_SERVER}" >&2
  exit 1
fi

export CAD_EVAL_DIR="${EVAL_DIR}"
export CAD_ISOLATE_EVAL="${CAD_ISOLATE_EVAL:-1}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

echo "=== IterCAD Reward Server ==="
echo "API  : http://${REWARD_HOST}:${REWARD_PORT}"
echo "Env  : ${CONDA_DEFAULT_ENV:-unknown} (expected: cadquery)"
echo "Stop : Ctrl+C"
echo "============================="

# Keep repo-relative STL paths resolvable.
cd "${REPO_ROOT}"
exec python "${REWARD_SERVER}" \
  --host "${REWARD_HOST}" \
  --port "${REWARD_PORT}" \
  --eval_dir "${EVAL_DIR}"
