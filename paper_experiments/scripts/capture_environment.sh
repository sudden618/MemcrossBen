#!/usr/bin/env bash
# Capture the actual per-run hardware/software environment required by checklist 4.8.
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY"
  exit 2
fi
OUTPUT_DIR="$1"
mkdir -p "${OUTPUT_DIR}"
{
  date --iso-8601=seconds
  uname -a
  python --version
  nvidia-smi
  nvcc --version || true
  lscpu
  free -h
  df -h .
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
} > "${OUTPUT_DIR}/hardware-software.txt"
python -m pip freeze > "${OUTPUT_DIR}/pip-freeze.txt"
python - <<'PY' > "${OUTPUT_DIR}/key-versions.txt"
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "transformers", "vllm", "faiss-cpu", "pymilvus",
             "mineru", "magic-pdf", "sentence-transformers", "colpali-engine"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "not installed")
PY
