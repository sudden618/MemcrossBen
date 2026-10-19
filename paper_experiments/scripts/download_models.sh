#!/usr/bin/env bash
# Download only explicitly selected model repositories; no weights live in this code package.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
MODEL_KEY="${1:-}"
REVISION="${2:-main}"

declare -A REPOS=(
  [text_bge]="BAAI/bge-m3"
  [text_kalm]="tencent/KaLM-Embedding-Gemma3-12B-2511"
  [text_llama]="nvidia/llama-embed-nemotron-8b"
  [text_qwen4b]="Qwen/Qwen3-Embedding-4B"
  [text_qwen8b]="Qwen/Qwen3-Embedding-8B"
  [image_qwen_vl_2b]="Qwen/Qwen3-VL-Embedding-2B"
  [image_qwen_vl_8b]="Qwen/Qwen3-VL-Embedding-8B"
  [image_colpali]="vidore/colpali-v1.3-merged"
  [image_rzen]="360zhinao/RzenEmbed"
  [image_vdr]="llamaindex/vdr-2b-multi-v1"
  [gme_qwen2_vl_2b]="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
  [colpali_v1_2]="vidore/colpali-v1.2-merged"
  [colqwen2_v1]="vidore/colqwen2-v1.0-merged"
)

if [[ -z "${MODEL_KEY}" || -z "${REPOS[${MODEL_KEY}]:-}" ]]; then
  echo "Usage: $0 <model-key> [revision-or-commit-sha]"
  printf 'Available: %s\n' "${!REPOS[@]}" | sort
  exit 2
fi

mkdir -p "${MODEL_DIR}"
hf download "${REPOS[${MODEL_KEY}]}" \
  --revision "${REVISION}" \
  --local-dir "${MODEL_DIR}/${MODEL_KEY}"
