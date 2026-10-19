#!/usr/bin/env bash
# Main retrieval → strict evaluation → NaiveRAG pipeline.
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 QA.jsonl PAGES.jsonl MODEL_KEY MODEL_DIR WORK_DIR GENERATOR_MODEL"
  exit 2
fi

QA_PATH="$1"
PAGES_PATH="$2"
MODEL_KEY="$3"
MODEL_DIR="$4"
WORK_DIR="$5"
GENERATOR_MODEL="$6"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"
python -m retrieval.run build --pages "${PAGES_PATH}" --model "${MODEL_KEY}" \
  --model-root "${MODEL_DIR}" --index-dir "${WORK_DIR}/index"
python -m retrieval.run search --qa "${QA_PATH}" --model "${MODEL_KEY}" \
  --model-root "${MODEL_DIR}" --index-dir "${WORK_DIR}/index" \
  --output "${WORK_DIR}/recall.jsonl" --top-k 20
python -m retrieval.run evaluate --qa "${QA_PATH}" --recall "${WORK_DIR}/recall.jsonl" \
  --output "${WORK_DIR}/strict_metrics.json"
python -m generation.run --method naive_rag --qa "${QA_PATH}" \
  --recall "${WORK_DIR}/recall.jsonl" --pages "${PAGES_PATH}" \
  --output "${WORK_DIR}/generation.jsonl" --model "${GENERATOR_MODEL}"
