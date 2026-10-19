# Multi-document Retrieval and Generation Experiments

This repository reproduces the paper's embedding retrieval, RAG retrieval, answer
generation, strict evidence evaluation, LLM judging, and breakdown analyses. It
contains source code and a tiny hand-verifiable example, but no benchmark data,
model weights, indexes, API credentials, or paper result files.

> Release blocker: `LICENSE` is currently an all-rights-reserved selection notice.
> The copyright holder must replace it with the approved license before publication.

## Repository layout

```text
data_processing/  portable input validation and normalization
retrieval/        text/multimodal embedding index, Top-K search, strict metrics
rag_retrieval/    HippoRAG, M3DocRAG, MDocAgent, ViDoRAG native retrieval
generation/       DirectQA, NaiveRAG, baseline generation, prompts and truncation
evaluation/       public Judge prompts, API runner and robust output parser
analysis/         per-question/grouped analysis, tables and SVG figures
configs/          model metadata, paper mapping and pipeline configuration
examples/         three-question hand-verifiable smoke-test data
tests/            metric and Judge parser unit tests
scripts/          named experiment wrappers and the total pipeline entry point
```

Generated data belongs under `data/`, and indexes/results under `outputs/`; both are
gitignored. Every path accepted by a script is configurable. The repository contains
no author-machine path, token, cache, model weight, or private package.

## Data

The benchmark cannot be redistributed with this code. Obtain it from the dataset
release specified by the paper, then provide:

- QA JSONL: `id`, `question`, `type` (`Text`, `Figure`, `Meta`), `subset`
  (`Papers`, `Patents`), optional reference `answer`, and evidence.
- Page JSONL: `file`, `page`, `text`, and/or a portable `image_path`.

Preferred evidence schema:

```json
{"evidence_sets": [
  [{"file": "docA", "pages": [1, 2]}, {"file": "docB", "pages": [4]}],
  [{"file": "docA", "pages": [3]}, {"file": "docB", "pages": [5, 6]}]
]}
```

Each outer entry is one complete valid combination. Alternatives are never mixed.
Legacy per-document `evidence[*].page` is supported and expanded as independent
alternatives.

Validate and normalize inputs:

```bash
python -m data_processing.prepare --qa /path/qa.jsonl --pages /path/pages.jsonl \
  --output-dir data/main
```

Page images/text must already be extracted. MinerU was used upstream for document
parsing, but its exact historical version was not preserved; this unresolved provenance
item must be filled from the original parsing environment before claiming exact raw-PDF
reproduction.

## Environment installation

Conda:

```bash
conda env create -f environment.yml
conda activate paper-experiments
pip install -r requirements-rag.txt  # only for native RAG retrieval
```

Docker:

```bash
docker build -t paper-experiments .
docker run --gpus all --rm -v "$PWD:/workspace/paper_experiments" paper-experiments
```

See `COMPUTE_ENVIRONMENT.md` for hardware, GPU counts, software versions and disk
estimates. Native HippoRAG may require its separate Python 3.10 environment because
its upstream dependency pins differ from the main environment.

## Model downloads and external access

Model weights are never included. `configs/models.yaml` records the complete repository
path, revision field, weight license, access requirement, precision, backend and
quantization. For the final run, replace mutable `main` revisions with commit hashes.

```bash
bash scripts/download_models.sh text_qwen8b
bash scripts/download_models.sh image_qwen_vl_8b
bash scripts/download_models.sh gme_qwen2_vl_2b
bash scripts/download_models.sh colpali_v1_2
bash scripts/download_models.sh colqwen2_v1
```

Public models do not require a token. Gated models require `HF_TOKEN` and prior model
license/access approval. The downloader reads authentication only through Hugging Face
CLI environment/configuration. Verify every upstream model license before redistribution.

## One-command main pipeline

Copy and edit the example configuration, especially data paths, complete model IDs and
immutable revisions:

```bash
cp configs/main_experiment.example.yaml configs/main_experiment.yaml
python scripts/run_all.py --config configs/main_experiment.yaml
```

Successful stages receive markers under `outputs/pipeline_state`. Re-running skips
completed stages; use `--force` to rerun them. Question-level retrieval, generation and
Judge commands also support `--resume`.

## Index and retrieval experiments

Text or multimodal embedding retrieval:

```bash
bash scripts/run_text_embedding.sh build --pages data/main/pages.jsonl \
  --model text_qwen8b --index-dir outputs/index/text_qwen8b
bash scripts/run_text_embedding.sh search --qa data/main/qa.jsonl \
  --model text_qwen8b --index-dir outputs/index/text_qwen8b \
  --output outputs/recall/text_qwen8b.jsonl --top-k 20

bash scripts/run_multimodal_embedding.sh build --pages data/main/pages.jsonl \
  --model image_qwen_vl_8b --index-dir outputs/index/image_qwen_vl_8b
```

Native RAG retrieval uses the same build/search interface:

```bash
bash scripts/run_rag_baselines.sh build --method mdocagent \
  --pages data/main/pages.jsonl --index-dir outputs/index/mdocagent
bash scripts/run_rag_baselines.sh search --method mdocagent \
  --qa data/main/qa.jsonl --index-dir outputs/index/mdocagent \
  --output outputs/recall/mdocagent.jsonl --top-k 20 --resume
```

Valid methods are `hipporag`, `m3docrag`, `mdocagent`, and `vidorag`:

| Method | Retrieval implementation |
|---|---|
| HippoRAG | LLM graph construction and graph retrieval |
| M3DocRAG | ColPali multi-vector page/query MaxSim |
| MDocAgent | BGE text + GME image dense retrieval, interleaved deduplication |
| ViDoRAG | BGE text + ColQwen MaxSim, per-route GMM filtering and fusion |

## Generation

All public system prompts and method prompts are in `generation/prompts.py`.

```bash
# DirectQA
bash scripts/run_directqa.sh --qa data/main/qa.jsonl \
  --output outputs/generation/directqa.jsonl \
  --model Qwen/Qwen3-VL-32B-Instruct --model-revision COMMIT_SHA --resume

# NaiveRAG or any RAG baseline
bash scripts/run_generation.sh --method naive_rag --qa data/main/qa.jsonl \
  --recall outputs/recall/text_qwen8b.jsonl --pages data/main/pages.jsonl \
  --output outputs/generation/naive_rag.jsonl \
  --model Qwen/Qwen3-VL-32B-Instruct --resume
```

Generation uses an OpenAI-compatible backend (`LLM_BASE_URL`) and reads keys only from
`OPENAI_API_KEY`. Ranked evidence blocks are injected whole until the token budget is
reached. Answers and citations are saved together. API calls retry five times by default
with exponential backoff; every successful item is flushed immediately.

Every output row contains `method`, `model`, `seed`, `config`, `run_id`, and `id`.
Generation additionally stores token usage, enabling call-count/cost estimation. Total
cost is `prompt_tokens × provider_input_price + completion_tokens × provider_output_price`;
prices are intentionally not hardcoded because providers change them.

## LLM Judge

Coverage, Accuracy, Attribution and Alignment prompts and 1–5 rubrics are public in
`evaluation/judge_prompts.py`. Judge input includes the question, optional reference
answer, gold evidence pages, candidate answer and candidate citations.

```bash
bash scripts/run_judge.sh --qa data/main/qa.jsonl \
  --answers outputs/generation/naive_rag.jsonl \
  --output outputs/judge/naive_rag.jsonl \
  --judge-models provider/exact-model-snapshot --api-call-date YYYY-MM-DD --resume
```

Use `JUDGE_BASE_URL` and `JUDGE_API_KEY`. Multiple `--judge-models` are independently
called; valid scores are averaged per dimension. Overall is the arithmetic mean of the
four aggregated dimensions. Raw responses, parsed responses and errors are retained.
Malformed responses and timeouts use the same bounded retry policy. The run config records
model names, date and inference parameters.

## Strict metrics and equivalent evidence

```bash
# Gold Only: only the first complete evidence combination is valid
python -m retrieval.run evaluate --qa data/main/qa.jsonl \
  --recall outputs/recall/method.jsonl --output outputs/metrics/gold.json \
  --evidence-mode gold

# Equivalent Allowed: any complete declared combination is valid
python -m retrieval.run evaluate --qa data/main/qa.jsonl \
  --recall outputs/recall/method.jsonl --output outputs/metrics/equivalent.json \
  --evidence-mode equivalent
```

- Strict Doc@K requires every document from one complete valid combination.
- Strict Page@K requires every page from one complete valid combination.
- Partial combinations fail; alternatives are not merged.
- Duplicate recalled pages are removed before Top-K and do not consume rank.
- If fewer than K unique pages exist, all available pages are evaluated.
- Invalid/no-evidence questions are excluded and counted explicitly.
- Reported overall values are micro-averages of per-question binary scores; the
  denominator is valid question count, not evidence count.
- Per-question rows and overall results are emitted together.

Run the hand-verifiable tests:

```bash
python -m unittest discover -v
python -m analysis.run --qa examples/qa.jsonl --recall examples/recall.jsonl \
  --judge examples/judge.jsonl --output-dir outputs/example
```

The example's Equivalent Strict Page metrics are 1.0 at 10/15/20. Gold Only is lower
because question 1 retrieves its declared equivalent combination rather than its first
gold combination.

