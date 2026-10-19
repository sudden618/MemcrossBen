"""
Unified DirectQA, NaiveRAG, and RAG-baseline answer generation.

Paper correspondence:
- Generation Experiments / Compared Methods
- Generation Experiments / Different Generator Models
- Generation Experiments / Prompt, Retrieval Injection, Truncation, Answers and Citations
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from experiment_utils import completed_ids, make_run_id, metadata, with_retry
from .context import inject_context
from .prompts import build_messages


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def token_counter(model: str):
    """Return a stable tokenizer counter, falling back to cl100k_base for local models."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


def main(args: argparse.Namespace) -> None:
    qa = read_jsonl(args.qa)
    recall_by_id = {}
    if args.recall:
        recall_by_id = {str(item["id"]): item.get("recall", []) for item in read_jsonl(args.recall)}
    pages = {}
    if args.pages:
        pages = {(str(item["file"]), int(item["page"])): item for item in read_jsonl(args.pages)}

    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url, timeout=args.timeout)
    count_tokens = token_counter(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and output.exists():
        completed = completed_ids(output)

    config = {
        "top_k": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "model_revision": args.model_revision,
        "inference_backend": args.inference_backend,
        "quantization": args.quantization,
        "api_call_date": args.api_call_date,
    }
    run_id = make_run_id(args.method, args.model, args.seed, config, args.run_id)
    run_meta = metadata(args.method, args.model, args.seed, config, run_id)
    with output.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for item in tqdm(qa, desc=args.method):
            if str(item["id"]) in completed:
                continue
            context, citations = "", []
            if args.method != "directqa":
                context, citations = inject_context(
                    recall_by_id.get(str(item["id"]), [])[: args.top_k],
                    pages,
                    count_tokens,
                    args.max_context_tokens,
                )
            messages = build_messages(args.method, item["question"], context)
            try:
                completion = with_retry(
                    lambda: client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_output_tokens,
                    ),
                    args.max_retries,
                    args.retry_initial_delay,
                    args.retry_max_delay,
                )
                record = {
                    "id": item["id"],
                    **run_meta,
                    "status": "complete",
                    "answer": completion.choices[0].message.content,
                    "citations": citations,
                    "usage": {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                        "total_tokens": getattr(completion.usage, "total_tokens", None),
                    },
                    "error": None,
                }
            except Exception as exc:
                record = {
                    "id": item["id"], **run_meta, "status": "failed", "answer": None,
                    "citations": citations, "usage": None, "error": str(exc),
                }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--method",
        choices=["directqa", "naive_rag", "hipporag", "m3docrag", "mdocagent", "vidorag"],
        required=True,
    )
    result.add_argument("--qa", required=True)
    result.add_argument("--recall")
    result.add_argument("--pages")
    result.add_argument("--output", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    result.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--max-context-tokens", type=int, default=32768)
    result.add_argument("--max-output-tokens", type=int, default=1024)
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--run-id")
    result.add_argument("--model-revision", default="unspecified")
    result.add_argument("--inference-backend", default="vLLM OpenAI-compatible API")
    result.add_argument("--quantization", default="none")
    result.add_argument("--api-call-date", default=date.today().isoformat())
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--retry-initial-delay", type=float, default=1.0)
    result.add_argument("--retry-max-delay", type=float, default=30.0)
    result.add_argument("--timeout", type=int, default=300)
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
