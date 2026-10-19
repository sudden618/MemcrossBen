"""
Resumable multi-Judge generation evaluation with raw-output preservation.

Paper correspondence:
- Generation Evaluation / Coverage, Accuracy, Attribution, Alignment
- Generation Evaluation / Multiple-Judge Mean Aggregation and Overall
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

from tqdm import tqdm

from experiment_utils import completed_ids, make_run_id, metadata, with_retry
from .judge_prompts import build_judge_messages

FIELDS = ("coverage", "accuracy", "attribution", "alignment")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_judge_output(raw: str) -> dict:
    """Parse plain/fenced JSON and reject missing or out-of-range dimension scores."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    scores = {}
    for field in FIELDS:
        score = int(value[field])
        if not 1 <= score <= 5:
            raise ValueError(f"{field} score outside 1..5: {score}")
        scores[field] = score
    return {**scores, "rationale": str(value.get("rationale", ""))}


def main(args: argparse.Namespace) -> None:
    from openai import OpenAI

    qa = {str(item["id"]): item for item in read_jsonl(args.qa)}
    answers = read_jsonl(args.answers)
    done = completed_ids(args.output) if args.resume else set()
    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url, timeout=args.timeout)
    config = {
        "judge_models": args.judge_models,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "aggregation": "arithmetic mean across valid judges; Overall=mean of four axes",
        "api_call_date": args.api_call_date,
    }
    model_label = "+".join(args.judge_models)
    run_id = make_run_id("generation_judge", model_label, args.seed, config, args.run_id)
    run_meta = metadata("generation_judge", model_label, args.seed, config, run_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for answer in tqdm(answers, desc="judge"):
            if str(answer["id"]) in done:
                continue
            item = qa[str(answer["id"])]
            messages = build_judge_messages(
                item["question"],
                str(item.get("answer") or item.get("reference_answer") or ""),
                item.get("evidence", []),
                answer.get("answer", ""),
                answer.get("citations", []),
            )
            outputs = []
            for judge_model in args.judge_models:
                try:
                    completion = with_retry(
                        lambda model=judge_model: client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        ),
                        args.max_retries,
                        args.retry_initial_delay,
                        args.retry_max_delay,
                    )
                    raw = completion.choices[0].message.content
                    outputs.append({"model": judge_model, "raw": raw, "parsed": parse_judge_output(raw), "error": None})
                except Exception as exc:
                    outputs.append({"model": judge_model, "raw": None, "parsed": None, "error": str(exc)})
            valid = [entry["parsed"] for entry in outputs if entry["parsed"]]
            scores = {
                field: sum(entry[field] for entry in valid) / len(valid) if valid else None
                for field in FIELDS
            }
            overall = sum(scores.values()) / len(FIELDS) if all(value is not None for value in scores.values()) else None
            stream.write(json.dumps({
                "id": answer["id"],
                **run_meta,
                "status": "complete" if valid else "failed",
                "judge_inputs": {
                    "has_reference_answer": bool(item.get("answer") or item.get("reference_answer")),
                    "has_evidence_pages": bool(item.get("evidence")),
                    "has_model_citations": bool(answer.get("citations")),
                },
                "judge_outputs": outputs,
                "judge_result": {**scores, "score": overall},
            }, ensure_ascii=False) + "\n")
            stream.flush()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--qa", required=True)
    result.add_argument("--answers", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--judge-models", nargs="+", required=True)
    result.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1"))
    result.add_argument("--api-key", default=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    result.add_argument("--temperature", type=float, default=0.0)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--timeout", type=int, default=180)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--run-id")
    result.add_argument("--api-call-date", default=date.today().isoformat())
    result.add_argument("--max-retries", type=int, default=5)
    result.add_argument("--retry-initial-delay", type=float, default=1.0)
    result.add_argument("--retry-max-delay", type=float, default=30.0)
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
