"""
Unified CLI for native HippoRAG, M3DocRAG, MDocAgent, and ViDoRAG retrieval.

Paper correspondence:
- RAG Baselines / Index Construction
- RAG Baselines / Top-K Retrieval
- Output is normalized for Strict Doc@K and Strict Page@K evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiment_utils import completed_ids, make_run_id, metadata
from .common import read_jsonl, write_recall

ROOT = Path(__file__).resolve().parents[1]


def config_for(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    return config["methods"][args.method]


def resolve_model(value: str, model_root: str) -> str:
    """Prefer a downloaded local model directory, otherwise allow a repository identifier."""
    local = Path(model_root) / value
    return str(local if local.exists() else value)


def command_build(args: argparse.Namespace) -> None:
    pages, config = read_jsonl(args.pages), config_for(args)
    if args.method == "hipporag":
        from . import hipporag

        hipporag.build(pages, args)
    elif args.method == "m3docrag":
        from . import m3docrag

        m3docrag.build(pages, args.index_dir, resolve_model(config["model"], args.model_root), args.device, args.batch_size)
    elif args.method == "mdocagent":
        from . import mdocagent

        mdocagent.build(
            pages,
            args.index_dir,
            resolve_model(config["text_model"], args.model_root),
            resolve_model(config["image_model"], args.model_root),
            args.batch_size,
        )
    else:
        from . import vidorag

        vidorag.build(
            pages,
            args.index_dir,
            resolve_model(config["text_model"], args.model_root),
            resolve_model(config["image_model"], args.model_root),
            args.device,
            args.batch_size,
        )


def command_search(args: argparse.Namespace) -> None:
    qa, config = read_jsonl(args.qa), config_for(args)
    done = completed_ids(args.output) if args.resume else set()
    qa = [item for item in qa if str(item["id"]) not in done]
    questions = [item["question"] for item in qa]
    if args.method == "hipporag":
        from . import hipporag

        results = hipporag.search(questions, args)
    elif args.method == "m3docrag":
        from . import m3docrag

        results = m3docrag.search(
            questions, args.index_dir, resolve_model(config["model"], args.model_root), args.device, args.top_k
        )
    elif args.method == "mdocagent":
        from . import mdocagent

        results = mdocagent.search(
            questions,
            args.index_dir,
            resolve_model(config["text_model"], args.model_root),
            resolve_model(config["image_model"], args.model_root),
            args.top_k,
            args.candidate_k,
        )
    else:
        from . import vidorag

        results = vidorag.search(
            questions,
            args.index_dir,
            resolve_model(config["text_model"], args.model_root),
            resolve_model(config["image_model"], args.model_root),
            args.device,
            args.top_k,
            args.candidate_k,
            args.min_candidates,
        )
    model_name = config.get("model") or "+".join(
        value for key, value in config.items() if key.endswith("_model")
    ) or f'{args.hippo_llm_model}+{args.hippo_embedding_model}'
    run_config = {
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "min_candidates": args.min_candidates,
        "backend": config["backend"],
    }
    run_id = make_run_id(args.method, model_name, args.seed, run_config, args.run_id)
    write_recall(
        args.output,
        qa,
        results,
        metadata(args.method, model_name, args.seed, run_config, run_id),
        append=args.resume,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=["build", "search"])
    result.add_argument("--method", choices=["hipporag", "m3docrag", "mdocagent", "vidorag"], required=True)
    result.add_argument("--pages", help="Page JSONL; required by build.")
    result.add_argument("--qa", help="QA JSONL; required by search.")
    result.add_argument("--index-dir", required=True)
    result.add_argument("--output", help="Recall JSONL; required by search.")
    result.add_argument("--config", default=str(ROOT / "configs/rag_models.yaml"))
    result.add_argument("--model-root", default=str(ROOT / "models"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--candidate-k", type=int, default=30)
    result.add_argument("--min-candidates", type=int, default=15)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--run-id")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--hippo-llm-model", default="Llama-3.1-8B")
    result.add_argument("--hippo-embedding-model", default="text-embedding-qwen")
    result.add_argument("--hippo-llm-base-url", default="http://127.0.0.1:40507/v1")
    result.add_argument("--hippo-embedding-base-url", default="http://127.0.0.1:40508/v1")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "build" and not arguments.pages:
        parser().error("--pages is required by build")
    if arguments.command == "search" and (not arguments.qa or not arguments.output):
        parser().error("--qa and --output are required by search")
    (command_build if arguments.command == "build" else command_search)(arguments)
