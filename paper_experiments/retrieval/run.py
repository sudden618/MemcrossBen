"""CLI for index construction, document/page retrieval, and strict evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index import PageEmbedder, build_index, read_jsonl, search_index
from experiment_utils import make_run_id, metadata
from .metrics import all_some_none, missed_documents, normalize_evidence_sets, strict_doc_at_k, strict_page_at_k

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def command_build(args: argparse.Namespace) -> None:
    pages = read_jsonl(args.pages)
    embedder = PageEmbedder(args.model, args.models_config, args.model_root)
    build_index(pages, embedder.encode_pages(pages, args.batch_size), args.index_dir)


def command_search(args: argparse.Namespace) -> None:
    qa = read_jsonl(args.qa)
    embedder = PageEmbedder(args.model, args.models_config, args.model_root)
    candidates = search_index(
        args.index_dir, embedder.encode_queries([item["question"] for item in qa], args.batch_size), args.top_k
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {"top_k": args.top_k, "index_dir": str(Path(args.index_dir)), "modality": embedder.modality}
    run_id = make_run_id(args.model, args.model, args.seed, config, args.run_id)
    run_meta = metadata(args.model, args.model, args.seed, config, run_id)
    with output.open("w", encoding="utf-8") as stream:
        for item, recall in zip(qa, candidates):
            # Both page-level and deduplicated document-level order are emitted.
            docs = list(dict.fromkeys(hit["file"] for hit in recall))
            stream.write(json.dumps({"id": item["id"], **run_meta, "recall": recall, "doc_recall": docs}, ensure_ascii=False) + "\n")


def command_evaluate(args: argparse.Namespace) -> None:
    qa = {str(item["id"]): item for item in read_jsonl(args.qa)}
    recalls = read_jsonl(args.recall)
    source_meta = {
        key: recalls[0].get(key) if recalls else None
        for key in ("method", "model", "seed", "config", "run_id")
    }
    rows = []
    for result in recalls:
        item = qa[str(result["id"])]
        recall = result.get("recall", [])
        if not normalize_evidence_sets(item, args.evidence_mode):
            rows.append({"id": item["id"], **source_meta, "valid_evidence": 0})
            continue
        row = {"id": item["id"], **source_meta, "valid_evidence": 1}
        for k in args.ks:
            row[f"Strict Doc@{k}"] = int(strict_doc_at_k(item, recall, k, args.evidence_mode))
            row[f"Strict Page@{k}"] = int(strict_page_at_k(item, recall, k, args.evidence_mode))
            row[f"All/Some/None@{k}"] = all_some_none(item, recall, k, args.evidence_mode)
            row[f"Miss@{k}"] = missed_documents(item, recall, k, args.evidence_mode)
        rows.append(row)
    valid = [row for row in rows if row["valid_evidence"]]
    summary = {**source_meta, "count": len(rows), "valid_count": len(valid), "excluded_invalid_evidence": len(rows) - len(valid),
               "aggregation": "micro-average over questions", "denominator": "valid questions",
               "evidence_mode": args.evidence_mode, "rows": rows, "means": {}}
    for k in args.ks:
        for metric in ("Strict Doc", "Strict Page"):
            key = f"{metric}@{k}"
            summary["means"][key] = sum(row[key] for row in valid) / len(valid) if valid else None
    write_json(args.output, summary)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--model-root", default=str(ROOT / "models"))
    common.add_argument("--models-config", default=str(ROOT / "configs/models.yaml"))
    common.add_argument("--batch-size", type=int, default=8)
    build = sub.add_parser("build", parents=[common])
    build.add_argument("--pages", required=True)
    build.add_argument("--index-dir", required=True)
    build.set_defaults(func=command_build)
    search = sub.add_parser("search", parents=[common])
    search.add_argument("--qa", required=True)
    search.add_argument("--index-dir", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--run-id")
    search.set_defaults(func=command_search)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--qa", required=True)
    evaluate.add_argument("--recall", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--ks", nargs="+", type=int, default=[10, 15, 20])
    evaluate.add_argument("--evidence-mode", choices=["gold", "equivalent"], default="equivalent")
    evaluate.set_defaults(func=command_evaluate)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
