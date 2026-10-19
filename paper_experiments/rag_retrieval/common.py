"""Shared page metadata, serialization, ranking, and recall-output helpers."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def page_key(item: dict) -> tuple[str, int]:
    return str(item["file"]), int(item["page"])


def page_id(item: dict) -> str:
    file_id, page = page_key(item)
    return f"{file_id}_{page}"


def parse_page_id(value: str) -> dict:
    """Parse `<document-id>_<page>` without breaking document IDs containing underscores."""
    file_id, page = Path(value).stem.rsplit("_", 1)
    return {"file": file_id, "page": int(page)}


def interleave(first: list[dict], second: list[dict], top_k: int) -> list[dict]:
    """Alternate two ranked lists and remove duplicate document/page candidates."""
    merged, seen = [], set()
    for index in range(max(len(first), len(second))):
        for source in (first, second):
            if index >= len(source):
                continue
            item = source[index]
            key = page_key(item)
            if key in seen:
                continue
            merged.append({**item, "rank": len(merged) + 1})
            seen.add(key)
            if len(merged) == top_k:
                return merged
    return merged


def write_recall(path: str | Path, qa: list[dict], results: list[dict], run_metadata: dict, append: bool = False) -> None:
    """Write the common recall JSONL consumed by generation and strict evaluation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as stream:
        for item, result in zip(qa, results):
            record = {"id": item["id"], **run_metadata, **result}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
