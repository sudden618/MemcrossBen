"""Shared run identity, configuration snapshots, retry, and resumable JSONL utilities."""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def make_run_id(method: str, model: str, seed: int, config: dict, explicit: str | None = None) -> str:
    """Create a stable, readable run ID from method, UTC time, and configuration digest."""
    if explicit:
        return explicit
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_method = method.replace("/", "_")
    return f"{safe_method}-{stamp}-s{seed}-{digest}"


def metadata(method: str, model: str, seed: int, config: dict, run_id: str) -> dict:
    """Return the mandatory metadata included in every question-level output."""
    return {"method": method, "model": model, "seed": seed, "config": config, "run_id": run_id}


def completed_ids(path: str | Path) -> set[str]:
    """Read valid completed records and ignore a possibly truncated final line."""
    output = Path(path)
    if not output.exists():
        return set()
    ids = set()
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in item and item.get("status") != "failed":
                ids.add(str(item["id"]))
    return ids


def append_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Append and flush each record so interruption loses at most one active sample."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


def with_retry(call: Callable[[], T], attempts: int, initial_delay: float, max_delay: float) -> T:
    """Retry transient API failures with exponential backoff and bounded jitter."""
    error = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # SDK/provider exceptions differ across backends.
            error = exc
            if attempt + 1 == attempts:
                break
            delay = min(max_delay, initial_delay * (2**attempt))
            time.sleep(delay + random.random() * min(1.0, delay / 4))
    raise RuntimeError(f"API call failed after {attempts} attempts") from error
