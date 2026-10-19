"""
Rank-preserving evidence injection and maximum-context truncation.

Paper correspondence:
- Generation Experiments / Retrieval Result Injection
- Generation Experiments / Maximum Context Truncation Strategy
"""

from __future__ import annotations

from typing import Callable


def inject_context(
    recalls: list[dict],
    pages: dict[tuple[str, int], dict],
    token_count: Callable[[str], int],
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """Add complete ranked evidence blocks until the context token budget is exhausted."""
    blocks, citations = [], []
    used = 0
    for hit in recalls:
        key = (str(hit["file"]), int(hit["page"]))
        page = pages.get(key)
        if not page:
            continue
        content = str(page.get("text") or page.get("caption") or "")
        block = f'[Source {key[0]}:{key[1]}]\n{content}'
        cost = token_count(block)
        # Paper step: truncate at evidence-block boundaries, preserving retrieval rank and citations.
        if used + cost > max_tokens:
            break
        blocks.append(block)
        citations.append({"file": key[0], "page": key[1], "rank": hit.get("rank")})
        used += cost
    return "\n\n".join(blocks), citations
