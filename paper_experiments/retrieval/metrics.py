"""
Strict document/page retrieval metrics with explicit equivalent evidence sets.

Paper correspondence:
- Retrieval Experiments / Strict Doc@K and Strict Page@K
- Evaluation / Gold Only versus Equivalent Allowed

Aggregation contract:
- Per-question binary success is micro-averaged over valid questions.
- The denominator is the number of questions with at least one valid evidence set.
- Empty/invalid evidence is excluded and reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable


def _deduplicated_top_k(recall: Iterable[dict], k: int) -> list[dict]:
    """Take the first K unique pages; repeated pages never consume additional rank."""
    output, seen = [], set()
    for item in recall:
        key = (str(item["file"]), int(item["page"]))
        if key in seen:
            continue
        output.append({"file": key[0], "page": key[1]})
        seen.add(key)
        if len(output) == k:
            break
    return output


def normalize_evidence_sets(item_or_evidence: dict | list, mode: str = "equivalent") -> list[list[dict]]:
    """Return full, non-mixing evidence combinations.

    Preferred schema is `evidence_sets`: a list of complete alternatives, each containing
    `{file, pages}` entries. Legacy per-document `evidence[*].page` is expanded as a
    Cartesian product because those alternatives were defined independently.
    """
    if mode not in {"gold", "equivalent"}:
        raise ValueError("mode must be 'gold' or 'equivalent'")
    if isinstance(item_or_evidence, dict) and item_or_evidence.get("evidence_sets"):
        combinations = item_or_evidence["evidence_sets"]
        combinations = combinations[:1] if mode == "gold" else combinations
        return [
            [{"file": str(entry["file"]), "pages": [int(page) for page in entry["pages"]]} for entry in combination]
            for combination in combinations
            if combination
        ]

    evidence = item_or_evidence.get("evidence", []) if isinstance(item_or_evidence, dict) else item_or_evidence
    choices = []
    for entry in evidence:
        alternatives = entry.get("page") or entry.get("pages") or []
        if alternatives and not isinstance(alternatives[0], list):
            alternatives = [[page] for page in alternatives]
        if mode == "gold":
            alternatives = alternatives[:1]
        choices.append([
            {"file": str(entry["file"]), "pages": [int(page) for page in pages]}
            for pages in alternatives
            if pages
        ])
    if not choices or any(not choice for choice in choices):
        return []
    return [list(combination) for combination in product(*choices)]


def _retrieved(recall: Iterable[dict], k: int) -> tuple[set[str], dict[str, set[int]]]:
    pages: dict[str, set[int]] = defaultdict(set)
    for item in _deduplicated_top_k(recall, k):
        pages[item["file"]].add(item["page"])
    return set(pages), pages


def _combination_page_hit(combination: list[dict], pages: dict[str, set[int]]) -> bool:
    return all(set(entry["pages"]) <= pages.get(entry["file"], set()) for entry in combination)


def strict_doc_at_k(item_or_evidence: dict | list, recall: list[dict], k: int, mode: str = "equivalent") -> bool:
    """Succeed when Top-K covers every required document in any valid full combination."""
    combinations = normalize_evidence_sets(item_or_evidence, mode)
    docs, _ = _retrieved(recall, k)
    return bool(combinations) and any({entry["file"] for entry in combination} <= docs for combination in combinations)


def strict_page_at_k(item_or_evidence: dict | list, recall: list[dict], k: int, mode: str = "equivalent") -> bool:
    """Succeed only after completely covering one valid full page combination."""
    combinations = normalize_evidence_sets(item_or_evidence, mode)
    _, pages = _retrieved(recall, k)
    return bool(combinations) and any(_combination_page_hit(combination, pages) for combination in combinations)


def matched_document_count(item_or_evidence: dict | list, recall: list[dict], k: int, mode: str = "equivalent") -> tuple[int, int]:
    """Return the best matched-document count and required count across valid combinations."""
    combinations = normalize_evidence_sets(item_or_evidence, mode)
    _, pages = _retrieved(recall, k)
    if not combinations:
        return 0, 0
    scored = [
        (sum(set(entry["pages"]) <= pages.get(entry["file"], set()) for entry in combination), len(combination))
        for combination in combinations
    ]
    return max(scored, key=lambda pair: (pair[0] / pair[1], pair[0]))


def all_some_none(item_or_evidence: dict | list, recall: list[dict], k: int, mode: str = "equivalent") -> str:
    """Decompose the best complete-document evidence coverage into All, Some, or None."""
    hits, required = matched_document_count(item_or_evidence, recall, k, mode)
    if required and hits == required:
        return "All"
    return "Some" if hits else "None"


def missed_documents(item_or_evidence: dict | list, recall: list[dict], k: int, mode: str = "equivalent") -> int:
    """Count missing documents for the best-matching valid evidence combination."""
    hits, required = matched_document_count(item_or_evidence, recall, k, mode)
    return required - hits
