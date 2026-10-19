"""
Native HippoRAG graph indexing and retrieval adapter.

Paper correspondence:
- RAG Baselines / HippoRAG
- Page IDs are embedded in indexed passages and recovered from graph retrieval.
"""

from __future__ import annotations

import re

from .common import parse_page_id

ID_PATTERN = re.compile(r"<<<ID:(.*?)>>>")


def _instance(args):
    from hipporag import HippoRAG

    return HippoRAG(
        save_dir=args.index_dir,
        llm_model_name=args.hippo_llm_model,
        embedding_model_name=args.hippo_embedding_model,
        llm_base_url=args.hippo_llm_base_url,
        embedding_base_url=args.hippo_embedding_base_url,
    )


def build(pages: list[dict], args) -> None:
    """Index page passages in HippoRAG while retaining stable page identifiers."""
    rag = _instance(args)
    docs = [f'<<<ID:{page["file"]}_{int(page["page"])}>>>\n{page["text"]}' for page in pages if page.get("text")]
    for start in range(0, len(docs), args.batch_size):
        rag.index(docs=docs[start : start + args.batch_size])


def search(questions: list[str], args) -> list[dict]:
    """Retrieve graph-ranked passages and normalize them to page-level candidates."""
    rag = _instance(args)
    batches = rag.retrieve(questions, num_to_retrieve=args.top_k)
    output = []
    for result in batches:
        recall, seen = [], set()
        for rank, doc in enumerate(result.docs, start=1):
            match = ID_PATTERN.search(doc)
            if not match:
                continue
            page = parse_page_id(match.group(1))
            key = (page["file"], page["page"])
            if key in seen:
                continue
            recall.append({**page, "rank": rank})
            seen.add(key)
        output.append({"recall": recall[: args.top_k]})
    return output
