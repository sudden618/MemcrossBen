"""
Implementation of page-level indexing and Top-K retrieval.

Paper correspondence:
- Retrieval Experiments / Page Index Construction
- Retrieval Experiments / Text and Multimodal Embedding Models
- Retrieval Experiments / Top-K Candidate Generation

Exact manuscript numbering is maintained in configs/paper_correspondence.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import yaml
from PIL import Image
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class PageEmbedder:
    """Load one configured text or multimodal model and embed pages/queries."""

    def __init__(self, model_key: str, config_path: str | Path, model_root: str | Path):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["models"][model_key]
        self.modality = config["modality"]
        local_path = Path(model_root) / model_key
        source = str(local_path if local_path.exists() else config["repo"])
        self.model = SentenceTransformer(
            source,
            revision=None if local_path.exists() else config.get("revision"),
            trust_remote_code="remote-code" in config.get("backend", ""),
        )

    def encode_pages(self, pages: Iterable[dict], batch_size: int = 8) -> np.ndarray:
        """Embed page text or page images and L2-normalize for cosine retrieval."""
        values = []
        for page in pages:
            values.append(page["text"] if self.modality == "text" else Image.open(page["image_path"]).convert("RGB"))
        return np.asarray(
            self.model.encode(values, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
            dtype="float32",
        )

    def encode_queries(self, questions: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed textual questions in the shared retrieval space."""
        return np.asarray(
            self.model.encode(questions, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
            dtype="float32",
        )


def build_index(pages: list[dict], vectors: np.ndarray, output_dir: str | Path) -> None:
    """Persist a FAISS cosine index plus its ordered page metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Paper step: each page is one candidate unit; normalized inner product is cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output / "pages.faiss"))
    with (output / "pages.jsonl").open("w", encoding="utf-8") as stream:
        for page in pages:
            stream.write(json.dumps({"file": str(page["file"]), "page": int(page["page"])}, ensure_ascii=False) + "\n")


def search_index(index_dir: str | Path, queries: np.ndarray, top_k: int) -> list[list[dict]]:
    """Return ranked document/page candidates and similarity scores."""
    root = Path(index_dir)
    index = faiss.read_index(str(root / "pages.faiss"))
    metadata = read_jsonl(root / "pages.jsonl")
    scores, indices = index.search(queries, top_k)
    results = []
    for row_scores, row_indices in zip(scores, indices):
        # Paper step: preserve global similarity order when producing Top-K candidates.
        results.append([
            {**metadata[int(idx)], "score": float(score), "rank": rank}
            for rank, (idx, score) in enumerate(zip(row_indices, row_scores), start=1)
            if idx >= 0
        ])
    return results
