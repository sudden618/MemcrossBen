"""
Native MDocAgent dual-route page retrieval.

Paper correspondence:
- RAG Baselines / MDocAgent Retrieval
- Text and image pages are indexed separately.
- The two ranked lists are interleaved with duplicate removal.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .common import interleave


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def build(pages: list[dict], index_dir: str, text_model: str, image_model: str, batch_size: int) -> None:
    """Build independent dense text and image page stores used by MDocAgent."""
    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    text_pages = [page for page in pages if page.get("text")]
    image_pages = [page for page in pages if page.get("image_path")]
    text_encoder = SentenceTransformer(text_model, trust_remote_code=True)
    image_encoder = SentenceTransformer(image_model, trust_remote_code=True)
    text_vectors = _normalize(text_encoder.encode([page["text"] for page in text_pages], batch_size=batch_size))
    image_vectors = _normalize(image_encoder.encode([page["image_path"] for page in image_pages], batch_size=batch_size))
    np.save(output / "text.npy", text_vectors)
    np.save(output / "image.npy", image_vectors)
    with (output / "metadata.pkl").open("wb") as stream:
        pickle.dump({"text": text_pages, "image": image_pages}, stream)


def search(
    questions: list[str], index_dir: str, text_model: str, image_model: str, top_k: int, candidate_k: int
) -> list[dict]:
    """Retrieve both modalities and apply the baseline's alternating fusion."""
    root = Path(index_dir)
    metadata = pickle.loads((root / "metadata.pkl").read_bytes())
    text_vectors, image_vectors = np.load(root / "text.npy"), np.load(root / "image.npy")
    text_encoder = SentenceTransformer(text_model, trust_remote_code=True)
    image_encoder = SentenceTransformer(image_model, trust_remote_code=True)
    tq = _normalize(text_encoder.encode(questions))
    iq = _normalize(image_encoder.encode(questions, prompt="Find an image that matches the given text."))
    output = []
    for text_query, image_query in zip(tq, iq):
        text_ids = np.argsort(-(text_query @ text_vectors.T))[:candidate_k]
        image_ids = np.argsort(-(image_query @ image_vectors.T))[:candidate_k]
        text_hits = [{**metadata["text"][int(idx)], "route": "text"} for idx in text_ids]
        image_hits = [{**metadata["image"][int(idx)], "route": "image"} for idx in image_ids]
        output.append({
            "text_recall": text_hits,
            "image_recall": image_hits,
            "recall": interleave(text_hits, image_hits, top_k),
        })
    return output
