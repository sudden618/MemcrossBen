"""
Native ViDoRAG multimodal retrieval with GMM candidate filtering.

Paper correspondence:
- RAG Baselines / ViDoRAG Retrieval
- Dense text retrieval and ColQwen late-interaction image retrieval
- Two-component GMM filtering followed by interleaved multimodal fusion
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

from .common import interleave


def _colqwen(model_path: str, device: str):
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model = ColQwen2.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
    return model, ColQwen2Processor.from_pretrained(model_path)


def filter_gmm(hits: list[dict], max_items: int, min_items: int) -> list[dict]:
    """Select the higher-score GMM component, bounded by the baseline min/max sizes."""
    if len(hits) <= min_items:
        return hits
    scores = np.asarray([hit["score"] for hit in hits], dtype=float).reshape(-1, 1)
    labels = GaussianMixture(n_components=2, n_init=1, random_state=0).fit_predict(scores)
    high_label = max(set(labels), key=lambda label: scores[labels == label].max())
    high = [hit for hit, label in zip(hits, labels) if label == high_label][:max_items]
    if len(high) < min_items:
        selected = {id(hit) for hit in high}
        high.extend(hit for hit in hits if id(hit) not in selected and len(high) < min_items)
    return high


def build(
    pages: list[dict], index_dir: str, text_model: str, image_model: str, device: str, batch_size: int
) -> None:
    """Build ViDoRAG text vectors and ColQwen page-token embeddings."""
    import torch
    from PIL import Image, ImageOps

    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    text_pages = [page for page in pages if page.get("text")]
    image_pages = [page for page in pages if page.get("image_path")]
    text_encoder = SentenceTransformer(text_model, trust_remote_code=True)
    text_vectors = text_encoder.encode(
        [page["text"] for page in text_pages], batch_size=batch_size, normalize_embeddings=True
    )
    model, processor = _colqwen(image_model, device)
    image_vectors = []
    for start in range(0, len(image_pages), batch_size):
        images = [
            ImageOps.pad(Image.open(page["image_path"]).convert("RGB"), (1191, 1684))
            for page in image_pages[start : start + batch_size]
        ]
        with torch.no_grad():
            image_vectors.extend(model(**processor.process_images(images).to(device)).cpu())
    np.save(output / "text.npy", np.asarray(text_vectors, dtype="float32"))
    torch.save(image_vectors, output / "image.pt")
    with (output / "metadata.pkl").open("wb") as stream:
        pickle.dump({"text": text_pages, "image": image_pages}, stream)


def search(
    questions: list[str],
    index_dir: str,
    text_model: str,
    image_model: str,
    device: str,
    top_k: int,
    candidate_k: int,
    min_items: int,
) -> list[dict]:
    """Run both retrieval routes, GMM-filter each, then interleave candidates."""
    import torch

    root = Path(index_dir)
    metadata = pickle.loads((root / "metadata.pkl").read_bytes())
    text_vectors = np.load(root / "text.npy")
    image_vectors = torch.load(root / "image.pt", map_location=device, weights_only=False)
    text_encoder = SentenceTransformer(text_model, trust_remote_code=True)
    text_queries = text_encoder.encode(questions, normalize_embeddings=True)
    image_encoder, processor = _colqwen(image_model, device)
    output = []
    for question, text_query in zip(questions, text_queries):
        text_scores = text_query @ text_vectors.T
        text_ids = np.argsort(-text_scores)[:candidate_k]
        text_hits = [{**metadata["text"][int(idx)], "score": float(text_scores[idx]), "route": "text"} for idx in text_ids]
        with torch.no_grad():
            query = image_encoder(**processor.process_queries([question]).to(device))
            image_scores = processor.score_multi_vector(query, image_vectors).squeeze(0)
        values, indices = torch.topk(image_scores, min(candidate_k, len(metadata["image"])))
        image_hits = [
            {**metadata["image"][int(idx)], "score": float(score), "route": "image"}
            for idx, score in zip(indices.cpu(), values.cpu())
        ]
        text_hits = filter_gmm(text_hits, top_k, min_items)
        image_hits = filter_gmm(image_hits, top_k, min_items)
        output.append({
            "text_recall": text_hits,
            "image_recall": image_hits,
            "recall": interleave(text_hits, image_hits, top_k),
        })
    return output
