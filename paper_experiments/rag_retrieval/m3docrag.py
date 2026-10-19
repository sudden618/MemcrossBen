"""
Native M3DocRAG ColPali multi-vector retrieval.

Paper correspondence:
- RAG Baselines / M3DocRAG Retrieval
- Page and query token embeddings are compared with late-interaction MaxSim.
"""

from __future__ import annotations

import pickle
from pathlib import Path


def _load_model(model_path: str, device: str):
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor

    model = ColPali.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
    return model, ColPaliProcessor.from_pretrained(model_path)


def build(pages: list[dict], index_dir: str, model_path: str, device: str, batch_size: int) -> None:
    """Encode every page as a variable-length token matrix and persist it on CPU."""
    import torch
    from PIL import Image

    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_pages = [page for page in pages if page.get("image_path")]
    model, processor = _load_model(model_path, device)
    embeddings = []
    for start in range(0, len(image_pages), batch_size):
        images = [Image.open(page["image_path"]).convert("RGB") for page in image_pages[start : start + batch_size]]
        with torch.no_grad():
            embeddings.extend(model(**processor.process_images(images).to(device)).cpu())
    torch.save(embeddings, output / "page_embeddings.pt")
    with (output / "metadata.pkl").open("wb") as stream:
        pickle.dump(image_pages, stream)


def search(questions: list[str], index_dir: str, model_path: str, device: str, top_k: int) -> list[dict]:
    """Rank pages using ColPali processor.score_multi_vector MaxSim scores."""
    import torch

    root = Path(index_dir)
    pages = pickle.loads((root / "metadata.pkl").read_bytes())
    page_embeddings = torch.load(root / "page_embeddings.pt", map_location=device, weights_only=False)
    model, processor = _load_model(model_path, device)
    output = []
    for question in questions:
        with torch.no_grad():
            query = model(**processor.process_queries([question]).to(device))
            scores = processor.score_multi_vector(query, page_embeddings).squeeze(0)
        values, indices = torch.topk(scores, min(top_k, len(pages)))
        recall = [
            {**pages[int(idx)], "score": float(score), "rank": rank}
            for rank, (idx, score) in enumerate(zip(indices.cpu(), values.cpu()), start=1)
        ]
        output.append({"recall": recall})
    return output
