"""
Prompt construction for DirectQA and retrieval-augmented generation.

Paper correspondence:
- Generation Experiments / DirectQA
- Generation Experiments / NaiveRAG and RAG Baselines
- Generation Experiments / Answer and Citation Output
"""

DIRECT_SYSTEM = """You are a question-answering assistant.
Answer the question directly. Output only the final answer and use the question's language."""

RAG_SYSTEM = """You are a multi-document question-answering assistant.
Use only the supplied sources. Include citations in the form [document:page].
If multiple documents answer the question, keep their claims distinguishable.
If the sources are insufficient, answer exactly: No relevant information found.
Output only the final answer and citations, using the question's language."""

METHOD_INSTRUCTIONS = {
    "naive_rag": "Synthesize an answer directly from the ranked Top-K evidence.",
    "hipporag": "Integrate the graph-retrieved evidence while preserving source identity.",
    "m3docrag": "Integrate multimodal multi-document evidence while preserving source identity.",
    "mdocagent": "Synthesize the evidence selected by specialized document agents.",
    "vidorag": "Answer from the retrieved document pages and cite every supporting page.",
}


def build_messages(method: str, question: str, context: str = "") -> list[dict]:
    """Construct the reproducible message sequence for one generation method."""
    if method == "directqa":
        return [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
    instruction = METHOD_INSTRUCTIONS.get(method, METHOD_INSTRUCTIONS["naive_rag"])
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {
            "role": "user",
            "content": f"Method instruction: {instruction}\n\nSources:\n{context}\n\nQuestion: {question}",
        },
    ]
