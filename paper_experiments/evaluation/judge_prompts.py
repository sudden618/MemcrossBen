"""
Public prompts for Coverage, Accuracy, Attribution, and Alignment judging.

All four dimensions use integer scores from 1 (worst) to 5 (best).
Overall is the arithmetic mean of the four dimension scores.
"""

COVERAGE_RUBRIC = """Coverage (1-5): judge whether the answer covers every answerable
required evidence document. 1=no required document covered; 3=some covered; 5=all covered."""

ACCURACY_RUBRIC = """Accuracy (1-5): judge factual consistency with the reference answer
and supplied evidence pages. 1=mostly false/unsupported; 3=mixed; 5=fully correct."""

ATTRIBUTION_RUBRIC = """Attribution (1-5): judge whether claims are assigned to the correct
documents/pages and whether citations support those claims. 1=misattributed; 3=partly
traceable; 5=all material claims correctly attributed."""

ALIGNMENT_RUBRIC = """Alignment (1-5): judge whether the response directly answers the
question and follows requested scope/language. 1=not aligned; 3=partly; 5=fully aligned."""

JUDGE_SYSTEM_PROMPT = f"""You are an impartial multi-document QA evaluator.
Use only the provided question, reference answer, evidence, candidate answer, and citations.

{COVERAGE_RUBRIC}
{ACCURACY_RUBRIC}
{ATTRIBUTION_RUBRIC}
{ALIGNMENT_RUBRIC}

Return only JSON:
{{"coverage": 1, "accuracy": 1, "attribution": 1, "alignment": 1,
  "rationale": "brief explanation"}}
All scores must be integers from 1 to 5."""


def build_judge_messages(
    question: str,
    reference_answer: str,
    evidence: list[dict],
    candidate_answer: str,
    citations: list[dict],
) -> list[dict]:
    """Expose every Judge input field explicitly in a reproducible order."""
    import json

    payload = {
        "question": question,
        "reference_answer": reference_answer,
        "evidence_pages": evidence,
        "candidate_answer": candidate_answer,
        "candidate_citations": citations,
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
