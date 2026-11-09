"""Citation quality metrics for attributed RAG answers."""

from __future__ import annotations

from rag_benchmark.schemas import PredictionRecord


def evaluate_citation_quality(
    prediction: PredictionRecord,
    *,
    gold_chunk_ids: set[str] | None = None,
    gold_sentence_refs: set[tuple[str, int]] | None = None,
) -> PredictionRecord:
    # Never retain semantic scores from an older/group-level evaluation.
    previous = {k: v for k, v in prediction.metrics.items() if k not in {
        "citation_precision", "citation_f1", "citation_support_rate",
        "unsupported_citation_rate", "contradicted_citation_rate", "cited_claim_coverage"}}
    metrics = {
        **previous,
        **citation_quality_metrics(
            prediction,
            gold_chunk_ids=gold_chunk_ids,
            gold_sentence_refs=gold_sentence_refs,
        ),
    }
    return prediction.model_copy(update={"metrics": metrics})


def citation_quality_metrics(
    prediction: PredictionRecord,
    *,
    gold_chunk_ids: set[str] | None = None,
    gold_sentence_refs: set[tuple[str, int]] | None = None,
) -> dict[str, float]:
    citations = prediction.citations or [
        citation for claim in prediction.claim_checks for citation in claim.citations
    ]
    citation_count = len(citations)
    claim_count = len(prediction.claim_checks)
    claims_with_citation = sum(1 for claim in prediction.claim_checks if claim.citations)
    invalid_count = int(
        prediction.metadata.get("citation_generation", {}).get("invalid_citation_count", 0)
    )

    individually_judged = [c for c in citations if c.metadata.get("judgment_scope") == "citation"]
    supported = sum(1 for citation in individually_judged if citation.support_status == "supported")
    unsupported = sum(1 for citation in individually_judged if citation.support_status == "unsupported")
    contradicted = sum(1 for citation in individually_judged if citation.support_status == "contradicted")
    judged = supported + unsupported + contradicted + sum(
        1 for citation in individually_judged if citation.support_status == "insufficient"
    )

    metrics = {
        "citation_count": float(citation_count),
        "claims_with_citation": float(claims_with_citation),
        "citation_judged_count": float(judged),
        "invalid_citation_count": float(invalid_count),
        "invalid_citation_rate": safe_divide(invalid_count, citation_count + invalid_count),
    }

    if claim_count:
        metrics["cited_claim_coverage"] = safe_divide(claims_with_citation, claim_count)
    if citation_count:
        metrics["citation_judgment_coverage"] = judged / citation_count
    if judged == citation_count and judged > 0:
        metrics.update({"citation_support_rate": supported / judged,
            "citation_precision": supported / judged,
            "unsupported_citation_rate": unsupported / judged,
            "contradicted_citation_rate": contradicted / judged})

    if gold_chunk_ids is not None:
        cited_gold_chunks = {citation.chunk_id for citation in citations if citation.chunk_id in gold_chunk_ids}
        gold_hits = sum(1 for citation in citations if citation.chunk_id in gold_chunk_ids)
        metrics.update(
            {
                "citation_gold_hit_rate": safe_divide(gold_hits, citation_count),
                "citation_recall": safe_divide(len(cited_gold_chunks), len(gold_chunk_ids)),
            }
        )
        # Gold evidence coverage is not the recall counterpart of semantic support.

    if gold_sentence_refs is not None:
        sentence_hits = sum(
            1
            for citation in citations
            if citation.sentence_id is not None
            and (citation.chunk_id, citation.sentence_id) in gold_sentence_refs
        )
        metrics["citation_sentence_gold_hit_rate"] = safe_divide(sentence_hits, citation_count)

    return metrics


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
