"""Evaluation availability is independent of generation and verification success."""
from rag_benchmark.schemas import PredictionRecord


def attach_evaluation_coverage(prediction: PredictionRecord) -> PredictionRecord:
    claims = prediction.claim_checks
    citations = prediction.citations
    judged = sum(c.support_status in {"supported", "unsupported", "contradicted", "insufficient"}
                 and c.metadata.get("judgment_scope") == "citation" for c in citations)
    coverage = {
        "schema_version": "evaluation-coverage-v1",
        "declared_claim_count": len(claims),
        "claim_source": "generator_declared" if claims else "none",
        "claim_support_scope": "declared_claims_only" if claims else "not_evaluated",
        "full_answer_claim_coverage": "unknown",
        "independent_claim_extraction": "not_run",
        "citation_count": len(citations), "citation_judged_count": judged,
        "citation_support_status": "evaluated" if citations and judged == len(citations) else "not_fully_evaluated",
    }
    if prediction.metadata.get("task_revision") and claims:
        coverage.update(claim_source="task_revision_replacements", claim_support_scope="replacement_claims_only")
    metrics = dict(prediction.metrics)
    if not claims:
        for key in ("claim_support_rate", "contradiction_rate", "unsupported_claim_rate", "evidence_consistency_score"):
            metrics.pop(key, None)
    return prediction.model_copy(update={"metrics": metrics,
        "metadata": {**prediction.metadata, "evaluation_coverage": coverage}})
