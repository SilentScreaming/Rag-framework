"""Observable evaluation outcomes and explicitly non-causal diagnostic hypotheses."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from rag_benchmark.evaluation.answers import assess_task

JsonObject = dict[str, Any]
DIAGNOSTIC_PROTOCOL = "diagnostics-v1"
ERROR_LABELS = {"correct_under_task_rule", "incorrect_unattributed", "not_evaluated",
                "annotated_evidence_not_retrieved", "annotated_evidence_rank_low",
                "reference_support_incomplete", "reference_contradiction", "reference_evidence_conflict",
                "citation_contribution_incomplete", "invalid_citation", "task_score_decreased",
                "correct_to_incorrect", "verification_evaluation_disagreement"}
HALLUCINATION_LABELS = set()  # No heuristic label is a human hallucination ground truth.
RETRIEVAL_FAILURE_LABELS = set()


@dataclass(frozen=True)
class DiagnosticRecord:
    example_id: str
    pipeline: str
    primary_error: str
    secondary_errors: list[str] = field(default_factory=list)
    needs_human_review: bool = True
    signals: JsonObject = field(default_factory=dict)
    rationale: str = ""
    hypotheses: list[dict] = field(default_factory=list)
    protocol_version: str = DIAGNOSTIC_PROTOCOL

    def to_json_dict(self):
        return asdict(self)


def diagnose_prediction(prediction, *, retrieval_trace=None):
    signals = collect_signals(prediction, retrieval_trace=retrieval_trace)
    labels = infer_error_labels(signals)
    primary = choose_primary_error(labels, signals)
    hypotheses = []
    for observed, candidate in (("annotated_evidence_not_retrieved", "possible_retrieval_miss"),
                                ("annotated_evidence_rank_low", "possible_ranking_issue"),
                                ("verification_evaluation_disagreement", "possible_verification_miss")):
        if observed in labels:
            hypotheses.append({"label": candidate, "supporting_signals": [observed],
                               "missing_evidence": ["controlled_counterfactual_or_manual_validation"],
                               "evidence_level": "heuristic", "rule_version": DIAGNOSTIC_PROTOCOL})
    secondary = [x for x in labels if x != primary]
    # Task correctness and evidence support are measured independently.
    review = bool(secondary) or primary != "correct_under_task_rule" or not signals["independent_evaluation_complete"]
    return DiagnosticRecord(prediction.example_id, prediction.pipeline, primary, secondary, review,
                            signals, build_rationale(primary, secondary, signals), hypotheses)


def collect_signals(prediction, *, retrieval_trace=None):
    task = prediction.metadata.get("task_assessment") or assess_task(prediction)
    metrics = prediction.metrics
    correct = task.get("task_correct")
    # Short-answer exact match is a task score, not a factuality judgment.
    if correct is None and prediction.task_track == "short_answer" and isinstance(metrics.get("em"), (int, float)):
        correct = metrics["em"] == 1.0
    sidecar = prediction.metadata.get("independent_evaluation", {})
    independent = sidecar.get("schema_version") == "evaluation-v1"
    gold_count = len(retrieval_trace.gold_chunk_ids) if retrieval_trace else None
    any_hit = next((bool(metrics[k]) for k in ("any_gold_hit@10", "any_gold_hit@5") if metrics.get(k) is not None), None)
    if retrieval_trace and gold_count:
        retrieved = {r.chunk_id for r in retrieval_trace.retrieved_chunks}
        any_hit = bool(retrieved & set(retrieval_trace.gold_chunk_ids))
    intervention = sidecar.get("intervention") or {}
    before, after = intervention.get("before", {}), intervention.get("after", {})
    decrease = None
    if before.get("score") is not None and after.get("score") is not None and before.get("score_metric") == after.get("score_metric"):
        decrease = after["score"] < before["score"] - 1e-9
    extraction = sidecar.get("extraction", {})
    return {
        "answer_correct": correct, "task_assessment": task,
        "any_gold_hit": any_hit, "gold_count": gold_count,
        "mrr_at_10": metrics.get("mrr@10"), "retrieval_count": len(prediction.retrieved_chunks),
        "claim_support_rate": metrics.get("reference_supported_claim_rate") if independent else None,
        "contradiction_rate": metrics.get("reference_contradicted_claim_rate") if independent else None,
        "conflicting_rate": metrics.get("reference_conflicting_claim_rate") if independent else None,
        "citation_support_rate": metrics.get("citation_contribution_precision") if independent else None,
        "invalid_citation_count": prediction.metadata.get("citation_generation", {}).get("invalid_citation_count"),
        "verification_passed": bool(prediction.verification and prediction.verification.status == "passed"),
        "task_score_decreased": decrease,
        "correct_to_incorrect": before.get("task_correct") is True and after.get("task_correct") is False,
        "independent_evaluation_complete": independent and extraction.get("status") == "completed"
            and bool(extraction.get("claims")) and metrics.get("reference_supported_claim_rate") is not None,
        "fact_coverage": sidecar.get("full_answer_fact_coverage", "unknown"),
        "f1_threshold_used": False,
    }


def infer_error_labels(signals):
    labels = []
    if signals["gold_count"] and signals["any_gold_hit"] is False:
        labels.append("annotated_evidence_not_retrieved")
    if signals["any_gold_hit"] is True and signals["mrr_at_10"] is not None and signals["mrr_at_10"] < 0.5:
        labels.append("annotated_evidence_rank_low")
    if signals["claim_support_rate"] is not None and signals["claim_support_rate"] < 1:
        labels.append("reference_support_incomplete")
    if signals["contradiction_rate"] is not None and signals["contradiction_rate"] > 0:
        labels.append("reference_contradiction")
    if signals["conflicting_rate"] is not None and signals["conflicting_rate"] > 0:
        labels.append("reference_evidence_conflict")
    if signals["citation_support_rate"] is not None and signals["citation_support_rate"] < 1:
        labels.append("citation_contribution_incomplete")
    if signals["invalid_citation_count"]:
        labels.append("invalid_citation")
    if signals["task_score_decreased"]:
        labels.append("task_score_decreased")
    if signals["correct_to_incorrect"]:
        labels.append("correct_to_incorrect")
    if signals["verification_passed"] and signals["answer_correct"] is False:
        labels.append("verification_evaluation_disagreement")
    labels.append("correct_under_task_rule" if signals["answer_correct"] is True else
                  "incorrect_unattributed" if signals["answer_correct"] is False else "not_evaluated")
    return dedupe(labels)


def choose_primary_error(labels, signals):
    # Presentation priority, explicitly not a unique root-cause inference.
    return labels[0]


def build_rationale(primary, secondary, signals):
    return (f"observed={primary}; task_correct={signals['answer_correct']}; "
            f"fact_coverage={signals['fact_coverage']}; causes_unconfirmed")


def summarize_diagnostics(records):
    counts = Counter(r.primary_error for r in records)
    all_counts = Counter(x for r in records for x in [r.primary_error, *r.secondary_errors])
    return {"protocol_version": DIAGNOSTIC_PROTOCOL, "num_records": len(records),
            "needs_human_review": sum(r.needs_human_review for r in records),
            "primary_error_counts": dict(sorted(counts.items())),
            "all_error_label_counts": dict(sorted(all_counts.items())),
            "interpretation": "observations_and_hypotheses_not_causal_or_human_ground_truth",
            **diagnostic_rates(records)}


def diagnostic_rates(records):
    total = len(records)
    rate = lambda condition: sum(condition(r) for r in records) / total if total else None
    return {"needs_human_review_rate": rate(lambda r: r.needs_human_review),
            "answer_evaluation_available_rate": rate(lambda r: r.signals.get("answer_correct") is not None),
            "claim_support_evaluation_available_rate": rate(lambda r: r.signals.get("claim_support_rate") is not None),
            "citation_support_evaluation_available_rate": rate(lambda r: r.signals.get("citation_support_rate") is not None),
            "hallucination_case_rate": None, "retrieval_failure_rate": None,
            "incorrect_unattributed_rate": rate(lambda r: "incorrect_unattributed" in [r.primary_error, *r.secondary_errors])}


def dedupe(labels):
    if not set(labels) <= ERROR_LABELS:
        raise ValueError("unknown diagnostic label")
    return list(dict.fromkeys(labels))
