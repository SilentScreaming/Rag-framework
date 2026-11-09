"""Canonical metric contract for the three benchmark tracks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


MetricTier = Literal["core", "module", "diagnostic", "audit", "secondary"]
MetricDirection = Literal["higher", "lower", "neutral"]


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    track: str
    group: str
    tier: MetricTier
    direction: MetricDirection
    description: str
    applicability: str = "all"
    unit: str = "ratio"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _metric(
    key: str,
    track: str,
    group: str,
    tier: MetricTier,
    direction: MetricDirection,
    description: str,
    applicability: str = "all",
    unit: str = "ratio",
) -> MetricDefinition:
    return MetricDefinition(key, track, group, tier, direction, description, applicability, unit)


METRIC_CATALOG: tuple[MetricDefinition, ...] = (
    # Track 1: short-answer QA.
    _metric("em", "short_answer", "answer_quality", "core", "higher", "Normalized exact match."),
    _metric("f1", "short_answer", "answer_quality", "core", "higher", "Token overlap F1."),
    _metric("r_precision", "short_answer", "retrieval", "module", "higher", "Relevant items among the top R results."),
    _metric("recall@5", "short_answer", "retrieval", "module", "higher", "Gold evidence recall in the top 5."),
    _metric("recall@10", "short_answer", "retrieval", "core", "higher", "Gold evidence recall in the top 10."),
    _metric("mrr@10", "short_answer", "retrieval", "module", "higher", "Reciprocal rank of the first gold item in the top 10."),
    _metric("ndcg@10", "short_answer", "retrieval", "secondary", "higher", "Graded relevance ranking quality at 10.", "graded_qrels"),
    _metric("supporting_fact_em", "short_answer", "evidence", "module", "higher", "Exact match of predicted and gold supporting sentences.", "hotpotqa"),
    _metric("supporting_fact_f1", "short_answer", "evidence", "module", "higher", "Set F1 over supporting sentences.", "hotpotqa"),
    _metric("hotpot_joint_em", "short_answer", "end_to_end", "core", "higher", "Answer EM and supporting-fact EM are both correct.", "hotpotqa"),
    _metric("hotpot_joint_f1", "short_answer", "end_to_end", "core", "higher", "Official-style joint answer/supporting-fact F1.", "hotpotqa"),
    _metric("kilt_em", "short_answer", "end_to_end", "core", "higher", "Answer EM gated by complete KILT provenance retrieval.", "kilt"),
    _metric("any_gold_hit@10", "short_answer", "retrieval", "diagnostic", "higher", "At least one gold evidence item is retrieved."),
    _metric("all_gold_hit@10", "short_answer", "retrieval", "core", "higher", "The complete gold evidence set is retrieved.", "multi_evidence"),
    _metric("qrel_coverage", "short_answer", "data_quality", "audit", "higher", "Fraction of examples with resolvable qrels."),
    _metric("gold_context_gap_f1", "short_answer", "diagnostic", "diagnostic", "lower", "Oracle-context F1 minus retrieved-context F1.", "paired_oracle_run"),
    # Track 2: long answers and citations.
    _metric("gold_answer_recall", "long_answer_citation", "content_correctness", "core", "higher", "ASQA exact-match recall over annotated short answers.", "asqa"),
    _metric("qampari_recall@5", "long_answer_citation", "content_correctness", "core", "higher", "QAMPARI recall capped at five correct list items.", "qampari"),
    _metric("qampari_precision", "long_answer_citation", "content_correctness", "core", "higher", "Precision of generated QAMPARI list items.", "qampari"),
    _metric("citation_recall", "long_answer_citation", "citation_quality", "core", "higher", "Gold evidence coverage by citations."),
    _metric("citation_precision", "long_answer_citation", "citation_quality", "core", "higher", "Fraction of judged citations that support their claims."),
    _metric("claim_support_rate", "long_answer_citation", "evidence", "core", "higher", "Fraction of answer claims supported by evidence."),
    _metric("cited_claim_coverage", "long_answer_citation", "citation_quality", "module", "higher", "Fraction of claims carrying at least one citation."),
    _metric("invalid_citation_rate", "long_answer_citation", "citation_quality", "module", "lower", "Fraction of citations that are invalid or unparseable."),
    _metric("attribution_macro_f1", "long_answer_citation", "judge_calibration", "diagnostic", "higher", "Macro-F1 of the attribution judge against human labels.", "calibration_labels"),
    _metric("mauve", "long_answer_citation", "text_quality", "secondary", "higher", "Distribution-level fluency similarity.", "reference_distribution"),
    _metric("answer_word_count", "long_answer_citation", "behavior", "diagnostic", "neutral", "Generated answer length in whitespace-delimited words.", unit="count"),
    _metric("claim_count", "long_answer_citation", "behavior", "diagnostic", "neutral", "Number of extracted factual claims.", unit="count"),
    _metric("citation_count", "long_answer_citation", "behavior", "diagnostic", "neutral", "Number of parsed citations.", unit="count"),
    # Track 3: verification, hallucination and deflection.
    _metric("verification_macro_f1", "verification_hallucination", "claim_verification", "core", "higher", "Macro-F1 over SUPPORTS, REFUTES and INSUFFICIENT labels.", "labeled_verification"),
    _metric("abstract_precision", "verification_hallucination", "claim_verification", "module", "higher", "SciFact abstract retrieval-and-label precision.", "scifact"),
    _metric("abstract_recall", "verification_hallucination", "claim_verification", "module", "higher", "SciFact abstract retrieval-and-label recall.", "scifact"),
    _metric("abstract_f1", "verification_hallucination", "claim_verification", "module", "higher", "SciFact abstract retrieval-and-label F1.", "scifact"),
    _metric("rationale_sentence_precision", "verification_hallucination", "rationale", "module", "higher", "Precision of predicted rationale sentences.", "scifact"),
    _metric("rationale_sentence_recall", "verification_hallucination", "rationale", "module", "higher", "Recall of gold rationale sentences.", "scifact"),
    _metric("rationale_sentence_f1", "verification_hallucination", "rationale", "module", "higher", "F1 over rationale sentences.", "scifact"),
    *tuple(_metric(f"{prefix}_{key}", "verification_hallucination", "rationale", "module", "higher",
        description, "scifact") for prefix, description in (
            ("rationale_source_id", "Source sentence ID match; does not imply the full sentence was visible."),
            ("rationale_visible_complete", "Full visible rationale sentence match; unavailable without visibility provenance."))
        for key in ("precision", "recall", "f1")),
    _metric("hallucination_response_precision", "verification_hallucination", "hallucination_detection", "module", "higher", "Response-level hallucination precision.", "ragtruth"),
    _metric("hallucination_response_recall", "verification_hallucination", "hallucination_detection", "module", "higher", "Response-level hallucination recall.", "ragtruth"),
    _metric("hallucination_response_f1", "verification_hallucination", "hallucination_detection", "module", "higher", "Response-level hallucination F1.", "ragtruth"),
    _metric("hallucination_span_precision", "verification_hallucination", "hallucination_detection", "module", "higher", "Character-span hallucination precision.", "ragtruth"),
    _metric("hallucination_span_recall", "verification_hallucination", "hallucination_detection", "module", "higher", "Character-span hallucination recall.", "ragtruth"),
    _metric("hallucination_span_f1", "verification_hallucination", "hallucination_detection", "core", "higher", "Character-span hallucination F1.", "ragtruth"),
    _metric("eligibility_accuracy", "verification_hallucination", "grounding", "module", "higher", "Accuracy of answer-versus-deflect eligibility decisions.", "garage"),
    _metric("answer_abstain_accuracy", "verification_hallucination", "grounding", "module", "higher", "Answer/abstain decision accuracy; not paper Eligibility.", "garage"),
    _metric("garage_eligibility", "verification_hallucination", "grounding", "module", "higher", "Sufficiently addresses the request relative to the human reference.", "garage_requirement_judge"),
    _metric("uraf", "verification_hallucination", "grounding", "module", "higher", "Whole response supported by relevant grounding (or sentences not requiring attribution).", "garage_requirement_judge"),
    _metric("raf", "verification_hallucination", "grounding", "core", "higher", "Joint per-answer eligibility and relevant-grounding support.", "garage_requirement_judge"),
    _metric("deflection_precision", "verification_hallucination", "deflection", "module", "higher", "Precision of abstention/deflection decisions.", "deflection_labels"),
    _metric("deflection_recall", "verification_hallucination", "deflection", "module", "higher", "Recall of required abstentions.", "deflection_labels"),
    _metric("deflection_f1", "verification_hallucination", "deflection", "core", "higher", "F1 of required abstentions.", "deflection_labels"),
    _metric("deflection_tpr", "verification_hallucination", "deflection", "module", "higher", "True-positive rate for required deflections.", "deflection_labels"),
    _metric("deflection_fpr", "verification_hallucination", "deflection", "core", "lower", "False-positive deflection rate on answerable examples.", "deflection_labels"),
    _metric("attribution_precision", "verification_hallucination", "attribution", "module", "higher", "Precision of selected attributed evidence.", "attribution_labels"),
    _metric("attribution_recall", "verification_hallucination", "attribution", "module", "higher", "Recall of gold attributed evidence.", "attribution_labels"),
    _metric("attribution_f1", "verification_hallucination", "attribution", "module", "higher", "F1 of attributed evidence selection.", "attribution_labels"),
    _metric("correction_success_rate", "verification_hallucination", "verification_safety", "module", "higher", "Initially incorrect examples corrected by verification."),
    _metric("harmful_revision_rate", "verification_hallucination", "verification_safety", "core", "lower", "Initially correct examples made incorrect by verification."),
    _metric("preserve_correct_rate", "verification_hallucination", "verification_safety", "module", "higher", "Initially correct examples remaining correct."),
    _metric("net_verification_gain", "verification_hallucination", "verification_safety", "core", "higher", "Mean paired task-score change under the recorded task protocol; token F1 is the fallback when task scores are absent."),
    _metric("grounding_gain", "verification_hallucination", "verification_safety", "module", "higher", "Final claim support minus initial claim support."),
    _metric("over_correction_rate", "verification_hallucination", "verification_safety", "diagnostic", "lower", "Rate of unnecessary harmful revisions."),
    _metric("verification_abstention_rate", "verification_hallucination", "verification_safety", "diagnostic", "neutral", "Fraction of final answers that abstain."),
    # Shared efficiency and reliability metrics are repeated into every track report.
    *tuple(
        _metric(key, "shared", group, tier, direction, description, unit=unit)
        for key, group, tier, direction, description, unit in (
            ("latency_p50_ms", "efficiency", "module", "lower", "Median end-to-end latency.", "ms"),
            ("latency_p95_ms", "efficiency", "module", "lower", "95th percentile end-to-end latency.", "ms"),
            ("avg_input_tokens", "efficiency", "module", "lower", "Mean input tokens per example.", "tokens"),
            ("avg_output_tokens", "efficiency", "module", "lower", "Mean output tokens per example.", "tokens"),
            ("total_input_tokens", "efficiency", "module", "lower", "Total input tokens.", "tokens"),
            ("total_output_tokens", "efficiency", "module", "lower", "Total output tokens.", "tokens"),
            ("avg_estimated_cost", "efficiency", "module", "lower", "Mean estimated cost per example.", "currency"),
            ("total_estimated_cost", "efficiency", "module", "lower", "Total estimated cost.", "currency"),
            ("api_failure_rate", "reliability", "diagnostic", "lower", "Fraction of examples ending in an API failure.", "ratio"),
            ("retry_rate", "reliability", "diagnostic", "lower", "Fraction of examples requiring at least one API retry.", "ratio"),
            ("invalid_output_rate", "reliability", "diagnostic", "lower", "Fraction of outputs encountering a schema/parse error, including recovered retries.", "ratio"),
            ("schema_failure_rate", "reliability", "diagnostic", "lower", "Fraction of outputs that still fail schema validation after all retries and use the audited abstention fallback.", "ratio"),
        )
    ),
)


METRIC_CATALOG += tuple(
    _metric(key, "shared", group, "diagnostic", direction, description)
    for key, group, direction, description in (
        ("asqa_str_em_alce", "content_correctness", "higher", "Grouped ALCE-compatible answer occurrence; not full factuality."),
        ("asqa_group_recall_strict", "content_correctness", "higher", "Grouped answer occurrence with token boundaries."),
        ("qampari_f1_top5_alce", "content_correctness", "higher", "ALCE-compatible grouped list F1 with recall capped at five."),
        ("qampari_unique_entity_precision", "content_correctness", "higher", "One-to-one matching of unique predicted entities."),
        ("qampari_unique_entity_recall", "content_correctness", "higher", "One-to-one matching recall over gold entity groups."),
        ("citation_contribution_precision", "citation_quality", "higher", "Judge-relative standalone or joint citation contribution."),
        ("supported_cited_claim_coverage", "citation_quality", "higher", "Independent factual claims supported by their cited evidence."),
        ("citation_gold_evidence_recall", "citation_quality", "higher", "Coverage of annotated evidence, not semantic recall."),
        ("reference_supported_claim_rate", "evidence", "higher", "Independent answer facts supported within the frozen reference scope."),
        ("reference_insufficient_claim_rate", "evidence", "neutral", "Reference evidence does not establish the claim; not proven false."),
        ("reference_contradicted_claim_rate", "evidence", "lower", "Reference-relative contradictory evidence."),
        ("reference_conflicting_claim_rate", "evidence", "neutral", "Conflicting reference evidence."),
        ("task_score_increase_rate", "verification_safety", "higher", "Paired task score increased; distinct from wrong-to-correct."),
        ("task_score_decrease_rate", "verification_safety", "lower", "Paired task score decreased; distinct from correct-to-wrong."),
    )
)

CATALOG_BY_KEY = {definition.key: definition for definition in METRIC_CATALOG}


def definitions_for_track(track: str) -> tuple[MetricDefinition, ...]:
    return tuple(row for row in METRIC_CATALOG if row.track in {track, "shared"})
