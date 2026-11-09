"""Track-specific metrics, grouping, and uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


from rag_benchmark.evaluation.metric_catalog import definitions_for_track
from rag_benchmark.data import Annotation, Example, Qrel
from rag_benchmark.evaluation.basic import (
    best_qa_prf,
    ndcg_at_k,
    normalize_answer,
    r_precision,
    set_prf,
)
from rag_benchmark.schemas import PredictionRecord
from rag_benchmark.evaluation.answers import ANSWER_PROTOCOL, grouped_answer_metrics


JsonObject = dict[str, Any]


TRACK_PRIMARY_METRICS = {
    "short_answer": (
        "em",
        "f1",
        "recall@10",
        "all_gold_hit@10",
        "kilt_em",
        "hotpot_joint_f1",
    ),
    "long_answer_citation": (
        "gold_answer_recall",
        "qampari_recall@5",
        "qampari_precision",
        "citation_recall",
        "citation_precision",
        "reference_supported_claim_rate",
        "citation_contribution_precision",
        "supported_cited_claim_coverage",
    ),
    "verification_hallucination": (
        "verification_macro_f1",
        "hallucination_span_f1",
        "raf",
        "deflection_f1",
        "deflection_fpr",
        "net_verification_gain",
        "harmful_revision_rate",
    ),
}


def evaluate_prediction_track_metrics(
    prediction: PredictionRecord,
    *,
    example: Example,
    qrels: list[Qrel],
    annotations: list[Annotation] | None = None,
    context_condition: str | None = None,
    context_set=None,
) -> PredictionRecord:
    """Attach only metrics justified by the available supervision."""

    if prediction.generator.get("schema_failure"):
        safe = {k: v for k, v in prediction.metrics.items() if k.startswith(
            ("recall@", "mrr@", "any_gold", "all_gold", "context_", "retrieval_"))
            or k.endswith(("_ms", "_tokens"))}
        from rag_benchmark.evaluation.task_registry import evaluate_task
        task, cells = evaluate_task(prediction, example=example, qrels=qrels, annotations=annotations, context_set=context_set)
        return prediction.model_copy(update={"metrics": {**safe, "generation_failed": 1.0},
            "metadata": {**prediction.metadata, "answer_metric_protocol": ANSWER_PROTOCOL,
                "task_assessment": task, "task_metric_results": cells}})
    metrics = {k: v for k, v in prediction.metrics.items() if k != "citation_f1"}
    if prediction.task_track == "long_answer_citation":
        metrics = {k: v for k, v in metrics.items() if not k.startswith(("gold_answer_", "asqa_", "qampari_", "list_"))}
    if prediction.task_track == "short_answer":
        metrics.update(short_answer_metrics(prediction, example=example, qrels=qrels))
    elif prediction.task_track == "long_answer_citation":
        metrics.update(long_answer_track_metrics(prediction, example=example))
    else:
        metrics.update(
            verification_track_metrics(
                prediction,
                example=example,
                qrels=qrels,
                annotations=annotations or [],
            )
        )
    metadata = dict(prediction.metadata)
    from rag_benchmark.evaluation.task_registry import evaluate_task, TASK_METRICS
    task, cells = evaluate_task(prediction, example=example, qrels=qrels, annotations=annotations, context_set=context_set)
    managed = {key for keys in TASK_METRICS.values() for key in keys} | {"accuracy"}
    metrics = {k: v for k, v in metrics.items() if k not in managed and not k.startswith(
        ("abstract_", "rationale_", "attribution_", "kilt_", "hotpot_joint_"))}
    metrics.update({key: cell["value"] for key, cell in cells.items() if cell["status"] == "computed"})
    metadata.update(task_assessment=task, task_metric_results=cells)
    metadata["answer_metric_protocol"] = ANSWER_PROTOCOL
    if prediction.task_track == "long_answer_citation":
        keys = ("gold_answer_recall", "qampari_precision", "qampari_recall@5")
        if not any(key in metrics for key in keys):
            metadata["answer_metric_status"] = "missing_answer_grouping"
    metadata["metric_strata"] = {
        "answer_type": str(example.metadata.get("answer_type") or example.task_type),
        "context_condition": context_condition or str(example.metadata.get("context_condition") or "default"),
    }
    return prediction.model_copy(update={"metrics": metrics, "metadata": metadata})


def short_answer_metrics(
    prediction: PredictionRecord,
    *,
    example: Example,
    qrels: list[Qrel],
) -> JsonObject:
    retrieved = [row.chunk_id for row in prediction.retrieved_chunks]
    gold_ids = {qrel.chunk_id for qrel in qrels}
    metrics: JsonObject = {"qrel_coverage": float(bool(gold_ids))}
    r_prec = r_precision(retrieved, gold_ids)
    if r_prec is not None:
        metrics["r_precision"] = r_prec
    relevance = {qrel.chunk_id: qrel.relevance for qrel in qrels}
    ndcg = ndcg_at_k(retrieved, relevance, k=10)
    if ndcg is not None:
        metrics["ndcg@10"] = ndcg
    if "kilt" in example.dataset.casefold() and r_prec is not None:
        metrics["kilt_em"] = float(float(prediction.metrics.get("em", 0.0)) == 1.0 and r_prec == 1.0)

    gold_sentences = gold_sentence_set(qrels)
    predicted_sentences = {
        (citation.chunk_id, citation.sentence_id)
        for citation in prediction.citations
        if citation.sentence_id is not None
    }
    if gold_sentences:
        support = set_prf(predicted_sentences, gold_sentences)
        metrics.update(
            {
                "supporting_fact_em": support["em"],
                "supporting_fact_precision": support["precision"],
                "supporting_fact_recall": support["recall"],
                "supporting_fact_f1": support["f1"],
            }
        )
        answer = best_qa_prf(prediction.answer, example.answers)
        joint_precision = answer["precision"] * support["precision"]
        joint_recall = answer["recall"] * support["recall"]
        metrics["hotpot_joint_em"] = float(
            float(prediction.metrics.get("em", 0.0)) == 1.0 and support["em"] == 1.0
        )
        metrics["hotpot_joint_f1"] = (
            2 * joint_precision * joint_recall / (joint_precision + joint_recall)
            if joint_precision + joint_recall
            else 0.0
        )
    return metrics


def long_answer_track_metrics(prediction: PredictionRecord, *, example: Example) -> JsonObject:
    dataset = example.dataset.casefold()
    metrics: JsonObject = {"answer_word_count": float(len(prediction.answer.split()))}
    kind = "asqa" if "asqa" in dataset else "qampari" if "qampari" in dataset else None
    if kind:
        try:
            from rag_benchmark.generation.answer_format import list_items_for_prediction
            items = list_items_for_prediction(prediction) if kind == 'qampari' and prediction.answer_bundle and prediction.answer_bundle.list_items is not None else None
            metrics.update(grouped_answer_metrics(prediction.answer, example.answers, kind=kind,
                                                  metadata=example.metadata, list_items=items))
        except ValueError as exc:
            if str(exc) != "missing_answer_grouping":
                raise
    return metrics


def verification_track_metrics(
    prediction: PredictionRecord,
    *,
    example: Example,
    qrels: list[Qrel],
    annotations: list[Annotation],
    answer_behavior=None,
) -> JsonObject:
    metrics: JsonObject = {}
    dataset = example.dataset.casefold()
    if "scifact" in dataset:
        predicted_label = normalize_verification_label(prediction.answer)
        gold_labels = {normalize_verification_label(str(row.get("text", ""))) for row in example.answers}
        label_correct = float(predicted_label in gold_labels)
        if gold_labels:
            metrics["verification_label_accuracy"] = label_correct
        predicted_abstracts = {row.chunk_id for row in prediction.retrieved_chunks}
        gold_abstracts = {qrel.chunk_id for qrel in qrels}
        abstract = set_prf(
            predicted_abstracts if label_correct else set(),
            gold_abstracts,
        )
        metrics.update({f"abstract_{key}": abstract[key] for key in ("precision", "recall", "f1")})
        gold_rationales = {
            (annotation.target_id, int(sentence_id))
            for annotation in annotations
            if annotation.target_type in {"sentence", "evidence"} and annotation.target_id is not None
            for sentence_id in annotation.metadata.get("sentence_ids", [])
        }
        if not gold_rationales:
            gold_rationales = gold_sentence_set(qrels)
        if gold_rationales:
            predicted_rationales = {
                (citation.chunk_id, citation.sentence_id)
                for citation in prediction.citations
                if citation.sentence_id is not None
            }
            rationale = set_prf(predicted_rationales, gold_rationales)
            metrics.update(
                {f"rationale_sentence_{key}": rationale[key] for key in ("precision", "recall", "f1")}
            )
            metrics.update({f"rationale_source_id_{key}": rationale[key] for key in ("precision", "recall", "f1")})
            if all(isinstance(c.metadata.get("fully_visible"), bool) for c in prediction.citations):
                full = {(c.chunk_id, c.sentence_id) for c in prediction.citations
                        if c.sentence_id is not None and c.metadata["fully_visible"]
                        and not c.metadata.get("budget_truncated", False)}
                visible = set_prf(full, gold_rationales)
                metrics.update({f"rationale_visible_complete_{key}": visible[key] for key in ("precision", "recall", "f1")})

    expected = str(example.metadata.get("expected_behavior", "")).casefold()
    if expected in {"answer", "abstain"} and not (
            'rgb' in dataset and expected == 'abstain'
            and example.metadata.get('insufficiency_analysis_eligible') is not True):
        predicted_abstain = is_abstention(prediction, answer_behavior=answer_behavior)
        if predicted_abstain is not None:
            metrics["eligibility_accuracy"] = float(predicted_abstain == (expected == "abstain"))

    if "garage" in dataset:
        cited = {citation.chunk_id for citation in prediction.citations}
        gold_attribution = {
            qrel.chunk_id for qrel in qrels if qrel.metadata.get("gold_cited", True)
        }
        attribution = set_prf(cited, gold_attribution)
        metrics.update(
            {f"attribution_{key}": attribution[key] for key in ("precision", "recall", "f1")}
        )
        requirement = numeric(prediction.metrics.get("requirement_satisfaction"))
        if requirement is not None:
            metrics["uraf"] = requirement
            metrics["raf"] = requirement * attribution["precision"] * float(
                float(prediction.metrics.get("claim_support_rate", 0.0)) == 1.0
            )
    return metrics


def aggregate_track_metrics(
    track: str,
    predictions: list[PredictionRecord],
    aggregate: JsonObject,
    *,
    examples_by_id: dict[str, Example],
) -> tuple[JsonObject, JsonObject]:
    """Add ratio-of-counts metrics and return grouped stratum summaries."""

    derived: JsonObject = {}
    derived.update(verification_intervention_metrics(predictions, track=track))
    if track == "verification_hallucination":
        labels = [
            (
                normalize_verification_label(str(examples_by_id[row.example_id].answers[0]["text"])),
                normalize_verification_label(row.answer),
            )
            for row in predictions
            if row.example_id in examples_by_id and examples_by_id[row.example_id].answers
            and str(examples_by_id[row.example_id].answers[0].get("type", "")) == "verification_label"
        ]
        if labels and not any(p.generator.get("schema_failure") for p in predictions):
            derived["verification_macro_f1"] = multiclass_macro_f1(labels)
        deflection = deflection_metrics(predictions, examples_by_id=examples_by_id)
        derived.update(deflection)

    reliability = reliability_metrics(predictions)
    derived.update(reliability)
    from rag_benchmark.evaluation.statistics import metric_value
    for key in ("hallucination_span_f1", "hallucination_response_f1"):
        if any(key in p.metrics for p in predictions):
            aggregate.pop(key, None)
            value = metric_value(predictions, key, examples_by_id=examples_by_id)
            if value is not None:
                derived[key] = value
    aggregate.update(derived)
    return aggregate, grouped_strata(predictions)


def verification_intervention_metrics(predictions, *, track="verification_hallucination"):
    """Report correctness rates and score changes with separate denominators."""
    rows = [p.metrics for p in predictions]
    # Fall back to exact match only for tracks other than long-answer citation.
    pre = "pre_verification_task_correct"
    post = "post_verification_task_correct"
    if not any(pre in r for r in rows) and track != "long_answer_citation":
        pre, post = "pre_verification_em", "post_verification_em"
    pairs = [(r[pre], r[post]) for r in rows if r.get(pre) in (0, 1) and r.get(post) in (0, 1)]
    result = {}
    if pairs:
        wrong = [(a, b) for a, b in pairs if a == 0]
        correct = [(a, b) for a, b in pairs if a == 1]
        if wrong:
            result["correction_success_rate"] = sum(b == 1 for _, b in wrong) / len(wrong)
        if correct:
            result["harmful_revision_rate"] = sum(b == 0 for _, b in correct) / len(correct)
            result["preserve_correct_rate"] = sum(b == 1 for _, b in correct) / len(correct)
            result["over_correction_rate"] = result["harmful_revision_rate"]
        result["verification_initial_incorrect_count"] = float(len(wrong))
        result["verification_initial_correct_count"] = float(len(correct))
        result["verification_correctness_coverage"] = len(pairs) / len(rows)
    score_pre, score_post = "pre_verification_task_score", "post_verification_task_score"
    named_scores = any(score_pre in r for r in rows)
    comparable = True
    if named_scores:
        names = [(p.metadata.get("pre_verification_task_score_metric"),
                  p.metadata.get("post_verification_task_score_metric")) for p in predictions]
        comparable = bool(names) and all(isinstance(a, str) and a and a == b for a, b in names)
        comparable = comparable and len({a for a, _ in names}) == 1
    else:
        score_pre, score_post = "pre_verification_f1", "post_verification_f1"
    delta_pairs = [(r[score_pre], r[score_post]) for r in rows
                   if comparable and numeric(r.get(score_pre)) is not None and numeric(r.get(score_post)) is not None]
    if delta_pairs:
        deltas = [b - a for a, b in delta_pairs]
        result.update(net_verification_gain=sum(deltas) / len(deltas),
                      task_score_increase_rate=sum(x > 1e-9 for x in deltas) / len(deltas),
                      task_score_decrease_rate=sum(x < -1e-9 for x in deltas) / len(deltas))
    grounding = [r["post_verification_claim_support_rate"] - r["pre_verification_claim_support_rate"]
                 for r in rows if numeric(r.get("pre_verification_claim_support_rate")) is not None
                 and numeric(r.get("post_verification_claim_support_rate")) is not None]
    if grounding:
        result["grounding_gain"] = sum(grounding) / len(grounding)
    if any(p.verification is not None for p in predictions):
        decisions = [is_abstention(p) for p in predictions]
        known = [v for v in decisions if v is not None]
        if known:
            result["verification_abstention_rate"] = sum(known) / len(known)
        result["verification_abstention_n_evaluated"] = len(known)
        result["verification_abstention_n_expected"] = len(decisions)
    return result


def deflection_metrics(
    predictions: list[PredictionRecord],
    *,
    examples_by_id: dict[str, Example],
) -> JsonObject:
    pairs = []
    for prediction in predictions:
        example = examples_by_id.get(prediction.example_id)
        if example is None:
            continue
        assessment = prediction.metadata.get("task_assessment")
        if assessment is not None and assessment.get("status") != "computed":
            return {}
        expected = str(assessment.get("expected_behavior", "") if assessment is not None
                       else example.metadata.get("expected_behavior", "")).casefold()
        if expected in {"answer", "abstain"}:
            decision = is_abstention(prediction)
            if decision is None:
                return {}  # unresolved behavior is not an answered or rejected instance
            pairs.append((expected == "abstain", decision))
        else:
            return {}  # no silent complete-case population
    if not pairs:
        return {}
    binary = binary_classification_metrics(pairs)
    negatives = sum(not gold for gold, _ in pairs)
    false_positives = sum((not gold) and predicted for gold, predicted in pairs)
    return {
        "deflection_precision": binary["precision"],
        "deflection_recall": binary["recall"],
        "deflection_f1": binary["f1"],
        "deflection_tpr": binary["recall"],
        "deflection_fpr": false_positives / negatives if negatives else None,
    }


def hallucination_detection_metrics(
    *,
    gold_response_labels: list[bool],
    predicted_response_labels: list[bool],
    gold_spans: list[list[tuple[int, int]]] | None = None,
    predicted_spans: list[list[tuple[int, int]]] | None = None,
) -> JsonObject:
    """RAGTruth-compatible response and character-span detection metrics."""

    if len(gold_response_labels) != len(predicted_response_labels):
        raise ValueError("response label vectors must have equal length")
    response = binary_classification_metrics(list(zip(gold_response_labels, predicted_response_labels)))
    metrics = {f"hallucination_response_{key}": value for key, value in response.items()}
    metrics.update(hallucination_response_tp=sum(g and p for g, p in zip(gold_response_labels, predicted_response_labels)),
                   hallucination_response_fp=sum(not g and p for g, p in zip(gold_response_labels, predicted_response_labels)),
                   hallucination_response_fn=sum(g and not p for g, p in zip(gold_response_labels, predicted_response_labels)))
    if gold_spans is not None or predicted_spans is not None:
        if gold_spans is None or predicted_spans is None or len(gold_spans) != len(predicted_spans):
            raise ValueError("gold_spans and predicted_spans must have equal length")
        gold_chars = set()
        predicted_chars = set()
        offset = 0
        for gold_rows, predicted_rows in zip(gold_spans, predicted_spans):
            for start, end in gold_rows:
                gold_chars.update(range(offset + start, offset + end))
            for start, end in predicted_rows:
                predicted_chars.update(range(offset + start, offset + end))
            offset += max(
                [end for _, end in [*gold_rows, *predicted_rows]] or [0]
            ) + 1
        span = set_prf(predicted_chars, gold_chars)
        metrics.update(
            {f"hallucination_span_{key}": span[key] for key in ("precision", "recall", "f1")}
        )
        metrics.update(hallucination_span_tp=len(gold_chars & predicted_chars),
                       hallucination_span_fp=len(predicted_chars - gold_chars),
                       hallucination_span_fn=len(gold_chars - predicted_chars))
    return metrics


def group_track_metrics(track: str, aggregate: JsonObject) -> JsonObject:
    """Keep dimensions separate; intentionally does not construct a blended score."""

    groups: dict[str, JsonObject] = defaultdict(dict)
    definition_keys = {definition.key: definition for definition in definitions_for_track(track)}
    for key, value in aggregate.items():
        definition = definition_keys.get(key)
        group = definition.group if definition else infer_group(key)
        groups[group][key] = value
    return {
        "track": track,
        "primary_metrics": {
            key: aggregate[key]
            for key in TRACK_PRIMARY_METRICS.get(track, ())
            if key in aggregate
        },
        "groups": dict(groups),
        "blended_score": None,
    }


def bootstrap_confidence_intervals(rows, *, keys, samples=1000, seed=42, predictions=None, examples_by_id=None):
    from rag_benchmark.evaluation.statistics import confidence_intervals
    if predictions is None:
        from rag_benchmark.evaluation.identity import EvaluationIdentity
        predictions = [PredictionRecord(example_id=str(i), pipeline="metrics", question="metric-only", answer="", metrics=row,
                       metadata=EvaluationIdentity("metric-only", "synthetic-v1", "synthetic", seed).metadata())
                       for i, row in enumerate(rows)]
    return confidence_intervals(predictions, keys=keys, samples=samples, seed=seed, examples_by_id=examples_by_id)


def paired_metric_comparisons(baseline, candidate, *, keys, samples=1000, seed=42, examples_by_id=None):
    from rag_benchmark.evaluation.statistics import paired_comparisons
    return paired_comparisons(baseline, candidate, keys=keys, samples=samples, seed=seed, examples_by_id=examples_by_id)


def grouped_strata(predictions: list[PredictionRecord]) -> JsonObject:
    buckets: dict[str, dict[str, list[JsonObject]]] = defaultdict(lambda: defaultdict(list))
    for prediction in predictions:
        strata = prediction.metadata.get("metric_strata", {})
        for dimension in ("answer_type", "context_condition"):
            buckets[dimension][str(strata.get(dimension, "unknown"))].append(prediction.metrics)
    result: JsonObject = {}
    for dimension, groups in buckets.items():
        result[dimension] = {}
        for name, rows in groups.items():
            summary = {}
            for key in ("em", "f1", "claim_support_rate", "citation_precision", "citation_recall"):
                values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
                if values:
                    summary[key] = sum(values) / len(values)
            summary["n"] = len(rows)
            result[dimension][name] = summary
    return result


def reliability_metrics(predictions: list[PredictionRecord]) -> JsonObject:
    if not predictions:
        return {}
    invalid_rows = []
    api_rows = []
    failure_rows = []
    schema_failure_rows = []
    for prediction in predictions:
        runtime = prediction.metadata.get("citation_generation_runtime", {})
        if runtime:
            invalid_rows.append(bool(runtime.get("used_fallback") or runtime.get("parse_error")))
        if prediction.generator.get("provider") == "openai":
            api_rows.append(int(prediction.generator.get("attempt_count", 1) or 1))
        if "api_failure" in prediction.generator:
            failure_rows.append(bool(prediction.generator["api_failure"]))
        if runtime or "schema_failure" in prediction.generator:
            schema_failure_rows.append(bool(prediction.generator.get("schema_failure", False)))
    result: JsonObject = {}
    if invalid_rows:
        result["invalid_output_rate"] = sum(invalid_rows) / len(invalid_rows)
    if api_rows:
        result["retry_rate"] = sum(attempts > 1 for attempts in api_rows) / len(api_rows)
    if failure_rows:
        result["api_failure_rate"] = sum(failure_rows) / len(failure_rows)
    if schema_failure_rows:
        result["schema_failure_rate"] = sum(schema_failure_rows) / len(schema_failure_rows)
    return result


def gold_sentence_set(qrels: Iterable[Qrel]) -> set[tuple[str, int]]:
    refs = set()
    for qrel in qrels:
        sentence_ids = list(qrel.metadata.get("sentence_ids", []))
        if qrel.metadata.get("sentence_id") is not None:
            sentence_ids.append(qrel.metadata["sentence_id"])
        refs.update((qrel.chunk_id, int(sentence_id)) for sentence_id in sentence_ids)
    return refs


def binary_classification_metrics(pairs: list[tuple[bool, bool]]) -> JsonObject:
    tp = sum(gold and predicted for gold, predicted in pairs)
    fp = sum((not gold) and predicted for gold, predicted in pairs)
    fn = sum(gold and (not predicted) for gold, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": harmonic_mean(precision, recall)}


def multiclass_macro_f1(pairs: list[tuple[str, str]], *, labels=("SUPPORTS", "REFUTES", "INSUFFICIENT")) -> float:
    scores = []
    for label in labels:
        binary = [(gold == label, predicted == label) for gold, predicted in pairs]
        scores.append(binary_classification_metrics(binary)["f1"])
    return sum(scores) / len(scores) if scores else 0.0


def normalize_verification_label(value: str) -> str:
    normalized = normalize_answer(value.replace("_", " ").replace("-", " ")).replace(" ", "_").upper()
    aliases = {
        "SUPPORT": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "ENTAILMENT": "SUPPORTS",
        "REFUTE": "REFUTES",
        "REFUTED": "REFUTES",
        "CONTRADICTION": "REFUTES",
        "CONTRADICT": "REFUTES",
        "NOT_ENOUGH_INFO": "INSUFFICIENT",
        "NEI": "INSUFFICIENT",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT",
    }
    return aliases.get(normalized, normalized)


def is_abstention(prediction: PredictionRecord, *, answer_behavior=None) -> bool | None:
    if answer_behavior is None:
        answer_behavior = prediction.metadata.get("task_assessment", {}).get("answer_behavior")
    if answer_behavior is not None:
        if answer_behavior.get("status") == "computed":
            label = answer_behavior.get("label")
            if label == "abstained":
                return True
            if label in {"answered", "partial"}:
                return False
            return None  # uncertainty is retained rather than credited as a pure refusal
        from rag_benchmark.generation.answer_format import is_explicit_abstention
        return True if is_explicit_abstention(prediction.answer) else None
    if prediction.answer_bundle is not None and prediction.answer_bundle.abstained:
        return True
    answer = normalize_answer(prediction.answer)
    return answer in {"unknown", "i dont know", "insufficient evidence", "cannot answer"}


def harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def infer_group(key: str) -> str:
    if key.startswith(("recall@", "mrr@", "any_gold", "all_gold", "r_precision", "ndcg")):
        return "retrieval"
    if key.startswith(("citation", "claim", "attribution", "unsupported", "contradiction")):
        return "evidence_and_citation"
    if key.startswith(("verification", "correction", "harmful", "preserve", "net_", "grounding", "over_")):
        return "verification_safety"
    if key.endswith("_ms") or "tokens" in key or "cost" in key:
        return "efficiency"
    return "other"
