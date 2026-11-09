"""RAGTruth-compatible response and character-span hallucination evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_benchmark.evaluation.tracks import binary_classification_metrics


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class HallucinationDetection:
    response_id: str
    has_hallucination: bool
    hallucination_probability: float
    spans: tuple[tuple[int, int], ...]
    rationale: str = ""
    invalid_span_quotes: int = 0


def parse_batched_hallucination_output(
    payload: JsonObject,
    *,
    responses: dict[str, str],
    threshold: float = 0.5,
) -> list[HallucinationDetection]:
    """Validate detector JSON and ground every predicted span in the response text."""

    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ValueError("hallucination detector output must contain an items list")
    by_id = {
        str(row.get("response_id")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("response_id", "")) in responses
    }
    detections = []
    for response_id, response_text in responses.items():
        row = by_id.get(response_id)
        if row is None:
            raise ValueError(f"detector output omitted response_id={response_id}")
        raw_spans = row.get("spans", [])
        if not isinstance(raw_spans, list):
            raise ValueError(f"spans must be a list for response_id={response_id}")
        spans = []
        invalid = 0
        search_start = 0
        for raw_span in raw_spans:
            quote = str(raw_span.get("quote", "") if isinstance(raw_span, dict) else raw_span).strip()
            if not quote:
                invalid += 1
                continue
            start = response_text.find(quote, search_start)
            if start < 0:
                start = response_text.casefold().find(quote.casefold(), search_start)
            if start < 0:
                invalid += 1
                continue
            end = start + len(quote)
            spans.append((start, end))
            search_start = end
        default_probability = 1.0 if bool(row.get("has_hallucination", bool(spans))) else 0.0
        probability = float(row.get("hallucination_probability", default_probability))
        probability = min(1.0, max(0.0, probability))
        predicted = probability >= threshold
        detections.append(
            HallucinationDetection(
                response_id=response_id,
                has_hallucination=predicted,
                hallucination_probability=probability,
                spans=tuple(merge_spans(spans)),
                rationale=str(row.get("rationale", "")),
                invalid_span_quotes=invalid,
            )
        )
    return detections


def calibrate_hallucination_threshold(
    rows: list[JsonObject],
    *,
    minimum_per_class_recall: float = 0.65,
) -> JsonObject:
    """Choose a deterministic calibration-only threshold and return its metrics."""

    probabilities = sorted(
        {min(1.0, max(0.0, float(row["hallucination_probability"]))) for row in rows}
    )
    thresholds = {0.0, 0.5, 1.0}
    thresholds.update(probabilities)
    thresholds.update(
        (left + right) / 2 for left, right in zip(probabilities, probabilities[1:])
    )
    candidates = []
    for threshold in sorted(thresholds):
        evaluated = apply_hallucination_threshold(rows, threshold=threshold)
        metrics = hallucination_benchmark_metrics(evaluated)
        feasible = (
            metrics["hallucination_response_recall"] >= minimum_per_class_recall
            and metrics["grounded_response_recall"] >= minimum_per_class_recall
        )
        candidates.append((feasible, metrics["hallucination_macro_f1"], metrics["hallucination_response_accuracy"], -threshold, threshold, metrics))
    _feasible, _macro_f1, _accuracy, _negative_threshold, threshold, metrics = max(candidates)
    return {
        "threshold": threshold,
        "minimum_per_class_recall": minimum_per_class_recall,
        "constraints_satisfied": bool(
            metrics["hallucination_response_recall"] >= minimum_per_class_recall
            and metrics["grounded_response_recall"] >= minimum_per_class_recall
        ),
        "metrics": metrics,
    }


def apply_hallucination_threshold(
    rows: list[JsonObject],
    *,
    threshold: float,
) -> list[JsonObject]:
    evaluated = []
    for row in rows:
        predicted = float(row.get("hallucination_probability", 0.0)) >= threshold
        evaluated.append(
            {
                **row,
                "predicted_has_hallucination": predicted,
                "predicted_spans": row.get("predicted_spans", []) if predicted else [],
                "predicted_span_quotes": row.get("predicted_span_quotes", []) if predicted else [],
            }
        )
    return evaluated


def hallucination_benchmark_metrics(rows: list[JsonObject]) -> JsonObject:
    pairs = [
        (bool(row["gold_has_hallucination"]), bool(row["predicted_has_hallucination"]))
        for row in rows
    ]
    response = binary_classification_metrics(pairs)
    grounded_pairs = [(not gold, not predicted) for gold, predicted in pairs]
    grounded = binary_classification_metrics(grounded_pairs)
    span_counts = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        gold_chars = span_characters(row.get("gold_spans", []))
        predicted_chars = span_characters(row.get("predicted_spans", []))
        span_counts["tp"] += len(gold_chars & predicted_chars)
        span_counts["fp"] += len(predicted_chars - gold_chars)
        span_counts["fn"] += len(gold_chars - predicted_chars)
    span_precision = safe_ratio(span_counts["tp"], span_counts["tp"] + span_counts["fp"])
    span_recall = safe_ratio(span_counts["tp"], span_counts["tp"] + span_counts["fn"])
    span_f1 = harmonic_mean(span_precision, span_recall)
    return {
        "examples": len(rows),
        "hallucinated_examples": sum(gold for gold, _predicted in pairs),
        "grounded_examples": sum(not gold for gold, _predicted in pairs),
        "hallucination_response_accuracy": safe_ratio(
            sum(gold == predicted for gold, predicted in pairs), len(pairs)
        ),
        "hallucination_response_precision": response["precision"],
        "hallucination_response_recall": response["recall"],
        "hallucination_response_f1": response["f1"],
        "grounded_response_recall": grounded["recall"],
        "hallucination_macro_f1": (response["f1"] + grounded["f1"]) / 2,
        "hallucination_span_precision": span_precision,
        "hallucination_span_recall": span_recall,
        "hallucination_span_f1": span_f1,
        "invalid_span_quote_rate": safe_ratio(
            sum(int(row.get("invalid_span_quotes", 0)) for row in rows),
            sum(
                len(row.get("predicted_span_quotes", []))
                + int(row.get("invalid_span_quotes", 0))
                for row in rows
            ),
        ),
    }


def span_characters(spans: list[list[int]] | list[tuple[int, int]]) -> set[int]:
    result = set()
    for start, end in spans:
        result.update(range(max(0, int(start)), max(0, int(end))))
    return result


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
