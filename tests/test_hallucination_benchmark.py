from rag_benchmark.evaluation.hallucination import (
    calibrate_hallucination_threshold,
    hallucination_benchmark_metrics,
    parse_batched_hallucination_output,
)


def test_detector_spans_are_grounded_and_invalid_quotes_are_counted() -> None:
    detections = parse_batched_hallucination_output(
        {
            "items": [
                {
                    "response_id": "r1",
                    "has_hallucination": True,
                    "spans": [{"quote": "invented fact"}, {"quote": "missing"}],
                }
            ]
        },
        responses={"r1": "An invented fact appears."},
    )

    assert detections[0].spans == ((3, 16),)
    assert detections[0].invalid_span_quotes == 1
    assert detections[0].hallucination_probability == 1.0


def test_hallucination_metrics_keep_response_and_span_dimensions_separate() -> None:
    metrics = hallucination_benchmark_metrics(
        [
            {
                "gold_has_hallucination": True,
                "predicted_has_hallucination": True,
                "gold_spans": [(2, 5)],
                "predicted_spans": [(3, 5)],
                "predicted_span_quotes": ["xx"],
                "invalid_span_quotes": 0,
            },
            {
                "gold_has_hallucination": False,
                "predicted_has_hallucination": False,
                "gold_spans": [],
                "predicted_spans": [],
                "predicted_span_quotes": [],
                "invalid_span_quotes": 0,
            },
        ]
    )

    assert metrics["hallucination_response_f1"] == 1.0
    assert metrics["hallucination_macro_f1"] == 1.0
    assert metrics["hallucination_response_accuracy"] == 1.0
    assert metrics["hallucination_span_precision"] == 1.0
    assert metrics["hallucination_span_recall"] == 2 / 3


def test_calibration_selects_threshold_without_using_eval_labels() -> None:
    rows = [
        {"gold_has_hallucination": True, "hallucination_probability": 0.9, "predicted_spans": [], "predicted_span_quotes": []},
        {"gold_has_hallucination": True, "hallucination_probability": 0.7, "predicted_spans": [], "predicted_span_quotes": []},
        {"gold_has_hallucination": False, "hallucination_probability": 0.6, "predicted_spans": [], "predicted_span_quotes": []},
        {"gold_has_hallucination": False, "hallucination_probability": 0.1, "predicted_spans": [], "predicted_span_quotes": []},
    ]

    calibration = calibrate_hallucination_threshold(rows, minimum_per_class_recall=0.5)

    assert 0.6 < calibration["threshold"] <= 0.7
    assert calibration["metrics"]["hallucination_macro_f1"] == 1.0
