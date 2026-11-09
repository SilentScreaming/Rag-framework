from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rag_benchmark.evaluation.tracks import (
    aggregate_track_metrics,
    evaluate_prediction_track_metrics,
    hallucination_detection_metrics,
    paired_metric_comparisons,
    reliability_metrics,
)
from rag_benchmark.data import Example, Qrel
from rag_benchmark.reporting import write_unified_metrics_report
from rag_benchmark.schemas import PredictionRecord


class TrackMetricsTest(unittest.TestCase):
    def test_reliability_metrics_separate_recovered_parse_errors_from_terminal_failure(self) -> None:
        recovered = PredictionRecord(
            example_id="recovered",
            pipeline="rag",
            question="Q?",
            answer="A",
            generator={"provider": "openai", "attempt_count": 2, "schema_failure": False},
            metadata={"citation_generation_runtime": {"parse_error": "retry", "used_fallback": False}},
        )
        failed = PredictionRecord(
            example_id="failed",
            pipeline="rag",
            question="Q?",
            answer="I don't know.",
            generator={"provider": "openai", "attempt_count": 3, "schema_failure": True},
            metadata={"citation_generation_runtime": {"parse_error": "failed", "used_fallback": True}},
        )

        metrics = reliability_metrics([recovered, failed])

        self.assertEqual(metrics["invalid_output_rate"], 1.0)
        self.assertEqual(metrics["schema_failure_rate"], 0.5)
        self.assertEqual(metrics["retry_rate"], 1.0)

    def test_kilt_and_hotpot_metrics_use_answer_and_complete_evidence(self) -> None:
        example = Example(
            example_id="e1",
            dataset="kilt_hotpotqa",
            split="dev",
            task_type="multi_hop_short_answer",
            question="Who?",
            answers=[{"text": "Ada"}],
            gold_evidence=[],
            metadata={},
        )
        qrels = [
            Qrel("e1", "a", 3, "gold", {"sentence_id": 0}),
            Qrel("e1", "b", 3, "gold", {"sentence_id": 1}),
        ]
        prediction = PredictionRecord(
            example_id="e1",
            pipeline="rag",
            question="Who?",
            answer="Ada",
            metrics={"em": 1.0, "f1": 1.0},
            retrieved_chunks=[
                {"chunk_id": "a", "rank": 1, "score": 1.0},
                {"chunk_id": "b", "rank": 2, "score": 0.9},
            ],
            citations=[
                {"chunk_id": "a", "sentence_id": 0},
                {"chunk_id": "b", "sentence_id": 1},
            ],
        )

        scored = evaluate_prediction_track_metrics(prediction, example=example, qrels=qrels)

        self.assertEqual(scored.metrics["r_precision"], 1.0)
        self.assertEqual(scored.metrics["kilt_em"], 1.0)
        self.assertEqual(scored.metrics["supporting_fact_f1"], 1.0)
        self.assertEqual(scored.metrics["hotpot_joint_f1"], 1.0)

    def test_verification_rates_use_conditional_denominators(self) -> None:
        predictions = [
            self.verification_prediction("a", 0, 1),
            self.verification_prediction("b", 0, 0),
            self.verification_prediction("c", 1, 1),
            self.verification_prediction("d", 1, 0),
        ]
        aggregate, _ = aggregate_track_metrics(
            "verification_hallucination",
            predictions,
            {},
            examples_by_id={},
        )

        self.assertEqual(aggregate["correction_success_rate"], 0.5)
        self.assertEqual(aggregate["harmful_revision_rate"], 0.5)
        self.assertEqual(aggregate["preserve_correct_rate"], 0.5)
        self.assertEqual(aggregate["net_verification_gain"], 0.0)

    def test_long_answer_verification_uses_paired_f1_harm_and_gain(self) -> None:
        predictions = [
            self.long_verification_prediction("a", 0.4, 0.6),
            self.long_verification_prediction("b", 0.7, 0.5),
            self.long_verification_prediction("c", 0.8, 0.8),
        ]

        aggregate, _ = aggregate_track_metrics(
            "long_answer_citation",
            predictions,
            {},
            examples_by_id={},
        )

        self.assertAlmostEqual(aggregate["net_verification_gain"], 0.0)
        self.assertAlmostEqual(aggregate["task_score_increase_rate"], 1 / 3)
        self.assertAlmostEqual(aggregate["task_score_decrease_rate"], 1 / 3)
        self.assertNotIn("harmful_revision_rate", aggregate)

    def test_verification_track_does_not_substitute_f1_for_empty_correctness_denominator(self) -> None:
        predictions = [
            self.verification_prediction_with_f1("a", 0.4, 0.4),
            self.verification_prediction_with_f1("b", 0.6, 0.6),
        ]

        aggregate, _ = aggregate_track_metrics(
            "verification_hallucination",
            predictions,
            {},
            examples_by_id={},
        )

        self.assertNotIn("harmful_revision_rate", aggregate)
        self.assertNotIn("preserve_correct_rate", aggregate)
        self.assertEqual(aggregate["task_score_decrease_rate"], 0.0)

    def test_asqa_does_not_emit_qampari_metrics(self) -> None:
        example = Example(
            example_id="asqa-1",
            dataset="alce_asqa",
            split="dev",
            task_type="ambiguous_long_answer_citation",
            question="q",
            answers=[{"text": "alpha", "type": "short", "metadata": {"qa_pair_index": 0}},
                     {"text": "beta", "type": "short", "metadata": {"qa_pair_index": 1}}],
            gold_evidence=[],
            metadata={},
        )
        prediction = PredictionRecord(
            example_id="asqa-1",
            pipeline="rag",
            question="q",
            answer="alpha and beta",
            task_track="long_answer_citation",
        )

        scored = evaluate_prediction_track_metrics(prediction, example=example, qrels=[])

        self.assertEqual(scored.metrics["gold_answer_recall"], 1.0)
        self.assertNotIn("qampari_recall@5", scored.metrics)

    def test_hallucination_detection_reports_response_and_character_span_f1(self) -> None:
        metrics = hallucination_detection_metrics(
            gold_response_labels=[True, False],
            predicted_response_labels=[True, False],
            gold_spans=[[(0, 3)], []],
            predicted_spans=[[(0, 2)], []],
        )

        self.assertEqual(metrics["hallucination_response_f1"], 1.0)
        self.assertAlmostEqual(metrics["hallucination_span_precision"], 1.0)
        self.assertAlmostEqual(metrics["hallucination_span_recall"], 2 / 3)

    def test_unified_report_includes_computed_and_unavailable_metrics(self) -> None:
        document = {
            "run_id": "run-1",
            "dataset": "kilt_nq",
            "split": "dev",
            "task_track": "short_answer",
            "num_examples": 2,
            "pipelines": {"closed_book": {"em": 0.5, "f1": 0.6}},
            "confidence_intervals": {
                "closed_book": {"em": {"ci95_low": 0.1, "ci95_high": 0.9, "n": 2}}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            rows = write_unified_metrics_report(Path(tmp), document)
            with (Path(tmp) / "all_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertTrue((Path(tmp) / "all_metrics.html").exists())
            self.assertEqual(len(rows), len(csv_rows))
            by_metric = {row["metric"]: row for row in csv_rows}
            self.assertEqual(by_metric["em"]["status"], "computed")
            self.assertEqual(by_metric["hotpot_joint_f1"]["status"], "not_applicable_dataset")
            self.assertEqual(by_metric["gold_context_gap_f1"]["status"], "requires_paired_oracle_run")

    def test_paired_comparison_uses_candidate_minus_baseline_delta(self) -> None:
        baseline = [self.metric_prediction("a", 0.0), self.metric_prediction("b", 0.5)]
        candidate = [self.metric_prediction("a", 1.0), self.metric_prediction("b", 0.5)]

        rows = paired_metric_comparisons(
            baseline,
            candidate,
            keys=("f1",),
            samples=50,
            seed=7,
        )

        self.assertEqual(rows[0]["mean_delta"], 0.5)
        self.assertEqual(rows[0]["n"], 2)
        self.assertLessEqual(rows[0]["ci95_low"], 0.5)
        self.assertGreaterEqual(rows[0]["ci95_high"], 0.5)

    @staticmethod
    def verification_prediction(example_id: str, initial: int, final: int) -> PredictionRecord:
        return PredictionRecord(
            example_id=example_id,
            pipeline="verification",
            question="q",
            answer="a",
            task_track="verification_hallucination",
            metrics={
                "pre_verification_em": float(initial),
                "post_verification_em": float(final),
                "pre_verification_f1": float(initial),
                "post_verification_f1": float(final),
            },
        )

    @staticmethod
    def metric_prediction(example_id: str, f1: float) -> PredictionRecord:
        from rag_benchmark.evaluation.identity import EvaluationIdentity
        return PredictionRecord(
            metadata=EvaluationIdentity("fixture", "v1", "dev", 42).metadata(),
            example_id=example_id,
            pipeline="p",
            question="q",
            answer="a",
            metrics={"f1": f1},
        )

    @staticmethod
    def long_verification_prediction(example_id: str, initial: float, final: float) -> PredictionRecord:
        return PredictionRecord(
            example_id=example_id,
            pipeline="verification",
            question="q",
            answer="a",
            task_track="long_answer_citation",
            metrics={
                "pre_verification_em": 0.0,
                "post_verification_em": 0.0,
                "pre_verification_f1": initial,
                "post_verification_f1": final,
            },
        )

    @staticmethod
    def verification_prediction_with_f1(
        example_id: str,
        initial: float,
        final: float,
    ) -> PredictionRecord:
        return PredictionRecord(
            example_id=example_id,
            pipeline="verification",
            question="q",
            answer="a",
            task_track="verification_hallucination",
            metrics={
                "pre_verification_em": 0.0,
                "post_verification_em": 0.0,
                "pre_verification_f1": initial,
                "post_verification_f1": final,
            },
        )


if __name__ == "__main__":
    unittest.main()
