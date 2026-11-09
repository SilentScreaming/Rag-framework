"""Failed outputs keep their pairing identity without gaining fallback scores."""
import unittest
from unittest.mock import Mock

from scripts.evaluate_predictions import pairing_view
from rag_benchmark.data import Example
from rag_benchmark.evaluation.answers import ANSWER_PROTOCOL
from rag_benchmark.evaluation.identity import EvaluationIdentity
from rag_benchmark.evaluation.service import EvaluationService
from rag_benchmark.evaluation.statistics import paired_comparisons
from rag_benchmark.evaluation.tracks import evaluate_prediction_track_metrics
from rag_benchmark.schemas import PredictionRecord


class FailedEvaluationTests(unittest.TestCase):
    def record(self, failed=False, protocol=ANSWER_PROTOCOL):
        metadata = EvaluationIdentity("kilt_nq", "fixture-v1", "formal", 42).metadata()
        if protocol is not None:
            metadata["answer_metric_protocol"] = protocol
        return PredictionRecord(example_id="e", pipeline="p", question="Q", answer="fallback",
            task_track="short_answer", generator={"schema_failure": failed}, metadata=metadata)

    def example(self):
        return Example("e", "kilt_nq", "formal", "short_answer", "Q",
            [{"text": "A", "type": "short"}], [], {})

    def test_failure_can_pair_without_mutating_saved_record(self):
        failed = self.record(True, None)
        paired_comparisons([pairing_view(failed)], [self.record()], keys=[], samples=0)
        self.assertNotIn("answer_metric_protocol", failed.metadata)
        self.assertTrue(pairing_view(failed).generator["schema_failure"])

    def test_explicit_incompatible_protocol_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            paired_comparisons([pairing_view(self.record(True, "old-protocol"))],
                [self.record()], keys=[], samples=0)
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            paired_comparisons([pairing_view(self.record(False, None))],
                [self.record()], keys=[], samples=0)

    def test_new_failure_records_keep_protocol_and_unavailable_score(self):
        out = evaluate_prediction_track_metrics(self.record(True, None), example=self.example(), qrels=[])
        self.assertEqual(out.metadata["answer_metric_protocol"], ANSWER_PROTOCOL)
        self.assertEqual(out.metadata["task_assessment"]["status"], "failed")
        self.assertNotIn("em", out.metrics)

    def test_failed_fallback_does_not_call_fact_extractor_or_judge(self):
        extractor, judge = Mock(), Mock()
        result = EvaluationService(extractor=extractor, judge=judge).evaluate(
            self.record(True, None), example=self.example(), reference_evidence=["A"])
        extractor.extract.assert_not_called()
        judge.judge_joint.assert_not_called()
        self.assertEqual(result["task"]["status"], "failed")
        self.assertEqual(result["extraction"]["reason"], "generation_failed")
        self.assertIsNone(result["metrics"]["em"]["value"])

    def test_replay_cli_forwards_model_and_actual_prices(self):
        from scripts.run_verification_replay import parse_args, replay_component_settings
        args = parse_args(["--dataset", "alce_qampari", "--split-profile", "formal",
            "--limit", "25", "--draft-predictions", "draft.jsonl", "--output-root", "output",
            "--run-id", "replay", "--model", "fixture-model", "--input-cost-per-1m-tokens", "0.10",
            "--output-cost-per-1m-tokens", "0.50", "--output-cap", "32768"])
        settings = replay_component_settings(args)
        self.assertEqual(settings.generator_model, "fixture-model")
        self.assertEqual(settings.query_rewriter, "openai")
        self.assertEqual(settings.input_cost_per_1m_tokens, 0.10)
        self.assertEqual(settings.output_cost_per_1m_tokens, 0.50)


if __name__ == "__main__":
    unittest.main()
