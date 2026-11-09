import unittest

from rag_benchmark.evaluation.statistics import metric_value
from rag_benchmark.evaluation.tracks import verification_intervention_metrics
from rag_benchmark.run_summary import summarize_run
from rag_benchmark.schemas import AnswerBundleRecord, PredictionRecord


def prediction(**changes):
    return PredictionRecord(run_id="fixture", pipeline="fixture", example_id="e", question="q", answer="a", **changes)


class RunReportingTests(unittest.TestCase):
    def test_plain_output_truncation_is_counted_outside_format_denominator(self):
        row = prediction(generator={"finish_reason": "length"})
        result = summarize_run([row])
        self.assertEqual(result["format_required_count"], 0)
        self.assertIsNone(result["final_format_failure_rate"])
        self.assertEqual(result["truncated_answer_count"], 1)

    def test_list_failure_and_valid_abstention_both_keep_format_denominator(self):
        bad = prediction(answer_bundle=AnswerBundleRecord(list_items=[], metadata={"list_format_status": "unavailable"}))
        refusal = prediction(answer_bundle=AnswerBundleRecord(abstained=True, list_items=[],
            metadata={"list_format_status": "computed", "list_format_reason": "abstention_marker"}))
        result = summarize_run([bad, refusal])
        self.assertEqual(result["format_required_count"], 2)
        self.assertEqual(result["first_pass_format_failure_rate"], 0.5)
        self.assertEqual(result["final_format_failure_rate"], 0.5)

    def test_mixed_score_meanings_do_not_become_one_revision_gain(self):
        rows = [prediction(metrics={"pre_verification_task_score": 0.0, "post_verification_task_score": 1.0},
                    metadata={"pre_verification_task_score_metric": name, "post_verification_task_score_metric": name})
                for name in ("f1", "eligibility_accuracy")]
        self.assertIsNone(metric_value(rows, "net_verification_gain"))
        self.assertNotIn("net_verification_gain", verification_intervention_metrics(rows, track="short_answer"))


if __name__ == "__main__":
    unittest.main()
