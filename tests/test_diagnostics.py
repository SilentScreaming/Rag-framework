from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.analysis.diagnostics import diagnose_prediction, summarize_diagnostics
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord, RetrievalTraceRecord


class DiagnosticsTest(unittest.TestCase):
    def test_retrieval_miss_is_primary_when_no_gold_hit(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-1",
            pipeline="rag",
            question="Who wrote it?",
            answer="wrong",
            metrics={"em": 0.0, "f1": 0.0, "any_gold_hit@10": 0.0, "recall@10": 0.0},
        )
        trace = RetrievalTraceRecord(
            example_id="ex-1",
            question="Who wrote it?",
            gold_chunk_ids=["gold"],
            metrics={"any_gold_hit@10": 0.0},
        )

        diagnostic = diagnose_prediction(prediction, retrieval_trace=trace)

        self.assertEqual(diagnostic.primary_error, "annotated_evidence_not_retrieved")
        self.assertNotIn("query_ambiguity", diagnostic.secondary_errors)
        self.assertEqual(diagnostic.hypotheses[0]["evidence_level"], "heuristic")
        self.assertTrue(diagnostic.needs_human_review)

    def test_citation_mismatch_and_unsupported_generation(self) -> None:
        citation = CitationRecord(chunk_id="chunk-1", support_status="unsupported")
        prediction = PredictionRecord(
            example_id="ex-2",
            pipeline="rag",
            question="Who wrote the notes?",
            answer="Ada wrote the notes.",
            citations=[citation],
            claim_checks=[
                ClaimCheckRecord(
                    claim_id="claim-1",
                    claim_text="Ada wrote the notes.",
                    status="unsupported",
                    citations=[citation],
                )
            ],
            metrics={
                "em": 1.0,
                "f1": 1.0,
                "claim_support_rate": 0.0,
                "citation_support_rate": 0.0,
            },
        )

        diagnostic = diagnose_prediction(prediction)

        self.assertEqual(diagnostic.primary_error, "correct_under_task_rule")
        self.assertIsNone(diagnostic.signals["claim_support_rate"])
        self.assertTrue(diagnostic.needs_human_review)

    def test_verification_miss_when_wrong_answer_passed(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-3",
            pipeline="rag_verification",
            question="Who wrote the notes?",
            answer="wrong",
            metrics={"em": 0.0, "f1": 0.0, "verification_passed": 1.0},
        )

        diagnostic = diagnose_prediction(prediction)

        self.assertEqual(diagnostic.primary_error, "incorrect_unattributed")

    def test_summary_counts_labels(self) -> None:
        records = [
            diagnose_prediction(
                PredictionRecord(
                    example_id="ex-1",
                    pipeline="rag",
                    question="Q",
                    answer="A",
                    metrics={"em": 1.0, "f1": 1.0},
                )
            ),
            diagnose_prediction(
                PredictionRecord(
                    example_id="ex-2",
                    pipeline="rag",
                    question="Q",
                    answer="wrong",
                    metrics={"em": 0.0, "f1": 0.0, "verification_passed": 1.0},
                )
            ),
        ]

        summary = summarize_diagnostics(records)

        self.assertEqual(summary["num_records"], 2)
        self.assertEqual(summary["primary_error_counts"]["correct_under_task_rule"], 1)
        self.assertEqual(summary["primary_error_counts"]["incorrect_unattributed"], 1)
        self.assertIsNone(summary["hallucination_case_rate"])
        self.assertIsNone(summary["retrieval_failure_rate"])
        self.assertEqual(summary["needs_human_review_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
