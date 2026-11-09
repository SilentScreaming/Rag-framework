from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.evaluation.citation import citation_quality_metrics, evaluate_citation_quality
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord


def make_prediction() -> PredictionRecord:
    citation = CitationRecord(
        citation_id="C1-S0-alpha",
        claim_id="claim-1",
        chunk_id="chunk-alpha",
        sentence_id=0,
        cited_text="Ada wrote the book.",
        support_status="supported",
        metadata={"judgment_scope": "citation"},
    )
    return PredictionRecord(
        example_id="ex-1",
        pipeline="best_practice_rag",
        question="Who wrote the book?",
        answer="Ada wrote the book.",
        citations=[citation],
        claim_checks=[
            ClaimCheckRecord(
                claim_id="claim-1",
                claim_text="Ada wrote the book.",
                status="supported",
                citations=[citation],
            ),
            ClaimCheckRecord(
                claim_id="claim-2",
                claim_text="It was published in 1999.",
                status="unsupported",
                citations=[],
            ),
        ],
        metadata={"citation_generation": {"invalid_citation_count": 1}},
    )


class CitationQualityTest(unittest.TestCase):
    def test_citation_quality_metrics(self) -> None:
        metrics = citation_quality_metrics(
            make_prediction(),
            gold_chunk_ids={"chunk-alpha", "chunk-beta"},
            gold_sentence_refs={("chunk-alpha", 0)},
        )

        self.assertEqual(metrics["citation_count"], 1.0)
        self.assertEqual(metrics["citation_support_rate"], 1.0)
        self.assertEqual(metrics["cited_claim_coverage"], 0.5)
        self.assertEqual(metrics["citation_gold_hit_rate"], 1.0)
        self.assertEqual(metrics["citation_recall"], 0.5)
        self.assertEqual(metrics["citation_sentence_gold_hit_rate"], 1.0)
        self.assertEqual(metrics["invalid_citation_rate"], 0.5)

    def test_evaluate_citation_quality_updates_prediction_metrics(self) -> None:
        updated = evaluate_citation_quality(make_prediction(), gold_chunk_ids={"chunk-alpha"})

        self.assertEqual(updated.metrics["citation_precision"], 1.0)
        self.assertEqual(updated.metrics["citation_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
