from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
from rag_benchmark.evaluation.citation import evaluate_citation_quality
from rag_benchmark.evaluation.evidence import evaluate_evidence_consistency
from rag_benchmark.generation.citation import apply_citation_generation, parse_citation_generation
from rag_benchmark.schemas import PredictionRecord
from rag_benchmark.verification.cove import CoVeVerifier


class AttributionVerificationFlowTest(unittest.TestCase):
    def test_citation_evidence_quality_and_verification_flow(self) -> None:
        context = CompressedContext(
            query="Who wrote the book?",
            sentences=[
                CompressedSentence(
                    context_id="C1-S0-alpha",
                    chunk_id="chunk-alpha",
                    doc_id="doc-alpha",
                    title="Alpha",
                    sentence_id=0,
                    text="Ada wrote the book.",
                    parent_rank=1,
                    parent_score=1.0,
                    selection_score=2.0,
                )
            ],
        )
        prediction = PredictionRecord(
            example_id="ex-1",
            pipeline="rag_verification",
            question="Who wrote the book?",
            answer="draft",
        )
        parsed = parse_citation_generation(
            json.dumps(
                {
                    "answer": "Ada wrote the book.",
                    "claims": [
                        {
                            "claim_id": "claim-1",
                            "text": "Ada wrote the book.",
                            "citations": ["C1-S0-alpha"],
                        }
                    ],
                }
            ),
            context=context,
        )

        cited = apply_citation_generation(prediction, parsed)
        checked = evaluate_evidence_consistency(cited)
        scored = evaluate_citation_quality(
            checked,
            gold_chunk_ids={"chunk-alpha"},
            gold_sentence_refs={("chunk-alpha", 0)},
        )
        verified = CoVeVerifier().verify(scored)

        self.assertEqual(verified.answer, "Ada wrote the book.")
        self.assertEqual(verified.claim_checks[0].status, "supported")
        self.assertNotIn("citation_support_rate", verified.metrics)
        self.assertEqual(verified.citations[0].support_status, "unjudged")
        self.assertEqual(verified.metrics["verification_passed"], 1.0)
        self.assertEqual(verified.verification.status, "passed")


if __name__ == "__main__":
    unittest.main()
