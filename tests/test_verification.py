from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord
from rag_benchmark.verification.cove import (
    CoVeVerificationConfig,
    CoVeVerifier,
    revise_answer_from_supported_claims,
)


def claim(claim_id: str, text: str, evidence: str) -> ClaimCheckRecord:
    return ClaimCheckRecord(
        claim_id=claim_id,
        claim_text=text,
        citations=[
            CitationRecord(
                citation_id=f"citation-{claim_id}",
                claim_id=claim_id,
                chunk_id=f"chunk-{claim_id}",
                cited_text=evidence,
            )
        ],
    )


def prediction(claims: list[ClaimCheckRecord]) -> PredictionRecord:
    return PredictionRecord(
        example_id="ex-1",
        pipeline="rag_verification",
        question="Who wrote the book?",
        answer="Ada wrote the book. It was published in 1999.",
        claim_checks=claims,
    )


class VerificationTest(unittest.TestCase):
    def test_verification_passes_supported_claims(self) -> None:
        verifier = CoVeVerifier()

        updated = verifier.verify(prediction([claim("claim-1", "Ada wrote the book.", "Ada wrote the book.")]))

        self.assertEqual(updated.verification.status, "passed")
        self.assertEqual(updated.answer, "Ada wrote the book. It was published in 1999.")
        self.assertEqual(updated.metrics["verification_passed"], 1.0)

    def test_verification_revises_unsupported_claims(self) -> None:
        verifier = CoVeVerifier()
        claims = [
            claim("claim-1", "Ada wrote the book.", "Ada wrote the book."),
            claim("claim-2", "It was published in 1999.", "Grace painted a mural."),
        ]

        updated = verifier.verify(prediction(claims))

        self.assertEqual(updated.verification.status, "revised")
        self.assertEqual(updated.answer, "Ada wrote the book.")
        self.assertEqual(updated.metrics["verification_failed_claim_count"], 1.0)

    def test_revision_falls_back_to_i_do_not_know(self) -> None:
        self.assertEqual(revise_answer_from_supported_claims([]), "I don't know.")

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CoVeVerificationConfig(questions_per_answer=0).validate()


if __name__ == "__main__":
    unittest.main()
