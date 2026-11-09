from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.evaluation.evidence import (
    RuleBasedEvidenceJudge,
    TransformersNLIEvidenceJudge,
    evaluate_evidence_consistency,
)
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord


def prediction_with_claim(claim: ClaimCheckRecord) -> PredictionRecord:
    return PredictionRecord(
        example_id="ex-1",
        pipeline="best_practice_rag",
        question="Who wrote the book?",
        answer="Ada wrote the book.",
        claim_checks=[claim],
    )


class EvidenceConsistencyTest(unittest.TestCase):
    def tearDown(self) -> None:
        TransformersNLIEvidenceJudge._PIPELINE_CACHE.clear()

    def test_supported_claim_updates_claim_and_citation_status(self) -> None:
        claim = ClaimCheckRecord(
            claim_id="claim-1",
            claim_text="Ada wrote the book.",
            citations=[
                CitationRecord(
                    citation_id="C1-S0-alpha",
                    claim_id="claim-1",
                    chunk_id="chunk-alpha",
                    sentence_id=0,
                    cited_text="Ada wrote the book in 1999.",
                )
            ],
        )

        updated = evaluate_evidence_consistency(prediction_with_claim(claim), judge=RuleBasedEvidenceJudge())

        self.assertEqual(updated.claim_checks[0].status, "supported")
        self.assertEqual(updated.claim_checks[0].citations[0].support_status, "unjudged")
        self.assertEqual(updated.claim_checks[0].citations[0].metadata["evidence_set_status"], "supported")
        self.assertEqual(updated.metrics["claim_support_rate"], 1.0)

    def test_unsupported_claim_updates_metrics(self) -> None:
        claim = ClaimCheckRecord(
            claim_id="claim-1",
            claim_text="Ada wrote the book.",
            citations=[
                CitationRecord(
                    citation_id="C1-S0-alpha",
                    claim_id="claim-1",
                    chunk_id="chunk-alpha",
                    cited_text="Grace painted a mural.",
                )
            ],
        )

        updated = evaluate_evidence_consistency(prediction_with_claim(claim), judge=RuleBasedEvidenceJudge())

        self.assertEqual(updated.claim_checks[0].status, "unsupported")
        self.assertEqual(updated.metrics["unsupported_claim_rate"], 1.0)

    def test_claim_without_citation_is_unsupported(self) -> None:
        claim = ClaimCheckRecord(claim_id="claim-1", claim_text="Ada wrote the book.", citations=[])

        updated = evaluate_evidence_consistency(prediction_with_claim(claim), judge=RuleBasedEvidenceJudge())

        self.assertEqual(updated.claim_checks[0].status, "unsupported")
        self.assertEqual(updated.claim_checks[0].judge["rationale"], "claim has no citations")

    def test_nli_model_and_tokenizer_are_shared_for_matching_configuration(self) -> None:
        tokenizer_calls = []
        model_calls = []

        class Loader:
            def __init__(self, calls, value) -> None:
                self.calls = calls
                self.value = value

            def from_pretrained(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self.value

        class FakeModel:
            def eval(self):
                return self

        tokenizer = object()
        model = FakeModel()
        module = types.SimpleNamespace(
            AutoTokenizer=Loader(tokenizer_calls, tokenizer),
            AutoModelForSequenceClassification=Loader(model_calls, model),
        )
        original = sys.modules.get("transformers")
        sys.modules["transformers"] = module
        try:
            first = TransformersNLIEvidenceJudge(model_name="test/nli", revision="frozen")
            second = TransformersNLIEvidenceJudge(model_name="test/nli", revision="frozen")
            self.assertEqual(first._load(), second._load())
        finally:
            if original is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original

        self.assertEqual(len(tokenizer_calls), 1)
        self.assertEqual(len(model_calls), 1)


if __name__ == "__main__":
    unittest.main()
