from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.evaluation.llm_judge import (  # noqa: E402
    LLMEvidenceJudge,
    judge_prediction,
    parse_llm_judge_output,
)
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord


class FakeJudgeGenerator:
    provider = "fake"
    model = "fake-judge"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        self.prompt = prompt
        return json.dumps(self.payload)


class LLMJudgeTest(unittest.TestCase):
    def test_parse_llm_judge_output(self) -> None:
        payload = parse_llm_judge_output(
            '```json\n{"status": "supported", "score": 0.9, "rationale": "direct"}\n```'
        )

        self.assertEqual(payload["status"], "supported")
        self.assertEqual(payload["score"], 0.9)

    def test_llm_judge_updates_prediction_claims_and_citations(self) -> None:
        generator = FakeJudgeGenerator(
            {"status": "supported", "score": 0.95, "rationale": "evidence matches"}
        )
        result = judge_prediction(make_prediction(), judge=LLMEvidenceJudge(generator))

        judged = result.prediction
        self.assertEqual(judged.claim_checks[0].status, "supported")
        self.assertEqual(judged.citations[0].support_status, "unjudged")
        self.assertEqual(judged.citations[0].metadata["evidence_set_status"], "supported")
        self.assertEqual(judged.metrics["judge_llm_called_count"], 1.0)
        self.assertEqual(result.traces[0].judge, "llm_evidence_judge")

    def test_only_uncertain_skips_high_confidence_rule_supported_claim(self) -> None:
        generator = FakeJudgeGenerator(
            {"status": "unsupported", "score": 0.2, "rationale": "would be wrong"}
        )
        result = judge_prediction(
            make_prediction(),
            judge=LLMEvidenceJudge(generator),
            only_uncertain=True,
            uncertainty_threshold=0.75,
        )

        self.assertEqual(generator.calls, 0)
        self.assertEqual(result.prediction.claim_checks[0].status, "supported")
        self.assertEqual(result.prediction.metrics["judge_llm_skipped_count"], 1.0)


def make_prediction() -> PredictionRecord:
    citation = CitationRecord(
        citation_id="C1-S0",
        claim_id="claim-1",
        chunk_id="chunk-1",
        sentence_id=0,
        cited_text="Ada wrote the notes.",
    )
    return PredictionRecord(
        example_id="ex-1",
        pipeline="rag",
        question="Who wrote the notes?",
        answer="Ada wrote the notes.",
        claim_checks=[
            ClaimCheckRecord(
                claim_id="claim-1",
                claim_text="Ada wrote the notes.",
                citations=[citation],
            )
        ],
        citations=[citation],
    )


if __name__ == "__main__":
    unittest.main()
