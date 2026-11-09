from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
from rag_benchmark.generation.citation import (
    CitationAwareGenerator,
    apply_citation_generation,
    build_citation_prompt,
    parse_citation_generation,
)
from rag_benchmark.schemas import PredictionRecord


def make_context() -> CompressedContext:
    return CompressedContext(
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
                parent_score=3.0,
                selection_score=4.0,
            )
        ],
    )


class FakePromptGenerator:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payload)


class CitationGenerationTest(unittest.TestCase):
    def test_parse_citation_generation_maps_context_to_citation_records(self) -> None:
        raw = json.dumps(
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
        )

        parsed = parse_citation_generation(raw, context=make_context())

        self.assertEqual(parsed.answer, "Ada wrote the book.")
        self.assertEqual(parsed.citations[0].chunk_id, "chunk-alpha")
        self.assertEqual(parsed.citations[0].sentence_id, 0)
        self.assertEqual(parsed.claim_checks[0].citations[0].citation_id, "C1-S0-alpha")

    def test_invalid_citation_is_recorded_without_creating_citation(self) -> None:
        raw = json.dumps(
            {
                "answer": "Ada wrote the book.",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "text": "Ada wrote the book.",
                        "citations": ["missing"],
                    }
                ],
            }
        )

        parsed = parse_citation_generation(raw, context=make_context())

        self.assertEqual(parsed.citations, [])
        self.assertEqual(parsed.invalid_citations[0]["citation_id"], "missing")
        self.assertEqual(parsed.claim_checks[0].metadata["invalid_citations"][0]["reason"], "unknown_context_id")

    def test_missing_top_level_answer_is_reconstructed_from_model_claims(self) -> None:
        parsed = parse_citation_generation(
            json.dumps(
                {
                    "claims": [
                        {
                            "claim_id": "claim-1",
                            "text": "Ada wrote the book.",
                            "citations": ["C1-S0-alpha"],
                        }
                    ]
                }
            ),
            context=make_context(),
            task_track="long_answer_citation",
        )

        self.assertEqual(parsed.answer, "Ada wrote the book.")
        self.assertEqual(parsed.repairs, ["answer_reconstructed_from_claim_texts"])

    def test_apply_citation_generation_updates_prediction(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-1",
            pipeline="best_practice_rag",
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
            context=make_context(),
        )

        updated = apply_citation_generation(prediction, parsed)

        self.assertEqual(updated.answer, "Ada wrote the book.")
        self.assertEqual(len(updated.citations), 1)
        self.assertEqual(len(updated.claim_checks), 1)

    def test_citation_aware_generator_builds_prompt_and_parses_output(self) -> None:
        fake = FakePromptGenerator(
            {
                "answer": "Ada wrote the book.",
                "claims": [{"claim_id": "claim-1", "text": "Ada wrote the book.", "citations": ["C1-S0-alpha"]}],
            }
        )
        generator = CitationAwareGenerator(fake)

        parsed = generator.generate(question="Who wrote the book?", context=make_context())

        self.assertEqual(parsed.answer, "Ada wrote the book.")
        self.assertIn("Context:", fake.prompts[0])
        self.assertIn("C1-S0-alpha", build_citation_prompt(question="Q?", context=make_context()))


if __name__ == "__main__":
    unittest.main()
