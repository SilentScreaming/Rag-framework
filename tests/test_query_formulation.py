from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.query_formulation import (  # noqa: E402
    IdentityQueryFormulator,
    LLMQueryRewriter,
    RuleBasedQueryRewriter,
)


class FakeTextGenerator:
    provider = "fake"
    model = "fake-model"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate_text(self, prompt: str) -> str:
        self.prompt = prompt
        return self.text


class QueryFormulationTest(unittest.TestCase):
    def test_identity_returns_original_question(self) -> None:
        result = IdentityQueryFormulator().formulate("Who wrote Dune?")
        self.assertEqual(result.query, "Who wrote Dune?")
        self.assertEqual(result.method, "identity")
        self.assertFalse(result.metadata["fallback"])

    def test_rule_rewriter_keeps_entity_terms(self) -> None:
        result = RuleBasedQueryRewriter(max_terms=4).formulate(
            "Who was the composer of The Magic Flute?"
        )
        self.assertEqual(result.query, "Who composer The Magic")
        self.assertEqual(result.method, "rule_rewrite")
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)

    def test_llm_rewriter_parses_json(self) -> None:
        generator = FakeTextGenerator(json.dumps({"query": "Dune author", "rationale": "short"}))
        result = LLMQueryRewriter(generator).formulate("Who wrote Dune?")
        self.assertEqual(result.query, "Dune author")
        self.assertFalse(result.metadata["fallback"])
        self.assertEqual(result.metadata["raw"]["rationale"], "short")

    def test_llm_rewriter_falls_back_on_bad_output(self) -> None:
        generator = FakeTextGenerator("not json")
        result = LLMQueryRewriter(generator).formulate("Who wrote Dune?")
        self.assertEqual(result.query, "Who wrote Dune?")
        self.assertTrue(result.metadata["fallback"])
        self.assertIn("JSONDecodeError", result.metadata["error"])


if __name__ == "__main__":
    unittest.main()
