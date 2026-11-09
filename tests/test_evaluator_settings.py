"""Evaluator source identity, output handling, and model settings."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_benchmark.evaluation.exact_evidence import TextExactJudge
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.evaluation.settings import evaluator_settings, identified_evidence, reasoning_parameter
from rag_benchmark.generation.usage import TextResponse


class Backend:
    model = "offline-fixture"
    max_output_tokens = 32768
    reasoning_effort = None

    def __init__(self, text='{"label":"supported","reason":"Direct statement."}'):
        self.text = text

    def generate_text_result(self, prompt):
        self.prompt = prompt
        return TextResponse(self.text, 50, 20, "offline-fixture", finish_reason="stop")


class EvaluatorSettingsTests(unittest.TestCase):
    def test_structured_sources_reach_judge_without_invented_passage_number(self):
        backend = Backend()
        supplied = [{"id": "opaque:3", "title": "Source heading", "text": "A statement."}]
        decision = TextExactJudge(backend).judge_joint("A statement.", identified_evidence(supplied))
        self.assertEqual(decision.status, "completed")
        self.assertEqual(decision.trace["evidence"], supplied)
        payload = json.loads(backend.prompt[backend.prompt.index('{"claim":'):])
        self.assertEqual(payload["evidence"], supplied)
        self.assertNotIn("passage_number", payload["evidence"][0])

    def test_plain_text_evidence_remains_compatible(self):
        backend = Backend()
        decision = TextExactJudge(backend).judge_joint("A statement.", ["A statement."])
        self.assertEqual(decision.label, "supported")
        self.assertEqual(decision.trace["evidence"], ["A statement."])

    def test_no_label_is_repaired_from_a_conflicting_reason(self):
        backend = Backend('{"label":"contradicted","reason":"The passage directly supports this claim."}')
        decision = TextExactJudge(backend).judge_joint("A statement.", ["A statement."])
        self.assertEqual(decision.label, "contradicted")
        self.assertIn("supports", decision.reason)

    def test_invalid_output_retains_response_and_safe_diagnostic(self):
        backend = Backend('{"label":"invented"}')
        decision = TextExactJudge(backend).judge_joint("claim", ["evidence"])
        self.assertEqual(decision.status, "failed")
        self.assertIsNone(decision.label)
        self.assertEqual(decision.trace["raw_response"], backend.text)
        self.assertEqual(decision.trace["validation_error"], "invalid semantic judgment")

    def test_provider_value_error_does_not_expose_its_message(self):
        class Failing(Backend):
            def generate_text_result(self, prompt):
                raise ValueError("private-credential-fixture")
        decision = TextExactJudge(Failing()).judge_joint("claim", ["evidence"])
        self.assertEqual(decision.status, "failed")
        self.assertNotIn("private-credential-fixture", json.dumps(decision.to_dict()))

    def test_default_profile_does_not_mean_no_reasoning(self):
        settings = evaluator_settings(model="fixture-model")
        self.assertEqual(settings["model"], "fixture-model")
        self.assertIsNone(settings["reasoning_effort"])
        self.assertEqual(settings["max_output_tokens"], 4096)
        self.assertEqual(reasoning_parameter("none"), "none")
        with self.assertRaises(ValueError):
            evaluator_settings(model="fixture-model", output_cap=0)


if __name__ == "__main__":
    unittest.main()
