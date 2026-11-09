"""Extraction failure/isolation regressions; all responses are offline fixtures."""
from __future__ import annotations

import json
import unittest

from rag_benchmark.evaluation.attribution import align_citations
from rag_benchmark.evaluation.claims import (
    ExtractionValidationError, TextClaimExtractor, parse_extraction,
)
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.evaluation.identity import EvaluationIdentity
from rag_benchmark.evaluation.service import EvaluationService
from rag_benchmark.generation.usage import TextResponse
from rag_benchmark.schemas import CitationRecord, PredictionRecord


class FixtureBackend:
    model = "offline-fixture"
    max_output_tokens = 4096

    def __init__(self, payload, *, finish_reason="stop"):
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.finish_reason = finish_reason

    def generate_text_result(self, prompt):
        self.prompt = prompt
        return TextResponse(self.text, 20, 30, "offline-fixture", finish_reason=self.finish_reason)


def segment(*claims, kind="factual"):
    return {"segments": [{"segment_id": "s1", "kind": kind, "claims": list(claims)}]}


class ClaimExtractionTest(unittest.TestCase):
    def test_failed_contract_retains_raw_response_usage_and_controlled_reason(self):
        backend = FixtureBackend({"segments": []})
        result = TextClaimExtractor(backend).extract(question="q", answer="A fact.", task_type="open_qa")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "ValueError")
        self.assertEqual(result.trace["error_code"], "unaccounted_segments")
        self.assertEqual(result.trace["validation_message"], "unaccounted answer segments")
        self.assertEqual(result.trace["raw_response"], backend.text)
        self.assertEqual(result.trace["usage"]["output_tokens"], 30)
        self.assertTrue(result.trace["prompt_hash"])

    def test_invalid_json_has_diagnostic_and_original_text(self):
        result = TextClaimExtractor(FixtureBackend("not json")).extract(
            question="q", answer="A fact.", task_type="open_qa")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.trace["error_code"], "invalid_json_object")
        self.assertEqual(result.trace["raw_response"], "not json")

    def test_truncation_is_not_recovered_as_successful_json(self):
        payload = segment({"text": "A fact.", "source_quote": "A fact."})
        result = TextClaimExtractor(FixtureBackend(payload, finish_reason="length")).extract(
            question="q", answer="A fact.", task_type="open_qa")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.trace["error_code"], "incomplete_response")
        self.assertIn("raw_response", result.trace)

    def test_provider_exception_text_is_not_persisted(self):
        class FailingBackend:
            def generate_text_result(self, prompt):
                raise RuntimeError("request headers: private-credential-fixture")
        result = TextClaimExtractor(FailingBackend()).extract(question="q", answer="a", task_type="open_qa")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "RuntimeError")
        self.assertNotIn("private-credential-fixture", json.dumps(result.to_dict()))

    def test_unlocatable_quote_preserves_facts_without_fabricating_spans(self):
        answer = "Mira wrote Lantern and Neri illustrated it."
        result = parse_extraction(segment(
            {"text": "Mira wrote Lantern.", "source_quote": "Mira wrote Lantern"},
            {"text": "Neri illustrated Lantern.", "source_quote": "Neri illustrated Lantern"},
        ), question="Who made Lantern?", answer=answer)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[1].answer_spans, ())
        self.assertEqual(result.trace["source_association_status"], "unknown")
        self.assertEqual(result.trace["source_quote_issues"][0]["code"], "source_quote_not_found")
        a, b = result.claims[0].answer_spans[0]
        self.assertEqual(answer[a:b], "Mira wrote Lantern")

    def test_quote_location_failure_does_not_block_support_or_create_inline_link(self):
        answer = "Mira wrote Lantern [C1]."
        backend = FixtureBackend(segment({"text": "Mira wrote Lantern.", "source_quote": "incorrect quote"}))
        identity = EvaluationIdentity("fixture", "fixture-v1", "dev", 42)
        prediction = PredictionRecord(example_id="e", pipeline="fixture", question="Who wrote Lantern?",
            answer=answer, metadata=identity.metadata(),
            citations=[CitationRecord(citation_id="C1", chunk_id="doc", cited_text="Mira wrote Lantern.")])
        class Judge:
            identity = {"model": "offline-fixture", "protocol": "fixture"}
            def judge_joint(self, claim, texts):
                return EvidenceDecision("completed", "supported")
        extracted = TextClaimExtractor(backend).extract(question=prediction.question, answer=answer, task_type="open_qa")
        links, _ = align_citations(prediction, extracted.claims)
        self.assertEqual(links[extracted.claims[0].claim_id]["status"], "unresolved")
        self.assertEqual(links[extracted.claims[0].claim_id]["citations"], [])
        result = EvaluationService(extractor=TextClaimExtractor(backend), judge=Judge()).evaluate(
            prediction, reference_evidence=["Mira wrote Lantern."])
        self.assertEqual(result["metrics"]["reference_supported_claim_rate"]["value"], 1.0)

    def test_repeated_quote_is_not_assigned_to_first_occurrence(self):
        result = parse_extraction(segment({"text": "Mira wrote Lantern.", "source_quote": "Mira"}),
            question="q", answer="Mira wrote Lantern and Mira illustrated it.")
        self.assertEqual(result.claims[0].answer_spans, ())
        self.assertEqual(result.trace["source_quote_issues"][0]["code"], "source_quote_ambiguous")

    def test_null_quote_keeps_explicit_unknown_association(self):
        result = parse_extraction(segment({"text": "Mira wrote Lantern.", "source_quote": None}),
            question="Who wrote Lantern?", answer="Mira")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.claims[0].answer_spans, ())
        self.assertEqual(result.trace["source_association_status"], "unknown")

    def test_short_answer_is_accepted_as_question_conditioned_atom(self):
        backend = FixtureBackend(segment({"text": "Mira wrote Lantern.", "source_quote": "Mira"}))
        result = TextClaimExtractor(backend).extract(question="Who wrote Lantern?", answer="Mira", task_type="open_qa")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.claims[0].claim_text, "Mira wrote Lantern.")
        self.assertEqual(result.claims[0].answer_spans, ((0, 4),))
        self.assertIn("unexpected answer type", backend.prompt)
        self.assertIn("never substitute the correct answer", backend.prompt)
        self.assertIn("Who wrote Lantern?", backend.prompt)

    def test_unresolved_model_answer_remains_partial_without_invented_fact(self):
        backend = FixtureBackend(segment(kind="unresolved"))
        result = TextClaimExtractor(backend).extract(question="Who wrote Lantern?", answer="Mira", task_type="open_qa")
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.claims, ())

    def test_list_item_quote_can_use_header_relation_without_copying_header(self):
        answer = "Moons of Planet X: Lyra, Neri"
        result = parse_extraction(segment(
            {"text": "Lyra is a moon of Planet X.", "source_quote": "Lyra"},
            {"text": "Neri is a moon of Planet X.", "source_quote": "Neri"},
        ), question="List moons of Planet X.", answer=answer)
        self.assertEqual(len(result.claims), 2)
        self.assertTrue(all(c.answer_spans for c in result.claims))

    def test_unknown_or_duplicate_segment_still_fails_contract(self):
        with self.assertRaises(ExtractionValidationError):
            parse_extraction({"segments": [{"segment_id": "invented", "kind": "factual", "claims": ["x"]}]},
                question="q", answer="A fact.")


if __name__ == "__main__":
    unittest.main()
