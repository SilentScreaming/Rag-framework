"""Experiment-level exact reuse contracts; fixtures only, never network calls."""
from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from rag_benchmark.evaluation.cache import ExperimentEvaluationCache
from rag_benchmark.evaluation.claims import ExtractionResult, parse_extraction
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.evaluation.service import EvaluationService, evaluated_prediction
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord
from rag_benchmark.evaluation.identity import EvaluationIdentity


USAGE = {"input_tokens": 10, "output_tokens": 4, "usage_source": "fixture", "attempts": 1}


class Extractor:
    identity = {"protocol": "exact-cache-fixture-v1", "model": "fixture"}
    def __init__(self, *, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first
        self.generator = SimpleNamespace(model="fixture", reasoning_effort=None, max_output_tokens=32768)
    def build_prompt(self, *, question, answer, task_type):
        return json.dumps([question, answer, task_type, self.identity])
    def extract(self, *, question, answer, task_type):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return ExtractionResult("failed", reason="fixture_transport_failure", trace={"usage": USAGE})
        result = parse_extraction({"segments": [{"segment_id": "s1", "kind": "factual",
            "claims": [answer]}]}, question=question, answer=answer)
        return ExtractionResult(result.status, result.claims, result.segments, trace={"usage": USAGE},
                                answer_behavior={"status": "computed", "label": "answered", "reason": "fixture"})


class Judge:
    identity = {"protocol": "exact-cache-judge-v1", "model": "fixture"}
    uses_question_context = True
    def __init__(self):
        self.calls = 0
        self.generator = SimpleNamespace(model="fixture", reasoning_effort=None, max_output_tokens=32768)
    def judge_joint(self, claim, texts, *, question=None):
        self.calls += 1
        return EvidenceDecision("completed", "supported" if claim in texts else "insufficient",
                                trace={"usage": USAGE, "question": question})


class Aligner:
    identity = {"protocol": "exact-cache-alignment-v1", "model": "fixture"}
    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first
    def align(self, *, answer, claims, declarations):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise ValueError("fixture_alignment_failure")
        return {c.claim_id: {"declaration_ids": [declarations[0].claim_id], "status": "matched"}
                for c in claims}, {"usage": USAGE}


def record(*, evidence="Paris is in France.", citation="Paris is in France."):
    p = PredictionRecord(example_id="e1", pipeline="a", question="Where is Paris?", answer="Paris is in France.",
        run_id="run-a", metadata={**EvaluationIdentity("fixture", "fixture-v1", "dev", 42).metadata(),
                                 "generation_context": {"visible_evidence": [{"text": evidence}]}})
    ref = CitationRecord(citation_id="1", chunk_id="d1", cited_text=citation)
    return p.model_copy(update={"citations": [ref], "claim_checks": [
        ClaimCheckRecord(claim_id="decl", claim_text="Paris belongs to France.", citations=[ref])]})


class EvaluationExactCacheTest(unittest.TestCase):
    def service(self, cache, **kwargs):
        return EvaluationService(extractor=kwargs.pop("extractor", Extractor()),
            judge=kwargs.pop("judge", Judge()), aligner=kwargs.pop("aligner", Aligner()),
            experiment_cache=cache, **kwargs)

    def test_new_run_reuses_valid_stages_and_does_not_recharge_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ExperimentEvaluationCache(Path(directory)/"experiment.sqlite")
            first_service = self.service(cache)
            p = record()
            first = first_service.evaluate(p)
            second_service = self.service(ExperimentEvaluationCache(cache.path))
            second = second_service.evaluate(p.model_copy(update={"run_id": "run-b", "pipeline": "b"}))
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["extraction"]["claims"], second["extraction"]["claims"])
            self.assertEqual(second["extraction"]["answer_behavior"]["label"], "answered")
            self.assertEqual(second_service.calls, [])
            self.assertTrue(all(s["cache_hit"] for s in second["evaluation_reuse"]["stages"]))
            self.assertEqual(second["extraction"]["trace"]["usage"]["input_tokens"], 0)
            self.assertEqual(second["attribution"]["alignment_trace"]["usage"]["output_tokens"], 0)
            self.assertEqual(first["extraction"]["trace"]["usage"]["input_tokens"], 10)

    def test_changed_context_or_citation_does_not_reuse_wrong_support(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ExperimentEvaluationCache(Path(directory)/"experiment.sqlite")
            first = self.service(cache).evaluate(record())
            changed_context = self.service(cache)
            second = changed_context.evaluate(record(evidence="Unrelated evidence."))
            self.assertEqual(changed_context.extractor.calls, 0)
            self.assertEqual(second["metrics"]["visible_context_supported_claim_rate"]["value"], 0)
            self.assertEqual(first["metrics"]["visible_context_supported_claim_rate"]["value"], 1)
            changed_citation = self.service(cache)
            third = changed_citation.evaluate(record(citation="Unrelated evidence."))
            self.assertEqual(changed_citation.aligner.calls, 0)
            self.assertEqual(third["metrics"]["citation_contribution_precision"]["value"], 0)

    def test_model_reasoning_cap_and_prompt_version_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ExperimentEvaluationCache(Path(directory)/"experiment.sqlite")
            self.service(cache).evaluate(record())
            for field, value in [("reasoning_effort", "none"), ("model", "different"), ("max_output_tokens", 16384)]:
                extractor = Extractor(); setattr(extractor.generator, field, value)
                self.service(cache, extractor=extractor).evaluate(record())
                self.assertEqual(extractor.calls, 1, field)
            extractor = Extractor(); extractor.identity = {**extractor.identity, "protocol": "changed-protocol"}
            self.service(cache, extractor=extractor).evaluate(record())
            self.assertEqual(extractor.calls, 1)

    def test_failures_are_not_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ExperimentEvaluationCache(Path(directory)/"experiment.sqlite")
            extractor = Extractor(fail_first=True); service = self.service(cache, extractor=extractor)
            first, second = service.evaluate(record()), service.evaluate(record())
            self.assertEqual(first["extraction"]["status"], "failed")
            self.assertEqual(second["extraction"]["status"], "completed")
            self.assertEqual(extractor.calls, 2)
            aligner = Aligner(fail_first=True)
            # A new declaration input prevents reusing the earlier alignment.
            p = record(); p.claim_checks[0].claim_text = "A distinct declaration."
            service = self.service(cache, aligner=aligner)
            service.evaluate(p); service.evaluate(p)
            self.assertEqual(aligner.calls, 2)
            self.assertEqual(sum(c["kind"] == "alignment" for c in service.calls), 2)

    def test_parallel_services_share_first_valid_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"experiment.sqlite"
            services = [self.service(ExperimentEvaluationCache(path)) for _ in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda s: s.evaluate(record()), services))
            self.assertEqual(sum(s.extractor.calls for s in services), 1)
            self.assertEqual(sum(s.aligner.calls for s in services), 1)
            self.assertTrue(all(r["metrics"] == results[0]["metrics"] for r in results))

    def test_failed_behavior_field_is_not_frozen_with_valid_claims(self):
        class BehaviorExtractor(Extractor):
            def extract(self, **inputs):
                row = super().extract(**inputs)
                return ExtractionResult(row.status, row.claims, row.segments, trace=row.trace,
                    answer_behavior={"status": "failed", "label": "unknown"} if self.calls == 1 else row.answer_behavior)
        with tempfile.TemporaryDirectory() as directory:
            extractor = BehaviorExtractor()
            service = self.service(ExperimentEvaluationCache(Path(directory)/"experiment.sqlite"), extractor=extractor)
            first, second = service.evaluate(record()), service.evaluate(record())
            self.assertEqual(first["extraction"]["status"], "completed")
            self.assertEqual(first["extraction"]["answer_behavior"]["status"], "failed")
            self.assertEqual(second["extraction"]["answer_behavior"]["label"], "answered")
            self.assertEqual(extractor.calls, 2)


if __name__ == "__main__":
    unittest.main()
