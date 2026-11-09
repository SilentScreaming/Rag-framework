"""Cross-layer regression contracts for evaluation and execution boundaries.

No network, model downloads, paid generation or external dataset writes.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from test_config_driven_runner import make_best_practice_pipeline, make_dataset_slice, make_experiment
import test_experiment_suite_resume as resume_fixtures
from rag_benchmark.analysis.diagnostics import diagnose_prediction
from rag_benchmark.component_factory import ComponentFactorySettings, OpenAITextPromptGenerator
from rag_benchmark.config import PipelineConfig
from rag_benchmark.data import Annotation, CorpusChunk, Example, Qrel
from rag_benchmark.evaluation.citation import citation_quality_metrics
from rag_benchmark.evaluation.coverage import attach_evaluation_coverage
from rag_benchmark.evaluation.evidence import EvidenceJudgment, apply_evidence_judgment, evidence_consistency_metrics
from rag_benchmark.evaluation.tracks import verification_track_metrics
from rag_benchmark.generation.backends import GenerationResult, OpenAIGenerator
from rag_benchmark.generation.context import compressed_context_from_retrieved, prepare_context
from rag_benchmark.generation.service import generate_citation_answer
from rag_benchmark.generation.usage import TextResponse, response_from_api
from rag_benchmark.query_formulation.rewrite import IdentityQueryFormulator, LLMQueryRewriter
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.run_identity import run_identity
from rag_benchmark.runner import ConfigDrivenPipelineRunner, FormalRunSettings, PipelineState, load_completed_formal_run
from rag_benchmark.schemas import CitationRecord, ClaimCheckRecord, PredictionRecord
from rag_benchmark.verification.cove import CoVeVerifier, FactoredCoVeVerifier

ROOT = Path(__file__).resolve().parents[1]


def retrieved():
    chunk = CorpusChunk("doc", "doc", "scifact", "Title", "Background. The answer is Paris.",
        [{"sentence_id": 0, "text": "Background."},
         {"sentence_id": 1, "text": "The answer is Paris."}], {})
    return [RetrievedChunk(chunk, 1, 1.0)]


def prediction(**kwargs):
    return PredictionRecord(example_id="ex-1", pipeline="test", question="Which city?", answer="Paris", **kwargs)


class ContextContractsTest(unittest.TestCase):
    def test_no_compression_keeps_all_sentences(self):
        context = compressed_context_from_retrieved(query="city", retrieved=retrieved())
        self.assertEqual([s.text for s in context.sentences], ["Background.", "The answer is Paris."])
        self.assertEqual(context.metadata["compressor"], "none")

    def test_incomplete_sentence_metadata_cannot_drop_text(self):
        row = retrieved()[0]
        row = replace(row, chunk=replace(row.chunk, text="Unmapped prefix. " + row.chunk.text))
        context = compressed_context_from_retrieved(query="city", retrieved=[row])
        self.assertIn("Unmapped prefix.", context.to_prompt_context())

    def test_both_generation_paths_consume_selected_content(self):
        selected = compressed_context_from_retrieved(query="city", retrieved=retrieved())
        selected = replace(selected, sentences=selected.sentences[1:], metadata={"compressor": "fixture_selector"})
        hashes = []
        for mode in ("context_grounded", "citation_aware"):
            config = make_best_practice_pipeline().to_json_dict()
            config["stages"]["generation"] = mode
            runner = ConfigDrivenPipelineRunner(repo_root=ROOT, experiment=make_experiment(PipelineConfig.model_validate(config)),
                dataset_slice=make_dataset_slice(), settings=FormalRunSettings(save_outputs=False,
                    component_settings=ComponentFactorySettings(generator_kind="extractive")))
            captured = []
            def generate(**kwargs):
                captured.extend(kwargs["contexts"])
                return GenerationResult("Paris", 0, 0, 0, None, {})
            runner.base_generator = SimpleNamespace(generate=generate)
            state = PipelineState(example=make_dataset_slice().examples[0], qrels=[], retrieved=retrieved(),
                compressed_context=selected, query_formulation=IdentityQueryFormulator().formulate("Which city?"))
            result = runner._run_generation(state)
            visible = result.metadata["generation_context"]["visible_evidence"]
            self.assertEqual([r["text"] for r in visible], ["The answer is Paris."])
            hashes.append(result.metadata["generation_context"]["content_sha256"])
            if mode == "context_grounded":
                self.assertEqual([r.chunk.text for r in captured], ["The answer is Paris."])
            else:
                self.assertEqual(result.citations[0].cited_text, "The answer is Paris.")
        self.assertEqual(hashes[0], hashes[1])

    def test_budget_counts_rendering_and_preserves_visible_provenance(self):
        row = retrieved()[0]
        row = replace(row, chunk=replace(row.chunk, text="word " * 300, sentences=[]))
        for cap in (0, 5, 40, 80):
            result = prepare_context(query="q", retrieved=[row], compressed=None, max_tokens=cap)
            self.assertLessEqual(result.metadata["used_tokens"], cap)
            self.assertEqual([s.text for s in result.evidence.sentences], [c.chunk.text for c in result.contexts])
            if result.evidence.sentences:
                self.assertTrue(result.evidence.sentences[0].metadata["budget_truncated"])


class EvaluationFoundationTest(unittest.TestCase):
    def test_group_support_never_blesses_irrelevant_citation(self):
        claim = ClaimCheckRecord(claim_id="c", claim_text="Paris is in France", citations=[
            CitationRecord(chunk_id="good", cited_text="Paris is in France"),
            CitationRecord(chunk_id="bad", cited_text="Mars is a planet")])
        checked = apply_evidence_judgment(claim, judge=SimpleNamespace(name="test"),
                                         judgment=EvidenceJudgment("supported", "one source supports", .99))
        self.assertEqual(checked.status, "supported")
        self.assertTrue(all(c.support_status == "unjudged" for c in checked.citations))
        metrics = citation_quality_metrics(prediction(claim_checks=[checked], citations=checked.citations))
        self.assertNotIn("citation_precision", metrics)
        self.assertEqual(metrics["citation_judged_count"], 0)

    def test_group_status_is_not_individual_judgment(self):
        metrics = citation_quality_metrics(prediction(citations=[CitationRecord(chunk_id="x", support_status="supported")]))
        self.assertNotIn("citation_precision", metrics)

    def test_empty_claims_are_not_verified_or_scored_as_supported(self):
        result = attach_evaluation_coverage(prediction(metrics={"claim_support_rate": 1.0}))
        self.assertEqual(result.metadata["evaluation_coverage"]["full_answer_claim_coverage"], "unknown")
        self.assertNotIn("claim_support_rate", result.metrics)
        self.assertNotIn("claim_support_rate", evidence_consistency_metrics([]))
        for verifier in (CoVeVerifier(), FactoredCoVeVerifier(judge=CoVeVerifier().judge)):
            checked = verifier.verify(result)
            self.assertEqual(checked.verification.status, "not_run")
            self.assertEqual(checked.metrics["verification_passed"], 0)

    def test_scifact_source_evidence_annotation_is_consumed(self):
        example = Example("ex-1", "scifact", "test", "verification", "Claim", [{"text": "SUPPORT"}], [], {})
        p = prediction(citations=[CitationRecord(chunk_id="doc", sentence_id=1)]).model_copy(update={"answer": "SUPPORT"})
        annotation = Annotation("a", "ex-1", "evidence", "doc", "SUPPORT", None, None, None, ["doc"], {"sentence_ids": [1]})
        metrics = verification_track_metrics(p, example=example,
            qrels=[Qrel("ex-1", "doc", 3, "rationale", {})], annotations=[annotation])
        self.assertEqual(metrics["rationale_sentence_f1"], 1.0)

    def test_incorrect_unattributed_and_unknown_are_not_benign(self):
        incorrect = diagnose_prediction(prediction(metrics={"em": 0.0, "f1": 0.0}))
        self.assertEqual(incorrect.primary_error, "incorrect_unattributed")
        self.assertIsNone(incorrect.signals["claim_support_rate"])
        self.assertTrue(incorrect.needs_human_review)
        self.assertEqual(diagnose_prediction(prediction()).primary_error, "not_evaluated")


class UsageContractsTest(unittest.TestCase):
    def test_raw_backend_enforces_output_cap_and_returns_usage(self):
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"), finish_reason="stop")],
                                                  usage=SimpleNamespace(prompt_tokens=137, completion_tokens=9)))
        generator = object.__new__(OpenAITextPromptGenerator)
        generator.model, generator.max_output_tokens = "test", 17
        generator.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = generator.generate_text_result("question")
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 17)
        self.assertEqual((result.input_tokens, result.output_tokens), (137, 9))
        self.assertEqual(result.usage_source, "api_response")

    def test_actual_zero_usage_is_not_replaced_by_estimate(self):
        raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                              usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0), _rag_benchmark_attempt_count=2)
        result = response_from_api(raw, messages=[{"content": "question"}], model="test")
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(result.usage_source, "api_response")
        self.assertTrue(result.unknown_attempt_usage)

    def test_schema_repair_sums_actual_usage_per_call(self):
        context = compressed_context_from_retrieved(query="q", retrieved=retrieved())
        payload = json.dumps({"answer": "Paris", "claims": [{"text": "Paris", "citations": [context.sentences[1].context_id]}]})
        generator = SimpleNamespace(generate_text_result=Mock(side_effect=[
            TextResponse("invalid", 101, 7, "api_response"), TextResponse(payload, 203, 11, "api_response")]))
        result, _, runtime = generate_citation_answer(question="Which city?", context=context,
            task_track="short_answer", task_type="qa", generator=generator, model="test", strict=True)
        self.assertEqual((result.input_tokens, result.output_tokens), (304, 18))
        self.assertEqual(runtime["format_attempt_count"], 2)

    def test_query_rewriter_preserves_actual_usage(self):
        generator = SimpleNamespace(provider="test", model="test", generate_text_result=lambda prompt:
            TextResponse('{"query":"Paris","rationale":"test"}', 111, 13, "api_response"))
        result = LLMQueryRewriter(generator).formulate("Which city?")
        self.assertEqual((result.input_tokens, result.output_tokens), (111, 13))
        self.assertEqual(result.metadata["call_usage"]["usage_source"], "api_response")

    def test_plain_backend_and_verifier_use_the_same_usage_contract(self):
        raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))],
                              usage=SimpleNamespace(prompt_tokens=81, completion_tokens=6))
        create = Mock(return_value=raw)
        plain = object.__new__(OpenAIGenerator)
        plain.model, plain.max_output_tokens = "test", 19
        plain.input_cost_per_1m_tokens = plain.output_cost_per_1m_tokens = None
        plain.ca_bundle, plain.cost_source = None, "none"
        plain.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = plain.generate(question="city", contexts=retrieved(), pipeline="test")
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 19)
        self.assertEqual((result.input_tokens, result.output_tokens), (81, 6))
        generator = SimpleNamespace(model="test", generate_text_result=Mock(side_effect=[
            TextResponse('{"questions":[{"claim_id":"c","question":"Which city?"}]}', 51, 8, "api_response"),
            TextResponse('{"answers":[{"question_id":"verify-1","answer":"Paris"}]}', 72, 9, "api_response"),
        ]))
        verifier = FactoredCoVeVerifier(judge=CoVeVerifier().judge, generator=generator)
        checked = verifier.verify(prediction(claim_checks=[ClaimCheckRecord(claim_id="c", claim_text="Paris",
            citations=[CitationRecord(chunk_id="d", cited_text="Paris")])]))
        self.assertEqual(checked.verification.metadata["input_tokens"], 123)
        self.assertEqual(checked.verification.metadata["output_tokens"], 17)
        self.assertEqual(checked.verification.metadata["call_usage"]["usage_source"], "api_response")


class ResumeContractsTest(unittest.TestCase):
    def test_no_identity_and_changed_identity_refuse_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = resume_fixtures.ExperimentSuiteResumeTest._write_run(Path(tmp), example_id="ex-1")
            args = dict(run_id="suite-vanilla_rag", pipeline="vanilla_rag", expected_example_ids={"ex-1"})
            self.assertIsNone(load_completed_formal_run(path, **args))
            self.assertIsNone(load_completed_formal_run(path, **args, expected_identity={"generation": "g2", "evaluation": "e1"}))

    def test_same_ids_but_changed_content_or_budget_change_fingerprint(self):
        experiment = make_experiment(make_best_practice_pipeline())
        data = make_dataset_slice()
        settings = FormalRunSettings(save_outputs=False, component_settings=ComponentFactorySettings(generator_kind="extractive"))
        def identity(exp=experiment, dataset=data):
            return run_identity(repo_root=ROOT, experiment=exp, dataset_slice=dataset, settings=settings)
        original = identity()
        changed_data = replace(data, examples=[replace(data.examples[0], question="A different question")])
        self.assertNotEqual(original, identity(dataset=changed_data))
        pipeline = experiment.pipeline.model_copy(update={"budget": experiment.pipeline.budget.model_copy(update={"max_context_tokens": 77})})
        self.assertNotEqual(original, identity(exp=experiment.model_copy(update={"pipeline": pipeline})))


    def test_existing_output_refuses_before_generator_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ConfigDrivenPipelineRunner(repo_root=ROOT,
                experiment=make_experiment(make_best_practice_pipeline(), outputs_root=Path(tmp)),
                dataset_slice=make_dataset_slice(), settings=FormalRunSettings(run_id="existing",
                    component_settings=ComponentFactorySettings(generator_kind="extractive")))
            runner.output_dir().mkdir(parents=True)
            runner.base_generator = SimpleNamespace(generate=Mock(side_effect=AssertionError("must not call")))
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.run()
            runner.base_generator.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
