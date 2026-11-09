"""Execution identity, usage accounting, and evidence-boundary contracts."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_benchmark.data import Example, CorpusChunk
from rag_benchmark.schemas import PredictionRecord, CitationRecord, ClaimCheckRecord
from rag_benchmark.evaluation.identity import EvaluationIdentity, prediction_identity
from rag_benchmark.evaluation.statistics import paired_comparisons
from rag_benchmark.evaluation.service import EvaluationService
from rag_benchmark.evaluation.claims import parse_extraction
from rag_benchmark.evaluation.attribution import align_citations
from rag_benchmark.evaluation.answers import grouped_answer_metrics
from rag_benchmark.verification.replay import validate_drafts
from rag_benchmark.verification.task_revision import TaskRevisionVerifier, TaskRevisionConfig, parse_list
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.generation.context import prepare_context
from rag_benchmark.generation.backends import build_prompt, create_chat_completion_with_retries
from rag_benchmark.generation.service import generate_citation_answer
from rag_benchmark.generation.usage import TextResponse, response_from_api
from rag_benchmark.retrieval.bm25 import RetrievedChunk


def example(dataset="scifact", task="scientific_claim_verification", answers=None, metadata=None):
    return Example("e1", dataset, "dev", task, "Question?", answers or [], [], metadata or {})


def prediction(answer="SUPPORT", dataset="scifact", **kwargs):
    return PredictionRecord(example_id="e1", pipeline="p", question="Question?", answer=answer,
        metadata={**EvaluationIdentity(dataset, "fixture-v1", "dev", 42).metadata(), **kwargs.pop("metadata", {})}, **kwargs)


class ExecutionContractsTest(unittest.TestCase):
    def test_formal_aggregate_does_not_restore_partial_cost_as_total(self):
        from test_config_driven_runner import make_dataset_slice, make_experiment, make_closed_book_pipeline, REPO_ROOT
        from rag_benchmark.runner import ConfigDrivenPipelineRunner, FormalRunSettings
        from rag_benchmark.component_factory import ComponentFactorySettings
        from rag_benchmark.generation.backends import GenerationResult
        runner = ConfigDrivenPipelineRunner(repo_root=REPO_ROOT, experiment=make_experiment(make_closed_book_pipeline()),
            dataset_slice=make_dataset_slice(), settings=FormalRunSettings(save_outputs=False, bootstrap_samples=0,
                component_settings=ComponentFactorySettings(generator_kind="extractive")))
        runner.base_generator.generate = lambda **_: GenerationResult("Ada Lovelace", 1, 7, 2, 0.1,
                                                                       {"unknown_attempt_usage": True})
        result = runner.run()
        self.assertEqual(result.cost_latency["known_estimated_cost"], 0.1)
        self.assertNotIn("total_estimated_cost", result.metrics["pipelines"]["closed_book"])
        self.assertEqual(result.metrics["cost_status"], "partial_unknown_attempt_usage")

    def test_replay_cli_accepts_explicit_seed_and_context_selection(self):
        from scripts.run_verification_replay import parse_args
        args = parse_args(["--dataset", "rgb", "--split-profile", "dev", "--limit", "1",
            "--draft-predictions", "draft.jsonl", "--output-root", "out", "--run-id", "replay",
            "--seed", "7", "--context-condition", "noise", "--context-metadata", '{"noise_rate":0.5}'])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.context_metadata, {"noise_rate": 0.5})


    def test_scoped_closed_book_and_oracle_keep_condition_identity(self):
        from test_config_driven_runner import make_dataset_slice, make_experiment, make_closed_book_pipeline, REPO_ROOT
        from rag_benchmark.data import ContextSet
        from rag_benchmark.runner import ConfigDrivenPipelineRunner, FormalRunSettings
        from rag_benchmark.component_factory import ComponentFactorySettings
        data = make_dataset_slice()
        data.context_sets_by_example["ex-1"] = [ContextSet("cx", "ex-1", "synthetic", "provided",
                                                        ["chunk-gold"], "answer", {})]
        for retrieval in ("none", "oracle"):
            pipe = make_closed_book_pipeline()
            pipe = pipe.model_copy(update={"stages": pipe.stages.model_copy(update={"retrieval": retrieval})})
            out = ConfigDrivenPipelineRunner(repo_root=REPO_ROOT, experiment=make_experiment(pipe),
                dataset_slice=data, settings=FormalRunSettings(save_outputs=False, bootstrap_samples=0,
                    component_settings=ComponentFactorySettings(generator_kind="extractive"))).run()
            self.assertEqual(prediction_identity(out.predictions[0]).context_set_id, "cx")

    def test_explicit_overlapping_spans_stay_unknown(self):
        p = prediction("Alice won [1] and Bob lost [2].", citations=[
            CitationRecord(citation_id="1", chunk_id="a"), CitationRecord(citation_id="2", chunk_id="b")])
        extracted = parse_extraction({"segments": [{"segment_id": "s1", "kind": "factual", "claims": [
            {"text": "Alice won", "source_quote": p.answer}, {"text": "Bob lost", "source_quote": p.answer}]}]},
            question=p.question, answer=p.answer)
        links, _ = align_citations(p, extracted.claims)
        self.assertTrue(all(v["status"] == "unresolved" for v in links.values()))
        self.assertTrue(all(not v["citations"] for v in links.values()))

    def test_nonoverlapping_source_quotes_resolve_locally(self):
        p = prediction("Alice won [1] and Bob lost [2].", citations=[
            CitationRecord(citation_id="1", chunk_id="a"), CitationRecord(citation_id="2", chunk_id="b")])
        extracted = parse_extraction({"segments": [{"segment_id": "s1", "kind": "factual", "claims": [
            {"text": "Alice won", "source_quote": "Alice won [1]"}, {"text": "Bob lost", "source_quote": "Bob lost [2]"}]}]},
            question=p.question, answer=p.answer)
        links, _ = align_citations(p, extracted.claims)
        self.assertEqual([[c.citation_id for c in v["citations"]] for v in links.values()], [["1"], ["2"]])

    def test_missing_answers_do_not_create_kilt_zero_score(self):
        e = example("kilt_nq", "short_answer")
        q = SimpleNamespace(chunk_id="d", relevance=3, metadata={})
        result = EvaluationService().evaluate(prediction("x", "kilt_nq"), example=e, qrels=[q])
        self.assertEqual(result["metrics"]["em"]["status"], "missing_input")
        self.assertNotIn("kilt_em", result["metrics"])

    def test_ragtruth_explicitly_requires_dedicated_entry(self):
        result = EvaluationService().evaluate(prediction("x", "ragtruth"), example=example("ragtruth"))
        self.assertEqual(result["metrics"]["hallucination_span_f1"]["status"], "unsupported")

    def test_rgb_refusal_is_scored_against_selected_instance(self):
        from rag_benchmark.evaluation.service import evaluated_prediction
        from rag_benchmark.evaluation.statistics import metric_value
        e = example("rgb", "rag_robustness_stress_test", [{"text": "Paris"}], {"expected_behavior": "answer"})
        p = prediction("UNKNOWN", "rgb", metadata={"context_set_id": "empty"})
        ctx = SimpleNamespace(context_set_id="empty", example_id="e1", dataset="rgb", condition="empty", expected_behavior="abstain",
                              metadata={"insufficiency_analysis_eligible": True, "evidence_review_status": "insufficient"})
        result = EvaluationService().evaluate(p, example=e, context_set=ctx)
        self.assertTrue(result["task"]["task_correct"])
        self.assertEqual(result["metrics"]["em"]["status"], "not_applicable")
        view = evaluated_prediction(p, result)
        self.assertEqual(metric_value([view], "deflection_recall", examples_by_id={"e1": e}), 1)

    def test_all_missing_and_all_failure_statuses_remain_distinct(self):
        from rag_benchmark.evaluation.contracts import aggregate_metric_results
        for status in ("failed", "missing_input", "unsupported", "not_applicable"):
            result = aggregate_metric_results([{"metrics": {"x": {"status": status, "value": None}}}])["x"]
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["value"])

    def test_previously_truncated_unit_never_becomes_complete(self):
        from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
        unit = CompressedSentence("c", "d", "doc", None, 0, "partial", 1, 1, 1,
                                  {"budget_truncated": True, "fully_visible": False, "source_text_sha256": "original"})
        result = prepare_context(query="q", retrieved=[], compressed=CompressedContext("q", [unit]), max_tokens=100)
        self.assertFalse(result.evidence.sentences[0].metadata["fully_visible"])
        self.assertEqual(result.evidence.sentences[0].metadata["source_text_sha256"], "original")

    def test_recovered_retry_is_not_complete_usage(self):
        import httpx
        from openai import OpenAI
        requests = []
        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(500, json={"error": {"message": "synthetic"}})
            return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 1,
                "model": "fixture", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-key"}), httpx.Client(
            transport=httpx.MockTransport(handler)
        ) as http:
            with patch("rag_benchmark.generation.backends.time.sleep"):
                response = create_chat_completion_with_retries(OpenAI(http_client=http),
                    model="fixture", messages=[], temperature=0)
        measured = response_from_api(response, messages=[], model="fixture")
        self.assertEqual((len(requests), measured.attempts, measured.input_tokens), (2, 2, 7))
        self.assertTrue(measured.unknown_attempt_usage)
        self.assertEqual(len(measured.attempt_records), 2)

    def test_format_repair_failure_preserves_prior_success_usage(self):
        from rag_benchmark.compression.sentence_select import CompressedContext
        calls = []
        def generate(_):
            calls.append(1)
            if len(calls) == 1:
                return TextResponse("bad json", 11, 3, "fixture")
            exc = RuntimeError("synthetic")
            exc._rag_benchmark_usage = {"attempt_count": 2, "unknown_attempt_usage": True}
            raise exc
        with self.assertRaises(RuntimeError) as raised:
            generate_citation_answer(question="q", context=CompressedContext("q", []), task_track="short_answer",
                task_type="short_answer", generator=SimpleNamespace(generate_text_result=generate), model="fixture", strict=True)
        usage = raised.exception._rag_benchmark_usage
        self.assertEqual((usage["input_tokens"], usage["output_tokens"], usage["attempt_count"]), (11, 3, 3))
        self.assertTrue(usage["unknown_attempt_usage"])

    def test_unknown_attempt_cost_is_not_reported_as_complete(self):
        from rag_benchmark.artifacts import summarize_cost_latency
        p = prediction(generator={"unknown_attempt_usage": True})
        p.cost.estimated_cost = 0.1
        cost = summarize_cost_latency([p])
        self.assertEqual(cost["known_estimated_cost"], 0.1)
        self.assertIsNone(cost["total_estimated_cost"])

    def test_rehashed_invalid_snapshot_cost_or_trace_is_rejected(self):
        from test_experiment_suite_resume import ExperimentSuiteResumeTest
        from rag_benchmark.artifacts import load_completed_formal_run
        from hashlib import sha256
        for name, content in (("config.snapshot.yaml", "run_id: suite-vanilla_rag\nexperiment: {}\n"),
                              ("cost_latency.json", '{"num_predictions":1}'),
                              ("stage_traces.jsonl", '{"example_id":"unknown","stage":"x"}\n')):
            with tempfile.TemporaryDirectory() as tmp:
                root = ExperimentSuiteResumeTest._write_run(Path(tmp), example_id="ex-1")
                path = root / name
                path.write_text(content)
                manifest_path = root / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["outputs"][name] = sha256(path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                self.assertIsNone(load_completed_formal_run(root, run_id="suite-vanilla_rag", pipeline="vanilla_rag",
                    expected_example_ids={"ex-1"}, expected_identity={"generation": "g1", "evaluation": "e1"}))

    def test_mode_budget_never_silently_raises_global_cap(self):
        from test_config_driven_runner import make_closed_book_pipeline
        from rag_benchmark.generation.budget import output_token_limit
        p = make_closed_book_pipeline()
        p = p.model_copy(update={"generation": {"output_tokens_by_mode": {"plain": 16, "citation_json": 32}},
                                 "budget": p.budget.model_copy(update={"max_generation_tokens": 32})})
        self.assertEqual(output_token_limit(p, "plain"), 16)
        self.assertEqual(output_token_limit(p, "citation_json"), 32)
        p.generation["output_tokens_by_mode"]["citation_json"] = 64
        with self.assertRaises(ValueError):
            output_token_limit(p, "citation_json")

    def test_default_stratum_is_not_context_identity(self):
        p = prediction(metadata={"metric_strata": {"context_condition": "default"}})
        validate_drafts([p], [example()], expected_track="short_answer", data_version="fixture-v1")
        p.metadata["context_set_id"] = "concrete-1"
        validate_drafts([p], [example()], expected_track="short_answer", data_version="fixture-v1",
                        context_ids={"e1": "concrete-1"})
        with self.assertRaises(ValueError):
            validate_drafts([p], [example()], expected_track="short_answer")

    def test_seed_version_and_context_cannot_cross_pair(self):
        a = prediction(metrics={"f1": 1})
        for field, value in (("seed", 43), ("data_version", "other"), ("context_set_id", "other")):
            b = a.model_copy(deep=True)
            b.metadata[field] = value
            with self.assertRaises(ValueError):
                paired_comparisons([a], [b], keys=["f1"])
        with self.assertRaises(ValueError):
            prediction_identity(a.model_copy(update={"metadata": {}}))

    def test_supported_claim_is_not_support_for_wrong_label(self):
        judge = SimpleNamespace(judge_joint=lambda *_: EvidenceDecision("completed", "supported"))
        p = prediction("CONTRADICT", metadata={"generation_context": {"visible_evidence": [
            {"chunk_id": "d", "sentence_id": 0, "text": "evidence", "metadata": {"fully_visible": True}}]}})
        for cfg in (TaskRevisionConfig(policy="detect_only"), TaskRevisionConfig(max_edits=0)):
            out = TaskRevisionVerifier(judge=judge, config=cfg).verify(p, task_type="scientific_claim_verification")
            self.assertEqual(out.answer, "CONTRADICT")
            self.assertEqual(out.verification.status, "failed")
            self.assertEqual(out.verification.metadata["execution_status"], "completed")

    def test_compound_sentence_does_not_fan_out_citations(self):
        p = prediction("Alice is a member [1] and Bob is a member [2].")
        p.citations = [CitationRecord(citation_id=str(i), chunk_id=str(i), cited_text=name)
                       for i, name in ((1, "Alice"), (2, "Bob"))]
        p.claim_checks = [ClaimCheckRecord(claim_id=name, claim_text=name+" is a member", citations=[c])
                          for name, c in zip(("Alice", "Bob"), p.citations)]
        extraction = parse_extraction({"segments": [{"segment_id": "s1", "kind": "factual",
            "claims": ["Alice is a member", "Bob is a member"]}]}, question=p.question, answer=p.answer)
        links, _ = align_citations(p, extraction.claims)
        self.assertEqual([[c.citation_id for c in v["citations"]] for v in links.values()], [["1"], ["2"]])
        p.claim_checks = []
        links, _ = align_citations(p, extraction.claims)
        self.assertTrue(all(v["status"] == "unresolved" for v in links.values()))

    def test_same_serialized_evidence_for_both_generation_modes(self):
        chunk = CorpusChunk("very-long-" * 100, "doc", "scifact", "Study",
            "Background. " * 200, [], {})
        prepared = prepare_context(query="q", retrieved=[RetrievedChunk(chunk, 1, 1)], compressed=None, max_tokens=80)
        prompt = build_prompt(question="q", contexts=list(prepared.contexts), pipeline="vanilla_rag")
        rendered = prompt.split("Context:\n", 1)[1].rsplit("\n\nAnswer:", 1)[0]
        self.assertEqual(rendered, prepared.evidence.to_prompt_context())
        self.assertLessEqual(prepared.metadata["used_tokens"], 80)

    def test_garage_decision_is_not_answer_correctness(self):
        e = example("garage", "grounding_and_selective_refusal", [{"text": "Paris"}], {"expected_behavior": "answer"})
        from rag_benchmark.evaluation.claims import ExtractionResult
        extractor = SimpleNamespace(identity={"fixture": "answered-behavior"}, extract=lambda **kwargs:
            ExtractionResult("completed", answer_behavior={"status": "computed", "label": "answered"}))
        result = EvaluationService(extractor=extractor).evaluate(prediction("London", "garage", task_track="verification_hallucination"), example=e)
        self.assertEqual(result["metrics"]["eligibility_accuracy"]["value"], 1)
        self.assertEqual(result["metrics"]["em"]["value"], 0)
        self.assertFalse(result["task"]["task_correct"])
        self.assertEqual(result["metrics"]["raf"]["status"], "unsupported")

    def test_rgb_uses_exact_context_expectation(self):
        e = example("rgb", "rag_robustness_stress_test", [{"text": "Paris"}], {"expected_behavior": "answer"})
        p = prediction("UNKNOWN", "rgb", metadata={"context_set_id": "no-evidence"})
        ctx = SimpleNamespace(context_set_id="no-evidence", example_id="e1", dataset="rgb",
                              expected_behavior="abstain", condition="noise",
                              metadata={"insufficiency_analysis_eligible": True, "evidence_review_status": "insufficient"})
        result = EvaluationService().evaluate(p, example=e, context_set=ctx)
        self.assertEqual(result["metrics"]["eligibility_accuracy"]["value"], 1)
        self.assertEqual(result["task"]["expected_behavior"], "abstain")

    def test_quoted_comma_entity_and_citation_adjacent_alias(self):
        answer = '"Washington, D.C." [1], Paris [2]'
        self.assertEqual([i.text for i in parse_list(answer)], ["Washington, D.C.", "Paris"])
        gold = [{"text": text, "type": "list_item", "metadata": {"item_index": i}}
                for i, text in enumerate(("Washington, D.C.", "Paris"))]
        self.assertEqual(grouped_answer_metrics(answer, gold, kind="qampari")["qampari_unique_entity_recall"], 1)
        gold = [{"text": "Paris", "type": "short", "metadata": {"qa_pair_index": 0}}]
        self.assertEqual(grouped_answer_metrics("Paris[1]", gold, kind="asqa")["asqa_group_recall_strict"], 1)

    def test_partial_sentence_id_is_not_complete_rationale(self):
        from rag_benchmark.evaluation.tracks import verification_track_metrics
        p = prediction(citations=[CitationRecord(citation_id="c", chunk_id="d", sentence_id=0,
            cited_text="Background", metadata={"fully_visible": False, "budget_truncated": True})])
        q = SimpleNamespace(chunk_id="d", relevance=1, metadata={"sentence_ids": [0]})
        a = SimpleNamespace(target_type="sentence", target_id="d", metadata={"sentence_ids": [0]})
        result = verification_track_metrics(p, example=example(answers=[{"text": "SUPPORT"}]), qrels=[q], annotations=[a])
        self.assertEqual(result["rationale_source_id_recall"], 1)
        self.assertEqual(result["rationale_visible_complete_recall"], 0)
        p.citations[0].metadata.clear()
        result = verification_track_metrics(p, example=example(answers=[{"text": "SUPPORT"}]), qrels=[q], annotations=[a])
        self.assertNotIn("rationale_visible_complete_recall", result)

    def test_single_retry_layer_and_failed_attempt_ledger(self):
        import httpx
        from openai import OpenAI
        count = []
        def handler(request):
            count.append(request)
            return httpx.Response(500, json={"error": {"message": "synthetic", "type": "server_error"}})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-key"}), httpx.Client(
            transport=httpx.MockTransport(handler)
        ) as http:
            client = OpenAI(http_client=http)
            with patch("rag_benchmark.generation.backends.time.sleep"):
                with self.assertRaises(Exception) as raised:
                    create_chat_completion_with_retries(client, model="fixture", messages=[], temperature=0, max_attempts=2)
            self.assertEqual(len(count), 2)
            usage = raised.exception._rag_benchmark_usage
            self.assertEqual(usage["attempt_count"], 2)
            self.assertIsNone(usage["input_tokens"])
            self.assertTrue(usage["unknown_attempt_usage"])

    def test_modified_or_semantically_inconsistent_outputs_do_not_resume(self):
        from test_experiment_suite_resume import ExperimentSuiteResumeTest
        from rag_benchmark.artifacts import load_completed_formal_run
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as tmp:
            directory = ExperimentSuiteResumeTest._write_run(Path(tmp), example_id="ex-1")
            path = directory / "metrics.json"
            metrics = json.loads(path.read_text())
            metrics["pipelines"]["vanilla_rag"]["f1"] = 0
            path.write_text(json.dumps(metrics))
            args = dict(run_id="suite-vanilla_rag", pipeline="vanilla_rag", expected_example_ids={"ex-1"},
                        expected_identity={"generation": "g1", "evaluation": "e1"})
            self.assertIsNone(load_completed_formal_run(directory, **args))
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["outputs"]["metrics.json"] = sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            self.assertIsNone(load_completed_formal_run(directory, **args))

    def test_valid_json_with_length_finish_is_failure_not_abstention(self):
        from rag_benchmark.compression.sentence_select import CompressedContext
        gen = SimpleNamespace(generate_text_result=lambda _: TextResponse(
            '{"answer":"SUPPORT","claims":[]}', 10, 32, "fixture", finish_reason="length"))
        output, _, runtime = generate_citation_answer(question="q", context=CompressedContext("q", []),
            task_track="verification_hallucination", task_type="scientific_claim_verification",
            generator=gen, model="fixture", strict=True)
        self.assertTrue(runtime["schema_failure"])
        self.assertEqual(runtime["format_attempt_count"], 1)
        p = prediction(output.answer, generator=output.generator)
        result = EvaluationService().evaluate(p, example=example(answers=[{"text": "SUPPORT"}]))
        self.assertEqual(result["task"]["status"], "failed")
        self.assertIsNone(result["task"]["task_correct"])


if __name__ == "__main__":
    unittest.main()
