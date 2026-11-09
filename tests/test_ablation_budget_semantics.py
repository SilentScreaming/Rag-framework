"""Ablation consistency, generation budgets, and evidence semantics."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rag_benchmark.ablation import build_default_ablation_steps, build_leave_one_out_steps, run_ablation_suite
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.data import Example, ContextSet
from rag_benchmark.data_semantics import garage_semantics, correct_garage_views
from rag_benchmark.evaluation.garage import GarageEvaluator, aggregate_garage_judgments
from rag_benchmark.evaluation.service import EvaluationService
from rag_benchmark.generation.budget import CallBudget, BudgetExhausted, execution_budget, with_shared_output_cap, output_token_limit
from rag_benchmark.generation.backends import create_chat_completion_with_retries
from rag_benchmark.generation.usage import TextResponse
from rag_benchmark.sampling import make_sample_manifest, load_sample_manifest
from rag_benchmark.run_summary import summarize_run
from rag_benchmark.runner import FormalRunSettings, adapt_pipeline_to_dataset, ConfigDrivenPipelineRunner
from rag_benchmark.schemas import PredictionRecord
from rag_benchmark.stage_cache import canonical_hash
from test_ablation_runner import make_experiment, make_dataset_slice, REPO_ROOT


class Backend:
    model = "fixture"
    max_output_tokens = 4096

    def __init__(self, values, finish="stop"):
        self.values = iter(values)
        self.finish = finish
        self.prompts = []

    def generate_text_result(self, prompt):
        self.prompts.append(prompt)
        return TextResponse(json.dumps(next(self.values)), 10, 10, "fixture", finish_reason=self.finish)


class AblationBudgetSemanticsTest(unittest.TestCase):
    def full(self):
        full = build_default_ablation_steps()[5].pipeline
        return full.model_copy(update={"retrieval": {**full.retrieval, "iterative_hops": 2},
                                       "compression": {**full.compression, "semantic_scoring": True}})

    def test_six_configurations_change_only_intended_component(self):
        steps = build_leave_one_out_steps(full=self.full())
        self.assertEqual(len(steps), 6)
        for step in steps:
            self.assertEqual(step.pipeline.stages.verification, "none")
            difference = [k for k,v in steps[0].pipeline.stages.model_dump().items() if getattr(step.pipeline.stages,k) != v]
            removed = step.pipeline.logging["removed_component"]
            self.assertEqual(difference, [removed] if removed else [])
            self.assertEqual(step.pipeline.budget, steps[0].pipeline.budget)
        self.assertEqual(steps[1].pipeline.retrieval["iterative_hops"], 1)
        self.assertFalse(steps[-1].pipeline.task_profile.citations_required)

    def test_leave_one_out_runs_preserve_each_variant(self):
        with tempfile.TemporaryDirectory() as d:
            exp = make_experiment(outputs_root=Path(d))
            full = self.full().model_copy(update={"reranking": {"top_k": 1}, "compression": {"semantic_scoring": True}})
            result = run_ablation_suite(repo_root=REPO_ROOT, base_experiment=exp, dataset_slice=make_dataset_slice(),
                settings=FormalRunSettings(split_profile="mvp", limit=1, output_root=Path(d), bootstrap_samples=0,
                    component_settings=ComponentFactorySettings(generator_kind="extractive", query_rewriter="rule")),
                ablation_id="loo", steps=build_leave_one_out_steps(full=full))
            self.assertEqual(len(result.step_results), 6)
            self.assertEqual(len(result.delta_rows), 5)
            for run in result.step_results:
                self.assertEqual(len(run.predictions[0].retrieved_chunks), 1)
                self.assertIn("effective_output_budget", run.predictions[0].metadata)
            self.assertFalse(result.step_results[-1].predictions[0].generator.get("schema_failure", False))

    def test_removing_reranker_preserves_compressor_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            exp = make_experiment(outputs_root=Path(d)).model_copy(update={"pipeline": build_leave_one_out_steps(full=self.full())[3].pipeline})
            runner = ConfigDrivenPipelineRunner(repo_root=REPO_ROOT, experiment=exp, dataset_slice=make_dataset_slice(),
                settings=FormalRunSettings(save_outputs=False, component_settings=ComponentFactorySettings(generator_kind="extractive")))
            self.assertIsNone(runner.reranker)
            self.assertIsNotNone(runner.compressor.semantic_scorer)

    def test_shared_cap_survives_repeated_dataset_adaptation(self):
        exp = make_experiment(outputs_root=(REPO_ROOT / "outputs"))
        changed = with_shared_output_cap(exp, 2345)
        adapted = adapt_pipeline_to_dataset(changed.pipeline, exp.dataset)
        self.assertEqual(output_token_limit(adapted, "plain"), 2345)
        self.assertEqual(output_token_limit(adapted, "citation_json"), 2345)
        self.assertNotEqual(exp.pipeline.budget.max_generation_tokens, 2345)

    def test_sample_manifest_is_random_reproducible_and_detects_source_drift(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "formal.jsonl"
            path.write_text("".join(json.dumps({"example_id": str(i)})+"\n" for i in range(101)))
            manifest = make_sample_manifest(dataset="fixture", data_version="v1", split="formal", split_path=path)
            self.assertEqual(len(manifest["sample_ids"]), 6)
            self.assertNotEqual(manifest["sample_ids"], [str(i) for i in range(6)])
            self.assertFalse(manifest["retrieval_corpus_subsampled"])
            saved = Path(d) / "sample.json"; saved.write_text(json.dumps(manifest))
            self.assertEqual(load_sample_manifest(saved, dataset="fixture", split_path=path, seed=42), manifest)
            path.write_text(path.read_text()+json.dumps({"example_id": "new"})+"\n")
            with self.assertRaises(ValueError):
                load_sample_manifest(saved, dataset="fixture", split_path=path)

    def test_response_sampling_does_not_collapse_different_preset_answers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "judge_eval.jsonl"
            path.write_text("".join(json.dumps({"example_id": "q", "response_id": str(i)})+"\n" for i in range(40)))
            result = make_sample_manifest(dataset="ragtruth_qa", data_version="v1", split="judge_eval", split_path=path)
            self.assertEqual(result["unit"], "response_id")
            self.assertEqual(len(result["sample_ids"]), 2)

    def test_garage_missing_direct_answer_label_is_unresolved(self):
        self.assertEqual(garage_semantics({"evidence_correct": ["RELATED-INFORMATION"]})["expected_behavior"], "unresolved")
        self.assertEqual(garage_semantics({"evidence_correct": ["ANSWER-THE-QUESTION"]})["expected_behavior"], "answer")

    def test_loader_corrects_abstention_type_without_altering_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"raw").mkdir()
            raw={"sample_id": "s", "evidence_correct": ["RELATED-INFORMATION"], "answer_generate": "Some answer"}
            path=root/"raw/GaRAGe_benchmark.jsonl";path.write_text(json.dumps(raw)+"\n\n")
            original=path.read_bytes()
            example=Example("e", "garage", "all", "long", "Q", [{"text":"Some answer","type":"abstention"}], [], {"source_id":"s"})
            e,c=correct_garage_views(root,{"e":example},{})
            self.assertEqual(e["e"].metadata["expected_behavior"],"unresolved")
            self.assertEqual(e["e"].answers[0]["type"],"long")
            self.assertEqual(e["e"].answers[0]["metadata"]["original_type"],"abstention")
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):
                correct_garage_views(root,{"e":example},{},reviews={"e":{"raw_annotation_hash":"wrong"}})

    def test_unknown_retry_consumes_budget_and_prevents_next_dispatch(self):
        create = Mock(side_effect=TimeoutError("fixture timeout"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch("rag_benchmark.generation.backends.time.sleep"):
            with execution_budget(max_calls=1) as budget:
                with self.assertRaises(BudgetExhausted):
                    create_chat_completion_with_retries(
                        client,
                        model="fixture",
                        messages=[{"role": "user", "content": "hi"}],
                        temperature=0,
                        max_output_tokens=100,
                    )
        self.assertEqual(create.call_count, 1)
        self.assertTrue(budget.snapshot()["unknown_usage"])
        self.assertGreater(budget.calls[0]["charged_tokens"], 0)


    def test_total_token_budget_reserves_full_next_call(self):
        budget = CallBudget(max_total_tokens=30)
        row = budget.reserve(input_tokens=10, output_cap=15, model="fixture")
        budget.finish(
            row,
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
                choices=[],
            ),
        )
        with self.assertRaises(BudgetExhausted):
            budget.reserve(input_tokens=10, output_cap=15, model="fixture")
        self.assertEqual(budget.calls[0]["charged_tokens"], 14)

    def test_truncation_and_plain_answers_are_not_format_failures(self):
        def p(gen):
            return PredictionRecord(example_id="e",pipeline="p",question="q",answer="a",generator=gen)
        result=summarize_run([p({"output_format":"citation_json","schema_failure":True,"failure_kind":"truncated"}),
                                p({"output_format":"citation_json","schema_failure":True,"failure_kind":"format"}),p({})])
        self.assertEqual(result["format_required_count"],2)
        self.assertEqual(result["final_format_failure_rate"],.5)
        self.assertEqual(result["truncated_answer_count"],1)

    def test_raf_is_joint_per_answer_not_product_of_average_scores(self):
        a=aggregate_garage_judgments("no_issues",[{"label":"unsupported"}])
        b=aggregate_garage_judgments("major_issues",[{"label":"supported"}])
        self.assertEqual((a["raf"]+b["raf"])/2,0)
        self.assertEqual((a["garage_eligibility"]+b["garage_eligibility"])*(a["uraf"]+b["uraf"])/4,.25)

    def test_garage_judgments_are_separate_and_uncalibrated(self):
        backend=Backend([{"label":"minor_issues"},{"sentences":[{"segment_id":"s1","label":"supported"}]}])
        result=GarageEvaluator(backend).evaluate(question="Q",answer="A fact.",reference_answers=["A fact."],relevant_evidence=["A fact."])
        self.assertEqual(result["development_values"]["raf"],1)
        self.assertIsNone(result["metrics"]["raf"]["value"])
        self.assertEqual(result["status"],"uncalibrated")
        self.assertNotIn('human_reference',backend.prompts[1])

    def test_garage_empty_missing_and_incomplete_outputs_never_pass(self):
        evaluator=GarageEvaluator(Backend([]))
        self.assertEqual(evaluator.evaluate(question="Q",answer="",reference_answers=["A"],relevant_evidence=[])["status"],"not_applicable")
        self.assertEqual(evaluator.evaluate(question="Q",answer="A",reference_answers=["A"],relevant_evidence=None)["status"],"missing_input")
        evaluator=GarageEvaluator(Backend([{"label":"no_issues"}],finish="length"))
        self.assertEqual(evaluator.evaluate(question="Q",answer="A",reference_answers=["A"],relevant_evidence=[])["status"],"failed")

    def test_failed_generator_skips_garage_judge(self):
        judge=Mock()
        example=Example("e","garage","all","long","Q",[{"text":"A","type":"long"}],[],{})
        p=PredictionRecord(example_id="e",pipeline="p",question="Q",answer="fallback",generator={"schema_failure":True})
        EvaluationService(garage_evaluator=judge).evaluate(p,example=example,garage_relevant_evidence=["A"])
        judge.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
