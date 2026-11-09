"""Controlled intervention regressions; all components are offline fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_ablation_runner import make_dataset_slice, make_experiment, REPO_ROOT
from rag_benchmark.ablation import build_leave_one_out_steps, run_ablation_suite
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.config import ConfigLoader
from rag_benchmark.execution_types import FormalRunSettings
from rag_benchmark.generation.budget import execution_budget, record_reused_calls, BudgetExhausted
from rag_benchmark.paired_artifacts import PairedArtifacts
from rag_benchmark.paired_artifacts import validate_detection_source
from rag_benchmark.schemas import PredictionRecord, VerificationRecord
from rag_benchmark.evaluation.identity import EvaluationIdentity
from rag_benchmark.artifacts import load_completed_formal_run
from rag_benchmark.planning import resolve_pipeline_with_base, build_experiment_with_resolved_pipeline
from rag_benchmark.query_formulation import QueryFormulationResult
from rag_benchmark.query_formulation.rewrite import IdentityQueryFormulator


class PairedInterventionTests(unittest.TestCase):
    def test_six_arms_share_only_unaffected_prefixes_and_visible_evidence(self):
        calls = []
        class ChangingRewriter:
            def formulate(self, question):
                calls.append(question)
                return QueryFormulationResult(original_query=question, query=question,
                    queries=(question, f"different rewrite {len(calls)}"), method="fixture",
                    input_tokens=11, output_tokens=3, estimated_cost=0.01)
        def formulator(*, stage, **kwargs):
            return ChangingRewriter() if stage == "multi_query_rewrite" else IdentityQueryFormulator()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiment = make_experiment(outputs_root=root)
            full = resolve_pipeline_with_base(ConfigLoader(REPO_ROOT), "best_practice_rag")
            with patch("rag_benchmark.runner.build_query_formulator", side_effect=formulator), \
                 patch("rag_benchmark.runner.source_fingerprint", return_value="fixture-implementation"):
                result = run_ablation_suite(repo_root=REPO_ROOT, base_experiment=experiment,
                    dataset_slice=make_dataset_slice(), ablation_id="controlled",
                    steps=build_leave_one_out_steps(full=full),
                    settings=FormalRunSettings(output_root=root, bootstrap_samples=0,
                        component_settings=ComponentFactorySettings(generator_kind="extractive", query_rewriter="rule")))
            rows = [r.predictions[0] for r in result.step_results]
            self.assertEqual(len(calls), 1)
            for row in rows[2:]:
                self.assertIn("query_formulation", row.metadata["paired_stage_reuse"])
            self.assertNotIn("retrieval", rows[2].metadata["paired_stage_reuse"])
            self.assertIn("retrieval", rows[3].metadata["paired_stage_reuse"])
            self.assertIn("reranking", rows[4].metadata["paired_stage_reuse"])
            self.assertIn("compression", rows[5].metadata["paired_stage_reuse"])
            self.assertEqual(rows[0].metadata["generation_context"]["content_sha256"],
                             rows[5].metadata["generation_context"]["content_sha256"])
            self.assertEqual(rows[0].metadata["query_formulation"]["queries"],
                             rows[3].metadata["query_formulation"]["queries"])
            self.assertNotIn("iterative_hop_queries", rows[1].metadata["query_formulation"]["metadata"])
            self.assertEqual(rows[0].metadata["query_formulation"]["input_tokens"],
                             rows[5].metadata["query_formulation"]["input_tokens"])
            self.assertEqual(rows[5].metadata["latency_measurement"],
                             "shared_stage_observations_plus_current_suffix_estimate")
            full_run = result.step_results[0]
            load_args = dict(run_id=full_run.run_id, pipeline=rows[0].pipeline,
                             expected_example_ids={rows[0].example_id}, expected_identity=full_run.manifest["run_identity"])
            self.assertIsNotNone(load_completed_formal_run(full_run.output_dir, **load_args))
            anchor = rows[0].metadata["paired_stage_artifacts"]["query_formulation"]
            Path(anchor["artifact_path"]).unlink()
            self.assertIsNone(load_completed_formal_run(full_run.output_dir, **load_args))

    def test_exact_artifacts_reject_changed_configuration_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            store = PairedArtifacts(Path(folder), {"corpus": "v1"})
            args = dict(example_id="e", stage="retrieval", inputs={"q": "a"}, config={"mode": "hybrid"})
            artifact = store.save(**args, payload={"results": []}, source_run_id="full", latency_ms=5)
            self.assertIsNotNone(store.load(**args))
            self.assertIsNone(store.load(**{**args, "config": {"mode": "dense"}}))
            row = json.loads(artifact.path.read_text(encoding="utf-8"))
            row["payload"]["results"] = ["changed"]
            artifact.path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                store.load(**args)

    def test_reused_prefix_still_consumes_complete_method_budget(self):
        with execution_budget(max_calls=1, max_total_tokens=100) as budget:
            record_reused_calls({"calls": [{"input_tokens": 10, "output_tokens": 5,
                "charged_tokens": 15, "usage_source": "api_response", "logical_call_id": "original"}]}, source_run_id="full")
            self.assertTrue(budget.calls[0]["reused"])
            self.assertEqual(budget.calls[0]["charged_tokens"], 15)
            with self.assertRaises(BudgetExhausted):
                budget.reserve(input_tokens=1, output_cap=10, model="fixture")

    def test_reused_calls_preserve_stage_identity_and_unknown_reservations(self):
        with execution_budget() as budget:
            record_reused_calls({"calls": [{"stage": "generation", "logical_call_id": "source",
                "input_tokens": None, "output_tokens": None, "reserved_tokens": 90,
                "charged_tokens": 90, "usage_source": "unavailable"}]}, source_run_id="draft")
            self.assertEqual(budget.calls[0]["stage"], "generation")
            self.assertEqual(budget.calls[0]["logical_call_id"], "source")
            self.assertEqual(budget.calls[0]["charged_tokens"], 90)
            self.assertTrue(budget.snapshot()["unknown_usage"])

    def test_explicit_shared_cap_overrides_resolved_dataset_defaults(self):
        from rag_benchmark.generation.budget import with_shared_output_cap
        for dataset in ("kilt_nq", "kilt_hotpotqa", "alce_asqa", "alce_qampari", "scifact", "garage", "rgb_en"):
            for pipeline in ("closed_book", "vanilla_rag", "best_practice_rag", "rag_verification"):
                experiment = build_experiment_with_resolved_pipeline(repo_root=REPO_ROOT,
                    dataset_name=dataset, pipeline_name=pipeline, use_local_paths=False,
                    execution_profile_name="quality_max")
                experiment = with_shared_output_cap(experiment, 4096)
                self.assertEqual(experiment.pipeline.budget.max_generation_tokens, 4096)
                self.assertEqual(experiment.pipeline.query_formulation["max_output_tokens"], 4096)

    def test_detection_source_requires_matching_context_and_frozen_protocol(self):
        metadata = {**EvaluationIdentity("synthetic", "v1", "mvp", 42, "context-a").metadata(),
                    "generation_context": {"content_sha256": "original-evidence"}}
        draft = PredictionRecord(run_id="draft", pipeline="best", example_id="e", question="q", answer="a", metadata=metadata)
        source = draft.model_copy(update={"run_id": "detection", "pipeline": "verification", "verification":
            VerificationRecord(status="failed", original_answer="a", metadata={"policy": "detect_only", "detection_artifacts": []})})
        validate_detection_source([source], [draft], selected_ids={"e"})
        changed = source.model_copy(update={"metadata": {**metadata, "context_set_id": "context-b"}})
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_detection_source([changed], [draft], selected_ids={"e"})
        incomplete = source.model_copy(update={"verification": VerificationRecord(status="failed", original_answer="a", metadata={"policy": "detect_only"})})
        with self.assertRaisesRegex(ValueError, "lacks frozen detection records"):
            validate_detection_source([incomplete], [draft], selected_ids={"e"})


if __name__ == "__main__":
    unittest.main()
