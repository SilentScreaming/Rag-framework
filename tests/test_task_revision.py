"""Task-aware revision policies, evidence selection, and edit contracts."""
import sys
import json
import unittest
import tempfile
from pathlib import Path

from rag_benchmark.data import Example
from rag_benchmark.evaluation.identity import EvaluationIdentity
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.generation.usage import TextResponse
from rag_benchmark.schemas import PredictionRecord, ClaimCheckRecord, CitationRecord
from rag_benchmark.verification.cove import FactoredCoVeConfig, FactoredCoVeVerifier
from rag_benchmark.verification.task_revision import (RevisionInput, TaskRevisionConfig,
    TaskRevisionVerifier, parse_list)
from rag_benchmark.verification.replay import validate_drafts, summarize_replay


class Judge:
    def __init__(self, label=None):
        self.label = label
        self.inputs = []
    def judge_joint(self, claim, texts):
        self.inputs.append((claim, texts))
        if any("ERROR" in t for t in texts):
            return EvidenceDecision("input_too_long")
        label = self.label or ("contradicted" if "Bob" in claim else "supported")
        if any("OPPOSING" in t for t in texts):
            # SciFact now submits the complete set in one joint judgment. The
            # fixture must model mixed support/opposition in that exact set.
            label = "conflicting" if len(texts) > 1 else "contradicted"
        return EvidenceDecision("completed", label)


class Generator:
    model = "fixture"
    def __init__(self, replacement="Charlie", finish="stop"):
        self.prompts, self.replacement, self.finish = [], replacement, finish
    def generate_text_result(self, prompt):
        self.prompts.append(prompt)
        payload = json.loads(prompt.split("\n", 1)[1])
        if "item" in payload:
            result = {"claim": payload["item"] + " is a member."}
        elif self.replacement is None:
            result = {"item": None}
        else:
            result = {"item": self.replacement, "claim": self.replacement + " is a member.", "evidence_ids": ["v1"]}
        return TextResponse(json.dumps(result), 10, 10, "fixture", finish_reason=self.finish)


def pred(answer="Alice, Bob", *, label=False, evidence=True):
    return PredictionRecord(example_id="e1", pipeline="rag_verification", question="Name the members." if not label else "Drug X reduces Y.",
        answer=answer, task_track="verification_hallucination" if label else "long_answer_citation",
        gold_answers=[{"text": "PRIVATE_GOLD_DO_NOT_SEND"}],
        metadata={**EvaluationIdentity("scifact" if label else "alce_qampari", "fixture-v1", "dev", 42).metadata(),
                  **({"generation_context": {"visible_evidence": [
            {"context_id": "c1", "chunk_id": "d1", "sentence_id": 0, "text": "Alice and Charlie are members. Bob is not a member."}
        ]}} if evidence else {})})


def verify(p=None, *, label=False, judge=None, generator=None, **config):
    return TaskRevisionVerifier(judge=judge or Judge(), generator=generator or Generator(),
        config=TaskRevisionConfig(**config)).verify(p or pred(),
            task_type="scientific_claim_verification" if label else "list_answer_citation")


class TaskRevisionTest(unittest.TestCase):
    def test_existing_abstention_is_not_an_execution_failure(self):
        v = verify(pred("UNKNOWN"))
        self.assertEqual(v.verification.status, "not_run")
        self.assertEqual(v.verification.metadata["status"], "not_applicable")

    def test_abstention_is_one_answer_level_action(self):
        v = verify(judge=Judge("contradicted"), policy="abstain_only", allow_abstention=True, max_edits=1)
        self.assertEqual(v.metrics["verification_applied_edit_count"], 1)


    def test_runner_replay_integration_without_models(self):
        from rag_benchmark.config import (PipelineConfig, ExperimentConfig, DatasetConfig,
            DatasetSourceConfig, PathConfig)
        from rag_benchmark.data import DatasetSlice, CorpusChunk
        from rag_benchmark.runner import ConfigDrivenPipelineRunner, FormalRunSettings
        from rag_benchmark.component_factory import ComponentFactorySettings
        p = pred("SUPPORT", label=True)
        p.metadata["data_version"] = "1"
        p.metrics["claim_support_rate"] = 1.0
        example = self.example()
        corpus = [CorpusChunk("d1", "d1", "scifact", "Study", "Drug X does not reduce Y.",
                              [{"sentence_id": 0, "text": "Drug X does not reduce Y."}], {})]
        pipeline = PipelineConfig.model_validate({"name": "rag_verification",
            "task_profile": {"track": "verification_hallucination", "answer_mode": "claims", "abstention_allowed": False},
            "stages": {"query_formulation": "none", "retrieval": "bm25", "reranking": "none", "compression": "none",
                       "generation": "context_grounded", "verification": "factored_cove"},
            "generation": {},
            "verification": {"task_revision": {"policy": "constrained_revision"}, "allow_abstention": False}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = ExperimentConfig(dataset=DatasetConfig(name="scifact", role="test", split="dev",
                source=DatasetSourceConfig(provider="fixture", version="1", root_key="scifact")), pipeline=pipeline,
                paths=PathConfig(datasets_root=root, models_root=root, outputs_root=root, datasets={}))
            data = DatasetSlice("scifact", "dev", "fixture", [example], corpus, {"e1": []})
            runner = ConfigDrivenPipelineRunner(repo_root=Path(__file__).resolve().parents[1], experiment=experiment,
                dataset_slice=data, settings=FormalRunSettings(run_id="revision-offline", save_outputs=False,
                    draft_predictions={"e1": p}, component_settings=ComponentFactorySettings(generator_kind="extractive")))
            # Production dispatch still receives a local-style judge, never gold labels.
            class LocalJudge:
                name = "fixture"
                def judge(self, **kwargs):
                    from rag_benchmark.evaluation.evidence import EvidenceJudgment
                    return EvidenceJudgment("contradicted", "fixture evidence", 0.99)
            runner.evidence_judge = LocalJudge()
            original = p.to_json_dict()
            output = runner.run().predictions[0]
            self.assertEqual(output.answer, "CONTRADICT")
            self.assertEqual(output.metrics["verification_repair_success"], 1)
            self.assertIsNone(output.metrics["post_verification_claim_support_rate"])
            self.assertEqual(p.to_json_dict(), original)

    def test_declared_citations_can_be_mapped_without_becoming_truth(self):
        p = pred()
        p.citations = [CitationRecord(citation_id="old", chunk_id="d1", cited_text="Original evidence")]
        p.claim_checks = [ClaimCheckRecord(claim_id="compound", claim_text="Alice and Bob are members.", citations=p.citations)]
        class MappingGenerator(Generator):
            def generate_text_result(self, prompt):
                if "allowed_citation_ids" in prompt:
                    self.prompts.append(prompt)
                    return TextResponse(json.dumps({"items": [
                        {"item_id": "i1", "citation_ids": ["old"], "status": "matched"},
                        {"item_id": "i2", "citation_ids": ["old"], "status": "matched"}]}), 10, 10, "fixture")
                return super().generate_text_result(prompt)
        v = verify(p, generator=MappingGenerator())
        self.assertTrue(v.answer.startswith("Alice [1], Charlie [2]"))
        self.assertEqual({c.citation_id for c in v.citations}, {"1", "2"})
        self.assertEqual({c.metadata["source_citation_id"] for c in v.citations}, {"old", "revision-i2-1"})
        self.assertEqual(v.verification.metadata["attribution_mapping"], {"i1": ["old"], "i2": ["old"]})

    def test_unmapped_declared_citation_does_not_disappear(self):
        p = pred()
        p.citations = [CitationRecord(citation_id="old", chunk_id="d1", cited_text="evidence")]
        v = verify(p)
        self.assertEqual(v.answer, p.answer)
        self.assertEqual(v.citations, p.citations)
        self.assertIn("citation_mapping_unresolved", [e["rejection_reason"] for e in v.verification.metadata["edits"]])

    def test_changed_answer_drops_stale_evaluation(self):
        p = pred()
        p.metrics = {"claim_support_rate": 1, "citation_precision": 1, "f1": 1, "cost": 0.1}
        p.metadata["independent_evaluation"] = {"old": True}
        v = verify(p)
        self.assertNotIn("claim_support_rate", v.metrics)
        self.assertNotIn("independent_evaluation", v.metadata)
        self.assertEqual(v.metrics["cost"], 0.1)

    def test_label_empty_claims_can_be_revised(self):
        p = pred("SUPPORT", label=True)
        v = verify(p, label=True, judge=Judge("contradicted"))
        self.assertEqual(v.answer, "CONTRADICT")
        self.assertEqual(v.answer_bundle.verification_label, "REFUTES")
        self.assertEqual(v.citations[0].sentence_id, 0)
        self.assertFalse(v.answer_bundle.abstained)
        self.assertEqual(p.answer, "SUPPORT")

    def test_correct_refutation_is_not_a_failed_answer(self):
        v = verify(pred("CONTRADICT", label=True), label=True, judge=Judge("contradicted"))
        self.assertEqual(v.answer, "CONTRADICT")
        self.assertEqual(v.verification.status, "passed")
        self.assertEqual(v.verification.metadata["edits"][0]["proposed_action"], "keep")

    def test_label_insufficient_is_not_abstention(self):
        v = verify(pred("SUPPORT", label=True), label=True, judge=Judge("insufficient"))
        self.assertEqual(v.answer, "NOT_ENOUGH_INFO")
        self.assertFalse(v.answer_bundle.abstained)
        self.assertEqual(v.citations, [])

    def test_label_failure_is_not_nei(self):
        p = pred("SUPPORT", label=True)
        p.metadata["generation_context"]["visible_evidence"][0]["text"] = "ERROR"
        v = verify(p, label=True)
        self.assertEqual(v.answer, p.answer)
        self.assertEqual(v.verification.metadata["status"], "partial")

    def test_conflicting_documents_preserve_label(self):
        p = pred("SUPPORT", label=True)
        p.metadata["generation_context"]["visible_evidence"].append({"chunk_id": "d2", "text": "OPPOSING"})
        v = verify(p, label=True)
        self.assertEqual(v.answer, p.answer)
        self.assertEqual(v.verification.metadata["observations"][0]["label"], "conflicting")

    def test_missing_evidence_is_not_empty_evidence(self):
        v = verify(pred("SUPPORT", label=True, evidence=False), label=True)
        self.assertEqual(v.answer, "SUPPORT")
        self.assertEqual(v.verification.status, "error")

    def test_incomplete_evidence_does_not_change_label(self):
        p = pred("SUPPORT", label=True)
        p.metadata["generation_context"]["visible_evidence"][0]["metadata"] = {"budget_truncated": True}
        v = verify(p, label=True, judge=Judge("contradicted"))
        self.assertEqual(v.answer, p.answer)

    def test_detect_only_never_mutates_payload(self):
        for p, label in ((pred(), False), (pred("SUPPORT", label=True), True)):
            v = verify(p, label=label, policy="detect_only", judge=Judge("contradicted"))
            self.assertEqual(v.answer, p.answer)
            self.assertEqual(v.citations, p.citations)
            self.assertEqual(v.claim_checks, p.claim_checks)
            self.assertTrue(all(e["applied_action"] == "keep" for e in v.verification.metadata["edits"]))

    def test_replace_item_and_synchronize_references(self):
        p = pred()
        original = p.to_json_dict()
        v = verify(p)
        self.assertTrue(v.answer.startswith("Alice, Charlie [1]"))
        self.assertEqual([i.text for i in v.answer_bundle.list_items], ["Alice", "Charlie"])
        self.assertEqual(v.claim_checks[0].claim_text, "Charlie is a member.")
        self.assertEqual(v.citations[0].citation_id, "1")
        self.assertEqual(p.to_json_dict(), original)
        v.metadata["generation_context"]["visible_evidence"][0]["text"] = "mutated output"
        self.assertEqual(p.to_json_dict(), original)

    def test_remove_only_contradicted_item(self):
        v = verify(generator=Generator(replacement=None))
        self.assertEqual(v.answer, "Alice")
        self.assertEqual(v.verification.metadata["edits"][-1].get("applied_action") in {"keep", "remove"}, True)
        self.assertEqual(v.metrics["verification_applied_edit_count"], 1)

    def test_insufficient_not_removed(self):
        v = verify(judge=Judge("insufficient"), generator=Generator(replacement=None))
        self.assertEqual(v.answer, "Alice, Bob")
        self.assertEqual(v.metrics["verification_applied_edit_count"], 0)

    def test_duplicate_replacement_is_rejected(self):
        v = verify(generator=Generator(replacement="Alice"))
        self.assertEqual(v.answer, "Alice, Bob")
        self.assertIn("duplicate_replacement", [e["rejection_reason"] for e in v.verification.metadata["edits"]])

    def test_removal_cannot_create_empty_list(self):
        v = verify(pred("Bob"), generator=Generator(replacement=None))
        self.assertEqual(v.answer, "Bob")
        self.assertEqual(v.verification.metadata["edits"][0]["rejection_reason"], "empty_list_requires_explicit_abstention_policy")

    def test_edit_budget_preserves_answer(self):
        v = verify(max_edits=0)
        self.assertEqual(v.answer, "Alice, Bob")
        self.assertIn("edit_budget_exhausted", [e["rejection_reason"] for e in v.verification.metadata["edits"]])

    def test_call_budget_failure_is_observable(self):
        generator = Generator()
        v = verify(generator=generator, max_calls=0)
        self.assertEqual(generator.prompts, [])
        self.assertEqual(v.answer, "Alice, Bob")
        self.assertEqual(v.verification.metadata["status"], "partial")

    def test_item_budget_does_not_report_passed(self):
        v = verify(judge=Judge("supported"), max_items=1)
        self.assertEqual(len(v.verification.metadata["unreviewed"]), 1)
        self.assertNotEqual(v.verification.status, "passed")

    def test_truncated_generation_cannot_modify(self):
        v = verify(generator=Generator(finish="length"))
        self.assertEqual(v.answer, "Alice, Bob")
        self.assertEqual(v.verification.metadata["status"], "partial")

    def test_gold_never_enters_controller_prompts(self):
        generator, judge = Generator(), Judge()
        verify(generator=generator, judge=judge)
        self.assertNotIn("PRIVATE_GOLD_DO_NOT_SEND", str(generator.prompts) + str(judge.inputs))

    def test_abstention_requires_permission(self):
        with self.assertRaises(ValueError):
            TaskRevisionConfig(policy="abstain_only")
        v = verify(judge=Judge("insufficient"), policy="abstain_only", allow_abstention=True)
        self.assertEqual(v.answer, "UNKNOWN")
        self.assertTrue(v.answer_bundle.abstained)

    def test_ambiguous_list_no_silent_split(self):
        for answer in ("Alice; Bob", "Alice\nBob", "Alice,", '"Alice, Bob'):
            with self.assertRaises(ValueError):
                parse_list(answer)
        self.assertEqual([i.text for i in parse_list('"Last, First", Bob')], ["Last, First", "Bob"])

    def test_source_offsets(self):
        text = " Alice [c1], Bob "
        for item in parse_list(text):
            a, b = item.source_span
            self.assertEqual(text[a:b], item.metadata["source_text"])

    def test_dispatch_only_for_explicit_task(self):
        p = pred("SUPPORT", label=True)
        v = FactoredCoVeVerifier(judge=Judge("contradicted"),
            config=FactoredCoVeConfig(task_type="scientific_claim_verification", answer_mode="claims")).verify(p)
        self.assertEqual(v.answer, "CONTRADICT")
        self.assertEqual(v.verification.metadata["verifier"], "task-revision-v1")

    def test_indirect_scifact_abstention_permission_is_not_overridden(self):
        from rag_benchmark.config import ConfigLoader
        from rag_benchmark.planning import build_experiment_with_resolved_pipeline
        repo = Path(__file__).resolve().parents[1]
        exp = build_experiment_with_resolved_pipeline(repo_root=repo, dataset_name="scifact",
            pipeline_name="rag_verification", use_local_paths=False, execution_profile_name="smoke")
        self.assertFalse(exp.pipeline.task_profile.abstention_allowed)

    def test_no_abstention_override_in_dispatch(self):
        with self.assertRaises(ValueError):
            FactoredCoVeVerifier(judge=Judge(), config=FactoredCoVeConfig(task_type="scientific_claim_verification",
                task_revision={"policy": "abstain_only", "allow_abstention": True})).verify(pred("SUPPORT", label=True))

    def example(self):
        return Example("e1", "scifact", "dev", "scientific_claim_verification", "Drug X reduces Y.",
            [{"text": "REFUTES", "type": "verification_label"}], [], {})

    def test_draft_preflight_rejects_duplicates_and_mismatch(self):
        p = pred("SUPPORT", label=True)
        with self.assertRaises(ValueError):
            validate_drafts([p, p], [self.example()], expected_track=p.task_track)
        with self.assertRaises(ValueError):
            validate_drafts([p.model_copy(update={"question": "wrong"})], [self.example()], expected_track=p.task_track)

    def test_replay_recomputes_task_metrics_without_zero_fill(self):
        p = pred("SUPPORT", label=True)
        v = verify(p, label=True, judge=Judge("contradicted"))
        summary = summarize_replay([p], [v], {"e1": self.example()}, samples=5)
        self.assertEqual(summary["mean_task_score_delta"], 1)
        self.assertEqual(summary["correction_success_rate"], 1)
        self.assertIsNone(summary["harmful_revision_rate"])
        self.assertEqual(summary["label_transitions"], {"SUPPORTS->REFUTES": 1})


if __name__ == "__main__":
    unittest.main()
