"""Offline contracts for full scientific evidence and neutral RGB inputs."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_benchmark.data import CorpusChunk, ContextSet, DatasetSlice, Example
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.evaluation.exact_evidence import NLIExactJudge
from rag_benchmark.generation.usage import TextResponse
from rag_benchmark.generation.context import prepare_context
from rag_benchmark.compression.sentence_select import context_identifier, CompressedContext, CompressedSentence
from rag_benchmark.indexing.dense_cache import corpus_fingerprint
from rag_benchmark.rgb_protocol import apply_rgb_protocol, rgb_review_identity
from rag_benchmark.stage_cache import canonical_hash
from rag_benchmark.generation.budget import execution_budget, CURRENT_BUDGET
from rag_benchmark.schemas import PredictionRecord
from rag_benchmark.verification.task_revision import TaskRevisionVerifier, TaskRevisionConfig


class NeverNLI:
    model_name, revision, max_length = "fixture-nli", "fixture", 512
    entailment_threshold = contradiction_threshold = .8
    def _load(self):
        raise AssertionError("SciFact must not fall back to the short NLI model")


class ExactGenerator:
    model, max_output_tokens, reasoning_effort = "fixture-model", 32768, None
    def __init__(self, label="supported"):
        self.prompts, self.label = [], label
    def generate_text_result(self, prompt):
        self.prompts.append(prompt)
        return TextResponse(json.dumps({"label": self.label, "reason": "fixture joint evidence"}),
                            3000, 20, "fixture", finish_reason="stop")


def scientific_prediction():
    return PredictionRecord(example_id="s1", pipeline="rag_verification", question="A and B imply C.",
                            answer="NOT_ENOUGH_INFO", task_track="verification_hallucination",
                            gold_answers=[{"text": "DO_NOT_SEND_GOLD"}])


class Retriever:
    def __init__(self, texts): self.texts = texts
    def retrieve(self, query, *, top_k):
        return [RetrievedChunk(CorpusChunk(f"d{i}", f"d{i}", "scifact", "Study", text, [], {}),
                               rank=i+1, score=1) for i, text in enumerate(self.texts[:top_k])]


def scientific(generator, texts, **config):
    config.setdefault("detection_corpus_hash", canonical_hash(texts))
    return TaskRevisionVerifier(judge=NLIExactJudge(NeverNLI()), generator=generator,
        retriever=Retriever(texts), config=TaskRevisionConfig(**config)).verify(
            scientific_prediction(), task_type="scientific_claim_verification")


class ScientificEvidenceTests(unittest.TestCase):
    def test_retrieval_exception_is_frozen_for_the_other_policy(self):
        class FlakyRetriever(Retriever):
            def __init__(self, fail):
                super().__init__(["Evidence."]); self.fail, self.calls = fail, 0
            def retrieve(self, query, *, top_k):
                self.calls += 1
                if self.fail:
                    raise ValueError("fixture_retrieval_failure")
                return super().retrieve(query, top_k=top_k)
        with tempfile.TemporaryDirectory() as folder:
            first, second = FlakyRetriever(True), FlakyRetriever(False)
            config = TaskRevisionConfig(detection_cache_root=folder, detection_corpus_hash="same-frozen-corpus")
            a = TaskRevisionVerifier(judge=NLIExactJudge(NeverNLI()), generator=ExactGenerator(),
                retriever=first, config=config).verify(scientific_prediction(), task_type="scientific_claim_verification")
            b = TaskRevisionVerifier(judge=NLIExactJudge(NeverNLI()), generator=ExactGenerator(),
                retriever=second, config=config).verify(scientific_prediction(), task_type="scientific_claim_verification")
            self.assertEqual(second.calls, 0)
            self.assertEqual(b.verification.metadata["execution_status"], "failed")
            self.assertEqual(a.verification.metadata["detection_artifacts"], b.verification.metadata["detection_artifacts"])
            self.assertEqual(a.verification.metadata["detection_cache"][0]["frozen_outcome"], "frozen_incomplete")

    def test_list_detection_prefix_is_shared_before_policy_candidate_generation(self):
        from test_task_revision import Generator, Judge, pred
        with tempfile.TemporaryDirectory() as folder:
            first_gen, second_gen = Generator(replacement=None), Generator(replacement=None)
            a = TaskRevisionVerifier(judge=Judge(), generator=first_gen,
                config=TaskRevisionConfig(policy="detect_only", detection_cache_root=folder)).verify(pred(), task_type="list_answer_citation")
            b = TaskRevisionVerifier(judge=Judge(), generator=second_gen,
                config=TaskRevisionConfig(allow_removal=False, detection_cache_root=folder)).verify(pred(), task_type="list_answer_citation")
            self.assertEqual(canonical_hash(a.verification.metadata["observations"]), canonical_hash(b.verification.metadata["observations"]))
            self.assertEqual(len(first_gen.prompts), 2)
            self.assertEqual(len(second_gen.prompts), 1)  # only the policy-specific candidate
            self.assertEqual(b.verification.metadata["physical_dispatch_count"], 1)
            self.assertEqual(b.verification.metadata["logical_call_count"], 3)

    def test_cached_calls_keep_complete_method_budget_without_dispatch(self):
        class LedgerGenerator(ExactGenerator):
            def generate_text_result(self, prompt):
                budget = CURRENT_BUDGET.get()
                call = budget.reserve(input_tokens=3000, output_cap=32768, model=self.model)
                call.update(logical_call_id="fixture-original-call", status="completed", input_tokens=3000,
                            output_tokens=20, charged_tokens=3020, usage_source="fixture")
                return super().generate_text_result(prompt)
        with tempfile.TemporaryDirectory() as folder:
            with execution_budget(max_calls=1):
                scientific(LedgerGenerator(), ["Evidence."], policy="detect_only", detection_cache_root=folder)
            generator = LedgerGenerator()
            with execution_budget(max_calls=1) as ledger:
                result = scientific(generator, ["Evidence."], detection_cache_root=folder)
                self.assertEqual(len(ledger.calls), 1)
                self.assertTrue(ledger.calls[0]["reused"])
                self.assertEqual(generator.prompts, [])
                self.assertEqual(result.verification.metadata["physical_dispatch_count"], 0)
                self.assertEqual(result.verification.metadata["logical_call_count"], 1)
            with execution_budget(max_calls=0):
                blocked = scientific(generator, ["Evidence."], detection_cache_root=folder)
                self.assertEqual(blocked.verification.metadata["execution_status"], "failed")
                self.assertEqual(generator.prompts, [])
    def test_full_selected_set_uses_judge_beyond_answer_context_budget(self):
        generator = ExactGenerator()
        texts = ["Background " * 2500 + "FIRST_COMPLETE_END", "B together with A implies C. SECOND_COMPLETE_END"]
        result = scientific(generator, texts, max_context_tokens=12, max_input_tokens=20)
        self.assertEqual(result.answer, "SUPPORT")
        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("FIRST_COMPLETE_END", generator.prompts[0])
        self.assertIn("SECOND_COMPLETE_END", generator.prompts[0])
        self.assertNotIn("DO_NOT_SEND_GOLD", generator.prompts[0])
        self.assertEqual(result.verification.metadata["checked_claim_count"], 1)
        self.assertEqual(result.verification.metadata["call_usage"]["input_tokens"], 3000)
        self.assertEqual(result.verification.metadata["judge"]["model"], "fixture-model")
        self.assertEqual(len(result.verification.metadata["scopes"][0]["selected_evidence"]), 2)

    def test_overlong_selected_set_is_saved_and_not_changed_to_nei(self):
        generator = ExactGenerator()
        result = scientific(generator, ["A " * 200], scientific_max_input_tokens=10)
        self.assertEqual(result.answer, "NOT_ENOUGH_INFO")
        self.assertEqual(generator.prompts, [])
        self.assertEqual(result.verification.metadata["execution_status"], "partial")
        self.assertEqual(len(result.verification.metadata["scopes"][0]["selected_evidence"]), 1)
        self.assertEqual(result.verification.metadata["checked_claim_count"], 0)

    def test_detection_shared_but_policy_edits_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = ExactGenerator(), ExactGenerator()
            a = scientific(first, ["A and B imply C."], policy="detect_only", detection_cache_root=folder)
            b = scientific(second, ["A and B imply C."], detection_cache_root=folder)
            self.assertEqual(a.answer, "NOT_ENOUGH_INFO")
            self.assertEqual(b.answer, "SUPPORT")
            self.assertEqual(len(first.prompts), 1)
            self.assertEqual(second.prompts, [])
            self.assertEqual(canonical_hash(a.verification.metadata["observations"]), canonical_hash(b.verification.metadata["observations"]))
            self.assertEqual(b.verification.metadata["call_usage"]["attempt_count"], 0)
            self.assertEqual(sum(row["call_usage"]["attempt_count"] for row in b.verification.metadata["reused_detection_usage"]), 1)
            self.assertEqual(a.verification.metadata["detection_artifacts"], b.verification.metadata["detection_artifacts"])

    def test_paired_failed_detection_is_frozen_and_new_experiment_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = ExactGenerator("invalid")
            first = scientific(bad, ["Evidence one."], detection_cache_root=folder)
            self.assertEqual(first.verification.metadata["execution_status"], "partial")
            good, changed = ExactGenerator(), ExactGenerator()
            replayed = scientific(good, ["Evidence one."], detection_cache_root=folder)
            scientific(changed, ["Evidence two."], detection_cache_root=folder)
            self.assertEqual(good.prompts, [])
            self.assertEqual(replayed.verification.metadata["execution_status"], "partial")
            self.assertIn("frozen_incomplete", [row.get("frozen_outcome") for row in replayed.verification.metadata["detection_cache"]])
            self.assertEqual(len(changed.prompts), 1)
            with tempfile.TemporaryDirectory() as new_experiment:
                scientific(good, ["Evidence one."], detection_cache_root=new_experiment)
                self.assertEqual(len(good.prompts), 1)

    def test_statement_exception_is_a_paired_incomplete_record(self):
        from test_task_revision import Generator, Judge, pred
        class BadGenerator(Generator):
            def generate_text_result(self, prompt):
                self.prompts.append(prompt)
                raise ValueError("fixture_statement_failure")
        with tempfile.TemporaryDirectory() as folder:
            bad, good = BadGenerator(), Generator()
            a = TaskRevisionVerifier(judge=Judge(), generator=bad,
                config=TaskRevisionConfig(policy="detect_only", detection_cache_root=folder)).verify(pred(), task_type="list_answer_citation")
            b = TaskRevisionVerifier(judge=Judge(), generator=good,
                config=TaskRevisionConfig(detection_cache_root=folder)).verify(pred(), task_type="list_answer_citation")
            self.assertEqual(good.prompts, [])
            self.assertEqual(canonical_hash(a.verification.metadata["observations"]), canonical_hash(b.verification.metadata["observations"]))
            self.assertEqual(b.answer, pred().answer)


class RGBContextTests(unittest.TestCase):
    def data(self):
        chunk = CorpusChunk("rgb:chunk:source:positive_wrong:1", "d", "rgb_en", "positive_wrong passage 1", "Supplied fact.", [], {})
        example = Example("e", "rgb_en", "formal", "noise_and_rejection", "Question?", [], [], {})
        context = ContextSet("c", "e", "rgb_en", "noise_robustness", [chunk.chunk_id], "abstain", {"positive_count": 0})
        return DatasetSlice("rgb_en", "formal", "provided", [example], [chunk], {}, {"e": [context]})

    def test_loaded_title_and_both_display_paths_are_neutral_with_source_mapping(self):
        source = self.data()
        data = apply_rgb_protocol(source)
        chunk = data.corpus[0]
        self.assertNotEqual(corpus_fingerprint(source.corpus), corpus_fingerprint(data.corpus))
        self.assertEqual(chunk.chunk_id, source.corpus[0].chunk_id)
        self.assertNotIn("wrong", chunk.title)
        context_id = context_identifier(rank=1, chunk_id=chunk.chunk_id, sentence_id=0, fallback_index=0)
        self.assertNotIn("wrong", context_id)
        retrieved = [RetrievedChunk(source.corpus[0], 1, 1)]
        compressed = CompressedContext("Question?", [CompressedSentence("positive_wrong_ID", chunk.chunk_id, "d",
            "positive_wrong title", 0, chunk.text, 1, 1, 1, {})])
        for selected in (None, compressed):
            prepared = prepare_context(query="Question?", retrieved=retrieved, compressed=selected, max_tokens=500)
            visible = prepared.evidence.to_prompt_context()
            self.assertNotIn("positive_wrong", visible)
            self.assertEqual(prepared.metadata["visible_evidence"][0]["chunk_id"], chunk.chunk_id)

    def test_only_reviewed_insufficient_is_in_semantic_cohort(self):
        source = self.data()
        input_hash = rgb_review_identity(source.examples[0].question, source.corpus)
        for decision, expected in [("answerable", False), ("ambiguous", False), ("insufficient", True)]:
            overlay = {"schema_version": "fixture", "reviews": {"c": {"input_hash": input_hash, "decision": decision}}}
            context = apply_rgb_protocol(source, overlay).context_sets_by_example["e"][0]
            self.assertEqual(context.metadata["insufficiency_analysis_eligible"], expected)
            self.assertEqual(context.expected_behavior, "abstain")
        self.assertFalse(apply_rgb_protocol(source).context_sets_by_example["e"][0].metadata["insufficiency_analysis_eligible"])

    def test_review_cannot_be_applied_to_changed_evidence(self):
        source = self.data()
        overlay = {"reviews": {"c": {"input_hash": "different", "decision": "insufficient"}}}
        with self.assertRaisesRegex(ValueError, "evidence_mismatch"):
            apply_rgb_protocol(source, overlay)

    def test_unknown_sentence_ids_keep_distinct_display_ids(self):
        chunk = self.data().corpus[0]
        sentences = [CompressedSentence("negative-ID", chunk.chunk_id, "d", "negative", None,
                       text, 1, 1, 1, {}) for text in ("First fact.", "Second fact.")]
        prepared = prepare_context(query="q", retrieved=[], compressed=CompressedContext("q", sentences), max_tokens=500)
        ids = [row["context_id"] for row in prepared.metadata["visible_evidence"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
