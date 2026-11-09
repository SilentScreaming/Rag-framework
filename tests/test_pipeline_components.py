from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from rag_benchmark.compression.sentence_select import (
    SentenceCompressionConfig,
    SentenceTopNCompressor,
)
from rag_benchmark.evaluation.evidence import EvidenceJudgment
from rag_benchmark.evaluation.tracks import bootstrap_confidence_intervals, group_track_metrics
from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.query_formulation import MultiQueryLLMFormulator
from rag_benchmark.schemas import PredictionRecord
from rag_benchmark.runner import retrieve_query_sets
from rag_benchmark.stage_cache import StageArtifactCache
from rag_benchmark.verification.cove import FactoredCoVeConfig, FactoredCoVeVerifier


class JsonGenerator:
    provider = "test"
    model = "test-model"

    def generate_text(self, prompt: str) -> str:
        return (
            '{"queries":["Dune author", "Frank Herbert bibliography"],'
            '"subquestions":["Who wrote Dune?"],"answer_type":"person"}'
        )


@dataclass
class FixedJudge:
    status: str
    score: float
    name: str = "fixed"

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        return EvidenceJudgment(status=self.status, score=self.score, rationale="fixed test judgment")


class ReplacementAwareJudge:
    name = "replacement-aware"

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        if "Corrected" in claim_text or "Alice is the only member" in claim_text:
            return EvidenceJudgment(status="supported", score=0.99, rationale="replacement supported")
        return EvidenceJudgment(status="contradicted", score=0.95, rationale="original contradicted")


class VerificationGenerator:
    provider = "test"
    model = "independent-verifier"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Revise the original answer minimally" in prompt:
            return '{"answer":"Corrected answer. Stable detail."}'
        if "key question" in prompt:
            return '{"question":"Who actually wrote the work?"}'
        if "original answer" in prompt:
            self.assert_original_answer_hidden(prompt)
        if "Answer each verification question independently" in prompt:
            return ('{"answers":[{"question_id":"verify-1","answer":"Corrected answer",'
                    '"short_answer":"Corrected answer","reason":"Evidence supplies the replacement."}]}')
        return "Corrected answer"

    @staticmethod
    def assert_original_answer_hidden(prompt: str) -> None:
        if "Wrong answer" in prompt:
            raise AssertionError("verification answerer saw the original answer")


class RepairingBatchVerificationGenerator:
    provider = "test"
    model = "repairing-batch-verifier"

    def __init__(self) -> None:
        self.plan_calls = 0
        self.answer_calls = 0

    def generate_text(self, prompt: str) -> str:
        if "key question" in prompt:
            self.plan_calls += 1
            if self.plan_calls == 1:
                return '{"questions":[{"claim_id":"c1","question":"Check first?"}]}'
            return (
                '{"questions":['
                '{"claim_id":"c1","question":"Check first?"},'
                '{"claim_id":"c2","question":"Check second?"}]}'
            )
        self.answer_calls += 1
        if self.answer_calls == 1:
            return '{"answers":[{"question_id":"verify-1","answer":"First answer"}]}'
        return (
            '{"answers":['
            '{"question_id":"verify-1","answer":"First answer"},'
            '{"question_id":"verify-2","answer":"Second answer"}]}'
        )


class BatchVerificationRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def retrieve_many(self, queries: list[str], *, top_k: int) -> list[list[RetrievedChunk]]:
        self.calls.append((queries, top_k))
        return [
            [RetrievedChunk(chunk("batch", "Evidence", "Batch evidence."), rank=1, score=1.0)]
            for _ in queries
        ]


class BadBatchRetriever:
    def retrieve_many(self, queries: list[str], *, top_k: int) -> list[list[RetrievedChunk]]:
        return []


def chunk(chunk_id: str, title: str, sentence: str) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        dataset="synthetic",
        title=title,
        text=sentence,
        sentences=[{"sentence_id": 0, "text": sentence}],
        metadata={},
    )


class PipelineComponentsTest(unittest.TestCase):
    def test_runner_uses_batch_retrieval_and_preserves_query_order(self) -> None:
        retriever = BatchVerificationRetriever()
        results = retrieve_query_sets(retriever, ["first", "second"], top_k=3)

        self.assertEqual([query for query, _ in results], ["first", "second"])
        self.assertEqual(retriever.calls, [(["first", "second"], 3)])

    def test_runner_rejects_incomplete_batch_results(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "batch result count"):
            retrieve_query_sets(BadBatchRetriever(), ["first"], top_k=3)

    def test_multi_query_keeps_original_and_deduplicates(self) -> None:
        result = MultiQueryLLMFormulator(JsonGenerator(), max_queries=4).formulate("Who wrote Dune?")

        self.assertEqual(result.query, "Who wrote Dune?")
        self.assertEqual(result.queries[0], "Who wrote Dune?")
        self.assertEqual(result.queries.count("Who wrote Dune?"), 1)
        self.assertIn("Dune author", result.queries)

    def test_compression_enforces_source_diversity_and_token_budget(self) -> None:
        retrieved = [
            RetrievedChunk(chunk=chunk("a", "Dune", "Frank Herbert wrote Dune."), rank=1, score=2.0),
            RetrievedChunk(chunk=chunk("b", "Herbert", "Herbert was an American author."), rank=2, score=1.0),
        ]
        compressed = SentenceTopNCompressor(
            SentenceCompressionConfig(max_sentences=2, max_tokens=20, min_chunks=2)
        ).compress(query="Who wrote Dune?", retrieved=retrieved)

        self.assertEqual(len({sentence.chunk_id for sentence in compressed.sentences}), 2)
        self.assertLessEqual(compressed.metadata["selected_tokens"], 20)
        self.assertEqual(compressed.metadata["compressor"], "evidence_set_mmr")

    def test_factored_verification_only_revises_high_confidence_contradiction(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-1",
            pipeline="rag_verification",
            question="Who wrote it?",
            answer="Wrong answer",
            answer_bundle={"short_answer": "Wrong answer"},
            claim_checks=[
                {
                    "claim_id": "c1",
                    "claim_text": "Wrong answer",
                    "citations": [{"chunk_id": "a", "cited_text": "Corrected answer"}],
                }
            ],
        )
        verifier = FactoredCoVeVerifier(
            judge=ReplacementAwareJudge(),
            generator=VerificationGenerator(),
            config=FactoredCoVeConfig(revise_short_answer=True, revision_threshold=0.8),
        )

        revised = verifier.verify(prediction)

        self.assertEqual(revised.answer, "Corrected answer")
        self.assertEqual(revised.verification.status, "revised")
        self.assertEqual(revised.verification.decisions[0].action, "replace")
        self.assertFalse(revised.verification.questions[0].metadata["original_answer_visible_to_answerer"])
        self.assertEqual(revised.claim_checks[0].claim_text, "Corrected answer")
        self.assertEqual(revised.answer_bundle.short_answer, "Corrected answer")
        self.assertEqual(revised.verification.questions[0].metadata["verification_reason"],
                         "Evidence supplies the replacement.")

    def test_factored_verification_rejects_unsupported_replacement(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-unsupported-replacement",
            pipeline="rag_verification",
            question="Who wrote it?",
            answer="Wrong answer",
            task_track="short_answer",
            claim_checks=[
                {
                    "claim_id": "c1",
                    "claim_text": "Wrong answer",
                    "citations": [{"chunk_id": "a", "cited_text": "Unrelated evidence"}],
                }
            ],
        )

        checked = FactoredCoVeVerifier(
            judge=FixedJudge("contradicted", 0.99),
            generator=VerificationGenerator(),
            config=FactoredCoVeConfig(revise_short_answer=True),
        ).verify(prediction)

        self.assertEqual(checked.answer, "Wrong answer")
        self.assertEqual(checked.verification.decisions[0].action, "keep")
        self.assertIn("failed the independent evidence-support gate", checked.verification.decisions[0].reason)

    def test_factored_verification_keeps_claim_when_replacement_is_unknown(self) -> None:
        class UnknownGenerator(VerificationGenerator):
            def generate_text(self, prompt: str) -> str:
                if "key question" in prompt:
                    return '{"question":"What is the answer?"}'
                return "UNKNOWN"

        prediction = PredictionRecord(
            example_id="ex-unknown",
            pipeline="rag_verification",
            question="What is the answer?",
            answer="Existing answer",
            claim_checks=[
                {
                    "claim_id": "c1",
                    "claim_text": "Existing answer",
                    "citations": [{"chunk_id": "a", "cited_text": "Conflicting evidence"}],
                }
            ],
        )

        checked = FactoredCoVeVerifier(
            judge=FixedJudge("contradicted", 0.99),
            generator=UnknownGenerator(),
        ).verify(prediction)

        self.assertEqual(checked.answer, "Existing answer")
        self.assertEqual(checked.verification.decisions[0].action, "keep")
        self.assertIn("no evidence-supported replacement", checked.verification.decisions[0].reason)

    def test_factored_verification_preserves_answer_when_evidence_is_uncertain(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-1",
            pipeline="rag_verification",
            question="Who wrote it?",
            answer="Existing answer",
            claim_checks=[{"claim_id": "c1", "claim_text": "Existing answer"}],
        )

        checked = FactoredCoVeVerifier(judge=FixedJudge("unsupported", 0.7)).verify(prediction)

        self.assertEqual(checked.answer, "Existing answer")
        self.assertEqual(checked.verification.decisions[0].action, "keep")

    def test_factored_verification_handles_answer_without_claims(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-empty",
            pipeline="rag_verification",
            question="What is known?",
            answer="UNKNOWN",
            claim_checks=[],
        )
        retriever = BatchVerificationRetriever()

        checked = FactoredCoVeVerifier(
            judge=FixedJudge("supported", 0.99),
            retriever=retriever,
        ).verify(prediction)

        self.assertEqual(retriever.calls, [])
        self.assertEqual(checked.answer, "UNKNOWN")
        self.assertEqual(checked.verification.status, "not_run")

    def test_factored_verification_does_not_rewrite_list_answers_as_prose(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-list",
            pipeline="rag_verification",
            question="Name the members.",
            answer="Alice, Bob",
            task_track="long_answer_citation",
            claim_checks=[
                {
                    "claim_id": "c1",
                    "claim_text": "Alice and Bob are members.",
                    "citations": [{"chunk_id": "a", "cited_text": "Alice is the only member."}],
                }
            ],
        )
        verifier = FactoredCoVeVerifier(
            judge=ReplacementAwareJudge(),
            generator=VerificationGenerator(),
            config=FactoredCoVeConfig(answer_mode="list", revision_threshold=0.8),
        )

        checked = verifier.verify(prediction)

        self.assertEqual(checked.answer, "Alice, Bob")
        self.assertEqual(checked.verification.status, "failed")
        self.assertEqual(checked.verification.decisions[0].action, "replace")

    def test_factored_verification_minimally_revises_long_answer_and_provenance(self) -> None:
        generator = VerificationGenerator()
        prediction = PredictionRecord(
            example_id="ex-long",
            pipeline="rag_verification",
            question="Who wrote the work?",
            answer="Wrong answer. Stable detail.",
            task_track="long_answer_citation",
            claim_checks=[
                {
                    "claim_id": "c1",
                    "claim_text": "Wrong answer",
                    "citations": [{"chunk_id": "a", "cited_text": "Corrected answer"}],
                }
            ],
        )

        revised = FactoredCoVeVerifier(
            judge=ReplacementAwareJudge(),
            generator=generator,
            config=FactoredCoVeConfig(answer_mode="long", minimum_revision_similarity=0.5),
        ).verify(prediction)

        self.assertEqual(revised.answer, "Corrected answer. Stable detail.")
        self.assertEqual(revised.claim_checks[0].claim_text, "Corrected answer")
        self.assertTrue(revised.verification.metadata["revision_guard"]["accepted"])
        self.assertTrue(any("Source question: Who wrote the work?" in prompt for prompt in generator.prompts))

    def test_factored_verification_batches_targeted_retrieval(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-batch",
            pipeline="rag_verification",
            question="Check both facts.",
            answer="First fact. Second fact.",
            claim_checks=[
                {"claim_id": "c1", "claim_text": "First fact."},
                {"claim_id": "c2", "claim_text": "Second fact."},
            ],
        )
        retriever = BatchVerificationRetriever()
        verifier = FactoredCoVeVerifier(
            judge=FixedJudge("supported", 0.99),
            retriever=retriever,
            config=FactoredCoVeConfig(retrieval_top_k=3),
        )

        checked = verifier.verify(prediction)

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1], 3)
        self.assertEqual(len(retriever.calls[0][0]), 2)
        self.assertEqual(
            [question.citations[0].chunk_id for question in checked.verification.questions],
            ["batch", "batch"],
        )

    def test_factored_verification_repairs_incomplete_batch_json(self) -> None:
        prediction = PredictionRecord(
            example_id="ex-schema-repair",
            pipeline="rag_verification",
            question="Check both facts.",
            answer="First fact. Second fact.",
            claim_checks=[
                {"claim_id": "c1", "claim_text": "First fact."},
                {"claim_id": "c2", "claim_text": "Second fact."},
            ],
        )
        generator = RepairingBatchVerificationGenerator()

        checked = FactoredCoVeVerifier(
            judge=FixedJudge("supported", 0.99),
            generator=generator,
            config=FactoredCoVeConfig(fail_on_generator_error=True, generator_schema_retries=2),
        ).verify(prediction)

        self.assertEqual(generator.plan_calls, 2)
        self.assertEqual(generator.answer_calls, 2)
        self.assertEqual(checked.verification.metadata["generator_schema_repair_count"], 2)
        self.assertEqual(checked.metrics["verification_generator_schema_repair_count"], 2.0)
        self.assertEqual([row.answer for row in checked.verification.questions], ["First answer", "Second answer"])

    def test_stage_cache_replays_only_matching_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = StageArtifactCache(Path(tmp))
            saved = cache.save(
                example_id="ex:1",
                stage="retrieval",
                method="dense",
                input_value={"query": "Dune"},
                payload={"results": [{"chunk_id": "a"}]},
            )

            loaded = cache.load(
                example_id="ex:1", stage="retrieval", input_value={"query": "Dune"}
            )
            stale = cache.load(
                example_id="ex:1", stage="retrieval", input_value={"query": "Foundation"}
            )

        self.assertEqual(loaded.output_hash, saved.output_hash)
        self.assertIsNone(stale)

    def test_track_metrics_have_confidence_intervals_but_no_blended_score(self) -> None:
        rows = [{"f1": 1.0, "em": 1.0}, {"f1": 0.0, "em": 0.0}]
        aggregate = {"f1": 0.5, "em": 0.5}

        grouped = group_track_metrics("short_answer", aggregate)
        intervals = bootstrap_confidence_intervals(rows, keys=("f1",), samples=50, seed=7)

        self.assertIsNone(grouped["blended_score"])
        self.assertEqual(intervals["f1"]["n"], 2)
        self.assertLessEqual(intervals["f1"]["ci95_low"], 0.5)
        self.assertGreaterEqual(intervals["f1"]["ci95_high"], 0.5)


if __name__ == "__main__":
    unittest.main()
