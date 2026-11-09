from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.schemas import (
    CorpusChunkRecord,
    ExampleRecord,
    PredictionRecord,
    QrelRecord,
    RetrievalTraceRecord,
    RunConfigRecord,
    read_jsonl_records,
    write_jsonl_records,
)


class SchemaTest(unittest.TestCase):
    def test_dataset_records_accept_canonical_rows(self) -> None:
        example = ExampleRecord(
            example_id="hotpotqa:dev:1",
            dataset="hotpotqa",
            split="dev_fullwiki",
            task_type="multi_hop_document_qa",
            question="Who wrote the book?",
            answers=[{"text": "Ada", "type": "short"}],
            gold_evidence=[
                {
                    "doc_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "title": "Ada",
                    "sentence_id": 0,
                    "text": "Ada wrote the book.",
                    "support_type": "supporting_fact",
                }
            ],
        )
        chunk = CorpusChunkRecord(
            chunk_id="chunk-1",
            doc_id="doc-1",
            dataset="hotpotqa",
            title="Ada",
            text="Ada wrote the book.",
            sentences=[{"sentence_id": 0, "text": "Ada wrote the book."}],
        )
        qrel = QrelRecord(example_id=example.example_id, chunk_id=chunk.chunk_id, relevance=3)

        self.assertEqual(example.schema_version, "example-v1")
        self.assertEqual(chunk.sentences[0].sentence_id, 0)
        self.assertEqual(qrel.relevance, 3)

    def test_prediction_and_retrieval_trace_accept_canonical_shape(self) -> None:
        retrieved_chunk = {
            "chunk_id": "chunk-1",
            "rank": 1,
            "score": 12.5,
            "title": "Ada",
            "text": "Ada wrote the book.",
            "is_gold": True,
            "gold_sent_ids": [0],
        }
        prediction = PredictionRecord(
            example_id="hotpotqa:dev:1",
            pipeline="bm25_rag",
            question="Who wrote the book?",
            gold_answers=[{"text": "Ada", "type": "short"}],
            answer="Ada",
            retrieved_chunks=[retrieved_chunk],
            citations=[],
            claim_checks=[],
            verification=None,
            latency={"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
            tokens={"input_tokens": 10, "output_tokens": 2},
            cost={"estimated_cost": 0.01},
            generator={"provider": "mock", "model": "mock-v0"},
            metrics={"em": 1.0, "f1": 1.0},
        )
        trace = RetrievalTraceRecord(
            example_id="hotpotqa:dev:1",
            question="Who wrote the book?",
            gold_chunk_ids=["chunk-1"],
            retrieved_chunks=[retrieved_chunk],
            metrics={"recall@1": 1.0},
        )

        self.assertEqual(prediction.retrieved_chunks[0].rank, 1)
        self.assertIsNone(prediction.verification)
        self.assertTrue(trace.retrieved_chunks[0].is_gold)

    def test_run_config_accepts_record_shape(self) -> None:
        config = RunConfigRecord(
            run_id="run-1",
            timestamp="2026-06-29T00:00:00+00:00",
            git_commit="abc123+dirty",
            dataset_name="hotpotqa",
            split_name="example",
            candidate_scope="per_example_pooled_context",
            num_examples=100,
            candidate_corpus_chunks=373,
            random_seed=42,
            pipeline_names=["closed_book", "bm25_rag"],
            retriever={"type": "bm25", "top_k": 10},
            generator={"type": "openai", "model": "fixture-model"},
            metrics={"qa": ["em", "f1"]},
            environment={"python": "3.11"},
        )

        self.assertEqual(config.pipeline_names, ["closed_book", "bm25_rag"])

    def test_jsonl_round_trip(self) -> None:
        rows = [
            QrelRecord(example_id="ex-1", chunk_id="chunk-1", relevance=3),
            QrelRecord(example_id="ex-2", chunk_id="chunk-2", relevance=2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.jsonl"
            self.assertEqual(write_jsonl_records(path, rows), 2)
            loaded = read_jsonl_records(path, QrelRecord)

        self.assertEqual([row.example_id for row in loaded], ["ex-1", "ex-2"])

    def test_invalid_qrel_relevance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QrelRecord(example_id="ex-1", chunk_id="chunk-1", relevance=4)


if __name__ == "__main__":
    unittest.main()
