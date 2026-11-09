from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)


def chunk(chunk_id: str, text: str | None = None) -> CorpusChunk:
    return CorpusChunk(chunk_id, f"doc-{chunk_id}", "test", chunk_id, text or chunk_id, [], {})


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class BatchFakeRetriever(FakeRetriever):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        super().__init__(results)
        self.batch_calls: list[tuple[list[str], int]] = []

    def retrieve_many(self, queries: list[str], *, top_k: int) -> list[list[RetrievedChunk]]:
        self.batch_calls.append((queries, top_k))
        return [self.results[:top_k] for _ in queries]


class HybridRetrievalTest(unittest.TestCase):
    def test_rrf_prefers_chunks_seen_by_both_retrievers(self) -> None:
        c1 = chunk("c1")
        c2 = chunk("c2")
        c3 = chunk("c3")
        lexical = [
            RetrievedChunk(c1, rank=1, score=10.0, metadata={"source": "bm25"}),
            RetrievedChunk(c2, rank=2, score=5.0, metadata={"source": "bm25"}),
        ]
        dense = [
            RetrievedChunk(c3, rank=1, score=0.9, metadata={"source": "dense"}),
            RetrievedChunk(c1, rank=2, score=0.8, metadata={"source": "dense"}),
        ]

        results = reciprocal_rank_fusion(
            lexical_results=lexical,
            dense_results=dense,
            final_top_k=3,
            rrf_k=60,
        )

        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertEqual(results[0].metadata["retriever"], "hybrid_rrf")
        self.assertEqual(results[0].metadata["bm25_rank"], 1)
        self.assertEqual(results[0].metadata["dense_rank"], 2)

    def test_hybrid_retriever_calls_both_retrievers(self) -> None:
        lexical = FakeRetriever([RetrievedChunk(chunk("c1"), rank=1, score=2.0)])
        dense = FakeRetriever([RetrievedChunk(chunk("c2"), rank=1, score=1.0)])
        retriever = HybridRetriever(
            lexical_retriever=lexical,
            dense_retriever=dense,
            config=HybridRetrievalConfig(bm25_top_k=7, dense_top_k=5, final_top_k=2),
        )

        results = retriever.retrieve("query")

        self.assertEqual(lexical.calls, [("query", 7)])
        self.assertEqual(dense.calls, [("query", 5)])
        self.assertEqual(len(results), 2)

    def test_hybrid_retrieve_many_uses_batch_interfaces(self) -> None:
        lexical = BatchFakeRetriever([RetrievedChunk(chunk("c1"), rank=1, score=2.0)])
        dense = BatchFakeRetriever([RetrievedChunk(chunk("c2"), rank=1, score=1.0)])
        retriever = HybridRetriever(
            lexical_retriever=lexical,
            dense_retriever=dense,
            config=HybridRetrievalConfig(bm25_top_k=7, dense_top_k=5, final_top_k=2),
        )

        results = retriever.retrieve_many(["first", "second"])

        self.assertEqual(lexical.batch_calls, [(["first", "second"], 7)])
        self.assertEqual(dense.batch_calls, [(["first", "second"], 5)])
        self.assertEqual([len(row) for row in results], [2, 2])

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetrievalConfig(final_top_k=0).validate()


if __name__ == "__main__":
    unittest.main()
