from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.reranking.cross_encoder import CrossEncoderReranker, RerankerConfig


class KeywordScorer:
    name = "keyword-scorer"

    def score(self, query: str, texts: list[str]) -> list[float]:
        token = query.lower()
        return [float(token in text.lower()) for text in texts]


class BadScorer:
    name = "bad-scorer"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0]


def result(chunk_id: str, text: str, rank: int, score: float) -> RetrievedChunk:
    chunk = CorpusChunk(chunk_id, f"doc-{chunk_id}", "test", chunk_id, text, [], {})
    return RetrievedChunk(chunk=chunk, rank=rank, score=score, metadata={"retriever": "hybrid_rrf"})


class RerankingTest(unittest.TestCase):
    def test_reranker_orders_by_pair_score(self) -> None:
        candidates = [
            result("c1", "unrelated text", rank=1, score=0.9),
            result("c2", "alpha answer evidence", rank=2, score=0.8),
        ]
        reranker = CrossEncoderReranker(KeywordScorer(), config=RerankerConfig(top_k=2))

        reranked = reranker.rerank("alpha", candidates)

        self.assertEqual(reranked[0].chunk.chunk_id, "c2")
        self.assertEqual(reranked[0].rank, 1)
        self.assertEqual(reranked[0].score, 1.0)
        self.assertEqual(reranked[0].metadata["pre_rerank_rank"], 2)
        self.assertEqual(reranked[0].metadata["reranker"], "keyword-scorer")

    def test_reranker_applies_top_k(self) -> None:
        candidates = [
            result("c1", "alpha one", rank=1, score=0.9),
            result("c2", "alpha two", rank=2, score=0.8),
        ]
        reranker = CrossEncoderReranker(KeywordScorer(), config=RerankerConfig(top_k=1))

        reranked = reranker.rerank("alpha", candidates)

        self.assertEqual(len(reranked), 1)

    def test_scorer_length_mismatch_is_rejected(self) -> None:
        reranker = CrossEncoderReranker(BadScorer())

        with self.assertRaises(ValueError):
            reranker.rerank("alpha", [result("c1", "alpha", 1, 1.0), result("c2", "beta", 2, 0.5)])


if __name__ == "__main__":
    unittest.main()
