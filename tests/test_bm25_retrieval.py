from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import BM25Retriever


class BM25RetrievalTest(unittest.TestCase):
    def test_bm25_smoke(self) -> None:
        corpus = [
            CorpusChunk("c1", "d1", "test", "Alpha", "alpha beta", [], {}),
            CorpusChunk("c2", "d2", "test", "Gamma", "gamma delta", [], {}),
        ]
        retriever = BM25Retriever()
        retriever.build(corpus)
        results = retriever.retrieve("alpha", top_k=1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")

        batched = retriever.retrieve_many(["alpha", "gamma"], top_k=1)
        individual = [retriever.retrieve(query, top_k=1) for query in ["alpha", "gamma"]]
        self.assertEqual(
            [[result.chunk.chunk_id for result in row] for row in batched],
            [[result.chunk.chunk_id for result in row] for row in individual],
        )


if __name__ == "__main__":
    unittest.main()
