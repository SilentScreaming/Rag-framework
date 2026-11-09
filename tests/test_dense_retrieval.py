from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.dense import (
    DenseRetriever,
    DenseVectorIndex,
    SentenceTransformerEmbedder,
)


class ToyEmbedder:
    name = "toy-embedder"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("alpha" in lowered),
                    float("beta" in lowered),
                    float("gamma" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def make_corpus() -> list[CorpusChunk]:
    return [
        CorpusChunk("c1", "d1", "test", "Alpha", "alpha topic", [], {}),
        CorpusChunk("c2", "d2", "test", "Beta", "beta topic", [], {}),
        CorpusChunk("c3", "d3", "test", "Gamma", "gamma topic", [], {}),
    ]


class DenseRetrievalTest(unittest.TestCase):
    def tearDown(self) -> None:
        SentenceTransformerEmbedder._MODEL_CACHE.clear()

    def test_dense_retriever_returns_nearest_chunks(self) -> None:
        retriever = DenseRetriever(ToyEmbedder())
        retriever.build(make_corpus())

        results = retriever.retrieve("alpha", top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].metadata["retriever"], "dense")
        self.assertEqual(results[0].metadata["embedding_model"], "toy-embedder")

    def test_dense_retrieve_many_matches_individual_queries(self) -> None:
        retriever = DenseRetriever(ToyEmbedder())
        retriever.build(make_corpus())

        batched = retriever.retrieve_many(["alpha", "beta"], top_k=2)
        individual = [
            retriever.retrieve(query, top_k=2)
            for query in ["alpha", "beta"]
        ]

        self.assertEqual(
            [[row.chunk.chunk_id for row in results] for results in batched],
            [[row.chunk.chunk_id for row in results] for results in individual],
        )
        self.assertEqual(
            [[row.score for row in results] for results in batched],
            [[row.score for row in results] for results in individual],
        )
        self.assertEqual(retriever.retrieve_many([], top_k=2), [])

    def test_dense_index_search_many_rejects_invalid_shape(self) -> None:
        retriever = DenseRetriever(ToyEmbedder())
        retriever.build(make_corpus())

        with self.assertRaises(ValueError):
            retriever.index.search_many(np.asarray([1.0, 0.0, 0.0]), top_k=1)

    def test_dense_retriever_requires_build(self) -> None:
        retriever = DenseRetriever(ToyEmbedder())

        with self.assertRaises(RuntimeError):
            retriever.retrieve("alpha", top_k=1)

    def test_dense_index_round_trip(self) -> None:
        corpus = make_corpus()
        retriever = DenseRetriever(ToyEmbedder())
        retriever.build(corpus)
        corpus_by_id = {chunk.chunk_id: chunk for chunk in corpus}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense-index"
            retriever.save_index(path)
            loaded = DenseVectorIndex.load(path, corpus_by_id=corpus_by_id)

        results = loaded.search(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_sentence_transformer_model_is_shared_across_scoped_retrievers(self) -> None:
        created = []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs) -> None:
                created.append((args, kwargs))

        module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = module
        try:
            first = SentenceTransformerEmbedder("test/model", revision="frozen")
            second = SentenceTransformerEmbedder("test/model", revision="frozen")
            self.assertIs(first._get_model(), second._get_model())
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original

        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
