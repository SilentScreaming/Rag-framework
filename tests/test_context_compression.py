from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.compression.sentence_select import (
    SentenceCompressionConfig,
    SentenceTopNCompressor,
)
from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk


def retrieved_chunk() -> RetrievedChunk:
    chunk = CorpusChunk(
        chunk_id="hotpotqa:chunk:alpha",
        doc_id="doc-alpha",
        dataset="hotpotqa",
        title="Alpha",
        text="Alpha wrote the book. Beta painted a mural.",
        sentences=[
            {"sentence_id": 0, "text": "Alpha wrote the book."},
            {"sentence_id": 1, "text": "Beta painted a mural."},
        ],
        metadata={},
    )
    return RetrievedChunk(chunk=chunk, rank=1, score=4.0, metadata={"retriever": "hybrid_rrf"})


class ContextCompressionTest(unittest.TestCase):
    def test_sentence_top_n_selects_relevant_sentence_with_provenance(self) -> None:
        compressor = SentenceTopNCompressor(SentenceCompressionConfig(max_sentences=1))

        context = compressor.compress(query="Who wrote the book?", retrieved=[retrieved_chunk()])

        self.assertEqual(len(context.sentences), 1)
        sentence = context.sentences[0]
        self.assertEqual(sentence.chunk_id, "hotpotqa:chunk:alpha")
        self.assertEqual(sentence.doc_id, "doc-alpha")
        self.assertEqual(sentence.sentence_id, 0)
        self.assertIn("Alpha wrote", sentence.text)
        self.assertEqual(sentence.metadata["parent_metadata"]["retriever"], "hybrid_rrf")

    def test_prompt_context_contains_context_ids(self) -> None:
        compressor = SentenceTopNCompressor(SentenceCompressionConfig(max_sentences=1))

        context = compressor.compress(query="book", retrieved=[retrieved_chunk()])
        prompt_context = context.to_prompt_context()

        self.assertIn("[C1-S0-", prompt_context)
        self.assertIn("Title: Alpha", prompt_context)

    def test_falls_back_to_chunk_text_when_sentences_missing(self) -> None:
        chunk = CorpusChunk("c1", "d1", "test", "Fallback", "Single fallback text.", [], {})
        retrieved = RetrievedChunk(chunk=chunk, rank=1, score=1.0)
        compressor = SentenceTopNCompressor(SentenceCompressionConfig(max_sentences=1))

        context = compressor.compress(query="fallback", retrieved=[retrieved])

        self.assertEqual(context.sentences[0].sentence_id, None)
        self.assertEqual(context.sentences[0].text, "Single fallback text.")

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SentenceCompressionConfig(max_sentences=0).validate()


if __name__ == "__main__":
    unittest.main()
