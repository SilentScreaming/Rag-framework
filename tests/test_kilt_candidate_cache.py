from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from rag_benchmark.retrieval.candidate_cache import (
    FrozenCandidateCache,
    FullWikiPageIndex,
    build_fts_match_query,
    canonical_chunks_from_kilt_page,
    corpus_chunk_to_json,
)


class KiltCandidateCacheTest(unittest.TestCase):
    def test_fullwiki_page_index_resolves_raw_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "knowledge.jsonl"
            page = {
                "_id": "42",
                "wikipedia_id": "42",
                "wikipedia_title": "Alpha Project",
                "text": ["Alpha is a benchmark.\n", "It evaluates retrieval quality.\n"],
            }
            payload = (json.dumps(page) + "\n").encode("utf-8")
            raw_path.write_bytes(payload)
            index_path = root / "index.sqlite"
            with closing(sqlite3.connect(index_path)) as connection:
                connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES ('source_path', ?)", (str(raw_path),))
                connection.execute(
                    "CREATE TABLE pages(rowid INTEGER PRIMARY KEY, wikipedia_id TEXT, title TEXT, "
                    "byte_offset INTEGER, byte_length INTEGER)"
                )
                connection.execute("INSERT INTO pages VALUES (1, '42', 'Alpha Project', 0, ?)", (len(payload),))
                connection.execute(
                    "CREATE VIRTUAL TABLE pages_fts USING fts5(title, body, content='', "
                    "tokenize='porter unicode61 remove_diacritics 2')"
                )
                connection.execute(
                    "INSERT INTO pages_fts(rowid, title, body) VALUES (1, 'Alpha Project', "
                    "'Alpha is a benchmark that evaluates retrieval quality')"
                )
                connection.commit()

            with FullWikiPageIndex(index_path) as index:
                hits = index.search("What evaluates retrieval quality?", top_k=5)
                self.assertEqual([hit.wikipedia_id for hit in hits], ["42"])
                loaded = index.load_page(hits[0])
            self.assertEqual(loaded["wikipedia_title"], "Alpha Project")
            chunks = canonical_chunks_from_kilt_page(loaded)
            self.assertEqual(chunks[0].chunk_id, "kilt:wiki:42:p0-1")

    def test_frozen_cache_loads_vectors_in_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = {
                "_id": "42",
                "wikipedia_id": "42",
                "wikipedia_title": "Alpha Project",
                "text": ["Alpha is a benchmark."],
            }
            chunk = canonical_chunks_from_kilt_page(page)[0]
            with closing(sqlite3.connect(root / "candidates.sqlite")) as connection:
                connection.execute(
                    "CREATE TABLE documents(chunk_id TEXT PRIMARY KEY, payload_json TEXT, vector_row INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE candidates(example_id TEXT, candidate_rank INTEGER, chunk_id TEXT)"
                )
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, 0)",
                    (chunk.chunk_id, json.dumps(corpus_chunk_to_json(chunk))),
                )
                connection.execute("INSERT INTO candidates VALUES ('q1', 1, ?)", (chunk.chunk_id,))
                connection.commit()
            np.save(root / "vectors.npy", np.ones((1, 8), dtype=np.float16))
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "kilt-candidate-cache-v1",
                        "cache_hash": "abc",
                        "embedding_model": "test-embedder@revision",
                        "vectors_file": "vectors.npy",
                    }
                ),
                encoding="utf-8",
            )
            cache = FrozenCandidateCache(root)
            bundle = cache.load("q1")
            with self.assertRaises(sqlite3.OperationalError):
                cache.connection.execute("INSERT INTO candidates VALUES ('q2', 1, ?)", (chunk.chunk_id,))
            cache.close()
            self.assertEqual(bundle.chunks[0].chunk_id, chunk.chunk_id)
            self.assertEqual(bundle.vectors.shape, (1, 8))
            self.assertEqual(bundle.embedding_model, "test-embedder@revision")

    def test_fts_query_removes_common_question_words(self) -> None:
        query = build_fts_match_query("What is the capital of New Zealand?")
        self.assertIn('"capital"', query)
        self.assertIn('"new"', query)
        self.assertNotIn('"what"', query)


if __name__ == "__main__":
    unittest.main()
