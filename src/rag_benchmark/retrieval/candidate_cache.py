"""Disk-backed KILT FullWiki page retrieval and frozen candidate caches."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from rag_benchmark.dataset_adapters import SECTION_RE, clean_text
from rag_benchmark.data import CorpusChunk


_QUERY_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
}


@dataclass(frozen=True)
class FullWikiPageHit:
    row_id: int
    wikipedia_id: str
    title: str
    byte_offset: int
    byte_length: int
    score: float
    query_mode: str = "fts"


@dataclass(frozen=True)
class FrozenCandidateBundle:
    chunks: list[CorpusChunk]
    vectors: np.ndarray | None
    embedding_model: str | None
    scope_hash: str


class FullWikiPageIndex:
    """Searches a contentless FTS5 index and resolves hits into the raw KILT file."""

    def __init__(self, index_path: Path, *, raw_path: Path | None = None) -> None:
        self.index_path = index_path.resolve()
        if not self.index_path.exists():
            raise FileNotFoundError(f"FullWiki FTS index not found: {self.index_path}")
        self.connection = sqlite3.connect(
            f"file:{self.index_path.as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row
        metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
        configured_raw = metadata.get("source_path")
        self.raw_path = (raw_path or (Path(configured_raw) if configured_raw else None))
        if self.raw_path is None or not self.raw_path.exists():
            raise FileNotFoundError("KILT knowledge source path is missing from the FTS index metadata")
        self.raw_path = self.raw_path.resolve()
        self._raw_handle: BinaryIO | None = None

    def search(self, query: str, *, top_k: int) -> list[FullWikiPageHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_plans = build_fts_match_queries(query)
        if not query_plans:
            return []
        fused: dict[int, tuple[float, sqlite3.Row, str]] = {}
        per_plan_limit = max(10, top_k)
        for mode, match_query in query_plans:
            rows = self.connection.execute(
                "SELECT p.rowid, p.wikipedia_id, p.title, p.byte_offset, p.byte_length, "
                "bm25(pages_fts, 8.0, 1.0) AS score "
                "FROM pages_fts JOIN pages AS p ON p.rowid = pages_fts.rowid "
                "WHERE pages_fts MATCH ? ORDER BY score, p.rowid LIMIT ?",
                (match_query, per_plan_limit),
            ).fetchall()
            for rank, row in enumerate(rows, start=1):
                row_id = int(row["rowid"])
                previous_score = fused.get(row_id, (0.0, row, mode))[0]
                fused[row_id] = (previous_score + 1.0 / (60 + rank), row, mode)
        ranked = sorted(fused.values(), key=lambda item: (-item[0], int(item[1]["rowid"])))[:top_k]
        return [self._page_hit(row, score=score, query_mode=mode) for score, row, mode in ranked]

    def get_by_wikipedia_ids(self, wikipedia_ids: list[str]) -> list[FullWikiPageHit]:
        unique_ids = list(dict.fromkeys(str(value) for value in wikipedia_ids if str(value)))
        rows_by_id: dict[str, sqlite3.Row] = {}
        for start in range(0, len(unique_ids), 900):
            batch = unique_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            for row in self.connection.execute(
                "SELECT rowid, wikipedia_id, title, byte_offset, byte_length FROM pages "
                f"WHERE wikipedia_id IN ({placeholders})",
                batch,
            ):
                rows_by_id[str(row["wikipedia_id"])] = row
        return [
            self._page_hit(rows_by_id[wiki_id], score=0.0, query_mode="anchor_expansion")
            for wiki_id in unique_ids
            if wiki_id in rows_by_id
        ]

    @staticmethod
    def _page_hit(row: sqlite3.Row, *, score: float, query_mode: str) -> FullWikiPageHit:
        return FullWikiPageHit(
            row_id=int(row["rowid"]),
            wikipedia_id=str(row["wikipedia_id"]),
            title=str(row["title"] or ""),
            byte_offset=int(row["byte_offset"]),
            byte_length=int(row["byte_length"]),
            score=float(score),
            query_mode=query_mode,
        )

    def load_page(self, hit: FullWikiPageHit) -> dict[str, Any]:
        if self._raw_handle is None:
            self._raw_handle = self.raw_path.open("rb")
        self._raw_handle.seek(hit.byte_offset)
        payload = self._raw_handle.read(hit.byte_length)
        return json.loads(payload.decode("utf-8"))

    def close(self) -> None:
        if self._raw_handle is not None:
            self._raw_handle.close()
            self._raw_handle = None
        self.connection.close()

    def __enter__(self) -> "FullWikiPageIndex":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class FrozenCandidateCache:
    """Loads per-example candidates and optional frozen document vectors."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.database_path = self.root / "candidates.sqlite"
        self.manifest_path = self.root / "manifest.json"
        if not self.database_path.exists() or not self.manifest_path.exists():
            raise FileNotFoundError(f"Frozen candidate cache is incomplete: {self.root}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "kilt-candidate-cache-v1":
            raise ValueError(f"Unsupported candidate cache schema: {self.manifest.get('schema_version')!r}")
        self.connection = sqlite3.connect(
            f"file:{self.database_path.as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row
        vectors_name = self.manifest.get("vectors_file")
        self._vectors = (
            np.load(self.root / str(vectors_name), mmap_mode="r") if vectors_name else None
        )
        self.embedding_model = self.manifest.get("embedding_model")
        self._closed = False
        if self._vectors is not None and not self.embedding_model:
            raise ValueError("Candidate cache vectors exist without an embedding_model")

    @property
    def scope_hash(self) -> str:
        return str(self.manifest.get("cache_hash") or self.manifest.get("content_hash") or "unknown")

    def load(self, example_id: str) -> FrozenCandidateBundle:
        rows = self.connection.execute(
            "SELECT d.payload_json, d.vector_row FROM candidates AS c "
            "JOIN documents AS d ON d.chunk_id = c.chunk_id "
            "WHERE c.example_id = ? ORDER BY c.candidate_rank",
            (example_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"Candidate cache has no candidates for example {example_id!r}")
        chunks = [corpus_chunk_from_json(json.loads(row["payload_json"])) for row in rows]
        vectors = None
        if self._vectors is not None:
            vector_rows = [row["vector_row"] for row in rows]
            if any(row is None for row in vector_rows):
                raise ValueError(f"Candidate cache has unembedded documents for {example_id!r}")
            vectors = np.asarray(self._vectors[np.asarray(vector_rows, dtype=np.int64)], dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(example_id.encode("utf-8"))
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\n")
        return FrozenCandidateBundle(
            chunks=chunks,
            vectors=vectors,
            embedding_model=str(self.embedding_model) if self.embedding_model else None,
            scope_hash=digest.hexdigest(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        if isinstance(self._vectors, np.memmap) and self._vectors._mmap is not None:
            self._vectors._mmap.close()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_fts_match_query(query: str, *, max_terms: int = 32) -> str:
    plans = build_fts_match_queries(query, max_terms=max_terms)
    return next((match_query for mode, match_query in plans if mode == "all_terms_and"), "")


def build_fts_match_queries(query: str, *, max_terms: int = 12) -> list[tuple[str, str]]:
    tokens = []
    seen = set()
    raw_tokens = _QUERY_TOKEN_RE.findall(query.casefold())
    for token in raw_tokens:
        if token in _STOP_WORDS or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_terms:
            break
    if not tokens:
        tokens = list(dict.fromkeys(token for token in raw_tokens if token))[:max_terms]
    quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    if not quoted:
        return []
    plans = []
    phrase_tokens = _QUERY_TOKEN_RE.findall(query.casefold())[:16]
    if len(phrase_tokens) >= 4:
        phrase_terms = [
            f'"{" ".join(phrase_tokens[start : start + 4])}"'
            for start in range(len(phrase_tokens) - 3)
        ]
        plans.append(("phrase_ngrams", f"({' OR '.join(phrase_terms)})"))
    plans.append(("all_terms_and", " AND ".join(quoted)))
    if len(quoted) >= 3:
        longest = sorted(tokens, key=lambda token: (-len(token), tokens.index(token)))[:2]
        anchors = [f'"{token}"' for token in longest]
        remaining = [value for token, value in zip(tokens, quoted) if token not in set(longest)]
        if remaining:
            plans.append(("rare_anchor", f"{' AND '.join(anchors)} AND ({' OR '.join(remaining)})"))
    title_terms = quoted[:8]
    plans.append(("title_terms", f"title:({' OR '.join(title_terms)})"))
    return plans


def canonical_chunks_from_kilt_page(
    page: dict[str, Any],
    *,
    max_words: int = 220,
    max_paragraphs: int = 3,
) -> list[CorpusChunk]:
    wiki_id = str(page.get("wikipedia_id") or page.get("_id"))
    title = clean_text(page.get("wikipedia_title"))
    groups: list[list[tuple[int, str, bool]]] = []
    current: list[tuple[int, str, bool]] = []
    current_words = 0
    for paragraph_id, raw_text in enumerate(list(page.get("text") or [])):
        text = clean_text(raw_text)
        if not text:
            continue
        section = SECTION_RE.match(text)
        if section:
            text = clean_text(section.group(1)) or "Section"
        words = len(text.split())
        if current and (len(current) >= max_paragraphs or current_words + words > max_words):
            groups.append(current)
            current = []
            current_words = 0
        current.append((paragraph_id, text, bool(section)))
        current_words += words
    if current:
        groups.append(current)

    chunks = []
    history = dict(page.get("history") or {})
    wikidata = dict(page.get("wikidata_info") or {})
    for group in groups:
        start, end = group[0][0], group[-1][0]
        chunks.append(
            CorpusChunk(
                chunk_id=f"kilt:wiki:{wiki_id}:p{start}-{end}",
                doc_id=f"kilt:wiki:{wiki_id}",
                dataset="kilt_wikipedia_2019",
                title=title,
                text="\n".join(text for _, text, _ in group),
                sentences=[
                    {
                        "sentence_id": paragraph_id,
                        "text": text,
                        "metadata": {"is_section_heading": is_heading},
                    }
                    for paragraph_id, text, is_heading in group
                ],
                metadata={
                    "wikipedia_id": wiki_id,
                    "paragraph_start": start,
                    "paragraph_end": end,
                    "revision_id": history.get("revid"),
                    "revision_timestamp": history.get("timestamp"),
                    "wikidata_id": wikidata.get("wikidata_id"),
                },
            )
        )
    return chunks


def corpus_chunk_to_json(chunk: CorpusChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "dataset": chunk.dataset,
        "title": chunk.title,
        "text": chunk.text,
        "sentences": chunk.sentences,
        "metadata": chunk.metadata,
    }


def corpus_chunk_from_json(row: dict[str, Any]) -> CorpusChunk:
    return CorpusChunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        dataset=str(row["dataset"]),
        title=row.get("title"),
        text=str(row["text"]),
        sentences=list(row.get("sentences", [])),
        metadata=dict(row.get("metadata", {})),
    )
