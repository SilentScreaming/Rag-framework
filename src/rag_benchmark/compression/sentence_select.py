"""Sentence-level context compression with provenance preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_benchmark.retrieval.bm25 import RetrievedChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class CompressedSentence:
    context_id: str
    chunk_id: str
    doc_id: str
    title: str | None
    sentence_id: int | None
    text: str
    parent_rank: int
    parent_score: float
    selection_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_block(self) -> str:
        title = self.title or "Untitled"
        return f"[{self.context_id}] Title: {title}\nText: {self.text}"


@dataclass(frozen=True)
class CompressedContext:
    query: str
    sentences: list[CompressedSentence]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        return "\n\n".join(sentence.prompt_block() for sentence in self.sentences)

    def provenance(self) -> list[dict[str, Any]]:
        return [
            {
                "context_id": sentence.context_id,
                "chunk_id": sentence.chunk_id,
                "doc_id": sentence.doc_id,
                "title": sentence.title,
                "sentence_id": sentence.sentence_id,
                "parent_rank": sentence.parent_rank,
                "parent_score": sentence.parent_score,
            }
            for sentence in self.sentences
        ]


@dataclass(frozen=True)
class SentenceCompressionConfig:
    max_sentences: int = 16
    per_chunk_limit: int | None = None
    max_tokens: int | None = None
    min_chunks: int = 1
    mmr_lambda: float = 0.8
    semantic_weight: float = 1.0
    lexical_weight: float = 0.35
    rank_weight: float = 0.15

    def validate(self) -> None:
        if self.max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        if self.per_chunk_limit is not None and self.per_chunk_limit < 1:
            raise ValueError("per_chunk_limit must be >= 1 when provided")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided")
        if self.min_chunks < 1:
            raise ValueError("min_chunks must be >= 1")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")


class SentenceScorer(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one semantic relevance score per sentence."""


class SentenceTopNCompressor:
    """Selects top-N sentences from ranked chunks using lightweight signals."""

    def __init__(
        self,
        config: SentenceCompressionConfig | None = None,
        *,
        semantic_scorer: SentenceScorer | None = None,
    ) -> None:
        self.config = config or SentenceCompressionConfig()
        self.config.validate()
        self.semantic_scorer = semantic_scorer

    def compress(
        self,
        *,
        query: str,
        retrieved: list[RetrievedChunk],
        queries: list[str] | tuple[str, ...] | None = None,
    ) -> CompressedContext:
        active_queries = list(queries or [query])
        query_token_sets = [set(tokenize(value)) for value in active_queries]
        candidates: list[CompressedSentence] = []
        for result in retrieved:
            sentences = extract_sentences(result)
            for local_index, sentence in enumerate(sentences):
                sentence_tokens = set(tokenize(sentence["text"]))
                query_coverages = [
                    len(tokens & sentence_tokens) / max(len(tokens), 1)
                    for tokens in query_token_sets
                ]
                lexical_score = max(query_coverages, default=0.0)
                rank_bonus = 1.0 / result.rank
                selection_score = (
                    self.config.lexical_weight * lexical_score
                    + self.config.rank_weight * rank_bonus
                )
                sentence_id = sentence.get("sentence_id")
                context_id = context_identifier(
                    rank=result.rank,
                    chunk_id=result.chunk.chunk_id,
                    sentence_id=sentence_id,
                    fallback_index=local_index,
                )
                candidates.append(
                    CompressedSentence(
                        context_id=context_id,
                        chunk_id=result.chunk.chunk_id,
                        doc_id=result.chunk.doc_id,
                        title=result.chunk.title,
                        sentence_id=sentence_id,
                        text=sentence["text"],
                        parent_rank=result.rank,
                        parent_score=result.score,
                        selection_score=selection_score,
                        metadata={
                            "query_coverages": query_coverages,
                            "lexical_score": lexical_score,
                            "parent_metadata": dict(result.metadata),
                        },
                    )
                )
        if self.semantic_scorer is not None and candidates:
            semantic_scores = self.semantic_scorer.score(query, [candidate.text for candidate in candidates])
            if len(semantic_scores) != len(candidates):
                raise ValueError("semantic sentence scorer returned an invalid number of scores")
            candidates = [
                CompressedSentence(
                    **{
                        **candidate.__dict__,
                        "selection_score": candidate.selection_score + self.config.semantic_weight * score,
                        "metadata": {
                            **candidate.metadata,
                            "semantic_score": score,
                            "semantic_scorer": self.semantic_scorer.name,
                        },
                    }
                )
                for candidate, score in zip(candidates, semantic_scores)
            ]
        selected = select_evidence_set(candidates, config=self.config)
        selected = sorted(
            selected,
            key=lambda item: (
                item.parent_rank,
                item.sentence_id if item.sentence_id is not None else 10**9,
                item.context_id,
            ),
        )
        return CompressedContext(
            query=query,
            sentences=selected,
            metadata={
                "compressor": "evidence_set_mmr",
                "candidate_sentences": len(candidates),
                "selected_sentences": len(selected),
                "max_sentences": self.config.max_sentences,
                "max_tokens": self.config.max_tokens,
                "selected_tokens": sum(approximate_tokens(sentence.text) for sentence in selected),
                "source_chunks": len({sentence.chunk_id for sentence in selected}),
                "query_count": len(active_queries),
                "semantic_scorer": self.semantic_scorer.name if self.semantic_scorer else None,
            },
        )


def select_evidence_set(
    candidates: list[CompressedSentence],
    *,
    config: SentenceCompressionConfig,
) -> list[CompressedSentence]:
    remaining = sorted(
        candidates,
        key=lambda item: (-item.selection_score, item.parent_rank, item.context_id),
    )
    selected: list[CompressedSentence] = []
    per_chunk_counts: dict[str, int] = {}
    used_tokens = 0
    while remaining and len(selected) < config.max_sentences:
        eligible = [
            candidate
            for candidate in remaining
            if config.per_chunk_limit is None
            or per_chunk_counts.get(candidate.chunk_id, 0) < config.per_chunk_limit
        ]
        if not eligible:
            break
        under_covered = len({item.chunk_id for item in selected}) < config.min_chunks
        if under_covered:
            unseen = [candidate for candidate in eligible if candidate.chunk_id not in per_chunk_counts]
            if unseen:
                eligible = unseen
        best = max(
            eligible,
            key=lambda candidate: (
                mmr_score(candidate, selected=selected, config=config),
                -candidate.parent_rank,
                candidate.context_id,
            ),
        )
        token_count = approximate_tokens(best.text)
        if config.max_tokens is not None and used_tokens + token_count > config.max_tokens:
            remaining.remove(best)
            continue
        selected.append(best)
        used_tokens += token_count
        per_chunk_counts[best.chunk_id] = per_chunk_counts.get(best.chunk_id, 0) + 1
        remaining.remove(best)
    return selected


def mmr_score(
    candidate: CompressedSentence,
    *,
    selected: list[CompressedSentence],
    config: SentenceCompressionConfig,
) -> float:
    redundancy = max((jaccard(candidate.text, item.text) for item in selected), default=0.0)
    return config.mmr_lambda * candidate.selection_score - (1.0 - config.mmr_lambda) * redundancy


def jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def approximate_tokens(text: str) -> int:
    return max(1, len(tokenize(text)))


def extract_sentences(result: RetrievedChunk) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(result.chunk.sentences):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        sentence_id = row.get("sentence_id", index)
        rows.append({"sentence_id": int(sentence_id), "text": text})
    if rows:
        return rows
    text = result.chunk.text.strip()
    return [{"sentence_id": None, "text": text}] if text else []


def context_identifier(*, rank: int, chunk_id: str, sentence_id: int | None, fallback_index: int) -> str:
    from hashlib import sha256
    suffix = sentence_id if sentence_id is not None else fallback_index
    compact_chunk_id = (sha256(chunk_id.encode("utf-8")).hexdigest()[:12] if chunk_id.startswith("rgb:chunk:")
                        else re.sub(r"[^A-Za-z0-9]+", "-", chunk_id).strip("-")[-16:] or "chunk")
    return f"C{rank}-S{suffix}-{compact_chunk_id}"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
