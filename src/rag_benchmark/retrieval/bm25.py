"""BM25 retrieval and shared retrieved-chunk contract."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from rag_benchmark.data import CorpusChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: CorpusChunk
    rank: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """BM25 retriever with explicit build/retrieve lifecycle."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        include_title: bool = True,
        title_weight: int = 2,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1 or epsilon < 0 or title_weight < 1:
            raise ValueError("invalid BM25 configuration")
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.include_title = include_title
        self.title_weight = title_weight
        self._chunks: list[CorpusChunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, corpus: list[CorpusChunk]) -> None:
        self._chunks = list(corpus)
        tokenized = []
        for chunk in self._chunks:
            tokens = tokenize(chunk.text)
            if self.include_title and chunk.title:
                tokens = tokenize(chunk.title) * self.title_weight + tokens
            tokenized.append(tokens)
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.build must be called before retrieve")
        scores = self._bm25.get_scores(tokenize(query))
        indexed_scores = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results = []
        for rank, (index, score) in enumerate(indexed_scores[:top_k], start=1):
            score_float = float(score)
            if math.isnan(score_float):
                score_float = 0.0
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[index],
                    rank=rank,
                    score=score_float,
                    metadata={
                        "retriever": "bm25",
                        "bm25_k1": self.k1,
                        "bm25_b": self.b,
                        "title_weight": self.title_weight if self.include_title else 0,
                    },
                )
            )
        return results

    def retrieve_many(self, queries: list[str], *, top_k: int) -> list[list[RetrievedChunk]]:
        return [self.retrieve(query, top_k=top_k) for query in queries]
