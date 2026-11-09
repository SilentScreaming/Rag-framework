"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_benchmark.retrieval.bm25 import RetrievedChunk


class PairScorer(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text for a query."""


@dataclass
class SentenceTransformersCrossEncoderScorer:
    model_name: str
    batch_size: int = 16
    device: str | None = None
    revision: str | None = None
    max_length: int = 512
    trust_remote_code: bool = False
    cache_folder: str | None = None
    identity_name: str | None = None
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        identity = self.identity_name or self.model_name
        return f"{identity}@{self.revision}" if self.revision else identity

    def score(self, query: str, texts: list[str]) -> list[float]:
        model = self._get_model()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in np.asarray(scores).reshape(-1)]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            revision=(None if Path(self.model_name).exists() else self.revision),
            max_length=self.max_length,
            trust_remote_code=self.trust_remote_code,
            cache_folder=self.cache_folder,
        )
        return self._model


@dataclass(frozen=True)
class RerankerConfig:
    top_k: int = 8
    max_candidates: int | None = None
    include_title: bool = True

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("reranker top_k must be >= 1")
        if self.max_candidates is not None and self.max_candidates < self.top_k:
            raise ValueError("max_candidates must be >= top_k")


class CrossEncoderReranker:
    """Reranks retrieved chunks using a query/chunk pair scorer."""

    def __init__(self, scorer: PairScorer, *, config: RerankerConfig | None = None) -> None:
        self.scorer = scorer
        self.config = config or RerankerConfig()
        self.config.validate()

    def rerank(self, query: str, candidates: list[RetrievedChunk], *, top_k: int | None = None) -> list[RetrievedChunk]:
        if not candidates:
            return []
        limit = top_k or self.config.top_k
        if limit < 1:
            raise ValueError("top_k must be >= 1")
        candidates = candidates[: self.config.max_candidates] if self.config.max_candidates else candidates
        texts = [
            f"{candidate.chunk.title}\n{candidate.chunk.text}"
            if self.config.include_title and candidate.chunk.title
            else candidate.chunk.text
            for candidate in candidates
        ]
        scores = self.scorer.score(query, texts)
        if len(scores) != len(candidates):
            raise ValueError("reranker scorer returned a different number of scores than candidates")
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk.chunk_id))
        results = []
        for rank, (candidate, score) in enumerate(scored[:limit], start=1):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "reranker": self.scorer.name,
                    "rerank_score": float(score),
                    "rerank_rank": rank,
                    "pre_rerank_rank": candidate.rank,
                    "pre_rerank_score": candidate.score,
                }
            )
            results.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    rank=rank,
                    score=float(score),
                    metadata=metadata,
                )
            )
        return results
