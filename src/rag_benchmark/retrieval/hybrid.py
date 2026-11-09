"""Hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_benchmark.retrieval.bm25 import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return ranked chunks for one query."""


@dataclass(frozen=True)
class HybridRetrievalConfig:
    bm25_top_k: int = 50
    dense_top_k: int = 50
    final_top_k: int = 12
    rrf_k: int = 60
    fusion: str = "rrf"
    lexical_weight: float = 0.5
    dense_weight: float = 0.5

    def validate(self) -> None:
        for name, value in (
            ("bm25_top_k", self.bm25_top_k),
            ("dense_top_k", self.dense_top_k),
            ("final_top_k", self.final_top_k),
            ("rrf_k", self.rrf_k),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.fusion not in {"rrf", "weighted"}:
            raise ValueError("fusion must be rrf or weighted")
        if self.lexical_weight < 0 or self.dense_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.lexical_weight + self.dense_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")


class HybridRetriever:
    """Combines lexical and dense candidates using reciprocal rank fusion."""

    def __init__(
        self,
        *,
        lexical_retriever: Retriever,
        dense_retriever: Retriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.config = config or HybridRetrievalConfig()
        self.config.validate()

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        final_top_k = top_k or self.config.final_top_k
        lexical = self.lexical_retriever.retrieve(query, top_k=self.config.bm25_top_k)
        dense = self.dense_retriever.retrieve(query, top_k=self.config.dense_top_k)
        if self.config.fusion == "weighted":
            return weighted_score_fusion(
                lexical_results=lexical,
                dense_results=dense,
                final_top_k=final_top_k,
                lexical_weight=self.config.lexical_weight,
                dense_weight=self.config.dense_weight,
            )
        return reciprocal_rank_fusion(
            lexical_results=lexical,
            dense_results=dense,
            final_top_k=final_top_k,
            rrf_k=self.config.rrf_k,
        )

    def retrieve_many(
        self,
        queries: list[str],
        *,
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        final_top_k = top_k or self.config.final_top_k
        lexical_many = getattr(self.lexical_retriever, "retrieve_many", None)
        dense_many = getattr(self.dense_retriever, "retrieve_many", None)
        lexical_rows = (
            lexical_many(queries, top_k=self.config.bm25_top_k)
            if callable(lexical_many)
            else [
                self.lexical_retriever.retrieve(query, top_k=self.config.bm25_top_k)
                for query in queries
            ]
        )
        dense_rows = (
            dense_many(queries, top_k=self.config.dense_top_k)
            if callable(dense_many)
            else [
                self.dense_retriever.retrieve(query, top_k=self.config.dense_top_k)
                for query in queries
            ]
        )
        if self.config.fusion == "weighted":
            return [
                weighted_score_fusion(
                    lexical_results=lexical,
                    dense_results=dense,
                    final_top_k=final_top_k,
                    lexical_weight=self.config.lexical_weight,
                    dense_weight=self.config.dense_weight,
                )
                for lexical, dense in zip(lexical_rows, dense_rows)
            ]
        return [
            reciprocal_rank_fusion(
                lexical_results=lexical,
                dense_results=dense,
                final_top_k=final_top_k,
                rrf_k=self.config.rrf_k,
            )
            for lexical, dense in zip(lexical_rows, dense_rows)
        ]


def reciprocal_rank_fusion(
    *,
    lexical_results: list[RetrievedChunk],
    dense_results: list[RetrievedChunk],
    final_top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")

    fused: dict[str, dict[str, object]] = {}
    for source_name, results in (("bm25", lexical_results), ("dense", dense_results)):
        for result in results:
            row = fused.setdefault(
                result.chunk.chunk_id,
                {
                    "chunk": result.chunk,
                    "score": 0.0,
                    "metadata": {},
                },
            )
            contribution = 1.0 / (rrf_k + result.rank)
            row["score"] = float(row["score"]) + contribution
            metadata = dict(row["metadata"])
            metadata[f"{source_name}_rank"] = result.rank
            metadata[f"{source_name}_score"] = result.score
            metadata[f"{source_name}_metadata"] = dict(result.metadata)
            row["metadata"] = metadata

    ranked = sorted(
        fused.values(),
        key=lambda row: (-float(row["score"]), str(row["chunk"].chunk_id)),
    )
    results = []
    for rank, row in enumerate(ranked[:final_top_k], start=1):
        metadata = dict(row["metadata"])
        metadata.update(
            {
                "retriever": "hybrid_rrf",
                "fusion": "rrf",
                "fusion_score": float(row["score"]),
                "fusion_rank": rank,
                "rrf_k": rrf_k,
            }
        )
        results.append(
            RetrievedChunk(
                chunk=row["chunk"],
                rank=rank,
                score=float(row["score"]),
                metadata=metadata,
            )
        )
    return results


def weighted_score_fusion(
    *,
    lexical_results: list[RetrievedChunk],
    dense_results: list[RetrievedChunk],
    final_top_k: int,
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Fuse calibrated per-list scores after min-max normalization."""

    fused: dict[str, dict[str, object]] = {}
    for source, results, weight in (
        ("bm25", lexical_results, lexical_weight),
        ("dense", dense_results, dense_weight),
    ):
        normalized = normalize_result_scores(results)
        for result, score in zip(results, normalized):
            row = fused.setdefault(
                result.chunk.chunk_id,
                {"chunk": result.chunk, "score": 0.0, "metadata": {}},
            )
            row["score"] = float(row["score"]) + weight * score
            metadata = dict(row["metadata"])
            metadata[f"{source}_rank"] = result.rank
            metadata[f"{source}_score"] = result.score
            metadata[f"{source}_normalized_score"] = score
            row["metadata"] = metadata
    ranked = sorted(fused.values(), key=lambda row: (-float(row["score"]), row["chunk"].chunk_id))
    return [
        RetrievedChunk(
            chunk=row["chunk"],
            rank=rank,
            score=float(row["score"]),
            metadata={
                **dict(row["metadata"]),
                "retriever": "hybrid_weighted",
                "fusion": "weighted",
                "fusion_rank": rank,
                "lexical_weight": lexical_weight,
                "dense_weight": dense_weight,
            },
        )
        for rank, row in enumerate(ranked[:final_top_k], start=1)
    ]


def fuse_multi_query_results(
    result_sets: list[tuple[str, list[RetrievedChunk]]],
    *,
    final_top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse results from the original query and additional retrieval queries."""

    fused: dict[str, dict[str, object]] = {}
    for query_index, (query, results) in enumerate(result_sets):
        for result in results:
            row = fused.setdefault(
                result.chunk.chunk_id,
                {"chunk": result.chunk, "score": 0.0, "metadata": {}, "queries": []},
            )
            row["score"] = float(row["score"]) + 1.0 / (rrf_k + result.rank)
            queries = list(row["queries"])
            queries.append({"query_index": query_index, "query": query, "rank": result.rank})
            row["queries"] = queries
            metadata = dict(row["metadata"])
            metadata.update(result.metadata)
            row["metadata"] = metadata
    ranked = sorted(fused.values(), key=lambda row: (-float(row["score"]), row["chunk"].chunk_id))
    return [
        RetrievedChunk(
            chunk=row["chunk"],
            rank=rank,
            score=float(row["score"]),
            metadata={
                **dict(row["metadata"]),
                "retriever": "multi_query_rrf",
                "query_hits": list(row["queries"]),
                "query_hit_count": len(row["queries"]),
                "fusion_rank": rank,
            },
        )
        for rank, row in enumerate(ranked[:final_top_k], start=1)
    ]


def normalize_result_scores(results: list[RetrievedChunk]) -> list[float]:
    if not results:
        return []
    scores = [float(result.score) for result in results]
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]
