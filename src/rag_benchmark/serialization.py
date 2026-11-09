"""Serialize retrieved evidence and retrieval diagnostics."""
from __future__ import annotations
from typing import Any
from rag_benchmark.data import Example, Qrel, gold_chunk_ids
from rag_benchmark.evaluation.basic import retrieval_scores
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.schemas import RetrievalTraceRecord, RetrievedChunkRecord


def build_retrieval_trace_record(
    *,
    example: Example,
    qrels: list[Qrel],
    retrieved: list[RetrievedChunk],
    retrieval_ms: float,
    metric_ks: tuple[int, ...],
) -> RetrievalTraceRecord:
    retrieved_ids = [result.chunk.chunk_id for result in retrieved]
    gold_ids = gold_chunk_ids(qrels)
    metrics: dict[str, Any] = {}
    for k in metric_ks:
        metrics.update(retrieval_scores(retrieved_ids, gold_ids, k=k))
    metrics["retrieval_ms"] = retrieval_ms
    return RetrievalTraceRecord(
        example_id=example.example_id,
        question=example.question,
        gold_chunk_ids=sorted(gold_ids),
        retrieved_chunks=serialize_retrieved_chunks(retrieved, qrels),
        metrics={key: value for key, value in metrics.items() if value is not None},
    )


def serialize_retrieved_chunks(results: list[RetrievedChunk], qrels: list[Qrel]) -> list[RetrievedChunkRecord]:
    gold_by_chunk: dict[str, list[Qrel]] = {}
    for qrel in qrels:
        gold_by_chunk.setdefault(qrel.chunk_id, []).append(qrel)
    rows = []
    for result in results:
        qrel_rows = gold_by_chunk.get(result.chunk.chunk_id, [])
        rows.append(
            RetrievedChunkRecord(
                chunk_id=result.chunk.chunk_id,
                rank=result.rank,
                score=result.score,
                title=result.chunk.title,
                text=result.chunk.text,
                is_gold=bool(qrel_rows),
                gold_sent_ids=[
                    int(qrel.metadata["sentence_id"])
                    for qrel in qrel_rows
                    if qrel.metadata.get("sentence_id") is not None
                ],
                metadata=dict(result.metadata),
            )
        )
    return rows
