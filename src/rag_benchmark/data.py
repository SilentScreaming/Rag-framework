"""Canonical benchmark data contracts and loaders, shared by all profiles."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Example:
    example_id: str
    dataset: str
    split: str
    task_type: str
    question: str
    answers: list[JsonObject]
    gold_evidence: list[JsonObject]
    metadata: JsonObject


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    dataset: str
    title: str | None
    text: str
    sentences: list[JsonObject]
    metadata: JsonObject


@dataclass(frozen=True)
class Qrel:
    example_id: str
    chunk_id: str
    relevance: int
    support_type: str
    metadata: JsonObject


@dataclass(frozen=True)
class CandidateResponse:
    response_id: str
    example_id: str
    dataset: str
    split: str
    text: str
    model: str | None
    metadata: JsonObject


@dataclass(frozen=True)
class ContextSet:
    context_set_id: str
    example_id: str
    dataset: str
    condition: str
    chunk_ids: list[str]
    expected_behavior: str
    metadata: JsonObject


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    example_id: str
    target_type: str
    target_id: str | None
    label: str
    start_char: int | None
    end_char: int | None
    text: str | None
    evidence_chunk_ids: list[str]
    metadata: JsonObject


@dataclass(frozen=True)
class DatasetSlice:
    dataset_name: str
    split_name: str
    candidate_scope: str
    examples: list[Example]
    corpus: list[CorpusChunk]
    qrels_by_example: dict[str, list[Qrel]]
    context_sets_by_example: dict[str, list[ContextSet]] = field(default_factory=dict)
    candidate_responses_by_example: dict[str, list[CandidateResponse]] = field(default_factory=dict)
    annotations_by_example: dict[str, list[Annotation]] = field(default_factory=dict)
    candidate_cache_path: Path | None = None


def read_jsonl(path: Path) -> Iterable[JsonObject]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonObject]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def load_examples(processed_root: Path) -> dict[str, Example]:
    examples = {}
    for row in read_jsonl(processed_root / "examples.jsonl"):
        example = Example(
            example_id=row["example_id"],
            dataset=row["dataset"],
            split=row["split"],
            task_type=row["task_type"],
            question=row["question"],
            answers=list(row.get("answers", [])),
            gold_evidence=list(row.get("gold_evidence", [])),
            metadata=dict(row.get("metadata", {})),
        )
        examples[example.example_id] = example
    return examples


def load_qrels(processed_root: Path, *, relevance_threshold: int = 3) -> dict[str, list[Qrel]]:
    qrels_by_example: dict[str, list[Qrel]] = defaultdict(list)
    for row in read_jsonl(processed_root / "qrels.jsonl"):
        relevance = int(row.get("relevance", 0))
        if relevance < relevance_threshold:
            continue
        qrels_by_example[row["example_id"]].append(
            Qrel(
                example_id=row["example_id"],
                chunk_id=row["chunk_id"],
                relevance=relevance,
                support_type=row.get("support_type", ""),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return dict(qrels_by_example)


def load_context_sets(
    processed_root: Path,
    *,
    example_ids: set[str] | None = None,
) -> dict[str, list[ContextSet]]:
    path = processed_root / "context_sets.jsonl"
    if not path.exists():
        return {}
    contexts_by_example: dict[str, list[ContextSet]] = defaultdict(list)
    for row in read_jsonl(path):
        example_id = str(row["example_id"])
        if example_ids is not None and example_id not in example_ids:
            continue
        contexts_by_example[example_id].append(
            ContextSet(
                context_set_id=row["context_set_id"],
                example_id=example_id,
                dataset=row["dataset"],
                condition=row["condition"],
                chunk_ids=list(row.get("chunk_ids", [])),
                expected_behavior=row["expected_behavior"],
                metadata=dict(row.get("metadata", {})),
            )
        )
    return dict(contexts_by_example)


def load_candidate_responses(
    processed_root: Path,
    *,
    example_ids: set[str] | None = None,
    response_ids: set[str] | None = None,
) -> dict[str, list[CandidateResponse]]:
    path = processed_root / "candidate_responses.jsonl"
    if not path.exists():
        return {}
    responses_by_example: dict[str, list[CandidateResponse]] = defaultdict(list)
    for row in read_jsonl(path):
        example_id = str(row["example_id"])
        if example_ids is not None and example_id not in example_ids:
            continue
        if response_ids is not None and str(row["response_id"]) not in response_ids:
            continue
        responses_by_example[example_id].append(
            CandidateResponse(
                response_id=row["response_id"],
                example_id=example_id,
                dataset=row["dataset"],
                split=row["split"],
                text=row["text"],
                model=row.get("model"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return dict(responses_by_example)


def load_annotations(
    processed_root: Path,
    *,
    example_ids: set[str] | None = None,
    target_ids: set[str] | None = None,
) -> dict[str, list[Annotation]]:
    path = processed_root / "annotations.jsonl"
    if not path.exists():
        return {}
    annotations_by_example: dict[str, list[Annotation]] = defaultdict(list)
    for row in read_jsonl(path):
        example_id = str(row["example_id"])
        if example_ids is not None and example_id not in example_ids:
            continue
        if target_ids is not None and row.get("target_id") not in target_ids:
            continue
        annotations_by_example[example_id].append(
            Annotation(
                annotation_id=row["annotation_id"],
                example_id=example_id,
                target_type=row["target_type"],
                target_id=row.get("target_id"),
                label=row["label"],
                start_char=row.get("start_char"),
                end_char=row.get("end_char"),
                text=row.get("text"),
                evidence_chunk_ids=list(row.get("evidence_chunk_ids", [])),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return dict(annotations_by_example)


def load_corpus_for_source_ids(
    processed_root: Path,
    source_ids: set[str],
    *,
    include_chunk_ids: set[str] | None = None,
) -> dict[str, CorpusChunk]:
    chunks = {}
    include_chunk_ids = include_chunk_ids or set()
    for row in read_jsonl(processed_root / "corpus.jsonl"):
        metadata = dict(row.get("metadata", {}))
        if metadata.get("source_example_id") not in source_ids and row["chunk_id"] not in include_chunk_ids:
            continue
        chunk = CorpusChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            dataset=row["dataset"],
            title=row.get("title"),
            text=row["text"],
            sentences=list(row.get("sentences", [])),
            metadata=metadata,
        )
        chunks[chunk.chunk_id] = chunk
    return chunks


def load_all_corpus(processed_root: Path) -> dict[str, CorpusChunk]:
    chunks = {}
    for row in read_jsonl(processed_root / "corpus.jsonl"):
        chunk = CorpusChunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            dataset=row["dataset"],
            title=row.get("title"),
            text=row["text"],
            sentences=list(row.get("sentences", [])),
            metadata=dict(row.get("metadata", {})),
        )
        chunks[chunk.chunk_id] = chunk
    return chunks






def gold_chunk_ids(qrels: list[Qrel]) -> set[str]:
    return {qrel.chunk_id for qrel in qrels if qrel.relevance >= 3}
