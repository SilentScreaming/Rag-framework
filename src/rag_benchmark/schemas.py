"""Canonical records shared by datasets, pipelines, and evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


JsonObject = dict[str, Any]
TaskTrack = Literal["short_answer", "long_answer_citation", "verification_hallucination"]
AnnotationTarget = Literal[
    "example",
    "response",
    "claim",
    "citation",
    "evidence",
    "span",
]
SupportStatus = Literal[
    "unjudged",
    "supported",
    "unsupported",
    "contradicted",
    "insufficient",
    "error",
]
VerificationStatus = Literal["not_run", "passed", "revised", "failed", "error"]
StageStatus = Literal["not_run", "completed", "replayed", "skipped", "failed"]


class BenchmarkRecord(BaseModel):
    """Base model for JSONL-compatible benchmark records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_dict(self) -> JsonObject:
        return self.model_dump(mode="json", exclude_none=False)


RecordT = TypeVar("RecordT", bound=BenchmarkRecord)


class AnswerRecord(BenchmarkRecord):
    text: str
    type: str = "short"
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer text must not be empty")
        return value


class EvidenceSpanRecord(BenchmarkRecord):
    doc_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    sentence_id: int | None = Field(default=None, ge=0)
    text: str
    support_type: str = "gold"
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence text must not be empty")
        return value


class SentenceRecord(BenchmarkRecord):
    sentence_id: int = Field(ge=0)
    text: str
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sentence text must not be empty")
        return value


class ExampleRecord(BenchmarkRecord):
    schema_version: str = "example-v1"
    example_id: str
    dataset: str
    split: str
    task_type: str
    question: str
    answers: list[AnswerRecord] = Field(default_factory=list)
    gold_evidence: list[EvidenceSpanRecord] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("example_id", "dataset", "split", "task_type", "question")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text field must not be empty")
        return value


class CorpusChunkRecord(BenchmarkRecord):
    schema_version: str = "corpus-chunk-v1"
    chunk_id: str
    doc_id: str
    dataset: str
    title: str | None = None
    url: str | None = None
    text: str
    sentences: list[SentenceRecord] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("chunk_id", "doc_id", "dataset", "text")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text field must not be empty")
        return value


class QrelRecord(BenchmarkRecord):
    schema_version: str = "qrel-v1"
    example_id: str
    chunk_id: str
    relevance: int = Field(ge=0, le=3)
    support_type: str = ""
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("example_id", "chunk_id")
    @classmethod
    def required_id_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required id field must not be empty")
        return value


class SplitMemberRecord(BenchmarkRecord):
    schema_version: str = "split-member-v1"
    example_id: str
    dataset: str
    source_split: str
    split_profile: str
    metadata: JsonObject = Field(default_factory=dict)


class CandidateResponseRecord(BenchmarkRecord):
    """A frozen model response used to calibrate hallucination judges."""

    schema_version: str = "candidate-response-v1"
    response_id: str
    example_id: str
    dataset: str
    split: str
    text: str
    model: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("response_id", "example_id", "dataset", "split")
    @classmethod
    def response_required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required response field must not be empty")
        return value

    @field_validator("text")
    @classmethod
    def response_text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("response text must not be empty")
        return value


class AnnotationRecord(BenchmarkRecord):
    """Dataset supervision for labels, rationales, citations, and text spans."""

    schema_version: str = "annotation-v1"
    annotation_id: str
    example_id: str
    target_type: AnnotationTarget
    target_id: str | None = None
    label: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("annotation_id", "example_id", "label")
    @classmethod
    def annotation_required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required annotation field must not be empty")
        return value


class ContextSetRecord(BenchmarkRecord):
    """A fixed provided-context condition, such as an RGB noise level."""

    schema_version: str = "context-set-v1"
    context_set_id: str
    example_id: str
    dataset: str
    condition: str
    chunk_ids: list[str]
    expected_behavior: str
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("context_set_id", "example_id", "dataset", "condition", "expected_behavior")
    @classmethod
    def context_required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required context-set field must not be empty")
        return value


class RetrievedChunkRecord(BenchmarkRecord):
    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    retriever: str | None = None
    title: str | None = None
    text: str | None = None
    is_gold: bool = False
    gold_sent_ids: list[int] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class CitationRecord(BenchmarkRecord):
    citation_id: str | None = None
    claim_id: str | None = None
    chunk_id: str
    sentence_id: int | None = Field(default=None, ge=0)
    cited_text: str | None = None
    support_status: SupportStatus = "unjudged"
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chunk_id must not be empty")
        return value


class ClaimCheckRecord(BenchmarkRecord):
    claim_id: str
    claim_text: str
    status: SupportStatus = "unjudged"
    citations: list[CitationRecord] = Field(default_factory=list)
    judge: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("claim_id", "claim_text")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text field must not be empty")
        return value


class ListAnswerItemRecord(BenchmarkRecord):
    item_id: str
    text: str
    source_span: tuple[int, int] | None = None
    citation_ids: list[str] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class AnswerBundleRecord(BenchmarkRecord):
    """Task-specific answer structure stored alongside the plain ``PredictionRecord.answer`` text."""

    short_answer: str | None = None
    long_answer: str | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    list_items: list[ListAnswerItemRecord] | None = None
    verification_label: Literal["SUPPORTS", "REFUTES", "INSUFFICIENT"] | None = None
    rationale_refs: list[CitationRecord] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class StageTraceRecord(BenchmarkRecord):
    schema_version: str = "stage-trace-v1"
    example_id: str
    stage: str
    method: str
    status: StageStatus = "completed"
    latency_ms: float = Field(default=0.0, ge=0)
    input_hash: str | None = None
    output_hash: str | None = None
    artifact_path: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class VerificationQuestionRecord(BenchmarkRecord):
    question_id: str
    question: str
    answer: str | None = None
    status: SupportStatus = "unjudged"
    citations: list[CitationRecord] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: JsonObject = Field(default_factory=dict)


class RevisionDecisionRecord(BenchmarkRecord):
    claim_id: str
    action: Literal["keep", "replace", "remove", "abstain"] = "keep"
    original_text: str
    revised_text: str | None = None
    reason: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: SupportStatus = "unjudged"
    metadata: JsonObject = Field(default_factory=dict)


class VerificationRecord(BenchmarkRecord):
    status: VerificationStatus = "not_run"
    original_answer: str | None = None
    revised_answer: str | None = None
    questions: list[VerificationQuestionRecord] = Field(default_factory=list)
    decisions: list[RevisionDecisionRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)


class LatencyRecord(BenchmarkRecord):
    query_formulation_ms: float | None = Field(default=None, ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    reranking_ms: float | None = Field(default=None, ge=0)
    compression_ms: float | None = Field(default=None, ge=0)
    generation_ms: float = Field(default=0.0, ge=0)
    verification_ms: float | None = Field(default=None, ge=0)
    overhead_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(default=0.0, ge=0)


class TokenUsageRecord(BenchmarkRecord):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class CostRecord(BenchmarkRecord):
    estimated_cost: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    metadata: JsonObject = Field(default_factory=dict)


class PredictionRecord(BenchmarkRecord):
    schema_version: str = "prediction-v1"
    run_id: str | None = None
    example_id: str
    pipeline: str
    question: str
    gold_answers: list[AnswerRecord] = Field(default_factory=list)
    answer: str
    task_track: TaskTrack = "short_answer"
    answer_bundle: AnswerBundleRecord | None = None
    retrieved_chunks: list[RetrievedChunkRecord] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    claim_checks: list[ClaimCheckRecord] = Field(default_factory=list)
    verification: VerificationRecord | None = None
    latency: LatencyRecord = Field(default_factory=LatencyRecord)
    tokens: TokenUsageRecord = Field(default_factory=TokenUsageRecord)
    cost: CostRecord = Field(default_factory=CostRecord)
    generator: JsonObject = Field(default_factory=dict)
    stage_traces: list[StageTraceRecord] = Field(default_factory=list)
    metrics: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("example_id", "pipeline", "question")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required text field must not be empty")
        return value


class RetrievalTraceRecord(BenchmarkRecord):
    schema_version: str = "retrieval-trace-v1"
    example_id: str
    question: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkRecord] = Field(default_factory=list)
    metrics: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)


class RunConfigRecord(BenchmarkRecord):
    schema_version: str = "run-config-v1"
    run_id: str
    timestamp: str
    git_commit: str
    dataset_name: str
    split_name: str
    candidate_scope: str | None = None
    num_examples: int = Field(ge=0)
    candidate_corpus_chunks: int = Field(ge=0)
    random_seed: int
    pipeline_names: list[str]
    retriever: JsonObject = Field(default_factory=dict)
    generator: JsonObject = Field(default_factory=dict)
    metrics: JsonObject = Field(default_factory=dict)
    environment: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)


def read_jsonl_records(path: Path, record_type: type[RecordT]) -> list[RecordT]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(record_type.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid {record_type.__name__}") from exc
    return records


def write_jsonl_records(path: Path, records: Iterable[BenchmarkRecord]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
