"""Execution settings and artifacts, independent of the pipeline runner."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.data import Example, Qrel, ContextSet
from rag_benchmark.compression.sentence_select import CompressedContext
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.query_formulation import QueryFormulationResult
from rag_benchmark.schemas import PredictionRecord, RetrievalTraceRecord, StageTraceRecord
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class FormalRunSettings:
    run_id: str | None = None
    split_profile: str = "mvp"
    limit: int = 20
    seed: int = 42
    gold_relevance_threshold: int = 3
    output_root: Path | None = None
    component_settings: ComponentFactorySettings = field(default_factory=ComponentFactorySettings)
    retrieval_metric_ks: tuple[int, ...] = (5, 10)
    save_outputs: bool = True
    stage_cache_root: Path | None = None
    replay_stages: tuple[str, ...] = ()
    write_stage_cache: bool = False
    bootstrap_samples: int = 1000
    draft_predictions: dict[str, PredictionRecord] | None = None
    context_condition: str | None = None
    context_metadata: JsonObject = field(default_factory=dict)
    # Immutable stage results scoped to this comparison.
    paired_artifact_root: Path | None = None


@dataclass
class PipelineState:
    example: Example
    qrels: list[Qrel]
    query_formulation: QueryFormulationResult | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    compressed_context: CompressedContext | None = None
    prediction: PredictionRecord | None = None
    retrieval_trace: RetrievalTraceRecord | None = None
    stage_latencies: dict[str, float] = field(default_factory=dict)
    logs: list[JsonObject] = field(default_factory=list)
    stage_traces: list[StageTraceRecord] = field(default_factory=list)
    context_set: ContextSet | None = None
    retriever: Any | None = None
    paired_reuse: dict[str, JsonObject] = field(default_factory=dict)
    paired_stage_artifacts: dict[str, JsonObject] = field(default_factory=dict)


@dataclass(frozen=True)
class FormalRunResult:
    run_id: str
    output_dir: Path | None
    predictions: list[PredictionRecord]
    retrieval_traces: list[RetrievalTraceRecord]
    metrics: JsonObject
    logs: list[JsonObject]
    cost_latency: JsonObject
    stage_traces: list[StageTraceRecord] = field(default_factory=list)
    manifest: JsonObject = field(default_factory=dict)

