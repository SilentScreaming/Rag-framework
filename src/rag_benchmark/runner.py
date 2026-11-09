"""Config-driven benchmark runner for formal pipelines."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from rag_benchmark.component_factory import (
    OpenAITextPromptGenerator,
    build_base_generator,
    build_compressor,
    build_evidence_judge,
    build_query_formulator,
    build_reranker,
    build_retriever,
    pipeline_retrieval_top_k,
)
from rag_benchmark.config import ExperimentConfig
from rag_benchmark.evaluation.citation import evaluate_citation_quality
from rag_benchmark.evaluation.evidence import evaluate_evidence_consistency
from rag_benchmark.evaluation.tracks import (
    TRACK_PRIMARY_METRICS,
    aggregate_track_metrics,
    bootstrap_confidence_intervals,
    evaluate_prediction_track_metrics,
    group_track_metrics,
)
from rag_benchmark.generation.citation import (
    apply_citation_generation,
)
from rag_benchmark.data import (
    ContextSet,
    DatasetSlice,
    Example,
    Qrel,
    gold_chunk_ids,
    load_candidate_responses,
    load_all_corpus,
    load_annotations,
    load_corpus_for_source_ids,
    load_context_sets,
    load_examples,
    load_qrels,
)
from rag_benchmark.generation.backends import GenerationResult, estimate_cost
from rag_benchmark.generation.backends import load_openai_local_config, resolve_openai_model
from rag_benchmark.evaluation.basic import aggregate_metrics, best_qa_scores, retrieval_scores
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.serialization import build_retrieval_trace_record, serialize_retrieved_chunks
from rag_benchmark.query_formulation import QueryFormulationResult
from rag_benchmark.schemas import (
    CostRecord, LatencyRecord, PredictionRecord, RetrievalTraceRecord, TokenUsageRecord,
    StageTraceRecord,
)
from rag_benchmark.stage_cache import StageArtifactCache, canonical_hash
from rag_benchmark.indexing.dense_cache import corpus_fingerprint
from rag_benchmark.verification.cove import (
    CoVeVerificationConfig,
    CoVeVerifier,
    FactoredCoVeConfig,
    FactoredCoVeVerifier,
)
from rag_benchmark.retrieval.hybrid import fuse_multi_query_results
from rag_benchmark.retrieval.candidate_cache import FrozenCandidateCache
from rag_benchmark.generation.context import prepare_context
from rag_benchmark.generation.service import generate_citation_answer
from rag_benchmark.generation.answer_format import build_answer_bundle
from rag_benchmark.evaluation.coverage import attach_evaluation_coverage
from rag_benchmark.run_identity import run_identity, source_fingerprint
from rag_benchmark.paired_artifacts import PairedArtifacts
# Re-exported for scripts and tests that import these names from the runner module.
from rag_benchmark.execution_types import FormalRunSettings, FormalRunResult, PipelineState
from rag_benchmark.planning import (
    resolve_pipeline_with_base, build_experiment_with_resolved_pipeline, adapt_pipeline_to_dataset,
)
from rag_benchmark.artifacts import (
    load_completed_formal_run, write_formal_outputs, summarize_cost_latency, make_run_id,
)


JsonObject = dict[str, Any]


class ConfigDrivenPipelineRunner:
    """Executes one pipeline config over one dataset slice."""

    def __init__(
        self,
        *,
        repo_root: Path,
        experiment: ExperimentConfig,
        dataset_slice: DatasetSlice,
        settings: FormalRunSettings,
    ) -> None:
        initialization_started = time.perf_counter()
        self.repo_root = repo_root.resolve()
        self.experiment = experiment
        self.pipeline = experiment.pipeline
        self.task_track = experiment.dataset.task_track
        self.dataset_slice = dataset_slice
        self.settings = settings
        self.identity = run_identity(repo_root=self.repo_root, experiment=experiment,
                                     dataset_slice=dataset_slice, settings=settings)
        self.run_id = settings.run_id or make_run_id(
            dataset=dataset_slice.dataset_name,
            split=settings.split_profile,
            pipeline=self.pipeline.name,
        )
        configured_profile = self.pipeline.budget.profile
        requested_profile = settings.component_settings.execution_profile
        execution_profile = configured_profile if requested_profile == "config" else requested_profile
        openai_local = (
            load_openai_local_config(self.repo_root)
            if settings.component_settings.generator_kind == "openai"
            else {}
        )
        self.component_settings = replace(
            settings.component_settings,
            generator_model=(
                resolve_openai_model(self.repo_root, settings.component_settings.generator_model,
                                     config=openai_local)
                if settings.component_settings.generator_kind == "openai"
                else settings.component_settings.generator_model
            ),
            input_cost_per_1m_tokens=(
                settings.component_settings.input_cost_per_1m_tokens
                if settings.component_settings.input_cost_per_1m_tokens is not None
                else openai_local.get("input_cost_per_1m_tokens")
            ),
            output_cost_per_1m_tokens=(
                settings.component_settings.output_cost_per_1m_tokens
                if settings.component_settings.output_cost_per_1m_tokens is not None
                else openai_local.get("output_cost_per_1m_tokens")
            ),
            execution_profile=execution_profile,
            strict_components=(
                settings.component_settings.strict_components
                or (self.pipeline.budget.strict_components and execution_profile != "smoke")
            ),
            model_configs=dict(experiment.model_refs),
            model_cache_dir=(
                settings.component_settings.model_cache_dir
                or (Path(experiment.paths.models_root) / "hf-cache" if experiment.paths else None)
            ),
        )
        self.stage_cache = StageArtifactCache(settings.stage_cache_root) if settings.stage_cache_root else None
        self.corpus_by_id = {chunk.chunk_id: chunk for chunk in dataset_slice.corpus}
        self.candidate_cache = (
            FrozenCandidateCache(dataset_slice.candidate_cache_path)
            if dataset_slice.candidate_cache_path is not None
            else None
        )
        self.corpus_hash = (
            self.candidate_cache.scope_hash
            if self.candidate_cache is not None
            else corpus_fingerprint(dataset_slice.corpus)
        )
        self.paired_artifacts = (PairedArtifacts(settings.paired_artifact_root, {
            "implementation": source_fingerprint(self.repo_root),
            "dataset": dataset_slice.dataset_name, "data_version": experiment.dataset.source.version,
            "corpus_hash": self.corpus_hash, "candidate_scope": dataset_slice.candidate_scope,
            "components": json.loads(json.dumps(asdict(self.component_settings), default=str)),
            "context_condition": settings.context_condition, "context_metadata": settings.context_metadata,
        }) if settings.paired_artifact_root else None)
        self.query_formulator = build_query_formulator(
            stage=self.pipeline.stages.query_formulation,
            repo_root=self.repo_root,
            settings=self.component_settings,
            stage_config=self.pipeline.query_formulation,
        )
        self._uses_scoped_contexts = any(dataset_slice.context_sets_by_example.values())
        self._scoped_retrievers: dict[str, Any] = {}
        self.retriever = (
            None
            if (
                self.candidate_cache is not None
                or (self._uses_scoped_contexts and self.pipeline.stages.retrieval not in {"none", "oracle"})
            )
            else build_retriever(
                pipeline=self.pipeline,
                corpus=dataset_slice.corpus,
                settings=self.component_settings,
            )
        )
        self.reranker = build_reranker(
            pipeline=self.pipeline,
            settings=self.component_settings,
        )
        compression_scorer = getattr(self.reranker, "scorer", None)
        if self.pipeline.logging.get("removed_component") == "reranking" and self.pipeline.compression.get("semantic_scoring"):
            # Removing ranking must not also remove the compressor's semantic scorer.
            scoring_pipeline = self.pipeline.model_copy(update={"stages": self.pipeline.stages.model_copy(update={"reranking": "cross_encoder"})})
            compression_scorer = getattr(build_reranker(pipeline=scoring_pipeline, settings=self.component_settings), "scorer", None)
        self.compressor = build_compressor(
            self.pipeline,
            semantic_scorer=compression_scorer,
        )
        self.evidence_judge = build_evidence_judge(
            pipeline=self.pipeline,
            settings=self.component_settings,
        )
        self.base_generator = build_base_generator(
            repo_root=self.repo_root,
            settings=self.component_settings,
            pipeline=self.pipeline,
        )
        self.text_generator = None
        from rag_benchmark.generation.budget import output_token_limit
        if (self.pipeline.stages.generation == "citation_aware" or self.pipeline.stages.verification == "factored_cove") and self.component_settings.generator_kind == "openai":
            self.text_generator = OpenAITextPromptGenerator(
                repo_root=self.repo_root,
                model=self.component_settings.generator_model,
                max_output_tokens=output_token_limit(self.pipeline, "citation_json" if self.pipeline.stages.generation == "citation_aware" else "plain"),
            )
        self.initialization_ms = (time.perf_counter() - initialization_started) * 1000

    def _warmup_local(self):
        """Warm local inference only; this method never calls an LLM backend."""
        if self.component_settings.execution_profile != "quality_max" or not self.dataset_slice.examples:
            return {"status": "not_applicable", "elapsed_ms": 0.0}
        started = time.perf_counter()
        example = self.dataset_slice.examples[0]
        _, retriever = self._retriever_for_example(example)
        retrieved = (retrieve_query_sets(retriever, [example.question], top_k=2)[0][1]
                     if retriever is not None and self.pipeline.stages.retrieval not in {"none", "oracle"} else [])
        if self.reranker is not None and retrieved:
            retrieved = self.reranker.rerank(example.question, retrieved)
        if self.compressor is not None and retrieved:
            self.compressor.compress(query=example.question, retrieved=retrieved, queries=(example.question,))
        from rag_benchmark.evaluation.evidence import TransformersNLIEvidenceJudge
        if isinstance(self.evidence_judge, TransformersNLIEvidenceJudge):
            self.evidence_judge.judge(claim_text="This is a warmup sentence.",
                                      evidence_texts=["This is a warmup sentence."])
        return {"status": "completed", "elapsed_ms": (time.perf_counter() - started) * 1000,
                "scope": "local_inference; API/server warmup excluded", "llm_requests": 0}

    def run(self) -> FormalRunResult:
        if self.settings.save_outputs and self.output_dir().exists():
            raise ValueError("Run output already exists. Validate its identity through the suite runner, or use a new run ID; existing artifacts are never overwritten.")
        warmup = self._warmup_local()
        predictions: list[PredictionRecord] = []
        traces: list[RetrievalTraceRecord] = []
        logs: list[JsonObject] = [
            {
                "event": "run_started",
                "run_id": self.run_id,
                "dataset": self.dataset_slice.dataset_name,
                "split": self.dataset_slice.split_name,
                "pipeline": self.pipeline.name,
                "timing_conditions": {"pipeline_jobs": 1, "initialization_ms": self.initialization_ms,
                                      "warmup": warmup, "per_question_excludes_initialization_and_warmup": True},
            }
        ]
        metric_rows: list[JsonObject] = []
        stage_traces: list[StageTraceRecord] = []
        checkpoint_rows = self._load_checkpoint()
        if checkpoint_rows:
            logs.append(
                {
                    "event": "checkpoint_recovered",
                    "run_id": self.run_id,
                    "examples": len(checkpoint_rows),
                }
            )

        for index, example in enumerate(self.dataset_slice.examples, start=1):
            checkpoint = checkpoint_rows.get(example.example_id)
            if checkpoint is not None:
                prediction = checkpoint["prediction"]
                predictions.append(prediction)
                metric_rows.append(prediction.metrics)
                if checkpoint["retrieval_trace"] is not None:
                    traces.append(checkpoint["retrieval_trace"])
                stage_traces.extend(checkpoint["stage_traces"])
                logs.extend(checkpoint["logs"])
                logs.append(
                    {
                        "event": "example_recovered",
                        "index": index,
                        "example_id": example.example_id,
                    }
                )
                continue
            try:
                state = self.run_example(example=example, qrels=self.dataset_slice.qrels_by_example.get(example.example_id, []))
            except Exception as exc:
                if self.settings.save_outputs:
                    from rag_benchmark.generation.usage import failure_usage
                    failure_dir = self.output_dir().parent / ".failures"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    from uuid import uuid4
                    record = {"run_id": self.run_id, "example_id": example.example_id,
                              "error_type": type(exc).__name__, "call_usage": failure_usage(exc)}
                    (failure_dir / f"{uuid4().hex}.json").write_text(json.dumps(record), encoding="utf-8")
                raise
            logs.extend(state.logs)
            if state.prediction is not None:
                predictions.append(state.prediction)
                metric_rows.append(state.prediction.metrics)
            if state.retrieval_trace is not None:
                traces.append(state.retrieval_trace)
            stage_traces.extend(state.stage_traces)
            logs.append({"event": "example_completed", "index": index, "example_id": example.example_id})
            self._append_checkpoint(state)

        aggregate = aggregate_metrics(metric_rows)
        track = self.task_track
        aggregate, grouped_results = aggregate_track_metrics(
            track,
            predictions,
            aggregate,
            examples_by_id={example.example_id: example for example in self.dataset_slice.examples},
        )
        # Missing or failed outcomes cannot become an optimistic complete-case mean.
        from rag_benchmark.evaluation.statistics import RECOMPUTED
        for key in list(aggregate):
            if key not in RECOMPUTED and key in {k for row in metric_rows for k in row}:
                if any(not isinstance(row.get(key), (int, float)) for row in metric_rows):
                    aggregate.pop(key)
        cost_latency = summarize_cost_latency(predictions)
        if cost_latency.get("cost_status") != "complete_estimate":
            for key in ("estimated_cost", "total_estimated_cost", "avg_estimated_cost"):
                aggregate.pop(key, None)
        aggregate.update(
            {
                key: value
                for key, value in cost_latency.items()
                if key not in {"num_predictions"} and isinstance(value, (int, float))
                and not (
                    key in {"total_estimated_cost", "avg_estimated_cost"}
                    and float(cost_latency.get("estimated_cost_available_rate", 0.0) or 0.0) == 0.0
                )
            }
        )
        from rag_benchmark.evaluation.contracts import aggregate_metric_results
        metrics = {
            "run_id": self.run_id,
            "evaluation_protocol": "evaluation-v1",
            "semantic_evaluation_scope": "operational_generator_declared_only",
            "dataset": self.dataset_slice.dataset_name,
            "split": self.dataset_slice.split_name,
            "candidate_scope": self.dataset_slice.candidate_scope,
            "num_examples": len(self.dataset_slice.examples),
            "generation_failures": sum(bool(p.generator.get("schema_failure")) for p in predictions),
            "cost_status": cost_latency.get("cost_status", "not_applicable"),
            "corpus_chunks": len(self.dataset_slice.corpus),
            "task_track": track,
            "pipelines": {self.pipeline.name: aggregate},
            "metric_results": {self.pipeline.name: aggregate_metric_results(
                [{"metrics": p.metadata.get("task_metric_results", {})} for p in predictions])},
            "track_results": {
                self.pipeline.name: group_track_metrics(track, aggregate),
            },
            "grouped_results": {self.pipeline.name: grouped_results},
            "confidence_intervals": {
                self.pipeline.name: bootstrap_confidence_intervals(
                    metric_rows,
                    predictions=predictions,
                    examples_by_id={e.example_id: e for e in self.dataset_slice.examples},
                    keys=TRACK_PRIMARY_METRICS.get(track, ()),
                    samples=self.settings.bootstrap_samples,
                    seed=self.settings.seed,
                )
            },
        }
        manifest = self._run_manifest()
        output_dir = None
        if self.settings.save_outputs:
            output_dir = self.output_dir()
            write_formal_outputs(
                output_dir=output_dir,
                repo_root=self.repo_root,
                experiment=self.experiment,
                settings=self.settings,
                run_id=self.run_id,
                predictions=predictions,
                retrieval_traces=traces,
                metrics=metrics,
                logs=logs,
                cost_latency=cost_latency,
                stage_traces=stage_traces,
                manifest=manifest,
            )
            self._clear_checkpoint()
        result = FormalRunResult(
            run_id=self.run_id,
            output_dir=output_dir,
            predictions=predictions,
            retrieval_traces=traces,
            metrics=metrics,
            logs=logs,
            cost_latency=cost_latency,
            stage_traces=stage_traces,
            manifest=manifest,
        )
        if self.candidate_cache is not None:
            self.candidate_cache.close()
        return result

    def _checkpoint_dir(self) -> Path | None:
        if not self.settings.save_outputs:
            return None
        output_dir = self.output_dir()
        return output_dir.parent / ".checkpoints" / self.run_id

    def _checkpoint_identity(self) -> JsonObject:
        example_ids = sorted(example.example_id for example in self.dataset_slice.examples)
        return {
            "schema_version": "formal-run-checkpoint-v1",
            "run_identity": self.identity,
            "run_id": self.run_id,
            "pipeline": self.pipeline.name,
            "pipeline_hash": canonical_hash(self.pipeline.to_json_dict()),
            "dataset": self.dataset_slice.dataset_name,
            "split": self.dataset_slice.split_name,
            "candidate_scope": self.dataset_slice.candidate_scope,
            "corpus_hash": self.corpus_hash,
            "example_ids_hash": canonical_hash(example_ids),
            "example_count": len(example_ids),
            "generator_kind": self.component_settings.generator_kind,
            "generator_model": self.component_settings.generator_model,
            "execution_profile": self.component_settings.execution_profile,
        }

    def _load_checkpoint(self) -> dict[str, JsonObject]:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return {}
        metadata_path = checkpoint_dir / "checkpoint.meta.json"
        rows_path = checkpoint_dir / "examples.jsonl"
        if not metadata_path.is_file() or not rows_path.is_file():
            raise ValueError(f"Incomplete formal-run checkpoint: {checkpoint_dir}")
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = self._checkpoint_identity()
        if observed != expected:
            raise ValueError(
                f"Formal-run checkpoint identity mismatch: {checkpoint_dir}. "
                "Keep it for audit or remove it before starting a different configuration."
            )
        raw_lines = rows_path.read_text(encoding="utf-8").splitlines()
        recovered: dict[str, JsonObject] = {}
        expected_ids = {example.example_id for example in self.dataset_slice.examples}
        for index, line in enumerate(raw_lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(raw_lines) - 1:
                    break
                raise
            prediction = PredictionRecord.model_validate(row["prediction"])
            from rag_benchmark.paired_artifacts import validate_prefix_anchors
            validate_prefix_anchors(prediction)
            if (
                prediction.example_id not in expected_ids
                or prediction.run_id != self.run_id
                or prediction.pipeline != self.pipeline.name
            ):
                raise ValueError(f"Invalid example in formal-run checkpoint: {prediction.example_id}")
            if prediction.example_id in recovered:
                raise ValueError(f"Duplicate example in formal-run checkpoint: {prediction.example_id}")
            retrieval_trace = (
                RetrievalTraceRecord.model_validate(row["retrieval_trace"])
                if row.get("retrieval_trace") is not None
                else None
            )
            recovered[prediction.example_id] = {
                "prediction": prediction,
                "retrieval_trace": retrieval_trace,
                "stage_traces": [
                    StageTraceRecord.model_validate(value)
                    for value in row.get("stage_traces", [])
                ],
                "logs": list(row.get("logs", [])),
            }
        return recovered

    def _append_checkpoint(self, state: PipelineState) -> None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None or state.prediction is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_dir / "checkpoint.meta.json"
        identity = self._checkpoint_identity()
        if metadata_path.exists():
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            if observed != identity:
                raise ValueError(f"Formal-run checkpoint identity mismatch: {checkpoint_dir}")
        else:
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(identity, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(metadata_path)
        row = {
            "prediction": state.prediction.to_json_dict(),
            "retrieval_trace": (
                state.retrieval_trace.to_json_dict()
                if state.retrieval_trace is not None
                else None
            ),
            "stage_traces": [trace.to_json_dict() for trace in state.stage_traces],
            "logs": state.logs,
        }
        with (checkpoint_dir / "examples.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _clear_checkpoint(self) -> None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is not None and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            parent = checkpoint_dir.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()

    def run_example(self, *, example: Example, qrels: list[Qrel]) -> PipelineState:
        from rag_benchmark.generation.budget import execution_budget, BudgetExhausted, output_token_limit
        with execution_budget(max_calls=self.pipeline.budget.max_model_calls,
                              max_total_tokens=self.pipeline.budget.max_total_model_tokens) as budget:
            try:
                state = self._run_example_unbudgeted(example=example, qrels=qrels)
                if budget.exhausted:
                    raise BudgetExhausted("a stage caught budget exhaustion; refusing a successful fallback answer")
                state.prediction = state.prediction.model_copy(update={"metadata": {
                    **state.prediction.metadata, "execution_budget": budget.snapshot(),
                    "effective_output_budget": {mode: output_token_limit(self.pipeline, mode)
                                                for mode in ("plain", "citation_json")}}})
                return state
            except Exception as exc:
                from rag_benchmark.generation.usage import failure_usage
                exc._rag_benchmark_usage = {**failure_usage(exc), "budget_ledger": budget.snapshot()}
                raise

    def _run_example_unbudgeted(self, *, example: Example, qrels: list[Qrel]) -> PipelineState:
        from rag_benchmark.generation.budget import set_budget_stage
        set_budget_stage("query_formulation")
        if self.settings.draft_predictions and example.example_id in self.settings.draft_predictions:
            return self._run_example_from_draft(
                example=example,
                qrels=qrels,
                draft=self.settings.draft_predictions[example.example_id],
            )
        example_started = time.perf_counter()
        state = PipelineState(example=example, qrels=qrels)
        query_inputs = {"question": example.question}
        query_config = {"method": self.pipeline.stages.query_formulation,
                        "options": self.pipeline.query_formulation}
        query_artifact = self._paired_load(state, "query_formulation", query_inputs, query_config)
        if query_artifact is not None:
            state.query_formulation = self._query_from_metadata(query_artifact.payload)
            from rag_benchmark.generation.budget import record_reused_calls
            record_reused_calls({"calls": query_artifact.payload.get("_budget_calls", [])},
                                source_run_id=query_artifact.metadata["source_run_id"])
        else:
            state.query_formulation = self.query_formulator.formulate(example.question)
            self._paired_save(state, "query_formulation", query_inputs, query_config,
                              state.query_formulation.to_metadata(), state.query_formulation.latency_ms)
        state.stage_latencies["query_formulation_ms"] = state.query_formulation.latency_ms
        self._record_stage(
            state,
            stage="query_formulation",
            method=state.query_formulation.method,
            input_value={"question": example.question},
            output_value=state.query_formulation.to_metadata(),
            latency_ms=state.query_formulation.latency_ms,
            status="replayed" if query_artifact is not None else "completed",
            artifact_path=str(query_artifact.path) if query_artifact is not None else None,
        )

        if self.pipeline.stages.retrieval == "oracle":
            contexts = self.dataset_slice.context_sets_by_example.get(example.example_id, [])
            if contexts:
                state.context_set = select_context_set(contexts, condition=self.settings.context_condition,
                                                       metadata=self.settings.context_metadata)
            retrieval_start = time.perf_counter()
            state.retrieved = self._oracle_retrieval(state.qrels)
            state.stage_latencies["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
            self._record_stage(
                state,
                stage="retrieval",
                method="oracle_gold_context",
                input_value={"qrels": [qrel.chunk_id for qrel in state.qrels]},
                output_value=self._serialize_retrieval(state.retrieved),
                latency_ms=state.stage_latencies["retrieval_ms"],
            )
            state.retrieval_trace = self._build_trace(state)
        else:
            state.context_set, state.retriever = self._retriever_for_example(example)
        if self.pipeline.stages.retrieval not in {"none", "oracle"} and state.retriever is not None:
            retrieval_top_k = pipeline_retrieval_top_k(self.pipeline)
            max_total_queries = self.pipeline.budget.max_queries
            retrieval_queries = list(
                (state.query_formulation.queries or (state.query_formulation.query,))[:max_total_queries]
            )
            retrieval_input = {
                "implementation_identity": self.identity["generation"],
                "queries": retrieval_queries,
                "top_k": retrieval_top_k,
                "candidate_scope": self.dataset_slice.candidate_scope,
                "context_set_id": state.context_set.context_set_id if state.context_set else None,
                "method": self.pipeline.stages.retrieval,
                "pipeline_hash": canonical_hash(self.pipeline.to_json_dict()),
                "corpus_hash": corpus_fingerprint(
                    [self.corpus_by_id[chunk_id] for chunk_id in state.context_set.chunk_ids]
                ) if state.context_set else self.corpus_hash,
            }
            artifact = None
            paired_retrieval_inputs = {k: v for k, v in retrieval_input.items()
                                       if k not in {"implementation_identity", "pipeline_hash"}}
            paired_retrieval_inputs["question"] = example.question
            retrieval_config = {"options": self.pipeline.retrieval, "max_queries": max_total_queries}
            paired_retrieval = self._paired_load(state, "retrieval", paired_retrieval_inputs, retrieval_config)
            if self.stage_cache is not None and "retrieval" in self.settings.replay_stages:
                artifact = self.stage_cache.load(
                    example_id=example.example_id,
                    stage="retrieval",
                    input_value=retrieval_input,
                )
            retrieval_start = time.perf_counter()
            if paired_retrieval is not None:
                state.retrieved = self._deserialize_retrieval(paired_retrieval.payload)
                state.query_formulation = self._query_from_metadata(paired_retrieval.payload["query_formulation"])
                stage_status = "replayed"
            elif artifact is not None:
                state.retrieved = self._deserialize_retrieval(artifact.payload)
                stage_status = "replayed"
            else:
                result_sets = retrieve_query_sets(
                    state.retriever,
                    retrieval_queries,
                    top_k=retrieval_top_k,
                )
                if int(self.pipeline.retrieval.get("iterative_hops", 1)) > 1:
                    seed_results = fuse_multi_query_results(
                        result_sets,
                        final_top_k=min(
                            retrieval_top_k,
                            int(self.pipeline.retrieval.get("hop1_seed_k", 3)),
                        ),
                        rrf_k=int(self.pipeline.retrieval.get("multi_query_rrf_k", 60)),
                    )
                    remaining_query_budget = max(0, max_total_queries - len(retrieval_queries))
                    hop_queries = derive_second_hop_queries(
                        question=state.example.question,
                        seeds=seed_results,
                        limit=min(
                            int(self.pipeline.retrieval.get("max_hop_queries", 3)),
                            remaining_query_budget,
                        ),
                    )
                    result_sets.extend(
                        retrieve_query_sets(
                            state.retriever,
                            hop_queries,
                            top_k=retrieval_top_k,
                        )
                    )
                    query_metadata = dict(state.query_formulation.metadata)
                    query_metadata["iterative_hop_queries"] = hop_queries
                    state.query_formulation = replace(
                        state.query_formulation,
                        queries=tuple([*retrieval_queries, *hop_queries]),
                        metadata=query_metadata,
                    )
                state.retrieved = (
                    fuse_multi_query_results(
                        result_sets,
                        final_top_k=retrieval_top_k,
                        rrf_k=int(self.pipeline.retrieval.get("multi_query_rrf_k", 60)),
                    )
                    if len(result_sets) > 1
                    else result_sets[0][1]
                )
                stage_status = "completed"
            state.stage_latencies["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
            if paired_retrieval is not None:
                state.stage_latencies["retrieval_ms"] = float(paired_retrieval.metadata["logical_latency_ms"])
            retrieval_payload = self._serialize_retrieval(state.retrieved)
            if paired_retrieval is None:
                self._paired_save(state, "retrieval", paired_retrieval_inputs, retrieval_config,
                                  {**retrieval_payload, "query_formulation": state.query_formulation.to_metadata()},
                                  state.stage_latencies["retrieval_ms"])
            saved_artifact = None
            if artifact is None and self.stage_cache is not None and self.settings.write_stage_cache:
                saved_artifact = self.stage_cache.save(
                    example_id=example.example_id,
                    stage="retrieval",
                    method=self.pipeline.stages.retrieval,
                    input_value=retrieval_input,
                    payload=retrieval_payload,
                )
            self._record_stage(
                state,
                stage="retrieval",
                method=self.pipeline.stages.retrieval,
                input_value=retrieval_input,
                output_value=retrieval_payload,
                latency_ms=state.stage_latencies["retrieval_ms"],
                status=stage_status,
                artifact_path=str((paired_retrieval or artifact or saved_artifact).path) if (paired_retrieval or artifact or saved_artifact) else None,
            )
            state.retrieval_trace = self._build_trace(state)

        if self.reranker is not None and state.retrieved:
            rerank_start = time.perf_counter()
            rerank_inputs = {"question": example.question, "candidates": self._serialize_retrieval(state.retrieved)}
            rerank_config = {"method": self.pipeline.stages.reranking, "options": self.pipeline.reranking}
            rerank_artifact = self._paired_load(state, "reranking", rerank_inputs, rerank_config)
            state.retrieved = (self._deserialize_retrieval(rerank_artifact.payload) if rerank_artifact is not None
                               else self.reranker.rerank(state.example.question, state.retrieved))
            state.stage_latencies["reranking_ms"] = (time.perf_counter() - rerank_start) * 1000
            if rerank_artifact is not None:
                state.stage_latencies["reranking_ms"] = float(rerank_artifact.metadata["logical_latency_ms"])
            else:
                self._paired_save(state, "reranking", rerank_inputs, rerank_config,
                                  self._serialize_retrieval(state.retrieved), state.stage_latencies["reranking_ms"])
            self._record_stage(
                state,
                stage="reranking",
                method=self.pipeline.stages.reranking,
                input_value={"query": state.example.question},
                output_value=self._serialize_retrieval(state.retrieved),
                latency_ms=state.stage_latencies["reranking_ms"],
                status="replayed" if rerank_artifact is not None else "completed",
                artifact_path=str(rerank_artifact.path) if rerank_artifact is not None else None,
            )

        if self.pipeline.logging.get("ablation_final_top_k"):
            state.retrieved = state.retrieved[:int(self.pipeline.logging["ablation_final_top_k"])]
        if self.compressor is not None and state.retrieved:
            compression_start = time.perf_counter()
            compression_inputs = {"query": state.query_formulation.query, "queries": list(state.query_formulation.queries),
                                  "ranked_candidates": self._serialize_retrieval(state.retrieved)}
            compression_config = {"method": self.pipeline.stages.compression, "options": self.pipeline.compression}
            compression_artifact = self._paired_load(state, "compression", compression_inputs, compression_config)
            if compression_artifact is not None:
                from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
                payload = compression_artifact.payload
                state.compressed_context = CompressedContext(query=payload["query"],
                    sentences=[CompressedSentence(**s) for s in payload["sentences"]], metadata=payload["metadata"])
            else:
                state.compressed_context = self.compressor.compress(
                    query=state.query_formulation.query, retrieved=state.retrieved,
                    queries=state.query_formulation.queries)
            state.stage_latencies["compression_ms"] = (time.perf_counter() - compression_start) * 1000
            if compression_artifact is not None:
                state.stage_latencies["compression_ms"] = float(compression_artifact.metadata["logical_latency_ms"])
            else:
                self._paired_save(state, "compression", compression_inputs, compression_config,
                                  asdict(state.compressed_context), state.stage_latencies["compression_ms"])
            self._record_stage(
                state,
                stage="compression",
                method=self.pipeline.stages.compression,
                input_value={"query": state.query_formulation.query},
                output_value={"provenance": state.compressed_context.provenance()},
                latency_ms=state.stage_latencies["compression_ms"],
                status="replayed" if compression_artifact is not None else "completed",
                artifact_path=str(compression_artifact.path) if compression_artifact is not None else None,
            )

        generation_start = time.perf_counter()
        state.prediction = self._run_generation(state)
        state.stage_latencies["generation_stage_ms"] = (time.perf_counter() - generation_start) * 1000
        self._record_stage(
            state,
            stage="generation",
            method=self.pipeline.stages.generation,
            input_value={"question": example.question},
            output_value={"answer": state.prediction.answer},
            latency_ms=state.stage_latencies["generation_stage_ms"],
        )
        state.prediction = self._evaluate_and_verify(state)
        state.prediction = self._attach_stage_metadata(state)
        end_to_end_ms = (time.perf_counter() - example_started) * 1000
        observed_ms = end_to_end_ms
        reused_detection_ms = (state.prediction.verification.metadata.get("reused_detection_ms", 0.0)
                               if state.prediction.verification else 0.0)
        if state.paired_reuse or reused_detection_ms:
            end_to_end_ms += sum(float(row["logical_latency_ms"]) for row in state.paired_reuse.values()) + reused_detection_ms
            state.prediction = state.prediction.model_copy(update={"metadata": {
                **state.prediction.metadata, "observed_execution_ms": observed_ms,
                "latency_measurement": "shared_stage_observations_plus_current_suffix_estimate"}})
        component_ms = component_total_ms(
            stage_latencies=state.stage_latencies,
            generation_ms=state.prediction.latency.generation_ms,
        )
        metrics = {
            **state.prediction.metrics,
            "component_total_ms": component_ms,
            "pipeline_overhead_ms": max(0.0, end_to_end_ms - component_ms),
            "total_ms": end_to_end_ms,
        }
        latency = state.prediction.latency.model_copy(
            update={
                "overhead_ms": max(0.0, end_to_end_ms - component_ms),
                "total_ms": end_to_end_ms,
            }
        )
        state.prediction = state.prediction.model_copy(update={"metrics": metrics, "latency": latency})
        state.prediction = evaluate_prediction_track_metrics(
            state.prediction,
            example=state.example,
            qrels=state.qrels,
            annotations=self.dataset_slice.annotations_by_example.get(state.example.example_id, []),
            context_condition=state.context_set.condition if state.context_set else self.settings.context_condition,
            context_set=state.context_set,
        )
        return state

    def _run_example_from_draft(
        self,
        *,
        example: Example,
        qrels: list[Qrel],
        draft: PredictionRecord,
    ) -> PipelineState:
        started = time.perf_counter()
        if self.pipeline.stages.verification == "none":
            raise ValueError("draft_predictions may only be used by a verification pipeline")
        query_metadata = draft.metadata.get("query_formulation", {})
        state = PipelineState(example=example, qrels=qrels)
        state.context_set, state.retriever = self._retriever_for_example(example)
        from rag_benchmark.evaluation.identity import validate_example_identity
        validate_example_identity(draft, example, data_version=self.experiment.dataset.source.version,
            seed=self.settings.seed, context_set_id=state.context_set.context_set_id if state.context_set else None)
        from rag_benchmark.generation.budget import record_reused_calls
        record_reused_calls({"calls": draft.metadata.get("execution_budget", {}).get("calls", [])},
                            source_run_id=draft.run_id)
        state.query_formulation = QueryFormulationResult(
            original_query=example.question,
            query=str(query_metadata.get("query", example.question)),
            queries=tuple(query_metadata.get("queries", [])),
            subquestions=tuple(query_metadata.get("subquestions", [])),
            answer_type=query_metadata.get("answer_type"),
            method="reused_best_practice_draft",
            metadata={"draft_run_id": draft.run_id, "draft_pipeline": draft.pipeline},
        )
        state.retrieved = self._retrieved_from_prediction(draft)
        state.stage_traces = [
            trace.model_copy(update={"status": "replayed"})
            for trace in draft.stage_traces
            if trace.stage != "verification"
        ]
        state.stage_traces.append(
            StageTraceRecord(
                example_id=example.example_id,
                stage="draft_reuse",
                method="paired_best_practice_draft",
                status="replayed",
                input_hash=canonical_hash({"draft_run_id": draft.run_id, "example_id": example.example_id}),
                output_hash=canonical_hash({"answer": draft.answer, "claims": len(draft.claim_checks)}),
                metadata={"draft_run_id": draft.run_id, "draft_pipeline": draft.pipeline},
            )
        )
        state.prediction = draft.model_copy(
            deep=True,
            update={
                "run_id": self.run_id,
                "pipeline": self.pipeline.name,
                "verification": None,
                "stage_traces": list(state.stage_traces),
                "metadata": {
                    **draft.metadata,
                    "draft_execution_budget": draft.metadata.get("execution_budget"),
                    "draft_reuse": {
                        "enabled": True,
                        "draft_run_id": draft.run_id,
                        "draft_pipeline": draft.pipeline,
                    },
                },
            }
        )
        state.prediction = self._evaluate_and_verify(state)
        state.prediction = self._attach_stage_metadata(state)
        verification_elapsed_ms = (time.perf_counter() - started) * 1000
        reused_detection_ms = (state.prediction.verification.metadata.get("reused_detection_ms", 0.0)
                               if state.prediction.verification else 0.0)
        total_ms = draft.latency.total_ms + verification_elapsed_ms + reused_detection_ms
        state.prediction = state.prediction.model_copy(update={"metadata": {**state.prediction.metadata,
            "observed_incremental_execution_ms": verification_elapsed_ms,
            "latency_measurement": "original_draft_plus_shared_detection_plus_observed_suffix_estimate"}})
        component_ms = float(draft.metrics.get("component_total_ms", draft.latency.total_ms)) + float(
            state.stage_latencies.get("verification_ms", 0.0)
        )
        metrics = {
            **state.prediction.metrics,
            "draft_reused": 1.0,
            "query_formulation_ms": draft.metrics.get("query_formulation_ms", 0.0),
            "retrieval_ms": draft.metrics.get("retrieval_ms", 0.0),
            "reranking_ms": draft.metrics.get("reranking_ms", 0.0),
            "compression_ms": draft.metrics.get("compression_ms", 0.0),
            "component_total_ms": component_ms,
            "pipeline_overhead_ms": max(0.0, total_ms - component_ms),
            "total_ms": total_ms,
        }
        latency = state.prediction.latency.model_copy(
            update={
                "overhead_ms": max(0.0, total_ms - component_ms),
                "query_formulation_ms": draft.latency.query_formulation_ms,
                "retrieval_ms": draft.latency.retrieval_ms,
                "reranking_ms": draft.latency.reranking_ms,
                "compression_ms": draft.latency.compression_ms,
                "total_ms": total_ms,
            }
        )
        state.prediction = state.prediction.model_copy(update={"metrics": metrics, "latency": latency})
        state.prediction = evaluate_prediction_track_metrics(
            state.prediction,
            example=state.example,
            qrels=state.qrels,
            annotations=self.dataset_slice.annotations_by_example.get(state.example.example_id, []),
            context_condition=state.context_set.condition if state.context_set else self.settings.context_condition,
            context_set=state.context_set,
        )
        return state

    def _run_generation(self, state: PipelineState) -> PredictionRecord:
        from rag_benchmark.generation.budget import set_budget_stage
        set_budget_stage("generation")
        prepared = prepare_context(
            query=state.query_formulation.query,
            retrieved=[] if self.pipeline.stages.retrieval == "none" else state.retrieved,
            compressed=state.compressed_context if self.pipeline.stages.retrieval != "none" else None,
            max_tokens=self.pipeline.budget.max_context_tokens,
            model=self.component_settings.generator_model,
        )
        if self.pipeline.stages.generation == "citation_aware":
            generation, parsed, runtime = generate_citation_answer(
                question=state.example.question, context=prepared.evidence,
                task_track=self.task_track, task_type=state.example.task_type,
                generator=self.text_generator, model=self.component_settings.generator_model,
                strict=self.component_settings.strict_components,
                input_price=self.component_settings.input_cost_per_1m_tokens,
                output_price=self.component_settings.output_cost_per_1m_tokens,
            )
        else:
            generation = self.base_generator.generate(
                question=state.example.question, contexts=list(prepared.contexts),
                pipeline=self.pipeline.name, task_track=self.task_track,
                task_type=state.example.task_type,
            )
        prediction = build_prediction(
            run_id=self.run_id, pipeline_name=self.pipeline.name, example=state.example,
            qrels=state.qrels, contexts=state.retrieved, generation=generation,
            query_formulation=state.query_formulation, stage_latencies=state.stage_latencies,
            metric_ks=self.settings.retrieval_metric_ks, task_track=self.task_track,
        )
        metadata = {**prediction.metadata, "generation_context": prepared.metadata}
        if self.pipeline.stages.generation == "citation_aware":
            prediction = apply_citation_generation(prediction, parsed)
            metadata.update(prediction.metadata)
            metadata["citation_generation_runtime"] = runtime
            metadata["compressed_context"] = {
                "sentence_count": len(prepared.evidence.sentences),
                "provenance": prepared.evidence.provenance(),
                "metadata": prepared.evidence.metadata,
            }
        return prediction.model_copy(update={
            "metadata": metadata,
            "metrics": {**prediction.metrics,
                "context_tokens": float(prepared.metadata["used_tokens"]),
                "context_chunks_used": float(prepared.metadata["chunks_used"])},
        })


    def _evaluate_and_verify(self, state: PipelineState) -> PredictionRecord:
        from rag_benchmark.generation.budget import set_budget_stage
        set_budget_stage("verification")
        prediction = state.prediction
        if prediction is None:
            raise RuntimeError("generation must run before evaluation")
        if prediction.generator.get("schema_failure"):
            from rag_benchmark.evaluation.answers import assess_task
            return prediction.model_copy(update={"metadata": {**prediction.metadata,
                "task_assessment": assess_task(prediction, state.example),
                "verification_skip_reason": "generation_failed"}})
        if prediction.claim_checks:
            prediction = evaluate_evidence_consistency(prediction, judge=self.evidence_judge)
            prediction = evaluate_citation_quality(
                prediction,
                gold_chunk_ids=gold_chunk_ids(state.qrels),
                gold_sentence_refs=gold_sentence_refs(state.qrels),
            )
        pre_verification_support = prediction.metrics.get("claim_support_rate")
        from rag_benchmark.evaluation.answers import assess_task
        assessment_example = (replace(state.example, metadata={**state.example.metadata,
            **state.context_set.metadata, "expected_behavior": state.context_set.expected_behavior,
            "context_condition": state.context_set.condition})
            if state.context_set else state.example)
        pre_task = assess_task(prediction, assessment_example)
        pre_verification_qa = best_qa_scores(prediction.answer, state.example.answers)
        if self.pipeline.stages.verification in {"cove_style", "factored_cove"}:
            verification_start = time.perf_counter()
            if self.pipeline.stages.verification == "factored_cove":
                prediction = FactoredCoVeVerifier(
                    judge=self.evidence_judge,
                    generator=self.text_generator,
                    retriever=state.retriever,
                    config=FactoredCoVeConfig(
                        task_type=state.example.task_type,
                        task_revision={**self.pipeline.verification.get("task_revision", {}),
                             **({"detection_cache_root": str(self.settings.paired_artifact_root / "detection")}
                                if self.settings.paired_artifact_root else {}),
                             "detection_corpus_hash": self.corpus_hash,
                            "max_items": min(int(self.pipeline.verification.get("task_revision", {}).get("max_items",
                                self.pipeline.verification.get("questions_per_answer", 8))), self.pipeline.budget.max_verification_questions),
                            "max_context_tokens": min(int(self.pipeline.verification.get("task_revision", {}).get("max_context_tokens", 2048)),
                                self.pipeline.budget.max_context_tokens)},
                        questions_per_answer=int(
                            self.pipeline.verification.get("questions_per_answer", 8)
                        ),
                        retrieval_top_k=int(self.pipeline.verification.get("retrieval_top_k", 5)),
                        revise_answer=bool(self.pipeline.verification.get("revise_answer", True)),
                        revision_threshold=float(
                            self.pipeline.verification.get("revision_threshold", 0.8)
                        ),
                        replacement_support_threshold=float(
                            self.pipeline.verification.get("replacement_support_threshold", 0.8)
                        ),
                        revise_short_answer=bool(
                            self.pipeline.verification.get("revise_short_answer", False)
                        ),
                        allow_abstention=bool(
                            self.pipeline.task_profile.abstention_allowed
                            if state.example.task_type in {"scientific_claim_verification", "list_answer_citation"}
                            else self.pipeline.verification.get("allow_abstention", False)
                        ),
                        allow_claim_removal=bool(
                            self.pipeline.verification.get("allow_claim_removal", False)
                        ),
                        fail_on_generator_error=self.component_settings.strict_components,
                        answer_mode=self.pipeline.task_profile.answer_mode,
                        minimum_revision_similarity=float(
                            self.pipeline.verification.get("minimum_revision_similarity", 0.95)
                        ),
                        minimum_revision_length_ratio=float(
                            self.pipeline.verification.get("minimum_revision_length_ratio", 0.8)
                        ),
                        maximum_revision_length_ratio=float(
                            self.pipeline.verification.get("maximum_revision_length_ratio", 1.25)
                        ),
                    ),
                ).verify(prediction)
            else:
                prediction = CoVeVerifier(
                    judge=self.evidence_judge,
                    config=CoVeVerificationConfig(
                        questions_per_answer=int(self.pipeline.verification.get("questions_per_answer", 5)),
                        revise_answer=bool(self.pipeline.verification.get("revise_answer", True)),
                    ),
                ).verify(prediction)
            verification_ms = (time.perf_counter() - verification_start) * 1000
            state.stage_latencies["verification_ms"] = verification_ms
            verification_metadata = prediction.verification.metadata if prediction.verification else {}
            reused_detection = {row["source"]: row for row in verification_metadata.get("reused_detection_usage", [])}
            if reused_detection:
                reused_input = sum(int(row.get("call_usage", {}).get("input_tokens", 0)) for row in reused_detection.values())
                reused_output = sum(int(row.get("call_usage", {}).get("output_tokens", 0)) for row in reused_detection.values())
                reused_ms = sum(float(row.get("source_duration_ms", 0.0)) for row in reused_detection.values())
                verification_metadata = {**verification_metadata,
                    "physical_input_tokens": verification_metadata.get("input_tokens", 0),
                    "physical_output_tokens": verification_metadata.get("output_tokens", 0),
                    "reused_input_tokens": reused_input, "reused_output_tokens": reused_output,
                    "input_tokens": int(verification_metadata.get("input_tokens", 0)) + reused_input,
                    "output_tokens": int(verification_metadata.get("output_tokens", 0)) + reused_output,
                    "observed_verification_ms": verification_ms,
                    "reused_detection_ms": reused_ms,
                    "resource_accounting": "complete_method_includes_frozen_detection; dispatch ledger counts physical calls once"}
                verification_ms += reused_ms
                state.stage_latencies["verification_ms"] = verification_ms
                prediction = prediction.model_copy(update={"verification": prediction.verification.model_copy(
                    update={"metadata": verification_metadata}), "metadata": {**prediction.metadata,
                        "latency_measurement": "shared_detection_observations_plus_current_suffix_estimate"}})
            verification_input_tokens = int(verification_metadata.get("input_tokens", 0) or 0)
            verification_output_tokens = int(verification_metadata.get("output_tokens", 0) or 0)
            if verification_input_tokens or verification_output_tokens:
                tokens = prediction.tokens.model_copy(
                    update={
                        "input_tokens": prediction.tokens.input_tokens + verification_input_tokens,
                        "output_tokens": prediction.tokens.output_tokens + verification_output_tokens,
                        "total_tokens": (
                            prediction.tokens.input_tokens
                            + prediction.tokens.output_tokens
                            + verification_input_tokens
                            + verification_output_tokens
                        ),
                    }
                )
                verification_cost = estimate_cost(
                    input_tokens=verification_input_tokens,
                    output_tokens=verification_output_tokens,
                    input_cost_per_1m_tokens=self.component_settings.input_cost_per_1m_tokens,
                    output_cost_per_1m_tokens=self.component_settings.output_cost_per_1m_tokens,
                )
                prediction = prediction.model_copy(
                    update={
                        "tokens": tokens,
                        "cost": prediction.cost.model_copy(
                            update={
                                "estimated_cost": (
                                    (prediction.cost.estimated_cost or 0.0) + verification_cost
                                    if verification_cost is not None
                                    else prediction.cost.estimated_cost
                                )
                            }
                        ),
                    }
                )
            latency = prediction.latency.model_copy(update={"verification_ms": verification_ms})
            prediction = prediction.model_copy(update={"latency": latency})
            self._record_stage(
                state,
                stage="verification",
                method=self.pipeline.stages.verification,
                input_value={"answer": prediction.verification.original_answer if prediction.verification else None},
                output_value=prediction.verification.to_json_dict() if prediction.verification else {},
                latency_ms=verification_ms,
            )
            if prediction.claim_checks:
                prediction = evaluate_evidence_consistency(prediction, judge=self.evidence_judge)
                prediction = evaluate_citation_quality(
                    prediction,
                    gold_chunk_ids=gold_chunk_ids(state.qrels),
                    gold_sentence_refs=gold_sentence_refs(state.qrels),
                )
        prediction = refresh_answer_metrics(prediction, answers=state.example.answers)
        post_verification_qa = {"em": prediction.metrics.get("em", 0.0), "f1": prediction.metrics.get("f1", 0.0)}
        if self.pipeline.stages.verification != "none":
            post_task = assess_task(prediction, assessment_example)
            metrics = {
                **prediction.metrics,
                "pre_verification_em": pre_verification_qa["em"],
                "pre_verification_f1": pre_verification_qa["f1"],
                "post_verification_em": post_verification_qa["em"],
                "post_verification_f1": post_verification_qa["f1"],
                "verification_delta_f1": post_verification_qa["f1"] - pre_verification_qa["f1"],
                "verification_overcorrection": (float(pre_task["task_correct"] and not post_task["task_correct"])
                    if pre_task["task_correct"] is not None and post_task["task_correct"] is not None else None),
                "verification_repair_success": (float(not pre_task["task_correct"] and post_task["task_correct"])
                    if pre_task["task_correct"] is not None and post_task["task_correct"] is not None else None),
            }
            if pre_verification_support is not None:
                metrics["pre_verification_claim_support_rate"] = float(pre_verification_support)
                metrics["post_verification_claim_support_rate"] = prediction.metrics.get("claim_support_rate")
            for prefix, task in (("pre", pre_task), ("post", post_task)):
                if task["task_correct"] is not None:
                    metrics[f"{prefix}_verification_task_correct"] = int(task["task_correct"])
                if task["score"] is not None:
                    metrics[f"{prefix}_verification_task_score"] = task["score"]
            prediction = prediction.model_copy(update={"metrics": metrics})
            prediction = prediction.model_copy(update={"metadata": {**prediction.metadata,
                "pre_verification_task_score_metric": pre_task["score_metric"],
                "post_verification_task_score_metric": post_task["score_metric"]}})
        return attach_evaluation_coverage(prediction)

    def _attach_stage_metadata(self, state: PipelineState) -> PredictionRecord:
        prediction = state.prediction
        if prediction is None:
            raise RuntimeError("prediction missing")
        metadata = dict(prediction.metadata)
        from rag_benchmark.evaluation.identity import EvaluationIdentity
        metadata.update(EvaluationIdentity(state.example.dataset, self.experiment.dataset.source.version,
            state.example.split, self.settings.seed,
            state.context_set.context_set_id if state.context_set else None).metadata())
        metadata["query_formulation"] = state.query_formulation.to_metadata()
        if state.paired_stage_artifacts:
            metadata["paired_stage_artifacts"] = dict(state.paired_stage_artifacts)
        if state.paired_reuse:
            metadata["paired_stage_reuse"] = dict(state.paired_reuse)
            metadata["resource_accounting"] = {
                "tokens_and_cost": "complete_method_including_reused_prefix",
                "physical_requests": "dispatch_ledger_only; reused stages are not dispatched",
                "latency": "sum_of_shared_stage_observations_and_current_suffix; estimate"}
        metadata["stage_latencies"] = dict(state.stage_latencies)
        metadata["usage_by_stage"] = {
            "generation": {k: prediction.generator.get(k) for k in
                ("usage_source", "calls", "unknown_attempt_usage")},
            "query_formulation": state.query_formulation.metadata.get("call_usage", {"usage_source": "not_applicable"}),
            "verification": (prediction.verification.metadata.get("call_usage", {"usage_source": "not_applicable"})
                if prediction.verification else {"usage_source": "not_applicable"}),
        }
        if metadata.get("draft_reuse", {}).get("enabled"):
            metadata["usage_by_stage"]["draft"] = {
                "reused": True, "cost_inclusion": "includes_original_draft_cost",
                "source_run_id": metadata["draft_reuse"].get("draft_run_id"),
            }
        metadata["pipeline_config"] = {
            "name": self.pipeline.name,
            "stages": self.pipeline.stages.model_dump(mode="json"),
        }
        if state.context_set is not None:
            metadata["context_set"] = {
                "context_set_id": state.context_set.context_set_id,
                "condition": state.context_set.condition,
                "expected_behavior": state.context_set.expected_behavior,
                "chunk_count": len(state.context_set.chunk_ids),
                "metadata": state.context_set.metadata,
            }
        tokens = prediction.tokens.model_copy(
            update={
                "input_tokens": prediction.tokens.input_tokens + state.query_formulation.input_tokens,
                "output_tokens": prediction.tokens.output_tokens + state.query_formulation.output_tokens,
                "total_tokens": (
                    prediction.tokens.input_tokens
                    + prediction.tokens.output_tokens
                    + state.query_formulation.input_tokens
                    + state.query_formulation.output_tokens
                ),
            }
        )
        cost_value = prediction.cost.estimated_cost
        if state.query_formulation.estimated_cost is not None:
            cost_value = (cost_value or 0.0) + state.query_formulation.estimated_cost
        total_ms = component_total_ms(
            stage_latencies=state.stage_latencies,
            generation_ms=prediction.latency.generation_ms,
        )
        pipeline_overhead = pipeline_overhead_ms(
            stage_latencies=state.stage_latencies,
            generation_ms=prediction.latency.generation_ms,
        )
        latency = prediction.latency.model_copy(
            update={
                "retrieval_ms": state.stage_latencies.get("retrieval_ms"),
                "query_formulation_ms": state.stage_latencies.get("query_formulation_ms"),
                "reranking_ms": state.stage_latencies.get("reranking_ms"),
                "compression_ms": state.stage_latencies.get("compression_ms"),
                "generation_ms": prediction.latency.generation_ms,
                "verification_ms": state.stage_latencies.get("verification_ms"),
                "total_ms": total_ms,
            }
        )
        metrics = {
            **prediction.metrics,
            "query_formulation_ms": state.query_formulation.latency_ms,
            "reranking_ms": state.stage_latencies.get("reranking_ms", 0.0),
            "compression_ms": state.stage_latencies.get("compression_ms", 0.0),
            "total_ms": total_ms,
            "input_tokens": tokens.input_tokens,
            "output_tokens": tokens.output_tokens,
            "pipeline_overhead_ms": pipeline_overhead,
        }
        if cost_value is not None:
            metrics["estimated_cost"] = cost_value
        if state.context_set is not None and isinstance(state.context_set.metadata.get("noise_rate"), (int, float)):
            metrics["context_noise_rate"] = float(state.context_set.metadata["noise_rate"])
        return prediction.model_copy(
            update={
                "metadata": metadata,
                "tokens": tokens,
                "cost": prediction.cost.model_copy(update={"estimated_cost": cost_value}),
                "latency": latency,
                "metrics": metrics,
                "stage_traces": list(state.stage_traces),
            }
        )

    def _build_trace(self, state: PipelineState) -> RetrievalTraceRecord:
        trace = build_retrieval_trace_record(
            example=state.example,
            qrels=state.qrels,
            retrieved=state.retrieved,
            retrieval_ms=state.stage_latencies.get("retrieval_ms", 0.0),
            metric_ks=self.settings.retrieval_metric_ks,
        )
        metadata = dict(trace.metadata)
        metadata["query_formulation"] = state.query_formulation.to_metadata()
        if state.context_set is not None:
            metadata["context_set_id"] = state.context_set.context_set_id
            metadata["context_condition"] = state.context_set.condition
            metadata["context_metadata"] = state.context_set.metadata
        return trace.model_copy(update={"metadata": metadata})

    def _record_stage(
        self,
        state: PipelineState,
        *,
        stage: str,
        method: str,
        input_value: Any,
        output_value: Any,
        latency_ms: float,
        status: str = "completed",
        artifact_path: str | None = None,
    ) -> None:
        state.stage_traces.append(
            StageTraceRecord(
                example_id=state.example.example_id,
                stage=stage,
                method=method,
                status=status,
                latency_ms=latency_ms,
                input_hash=canonical_hash(input_value),
                output_hash=canonical_hash(output_value),
                artifact_path=artifact_path,
            )
        )

    @staticmethod
    def _query_from_metadata(payload):
        row = dict(payload)
        row.pop("_budget_calls", None)
        row["queries"] = tuple(row.get("queries", ()))
        row["subquestions"] = tuple(row.get("subquestions", ()))
        return QueryFormulationResult(**row)

    def _paired_load(self, state, stage, inputs, config):
        if self.paired_artifacts is None:
            return None
        artifact = self.paired_artifacts.load(example_id=state.example.example_id,
            stage=stage, inputs=inputs, config=config)
        if artifact is not None:
            anchor = {"source_run_id": artifact.metadata["source_run_id"],
                "input_hash": artifact.input_hash, "output_hash": artifact.output_hash,
                "artifact_path": str(artifact.path),
                "logical_latency_ms": artifact.metadata["logical_latency_ms"]}
            state.paired_reuse[stage] = anchor
            state.paired_stage_artifacts[stage] = anchor
        return artifact

    def _paired_save(self, state, stage, inputs, config, payload, latency_ms):
        if self.paired_artifacts is not None:
            if stage == "query_formulation":
                from rag_benchmark.generation.budget import CURRENT_BUDGET
                budget = CURRENT_BUDGET.get()
                payload = {**payload, "_budget_calls": budget.snapshot()["calls"] if budget else []}
            artifact = self.paired_artifacts.save(example_id=state.example.example_id, stage=stage,
                inputs=inputs, config=config, payload=payload,
                source_run_id=self.run_id, latency_ms=latency_ms)
            state.paired_stage_artifacts[stage] = {"source_run_id": artifact.metadata["source_run_id"],
                "input_hash": artifact.input_hash, "output_hash": artifact.output_hash,
                "artifact_path": str(artifact.path), "logical_latency_ms": artifact.metadata["logical_latency_ms"]}

    @staticmethod
    def _serialize_retrieval(retrieved: list[RetrievedChunk]) -> JsonObject:
        return {
            "results": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "rank": result.rank,
                    "score": result.score,
                    "metadata": dict(result.metadata),
                }
                for result in retrieved
            ]
        }

    def _deserialize_retrieval(self, payload: JsonObject) -> list[RetrievedChunk]:
        results = []
        for row in payload.get("results", []):
            chunk_id = str(row["chunk_id"])
            if chunk_id not in self.corpus_by_id:
                raise ValueError(f"stage cache references missing chunk: {chunk_id}")
            results.append(
                RetrievedChunk(
                    chunk=self.corpus_by_id[chunk_id],
                    rank=int(row["rank"]),
                    score=float(row["score"]),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return results

    def _oracle_retrieval(self, qrels: list[Qrel]) -> list[RetrievedChunk]:
        ranked_qrels = sorted(qrels, key=lambda row: (-row.relevance, row.chunk_id))
        results = []
        seen = set()
        for qrel in ranked_qrels:
            if qrel.chunk_id in seen or qrel.chunk_id not in self.corpus_by_id:
                continue
            seen.add(qrel.chunk_id)
            results.append(
                RetrievedChunk(
                    chunk=self.corpus_by_id[qrel.chunk_id],
                    rank=len(results) + 1,
                    score=float(qrel.relevance),
                    metadata={"retriever": "oracle_gold_context", "relevance": qrel.relevance},
                )
            )
        return results

    def _retriever_for_example(self, example: Example) -> tuple[ContextSet | None, Any | None]:
        if self.pipeline.stages.retrieval in {"none", "oracle"}:
            contexts = self.dataset_slice.context_sets_by_example.get(example.example_id, [])
            context = select_context_set(contexts, condition=self.settings.context_condition,
                metadata=self.settings.context_metadata) if contexts else None
            return context, self.retriever
        if self.candidate_cache is not None:
            bundle = self.candidate_cache.load(example.example_id)
            self.corpus_by_id.update({chunk.chunk_id: chunk for chunk in bundle.chunks})
            retriever = build_retriever(
                pipeline=self.pipeline,
                corpus=bundle.chunks,
                settings=replace(self.component_settings, dense_index_path=None),
                precomputed_vectors=bundle.vectors,
                precomputed_embedding_model=bundle.embedding_model,
            )
            return None, retriever
        contexts = self.dataset_slice.context_sets_by_example.get(example.example_id, [])
        if not contexts:
            return None, self.retriever
        context = select_context_set(
            contexts,
            condition=self.settings.context_condition,
            metadata=self.settings.context_metadata,
        )
        if context.context_set_id in self._scoped_retrievers:
            return context, self._scoped_retrievers[context.context_set_id]
        missing = [chunk_id for chunk_id in context.chunk_ids if chunk_id not in self.corpus_by_id]
        if missing:
            raise ValueError(
                f"Context set {context.context_set_id} references {len(missing)} unloaded chunks; "
                f"first={missing[:3]}"
            )
        scoped_corpus = [self.corpus_by_id[chunk_id] for chunk_id in context.chunk_ids]
        scoped_settings = self.component_settings
        if scoped_settings.dense_index_path is not None:
            scoped_settings = replace(
                scoped_settings,
                dense_index_path=context_index_dir(scoped_settings.dense_index_path, context),
            )
        retriever = build_retriever(
            pipeline=self.pipeline,
            corpus=scoped_corpus,
            settings=scoped_settings,
        )
        self._scoped_retrievers[context.context_set_id] = retriever
        return context, retriever

    def _retrieved_from_prediction(self, prediction: PredictionRecord) -> list[RetrievedChunk]:
        retrieved = []
        for row in prediction.retrieved_chunks:
            if row.chunk_id not in self.corpus_by_id:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=self.corpus_by_id[row.chunk_id],
                    rank=row.rank,
                    score=row.score,
                    metadata={**row.metadata, "reused_from_draft": True},
                )
            )
        return retrieved

    def _run_manifest(self) -> JsonObject:
        return {
            "schema_version": "run-manifest-v1",
            "run_identity": self.identity,
            "run_id": self.run_id,
            "dataset": self.dataset_slice.dataset_name,
            "split": self.dataset_slice.split_name,
            "task_track": self.task_track,
            "candidate_scope": self.dataset_slice.candidate_scope,
            "candidate_cache": (
                {
                    "path": str(self.candidate_cache.root),
                    "schema_version": self.candidate_cache.manifest.get("schema_version"),
                    "cache_hash": self.candidate_cache.scope_hash,
                    "embedding_model": self.candidate_cache.embedding_model,
                }
                if self.candidate_cache is not None
                else None
            ),
            "context_condition": self.settings.context_condition,
            "context_metadata": self.settings.context_metadata,
            "corpus_hash": self.corpus_hash,
            "pipeline_hash": canonical_hash(self.pipeline.to_json_dict()),
            "pipeline": self.pipeline.to_json_dict(),
            "method_definition": self.pipeline.method.to_json_dict() if self.pipeline.method else None,
            "execution_profile": self.component_settings.execution_profile,
            "strict_components": self.component_settings.strict_components,
            "paired_draft_reuse": bool(self.settings.draft_predictions),
            "draft_run_ids": sorted(
                {
                    prediction.run_id
                    for prediction in (self.settings.draft_predictions or {}).values()
                    if prediction.run_id
                }
            ),
            "model_refs": self.experiment.model_refs,
            "components": {
                "query_formulator": getattr(self.query_formulator, "name", type(self.query_formulator).__name__),
                "retriever": (
                    "fullwiki_frozen_candidate_scope"
                    if self.candidate_cache is not None and self.pipeline.stages.retrieval not in {"none", "oracle"}
                    else "example_scoped"
                    if self._uses_scoped_contexts and self.pipeline.stages.retrieval not in {"none", "oracle"}
                    else type(self.retriever).__name__ if self.retriever is not None else self.pipeline.stages.retrieval
                ),
                "reranker": type(self.reranker).__name__ if self.reranker is not None else "none",
                "compressor": type(self.compressor).__name__ if self.compressor is not None else "none",
                "evidence_judge": getattr(self.evidence_judge, "name", type(self.evidence_judge).__name__),
                "generator": {
                    "provider": getattr(self.base_generator, "provider", "unknown"),
                    "model": getattr(self.base_generator, "model", "unknown"),
                },
            },
        }

    def output_dir(self) -> Path:
        root = self.settings.output_root or Path(self.experiment.paths.outputs_root if self.experiment.paths else "outputs")
        return (root / "formal" / self.run_id).resolve()


def build_prediction(
    *,
    run_id: str,
    pipeline_name: str,
    example: Example,
    qrels: list[Qrel],
    contexts: list[RetrievedChunk],
    generation: GenerationResult,
    query_formulation: QueryFormulationResult,
    stage_latencies: dict[str, float],
    metric_ks: tuple[int, ...],
    task_track: str = "short_answer",
) -> PredictionRecord:
    qa = best_qa_scores(generation.answer, example.answers)
    retrieved_ids = [result.chunk.chunk_id for result in contexts]
    gold_ids = gold_chunk_ids(qrels)
    retrieval_metric_values: dict[str, Any] = {}
    if contexts:
        for k in metric_ks:
            retrieval_metric_values.update(retrieval_scores(retrieved_ids, gold_ids, k=k))
    total_ms = (
        stage_latencies.get("query_formulation_ms", 0.0)
        + stage_latencies.get("retrieval_ms", 0.0)
        + generation.generation_ms
        + stage_latencies.get("reranking_ms", 0.0)
        + stage_latencies.get("compression_ms", 0.0)
    )
    metrics = {
        **qa,
        **{key: value for key, value in retrieval_metric_values.items() if value is not None},
        "retrieval_ms": stage_latencies.get("retrieval_ms", 0.0),
        "generation_ms": generation.generation_ms,
        "total_ms": total_ms,
        "input_tokens": generation.input_tokens + query_formulation.input_tokens,
        "output_tokens": generation.output_tokens + query_formulation.output_tokens,
    }
    if generation.estimated_cost is not None or query_formulation.estimated_cost is not None:
        metrics["estimated_cost"] = (generation.estimated_cost or 0.0) + (
            query_formulation.estimated_cost or 0.0
        )
    if gold_ids and pipeline_name != "closed_book":
        largest_k = max(metric_ks)
        retrieval_success = float(retrieval_metric_values.get(f"any_gold_hit@{largest_k}", 0.0) or 0.0)
        metrics.update(
            {
                "retrieval_success": retrieval_success,
                "retrieval_failure": 1.0 - retrieval_success,
                "context_utilization_success": float(retrieval_success == 1.0 and qa["em"] == 1.0),
                "context_utilization_failure": float(retrieval_success == 1.0 and qa["em"] == 0.0),
            }
        )
    return PredictionRecord(
        run_id=run_id,
        example_id=example.example_id,
        pipeline=pipeline_name,
        question=example.question,
        gold_answers=example.answers,
        answer=generation.answer,
        task_track=task_track,
        answer_bundle=build_answer_bundle(generation.answer, task_track=task_track, task_type=example.task_type),
        retrieved_chunks=serialize_retrieved_chunks(contexts, qrels),
        latency=LatencyRecord(
            retrieval_ms=stage_latencies.get("retrieval_ms"),
            generation_ms=generation.generation_ms,
            total_ms=total_ms,
        ),
        tokens=TokenUsageRecord(
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            total_tokens=generation.input_tokens + generation.output_tokens,
        ),
        cost=CostRecord(estimated_cost=generation.estimated_cost),
        generator=generation.generator,
        metrics=metrics,
    )


def component_total_ms(*, stage_latencies: dict[str, float], generation_ms: float) -> float:
    return (
        stage_latencies.get("query_formulation_ms", 0.0)
        + stage_latencies.get("retrieval_ms", 0.0)
        + stage_latencies.get("reranking_ms", 0.0)
        + stage_latencies.get("compression_ms", 0.0)
        + generation_ms
        + stage_latencies.get("verification_ms", 0.0)
    )


def pipeline_overhead_ms(*, stage_latencies: dict[str, float], generation_ms: float) -> float:
    return max(0.0, stage_latencies.get("generation_stage_ms", 0.0) - generation_ms)


def refresh_answer_metrics(prediction: PredictionRecord, *, answers: list[JsonObject]) -> PredictionRecord:
    scored_answer = prediction.answer
    if prediction.task_track == "short_answer" and prediction.answer_bundle is not None:
        scored_answer = prediction.answer_bundle.short_answer or prediction.answer
    qa = best_qa_scores(scored_answer, answers)
    metrics = {**prediction.metrics, **qa}
    return prediction.model_copy(update={"metrics": metrics})


def gold_sentence_refs(qrels: list[Qrel]) -> set[tuple[str, int]]:
    refs = set()
    for qrel in qrels:
        sentence_id = qrel.metadata.get("sentence_id")
        if sentence_id is not None:
            refs.add((qrel.chunk_id, int(sentence_id)))
    return refs


def load_formal_dataset_slice(
    *,
    experiment: ExperimentConfig,
    split_profile: str,
    limit: int,
    seed: int,
    relevance_threshold: int,
    candidate_cache_path: Path | None = None,
    sample_manifest: Path | None = None,
) -> DatasetSlice:
    if experiment.paths is None:
        raise ValueError("ExperimentConfig.paths is required for dataset loading")
    root_key = experiment.dataset.source.root_key
    if root_key not in experiment.paths.datasets:
        raise KeyError(f"Dataset root key {root_key!r} not found in paths config")
    dataset_root = experiment.paths.datasets[root_key].root
    processed_root = (
        dataset_root
        if (dataset_root / "examples.jsonl").exists()
        else dataset_root / "processed"
    )
    if sample_manifest is not None:
        from rag_benchmark.sampling import load_sample_manifest
        load_sample_manifest(sample_manifest, dataset=experiment.dataset.name,
                             split_path=processed_root / "splits" / f"{split_profile}.jsonl",
                             seed=seed, data_version=experiment.dataset.source.version)
    result = build_normalized_split_slice(
        processed_root=processed_root,
        dataset_name=experiment.dataset.name,
        split_profile=split_profile,
        limit=limit,
        relevance_threshold=relevance_threshold,
        corpus_scope=str(experiment.dataset.processing.get("corpus_scope", "per_example_pooled_context")),
        retrieval_required=experiment.pipeline.stages.retrieval != "none",
        candidate_cache_path=candidate_cache_path,
        sample_manifest=sample_manifest,
    )
    if experiment.dataset.name == "garage":
        from rag_benchmark.data_semantics import correct_garage_views
        review_path = experiment.dataset.processing.get("semantic_reviews_path")
        reviews = json.loads(Path(review_path).read_text(encoding="utf-8")) if review_path else None
        examples, contexts = correct_garage_views(processed_root, {e.example_id: e for e in result.examples}, result.context_sets_by_example, reviews=reviews)
        result = replace(result, examples=list(examples.values()), context_sets_by_example=contexts)
    if experiment.dataset.name == "rgb_en":
        from rag_benchmark.rgb_protocol import apply_rgb_protocol
        review_path = experiment.dataset.processing.get("semantic_reviews_path")
        reviews = json.loads(Path(review_path).read_text(encoding="utf-8")) if review_path else None
        result = apply_rgb_protocol(result, review_overlay=reviews)
    return result


def select_context_set(
    contexts: list[ContextSet],
    *,
    condition: str | None = None,
    metadata: JsonObject | None = None,
) -> ContextSet:
    matches = list(contexts)
    if condition is not None:
        matches = [context for context in matches if context.condition == condition]
    for key, expected in (metadata or {}).items():
        matches = [context for context in matches if context.metadata.get(key) == expected]
    if len(matches) != 1:
        available = [
            {
                "context_set_id": context.context_set_id,
                "condition": context.condition,
                "metadata": context.metadata,
            }
            for context in contexts
        ]
        raise ValueError(
            f"Expected exactly one context set after filtering, found {len(matches)}. "
            f"condition={condition!r}, metadata={metadata or {}}, available={available}"
        )
    return matches[0]


def filter_dataset_slice_by_context(
    dataset_slice: DatasetSlice,
    *,
    condition: str | None,
    metadata: JsonObject | None = None,
) -> DatasetSlice:
    """Keep only examples with exactly one context matching the frozen stress condition."""

    if condition is None and not metadata:
        return dataset_slice
    selected_examples = []
    selected_contexts = {}
    for example in dataset_slice.examples:
        contexts = dataset_slice.context_sets_by_example.get(example.example_id, [])
        matches = [context for context in contexts if condition is None or context.condition == condition]
        for key, expected in (metadata or {}).items():
            matches = [context for context in matches if context.metadata.get(key) == expected]
        if len(matches) == 1:
            selected_examples.append(example)
            selected_contexts[example.example_id] = contexts
        elif len(matches) > 1:
            raise ValueError(
                f"Context filter is ambiguous for {example.example_id}: condition={condition!r}, "
                f"metadata={metadata or {}}"
            )
    selected_ids = {example.example_id for example in selected_examples}
    return replace(
        dataset_slice,
        examples=selected_examples,
        qrels_by_example={
            example_id: rows
            for example_id, rows in dataset_slice.qrels_by_example.items()
            if example_id in selected_ids
        },
        context_sets_by_example=selected_contexts,
        candidate_responses_by_example={
            example_id: rows
            for example_id, rows in dataset_slice.candidate_responses_by_example.items()
            if example_id in selected_ids
        },
        annotations_by_example={
            example_id: rows
            for example_id, rows in dataset_slice.annotations_by_example.items()
            if example_id in selected_ids
        },
    )


def context_index_dir(base: Path, context: ContextSet) -> Path:
    scope_hash = canonical_hash(
        {"context_set_id": context.context_set_id, "chunk_ids": context.chunk_ids}
    )[:20]
    return base / "scopes" / scope_hash


def parse_context_metadata_filters(values: list[str]) -> JsonObject:
    filters: JsonObject = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Context metadata filter must use KEY=JSON_VALUE: {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Context metadata filter key must not be empty")
        try:
            filters[key] = json.loads(raw)
        except json.JSONDecodeError:
            filters[key] = raw
    return filters


def build_normalized_split_slice(
    *,
    processed_root: Path,
    dataset_name: str,
    split_profile: str,
    limit: int,
    relevance_threshold: int,
    corpus_scope: str = "per_example_pooled_context",
    retrieval_required: bool = True,
    candidate_cache_path: Path | None = None,
    sample_manifest: Path | None = None,
) -> DatasetSlice:
    split_path = processed_root / "splits" / f"{split_profile}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Requested split {split_profile!r} does not exist: {split_path}")
    examples_by_id = load_examples(processed_root)
    qrels_by_example = load_qrels(processed_root, relevance_threshold=relevance_threshold)
    selected_response_ids: set[str] | None = None
    split_rows = read_jsonl_rows(split_path)
    if sample_manifest is not None:
        from rag_benchmark.sampling import load_sample_manifest
        selection = load_sample_manifest(sample_manifest, dataset=dataset_name, split_path=split_path)
        wanted = set(selection["sample_ids"])
        split_rows = [r for r in split_rows if str(r.get(selection["unit"])) in wanted]
        limit = len(split_rows)
    response_rows = [row for row in split_rows if row.get("response_id")]
    if response_rows:
        response_rows = response_rows[:limit]
        selected_response_ids = {str(row["response_id"]) for row in response_rows}
        selected_ids = list(dict.fromkeys(str(row["example_id"]) for row in response_rows))
    else:
        selected_ids = [row["example_id"] for row in split_rows]
    unknown_ids = set(selected_ids) - examples_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Requested split contains {len(unknown_ids)} unknown example IDs")
    examples = [examples_by_id[example_id] for example_id in selected_ids]
    if selected_response_ids is None:
        examples = examples[:limit]
    selected_example_ids = {example.example_id for example in examples}
    context_sets_by_example = load_context_sets(processed_root, example_ids=selected_example_ids)
    candidate_responses_by_example = load_candidate_responses(
        processed_root,
        example_ids=selected_example_ids,
        response_ids=selected_response_ids,
    )
    annotations_by_example = load_annotations(
        processed_root,
        example_ids=selected_example_ids,
        target_ids=selected_response_ids,
    )
    source_ids = {str(example.metadata.get("source_id")) for example in examples}
    required_gold_ids = {
        qrel.chunk_id for example in examples for qrel in qrels_by_example.get(example.example_id, [])
    }
    if corpus_scope == "kilt_full_wikipedia_2019":
        if retrieval_required and candidate_cache_path is None:
            raise RuntimeError(
                "KILT FullWiki retrieval requires a disk-backed full-corpus index or a frozen "
                "candidate cache. Refusing to substitute the selected examples' gold chunks, "
                "because that would leak relevance labels and invalidate retrieval metrics."
            )
        corpus_by_id = {}
    elif corpus_scope in {"scifact_full_corpus", "full_corpus"}:
        corpus_by_id = load_all_corpus(processed_root)
    else:
        corpus_by_id = load_corpus_for_source_ids(
            processed_root,
            source_ids,
            include_chunk_ids=required_gold_ids,
        )
    return DatasetSlice(
        dataset_name=dataset_name,
        split_name=split_profile,
        candidate_scope=(
            f"{corpus_scope}:fullwiki_fts_candidate_restricted"
            if corpus_scope == "kilt_full_wikipedia_2019" and candidate_cache_path is not None
            else corpus_scope
        ),
        examples=examples,
        corpus=sorted(corpus_by_id.values(), key=lambda chunk: chunk.chunk_id),
        qrels_by_example={example.example_id: qrels_by_example.get(example.example_id, []) for example in examples},
        context_sets_by_example=context_sets_by_example,
        candidate_responses_by_example=candidate_responses_by_example,
        annotations_by_example=annotations_by_example,
        candidate_cache_path=(candidate_cache_path.resolve() if candidate_cache_path else None),
    )


def read_jsonl_rows(path: Path) -> list[JsonObject]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_second_hop_queries(
    *,
    question: str,
    seeds: list[RetrievedChunk],
    limit: int,
) -> list[str]:
    if limit < 1:
        return []
    question_key = question.casefold()
    queries = []
    seen = set()
    for result in seeds:
        title = (result.chunk.title or "").strip()
        if not title or title.casefold() in question_key or title.casefold() in seen:
            continue
        seen.add(title.casefold())
        queries.append(f"{question} {title}")
        if len(queries) >= limit:
            break
    return queries


def retrieve_query_sets(
    retriever: Any,
    queries: list[str],
    *,
    top_k: int,
) -> list[tuple[str, list[RetrievedChunk]]]:
    """Use a retriever's batch API when available while preserving query order."""

    if not queries:
        return []
    retrieve_many = getattr(retriever, "retrieve_many", None)
    if callable(retrieve_many):
        result_batches = retrieve_many(queries, top_k=top_k)
        if len(result_batches) != len(queries):
            raise RuntimeError(
                "Retriever batch result count does not match the number of retrieval queries"
            )
        return list(zip(queries, result_batches))
    return [(query, retriever.retrieve(query, top_k=top_k)) for query in queries]
