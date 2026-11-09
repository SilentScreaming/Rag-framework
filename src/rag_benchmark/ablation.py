"""Ablation runner built on top of the formal config-driven pipeline runner."""

from __future__ import annotations
from rag_benchmark.run_identity import run_identity

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_benchmark.config import ExperimentConfig, PipelineConfig
from rag_benchmark.evaluation.tracks import reliability_metrics
from rag_benchmark.data import DatasetSlice
from rag_benchmark.reporting import write_unified_metrics_report
from rag_benchmark.runner import (
    ConfigDrivenPipelineRunner,
    FormalRunResult,
    FormalRunSettings,
    adapt_pipeline_to_dataset,
    load_completed_formal_run,
)


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class AblationStep:
    name: str
    pipeline: PipelineConfig


@dataclass(frozen=True)
class AblationRunResult:
    ablation_id: str
    output_dir: Path | None
    step_results: list[FormalRunResult]
    metrics_rows: list[JsonObject]
    delta_rows: list[JsonObject]


def build_leave_one_out_steps(*, full: PipelineConfig) -> list[AblationStep]:
    """Five conditional component contributions, with one shared full configuration."""
    from copy import deepcopy
    variants = [("01_full", None), ("02_without_query", "query_formulation"),
                ("03_without_hybrid", "retrieval"), ("04_without_reranker", "reranking"),
                ("05_without_compression", "compression"), ("06_without_citation", "generation")]
    expected = {"query_formulation": "multi_query_rewrite", "retrieval": "hybrid",
                "reranking": "cross_encoder", "compression": "sentence_top_n", "generation": "citation_aware"}
    if any(getattr(full.stages, k) != v for k, v in expected.items()):
        raise ValueError("leave-one-out requires an explicitly enabled five-component full pipeline")
    top_k = int(full.reranking.get("top_k", 12))
    candidates = int(full.retrieval.get("final_top_k", 100))
    steps = []
    for name, removed in variants:
        payload = deepcopy(full.to_json_dict())
        payload.update(name=f"ablation_{name}", base_pipeline=None)
        payload["stages"]["verification"] = "none"
        payload["logging"].update(ablation_design="leave-one-out-v1", removed_component=removed,
                                  ablation_final_top_k=top_k, selection_basis="fixed_research_design")
        if removed:
            payload["stages"][removed] = {"query_formulation": "original_question", "retrieval": "dense",
                                          "generation": "context_grounded"}.get(removed, "none")
        if removed == "retrieval":
            payload["retrieval"]["top_k"] = candidates
        if removed == "query_formulation":
            # Derived second-hop queries also transform the original question.
            payload["retrieval"]["iterative_hops"] = 1
        if removed == "generation":
            payload["task_profile"]["citations_required"] = False
        steps.append(AblationStep(name, PipelineConfig.model_validate(payload)))
    return steps


def build_default_ablation_steps() -> list[AblationStep]:
    return [
        AblationStep("01_vanilla", make_pipeline_config(name="ablation_vanilla", query="original_question", retrieval="dense")),
        AblationStep("02_query_rewrite", make_pipeline_config(name="ablation_query_rewrite", query="multi_query_rewrite", retrieval="dense")),
        AblationStep("03_hybrid", make_pipeline_config(name="ablation_hybrid", query="multi_query_rewrite", retrieval="hybrid")),
        AblationStep(
            "04_reranker",
            make_pipeline_config(name="ablation_reranker", query="multi_query_rewrite", retrieval="hybrid", reranking="cross_encoder"),
        ),
        AblationStep(
            "05_compression",
            make_pipeline_config(
                name="ablation_compression",
                query="multi_query_rewrite",
                retrieval="hybrid",
                reranking="cross_encoder",
                compression="sentence_top_n",
            ),
        ),
        AblationStep(
            "06_citation",
            make_pipeline_config(
                name="ablation_citation",
                query="multi_query_rewrite",
                retrieval="hybrid",
                reranking="cross_encoder",
                compression="sentence_top_n",
                generation="citation_aware",
            ),
        ),
        AblationStep(
            "07_verification",
            make_pipeline_config(
                name="ablation_verification",
                query="multi_query_rewrite",
                retrieval="hybrid",
                reranking="cross_encoder",
                compression="sentence_top_n",
                generation="citation_aware",
                verification="factored_cove",
            ),
        ),
    ]


def build_profile_ablation_steps(
    *,
    vanilla: PipelineConfig,
    best_practice: PipelineConfig,
    verification: PipelineConfig,
) -> list[AblationStep]:
    """Compose the ablation from the exact resolved formal pipeline settings."""

    best_payload = best_practice.to_json_dict()
    vanilla_payload = vanilla.to_json_dict()
    verification_payload = verification.to_json_dict()

    def make_step(
        name: str,
        *,
        query: bool = False,
        hybrid: bool = False,
        reranker: bool = False,
        compression: bool = False,
        citation: bool = False,
        verify: bool = False,
    ) -> AblationStep:
        payload = dict(best_payload)
        payload["name"] = f"ablation_{name.removeprefix('0').replace('_', '')}"
        payload["base_pipeline"] = None
        payload["stages"] = {
            "query_formulation": (
                best_practice.stages.query_formulation if query else vanilla.stages.query_formulation
            ),
            "retrieval": best_practice.stages.retrieval if hybrid else vanilla.stages.retrieval,
            "reranking": best_practice.stages.reranking if reranker else "none",
            "compression": best_practice.stages.compression if compression else "none",
            "generation": best_practice.stages.generation if citation else vanilla.stages.generation,
            "verification": verification.stages.verification if verify else "none",
        }
        payload["query_formulation"] = (
            dict(best_payload.get("query_formulation", {}))
            if query
            else dict(vanilla_payload.get("query_formulation", {}))
        )
        payload["retrieval"] = (
            dict(best_payload.get("retrieval", {}))
            if hybrid
            else dict(vanilla_payload.get("retrieval", {}))
        )
        payload["generation"] = (
            dict(best_payload.get("generation", {}))
            if citation
            else dict(vanilla_payload.get("generation", {}))
        )
        if verify:
            payload["verification"] = dict(verification_payload.get("verification", {}))
            payload["budget"] = dict(verification_payload.get("budget", {}))
        return AblationStep(name, PipelineConfig.model_validate(payload))

    return [
        make_step("01_vanilla"),
        make_step("02_query_rewrite", query=True),
        make_step("03_hybrid", query=True, hybrid=True),
        make_step("04_reranker", query=True, hybrid=True, reranker=True),
        make_step(
            "05_compression",
            query=True,
            hybrid=True,
            reranker=True,
            compression=True,
        ),
        make_step(
            "06_citation",
            query=True,
            hybrid=True,
            reranker=True,
            compression=True,
            citation=True,
        ),
        make_step(
            "07_verification",
            query=True,
            hybrid=True,
            reranker=True,
            compression=True,
            citation=True,
            verify=True,
        ),
    ]


def run_ablation_suite(
    *,
    repo_root: Path,
    base_experiment: ExperimentConfig,
    dataset_slice: DatasetSlice,
    settings: FormalRunSettings,
    ablation_id: str,
    steps: list[AblationStep] | None = None,
) -> AblationRunResult:
    steps = steps or build_default_ablation_steps()
    output_dir = ablation_output_dir(base_experiment=base_experiment, settings=settings, ablation_id=ablation_id)
    step_results: list[FormalRunResult] = []
    metrics_rows: list[JsonObject] = []
    fixed_design = all(step.pipeline.logging.get("ablation_design") == "leave-one-out-v1" for step in steps)
    paired_root = settings.paired_artifact_root or (output_dir / "paired_artifacts" if fixed_design and output_dir else None)
    if fixed_design and paired_root is None:
        raise ValueError("leave-one-out requires saved paired artifacts; provide an output root or paired_artifact_root")

    for index, step in enumerate(steps, start=1):
        pipeline = adapt_pipeline_to_dataset(step.pipeline, base_experiment.dataset)
        step_experiment = base_experiment.model_copy(update={"pipeline": pipeline})
        step_settings = FormalRunSettings(
            run_id=step.name,
            split_profile=settings.split_profile,
            limit=settings.limit,
            seed=settings.seed,
            gold_relevance_threshold=settings.gold_relevance_threshold,
            output_root=(output_dir / step.name) if output_dir else None,
            component_settings=settings.component_settings,
            retrieval_metric_ks=settings.retrieval_metric_ks,
            save_outputs=settings.save_outputs,
            stage_cache_root=settings.stage_cache_root,
            replay_stages=settings.replay_stages,
            write_stage_cache=settings.write_stage_cache,
            bootstrap_samples=settings.bootstrap_samples,
            context_condition=settings.context_condition,
            context_metadata=settings.context_metadata,
            paired_artifact_root=paired_root,
        )
        result = None
        if output_dir is not None and settings.save_outputs:
            result = load_completed_formal_run(
                output_dir / step.name / "formal" / step.name,
                run_id=step.name,
                pipeline=pipeline.name,
                expected_identity=run_identity(repo_root=repo_root, experiment=step_experiment, dataset_slice=dataset_slice, settings=step_settings),
                expected_example_ids={example.example_id for example in dataset_slice.examples},
            )
        if result is None:
            if output_dir is not None and settings.save_outputs and (output_dir / step.name / "formal" / step.name).exists():
                raise ValueError(f"Existing ablation run failed identity validation: {step.name}; use a new ablation ID.")
            result = ConfigDrivenPipelineRunner(
                repo_root=repo_root,
                experiment=step_experiment,
                dataset_slice=dataset_slice,
                settings=step_settings,
            ).run()
        step_results.append(result)
        aggregate = {
            **result.metrics["pipelines"][pipeline.name],
            **reliability_metrics(result.predictions),
        }
        metrics_rows.append({"step_index": index, "step_name": step.name, "pipeline": pipeline.name,
                             "evaluation_protocol": result.metrics.get("evaluation_protocol"), **aggregate})

    delta_rows = compute_leave_one_out_deltas(metrics_rows) if fixed_design else compute_delta_rows(metrics_rows)
    if settings.save_outputs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_ablation_outputs(
            output_dir=output_dir,
            ablation_id=ablation_id,
            metrics_rows=metrics_rows,
            delta_rows=delta_rows,
        )
        write_unified_metrics_report(
            output_dir,
            [result.metrics for result in step_results],
            stem="all_metrics",
        )
    return AblationRunResult(
        ablation_id=ablation_id,
        output_dir=output_dir,
        step_results=step_results,
        metrics_rows=metrics_rows,
        delta_rows=delta_rows,
    )


def make_pipeline_config(
    *,
    name: str,
    query: str,
    retrieval: str,
    reranking: str = "none",
    compression: str = "none",
    generation: str = "context_grounded",
    verification: str = "none",
) -> PipelineConfig:
    data: JsonObject = {
        "name": name,
        "stages": {
            "query_formulation": query,
            "retrieval": retrieval,
            "reranking": reranking,
            "compression": compression,
            "generation": generation,
            "verification": verification,
        },
        "generation": {"temperature": 0.0, "max_output_tokens": 384},
        "query_formulation": {"max_queries": 4},
    }
    if retrieval == "dense":
        data["retrieval"] = {"model_ref": "default_retrieval", "top_k": 10}
    elif retrieval == "hybrid":
        data["retrieval"] = {
            "model_ref": "default_retrieval",
            "bm25_top_k": 100,
            "dense_top_k": 100,
            "final_top_k": 100,
            "fusion": "rrf",
        }
    if reranking == "cross_encoder":
        data["reranking"] = {"model_ref": "default_retrieval", "max_candidates": 100, "top_k": 12}
    if compression == "sentence_top_n":
        data["compression"] = {
            "max_sentences": 24,
            "max_tokens": 1800,
            "min_chunks": 2,
            "per_chunk_limit": 4,
            "mmr_lambda": 0.8,
            "semantic_scoring": True,
        }
    if generation == "citation_aware":
        data["evidence_judge"] = {"type": "nli", "model_ref": "default_retrieval"}
    if verification == "factored_cove":
        data["verification"] = {
            "questions_per_answer": 8,
            "retrieval_top_k": 5,
            "revise_answer": True,
            "revision_threshold": 0.8,
            "revise_short_answer": False,
        }
    return PipelineConfig.model_validate(data)


def compute_delta_rows(metrics_rows: list[JsonObject]) -> list[JsonObject]:
    rows = []
    for previous, current in zip(metrics_rows, metrics_rows[1:]):
        row: JsonObject = {
            "from_step": previous["step_name"],
            "to_step": current["step_name"],
        }
        numeric_keys = sorted(
            key
            for key, value in current.items()
            if isinstance(value, (int, float)) and isinstance(previous.get(key), (int, float))
        )
        for key in numeric_keys:
            if key == "step_index":
                continue
            row[f"delta_{key}"] = float(current[key]) - float(previous[key])
        rows.append(row)
    return rows


def compute_leave_one_out_deltas(metrics_rows):
    if not metrics_rows or metrics_rows[0]["step_name"] != "01_full":
        raise ValueError("full configuration must be first")
    full = metrics_rows[0]
    return [{"from_step": row["step_name"], "to_step": full["step_name"],
             "interpretation": "full_minus_removed_conditional_contribution",
             **{f"delta_{k}": float(v) - float(row[k]) for k, v in full.items()
                if k != "step_index" and type(v) in (int, float) and type(row.get(k)) in (int, float)}}
            for row in metrics_rows[1:]]








def ablation_output_dir(
    *,
    base_experiment: ExperimentConfig,
    settings: FormalRunSettings,
    ablation_id: str,
) -> Path | None:
    if not settings.save_outputs:
        return None
    root = settings.output_root
    if root is None:
        if base_experiment.paths is None:
            root = Path("outputs")
        else:
            root = base_experiment.paths.outputs_root
    return (root / "ablations" / ablation_id).resolve()


def write_ablation_outputs(
    *,
    output_dir: Path,
    ablation_id: str,
    metrics_rows: list[JsonObject],
    delta_rows: list[JsonObject],
) -> None:
    payload = {
        "ablation_id": ablation_id,
        "steps": metrics_rows,
        "deltas": delta_rows,
    }
    (output_dir / "ablation_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "ablation_metrics.csv", metrics_rows)
    write_csv(output_dir / "ablation_delta.csv", delta_rows)


def write_csv(path: Path, rows: list[JsonObject]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
