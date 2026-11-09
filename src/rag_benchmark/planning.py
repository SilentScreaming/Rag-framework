"""Resolve configuration and task contracts before execution."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from rag_benchmark.config import ConfigLoader, ExperimentConfig, PipelineConfig
JsonObject = dict[str, Any]


def resolve_pipeline_with_base(loader: ConfigLoader, pipeline_name: str) -> PipelineConfig:
    pipeline = loader.load_pipeline(pipeline_name)
    if not pipeline.base_pipeline:
        return pipeline
    base = resolve_pipeline_with_base(loader, pipeline.base_pipeline)
    merged = base.to_json_dict()
    child = pipeline.to_json_dict()
    for key, value in child.items():
        if key in {
            "query_formulation",
            "retrieval",
            "reranking",
            "compression",
            "generation",
            "verification",
            "evidence_judge",
            "logging",
        }:
            merged[key] = {**merged.get(key, {}), **value}
        elif value is not None:
            merged[key] = value
    return PipelineConfig.model_validate(merged)


def build_experiment_with_resolved_pipeline(
    *,
    repo_root: Path,
    dataset_name: str,
    pipeline_name: str,
    use_local_paths: bool,
    execution_profile_name: str | None = None,
) -> ExperimentConfig:
    loader = ConfigLoader(repo_root)
    experiment = loader.build_experiment_config(
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
        use_local_paths=use_local_paths,
    )
    pipeline = resolve_pipeline_with_base(loader, pipeline_name)
    config_files = dict(experiment.config_files)
    if execution_profile_name is not None:
        profile = loader.load_execution_profile(execution_profile_name)
        override = profile.pipeline_overrides.get(pipeline_name, {})
        pipeline = PipelineConfig.model_validate(deep_merge(pipeline.to_json_dict(), override))
        config_files["execution_profile"] = f"configs/profiles/{execution_profile_name}.example.yaml"
    pipeline = adapt_pipeline_to_dataset(pipeline, experiment.dataset)
    model_refs = loader.resolve_model_refs(pipeline.model_refs())
    resolved = experiment.model_copy(
        update={"pipeline": pipeline, "config_files": config_files, "model_refs": model_refs}
    )
    return resolved


def adapt_pipeline_to_dataset(pipeline: PipelineConfig, dataset: Any) -> PipelineConfig:
    processing = dict(dataset.processing)
    generation_key = f"{pipeline.budget.profile}_generation_max_tokens"
    generation_max_tokens = int(pipeline.generation.get("shared_output_cap") or
        processing.get(
            generation_key,
            processing.get("generation_max_tokens", pipeline.budget.max_generation_tokens),
        )
    )
    payload = pipeline.to_json_dict()
    payload["task_profile"] = {
        **payload.get("task_profile", {}),
        "track": dataset.task_track,
        "answer_mode": processing.get("answer_mode", payload.get("task_profile", {}).get("answer_mode", "short")),
        "citations_required": False if pipeline.logging.get("removed_component") == "generation" else bool(
            processing.get(
                "citation_required",
                payload.get("task_profile", {}).get("citations_required", False),
            )
        ),
        "max_answer_tokens": generation_max_tokens,
        "abstention_allowed": bool(
            processing.get(
                "abstention_allowed",
                payload.get("task_profile", {}).get("abstention_allowed", True),
            )
        ),
    }
    payload["budget"] = {
        **payload.get("budget", {}),
        "max_generation_tokens": generation_max_tokens,
    }
    payload["generation"] = {
        **payload.get("generation", {}),
        "max_output_tokens": generation_max_tokens,
    }
    if pipeline.stages.verification not in {"none", "disabled"}:
        payload["verification"] = {
            **payload.get("verification", {}),
            "revise_short_answer": bool(
                processing.get(
                    "revise_short_answer",
                    payload.get("verification", {}).get("revise_short_answer", False),
                )
            ),
            "allow_abstention": bool(
                processing.get(
                    "abstention_allowed",
                    payload.get("verification", {}).get("allow_abstention", False),
                )
            ),
        }
    return PipelineConfig.model_validate(payload)


def deep_merge(base: JsonObject, override: JsonObject) -> JsonObject:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged

