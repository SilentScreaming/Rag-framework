"""Typed configuration loading for reproducible benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator

from rag_benchmark.schemas import BenchmarkRecord, JsonObject, TaskTrack
from rag_benchmark.paths import load_local_paths, path_config_file, resolve_project_root


class DatasetPathEntry(BenchmarkRecord):
    version: str
    root: Path

    @field_validator("root")
    @classmethod
    def root_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"dataset root path must be absolute: {value}")
        return value


class PathConfig(BenchmarkRecord):
    schema_version: int = 1
    datasets_root: Path
    models_root: Path
    outputs_root: Path
    datasets: dict[str, DatasetPathEntry] = Field(default_factory=dict)

    @field_validator("datasets_root", "models_root", "outputs_root")
    @classmethod
    def root_paths_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"path config values must be absolute: {value}")
        return value


class DatasetSourceConfig(BenchmarkRecord):
    provider: str
    version: str
    root_key: str


class DatasetConfig(BenchmarkRecord):
    name: str
    role: str
    task_track: TaskTrack = "short_answer"
    enabled: bool = True
    split: str
    seed: int = 42
    source: DatasetSourceConfig
    sampling: JsonObject = Field(default_factory=dict)
    processing: JsonObject = Field(default_factory=dict)


class PipelineStagesConfig(BenchmarkRecord):
    query_formulation: str = "none"
    retrieval: str = "none"
    reranking: str = "none"
    compression: str = "none"
    generation: str
    verification: str = "none"


class TaskProfileConfig(BenchmarkRecord):
    track: TaskTrack = "short_answer"
    answer_mode: Literal["short", "long", "list", "claims"] = "short"
    citations_required: bool = False
    abstention_allowed: bool = True
    max_answer_tokens: int = Field(default=64, ge=1)


class ExecutionBudgetConfig(BenchmarkRecord):
    profile: Literal["smoke", "quality_max", "budget_matched"] = "smoke"
    max_context_tokens: int = Field(default=4096, ge=128)
    max_generation_tokens: int = Field(default=256, ge=1)
    max_queries: int = Field(default=1, ge=1)
    max_verification_questions: int = Field(default=5, ge=1)
    strict_components: bool = False
    max_model_calls: int | None = Field(default=None, ge=1)
    max_total_model_tokens: int | None = Field(default=None, ge=1)


class MethodDefinitionConfig(BenchmarkRecord):
    family: Literal["closed_book", "vanilla_rag", "enhanced_rag", "rag_verification"]
    paper_reference: str | None = None
    project_implementation: str
    controlled_parent: str | None = None


class PipelineConfig(BenchmarkRecord):
    name: str
    base_pipeline: str | None = None
    method: MethodDefinitionConfig | None = None
    task_profile: TaskProfileConfig = Field(default_factory=TaskProfileConfig)
    budget: ExecutionBudgetConfig = Field(default_factory=ExecutionBudgetConfig)
    stages: PipelineStagesConfig
    query_formulation: JsonObject = Field(default_factory=dict)
    retrieval: JsonObject = Field(default_factory=dict)
    reranking: JsonObject = Field(default_factory=dict)
    compression: JsonObject = Field(default_factory=dict)
    generation: JsonObject = Field(default_factory=dict)
    verification: JsonObject = Field(default_factory=dict)
    evidence_judge: JsonObject = Field(default_factory=dict)
    logging: JsonObject = Field(default_factory=dict)

    def model_refs(self) -> set[str]:
        refs = set()
        for section in (
            self.query_formulation,
            self.generation,
            self.reranking,
            self.retrieval,
            self.compression,
            self.verification,
            self.evidence_judge,
        ):
            value = section.get("model_ref")
            if value:
                refs.add(str(value))
        return refs


class ExperimentConfig(BenchmarkRecord):
    schema_version: str = "experiment-config-v1"
    dataset: DatasetConfig
    pipeline: PipelineConfig
    paths: PathConfig | None = None
    model_refs: dict[str, JsonObject] = Field(default_factory=dict)
    config_files: dict[str, str] = Field(default_factory=dict)


class ExecutionProfileConfig(BenchmarkRecord):
    name: Literal["smoke", "quality_max", "budget_matched"]
    description: str
    pipeline_overrides: dict[str, JsonObject] = Field(default_factory=dict)


def load_yaml_file(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


class ConfigLoader:
    """Loads committed and local benchmark configuration files."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = resolve_project_root(repo_root)
        self.config_root = self.repo_root / "configs"

    def load_paths(self, *, use_local: bool = True) -> PathConfig:
        return PathConfig.model_validate(load_local_paths(self.repo_root, use_local=use_local))

    def load_dataset(self, name: str) -> DatasetConfig:
        path, data = self._load_named_yaml("datasets", name)
        return DatasetConfig.model_validate(data)

    def load_pipeline(self, name: str) -> PipelineConfig:
        path, data = self._load_named_yaml("pipelines", name)
        return PipelineConfig.model_validate(data)

    def load_model_dict(self, name: str) -> JsonObject:
        path, data = self._load_named_yaml("models", name)
        return data

    def load_execution_profile(self, name: str) -> ExecutionProfileConfig:
        _, data = self._load_named_yaml("profiles", name)
        return ExecutionProfileConfig.model_validate(data)

    def resolve_model_refs(self, refs: set[str]) -> dict[str, JsonObject]:
        return self._resolve_available_model_refs(refs)

    def build_experiment_config(
        self,
        *,
        dataset_name: str,
        pipeline_name: str,
        use_local_paths: bool = True,
    ) -> ExperimentConfig:
        dataset_path, dataset_data = self._load_named_yaml("datasets", dataset_name)
        pipeline_path, pipeline_data = self._load_named_yaml("pipelines", pipeline_name)
        dataset = DatasetConfig.model_validate(dataset_data)
        pipeline = PipelineConfig.model_validate(pipeline_data)
        paths = self.load_paths(use_local=use_local_paths)
        model_refs = self._resolve_available_model_refs(pipeline.model_refs())
        return ExperimentConfig(
            dataset=dataset,
            pipeline=pipeline,
            paths=paths,
            model_refs=model_refs,
            config_files={
                "dataset": str(dataset_path.relative_to(self.repo_root)),
                "pipeline": str(pipeline_path.relative_to(self.repo_root)),
                "paths": path_config_file(self.repo_root, use_local=use_local_paths).relative_to(self.repo_root).as_posix(),
            },
        )

    def _resolve_available_model_refs(self, refs: set[str]) -> dict[str, JsonObject]:
        resolved = {}
        for ref in sorted(refs):
            try:
                resolved[ref] = self.load_model_dict(ref)
            except FileNotFoundError:
                continue
        return resolved

    def _load_named_yaml(self, group: str, name: str) -> tuple[Path, JsonObject]:
        directory = self.config_root / group
        if not directory.exists():
            raise FileNotFoundError(directory)
        matches: list[tuple[Path, JsonObject]] = []
        for path in sorted(directory.glob("*.yaml")):
            data = load_yaml_file(path)
            if _yaml_name_matches(path, data, name):
                matches.append((path, data))
        if not matches:
            raise FileNotFoundError(f"No {group} config named {name!r} under {directory}")
        if len(matches) > 1:
            paths = ", ".join(str(path) for path, _ in matches)
            raise ValueError(f"Multiple {group} configs named {name!r}: {paths}")
        return matches[0]


def _yaml_name_matches(path: Path, data: JsonObject, name: str) -> bool:
    configured_name = data.get("name")
    if configured_name == name:
        return True
    stem = path.name.removesuffix(".yaml").removesuffix(".example")
    return stem == name
