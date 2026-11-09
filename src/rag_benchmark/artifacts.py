"""Run persistence, completion manifests, and artifact validation."""
from __future__ import annotations
import csv
import json
import platform
import os
import math
from hashlib import sha256
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any
import yaml
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.config import ExperimentConfig
from rag_benchmark.data import write_jsonl
from rag_benchmark.execution_types import FormalRunResult, FormalRunSettings
from rag_benchmark.schemas import PredictionRecord, RetrievalTraceRecord, StageTraceRecord
from rag_benchmark.reporting import write_unified_metrics_report
JsonObject = dict[str, Any]

FORMAL_ARTIFACTS = ("config.snapshot.yaml", "metrics.json", "cost_latency.json", "predictions.jsonl",
                    "logs.jsonl", "retrieval_traces.jsonl", "stage_traces.jsonl")


def validate_result_consistency(predictions, metrics, snapshot):
    """Validate core deterministic aggregates, not just byte integrity."""
    if not isinstance(metrics, dict) or not isinstance(snapshot, dict) or not isinstance(snapshot.get("run_id"), str) or snapshot.get("run_id") != metrics.get("run_id"):
        raise ValueError("snapshot/run mismatch")
    if not isinstance(metrics.get("pipelines"), dict) or set(metrics["pipelines"]) != {p.pipeline for p in predictions}:
        raise ValueError("pipeline population mismatch")
    if metrics.get("num_examples", len(predictions)) != len(predictions):
        raise ValueError("prediction count mismatch")
    experiment = ExperimentConfig.model_validate(snapshot.get("experiment"))
    settings = snapshot.get("settings")
    if not isinstance(settings, dict) or not isinstance(settings.get("seed"), int):
        raise ValueError("invalid settings snapshot")
    from rag_benchmark.evaluation.identity import prediction_identity
    for prediction in predictions:
        identity = prediction_identity(prediction)
        if (prediction.run_id != snapshot["run_id"] or prediction.pipeline != experiment.pipeline.name
                or identity.dataset != experiment.dataset.name
                or identity.data_version != experiment.dataset.source.version or identity.seed != settings["seed"]
                or prediction.task_track != experiment.pipeline.task_profile.track):
            raise ValueError("prediction/snapshot identity mismatch")
    for pipeline, values in metrics.get("pipelines", {}).items():
        if not isinstance(values, dict):
            raise ValueError("invalid pipeline metrics")
        rows = [p for p in predictions if p.pipeline == pipeline]
        for key in ("em", "f1", "verification_label_accuracy"):
            if key in values:
                if not rows or not all(isinstance(p.metrics.get(key), (int, float)) for p in rows):
                    raise ValueError(f"aggregate requires complete per-row scores: {key}")
                mean = sum(p.metrics[key] for p in rows) / len(rows)
                if not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or not math.isfinite(mean) or abs(values[key] - mean) > 1e-9:
                    raise ValueError(f"aggregate/prediction mismatch: {key}")


def validate_auxiliary_outputs(output_dir, predictions):
    cost = json.loads((output_dir / "cost_latency.json").read_text(encoding="utf-8"))
    if not isinstance(cost, dict):
        raise ValueError("invalid cost summary")
    for key, value in summarize_cost_latency(predictions).items():
        actual = cost.get(key)
        if isinstance(value, str):
            if actual != value:
                raise ValueError(f"cost/prediction mismatch: {key}")
            continue
        if key not in cost or (value is None and actual is not None) or (value is not None and
            (not isinstance(actual, (int, float)) or not math.isfinite(actual)
             or not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9))):
            raise ValueError(f"cost/prediction mismatch: {key}")
    def read_rows(name):
        return [json.loads(line) for line in (output_dir / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = [RetrievalTraceRecord.model_validate(r) for r in read_rows("retrieval_traces.jsonl")]
    stages = [StageTraceRecord.model_validate(r) for r in read_rows("stage_traces.jsonl")]
    ids = {p.example_id for p in predictions}
    if any(t.example_id not in ids for t in [*traces, *stages]):
        raise ValueError("unknown trace example")
    logs = read_rows("logs.jsonl")
    if any(not isinstance(row, dict) or (row.get("example_id") is not None and row["example_id"] not in ids) for row in logs):
        raise ValueError("invalid run log")
    return cost, traces, stages, logs


def seal_formal_outputs(output_dir, manifest):
    """Finalize the manifest after the new run's artifacts have been written."""
    predictions = [PredictionRecord.model_validate(json.loads(line)) for line in
                   (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    snapshot = yaml.safe_load((output_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    validate_result_consistency(predictions, metrics, snapshot)
    for name in FORMAL_ARTIFACTS:
        if not (output_dir / name).is_file():
            raise ValueError(f"missing formal artifact: {name}")
    validate_auxiliary_outputs(output_dir, predictions)
    completed = {**manifest, "artifact_schema": "formal-artifacts-v1", "status": "completed",
        "outputs": {p.name: sha256(p.read_bytes()).hexdigest() for p in output_dir.iterdir()
                    if p.is_file() and p.name not in {"run_manifest.json", "run_manifest.json.tmp"}}}
    temporary = output_dir / "run_manifest.json.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(completed, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir / "run_manifest.json")
    manifest.update(completed)


def load_completed_formal_run(
    run_dir: Path,
    *,
    run_id: str,
    pipeline: str,
    expected_example_ids: set[str],
    expected_identity: JsonObject | None = None,
) -> FormalRunResult | None:
    """Recover only fully written runs whose examples and metrics match the caller."""

    if expected_identity is None:
        return None  # Without an expected identity, existing artifacts are never resumed automatically.
    required = (*FORMAL_ARTIFACTS, "run_manifest.json")
    if not run_dir.exists() or any(not (run_dir / name).is_file() for name in required):
        return None
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        cost_latency = json.loads((run_dir / "cost_latency.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("artifact_schema") != "formal-artifacts-v1" or manifest.get("status") != "completed":
            return None
        hashes = manifest.get("outputs", {})
        if not isinstance(hashes, dict) or not set(FORMAL_ARTIFACTS) <= hashes.keys():
            return None
        for name, expected_hash in hashes.items():
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                return None
            if sha256((run_dir / name).read_bytes()).hexdigest() != expected_hash:
                return None
        predictions = [
            PredictionRecord.model_validate(json.loads(line))
            for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        from rag_benchmark.paired_artifacts import validate_prefix_anchors
        for prediction in predictions:
            validate_prefix_anchors(prediction)
        snapshot = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
        validate_result_consistency(predictions, metrics, snapshot)
        cost_latency, traces, stages, logs = validate_auxiliary_outputs(run_dir, predictions)
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return None
    prediction_ids = {prediction.example_id for prediction in predictions}
    if (
        manifest.get("run_identity") != expected_identity
        or metrics.get("run_id") != run_id
        or pipeline not in metrics.get("pipelines", {})
        or len(predictions) != len(expected_example_ids)
        or prediction_ids != expected_example_ids
        or any(p.pipeline != pipeline or p.run_id != run_id for p in predictions)
    ):
        return None
    return FormalRunResult(
        run_id=run_id,
        output_dir=run_dir,
        predictions=predictions,
        retrieval_traces=traces,
        metrics=metrics,
        logs=logs,
        cost_latency=cost_latency,
        stage_traces=stages,
        manifest=manifest,
    )


def write_formal_outputs(
    *,
    output_dir: Path,
    repo_root: Path,
    experiment: ExperimentConfig,
    settings: FormalRunSettings,
    run_id: str,
    predictions: list[PredictionRecord],
    retrieval_traces: list[RetrievalTraceRecord],
    metrics: JsonObject,
    logs: list[JsonObject],
    cost_latency: JsonObject,
    stage_traces: list[StageTraceRecord],
    manifest: JsonObject,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": experiment.to_json_dict(),
        "settings": {
            "split_profile": settings.split_profile,
            "limit": settings.limit,
            "seed": settings.seed,
            "gold_relevance_threshold": settings.gold_relevance_threshold,
            "component_settings": component_settings_snapshot(settings.component_settings),
            "retrieval_metric_ks": list(settings.retrieval_metric_ks),
            "bootstrap_samples": settings.bootstrap_samples,
            "paired_artifact_root": str(settings.paired_artifact_root) if settings.paired_artifact_root else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(repo_root),
            "git_status_short": git_status_short(repo_root),
        },
    }
    (output_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cost_latency.json").write_text(
        json.dumps(cost_latency, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions.jsonl", (prediction.to_json_dict() for prediction in predictions))
    write_jsonl(output_dir / "retrieval_traces.jsonl", (trace.to_json_dict() for trace in retrieval_traces))
    write_jsonl(output_dir / "stage_traces.jsonl", (trace.to_json_dict() for trace in stage_traces))
    write_jsonl(output_dir / "logs.jsonl", logs)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    write_unified_metrics_report(output_dir, metrics)
    seal_formal_outputs(output_dir, manifest)


def write_metrics_csv(path: Path, metrics: JsonObject) -> None:
    rows = []
    for pipeline, values in metrics.get("pipelines", {}).items():
        row = {"pipeline": pipeline}
        row.update(values)
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def component_settings_snapshot(settings: ComponentFactorySettings) -> JsonObject:
    data = dict(settings.__dict__)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def summarize_cost_latency(predictions: list[PredictionRecord]) -> JsonObject:
    if not predictions:
        return {
            "num_predictions": 0,
            "total_estimated_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_total_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_estimated_cost": 0.0,
        }
    latencies = sorted(prediction.latency.total_ms for prediction in predictions)
    available_costs = [
        prediction.cost.estimated_cost
        for prediction in predictions
        if prediction.cost.estimated_cost is not None
    ]
    unknown_usage = any(p.generator.get("unknown_attempt_usage") or
        any(stage.get("unknown_attempt_usage") for stage in p.metadata.get("usage_by_stage", {}).values()
            if isinstance(stage, dict)) for p in predictions)
    complete_cost = len(available_costs) == len(predictions) and not unknown_usage
    return {
        "num_predictions": len(predictions),
        "total_estimated_cost": sum(available_costs) if complete_cost else None,
        "known_estimated_cost": sum(available_costs) if available_costs else None,
        "cost_status": "complete_estimate" if complete_cost else "partial_unknown_attempt_usage" if unknown_usage else "missing_prices_or_usage",
        "total_input_tokens": sum(prediction.tokens.input_tokens for prediction in predictions),
        "total_output_tokens": sum(prediction.tokens.output_tokens for prediction in predictions),
        "avg_total_ms": sum(latencies) / len(predictions),
        "latency_p50_ms": median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "avg_input_tokens": sum(prediction.tokens.input_tokens for prediction in predictions) / len(predictions),
        "avg_output_tokens": sum(prediction.tokens.output_tokens for prediction in predictions) / len(predictions),
        "avg_estimated_cost": (
            sum(available_costs) / len(available_costs) if complete_cost else None
        ),
        "estimated_cost_available_rate": len(available_costs) / len(predictions),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def make_run_id(*, dataset: str, split: str, pipeline: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{dataset}-{split}-{pipeline}"


def git_commit(repo_root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{commit}+dirty" if status else commit
    except Exception:
        return "no commit"


def git_status_short(repo_root: Path) -> list[str]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line for line in status.splitlines() if line.strip()]
    except Exception:
        return ["git status unavailable"]
