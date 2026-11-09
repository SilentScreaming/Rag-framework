"""Replay only RAG verification from a frozen best-practice prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.runner import (
    ConfigDrivenPipelineRunner,
    FormalRunSettings,
    build_experiment_with_resolved_pipeline,
    load_formal_dataset_slice,
)
from rag_benchmark.schemas import PredictionRecord, read_jsonl_records
from rag_benchmark.verification.replay import validate_drafts, summarize_replay


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-profile", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--draft-predictions", required=True)
    parser.add_argument("--execution-profile", choices=["quality_max", "budget_matched"], default="quality_max")
    parser.add_argument("--model", default=None, help="Model name; defaults to local API configuration")
    parser.add_argument("--input-cost-per-1m-tokens", type=float)
    parser.add_argument("--output-cost-per-1m-tokens", type=float)
    parser.add_argument("--dense-index-path", default=None)
    parser.add_argument("--candidate-cache-path", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--paired-artifact-root", type=Path,
                        help="Shared experiment-local detection artifacts for paired policy replays")
    parser.add_argument("--detection-source-predictions", type=Path,
                        help="Frozen detection arm; its saved detection artifacts must remain intact")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--policy", choices=["detect_only", "constrained_revision", "abstain_only"], default=None)
    parser.add_argument("--max-calls-per-answer", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-cap", type=int)
    parser.add_argument("--context-condition")
    parser.add_argument("--context-metadata", type=json.loads, default={},
                        help='Exact context metadata filters, e.g. {"noise_rate": 0.4}')
    args = parser.parse_args(argv)
    if not isinstance(args.context_metadata, dict):
        parser.error("--context-metadata must be a JSON object")
    return args


def replay_component_settings(args):
    return ComponentFactorySettings(
        generator_kind="openai", generator_model=args.model, query_rewriter="openai",
        input_cost_per_1m_tokens=args.input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=args.output_cost_per_1m_tokens,
        dense_index_path=Path(args.dense_index_path).resolve() if args.dense_index_path else None,
        execution_profile=args.execution_profile, strict_components=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (args.dense_index_path or args.candidate_cache_path):
        raise ValueError("verification replay requires a dense index or candidate cache")
    experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name="rag_verification",
        use_local_paths=True,
        execution_profile_name=args.execution_profile,
    )
    from rag_benchmark.verification.task_revision import TaskRevisionConfig
    if args.output_cap is not None:
        from rag_benchmark.generation.budget import with_shared_output_cap
        experiment = with_shared_output_cap(experiment, args.output_cap)
    if args.sample_manifest:
        args.limit = len(json.loads(args.sample_manifest.read_text(encoding="utf-8"))["sample_ids"])
    options = {**experiment.pipeline.verification.get("task_revision", {}),
               "max_calls": args.max_calls_per_answer}
    if args.policy is not None:
        options["policy"] = args.policy
    TaskRevisionConfig(**{**options, "allow_abstention": experiment.pipeline.task_profile.abstention_allowed})
    experiment = experiment.model_copy(update={"pipeline": experiment.pipeline.model_copy(update={
        "verification": {**experiment.pipeline.verification, "task_revision": options}})})
    dataset_slice = load_formal_dataset_slice(
        experiment=experiment,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        relevance_threshold=3,
        sample_manifest=args.sample_manifest,
        candidate_cache_path=(
            Path(args.candidate_cache_path).resolve() if args.candidate_cache_path else None
        ),
    )
    drafts = read_jsonl_records(Path(args.draft_predictions).resolve(), PredictionRecord)
    if args.detection_source_predictions:
        from rag_benchmark.paired_artifacts import validate_detection_source
        validate_detection_source(read_jsonl_records(args.detection_source_predictions, PredictionRecord),
                                  drafts, selected_ids={e.example_id for e in dataset_slice.examples},
                                  detection_root=(args.paired_artifact_root / "detection") if args.paired_artifact_root else None)
    from rag_benchmark.verification.task_revision import TASKS
    if args.policy is not None and any(e.task_type not in TASKS for e in dataset_slice.examples):
        raise ValueError("explicit task-revision policy is only supported for SciFact/QAMPARI task types")
    from rag_benchmark.runner import select_context_set
    context_ids = {e.example_id: select_context_set(dataset_slice.context_sets_by_example[e.example_id],
                   condition=args.context_condition, metadata=args.context_metadata).context_set_id
                   for e in dataset_slice.examples if dataset_slice.context_sets_by_example.get(e.example_id)}
    drafts_by_id = validate_drafts(drafts, dataset_slice.examples, expected_track=experiment.pipeline.task_profile.track,
        data_version=experiment.dataset.source.version, context_ids=context_ids, seed=args.seed)
    selected_ids = {row.example_id for row in dataset_slice.examples}
    missing = sorted(selected_ids - drafts_by_id.keys())
    if missing:
        raise ValueError(f"draft prediction file is missing {len(missing)} selected examples")
    result = ConfigDrivenPipelineRunner(
        repo_root=REPO_ROOT,
        experiment=experiment,
        dataset_slice=dataset_slice,
        settings=FormalRunSettings(
            run_id=args.run_id,
            split_profile=args.split_profile,
            limit=args.limit,
            seed=args.seed,
            context_condition=args.context_condition,
            context_metadata=args.context_metadata,
            output_root=Path(args.output_root).resolve(),
            paired_artifact_root=args.paired_artifact_root.resolve() if args.paired_artifact_root else None,
            bootstrap_samples=args.bootstrap_samples,
            draft_predictions={example_id: drafts_by_id[example_id] for example_id in selected_ids},
            component_settings=replay_component_settings(args),
        ),
    ).run()
    summary = {
        **summarize_replay(list(drafts_by_id.values()), result.predictions,
                          {e.example_id: e for e in dataset_slice.examples}, samples=args.bootstrap_samples),
        "dataset": args.dataset,
        "run_id": result.run_id,
        "draft_predictions": str(Path(args.draft_predictions).resolve()),
        "policy": options.get("policy", "constrained_revision") if all(e.task_type in TASKS for e in dataset_slice.examples) else "evidence_guarded",
        "metrics": result.metrics,
    }
    summary_path = result.output_dir / "verification_replay_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(result.output_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
