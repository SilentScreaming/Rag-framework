"""Run the fixed RAG module ablation chain."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_benchmark.ablation import build_profile_ablation_steps, build_leave_one_out_steps, run_ablation_suite
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.runner import (
    FormalRunSettings,
    build_experiment_with_resolved_pipeline,
    load_formal_dataset_slice,
    parse_context_metadata_filters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="alce_asqa")
    parser.add_argument("--split-profile", default="mvp")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-relevance-threshold", type=int, default=3)
    parser.add_argument("--ablation-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--generator", default="extractive", choices=["mock", "extractive", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-rewriter", default="rule", choices=["auto", "identity", "rule", "openai"])
    parser.add_argument(
        "--execution-profile",
        default="smoke",
        choices=["smoke", "quality_max", "budget_matched"],
    )
    parser.add_argument("--strict-components", action="store_true")
    parser.add_argument("--use-real-dense", action="store_true")
    parser.add_argument("--use-real-reranker", action="store_true")
    parser.add_argument("--dense-index-path", default=None)
    parser.add_argument("--candidate-cache-path", default=None)
    parser.add_argument("--build-index-if-missing", action="store_true")
    parser.add_argument("--input-cost-per-1m-tokens", type=float, default=None)
    parser.add_argument("--output-cost-per-1m-tokens", type=float, default=None)
    parser.add_argument("--stage-cache-root", default=None)
    parser.add_argument("--paired-artifact-root", type=Path)
    parser.add_argument("--replay-stage", action="append", default=[])
    parser.add_argument("--write-stage-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--context-condition", default=None)
    parser.add_argument("--design", choices=["leave-one-out", "sequential"], default="leave-one-out")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-cap", type=int)
    parser.add_argument("--garage-reviews", type=Path)
    parser.add_argument("--context-metadata", action="append", default=[], metavar="KEY=JSON_VALUE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name="vanilla_rag",
        use_local_paths=True,
        execution_profile_name=args.execution_profile,
    )
    best_experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name="best_practice_rag",
        use_local_paths=True,
        execution_profile_name=args.execution_profile,
    )
    verification_experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name="rag_verification",
        use_local_paths=True,
        execution_profile_name=args.execution_profile,
    )
    if args.garage_reviews:
        from rag_benchmark.data_semantics import with_semantic_reviews
        experiment = with_semantic_reviews(experiment, args.garage_reviews)
        best_experiment = with_semantic_reviews(best_experiment, args.garage_reviews)
        verification_experiment = with_semantic_reviews(verification_experiment, args.garage_reviews)
    if args.output_cap is not None:
        from rag_benchmark.generation.budget import with_shared_output_cap
        experiment = with_shared_output_cap(experiment, args.output_cap)
        best_experiment = with_shared_output_cap(best_experiment, args.output_cap)
        verification_experiment = with_shared_output_cap(verification_experiment, args.output_cap)
    if args.sample_manifest:
        args.limit = len(json.loads(args.sample_manifest.read_text(encoding="utf-8"))["sample_ids"])
    dataset_slice = load_formal_dataset_slice(
        experiment=experiment,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        relevance_threshold=args.gold_relevance_threshold,
        candidate_cache_path=(Path(args.candidate_cache_path).resolve() if args.candidate_cache_path else None),
        sample_manifest=args.sample_manifest,
    )
    ablation_id = args.ablation_id or make_ablation_id(dataset_slice.dataset_name, args.split_profile)
    settings = FormalRunSettings(
        run_id=None,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        gold_relevance_threshold=args.gold_relevance_threshold,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        save_outputs=not args.no_save,
        stage_cache_root=Path(args.stage_cache_root).resolve() if args.stage_cache_root else None,
        paired_artifact_root=args.paired_artifact_root.resolve() if args.paired_artifact_root else None,
        replay_stages=tuple(args.replay_stage),
        write_stage_cache=args.write_stage_cache,
        bootstrap_samples=args.bootstrap_samples,
        context_condition=args.context_condition,
        context_metadata=parse_context_metadata_filters(args.context_metadata),
        component_settings=ComponentFactorySettings(
            generator_kind=args.generator,
            generator_model=args.model,
            query_rewriter=args.query_rewriter,
            use_real_dense=args.use_real_dense,
            use_real_reranker=args.use_real_reranker,
            dense_index_path=Path(args.dense_index_path).resolve() if args.dense_index_path else None,
            build_index_if_missing=args.build_index_if_missing,
            input_cost_per_1m_tokens=args.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=args.output_cost_per_1m_tokens,
            execution_profile=args.execution_profile,
            strict_components=args.strict_components,
        ),
    )
    result = run_ablation_suite(
        repo_root=REPO_ROOT,
        base_experiment=experiment,
        dataset_slice=dataset_slice,
        settings=settings,
        ablation_id=ablation_id,
        steps=build_leave_one_out_steps(full=best_experiment.pipeline) if args.design == "leave-one-out" else build_profile_ablation_steps(
            vanilla=experiment.pipeline,
            best_practice=best_experiment.pipeline,
            verification=verification_experiment.pipeline,
        ),
    )
    print(
        json.dumps(
            {
                "ablation_id": result.ablation_id,
                "output_dir": str(result.output_dir) if result.output_dir else None,
                "steps": result.metrics_rows,
                "deltas": result.delta_rows,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def make_ablation_id(dataset: str, split_profile: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{dataset}-{split_profile}"


if __name__ == "__main__":
    raise SystemExit(main())
