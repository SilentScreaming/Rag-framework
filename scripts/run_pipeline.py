"""Run a config-driven benchmark pipeline."""

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
    parse_context_metadata_filters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="alce_asqa")
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=[
            "closed_book",
            "vanilla_rag",
            "best_practice_rag",
            "rag_verification",
            "oracle_context_rag",
        ],
    )
    parser.add_argument("--split-profile", default="mvp")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-relevance-threshold", type=int, default=3)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--use-example-paths", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--generator", default="extractive", choices=["mock", "extractive", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-rewriter", default="auto", choices=["auto", "identity", "rule", "openai"])
    parser.add_argument(
        "--execution-profile",
        default="config",
        choices=["config", "smoke", "quality_max", "budget_matched"],
    )
    parser.add_argument("--strict-components", action="store_true")
    parser.add_argument("--use-real-dense", action="store_true")
    parser.add_argument("--use-real-reranker", action="store_true")
    parser.add_argument("--dense-index-path", default=None)
    parser.add_argument("--candidate-cache-path", default=None)
    parser.add_argument("--build-index-if-missing", action="store_true")
    parser.add_argument("--stage-cache-root", default=None)
    parser.add_argument("--replay-stage", action="append", default=[])
    parser.add_argument("--write-stage-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--context-condition", default=None)
    parser.add_argument(
        "--context-metadata",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument("--input-cost-per-1m-tokens", type=float, default=None)
    parser.add_argument("--output-cost-per-1m-tokens", type=float, default=None)
    args = parser.parse_args()
    if (args.input_cost_per_1m_tokens is None) != (args.output_cost_per_1m_tokens is None):
        parser.error("--input-cost-per-1m-tokens and --output-cost-per-1m-tokens must be provided together")
    return args


def main() -> int:
    args = parse_args()
    experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name=args.pipeline,
        use_local_paths=not args.use_example_paths,
        execution_profile_name=(None if args.execution_profile == "config" else args.execution_profile),
    )
    dataset_slice = load_formal_dataset_slice(
        experiment=experiment,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        relevance_threshold=args.gold_relevance_threshold,
        candidate_cache_path=(Path(args.candidate_cache_path).resolve() if args.candidate_cache_path else None),
    )
    output_root = Path(args.output_root).resolve() if args.output_root else None
    settings = FormalRunSettings(
        run_id=args.run_id,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        gold_relevance_threshold=args.gold_relevance_threshold,
        output_root=output_root,
        save_outputs=not args.no_save,
        stage_cache_root=Path(args.stage_cache_root).resolve() if args.stage_cache_root else None,
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
    runner = ConfigDrivenPipelineRunner(
        repo_root=REPO_ROOT,
        experiment=experiment,
        dataset_slice=dataset_slice,
        settings=settings,
    )
    result = runner.run()
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "output_dir": str(result.output_dir) if result.output_dir else None,
                "metrics": result.metrics,
                "cost_latency": result.cost_latency,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
