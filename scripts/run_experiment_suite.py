"""Run the four controlled RAG methods with shared data, models, and paired verification drafts."""

from __future__ import annotations
from rag_benchmark.run_identity import run_identity

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.evaluation.tracks import TRACK_PRIMARY_METRICS, paired_metric_comparisons
from rag_benchmark.runner import (
    ConfigDrivenPipelineRunner,
    FormalRunSettings,
    build_experiment_with_resolved_pipeline,
    filter_dataset_slice_by_context,
    load_formal_dataset_slice,
    load_completed_formal_run,
    parse_context_metadata_filters,
)
from rag_benchmark.reporting import write_unified_metrics_report


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ("closed_book", "vanilla_rag", "best_practice_rag", "rag_verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="alce_asqa")
    parser.add_argument("--split-profile", default="mvp")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-relevance-threshold", type=int, default=3)
    parser.add_argument(
        "--execution-profile",
        default="quality_max",
        choices=["smoke", "quality_max", "budget_matched"],
    )
    parser.add_argument("--generator", default="openai", choices=["mock", "extractive", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-rewriter", default="auto", choices=["auto", "identity", "rule", "openai"])
    parser.add_argument("--dense-index-path", default=None)
    parser.add_argument("--candidate-cache-path", default=None)
    parser.add_argument("--build-index-if-missing", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--suite-id", default=None)
    parser.add_argument("--stage-cache-root", default=None)
    parser.add_argument("--write-stage-cache", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--context-condition", default=None)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-cap", type=int)
    parser.add_argument("--garage-reviews", type=Path)
    parser.add_argument("--context-metadata", action="append", default=[], metavar="KEY=JSON_VALUE")
    parser.add_argument("--input-cost-per-1m-tokens", type=float, default=None)
    parser.add_argument("--output-cost-per-1m-tokens", type=float, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if (args.input_cost_per_1m_tokens is None) != (args.output_cost_per_1m_tokens is None):
        parser.error("input and output token prices must be provided together")
    if args.execution_profile != "smoke" and args.generator != "openai":
        parser.error("formal suites require --generator openai")
    if (
        args.execution_profile != "smoke"
        and args.dataset != "rgb_en"
        and not (args.dense_index_path or args.candidate_cache_path)
    ):
        parser.error("formal suites require --dense-index-path or --candidate-cache-path")
    return args


def main() -> int:
    args = parse_args()
    suite_id = args.suite_id or make_suite_id(args.dataset, args.execution_profile)
    experiments = {
        pipeline: build_experiment_with_resolved_pipeline(
            repo_root=REPO_ROOT,
            dataset_name=args.dataset,
            pipeline_name=pipeline,
            use_local_paths=True,
            execution_profile_name=args.execution_profile,
        )
        for pipeline in PIPELINES
    }
    if getattr(args, "garage_reviews", None):
        from rag_benchmark.data_semantics import with_semantic_reviews
        experiments = {name: with_semantic_reviews(exp, args.garage_reviews) for name,exp in experiments.items()}
    if getattr(args, "output_cap", None) is not None:
        from rag_benchmark.generation.budget import with_shared_output_cap
        experiments = {name: with_shared_output_cap(exp, args.output_cap) for name, exp in experiments.items()}
    if getattr(args, "sample_manifest", None):
        args.limit = len(json.loads(args.sample_manifest.read_text(encoding="utf-8"))["sample_ids"])
    dataset_slice = load_formal_dataset_slice(
        experiment=experiments["best_practice_rag"],
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        relevance_threshold=args.gold_relevance_threshold,
        candidate_cache_path=(Path(args.candidate_cache_path).resolve() if args.candidate_cache_path else None),
        sample_manifest=getattr(args, "sample_manifest", None),
    )
    context_metadata = parse_context_metadata_filters(args.context_metadata)
    dataset_slice = filter_dataset_slice_by_context(
        dataset_slice,
        condition=args.context_condition,
        metadata=context_metadata,
    )
    if not dataset_slice.examples:
        raise ValueError(
            f"No examples match context condition={args.context_condition!r}, "
            f"metadata={context_metadata}"
        )
    output_root = Path(args.output_root).resolve() if args.output_root else None
    component_settings = ComponentFactorySettings(
        generator_kind=args.generator,
        generator_model=args.model,
        query_rewriter=args.query_rewriter,
        dense_index_path=Path(args.dense_index_path).resolve() if args.dense_index_path else None,
        build_index_if_missing=args.build_index_if_missing,
        input_cost_per_1m_tokens=args.input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=args.output_cost_per_1m_tokens,
        execution_profile=args.execution_profile,
        strict_components=args.execution_profile != "smoke",
    )
    results = {}
    best_drafts = None
    for pipeline in PIPELINES:
        run_id = f"{suite_id}-{pipeline}"
        run_root = output_root or experiments[pipeline].paths.outputs_root
        settings = FormalRunSettings(
            run_id=run_id,
            split_profile=args.split_profile,
            limit=args.limit,
            seed=args.seed,
            gold_relevance_threshold=args.gold_relevance_threshold,
            output_root=output_root,
            component_settings=component_settings,
            save_outputs=not args.no_save,
            stage_cache_root=Path(args.stage_cache_root).resolve() if args.stage_cache_root else None,
            write_stage_cache=args.write_stage_cache,
            bootstrap_samples=args.bootstrap_samples,
            context_condition=args.context_condition,
            context_metadata=context_metadata,
            draft_predictions=best_drafts if pipeline == "rag_verification" else None,
        )
        completed_run = None if args.no_save else load_completed_formal_run(
            Path(run_root) / "formal" / run_id,
            run_id=run_id,
            pipeline=pipeline,
            expected_identity=run_identity(repo_root=REPO_ROOT, experiment=experiments[pipeline], dataset_slice=dataset_slice, settings=settings),
            expected_example_ids={example.example_id for example in dataset_slice.examples},
        )
        if completed_run is not None:
            results[pipeline] = completed_run
            if pipeline == "best_practice_rag":
                best_drafts = {
                    prediction.example_id: prediction
                    for prediction in completed_run.predictions
                }
            print(
                json.dumps(
                    {"event": "pipeline_recovered", "pipeline": pipeline, "run_id": run_id}
                ),
                flush=True,
            )
            continue
        if not args.no_save and (Path(run_root) / "formal" / run_id).exists():
            raise ValueError(f"Existing run failed identity or completeness validation: {run_id}. Preserve it and choose a new suite ID/output root.")
        result = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=experiments[pipeline],
            dataset_slice=dataset_slice,
            settings=settings,
        ).run()
        results[pipeline] = result
        if pipeline == "best_practice_rag":
            best_drafts = {prediction.example_id: prediction for prediction in result.predictions}

    summary = build_suite_summary(
        suite_id=suite_id,
        results=results,
        bootstrap_samples=args.bootstrap_samples,
    )
    from rag_benchmark.run_summary import summarize_run
    summary["run_usage"] = {name: summarize_run(result.predictions) for name, result in results.items()}
    summary["sample_manifest"] = str(args.sample_manifest) if getattr(args, "sample_manifest", None) else None
    suite_dir = None
    if not args.no_save:
        root = output_root or experiments["closed_book"].paths.outputs_root
        suite_dir = (Path(root) / "suites" / suite_id).resolve()
        suite_dir.mkdir(parents=True, exist_ok=False)
        (suite_dir / "suite_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_unified_metrics_report(
            suite_dir,
            [result.metrics for result in results.values()],
            comparisons=summary["paired_comparisons"],
        )
    print(json.dumps({"suite_dir": str(suite_dir) if suite_dir else None, **summary}, indent=2, ensure_ascii=False))
    return 0


def build_suite_summary(
    *,
    suite_id: str,
    results: dict[str, object],
    bootstrap_samples: int = 1000,
) -> dict[str, object]:
    aggregates = {
        pipeline: result.metrics["pipelines"][pipeline]
        for pipeline, result in results.items()
    }
    best = {row.example_id: row for row in results["best_practice_rag"].predictions}
    verified = {row.example_id: row for row in results["rag_verification"].predictions}
    paired = []
    for example_id in sorted(best.keys() & verified.keys()):
        before, after = best[example_id], verified[example_id]
        paired.append(
            {
                "example_id": example_id,
                "delta_em": float(after.metrics.get("em", 0.0)) - float(before.metrics.get("em", 0.0)),
                "delta_f1": float(after.metrics.get("f1", 0.0)) - float(before.metrics.get("f1", 0.0)),
                "overcorrection": after.metrics.get("verification_overcorrection"),
                "repair_success": after.metrics.get("verification_repair_success"),
                "draft_reused": float(after.metrics.get("draft_reused", 0.0)),
            }
        )
    paired_summary = {
        key: (sum(float(row[key]) for row in paired) / len(paired)
              if paired and all(row[key] is not None for row in paired) else None)
        for key in ("delta_em", "delta_f1", "overcorrection", "repair_success", "draft_reused")
    }
    track = str(next(iter(results.values())).metrics.get("task_track", "short_answer"))
    comparison_keys = tuple(
        dict.fromkeys(
            [
                *TRACK_PRIMARY_METRICS.get(track, ()),
                "em",
                "f1",
                "claim_support_rate",
                "citation_precision",
                "citation_recall",
            ]
        )
    )
    paired_comparisons = []
    for baseline_name, candidate_name in (
        ("closed_book", "vanilla_rag"),
        ("vanilla_rag", "best_practice_rag"),
        ("best_practice_rag", "rag_verification"),
        ("closed_book", "best_practice_rag"),
        ("closed_book", "rag_verification"),
    ):
        for row in paired_metric_comparisons(
            results[baseline_name].predictions,
            results[candidate_name].predictions,
            keys=comparison_keys,
            samples=bootstrap_samples,
            seed=42,
        ):
            paired_comparisons.append(
                {"baseline": baseline_name, "candidate": candidate_name, **row}
            )
    return {
        "suite_id": suite_id,
        "pipeline_order": list(PIPELINES),
        "run_ids": {pipeline: result.run_id for pipeline, result in results.items()},
        "aggregates": aggregates,
        "paired_best_to_verification": paired_summary,
        "paired_example_count": len(paired),
        "paired_comparisons": paired_comparisons,
    }


def make_suite_id(dataset: str, profile: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{dataset}-{profile}"


if __name__ == "__main__":
    raise SystemExit(main())
