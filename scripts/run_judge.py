"""Run evidence/citation judge pre-screening over prediction records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_benchmark.component_factory import OpenAITextPromptGenerator
from rag_benchmark.evaluation.evidence import RuleBasedEvidenceJudge
from rag_benchmark.evaluation.llm_judge import LLMEvidenceJudge, judge_prediction
from rag_benchmark.data import write_jsonl
from rag_benchmark.evaluation.basic import aggregate_metrics
from rag_benchmark.reporting import write_unified_metrics_report
from rag_benchmark.schemas import PredictionRecord, read_jsonl_records, write_jsonl_records


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--judge", choices=["rule", "openai"], default="rule")
    parser.add_argument("--model", default=None)
    parser.add_argument("--only-uncertain", action="store_true")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.75)
    args = parser.parse_args()
    if not args.run_dir and not args.predictions:
        parser.error("Provide --run-dir or --predictions")
    return args


def main() -> int:
    args = parse_args()
    predictions_path = resolve_predictions_path(args)
    output_dir = resolve_output_dir(args, predictions_path=predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl_records(predictions_path, PredictionRecord)
    judge = build_judge(args)
    judged_predictions = []
    traces = []
    metric_rows = []
    for prediction in predictions:
        result = judge_prediction(
            prediction,
            judge=judge,
            only_uncertain=args.only_uncertain,
            uncertainty_threshold=args.uncertainty_threshold,
        )
        judged_predictions.append(result.prediction)
        traces.extend(result.traces)
        metric_rows.append(result.prediction.metrics)

    metrics = {
        "input_predictions": len(predictions),
        "judged_predictions": len(judged_predictions),
        "judge": args.judge,
        "only_uncertain": args.only_uncertain,
        "aggregate": aggregate_metrics(metric_rows),
    }
    write_jsonl_records(output_dir / "predictions.judged.jsonl", judged_predictions)
    write_jsonl(output_dir / "judge_traces.jsonl", (trace.to_json_dict() for trace in traces))
    (output_dir / "judge_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = metrics["aggregate"]
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        unified_document = json.loads(metrics_path.read_text(encoding="utf-8"))
        pipeline = judged_predictions[0].pipeline if judged_predictions else "judge"
        unified_document.setdefault("pipelines", {})[pipeline] = aggregate
    else:
        first = judged_predictions[0] if judged_predictions else None
        pipeline = first.pipeline if first else "judge"
        unified_document = {
            "run_id": first.run_id if first else "judge",
            "dataset": first.metadata.get("dataset", "unknown") if first else "unknown",
            "split": first.metadata.get("split", "unknown") if first else "unknown",
            "task_track": first.task_track if first else "verification_hallucination",
            "num_examples": len(judged_predictions),
            "pipelines": {pipeline: aggregate},
            "confidence_intervals": {pipeline: {}},
        }
    write_unified_metrics_report(output_dir, unified_document)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions:
        return Path(args.predictions).resolve()
    return (Path(args.run_dir).resolve() / "predictions.jsonl")


def resolve_output_dir(args: argparse.Namespace, *, predictions_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if args.run_dir:
        return Path(args.run_dir).resolve()
    return predictions_path.parent


def build_judge(args: argparse.Namespace):
    if args.judge == "rule":
        return RuleBasedEvidenceJudge()
    return LLMEvidenceJudge(OpenAITextPromptGenerator(repo_root=REPO_ROOT, model=args.model))


if __name__ == "__main__":
    raise SystemExit(main())
