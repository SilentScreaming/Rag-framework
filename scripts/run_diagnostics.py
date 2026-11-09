"""Generate hallucination/error diagnostics from benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, UTC

from rag_benchmark.analysis.diagnostics import diagnose_prediction, summarize_diagnostics
from rag_benchmark.data import write_jsonl
from rag_benchmark.schemas import PredictionRecord, RetrievalTraceRecord, read_jsonl_records
from rag_benchmark.data import read_jsonl
from rag_benchmark.evaluation.service import evaluated_prediction
from rag_benchmark.stage_cache import canonical_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--retrieval-traces", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluations", type=Path, help="Independent evaluation-v1 sidecar JSONL")
    args = parser.parse_args()
    if not args.run_dir and not args.predictions:
        parser.error("Provide --run-dir or --predictions")
    return args


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions).resolve() if args.predictions else Path(args.run_dir).resolve() / "predictions.jsonl"
    traces_path = (
        Path(args.retrieval_traces).resolve()
        if args.retrieval_traces
        else Path(args.run_dir).resolve() / "retrieval_traces.jsonl"
        if args.run_dir
        else None
    )
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else
                  predictions_path.parent / ("diagnostics-v1-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")))
    if output_dir.exists():
        raise FileExistsError("Use a new diagnostics output directory")

    predictions = read_jsonl_records(predictions_path, PredictionRecord)
    if args.evaluations:
        sidecars = {}
        for row in read_jsonl(args.evaluations):
            if row.get("schema_version") != "evaluation-v1" or row["prediction_hash"] in sidecars:
                raise ValueError("unsupported or duplicate evaluation sidecar")
            sidecars[row["prediction_hash"]] = row
        predictions = [evaluated_prediction(p, sidecars[canonical_hash(p.to_json_dict())]) for p in predictions]
    traces_by_example = load_traces_by_example(traces_path) if traces_path and traces_path.exists() else {}
    diagnostics = [
        diagnose_prediction(
            prediction,
            retrieval_trace=traces_by_example.get(prediction.example_id),
        )
        for prediction in predictions
    ]
    summary = summarize_diagnostics(diagnostics)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "diagnostics.jsonl", (record.to_json_dict() for record in diagnostics))
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def load_traces_by_example(path: Path) -> dict[str, RetrievalTraceRecord]:
    return {trace.example_id: trace for trace in read_jsonl_records(path, RetrievalTraceRecord)}


if __name__ == "__main__":
    raise SystemExit(main())
