"""Run a resumable evidence-grounded hallucination detector on RAGTruth QA."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_benchmark.component_factory import OpenAITextPromptGenerator
from rag_benchmark.config import ConfigLoader
from rag_benchmark.evaluation.hallucination import (
    apply_hallucination_threshold,
    calibrate_hallucination_threshold,
    hallucination_benchmark_metrics,
    parse_batched_hallucination_output,
)
from rag_benchmark.evaluation.llm_judge import extract_json_object
from rag_benchmark.generation.backends import load_openai_local_config, resolve_openai_model
from rag_benchmark.stage_cache import canonical_hash
from rag_benchmark.file_utils import file_sha256
from rag_benchmark.run_identity import source_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-profile", choices=["judge_calibration", "judge_eval"], required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", default=None, help="Model name; defaults to local API configuration")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prompt-profile", choices=["balanced", "strict_attribution"], default="balanced")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-cap", type=int, default=512)
    parser.add_argument("--input-cost-per-1m-tokens",type=float)
    parser.add_argument("--output-cost-per-1m-tokens",type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = resolve_openai_model(REPO_ROOT, args.model)
    if args.limit < 1 or args.batch_size < 1:
        raise ValueError("limit and batch-size must be >= 1")
    paths = ConfigLoader(REPO_ROOT).load_paths(use_local=True)
    root = paths.datasets["ragtruth_qa"].root
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "detections.jsonl"

    split_path = root / "splits" / f"{args.split_profile}.jsonl"
    split_rows = read_jsonl(split_path)
    if args.sample_manifest:
        from rag_benchmark.sampling import load_sample_manifest
        selected = load_sample_manifest(args.sample_manifest, dataset="ragtruth_qa", split_path=split_path, seed=42,
                    data_version=ConfigLoader(REPO_ROOT).load_dataset("ragtruth_qa").source.version)
        split_rows = [r for r in split_rows if str(r["response_id"]) in set(selected["sample_ids"])]
    else:
        split_rows = split_rows[:args.limit]
    selected_ids = {str(row["response_id"]) for row in split_rows}
    identity = {"schema_version": "ragtruth-run-identity-v1", "split": args.split_profile,
                "response_ids": sorted(selected_ids), "model": args.model, "prompt_profile": args.prompt_profile,
                "output_cap": args.output_cap, "batch_size": args.batch_size, "threshold": args.threshold,
                "prices":{"input":args.input_cost_per_1m_tokens,"output":args.output_cost_per_1m_tokens},
                "source_code": source_fingerprint(REPO_ROOT),
                "data": {name: file_sha256(root / name) for name in
                         ("examples.jsonl", "candidate_responses.jsonl", "corpus.jsonl", "context_sets.jsonl", "annotations.jsonl")}}
    identity["identity_hash"] = canonical_hash(identity)
    manifest_path = output_dir / "run_identity.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("RAGTruth resume identity changed; use a new output directory")
    elif traces_path.exists():
        raise ValueError("Existing RAGTruth traces have no identity manifest; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    responses = {
        str(row["response_id"]): row
        for row in read_jsonl(root / "candidate_responses.jsonl")
        if str(row.get("response_id")) in selected_ids
    }
    examples = {str(row["example_id"]): row for row in read_jsonl(root / "examples.jsonl")}
    corpus = {str(row["chunk_id"]): row for row in read_jsonl(root / "corpus.jsonl")}
    contexts = {str(row["example_id"]): row for row in read_jsonl(root / "context_sets.jsonl")}
    annotations = read_jsonl(root / "annotations.jsonl")
    gold_response = {
        str(row["target_id"]): str(row["label"]) == "hallucinated"
        for row in annotations
        if row.get("target_type") == "response" and str(row.get("target_id")) in selected_ids
    }
    gold_spans: dict[str, list[tuple[int, int]]] = {response_id: [] for response_id in selected_ids}
    for row in annotations:
        response_id = str(row.get("target_id"))
        if response_id in selected_ids and row.get("target_type") == "span":
            if row.get("start_char") is not None and row.get("end_char") is not None:
                gold_spans[response_id].append((int(row["start_char"]), int(row["end_char"])))

    completed_rows = read_jsonl(traces_path) if traces_path.exists() else []
    completed = {str(row["response_id"]) for row in completed_rows}
    pending = [row for row in split_rows if str(row["response_id"]) not in completed]
    generator = OpenAITextPromptGenerator(repo_root=REPO_ROOT, model=args.model, max_output_tokens=args.output_cap,
        max_attempts=1,timeout_seconds=240)
    local_config = load_openai_local_config(REPO_ROOT)
    input_price = args.input_cost_per_1m_tokens if args.input_cost_per_1m_tokens is not None else local_config.get("input_cost_per_1m_tokens")
    output_price = args.output_cost_per_1m_tokens if args.output_cost_per_1m_tokens is not None else local_config.get("output_cost_per_1m_tokens")
    started = time.perf_counter()
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        items = []
        response_texts = {}
        for member in batch:
            response_id = str(member["response_id"])
            response = responses[response_id]
            example_id = str(response["example_id"])
            context = contexts[example_id]
            evidence = [str(corpus[chunk_id]["text"]) for chunk_id in context["chunk_ids"]]
            response_text = str(response["text"])
            response_texts[response_id] = response_text
            items.append(
                {
                    "response_id": response_id,
                    "question": str(examples[example_id]["question"]),
                    "evidence": evidence,
                    "response": response_text,
                }
            )
        prompt = build_prompt(items, profile=args.prompt_profile)
        call_started = time.perf_counter()
        from rag_benchmark.generation.budget import execution_budget
        with execution_budget() as budget:
            budget.stage = "ragtruth_detection"
            try:
                response_result = generator.generate_text_result(prompt)
            except Exception as exc:
                from rag_benchmark.generation.usage import failure_usage
                (output_dir / "failed_batch.json").write_text(json.dumps({"response_ids": list(response_texts),
                    "failure_kind": "api_error", "error_type": type(exc).__name__, "usage": failure_usage(exc),
                    "budget": budget.snapshot()}, indent=2), encoding="utf-8")
                raise
        raw = response_result.text
        if response_result.finish_reason in {"length", "content_filter"}:
            (output_dir / "failed_batch.json").write_text(json.dumps({"response_ids": list(response_texts),
                "finish_reason": response_result.finish_reason, "usage": response_result.usage_record(),
                "budget": budget.snapshot()}, indent=2), encoding="utf-8")
            raise ValueError("RAGTruth judge output was truncated/filtered; not a semantic detection")
        latency_ms = (time.perf_counter() - call_started) * 1000
        with (output_dir / "batch_usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"response_ids": list(response_texts), "usage": response_result.usage_record(),
                                     "budget": budget.snapshot(), "raw_response": raw}) + "\n")
        try:
            payload = extract_json_object(raw)
            detections = parse_batched_hallucination_output(payload, responses=response_texts, threshold=args.threshold)
        except Exception as exc:
            (output_dir / "failed_batch.json").write_text(json.dumps({"response_ids": list(response_texts),
                "failure_kind": "format", "error_type": type(exc).__name__, "usage": response_result.usage_record()}), encoding="utf-8")
            raise
        input_tokens = response_result.input_tokens
        output_tokens = response_result.output_tokens
        per_item_cost = (
            input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000
        ) / len(batch) if input_price is not None and output_price is not None else None
        with traces_path.open("a", encoding="utf-8") as handle:
            for detection in detections:
                response = responses[detection.response_id]
                handle.write(
                    json.dumps(
                        {
                            "schema_version": "ragtruth-detection-v1",
                            "response_id": detection.response_id,
                            "example_id": response["example_id"],
                            "source_model": response.get("model"),
                            "gold_has_hallucination": gold_response[detection.response_id],
                            "predicted_has_hallucination": detection.has_hallucination,
                            "hallucination_probability": detection.hallucination_probability,
                            "gold_spans": gold_spans[detection.response_id],
                            "predicted_spans": detection.spans,
                            "predicted_span_quotes": [
                                response_texts[detection.response_id][start:end]
                                for start, end in detection.spans
                            ],
                            "invalid_span_quotes": detection.invalid_span_quotes,
                            "rationale": detection.rationale,
                            "model": args.model,
                            "prompt_profile": args.prompt_profile,
                            "latency_ms_share": latency_ms / len(batch),
                            "input_tokens_share": input_tokens / len(batch),
                            "output_tokens_share": output_tokens / len(batch),
                            "estimated_cost": per_item_cost,
                            "configured_prices_per_million": {"input": input_price, "output": output_price},
                            "usage_source": response_result.usage_source,
                            "unknown_attempt_usage": response_result.unknown_attempt_usage,
                            "output_cap": args.output_cap,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        completed_count = min(len(pending), batch_start + len(batch))
        print(
            json.dumps(
                {
                    "event": "ragtruth_progress",
                    "new_completed": completed_count,
                    "pending_total": len(pending),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }
            ),
            flush=True,
        )

    rows = read_jsonl(traces_path)
    selected_order = [str(row["response_id"]) for row in split_rows]
    rows_by_id = {str(row["response_id"]): row for row in rows}
    rows = [rows_by_id[response_id] for response_id in selected_order if response_id in rows_by_id]
    default_metrics = hallucination_benchmark_metrics(
        apply_hallucination_threshold(rows, threshold=args.threshold)
    )
    calibration = None
    if args.split_profile == "judge_calibration" and not args.sample_manifest:
        calibration = calibrate_hallucination_threshold(rows, minimum_per_class_recall=0.65)
        metrics = dict(calibration["metrics"])
        decision_threshold = float(calibration["threshold"])
    else:
        metrics = dict(default_metrics)
        decision_threshold = args.threshold
    metrics.update(
        {
            "schema_version": "ragtruth-benchmark-metrics-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": "ragtruth_qa",
            "split_profile": args.split_profile,
            "model": args.model,
            "prompt_profile": args.prompt_profile,
            "decision_threshold": decision_threshold,
            "default_threshold": args.threshold,
            "default_threshold_metrics": default_metrics,
            "calibration": calibration,
            "total_input_tokens": sum(float(row.get("input_tokens_share", 0.0)) for row in rows),
            "total_output_tokens": sum(float(row.get("output_tokens_share", 0.0)) for row in rows),
            "known_estimated_cost": sum(float(row["estimated_cost"]) for row in rows if row.get("estimated_cost") is not None),
            "total_estimated_cost": (sum(float(row["estimated_cost"]) for row in rows)
                 if all(row.get("estimated_cost") is not None and not row.get("unknown_attempt_usage") for row in rows) else None),
            "unknown_usage_count": sum(bool(row.get("unknown_attempt_usage")) for row in rows),
            "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
            "purpose": selected.get("purpose", "sampled_judge_evaluation") if args.sample_manifest else "judge_evaluation",
            "avg_latency_ms": (
                sum(float(row.get("latency_ms_share", 0.0)) for row in rows) / len(rows) if rows else 0.0
            ),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "metrics": metrics}, indent=2))
    return 0


def build_prompt(items: list[dict[str, Any]], *, profile: str) -> str:
    if profile == "strict_attribution":
        policy = (
            "Apply strict RAGTruth-style attribution. Inspect every atomic assertion against the supplied "
            "evidence and ignore outside knowledge or plausibility. Mark hallucination when any response "
            "word or phrase adds information not explicitly supported, including unsupported adjectives, "
            "causal explanations, thematic interpretations, procedural steps, quantities, or broadened or "
            "narrowed ranges. Evidence-faithful paraphrases and purely connective wording are allowed. "
        )
    else:
        policy = (
            "Mark hallucination when any material factual statement is unsupported or contradicted. "
            "Evidence-supported paraphrases, harmless synthesis, and direct logical inferences are not "
            "hallucinations. "
        )
    return (
        "Evaluate each response only against its supplied evidence. "
        + policy
        + "For every hallucinated passage, return "
        "the shortest exact contiguous quote copied from the response; do not paraphrase quotes. Return "
        "a calibrated hallucination_probability from 0 to 1 for the event that at least one material "
        "hallucination exists. Return only valid JSON as {\"items\":[{\"response_id\":\"...\","
        "\"has_hallucination\":true,\"hallucination_probability\":0.9,"
        "\"spans\":[{\"quote\":\"exact response text\"}],\"rationale\":\"brief\"}]} .\n\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n\nJSON:"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
