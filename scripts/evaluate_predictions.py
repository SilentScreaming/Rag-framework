"""Create versioned, non-destructive evaluation sidecars for saved predictions.

Default: deterministic task metrics only, no model calls. Independent extraction
requires an explicitly selected backend. Reference pools are fixed per question.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

from rag_benchmark.analysis.diagnostics import diagnose_prediction, summarize_diagnostics
from rag_benchmark.data import load_examples, load_qrels, load_annotations, load_context_sets, write_jsonl, read_jsonl
from rag_benchmark.evaluation.claims import TextClaimExtractor
from rag_benchmark.evaluation.attribution import TextCitationAligner
from rag_benchmark.evaluation.contracts import EVALUATION_VERSION
from rag_benchmark.evaluation.exact_evidence import TextExactJudge, NLIExactJudge
from rag_benchmark.evaluation.settings import (
    EVALUATOR_OUTPUT_CAP, EVALUATOR_INPUT_CAP, EVALUATOR_TIMEOUT_SECONDS,
    REASONING_CHOICES, reasoning_parameter,
)
from rag_benchmark.evaluation.evidence import TransformersNLIEvidenceJudge
from rag_benchmark.evaluation.service import EvaluationService, evaluated_prediction
from rag_benchmark.evaluation.cache import CACHE_PROTOCOL, ExperimentEvaluationCache
from rag_benchmark.evaluation.statistics import confidence_intervals, paired_comparisons, prediction_key, metric_value, CONDITIONAL
from rag_benchmark.generation.text_backend import OpenAITextPromptGenerator
from rag_benchmark.file_utils import file_sha256
from rag_benchmark.reporting import write_unified_metrics_report
from rag_benchmark.run_identity import source_fingerprint
from rag_benchmark.schemas import PredictionRecord, read_jsonl_records
from rag_benchmark.stage_cache import canonical_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


def pairing_view(prediction):
    """Record the metric protocol for failed predictions without scoring fallback text."""
    if prediction.generator.get("schema_failure") and prediction.metadata.get("answer_metric_protocol") is None:
        from rag_benchmark.evaluation.answers import ANSWER_PROTOCOL
        return prediction.model_copy(update={"metadata": {**prediction.metadata,
            "answer_metric_protocol": ANSWER_PROTOCOL}})
    return prediction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-evidence", type=Path)
    parser.add_argument("--cache-from", type=Path, help="Completed evaluation directory with verified cache hashes")
    parser.add_argument("--experiment-cache", type=Path,
                        help="Shared SQLite exact-input cache for new independent evaluations in one experiment")
    parser.add_argument("--extractor-model")
    parser.add_argument("--judge", choices=("none", "local-nli", "openai"), default="none")
    parser.add_argument("--judge-model")
    parser.add_argument("--garage-judge", action="store_true")
    parser.add_argument("--garage-calibration", type=Path)
    parser.add_argument("--garage-reviews", type=Path)
    parser.add_argument("--rgb-reviews", type=Path,
                        help="Optional question/evidence-hash-validated RGB review overlay")
    parser.add_argument("--garage-output-cap", type=int, default=EVALUATOR_OUTPUT_CAP)
    parser.add_argument("--evaluator-output-cap", type=int, default=EVALUATOR_OUTPUT_CAP)
    parser.add_argument("--reasoning-effort", choices=REASONING_CHOICES, default="default")
    parser.add_argument("--model-revision")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args(argv)
    if not 1<=args.workers<=8:
        parser.error("workers must be between 1 and 8")
    if (args.extractor_model or args.judge == "openai" or args.garage_judge) and not args.allow_network:
        parser.error("API evaluation requires explicit --allow-network")
    if args.judge != "none" and not args.judge_model:
        parser.error("--judge-model is required")
    if args.garage_judge and not args.judge_model:
        parser.error("--garage-judge requires the existing --judge-model")
    if args.judge == "local-nli" and not Path(args.judge_model).is_dir():
        parser.error("local-nli requires an existing local model directory; downloads are disabled")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap samples must be nonnegative")
    return args


def load_reference_pools(path, examples):
    if path is None:
        return {}
    pools = {}
    for row in read_jsonl(path):
        key = row["example_id"]
        if key in pools or key not in examples:
            raise ValueError("duplicate or unknown reference-pool example")
        if row.get("question_hash") != canonical_hash(examples[key].question):
            raise ValueError("reference pool question hash mismatch")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(x, str) or not x.strip() for x in evidence):
            raise ValueError("reference evidence must be a list of nonempty text strings")
        pools[key] = evidence
    return pools


def aggregate_results(results):
    from rag_benchmark.evaluation.contracts import aggregate_metric_results
    return aggregate_metric_results(results)


def canonical_rgb_views(processed_root, examples, contexts, *, review_path=None):
    """Use the generation path's canonical RGB protocol, never prediction claims."""
    if not any(example.dataset == "rgb_en" for example in examples.values()):
        return examples, contexts, None
    from rag_benchmark.data import DatasetSlice, load_all_corpus
    from rag_benchmark.rgb_protocol import apply_rgb_protocol, RGB_INPUT_PROTOCOL
    path = Path(review_path) if review_path is not None else None
    if path is not None and not path.is_file():
        raise FileNotFoundError("RGB review overlay not found")
    overlay = json.loads(path.read_text(encoding="utf-8")) if path is not None else None
    raw_corpus = load_all_corpus(processed_root)
    dataset_slice = DatasetSlice("rgb_en", "canonical_evaluation", "canonical_context_sets",
        list(examples.values()), list(raw_corpus.values()), {}, context_sets_by_example=contexts)
    canonical = apply_rgb_protocol(dataset_slice, review_overlay=overlay)
    return {example.example_id: example for example in canonical.examples}, canonical.context_sets_by_example, {
        "input_display_protocol": RGB_INPUT_PROTOCOL,
        "review_overlay_path": str(path.resolve()) if overlay is not None else None,
        "review_overlay_hash": file_sha256(path) if overlay is not None else None,
        "canonical_corpus_hash": canonical_hash([vars(chunk) for chunk in canonical.corpus]),
        "review_basis": "canonical_question_and_ordered_chunk_ids_text_not_prediction_metadata",
    }


def neutral_rgb_reference_pools(pools):
    """Strip source-category display titles, retaining reference passage bodies."""
    from rag_benchmark.rgb_protocol import rgb_display_title
    result = {}
    for example_id, passages in pools.items():
        normalized = []
        for passage in passages:
            if passage.startswith("Title:"):
                _, separator, body = passage.partition("\n")
                passage = f"Title: {rgb_display_title(canonical_hash(body))}" + (f"\n{body}" if separator else "")
            normalized.append(passage)
        result[example_id] = normalized
    return result


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new evaluation output directory; existing artifacts are immutable")
    example_rows = list(read_jsonl(args.processed_root / "examples.jsonl"))
    if len({row["example_id"] for row in example_rows}) != len(example_rows):
        raise ValueError("duplicate processed example IDs")
    examples = load_examples(args.processed_root)
    qrels = load_qrels(args.processed_root)
    annotations = load_annotations(args.processed_root)
    contexts = load_context_sets(args.processed_root)
    from rag_benchmark.data_semantics import correct_garage_views
    reviews = json.loads(args.garage_reviews.read_text(encoding="utf-8")) if args.garage_reviews else None
    examples, contexts = correct_garage_views(args.processed_root, examples, contexts, reviews=reviews)
    examples, contexts, rgb_protocol = canonical_rgb_views(args.processed_root, examples, contexts,
                                                         review_path=args.rgb_reviews)
    selected_contexts = {}
    pools = load_reference_pools(args.reference_evidence, examples)
    if rgb_protocol is not None:
        pools = neutral_rgb_reference_pools(pools)
    predictions = [p for path in args.predictions for p in read_jsonl_records(path, PredictionRecord)]
    if not predictions:
        raise ValueError("no predictions")
    seen = set()
    method_keys = defaultdict(set)
    for prediction in predictions:
        key = (prediction.pipeline, prediction_key(prediction))
        if key in seen:
            raise ValueError("duplicate prediction pairing key")
        seen.add(key)
        method_keys[prediction.pipeline].add(key[1])
        if prediction.example_id not in examples or prediction.question != examples[prediction.example_id].question:
            raise ValueError("prediction/example question mismatch")
        from rag_benchmark.evaluation.identity import validate_example_identity
        validate_example_identity(prediction, examples[prediction.example_id])
        context_id = prediction.metadata["context_set_id"]
        if context_id is not None:
            matches = [c for c in contexts.get(prediction.example_id, []) if c.context_set_id == context_id]
            if len(matches) != 1:
                raise ValueError("missing or duplicate canonical context instance")
            selected_contexts[key] = matches[0]
        elif contexts.get(prediction.example_id):
            raise ValueError("canonical example requires a concrete context instance")
    if len({frozenset(keys) for keys in method_keys.values()}) != 1:
        raise ValueError("all compared methods must have identical sample/condition/seed keys")
    if len({(examples[p.example_id].dataset, examples[p.example_id].split, p.task_track) for p in predictions}) != 1:
        raise ValueError("evaluate one dataset/split/task track per invocation")
    # Validate all pairing contracts before any potentially paid evaluation call.
    preflight = defaultdict(list)
    for prediction in predictions:
        preflight[prediction.pipeline].append(pairing_view(prediction))
    groups = list(preflight.values())
    for rows in groups[1:]:
        paired_comparisons(groups[0], rows, keys=[], samples=0, examples_by_id=examples)
    extractor = aligner = judge = None
    if args.extractor_model:
        backend = OpenAITextPromptGenerator(repo_root=REPO_ROOT, model=args.extractor_model,
            max_output_tokens=args.evaluator_output_cap, reasoning_effort=reasoning_parameter(args.reasoning_effort),
            max_attempts=1, timeout_seconds=EVALUATOR_TIMEOUT_SECONDS)
        extractor = TextClaimExtractor(backend, revision=args.model_revision)
        aligner = TextCitationAligner(backend)
    if args.judge == "openai":
        backend = OpenAITextPromptGenerator(repo_root=REPO_ROOT, model=args.judge_model,
            max_output_tokens=args.evaluator_output_cap, reasoning_effort=reasoning_parameter(args.reasoning_effort),
            max_attempts=1, timeout_seconds=EVALUATOR_TIMEOUT_SECONDS)
        judge = TextExactJudge(backend, revision=args.model_revision, max_input_tokens=EVALUATOR_INPUT_CAP)
    elif args.judge == "local-nli":
        judge = NLIExactJudge(TransformersNLIEvidenceJudge(model_name=args.judge_model, revision=args.model_revision))
    garage_evaluator, garage_pools = None, {}
    if args.garage_judge:
        if any(e.dataset != "garage" for e in (examples[p.example_id] for p in predictions)):
            raise ValueError("GaRAGe evaluator requires the GaRAGe track")
        from rag_benchmark.evaluation.garage import GarageEvaluator
        from rag_benchmark.data import load_all_corpus
        corpus = load_all_corpus(args.processed_root)
        calibration = json.loads(args.garage_calibration.read_text(encoding="utf-8")) if args.garage_calibration else None
        garage_evaluator = GarageEvaluator(OpenAITextPromptGenerator(repo_root=REPO_ROOT,
            model=args.judge_model, max_output_tokens=args.garage_output_cap,
            reasoning_effort=reasoning_parameter(args.reasoning_effort), max_attempts=1,
            timeout_seconds=EVALUATOR_TIMEOUT_SECONDS), max_input_tokens=EVALUATOR_INPUT_CAP, calibration=calibration)
        for p in predictions:
            context = selected_contexts.get((p.pipeline, prediction_key(p)))
            if context is not None and all(cid in corpus and "evidence_relevant" in corpus[cid].metadata for cid in context.chunk_ids):
                garage_pools[p.example_id] = [corpus[cid].text for cid in context.chunk_ids
                    if str(corpus[cid].metadata["evidence_relevant"]).upper() == "YES"]
    experiment_cache = ExperimentEvaluationCache(args.experiment_cache) if args.experiment_cache else None
    service = EvaluationService(extractor=extractor, judge=judge, aligner=aligner,
                                garage_evaluator=garage_evaluator, experiment_cache=experiment_cache)
    implementation_hash = source_fingerprint(REPO_ROOT)
    cache_origin = None
    if args.cache_from:
        previous = json.loads((args.cache_from / "evaluation_manifest.json").read_text(encoding="utf-8"))
        cache_path = args.cache_from / "evaluation_cache.json"
        if previous.get("status") != "completed" or previous.get("implementation_hash") != implementation_hash:
            raise ValueError("cache requires a completed run of the same implementation")
        if previous.get("outputs", {}).get(cache_path.name) != file_sha256(cache_path):
            raise ValueError("evaluation cache hash mismatch")
        service.restore_cache(json.loads(cache_path.read_text(encoding="utf-8")))
        cache_origin = {"manifest_hash": file_sha256(args.cache_from / "evaluation_manifest.json"),
                        "cache_hash": file_sha256(cache_path)}
    manifest = {"schema_version": EVALUATION_VERSION, "implementation_hash": source_fingerprint(REPO_ROOT),
                "prediction_files": {str(p.resolve()): file_sha256(p) for p in args.predictions},
                "examples_hash": canonical_hash([vars(e) for e in examples.values()]),
                "qrels_hash": canonical_hash({k: [vars(q) for q in v] for k, v in qrels.items()}),
                "annotations_hash": canonical_hash({k: [vars(a) for a in v] for k, v in annotations.items()}),
                "context_sets_hash": canonical_hash({k: [vars(c) for c in v] for k, v in contexts.items()}),
                "rgb_protocol": rgb_protocol,
                "reference_pools_hash": canonical_hash(pools), "seed": args.seed,
                "bootstrap_samples": args.bootstrap_samples, "extractor": getattr(extractor, "identity", None),
                "judge": getattr(judge, "identity", None), "human_validation": "not_performed",
                "aligner": getattr(aligner, "identity", None), "cache_origin": cache_origin,
                "experiment_cache": {"protocol": CACHE_PROTOCOL, "path": str(experiment_cache.path)} if experiment_cache else None,
                "garage_evaluator": getattr(garage_evaluator, "identity", None),
                "cost_scope": "new_evaluation_calls_only_not_method_runtime", "workers":args.workers,"status": "running"}
    output.mkdir(parents=True, exist_ok=False)
    def save_json(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_json("evaluation_manifest.json", manifest)
    results, views, diagnostics = [], defaultdict(list), []
    def evaluate_one(prediction):
        from rag_benchmark.generation.budget import execution_budget
        with execution_budget() as budget:
            result = service.evaluate(prediction, example=examples[prediction.example_id],
                reference_evidence=pools.get(prediction.example_id), garage_relevant_evidence=garage_pools.get(prediction.example_id),
                qrels=qrels.get(prediction.example_id), annotations=annotations.get(prediction.example_id),
                context_set=selected_contexts.get((prediction.pipeline, prediction_key(prediction))),
                gold_chunk_ids={q.chunk_id for q in qrels.get(prediction.example_id, [])})
            result["execution_budget"] = budget.snapshot()
        return prediction,result
    try:
        with (output / "evaluations.jsonl").open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for prediction,result in pool.map(evaluate_one,predictions):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
                view = evaluated_prediction(prediction, result)
                views[prediction.pipeline].append(view)
                diagnostics.append(diagnose_prediction(view))
                print(json.dumps({"evaluated":len(results),"total":len(predictions),"output":str(output)}),flush=True)
        documents = []
        for pipeline, rows in views.items():
            cells = aggregate_results([r for r in results if r["pipeline"] == pipeline])
            derived_keys = set(CONDITIONAL) if any(r.verification for r in rows) else set()
            if all("eligibility_accuracy" in r.metrics for r in rows):
                derived_keys.update(("deflection_precision", "deflection_recall", "deflection_f1", "deflection_tpr", "deflection_fpr"))
            if all(examples[r.example_id].answers and examples[r.example_id].answers[0].get("type") == "verification_label" for r in rows):
                derived_keys.add("verification_macro_f1")
            for key in sorted(derived_keys):
                value = metric_value(rows, key, examples_by_id=examples)
                cells[key] = {"value": value, "status": "computed" if value is not None else "not_estimable",
                              "protocol_version": EVALUATION_VERSION, "n_expected": len(rows),
                              "n_evaluated": len(rows) if value is not None else 0}
            documents.append({"run_id": f"{output.name}:{pipeline}", "dataset": examples[rows[0].example_id].dataset,
                              "split": examples[rows[0].example_id].split, "task_track": rows[0].task_track,
                              "num_examples": len(rows), "evaluation_protocol": EVALUATION_VERSION,
                              "pipelines": {pipeline: {k: v["value"] for k, v in cells.items() if v["status"] == "computed"}},
                              "metric_results": {pipeline: cells},
                              "confidence_intervals": {pipeline: confidence_intervals(rows, keys=list(cells),
                                  samples=args.bootstrap_samples, seed=args.seed, examples_by_id=examples)}})
        comparisons = []
        names = sorted(views)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                keys = sorted({k for doc in documents for values in doc["metric_results"].values() for k in values})
                for row in paired_comparisons(views[left], views[right], keys=keys, samples=args.bootstrap_samples,
                                              seed=args.seed, examples_by_id=examples):
                    comparisons.append({"baseline": left, "candidate": right, **row})
        write_unified_metrics_report(output, documents, comparisons=comparisons)
        save_json("metrics.json", {"documents": documents, "comparisons": comparisons})
        write_jsonl(output / "diagnostics.jsonl", (d.to_json_dict() for d in diagnostics))
        save_json("diagnostics_summary.json", summarize_diagnostics(diagnostics))
        write_jsonl(output / "evaluation_calls.jsonl", service.calls)
        save_json("evaluation_cache.json", service.cache_payload())
        reuse = [stage for result in results for stage in result["evaluation_reuse"]["stages"]]
        save_json("evaluation_reuse.json", {
            "protocol": CACHE_PROTOCOL, "logical_stage_requests": len(reuse),
            "reused_stage_results": sum(stage["cache_hit"] for stage in reuse),
            "new_stage_attempts": len(service.calls),
            "physical_usage_source": "execution_budget_and_evaluation_calls_not_reused_trace_usage",
        })
        manifest["status"] = "completed"
        manifest["outputs"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p.name != "evaluation_manifest.json"}
        save_json("evaluation_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        write_jsonl(output / "evaluation_calls.jsonl", service.calls)
        save_json("failed_evaluation.json", {"error_type": type(exc).__name__,
                  "budget": budget.snapshot() if "budget" in locals() else None})
        save_json("evaluation_manifest.json", manifest)
        raise
    print(json.dumps({"output_dir": str(output), "predictions": len(predictions), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
