"""Deterministic task matrix for independent offline evaluation."""
from dataclasses import replace
from rag_benchmark.evaluation.answers import assess_task
from rag_benchmark.evaluation.contracts import MetricResult
from rag_benchmark.evaluation.basic import retrieval_scores

TASK_MATRIX_PROTOCOL = "task-matrix-v1"
TASK_METRICS = {
    "short_answer": ("em", "f1", "recall@10", "all_gold_hit@10"),
    "asqa": ("asqa_str_em_alce", "asqa_group_recall_strict"),
    "qampari": ("qampari_precision_alce", "qampari_recall_alce", "qampari_f1_top5_alce",
                 "qampari_unique_entity_precision", "qampari_unique_entity_recall"),
    "scifact": ("verification_label_accuracy", "abstract_f1", "rationale_source_id_f1", "rationale_visible_complete_f1"),
    "garage": ("eligibility_accuracy", "answer_abstain_accuracy", "garage_eligibility", "em", "f1", "attribution_f1", "raf", "uraf"),
    "rgb": ("em", "f1", "eligibility_accuracy"),
    "ragtruth": ("hallucination_response_f1", "hallucination_span_f1"),
}


def scoped_example(prediction, example, context_set=None):
    if example is None:
        return None
    expected_id = prediction.metadata.get("context_set_id")
    if context_set is not None:
        if (context_set.context_set_id != expected_id or context_set.example_id != example.example_id
                or context_set.dataset != example.dataset):
            raise ValueError("context instance mismatch")
        return replace(example, metadata={**example.metadata,
            **{k: v for k, v in getattr(context_set, 'metadata', {}).items()
               if k in {'insufficiency_analysis_eligible', 'evidence_review_status'}},
            "expected_behavior": context_set.expected_behavior, "context_condition": context_set.condition})
    if expected_id is not None:
        # Do not substitute example-level expectations for an unknown context instance.
        return replace(example, metadata={k: v for k, v in example.metadata.items() if k != "expected_behavior"})
    return example


def evaluate_task(prediction, *, example=None, qrels=None, annotations=None, context_set=None, answer_behavior=None):
    from rag_benchmark.evaluation.tracks import short_answer_metrics, verification_track_metrics
    scoped = scoped_example(prediction, example, context_set)
    dataset = (scoped.dataset if scoped else prediction.metadata.get("dataset", "")).lower()
    kind = next((k for k in ("ragtruth", "scifact", "garage", "qampari", "asqa", "rgb") if k in dataset),
                "short_answer" if prediction.task_track == "short_answer" else "unsupported")
    task = assess_task(prediction, scoped, answer_behavior=answer_behavior)
    values = dict(task["metrics"])
    clean = prediction.model_copy(update={"metrics": values})
    if scoped is not None and task["status"] != "failed" and kind != "ragtruth":
        if kind in {"garage", "rgb", "scifact"}:
            extra = verification_track_metrics(clean, example=scoped, qrels=qrels or [], annotations=annotations or [],
                                               answer_behavior=answer_behavior)
            # Missing qrels are not empty gold supervision.
            values.update({k: v for k, v in extra.items()
                           if (k == "eligibility_accuracy"
                               or (k.startswith("abstract_") and qrels and task["status"] == "computed")
                               or (k.startswith("attribution_") and qrels)
                               or (k.startswith("rationale_") and (qrels or annotations)))})
        if kind == "short_answer" and qrels:
            values.update(short_answer_metrics(clean, example=scoped, qrels=qrels))
            if "em" not in task["metrics"]:
                values = {k: v for k, v in values.items() if not k.startswith(("kilt_", "hotpot_joint_"))}
        if qrels:
            for k in (1, 5, 10):
                values.update({name: value for name, value in retrieval_scores(
                    [r.chunk_id for r in prediction.retrieved_chunks],
                    {q.chunk_id for q in qrels if q.relevance > 0}, k=k).items() if value is not None})
    results = {k: MetricResult(float(v), n_expected=1, n_evaluated=1).to_dict()
               for k, v in values.items() if v is not None}
    for key in TASK_METRICS.get(kind, ("task_score",)):
        if key not in results:
            status = "failed" if task["status"] == "failed" else "unsupported" if kind in {"ragtruth", "unsupported"} or key in {"garage_eligibility", "raf", "uraf"} else "missing_input"
            if task["status"] != "failed" and kind in {"garage", "rgb"} and scoped and scoped.metadata.get("expected_behavior") == "abstain" and key in {"em", "f1"}:
                status = "not_applicable"
            results[key] = MetricResult(None, status=status, n_expected=1,
                reason="use_dedicated_ragtruth_entry" if kind == "ragtruth" else
                       "independent_garage_evaluator_not_configured" if key in {"garage_eligibility", "raf", "uraf"} else
                       "required_supervision_or_visibility_unavailable").to_dict()
    task.update(matrix_protocol=TASK_MATRIX_PROTOCOL, evaluator=kind,
                expected_behavior=scoped.metadata.get("expected_behavior") if scoped else None)
    return task, results
