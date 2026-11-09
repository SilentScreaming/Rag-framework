"""Prediction-level paired/cluster resampling, including non-additive metrics."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable

import numpy as np


STATISTICS_PROTOCOL = "paired-cluster-v1"
CONDITIONAL = {"correction_success_rate", "harmful_revision_rate", "preserve_correct_rate",
               "net_verification_gain", "over_correction_rate", "grounding_gain",
               "task_score_increase_rate", "task_score_decrease_rate"}
RECOMPUTED = {"verification_macro_f1", "deflection_precision", "deflection_recall", "deflection_f1",
              "deflection_tpr", "deflection_fpr", "hallucination_span_f1", "hallucination_response_f1"} | CONDITIONAL


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def prediction_key(row):
    from rag_benchmark.evaluation.identity import prediction_identity
    return prediction_identity(row).key(row.example_id)


def _index(rows):
    result = {}
    for row in rows:
        key = prediction_key(row)
        if key in result:
            raise ValueError(f"duplicate evaluation pairing key: {key}")
        result[key] = row
    return result


def metric_value(rows, key, *, examples_by_id=None):
    from rag_benchmark.evaluation.tracks import multiclass_macro_f1, normalize_verification_label, verification_intervention_metrics
    if not rows:
        return None
    if key == "verification_macro_f1":
        pairs = []
        for row in rows:
            if row.metadata.get("task_assessment", {}).get("status") == "failed":
                return None
            example = (examples_by_id or {}).get(row.example_id)
            answers = example.answers if example else [a.to_json_dict() for a in row.gold_answers]
            if not answers or answers[0].get("type") != "verification_label":
                return None
            gold = normalize_verification_label(answers[0]["text"])
            if gold not in {"SUPPORTS", "REFUTES", "INSUFFICIENT"}:
                raise ValueError("invalid gold verification label")
            pairs.append((gold, normalize_verification_label(row.answer)))
        return multiclass_macro_f1(pairs)
    if key in CONDITIONAL:
        tracks = {r.task_track for r in rows}
        if len(tracks) != 1:
            raise ValueError("cannot mix task tracks in an intervention metric")
        if key == "grounding_gain":
            required = ("pre_verification_claim_support_rate", "post_verification_claim_support_rate")
        elif key in {"net_verification_gain", "task_score_increase_rate", "task_score_decrease_rate"}:
            required = ("pre_verification_task_score", "post_verification_task_score")
            if not any(required[0] in r.metrics for r in rows):
                required = ("pre_verification_f1", "post_verification_f1")
            else:
                metric_names = {(r.metadata.get("pre_verification_task_score_metric"),
                                 r.metadata.get("post_verification_task_score_metric")) for r in rows}
                if len(metric_names) != 1:
                    return None
                before_metric, after_metric = next(iter(metric_names))
                if not before_metric or before_metric != after_metric:
                    return None
        else:
            required = ("pre_verification_task_correct", "post_verification_task_correct")
            if not any(required[0] in r.metrics for r in rows) and next(iter(tracks)) != "long_answer_citation":
                required = ("pre_verification_em", "post_verification_em")
        if any(not all(finite(r.metrics.get(k)) for k in required) for r in rows):
            return None
        return verification_intervention_metrics(rows, track=next(iter(tracks))).get(key)
    if key.startswith("deflection_"):
        from rag_benchmark.evaluation.tracks import deflection_metrics
        if examples_by_id is None or any(r.example_id not in examples_by_id for r in rows):
            return None
        return deflection_metrics(rows, examples_by_id=examples_by_id).get(key)
    if key in {"hallucination_span_f1", "hallucination_response_f1"}:
        prefix = key.removesuffix("_f1")
        counts = [r.metrics.get(f"{prefix}_{k}") for r in rows for k in ("tp", "fp", "fn")]
        if not all(finite(v) and v >= 0 for v in counts):
            return None
        tp, fp, fn = (sum(r.metrics[f"{prefix}_{k}"] for r in rows) for k in ("tp", "fp", "fn"))
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    values = [r.metrics.get(key) for r in rows]
    return float(np.mean(values)) if all(finite(v) for v in values) else None


def _clusters(keys):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key[:2]].append(index)
    return [groups[k] for k in sorted(groups)]


def _sample_indices(groups, rng):
    return [i for g in rng.integers(0, len(groups), size=len(groups)) for i in groups[g]]


def confidence_intervals(predictions, *, keys, samples=1000, seed=42, examples_by_id=None):
    indexed = _index(predictions)
    ordered = sorted(indexed)
    rows = [indexed[k] for k in ordered]
    groups = _clusters(ordered)
    result = {}
    for key in dict.fromkeys(keys):
        point = metric_value(rows, key, examples_by_id=examples_by_id)
        rng = np.random.default_rng(seed)
        draws = []
        attempted = samples if point is not None and len(groups) >= 2 and samples > 0 else 0
        if attempted:
            for _ in range(samples):
                value = metric_value([rows[i] for i in _sample_indices(groups, rng)], key, examples_by_id=examples_by_id)
                if finite(value):
                    draws.append(value)
        low, high = np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) >= 2 else (None, None)
        result[key] = {"mean": point, "estimate": point, "ci95_low": low, "ci95_high": high,
                       "n": len(rows), "n_clusters": len(groups), "bootstrap_samples": samples,
                       "valid_resamples": len(draws), "attempted_resamples": attempted,
                       "undefined_resamples": attempted - len(draws),
                       "status": "computed" if low is not None else "not_estimable",
                       "reason": None if low is not None else "missing_inputs_or_insufficient_clusters",
                       "protocol_version": STATISTICS_PROTOCOL}
    return result


def paired_comparisons(baseline, candidate, *, keys: Iterable[str], samples=1000, seed=42, examples_by_id=None):
    left, right = _index(baseline), _index(candidate)
    if left.keys() != right.keys():
        raise ValueError("paired evaluation requires identical sample/condition/seed keys")
    ordered = sorted(left)
    a, b = [left[k] for k in ordered], [right[k] for k in ordered]
    for x, y in zip(a, b):
        if x.question != y.question or x.task_track != y.task_track or x.gold_answers != y.gold_answers:
            raise ValueError("paired prediction task/reference mismatch")
        for field in ("answer_metric_protocol", "evaluation_protocol"):
            if x.metadata.get(field) != y.metadata.get(field):
                raise ValueError("paired evaluation protocol mismatch")
    groups = _clusters(ordered)
    results = []
    for key in dict.fromkeys(keys):
        va = metric_value(a, key, examples_by_id=examples_by_id)
        vb = metric_value(b, key, examples_by_id=examples_by_id)
        observed = vb - va if va is not None and vb is not None else None
        rng = np.random.default_rng(seed)
        draws, null = [], []
        if observed is not None and len(groups) >= 2 and samples > 0:
            for _ in range(samples):
                idx = _sample_indices(groups, rng)
                sa = metric_value([a[i] for i in idx], key, examples_by_id=examples_by_id)
                sb = metric_value([b[i] for i in idx], key, examples_by_id=examples_by_id)
                if sa is not None and sb is not None:
                    draws.append(sb - sa)
                aa, bb = list(a), list(b)
                for group, swap in zip(groups, rng.integers(0, 2, len(groups))):
                    if swap:
                        for i in group:
                            aa[i], bb[i] = bb[i], aa[i]
                na = metric_value(aa, key, examples_by_id=examples_by_id)
                nb = metric_value(bb, key, examples_by_id=examples_by_id)
                if na is not None and nb is not None:
                    null.append(nb - na)
        low, high = np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) >= 2 else (None, None)
        p = (sum(abs(v) >= abs(observed) - 1e-12 for v in null) + 1) / (len(null) + 1) if null else None
        results.append({"metric": key, "mean_delta": observed, "ci95_low": low, "ci95_high": high,
                        "paired_p_value": p, "n": len(a), "n_clusters": len(groups),
                        "bootstrap_samples": samples, "valid_resamples": len(draws),
                        "valid_permutations": len(null), "status": "computed" if low is not None else "not_estimable",
                        "test": "paired_cluster_prediction_permutation", "protocol_version": STATISTICS_PROTOCOL,
                        "comparison_role": "exploratory", "multiplicity_adjustment": "none"})
    return results
