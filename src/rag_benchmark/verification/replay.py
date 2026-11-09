"""Pairing preflight and task-aware summaries for verification replay."""
from __future__ import annotations

from collections import Counter

from rag_benchmark.evaluation.answers import assess_task
from rag_benchmark.evaluation.contracts import EVALUATION_VERSION
from rag_benchmark.evaluation.statistics import paired_comparisons, prediction_key
from rag_benchmark.stage_cache import canonical_hash


def entity_edit_audit(prediction, example):
    """Match gold references for reporting after revision has completed."""
    from rag_benchmark.evaluation.answers import answer_groups
    from rag_benchmark.evaluation.basic import normalize_answer
    if example.task_type != "list_answer_citation":
        return None
    try:
        groups = answer_groups(example.answers, kind="qampari", metadata=example.metadata)
    except ValueError:
        return {"status": "missing_answer_grouping"}
    def matches(text):
        return sum(normalize_answer(text) in {normalize_answer(a) for a in g.aliases} for g in groups)
    edits = []
    for edit in prediction.verification.metadata.get("edits", []):
        if edit["applied_action"] not in {"remove", "replace"}:
            continue
        count = matches(edit["original"])
        new_count = matches(edit["candidate"]) if edit["applied_action"] == "replace" else None
        edits.append({"target_id": edit["target_id"], "action": edit["applied_action"],
            "original_gold_group_matches": count, "replacement_gold_group_matches": new_count})
    return {"status": "computed", "interpretation": "gold_alias_membership_not_universal_entity_truth", "edits": edits,
        "removed_gold_matching_items": sum(e["original_gold_group_matches"] > 0 for e in edits),
        "removed_nonmatching_items": sum(e["original_gold_group_matches"] == 0 for e in edits),
        "added_nonmatching_items": sum(e["replacement_gold_group_matches"] == 0 for e in edits),
        "ambiguous_alias_edits": sum(e["original_gold_group_matches"] > 1 or (e["replacement_gold_group_matches"] or 0) > 1 for e in edits)}


def validate_drafts(drafts, examples, *, expected_track, seed=42, data_version=None, context_ids=None):
    by_id = {}
    for row in drafts:
        if row.example_id in by_id:
            raise ValueError("duplicate draft ID; select one context/seed before replay")
        by_id[row.example_id] = row
    for example in examples:
        if example.example_id not in by_id:
            raise ValueError("draft prediction file is missing selected examples")
        row = by_id[example.example_id]
        if row.question != example.question or row.task_track != expected_track:
            raise ValueError("draft question/task mismatch")
        from rag_benchmark.evaluation.identity import validate_example_identity
        validate_example_identity(row, example, data_version=data_version, seed=seed,
                                  context_set_id=(context_ids or {}).get(example.example_id))
    return {e.example_id: by_id[e.example_id] for e in examples}


def summarize_replay(before, after, examples, *, samples=1000, seed=42):
    left = {prediction_key(p): p for p in before}
    right = {prediction_key(p): p for p in after}
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError("replay requires unique identical sample/condition/seed keys")
    pairs, a_views, b_views, reasons, transitions = [], [], [], Counter(), Counter()
    def behavior(row):
        return row.metadata.get('task_assessment', {}).get('answer_behavior') or {
            'status': 'not_run', 'label': 'unknown', 'reason': 'behavior_not_independently_evaluated'}
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.question != b.question or a.task_track != b.task_track:
            raise ValueError("replay task mismatch")
        if not b.verification or b.verification.original_answer != a.answer:
            raise ValueError("replay original answer does not match frozen draft")
        example = examples[a.example_id]
        av, bv = assess_task(a, example, answer_behavior=behavior(a)), assess_task(b, example, answer_behavior=behavior(b))
        for row, task, target in ((a, av, a_views), (b, bv, b_views)):
            target.append(row.model_copy(update={"metrics": task["metrics"], "gold_answers": a.gold_answers,
                "metadata": {**row.metadata, "evaluation_protocol": EVALUATION_VERSION,
                             "answer_metric_protocol": task["protocol_version"], "task_assessment": task}}))
        known = av["task_correct"] is not None and bv["task_correct"] is not None
        pairs.append({"example_id": a.example_id, "draft_hash": canonical_hash(a.to_json_dict()),
            "entity_edit_audit": entity_edit_audit(b, example),
            "before": av, "after": bv, "answer_changed": a.answer != b.answer,
            "draft_cost": a.cost.estimated_cost if cost_complete(a) else None,
            "complete_method_cost": b.cost.estimated_cost if cost_complete(a) and cost_complete(b) else None,
            "incremental_cost": b.cost.estimated_cost - a.cost.estimated_cost
                if cost_complete(a) and cost_complete(b) else None,
            "known_draft_cost": a.cost.estimated_cost, "known_complete_method_cost": b.cost.estimated_cost,
            "draft_output_tokens": a.tokens.output_tokens, "complete_method_output_tokens": b.tokens.output_tokens,
            "score_metric": av['score_metric'] if av['score_metric'] == bv['score_metric'] else None,
            "score_delta": bv["score"] - av["score"] if av["score"] is not None and bv["score"] is not None
                and av['score_metric'] and av['score_metric'] == bv['score_metric'] else None,
            "repair": bool(not av["task_correct"] and bv["task_correct"]) if known else None,
            "harm": bool(av["task_correct"] and not bv["task_correct"]) if known else None})
        reasons.update(e["rejection_reason"] for e in b.verification.metadata.get("edits", []) if e.get("rejection_reason"))
        if example.task_type == "scientific_claim_verification":
            from rag_benchmark.evaluation.tracks import normalize_verification_label
            transitions[f"{normalize_verification_label(a.answer)}->{normalize_verification_label(b.answer)}"] += 1
    keys = sorted({k for p in a_views + b_views for k in p.metrics})
    if all(examples[p.example_id].task_type == "scientific_claim_verification" for p in before):
        keys.append("verification_macro_f1")
    deltas = [p["score_delta"] for p in pairs]
    known = all(p["repair"] is not None for p in pairs)
    wrong = sum(p["before"]["task_correct"] is False for p in pairs)
    correct = sum(p["before"]["task_correct"] is True for p in pairs)
    from rag_benchmark.evaluation.tracks import is_abstention, normalize_verification_label
    behavior_decisions = [False if examples[p.example_id].task_type == 'scientific_claim_verification'
                          and normalize_verification_label(p.answer) in {'SUPPORTS', 'REFUTES', 'INSUFFICIENT'}
                          else is_abstention(p, answer_behavior=behavior(p)) for p in after]
    answered = [p for p, decision in zip(after, behavior_decisions) if decision is False]
    behavior_complete = all(d is not None for d in behavior_decisions)
    answered_tasks = [assess_task(p, examples[p.example_id], answer_behavior=behavior(p)) for p in answered]
    shared_score_metric = ({p['score_metric'] for p in pairs} if pairs else set())
    return {"schema_version": "verification-replay-summary-v1", "evaluation_protocol": EVALUATION_VERSION,
        "examples": len(pairs), "paired": pairs,
        "mean_task_score_delta": sum(deltas) / len(deltas) if deltas and all(v is not None for v in deltas)
            and len(shared_score_metric) == 1 and None not in shared_score_metric else None,
        "mean_task_score_metric": next(iter(shared_score_metric)) if len(shared_score_metric) == 1 and None not in shared_score_metric else None,
        "correction_success_rate": sum(p["repair"] for p in pairs) / wrong if known and wrong else None,
        "harmful_revision_rate": sum(p["harm"] for p in pairs) / correct if known and correct else None,
        "initially_correct_count": correct, "initially_incorrect_count": wrong,
        "answer_changed_rate": sum(p["answer_changed"] for p in pairs) / len(pairs) if pairs else None,
        "answer_coverage": len(answered) / len(after) if after and behavior_complete else None,
        "answer_behavior_n_evaluated": sum(d is not None for d in behavior_decisions),
        "answer_behavior_n_expected": len(after), "known_answered_count": len(answered),
        "selective_risk": sum(not t["task_correct"] for t in answered_tasks) / len(answered_tasks)
            if behavior_complete and answered_tasks and all(t["task_correct"] is not None for t in answered_tasks) else None,
        "rejection_reasons": dict(reasons), "label_transitions": dict(transitions),
        "paired_comparisons": paired_comparisons(a_views, b_views, keys=keys, samples=samples, seed=seed, examples_by_id=examples)}


def cost_complete(prediction):
    return (prediction.cost.estimated_cost is not None
            and not prediction.generator.get("unknown_attempt_usage", False)
            and not prediction.metadata.get("execution_budget", {}).get("unknown_usage", False))
