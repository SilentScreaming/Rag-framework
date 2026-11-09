"""Grouped answer scoring. ALCE-compatible and strict diagnostics are distinct."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_benchmark.evaluation.basic import normalize_answer
from rag_benchmark.generation.answer_format import parse_entity_list, strip_citation_markers

ANSWER_PROTOCOL = "grouped-answers-v1"
REFERENCE_PROTOCOL = "alce-eval-reference-v1-no-cot-ratio-scale"


def alce_reference_text(answer: str) -> str:
    """Explicit reference CLI preprocessing; strict/full-answer metrics do not truncate."""
    first_line = answer.strip().split("\n")[0].replace("<|im_end|>", "")
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", first_line)).replace(" |", "").replace("]", "")


@dataclass(frozen=True)
class AnswerGroup:
    group_id: str
    aliases: tuple[str, ...]


def answer_groups(rows: list[dict], *, kind: str, metadata: dict | None = None) -> list[AnswerGroup]:
    if kind == "asqa" and (metadata or {}).get("qa_pairs"):
        groups = [AnswerGroup(str(i), tuple(dict.fromkeys(normalize_answer(str(a)) for a in pair.get("short_answers", []) if normalize_answer(str(a)))))
                  for i, pair in enumerate(metadata["qa_pairs"])]
        if any(not group.aliases for group in groups):
            raise ValueError("missing_answer_grouping")
        return groups
    key = "qa_pair_index" if kind == "asqa" else "item_index"
    grouped: dict[str, list[str]] = {}
    relevant_type = "short" if kind == "asqa" else "list_item"
    for row in rows:
        if row.get("type") != relevant_type:
            continue
        index = row.get("metadata", {}).get(key)
        if index is None:
            raise ValueError("missing_answer_grouping")
        text = normalize_answer(str(row.get("text", "")))
        if text:
            grouped.setdefault(str(index), []).append(text)
    if not grouped and kind == "asqa":
        for index, pair in enumerate((metadata or {}).get("qa_pairs", [])):
            grouped[str(index)] = [normalize_answer(str(x)) for x in pair.get("short_answers", []) if str(x).strip()]
    if not grouped or any(not aliases for aliases in grouped.values()):
        raise ValueError("missing_answer_grouping")
    return [AnswerGroup(key, tuple(dict.fromkeys(values))) for key, values in sorted(grouped.items())]


def strict_list_items(answer: str) -> list[str]:
    return [normalize_answer(item.text) for item in parse_entity_list(answer)]


def unique_matching(items: list[str], groups: list[AnswerGroup]) -> tuple[int, int]:
    """Maximum one-to-one matching, including overlapping alias sets."""
    owners: dict[int, int] = {}
    edges = [[j for j, group in enumerate(groups) if item in group.aliases] for item in items]
    def visit(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False
    return sum(visit(i, set()) for i in range(len(items))), sum(len(e) > 1 for e in edges)


def grouped_answer_metrics(answer: str, rows: list[dict], *, kind: str, metadata: dict | None = None,
                           list_items=None) -> dict[str, Any]:
    groups = answer_groups(rows, kind=kind, metadata=metadata)
    if kind == "asqa":
        normalized = normalize_answer(strip_citation_markers(answer))
        reference = normalize_answer(alce_reference_text(answer))
        hits = sum(any(a in reference for a in g.aliases) for g in groups)
        strict_hits = sum(any(f" {a} " in f" {normalized} " for a in g.aliases) for g in groups)
        return {"asqa_str_em_alce": hits / len(groups),
                "asqa_group_recall_strict": strict_hits / len(groups),
                "gold_answer_recall": hits / len(groups),
                "gold_answer_count": float(len(groups)), "gold_answer_match_count": float(hits)}
    # ALCE eval.py: comma splitting and complete normalized alias equality.
    official = [normalize_answer(x.strip()) for x in alce_reference_text(answer).rstrip().rstrip(".").rstrip(",").split(",")]
    official = [x for x in official if x]
    aliases = {a for group in groups for a in group.aliases}
    hits = sum(any(a in official for a in g.aliases) for g in groups)
    precision = sum(x in aliases for x in official) / len(official) if official else 0.0
    recall = hits / len(groups)
    recall5 = min(hits, 5) / min(len(groups), 5)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    reference_metrics = {"qampari_precision": precision, "qampari_precision_alce": precision,
        "qampari_recall_alce": recall, "qampari_recall@5": recall5,
        "qampari_f1_alce": f1(precision, recall), "qampari_f1_top5_alce": f1(precision, recall5),
        "list_item_count": float(len(official))}
    try:
        raw = ([normalize_answer(item.text) for item in list_items] if list_items is not None else
               [] if normalize_answer(answer) in {"unknown", "i dont know"} else strict_list_items(answer))
    except ValueError:
        return reference_metrics  # strict grammar unavailable, reference score remains reproducible
    unique = list(dict.fromkeys(raw))
    matched, ambiguous = unique_matching(unique, groups)
    return {"qampari_precision": precision, "qampari_precision_alce": precision,
            "qampari_recall_alce": recall, "qampari_recall@5": recall5,
            "qampari_f1_alce": f1(precision, recall), "qampari_f1_top5_alce": f1(precision, recall5),
            "qampari_unique_entity_precision": matched / len(unique) if unique else 0.0,
            "qampari_unique_entity_recall": matched / len(groups),
            "list_item_count": float(len(raw)), "alce_list_item_count": float(len(official)),
            "list_duplicate_rate": (len(raw) - len(unique)) / len(raw) if raw else 0.0,
            "list_ambiguous_alias_count": float(ambiguous)}


def assess_task(prediction, example=None, *, answer_behavior=None) -> dict:
    """Task outcomes are separate from full-answer factuality and abstention."""
    from rag_benchmark.evaluation.basic import best_qa_scores
    from rag_benchmark.evaluation.tracks import normalize_verification_label
    answers = example.answers if example else [a.to_json_dict() for a in prediction.gold_answers]
    dataset = (example.dataset if example else str(prediction.metadata.get("dataset", ""))).casefold()
    metadata = example.metadata if example else {}
    metrics, correct, score, metric = {}, None, None, None
    status = "computed"
    failed = (prediction.metadata.get("citation_generation", {}).get("schema_failure")
              or prediction.generator.get("schema_failure")
              or (prediction.answer_bundle and prediction.answer_bundle.metadata.get('list_format_status') == 'unavailable'))
    if failed:
        status = "failed"
    elif "ragtruth" in dataset:
        status = "unsupported"
    elif "garage" in dataset or "rgb" in dataset or (example and example.task_type in {"grounding_and_selective_refusal", "rag_robustness_stress_test"}):
        from rag_benchmark.evaluation.tracks import is_abstention
        expected = metadata.get("expected_behavior")
        abstained = is_abstention(prediction, answer_behavior=answer_behavior)
        rgb_ineligible = ("rgb" in dataset and expected == "abstain"
                          and metadata.get("insufficiency_analysis_eligible") is not True)
        if expected not in {"answer", "abstain"} or rgb_ineligible:
            status = "missing_input"
        else:
            if abstained is not None:
                metrics["eligibility_accuracy"] = float(abstained == (expected == "abstain"))
                metrics["answer_abstain_accuracy"] = metrics["eligibility_accuracy"]
            if expected == "abstain" and abstained is not None:
                correct, score, metric = abstained, float(abstained), "eligibility_accuracy"
            elif expected == "answer" and any(str(a.get("text", "")).strip() for a in answers):
                metrics.update(best_qa_scores(prediction.answer, answers))
                correct = None if abstained is None else not abstained and metrics["em"] == 1
                score, metric = metrics["f1"], "f1"
            elif abstained is None:
                status = "missing_input"
            # Eligibility alone cannot establish correctness of a supplied answer.
    elif not any(str(a.get("text", "")).strip() for a in answers):
        status = "missing_input"
    elif "scifact" in dataset or answers[0].get("type") == "verification_label":
        labels = {normalize_verification_label(a["text"]) for a in answers}
        if not labels <= {"SUPPORTS", "REFUTES", "INSUFFICIENT"}:
            raise ValueError("invalid gold verification label")
        correct = normalize_verification_label(prediction.answer) in labels
        score, metric = float(correct), "verification_label_accuracy"
        metrics[metric] = score
    elif "asqa" in dataset or "qampari" in dataset:
        kind = "asqa" if "asqa" in dataset else "qampari"
        try:
            items = None
            if kind == "qampari" and prediction.answer_bundle is not None and prediction.answer_bundle.list_items is not None:
                from rag_benchmark.generation.answer_format import list_items_for_prediction
                items = list_items_for_prediction(prediction)
            metrics = grouped_answer_metrics(prediction.answer, answers, kind=kind, metadata=metadata,
                                             list_items=items)
            metric = "asqa_str_em_alce" if kind == "asqa" else "qampari_f1_top5_alce"
            score = metrics[metric]
            if kind == "qampari":
                if "qampari_unique_entity_precision" in metrics:
                    correct = metrics["qampari_unique_entity_precision"] == 1 and metrics["qampari_unique_entity_recall"] == 1
            # ASQA answer occurrence cannot establish complete answer correctness.
        except ValueError as exc:
            status = "missing_answer_grouping" if str(exc) == "missing_answer_grouping" else "failed"
    elif prediction.task_track == "short_answer":
        canonical_answer = prediction.answer_bundle.short_answer if prediction.answer_bundle and prediction.answer_bundle.short_answer is not None else prediction.answer
        metrics = best_qa_scores(canonical_answer, answers)
        correct, score, metric = metrics["em"] == 1, metrics["f1"], "f1"
    else:
        status = "not_applicable"
    return {"status": status, "protocol_version": ANSWER_PROTOCOL, "metrics": metrics,
            "answer_behavior": answer_behavior,
            "insufficiency_analysis_eligible": metadata.get("insufficiency_analysis_eligible"),
            "evidence_review_status": metadata.get("evidence_review_status"),
            "expected_behavior": metadata.get("expected_behavior"),
            "reference_protocol": REFERENCE_PROTOCOL if "asqa" in dataset or "qampari" in dataset else None,
            "task_correct": correct, "score": score, "score_metric": metric,
            "correctness_status": "correct_under_task_rule" if correct is True else
            "incorrect_under_task_rule" if correct is False else "undetermined"}
