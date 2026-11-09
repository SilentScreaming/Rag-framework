"""Dataset semantic views with source labels separated from model inputs."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
from rag_benchmark.stage_cache import canonical_hash

GARAGE_SEMANTICS = "garage-semantics-v1"


def with_semantic_reviews(experiment, path):
    if path is None:
        return experiment
    if experiment.dataset.name != "garage" or not Path(path).is_file():
        raise ValueError("semantic reviews require GaRAGe and an existing overlay file")
    return experiment.model_copy(update={"dataset": experiment.dataset.model_copy(update={
        "processing": {**experiment.dataset.processing, "semantic_reviews_path": str(Path(path).resolve())}})})


def garage_semantics(row):
    raw = {k: row.get(k) for k in ("sample_id", "evidence_relevant", "evidence_correct",
           "answer_generate", "answer_related_info", "answer_validate", "question_valid", "question_date")}
    direct = "ANSWER-THE-QUESTION" in [str(x).strip().upper() for x in row.get("evidence_correct", [])]
    return {"expected_behavior": "answer" if direct else "unresolved",
            "semantics_protocol": GARAGE_SEMANTICS, "raw_annotation": raw,
            "raw_annotation_hash": canonical_hash(raw),
            "behavior_basis": "explicit_answer_evidence" if direct else "absence_of_direct_evidence_is_not_abstention_gold",
            "human_review_status": "not_reviewed"}


def find_raw(processed_root, filename):
    root = Path(processed_root).resolve()
    for parent in (root, *list(root.parents)[:3]):
        path = parent / "raw" / filename
        if path.is_file():
            return path
    return None


def correct_garage_views(processed_root, examples, contexts, reviews=None):
    if not any(e.dataset == "garage" for e in examples.values()):
        return examples, contexts
    raw_path = find_raw(processed_root, "GaRAGe_benchmark.jsonl")
    raw_rows = {}
    if raw_path:
        with raw_path.open(encoding="utf-8") as handle:
            raw_rows = {str(r["sample_id"]): r for r in (json.loads(s) for s in handle if s.strip())}
    corrected, corrected_contexts = dict(examples), dict(contexts)
    for key, example in examples.items():
        if example.dataset != "garage":
            continue
        raw = raw_rows.get(str(example.metadata.get("source_id")))
        if raw is None:
            if example.metadata.get("semantics_protocol") == GARAGE_SEMANTICS:
                continue
            meta = {"expected_behavior": "unresolved", "semantics_protocol": GARAGE_SEMANTICS,
                    "behavior_basis": "raw_annotation_unavailable", "human_review_status": "not_reviewed"}
        else:
            meta = garage_semantics(raw)
        review = (reviews or {}).get(key)
        if review:
            if review.get("raw_annotation_hash") != meta.get("raw_annotation_hash") or review.get("expected_behavior") not in {"answer", "abstain", "unresolved"} or not review.get("reviewer") or not review.get("reason"):
                raise ValueError(f"invalid semantic review for {key}")
            if str(review["reviewer"]).strip().lower().startswith("model:"):
                raise ValueError("model-only semantic candidates cannot be activated as human reviews")
            meta.update(expected_behavior=review["expected_behavior"], human_review_status="reviewed", human_review=review)
        answers = [{**a, "type": "long", "metadata": {**a.get("metadata", {}), "original_type": a.get("type")}}
                   for a in example.answers]
        corrected[key] = replace(example, answers=answers, metadata={**example.metadata, **meta})
        corrected_contexts[key] = [replace(c, expected_behavior=meta["expected_behavior"],
            metadata={**c.metadata, "semantics_protocol": GARAGE_SEMANTICS, "behavior_basis": meta["behavior_basis"]})
            for c in contexts.get(key, [])]
    return corrected, corrected_contexts
