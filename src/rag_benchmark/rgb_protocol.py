"""Neutral RGB model inputs and a scoped, source-preserving evidence review."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from rag_benchmark.stage_cache import canonical_hash

RGB_INPUT_PROTOCOL = "rgb-neutral-inputs-v1"


def rgb_display_title(chunk_id: str) -> str:
    return "Passage " + sha256(chunk_id.encode("utf-8")).hexdigest()[:12]


def neutral_rgb_chunk(chunk):
    if chunk.dataset != "rgb_en":
        return chunk
    return replace(chunk, title=rgb_display_title(chunk.chunk_id),
                   metadata={**chunk.metadata, "input_display_protocol": RGB_INPUT_PROTOCOL})


def rgb_review_identity(question, chunks):
    return canonical_hash({"question": question,
                           "evidence": [{"chunk_id": chunk.chunk_id, "text": chunk.text} for chunk in chunks]})


def apply_rgb_protocol(dataset_slice, review_overlay=None):
    """Call once on the loaded slice, before encoding/search/query rewriting.

    The original source labels and expected_behavior are kept. The reviewed
    insufficiency cohort is a distinct metadata flag, not a replacement benchmark.
    """
    if dataset_slice.dataset_name != "rgb_en":
        return dataset_slice
    corpus = [neutral_rgb_chunk(chunk) for chunk in dataset_slice.corpus]
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    examples = {example.example_id: example for example in dataset_slice.examples}
    reviews = (review_overlay or {}).get("reviews", {})
    contexts = {}
    for example_id, sets in dataset_slice.context_sets_by_example.items():
        updated = []
        for context in sets:
            metadata = {**context.metadata, "input_display_protocol": RGB_INPUT_PROTOCOL}
            if context.metadata.get("positive_count") == 0:
                metadata.update(insufficiency_analysis_eligible=False, evidence_review_status="not_reviewed")
                review = reviews.get(context.context_set_id)
                if review is not None:
                    chunks = [by_id[key] for key in context.chunk_ids]
                    if review["input_hash"] != rgb_review_identity(examples[example_id].question, chunks):
                        raise ValueError("rgb_review_evidence_mismatch")
                    decision = review["decision"]
                    if decision not in {"answerable", "insufficient", "ambiguous"}:
                        raise ValueError("invalid_rgb_evidence_review")
                    metadata.update(insufficiency_analysis_eligible=decision == "insufficient",
                                    evidence_review_status=decision, evidence_review=review,
                                    evidence_review_protocol=(review_overlay or {}).get("schema_version"))
            updated.append(replace(context, metadata=metadata))
        contexts[example_id] = updated
    return replace(dataset_slice, corpus=corpus, context_sets_by_example=contexts)
