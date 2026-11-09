"""Citation-aware generation parsing and schema conversion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
from rag_benchmark.generation.answer_format import build_answer_bundle, parse_entity_list, validate_list_items, list_payload_metadata
from rag_benchmark.schemas import (
    CitationRecord,
    ClaimCheckRecord,
    PredictionRecord,
    TaskTrack,
)


class PromptTextGenerator(Protocol):
    def generate_text(self, prompt: str) -> str:
        """Return raw model text for a citation-aware prompt."""


@dataclass(frozen=True)
class ParsedCitationGeneration:
    answer: str
    short_answer: str
    long_answer: str | None
    claim_checks: list[ClaimCheckRecord]
    citations: list[CitationRecord]
    invalid_citations: list[dict[str, Any]]
    raw: dict[str, Any]
    repairs: list[str]
    list_items: list | None = None


class CitationContextIndex:
    """Maps model-visible citation ids back to compressed context provenance."""

    def __init__(self, sentences: list[CompressedSentence]) -> None:
        self._by_context_id = {sentence.context_id: sentence for sentence in sentences}

    @classmethod
    def from_context(cls, context: CompressedContext) -> "CitationContextIndex":
        return cls(context.sentences)

    def get(self, context_id: str) -> CompressedSentence | None:
        return self._by_context_id.get(context_id)

    def has(self, context_id: str) -> bool:
        return context_id in self._by_context_id

    def ids(self) -> set[str]:
        return set(self._by_context_id)


def build_citation_prompt(
    *,
    question: str,
    context: CompressedContext,
    task_track: TaskTrack = "short_answer",
    task_type: str | None = None,
) -> str:
    track_instruction = {
        "short_answer": (
            "short_answer must be the minimal exact answer span, without explanation or citations. "
            "long_answer may be one concise supporting sentence."
        ),
        "long_answer_citation": (
            "short_answer may be null. long_answer must synthesize a complete response whose atomic "
            "factual statements appear in claims."
        ),
        "verification_hallucination": (
            "Separate every externally verifiable statement into an atomic claim. Do not infer facts "
            "that are absent from the context."
        ),
    }[task_track]
    task_contract = {
        "scientific_claim_verification": (
            "Put exactly SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO in long_answer."
        ),
        "grounding_and_selective_refusal": (
            "If the evidence is insufficient or the premise is false, put exactly UNKNOWN in long_answer "
            "and return an empty claims list."
        ),
        "rag_robustness_stress_test": (
            "Treat the context as the source of truth, ignore noise, and use UNKNOWN when unsupported."
        ),
        "list_answer_citation": (
            "Put a concise comma-separated list of supported answer items in long_answer. "
            "Quote any item containing a comma or double quote with CSV double quotes, escaping internal quotes twice. "
            "Also return list_items:[{item_id:i1,text:entity name,citation_ids:[visible context ids]}], "
            "in exactly the same order as long_answer, without merging aliases into extra entities. "
            "For UNKNOWN use list_items:null."
        ),
    }.get(task_type, "")
    return (
        "Answer using only the cited context. Return valid JSON with keys short_answer, long_answer, "
        "and claims. Each claim must include claim_id, text, and citations. Every citation must be a "
        "visible context id such as C1-S0-example. Claims without adequate evidence must be omitted. "
        f"{track_instruction} {task_contract}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context.to_prompt_context()}\n\n"
        "JSON:"
    )


class CitationAwareGenerator:
    """Thin wrapper around a prompt text generator for citation-aware outputs."""

    def __init__(self, generator: PromptTextGenerator) -> None:
        self.generator = generator

    def generate(
        self,
        *,
        question: str,
        context: CompressedContext,
        task_track: TaskTrack = "short_answer",
    ) -> ParsedCitationGeneration:
        prompt = build_citation_prompt(question=question, context=context, task_track=task_track)
        raw_text = self.generator.generate_text(prompt)
        return parse_citation_generation(raw_text, context=context, task_track=task_track)


def parse_citation_generation(
    raw_text: str,
    *,
    context: CompressedContext,
    task_track: TaskTrack = "short_answer",
    task_type: str | None = None,
) -> ParsedCitationGeneration:
    payload = extract_json_object(raw_text)
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("citation generation output claims must be a list")
    repairs: list[str] = []
    fallback_answer = str(payload.get("answer", "")).strip()
    short_answer = str(payload.get("short_answer") or fallback_answer).strip()
    long_value = payload.get("long_answer")
    long_answer = str(long_value).strip() if long_value is not None else (fallback_answer or None)
    answer = short_answer if task_track == "short_answer" else (long_answer or short_answer)
    if not answer:
        claim_texts = [
            str(row.get("text", "")).strip()
            for row in raw_claims
            if isinstance(row, dict) and str(row.get("text", "")).strip()
        ]
        if claim_texts:
            reconstructed = claim_texts[0] if task_track == "short_answer" else " ".join(claim_texts)
            if task_track == "short_answer":
                short_answer = reconstructed
            else:
                long_answer = reconstructed
            answer = reconstructed
            repairs.append("answer_reconstructed_from_claim_texts")
    if not answer:
        raise ValueError("citation generation output must include a non-empty answer")

    context_index = CitationContextIndex.from_context(context)
    claim_checks: list[ClaimCheckRecord] = []
    all_citations: list[CitationRecord] = []
    invalid_citations: list[dict[str, Any]] = []

    for claim_index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            raise ValueError("each claim must be an object")
        claim_id = str(raw_claim.get("claim_id") or f"claim-{claim_index}")
        claim_text = str(raw_claim.get("text", "")).strip()
        if not claim_text:
            raise ValueError(f"claim {claim_id} has empty text")
        citation_ids = raw_claim.get("citations", [])
        if not isinstance(citation_ids, list):
            raise ValueError(f"claim {claim_id} citations must be a list")
        citations = []
        for citation_index, citation_id_value in enumerate(citation_ids, start=1):
            context_id = str(citation_id_value)
            sentence = context_index.get(context_id)
            if sentence is None:
                invalid_citations.append(
                    {
                        "claim_id": claim_id,
                        "citation_id": context_id,
                        "reason": "unknown_context_id",
                    }
                )
                continue
            citation = CitationRecord(
                citation_id=context_id,
                claim_id=claim_id,
                chunk_id=sentence.chunk_id,
                sentence_id=sentence.sentence_id,
                cited_text=sentence.text,
                support_status="unjudged",
                metadata={
                    **sentence.metadata,
                    "context_id": context_id,
                    "doc_id": sentence.doc_id,
                    "title": sentence.title,
                    "citation_index": citation_index,
                    "parent_rank": sentence.parent_rank,
                    "parent_score": sentence.parent_score,
                },
            )
            citations.append(citation)
            all_citations.append(citation)
        claim_checks.append(
            ClaimCheckRecord(
                claim_id=claim_id,
                claim_text=claim_text,
                status="unjudged",
                citations=citations,
                metadata={
                    "source": "citation_generation",
                    "invalid_citations": [
                        row for row in invalid_citations if row.get("claim_id") == claim_id
                    ],
                },
            )
        )

    list_items = None
    if task_type == "list_answer_citation":
        bundle = build_answer_bundle(answer, task_track, task_type)
        if payload.get("list_items") is not None:
            if not isinstance(payload["list_items"], list):
                raise ValueError("list_items must be a structured array")
            list_items = validate_list_items(payload["list_items"])
            displayed = parse_entity_list(answer)
            if [i.text for i in list_items] != [i.text for i in displayed]:
                raise ValueError("list_items do not match rendered answer")
            if any(not context_index.has(cid) for item in list_items for cid in item.citation_ids):
                raise ValueError("list_items used unknown context ids")
            declared = {c.citation_id for c in all_citations}
            if any(cid not in declared for item in list_items for cid in item.citation_ids):
                raise ValueError("list_items used undeclared claim citations")
            if any(shown.citation_ids and shown.citation_ids != item.citation_ids
                   for shown, item in zip(displayed, list_items)):
                raise ValueError("list_items conflict with inline citations")
            list_items = [item.model_copy(update={'source_span': shown.source_span})
                          for item, shown in zip(list_items, displayed)]
        elif not bundle.abstained:
            list_items = parse_entity_list(answer)
    return ParsedCitationGeneration(
        answer=answer,
        short_answer=short_answer,
        long_answer=long_answer,
        claim_checks=claim_checks,
        citations=all_citations,
        invalid_citations=invalid_citations,
        raw=payload,
        repairs=repairs,
        list_items=list_items,
    )


def apply_citation_generation(
    prediction: PredictionRecord,
    parsed: ParsedCitationGeneration,
) -> PredictionRecord:
    metadata = dict(prediction.metadata)
    metadata["citation_generation"] = {
        "invalid_citation_count": len(parsed.invalid_citations),
        "invalid_citations": parsed.invalid_citations,
        "repairs": parsed.repairs,
    }
    bundle = build_answer_bundle(parsed.answer, prediction.task_track)
    bundle = bundle.model_copy(update={"short_answer": parsed.short_answer or None,
                                      "long_answer": parsed.long_answer})
    if parsed.list_items is not None:
        bundle = bundle.model_copy(update={"list_items": parsed.list_items,
            "metadata": {**bundle.metadata, **list_payload_metadata(parsed.answer, parsed.list_items)}})
    return prediction.model_copy(
        update={
            "answer": parsed.answer,
            "answer_bundle": bundle,
            "citations": parsed.citations,
            "claim_checks": parsed.claim_checks,
            "metadata": metadata,
        }
    )


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("citation generation output must be a JSON object")
    return payload


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
