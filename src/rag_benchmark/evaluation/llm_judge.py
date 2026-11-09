"""LLM and rule-based judge utilities for evidence/citation pre-screening."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_benchmark.evaluation.evidence import (
    EvidenceJudge,
    EvidenceJudgment,
    RuleBasedEvidenceJudge,
    evaluate_claim_check,
    evidence_consistency_metrics,
)
from rag_benchmark.schemas import ClaimCheckRecord, PredictionRecord, SupportStatus


JsonObject = dict[str, Any]


class JudgeTextGenerator(Protocol):
    provider: str
    model: str

    def generate_text(self, prompt: str) -> str:
        """Return raw model output for one judge prompt."""


@dataclass(frozen=True)
class JudgeTrace:
    example_id: str
    pipeline: str
    claim_id: str
    judge: str
    status: SupportStatus
    score: float
    rationale: str
    metadata: JsonObject

    def to_json_dict(self) -> JsonObject:
        return {
            "example_id": self.example_id,
            "pipeline": self.pipeline,
            "claim_id": self.claim_id,
            "judge": self.judge,
            "status": self.status,
            "score": self.score,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class JudgeRunResult:
    prediction: PredictionRecord
    traces: list[JudgeTrace]


@dataclass(frozen=True)
class LLMEvidenceJudge:
    generator: JudgeTextGenerator
    name: str = "llm_evidence_judge"

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        prompt = build_llm_judge_prompt(claim_text=claim_text, evidence_texts=evidence_texts)
        raw_text = self.generator.generate_text(prompt)
        payload = parse_llm_judge_output(raw_text)
        return EvidenceJudgment(
            status=payload["status"],
            rationale=str(payload.get("rationale", "")),
            score=float(payload.get("score", 0.0)),
        )


def judge_prediction(
    prediction: PredictionRecord,
    *,
    judge: EvidenceJudge | None = None,
    only_uncertain: bool = False,
    uncertainty_threshold: float = 0.75,
    pre_judge: EvidenceJudge | None = None,
) -> JudgeRunResult:
    judge = judge or RuleBasedEvidenceJudge()
    pre_judge = pre_judge or RuleBasedEvidenceJudge()
    traces: list[JudgeTrace] = []
    updated_claims: list[ClaimCheckRecord] = []

    for claim in prediction.claim_checks:
        candidate = claim
        used_judge_name = judge.name
        if only_uncertain:
            candidate = evaluate_claim_check(claim, judge=pre_judge)
            score = float(candidate.judge.get("score", 0.0))
            if candidate.status == "supported" and score >= uncertainty_threshold:
                updated_claims.append(candidate)
                traces.append(trace_from_claim(prediction, candidate, judge_name=pre_judge.name, skipped_llm=True))
                continue
        updated = evaluate_claim_check(claim, judge=judge)
        updated_claims.append(updated)
        traces.append(trace_from_claim(prediction, updated, judge_name=used_judge_name, skipped_llm=False))

    updated_citations = [citation for claim in updated_claims for citation in claim.citations]
    metrics = {
        **prediction.metrics,
        **evidence_consistency_metrics(updated_claims),
        **judge_metrics(updated_claims, traces),
    }
    judged = prediction.model_copy(
        update={
            "claim_checks": updated_claims,
            "citations": updated_citations,
            "metrics": metrics,
        }
    )
    return JudgeRunResult(prediction=judged, traces=traces)


def build_llm_judge_prompt(*, claim_text: str, evidence_texts: list[str]) -> str:
    evidence = "\n".join(f"[{index}] {text}" for index, text in enumerate(evidence_texts, start=1))
    return (
        "Judge whether the cited evidence supports the claim. Return only valid JSON with "
        'keys "status", "score", and "rationale". Status must be one of supported, '
        "unsupported, contradicted, or insufficient. Score is a confidence/support score from 0 to 1.\n\n"
        f"Claim: {claim_text}\n\nEvidence:\n{evidence}\n\nJSON:"
    )


def parse_llm_judge_output(text: str) -> JsonObject:
    payload = extract_json_object(text)
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"supported", "unsupported", "contradicted", "insufficient", "error"}:
        raise ValueError(f"unsupported judge status: {status!r}")
    score = float(payload.get("score", 0.0))
    if score < 0.0 or score > 1.0:
        raise ValueError("judge score must be between 0 and 1")
    payload["status"] = status
    payload["score"] = score
    return payload


def extract_json_object(text: str) -> JsonObject:
    cleaned = strip_code_fence(text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge output must be a JSON object")
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


def trace_from_claim(
    prediction: PredictionRecord,
    claim: ClaimCheckRecord,
    *,
    judge_name: str,
    skipped_llm: bool,
) -> JudgeTrace:
    return JudgeTrace(
        example_id=prediction.example_id,
        pipeline=prediction.pipeline,
        claim_id=claim.claim_id,
        judge=judge_name,
        status=claim.status,
        score=float(claim.judge.get("score", 0.0)),
        rationale=str(claim.judge.get("rationale", "")),
        metadata={
            "skipped_llm": skipped_llm,
            "citation_count": len(claim.citations),
        },
    )


def judge_metrics(claims: list[ClaimCheckRecord], traces: list[JudgeTrace]) -> JsonObject:
    llm_skipped = sum(1 for trace in traces if trace.metadata.get("skipped_llm"))
    return {
        "judge_claim_count": float(len(claims)),
        "judge_trace_count": float(len(traces)),
        "judge_llm_skipped_count": float(llm_skipped),
        "judge_llm_called_count": float(len(traces) - llm_skipped),
    }
