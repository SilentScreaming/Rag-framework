"""Evidence consistency evaluation for cited claims."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from rag_benchmark.schemas import ClaimCheckRecord, PredictionRecord, SupportStatus


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
}


@dataclass(frozen=True)
class EvidenceJudgment:
    status: SupportStatus
    rationale: str
    score: float = 0.0


class EvidenceJudge(Protocol):
    name: str

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        """Judge whether evidence supports one claim."""


@dataclass(frozen=True)
class RuleBasedEvidenceJudge:
    """Lightweight deterministic judge for tests and cheap pre-screening."""

    min_claim_token_recall: float = 0.5
    name: str = "rule_token_overlap"

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        evidence_text = " ".join(evidence_texts).strip()
        if not evidence_text:
            return EvidenceJudgment(status="insufficient", rationale="no cited evidence text", score=0.0)
        claim_tokens = content_tokens(claim_text)
        if not claim_tokens:
            return EvidenceJudgment(status="insufficient", rationale="no claim content tokens", score=0.0)
        evidence_tokens = content_tokens(evidence_text)
        overlap = len(claim_tokens & evidence_tokens)
        score = overlap / len(claim_tokens)
        if simple_negation_conflict(claim_text, evidence_text):
            return EvidenceJudgment(status="contradicted", rationale="simple negation conflict", score=score)
        if score >= self.min_claim_token_recall:
            return EvidenceJudgment(status="supported", rationale="token overlap above threshold", score=score)
        return EvidenceJudgment(status="unsupported", rationale="token overlap below threshold", score=score)


@dataclass
class TransformersNLIEvidenceJudge:
    """Lazy NLI attribution judge with per-evidence and joint-evidence aggregation."""

    model_name: str = "cross-encoder/nli-deberta-v3-base"
    revision: str | None = None
    device: str | None = None
    max_length: int = 512
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.65
    cache_dir: str | None = None
    identity_name: str | None = None
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _PIPELINE_CACHE: ClassVar[dict[tuple[Any, ...], tuple[Any, Any]]] = {}

    @property
    def name(self) -> str:
        identity = self.identity_name or self.model_name
        return f"nli:{identity}@{self.revision}" if self.revision else f"nli:{identity}"

    def judge(self, *, claim_text: str, evidence_texts: list[str]) -> EvidenceJudgment:
        return self.judge_many([(claim_text, evidence_texts)])[0]

    def judge_many(
        self,
        rows: list[tuple[str, list[str]]],
    ) -> list[EvidenceJudgment]:
        """Score multiple claims in one model batch without changing per-claim aggregation."""

        flattened_premises = []
        flattened_hypotheses = []
        slices = []
        empty = []
        for claim_text, raw_evidence_texts in rows:
            evidence_texts = [text.strip() for text in raw_evidence_texts if text.strip()]
            if not evidence_texts:
                slices.append((0, 0))
                empty.append(True)
                continue
            premises = list(evidence_texts)
            if len(evidence_texts) > 1:
                premises.append("\n".join(evidence_texts))
            start = len(flattened_premises)
            flattened_premises.extend(premises)
            flattened_hypotheses.extend([claim_text] * len(premises))
            slices.append((start, len(flattened_premises)))
            empty.append(False)
        probabilities = (
            self._predict_pairs(flattened_premises, flattened_hypotheses)
            if flattened_premises
            else []
        )
        judgments = []
        for is_empty, (start, end) in zip(empty, slices):
            if is_empty:
                judgments.append(
                    EvidenceJudgment(status="insufficient", rationale="no cited evidence text", score=0.0)
                )
            else:
                judgments.append(self._aggregate_probabilities(probabilities[start:end]))
        return judgments

    def _aggregate_probabilities(
        self,
        probabilities: list[dict[str, float]],
    ) -> EvidenceJudgment:
        entailment = max(row.get("entailment", 0.0) for row in probabilities)
        contradiction = max(row.get("contradiction", 0.0) for row in probabilities)
        neutral = max(row.get("neutral", 0.0) for row in probabilities)
        if contradiction >= self.contradiction_threshold and contradiction > entailment:
            return EvidenceJudgment(
                status="contradicted",
                rationale=f"NLI contradiction={contradiction:.3f}, entailment={entailment:.3f}",
                score=contradiction,
            )
        if entailment >= self.entailment_threshold:
            return EvidenceJudgment(
                status="supported",
                rationale=f"NLI entailment={entailment:.3f}, contradiction={contradiction:.3f}",
                score=entailment,
            )
        return EvidenceJudgment(
            status="unsupported",
            rationale=(
                f"NLI below calibrated thresholds: entailment={entailment:.3f}, "
                f"contradiction={contradiction:.3f}, neutral={neutral:.3f}"
            ),
            score=max(entailment, contradiction),
        )

    def _predict(self, *, premises: list[str], hypothesis: str) -> list[dict[str, float]]:
        return self._predict_pairs(premises, [hypothesis] * len(premises))

    def _predict_pairs(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[dict[str, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premise and hypothesis batches must have equal length")
        tokenizer, model = self._load()
        import torch

        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            rows = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        labels = normalize_nli_labels(model.config.id2label)
        return [
            {labels.get(index, f"label_{index}"): float(score) for index, score in enumerate(row)}
            for row in rows
        ]

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved_revision = None if Path(self.model_name).exists() else self.revision
        cache_key = (
            str(Path(self.model_name).resolve()) if Path(self.model_name).exists() else self.model_name,
            resolved_revision,
            self.device,
            self.cache_dir,
        )
        cached = self._PIPELINE_CACHE.get(cache_key)
        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=resolved_revision,
                cache_dir=self.cache_dir,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                revision=resolved_revision,
                cache_dir=self.cache_dir,
            )
            if self.device:
                model.to(self.device)
            model.eval()
            cached = (tokenizer, model)
            self._PIPELINE_CACHE[cache_key] = cached
        self._tokenizer, self._model = cached
        return self._tokenizer, self._model


def evaluate_evidence_consistency(
    prediction: PredictionRecord,
    *,
    judge: EvidenceJudge | None = None,
) -> PredictionRecord:
    judge = judge or RuleBasedEvidenceJudge()
    claims = list(prediction.claim_checks)
    judge_many = getattr(judge, "judge_many", None)
    if callable(judge_many):
        judgments = judge_many(
            [
                (
                    claim.claim_text,
                    [citation.cited_text or "" for citation in claim.citations if citation.cited_text],
                )
                for claim in claims
            ]
        )
        judgments = [
            (
                judgment
                if claim.citations
                else EvidenceJudgment(
                    status="unsupported",
                    rationale="claim has no citations",
                    score=0.0,
                )
            )
            for claim, judgment in zip(claims, judgments)
        ]
        updated_claims = [
            apply_evidence_judgment(claim, judge=judge, judgment=judgment)
            for claim, judgment in zip(claims, judgments)
        ]
    else:
        updated_claims = [evaluate_claim_check(claim_check, judge=judge) for claim_check in claims]
    updated_citations = [citation for claim_check in updated_claims for citation in claim_check.citations]
    metrics = {
        **prediction.metrics,
        **evidence_consistency_metrics(updated_claims),
    }
    return prediction.model_copy(
        update={"claim_checks": updated_claims, "citations": updated_citations, "metrics": metrics}
    )


def evaluate_claim_check(claim_check: ClaimCheckRecord, *, judge: EvidenceJudge) -> ClaimCheckRecord:
    if not claim_check.citations:
        return claim_check.model_copy(
            update={
                "status": "unsupported",
                "judge": {"name": judge.name, "rationale": "claim has no citations", "score": 0.0},
            }
        )

    evidence_texts = [citation.cited_text or "" for citation in claim_check.citations if citation.cited_text]
    judgment = judge.judge(claim_text=claim_check.claim_text, evidence_texts=evidence_texts)
    return apply_evidence_judgment(claim_check, judge=judge, judgment=judgment)


def apply_evidence_judgment(
    claim_check: ClaimCheckRecord,
    *,
    judge: EvidenceJudge,
    judgment: EvidenceJudgment,
) -> ClaimCheckRecord:
    updated_citations = [
        citation.model_copy(
            update={
                # A set-level judgment is not a judgment of each member.
                "support_status": (citation.support_status
                    if citation.metadata.get("judgment_scope") == "citation" else "unjudged"),
                "metadata": {
                    **citation.metadata,
                    "evidence_set_judge": judge.name,
                    "evidence_set_status": judgment.status,
                    "evidence_set_score": judgment.score,
                },
            }
        )
        for citation in claim_check.citations
    ]
    return claim_check.model_copy(
        update={
            "status": judgment.status,
            "citations": updated_citations,
            "judge": {
                "name": judge.name,
                "rationale": judgment.rationale,
                "score": judgment.score,
            },
        }
    )


def evidence_consistency_metrics(claim_checks: list[ClaimCheckRecord]) -> dict[str, float]:
    total = len(claim_checks)
    if total == 0:
        return {"claim_count": 0.0}
    supported = sum(1 for claim in claim_checks if claim.status == "supported")
    unsupported = sum(1 for claim in claim_checks if claim.status == "unsupported")
    contradicted = sum(1 for claim in claim_checks if claim.status == "contradicted")
    insufficient = sum(1 for claim in claim_checks if claim.status == "insufficient")
    return {
        "claim_count": float(total),
        "supported_claims": float(supported),
        "unsupported_claims": float(unsupported),
        "contradicted_claims": float(contradicted),
        "insufficient_claims": float(insufficient),
        "claim_support_rate": supported / total,
        "unsupported_claim_rate": unsupported / total,
        "contradiction_rate": contradicted / total,
        "insufficient_claim_rate": insufficient / total,
        "evidence_consistency_score": supported / total,
    }


def content_tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS}


def simple_negation_conflict(claim_text: str, evidence_text: str) -> bool:
    claim_tokens = set(TOKEN_PATTERN.findall(claim_text.lower()))
    evidence_tokens = set(TOKEN_PATTERN.findall(evidence_text.lower()))
    negators = {"not", "never", "no"}
    return bool((claim_tokens & negators) ^ (evidence_tokens & negators)) and bool(
        (claim_tokens - negators) & (evidence_tokens - negators)
    )


def normalize_nli_labels(id2label: dict[int, str] | dict[str, str]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_index, raw_label in id2label.items():
        index = int(raw_index)
        label = str(raw_label).lower()
        if "entail" in label:
            normalized[index] = "entailment"
        elif "contrad" in label:
            normalized[index] = "contradiction"
        elif "neutral" in label:
            normalized[index] = "neutral"
    if len(normalized) < 3 and len(id2label) == 3:
        return {0: "contradiction", 1: "neutral", 2: "entailment"}
    return normalized
