"""CoVe-style evidence-only verification for attributed predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

from rag_benchmark.evaluation.evidence import (
    EvidenceJudge,
    EvidenceJudgment,
    RuleBasedEvidenceJudge,
)
from rag_benchmark.schemas import (
    ClaimCheckRecord,
    CitationRecord,
    PredictionRecord,
    RevisionDecisionRecord,
    VerificationQuestionRecord,
    VerificationRecord,
)
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.generation.usage import call_text_generator, summarize_calls, failure_usage


@dataclass(frozen=True)
class CoVeVerificationConfig:
    questions_per_answer: int = 5
    revise_answer: bool = True

    def validate(self) -> None:
        if self.questions_per_answer < 1:
            raise ValueError("questions_per_answer must be >= 1")


class CoVeVerifier:
    """Plans verification questions, answers them against evidence, and revises if needed."""

    def __init__(
        self,
        *,
        judge: EvidenceJudge | None = None,
        config: CoVeVerificationConfig | None = None,
    ) -> None:
        self.judge = judge or RuleBasedEvidenceJudge()
        self.config = config or CoVeVerificationConfig()
        self.config.validate()

    def verify(self, prediction: PredictionRecord) -> PredictionRecord:
        claim_checks = prediction.claim_checks[: self.config.questions_per_answer]
        questions = [self._verify_claim(index, claim) for index, claim in enumerate(claim_checks, start=1)]
        failed_claim_ids = [
            question.metadata["claim_id"]
            for question in questions
            if question.status in {"unsupported", "contradicted", "insufficient", "error"}
        ]
        supported_claims = [
            claim
            for claim, question in zip(claim_checks, questions)
            if question.status == "supported"
        ]

        revised_answer = None
        status = "passed" if questions else "not_run"
        if failed_claim_ids:
            if self.config.revise_answer:
                revised_answer = revise_answer_from_supported_claims(supported_claims)
                status = "revised"
            else:
                status = "failed"

        verification = VerificationRecord(
            status=status,
            original_answer=prediction.answer,
            revised_answer=revised_answer,
            questions=questions,
            notes=[
                (f"failed_claim_ids={','.join(failed_claim_ids)}" if failed_claim_ids else "all checked claims passed")
                if questions else "no claims were evaluated"
            ],
            metadata={
                "verifier": "cove_evidence_only",
                "judge": self.judge.name,
                "checked_claim_count": len(questions),
                "failed_claim_ids": failed_claim_ids,
            },
        )
        metrics = {
            **prediction.metrics,
            "verification_question_count": float(len(questions)),
            "verification_failed_claim_count": float(len(failed_claim_ids)),
            "verification_revised": 1.0 if status == "revised" else 0.0,
            "verification_passed": 1.0 if status == "passed" else 0.0,
        }
        updates = {"verification": verification, "metrics": metrics}
        if revised_answer is not None:
            updates["answer"] = revised_answer
            from rag_benchmark.generation.answer_format import build_answer_bundle
            updates["answer_bundle"] = build_answer_bundle(revised_answer, prediction.task_track)
        return prediction.model_copy(update=updates)

    def _verify_claim(self, index: int, claim: ClaimCheckRecord) -> VerificationQuestionRecord:
        evidence_texts = [citation.cited_text or "" for citation in claim.citations if citation.cited_text]
        judgment = self.judge.judge(claim_text=claim.claim_text, evidence_texts=evidence_texts)
        return VerificationQuestionRecord(
            question_id=f"verify-{index}",
            question=f"Is the following claim supported by the cited evidence? {claim.claim_text}",
            answer=judgment.status,
            status=judgment.status,
            citations=claim.citations,
            metadata={
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "judge": self.judge.name,
                "judge_rationale": judgment.rationale,
                "judge_score": judgment.score,
            },
        )


def revise_answer_from_supported_claims(claims: list[ClaimCheckRecord]) -> str:
    texts = [claim.claim_text.strip() for claim in claims if claim.claim_text.strip()]
    if not texts:
        return "I don't know."
    return " ".join(text if text.endswith((".", "!", "?")) else f"{text}." for text in texts)


class VerificationTextGenerator(Protocol):
    provider: str
    model: str

    def generate_text(self, prompt: str) -> str:
        """Return text without sharing state between verification questions."""


class VerificationRetriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Retrieve fresh evidence for one verification question."""


@dataclass(frozen=True)
class FactoredCoVeConfig:
    task_type: str | None = None
    task_revision: dict[str, Any] | None = None
    questions_per_answer: int = 8
    retrieval_top_k: int = 5
    revise_answer: bool = True
    revision_threshold: float = 0.8
    replacement_support_threshold: float = 0.8
    revise_short_answer: bool = False
    allow_abstention: bool = False
    allow_claim_removal: bool = False
    fail_on_generator_error: bool = False
    generator_schema_retries: int = 2
    answer_mode: str = "long"
    minimum_revision_similarity: float = 0.95
    minimum_revision_length_ratio: float = 0.8
    maximum_revision_length_ratio: float = 1.25

    def validate(self) -> None:
        if self.questions_per_answer < 1 or self.retrieval_top_k < 1:
            raise ValueError("questions_per_answer and retrieval_top_k must be >= 1")
        if not 0.0 <= self.revision_threshold <= 1.0:
            raise ValueError("revision_threshold must be between 0 and 1")
        if not 0.0 <= self.replacement_support_threshold <= 1.0:
            raise ValueError("replacement_support_threshold must be between 0 and 1")
        if not 0.0 <= self.minimum_revision_similarity <= 1.0:
            raise ValueError("minimum_revision_similarity must be between 0 and 1")
        if not 0.0 < self.minimum_revision_length_ratio <= self.maximum_revision_length_ratio:
            raise ValueError("revision length ratios are invalid")
        if self.answer_mode not in {"short", "long", "list", "claims"}:
            raise ValueError("answer_mode must be short, long, list, or claims")
        if self.generator_schema_retries < 0:
            raise ValueError("generator_schema_retries must be >= 0")


class FactoredCoVeVerifier:
    """Independent question answering plus confidence-gated claim revision."""

    def __init__(
        self,
        *,
        judge: EvidenceJudge,
        generator: VerificationTextGenerator | None = None,
        retriever: VerificationRetriever | None = None,
        config: FactoredCoVeConfig | None = None,
    ) -> None:
        self.judge = judge
        self.generator = generator
        self.retriever = retriever
        self.config = config or FactoredCoVeConfig()
        self.config.validate()

    def verify(self, prediction: PredictionRecord) -> PredictionRecord:
        from rag_benchmark.verification.task_revision import TASKS, TaskRevisionConfig, TaskRevisionVerifier
        if self.config.task_type in TASKS:
            options = dict(self.config.task_revision or {})
            options.setdefault("policy", "constrained_revision" if self.config.revise_answer else "detect_only")
            options.setdefault("max_items", self.config.questions_per_answer)
            options.setdefault("retrieval_top_k", self.config.retrieval_top_k)
            options.setdefault("allow_abstention", self.config.allow_abstention)
            if options["policy"] != "detect_only" and not self.config.revise_answer:
                raise ValueError("task revision policy conflicts with revise_answer=false")
            if options.get("allow_abstention") and not self.config.allow_abstention:
                raise ValueError("task revision cannot override task abstention prohibition")
            return TaskRevisionVerifier(judge=self.judge, generator=self.generator, retriever=self.retriever,
                config=TaskRevisionConfig(**options)).verify(prediction, task_type=self.config.task_type)
        self._calls = []
        self._failed_calls = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._llm_calls = 0
        self._generator_repairs: list[dict[str, Any]] = []
        self._revision_guard: dict[str, Any] = {"accepted": False, "reason": "not_requested"}
        self._answer_details: dict[str, dict] = {}
        self._short_answer_question = prediction.question if prediction.task_track == "short_answer" else None
        checked_claims = prediction.claim_checks[: self.config.questions_per_answer]
        planned_questions = self._plan_questions(
            checked_claims,
            source_question=prediction.question,
        )
        evidence_sets = self._retrieve_evidence_sets(planned_questions, checked_claims)
        independent_answers = self._answer_questions(planned_questions, evidence_sets)
        judgments = self._judge_texts(
            [claim.claim_text for claim in checked_claims],
            evidence_sets,
        )
        replacement_judgments = self._judge_texts(
            independent_answers,
            evidence_sets,
        )
        questions = []
        decisions = []
        for index, (
            claim,
            question_text,
            evidence,
            independent_answer,
            judgment,
            replacement_judgment,
        ) in enumerate(
            zip(
                checked_claims,
                planned_questions,
                evidence_sets,
                independent_answers,
                judgments,
                replacement_judgments,
            ),
            start=1,
        ):
            questions.append(
                VerificationQuestionRecord(
                    question_id=f"verify-{index}",
                    question=question_text,
                    answer=independent_answer,
                    status=judgment.status,
                    confidence=judgment.score,
                    citations=evidence,
                    metadata={
                        "claim_id": claim.claim_id,
                        "claim_text": claim.claim_text,
                        "judge": self.judge.name,
                        "judge_rationale": judgment.rationale,
                        "replacement_support_status": replacement_judgment.status,
                        "replacement_support_score": replacement_judgment.score,
                        "replacement_support_rationale": replacement_judgment.rationale,
                        "factored": True,
                        "original_answer_visible_to_answerer": False,
                        "source_question_visible_to_answerer": True,
                        "replacement_answer": self._answer_details.get(f"verify-{index}", {}).get("short_answer"),
                        "verification_reason": self._answer_details.get(f"verify-{index}", {}).get("reason"),
                    },
                )
            )
            decisions.append(
                self._decision(
                    claim,
                    judgment,
                    independent_answer,
                    replacement_judgment,
                    answer_details=self._answer_details.get(f"verify-{index}", {}),
                )
            )

        revised_answer = self._apply_decisions(prediction, decisions)
        changed = revised_answer is not None and revised_answer.strip() != prediction.answer.strip()
        failed = [decision.claim_id for decision in decisions if decision.status != "supported"]
        status = "revised" if changed else ("passed" if not failed else "failed") if questions else "not_run"
        verification = VerificationRecord(
            status=status,
            original_answer=prediction.answer,
            revised_answer=revised_answer if changed else None,
            questions=questions,
            decisions=decisions,
            notes=[
                "verification answers were generated without the original answer in context",
                (f"failed_claim_ids={','.join(failed)}" if failed else "all checked claims passed")
                if questions else "no claims were evaluated",
            ],
            metadata={
                "verifier": "factored-cove-v1",
                "revision_policy": "evidence_guarded",
                "judge": self.judge.name,
                "generator": getattr(self.generator, "model", None),
                "targeted_retrieval": self.retriever is not None,
                "revision_threshold": self.config.revision_threshold,
                "replacement_support_threshold": self.config.replacement_support_threshold,
                "checked_claim_count": len(questions),
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "call_usage": summarize_calls(self._calls, self._failed_calls),
                "llm_batch_calls": self._llm_calls,
                "generator_schema_repairs": self._generator_repairs,
                "generator_schema_repair_count": len(self._generator_repairs),
                "answer_mode": self.config.answer_mode,
                "revision_guard": self._revision_guard,
            },
        )
        metrics = {
            **prediction.metrics,
            "verification_question_count": float(len(questions)),
            "verification_failed_claim_count": float(len(failed)),
            "verification_high_confidence_revision_count": float(
                sum(decision.action != "keep" for decision in decisions)
            ),
            "verification_revised": 1.0 if changed else 0.0,
            "verification_passed": 1.0 if status == "passed" else 0.0,
            "verification_generator_schema_repair_count": float(len(self._generator_repairs)),
        }
        updates: dict[str, Any] = {"verification": verification, "metrics": metrics}
        if changed and revised_answer is not None:
            updates["answer"] = revised_answer
            updated_claims = self._apply_claim_decisions(
                prediction.claim_checks,
                decisions=decisions,
                evidence_sets=evidence_sets,
            )
            updates["claim_checks"] = updated_claims
            updates["citations"] = [
                citation for claim in updated_claims for citation in claim.citations
            ]
            if prediction.answer_bundle is not None:
                if prediction.task_track == "short_answer":
                    updates["answer_bundle"] = prediction.answer_bundle.model_copy(
                        update={"short_answer": revised_answer, "long_answer": None,
                                "abstained": revised_answer == "UNKNOWN"}
                    )
                else:
                    updates["answer_bundle"] = prediction.answer_bundle.model_copy(
                        update={"long_answer": revised_answer}
                    )
        return prediction.model_copy(update=updates)

    def _plan_questions(
        self,
        claims: list[ClaimCheckRecord],
        *,
        source_question: str,
    ) -> list[str]:
        defaults = [
            f"For the question {source_question!r}, what evidence determines whether this fact is correct: "
            f"{claim.claim_text}"
            for claim in claims
        ]
        if self.generator is None or not claims:
            return defaults
        prompt = (
            "Write one open-ended key question for each claim below. Questions must not "
            "suggest that a claim is true and must be answerable without seeing the original response. "
            "Use the source question to disambiguate entities, titles, dates, and scope. "
            "Return only valid JSON as {\"questions\":[{\"claim_id\":\"...\",\"question\":\"...\"}]}.\n\n"
            f"Source question: {source_question}\n"
            f"Claims: {json.dumps([{'claim_id': claim.claim_id, 'claim': claim.claim_text} for claim in claims], ensure_ascii=False)}\n\nJSON:"
        )
        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(self.config.generator_schema_retries + 1):
            try:
                payload = extract_json(self._generate(current_prompt))
                if len(claims) == 1 and payload.get("question"):
                    return [str(payload["question"]).strip() or defaults[0]]
                rows = payload.get("questions", [])
                by_id = {
                    str(row.get("claim_id")): str(row.get("question", "")).strip()
                    for row in rows
                    if isinstance(row, dict)
                }
                missing_ids = [claim.claim_id for claim in claims if not by_id.get(claim.claim_id)]
                if missing_ids:
                    raise ValueError(
                        "batched verification plan omitted claim ids: " + ",".join(missing_ids)
                    )
                return [by_id[claim.claim_id] for claim in claims]
            except Exception as error:
                last_error = error
                if attempt >= self.config.generator_schema_retries:
                    break
                self._generator_repairs.append(
                    {
                        "stage": "plan_questions",
                        "attempt": attempt + 1,
                        "error": str(error),
                    }
                )
                current_prompt = (
                    prompt
                    + "\n\nYour previous response failed validation: "
                    + str(error)
                    + "\nReturn a complete corrected JSON object containing every required claim_id."
                )
        if self.config.fail_on_generator_error and last_error is not None:
            raise last_error
        return defaults

    def _retrieve_evidence(
        self,
        question: str,
        claim: ClaimCheckRecord,
    ) -> list[CitationRecord]:
        if self.retriever is None:
            return list(claim.citations)
        results = self.retriever.retrieve(question, top_k=self.config.retrieval_top_k)
        return [
            CitationRecord(
                citation_id=f"verify-{claim.claim_id}-{rank}",
                claim_id=claim.claim_id,
                chunk_id=result.chunk.chunk_id,
                cited_text=result.chunk.text,
                metadata={
                    "verification_retrieval_rank": rank,
                    "verification_retrieval_score": result.score,
                    "title": result.chunk.title,
                },
            )
            for rank, result in enumerate(results, start=1)
        ]

    def _retrieve_evidence_sets(
        self,
        questions: list[str],
        claims: list[ClaimCheckRecord],
    ) -> list[list[CitationRecord]]:
        if not claims:
            return []
        if self.retriever is None:
            return [list(claim.citations) for claim in claims]
        retrieve_many = getattr(self.retriever, "retrieve_many", None)
        if not callable(retrieve_many):
            return [
                self._retrieve_evidence(question, claim)
                for question, claim in zip(questions, claims)
            ]
        result_sets = retrieve_many(questions, top_k=self.config.retrieval_top_k)
        if len(result_sets) != len(claims):
            raise ValueError("verification retrieve_many returned an unexpected number of result sets")
        return [
            [
                CitationRecord(
                    citation_id=f"verify-{claim.claim_id}-{rank}",
                    claim_id=claim.claim_id,
                    chunk_id=result.chunk.chunk_id,
                    cited_text=result.chunk.text,
                    metadata={
                        "verification_retrieval_rank": rank,
                        "verification_retrieval_score": result.score,
                        "title": result.chunk.title,
                    },
                )
                for rank, result in enumerate(results, start=1)
            ]
            for claim, results in zip(claims, result_sets)
        ]

    def _answer_questions(
        self,
        questions: list[str],
        evidence_sets: list[list[CitationRecord]],
    ) -> list[str]:
        if self.generator is None:
            return ["EVIDENCE_REVIEW_ONLY"] * len(questions)
        items = [
            {
                "question_id": f"verify-{index}",
                "question": question,
                "evidence": [citation.cited_text or "" for citation in evidence],
            }
            for index, (question, evidence) in enumerate(zip(questions, evidence_sets), start=1)
        ]
        prompt = (
            "Answer each verification question independently using only that item's evidence. You have not "
            "seen the original answer. If evidence is insufficient, use exactly UNKNOWN. Return only valid "
            "JSON as {\"answers\":[{\"question_id\":\"verify-1\",\"answer\":\"...\","
            "\"short_answer\":null,\"reason\":\"...\"}]}. Put only the evidence-derived factual answer "
            "in answer and keep its explanation separately in reason. "
            + (("For the source short-answer question below, also put the minimal replacement answer span in "
                "short_answer (no explanation, citation, or restatement); use null if no replacement span follows "
                "from this item's evidence. Source question: " + self._short_answer_question + "\n")
               if getattr(self, '_short_answer_question', None) else "")
            + "\n"
            f"Items: {json.dumps(items, ensure_ascii=False)}\n\nJSON:"
        )
        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(self.config.generator_schema_retries + 1):
            try:
                raw = self._generate(current_prompt).strip()
                if len(questions) == 1:
                    try:
                        payload = extract_json(raw)
                    except Exception:
                        return [raw or "UNKNOWN"]
                else:
                    payload = extract_json(raw)
                rows = payload.get("answers", [])
                self._answer_details = {
                    str(row.get("question_id")): {
                        "short_answer": row.get("short_answer") if isinstance(row.get("short_answer"), str) else None,
                        "reason": row.get("reason") if isinstance(row.get("reason"), str) else None}
                    for row in rows if isinstance(row, dict)}
                by_id = {
                    str(row.get("question_id")): str(row.get("answer", "")).strip()
                    for row in rows
                    if isinstance(row, dict)
                }
                missing_ids = [
                    f"verify-{index}"
                    for index in range(1, len(questions) + 1)
                    if not by_id.get(f"verify-{index}")
                ]
                if missing_ids:
                    raise ValueError(
                        "batched verification answers omitted question ids: " + ",".join(missing_ids)
                    )
                return [by_id[f"verify-{index}"] for index in range(1, len(questions) + 1)]
            except Exception as error:
                last_error = error
                if attempt >= self.config.generator_schema_retries:
                    break
                self._generator_repairs.append(
                    {
                        "stage": "answer_questions",
                        "attempt": attempt + 1,
                        "error": str(error),
                    }
                )
                current_prompt = (
                    prompt
                    + "\n\nYour previous response failed validation: "
                    + str(error)
                    + "\nReturn a complete corrected JSON object containing every required question_id."
                )
        if self.config.fail_on_generator_error and last_error is not None:
            raise last_error
        return ["UNKNOWN"] * len(questions)

    def _judge_texts(
        self,
        texts: list[str],
        evidence_sets: list[list[CitationRecord]],
    ) -> list[EvidenceJudgment]:
        rows = [
            (
                text,
                [citation.cited_text or "" for citation in evidence],
            )
            for text, evidence in zip(texts, evidence_sets)
        ]
        judge_many = getattr(self.judge, "judge_many", None)
        if callable(judge_many):
            return judge_many(rows)
        return [
            self.judge.judge(claim_text=text, evidence_texts=evidence_texts)
            for text, evidence_texts in rows
        ]

    def _generate(self, prompt: str) -> str:
        if self.generator is None:
            return ""
        self._llm_calls += 1
        try:
            response = call_text_generator(self.generator, prompt)
        except Exception as exc:
            self._failed_calls.append(failure_usage(exc))
            exc._rag_benchmark_usage = summarize_calls(self._calls, self._failed_calls)
            raise
        self._calls.append(response)
        self._input_tokens += response.input_tokens
        self._output_tokens += response.output_tokens
        return response.text

    def _decision(
        self,
        claim: ClaimCheckRecord,
        judgment: EvidenceJudgment,
        independent_answer: str,
        replacement_judgment: EvidenceJudgment,
        answer_details: dict | None = None,
    ) -> RevisionDecisionRecord:
        candidate = (answer_details or {}).get("short_answer")
        if not isinstance(candidate, str) or not candidate.strip() or candidate.strip().casefold() not in independent_answer.casefold():
            candidate = None
        action = "keep"
        revised_text = None
        reason = "claim retained because no high-confidence contradiction was established"
        if (
            self.config.revise_answer
            and judgment.status == "contradicted"
            and judgment.score >= self.config.revision_threshold
        ):
            if (
                independent_answer.casefold() not in {"unknown", "evidence_review_only"}
                and replacement_judgment.status == "supported"
                and replacement_judgment.score >= self.config.replacement_support_threshold
            ):
                action = "replace"
                revised_text = independent_answer.strip()
                reason = (
                    "high-confidence contradiction with an independently derived, "
                    "evidence-supported replacement"
                )
            elif (
                self.config.allow_claim_removal
                and independent_answer.casefold() in {"unknown", "evidence_review_only"}
            ):
                action = "remove"
                reason = "high-confidence contradiction without a supported replacement"
            elif independent_answer.casefold() in {"unknown", "evidence_review_only"}:
                reason = "claim retained because no evidence-supported replacement was available"
            else:
                reason = "candidate replacement failed the independent evidence-support gate"
        return RevisionDecisionRecord(
            claim_id=claim.claim_id,
            action=action,
            original_text=claim.claim_text,
            revised_text=revised_text,
            reason=reason,
            confidence=judgment.score,
            status=judgment.status,
            metadata={
                "replacement_support_status": replacement_judgment.status,
                "replacement_support_score": replacement_judgment.score,
                "replacement_answer": candidate.strip() if candidate else None,
                "verification_reason": (answer_details or {}).get("reason"),
            },
        )

    def _apply_decisions(
        self,
        prediction: PredictionRecord,
        decisions: list[RevisionDecisionRecord],
    ) -> str | None:
        changed = [decision for decision in decisions if decision.action != "keep"]
        if not changed:
            return None
        if self.config.answer_mode in {"list", "claims"}:
            self._revision_guard = {
                "accepted": False,
                "reason": f"answer_mode_{self.config.answer_mode}_is_structurally_locked",
            }
            return None
        if prediction.task_track == "short_answer":
            if not self.config.revise_short_answer:
                return None
            replacement = next(
                (decision.metadata.get("replacement_answer") for decision in changed
                 if decision.action == "replace" and decision.metadata.get("replacement_answer")),
                None,
            )
            if replacement:
                self._revision_guard = {"accepted": True, "reason": "short_answer_replacement"}
                return replacement.strip()
            if self.config.allow_abstention and any(d.action == "abstain" for d in changed):
                return "UNKNOWN"
            self._revision_guard = {"accepted": False, "reason": "missing_minimal_replacement_answer"}
            return None
        return self._revise_long_answer(prediction.answer, changed)

    def _revise_long_answer(
        self,
        original_answer: str,
        decisions: list[RevisionDecisionRecord],
    ) -> str | None:
        if self.generator is None:
            self._revision_guard = {"accepted": False, "reason": "no_revision_generator"}
            return None
        corrections = [
            {
                "claim_id": decision.claim_id,
                "original_claim": decision.original_text,
                "action": decision.action,
                "replacement": decision.revised_text,
            }
            for decision in decisions
        ]
        prompt = (
            "Revise the original answer minimally using only the high-confidence corrections below. "
            "Preserve all unaffected wording, ordering, and existing citation markers. Do not add facts, "
            "explanations, or citation identifiers. Return only valid JSON as {\"answer\":\"...\"}.\n\n"
            f"Original answer: {original_answer}\n"
            f"Corrections: {json.dumps(corrections, ensure_ascii=False)}\n\nJSON:"
        )
        try:
            revised = str(extract_json(self._generate(prompt)).get("answer", "")).strip()
        except Exception as exc:
            self._revision_guard = {
                "accepted": False,
                "reason": f"revision_generation_error:{type(exc).__name__}",
            }
            return None
        accepted, guard = validate_long_answer_revision(
            original_answer,
            revised,
            minimum_similarity=self.config.minimum_revision_similarity,
            minimum_length_ratio=self.config.minimum_revision_length_ratio,
            maximum_length_ratio=self.config.maximum_revision_length_ratio,
        )
        self._revision_guard = guard
        return revised if accepted else None

    @staticmethod
    def _apply_claim_decisions(
        claims: list[ClaimCheckRecord],
        *,
        decisions: list[RevisionDecisionRecord],
        evidence_sets: list[list[CitationRecord]],
    ) -> list[ClaimCheckRecord]:
        evidence_by_claim = {
            decision.claim_id: evidence
            for decision, evidence in zip(decisions, evidence_sets)
        }
        decision_by_claim = {decision.claim_id: decision for decision in decisions}
        updated = []
        for claim in claims:
            decision = decision_by_claim.get(claim.claim_id)
            if decision is None or decision.action == "keep":
                updated.append(claim)
                continue
            if decision.action == "remove":
                continue
            if decision.action == "replace" and decision.revised_text:
                citations = [
                    citation.model_copy(update={"claim_id": claim.claim_id})
                    for citation in evidence_by_claim.get(claim.claim_id, [])
                ]
                updated.append(
                    claim.model_copy(
                        update={
                            "claim_text": decision.revised_text,
                            "citations": citations,
                            "status": "unjudged",
                            "judge": {},
                        }
                    )
                )
        return updated


def validate_long_answer_revision(
    original: str,
    revised: str,
    *,
    minimum_similarity: float,
    minimum_length_ratio: float,
    maximum_length_ratio: float,
) -> tuple[bool, dict[str, Any]]:
    if not revised:
        return False, {"accepted": False, "reason": "empty_revision"}
    original_tokens = re.findall(r"\w+|[^\w\s]", original, flags=re.UNICODE)
    revised_tokens = re.findall(r"\w+|[^\w\s]", revised, flags=re.UNICODE)
    length_ratio = len(revised_tokens) / max(len(original_tokens), 1)
    similarity = SequenceMatcher(None, original.casefold(), revised.casefold()).ratio()
    original_citations = set(re.findall(r"\[[^\[\]]+\]", original))
    revised_citations = set(re.findall(r"\[[^\[\]]+\]", revised))
    new_citations = sorted(revised_citations - original_citations)
    reason = "accepted"
    accepted = True
    if not minimum_length_ratio <= length_ratio <= maximum_length_ratio:
        accepted = False
        reason = "length_ratio_outside_guardrail"
    elif similarity < minimum_similarity:
        accepted = False
        reason = "similarity_below_guardrail"
    elif new_citations:
        accepted = False
        reason = "revision_introduced_new_citation_ids"
    return accepted, {
        "accepted": accepted,
        "reason": reason,
        "similarity": similarity,
        "length_ratio": length_ratio,
        "new_citation_ids": new_citations,
    }


def extract_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload
