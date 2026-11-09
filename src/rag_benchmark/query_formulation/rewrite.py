"""Query formulation stages for config-driven benchmark pipelines."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_benchmark.generation.backends import estimate_cost, estimate_tokens
from rag_benchmark.generation.usage import failure_usage


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


@dataclass(frozen=True)
class QueryFormulationResult:
    original_query: str
    query: str
    method: str
    queries: tuple[str, ...] = ()
    subquestions: tuple[str, ...] = ()
    answer_type: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "query": self.query,
            "method": self.method,
            "queries": list(self.queries or (self.query,)),
            "subquestions": list(self.subquestions),
            "answer_type": self.answer_type,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.estimated_cost,
            "metadata": dict(self.metadata),
        }


class QueryFormulator(Protocol):
    name: str

    def formulate(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryFormulationResult:
        """Return the query that downstream retrieval should use."""


@dataclass(frozen=True)
class IdentityQueryFormulator:
    name: str = "identity"

    def formulate(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryFormulationResult:
        start = time.perf_counter()
        query = question.strip()
        return QueryFormulationResult(
            original_query=question,
            query=query,
            method=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"fallback": False, **(metadata or {})},
        )


@dataclass(frozen=True)
class RuleBasedQueryRewriter:
    """Deterministic local rewrite used for cheap smoke runs and ablations."""

    max_terms: int = 16
    name: str = "rule_rewrite"

    def formulate(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryFormulationResult:
        start = time.perf_counter()
        tokens = TOKEN_PATTERN.findall(question)
        kept: list[str] = []
        seen = set()
        for token in tokens:
            normalized = token.lower()
            if normalized in STOPWORDS and not token[:1].isupper():
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            kept.append(token)
            if len(kept) >= self.max_terms:
                break
        query = " ".join(kept).strip() or question.strip()
        return QueryFormulationResult(
            original_query=question,
            query=query,
            method=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=estimate_tokens(question),
            output_tokens=estimate_tokens(query),
            metadata={
                "fallback": query == question.strip(),
                "max_terms": self.max_terms,
                "kept_terms": kept,
                **(metadata or {}),
            },
        )


from rag_benchmark.generation.usage import call_text_generator


class PromptTextGenerator(Protocol):
    provider: str
    model: str

    def generate_text(self, prompt: str) -> str:
        """Return raw model text for a prompt."""


@dataclass(frozen=True)
class LLMQueryRewriter:
    """LLM-backed query rewriter with safe fallback to the original question."""

    generator: PromptTextGenerator
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    name: str = "llm_rewrite"
    fail_on_error: bool = False

    def formulate(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryFormulationResult:
        prompt = build_query_rewrite_prompt(question)
        input_tokens = estimate_tokens(prompt, model=self.generator.model)
        start = time.perf_counter()
        response = None
        failed_usage = {}
        try:
            response = call_text_generator(self.generator, prompt)
            raw_text = response.text
            latency_ms = (time.perf_counter() - start) * 1000
            query, parsed = parse_query_rewrite_output(raw_text)
            fallback = False
            error = None
        except Exception as exc:
            failed_usage = failure_usage(exc)
            if self.fail_on_error:
                raise
            raw_text = ""
            latency_ms = (time.perf_counter() - start) * 1000
            query = question.strip()
            parsed = {}
            fallback = True
            error = type(exc).__name__
        output_tokens = estimate_tokens(raw_text or query, model=self.generator.model)
        if response is not None:
            input_tokens, output_tokens = response.input_tokens, response.output_tokens
        estimated = estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1m_tokens=self.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=self.output_cost_per_1m_tokens,
        )
        return QueryFormulationResult(
            original_query=question,
            query=query.strip() or question.strip(),
            method=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated if response is not None else None,
            metadata={
                "fallback": fallback,
                "error": error,
                "provider": self.generator.provider,
                "model": self.generator.model,
                "raw": parsed,
                "call_usage": response.usage_record() if response is not None else failed_usage,
                **(metadata or {}),
            },
        )


@dataclass(frozen=True)
class MultiQueryLLMFormulator:
    """Preserves the original query and adds diverse, hop-aware retrieval queries."""

    generator: PromptTextGenerator
    max_queries: int = 4
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    name: str = "multi_query_rewrite"
    fail_on_error: bool = False

    def formulate(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryFormulationResult:
        prompt = build_multi_query_prompt(question=question, max_queries=self.max_queries)
        input_tokens = estimate_tokens(prompt, model=self.generator.model)
        start = time.perf_counter()
        error = None
        raw_text = ""
        response = None
        failed_usage = {}
        try:
            response = call_text_generator(self.generator, prompt)
            raw_text = response.text
            payload = extract_json_object(raw_text)
            generated = payload.get("queries", [])
            if not isinstance(generated, list):
                raise ValueError("queries must be a list")
            subquestions = payload.get("subquestions", [])
            if not isinstance(subquestions, list):
                raise ValueError("subquestions must be a list")
            answer_type = str(payload.get("answer_type", "")).strip() or None
        except Exception as exc:
            failed_usage = failure_usage(exc)
            if self.fail_on_error:
                raise
            payload = {}
            generated = []
            subquestions = []
            answer_type = None
            error = type(exc).__name__
        queries = deduplicate_queries(
            [question, *[str(value) for value in generated], *[str(value) for value in subquestions]],
            limit=self.max_queries,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        output_tokens = estimate_tokens(raw_text or question, model=self.generator.model)
        if response is not None:
            input_tokens, output_tokens = response.input_tokens, response.output_tokens
        return QueryFormulationResult(
            original_query=question,
            query=question.strip(),
            queries=tuple(queries),
            subquestions=tuple(str(value).strip() for value in subquestions if str(value).strip()),
            answer_type=answer_type,
            method=self.name,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_per_1m_tokens=self.input_cost_per_1m_tokens,
                output_cost_per_1m_tokens=self.output_cost_per_1m_tokens,
            ) if response is not None else None,
            metadata={
                "fallback": not bool(generated),
                "error": error,
                "provider": self.generator.provider,
                "model": self.generator.model,
                "raw": payload,
                "call_usage": response.usage_record() if response is not None else failed_usage,
                **(metadata or {}),
            },
        )
def build_query_rewrite_prompt(question: str) -> str:
    return (
        "Rewrite the question as a concise retrieval query for a QA benchmark. "
        "Preserve named entities and the information need. Return only valid JSON "
        'with keys "query" and "rationale".\n\n'
        f"Question: {question}\n\nJSON:"
    )


def build_multi_query_prompt(*, question: str, max_queries: int) -> str:
    return (
        "Create retrieval queries for an open-domain QA system. Preserve every named entity, date, "
        "constraint, and comparison in the original question. For multi-hop questions, add focused "
        "subquestions whose answers can be joined. Do not guess the final answer. Return only valid "
        'JSON with keys "queries", "subquestions", and "answer_type". '
        f"Return at most {max_queries - 1} additional queries because the original question is always kept.\n\n"
        f"Question: {question}\n\nJSON:"
    )


def deduplicate_queries(values: list[str], *, limit: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = " ".join(value.split()).strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def parse_query_rewrite_output(text: str) -> tuple[str, dict[str, Any]]:
    payload = extract_json_object(text)
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("query rewrite output must include a non-empty query")
    return query, payload


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
        raise ValueError("query rewrite output must be a JSON object")
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
