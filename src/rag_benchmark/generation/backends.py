"""Generator backends shared by smoke and formal execution."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_benchmark.retrieval.bm25 import RetrievedChunk


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    generation_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float | None
    generator: dict[str, Any]


def estimate_tokens(text: str, *, model: str | None = None) -> int:
    # The offline paths must not fetch tokenizer assets. Their token counts are
    # deliberately approximate; API-backed runs may use a model tokenizer.
    if not model or model in {"mock", "mock-v0", "extractive", "first-context-sentence-v0", "example-model"}:
        return max(1, len(text.split()))
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text.split()))


def create_chat_completion_with_retries(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_attempts: int = 4,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> Any:
    """Retry transient OpenAI API failures without hiding configuration errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    # One retry owner, including when callers supply an SDK client.
    if hasattr(client, "with_options"):
        client = client.with_options(max_retries=0)
    delay_seconds = 1.0
    ledger = []
    from rag_benchmark.generation.budget import CURRENT_BUDGET
    budget = CURRENT_BUDGET.get()
    for attempt in range(1, max_attempts + 1):
        reservation = None
        if budget is not None:
            if max_output_tokens is None:
                raise ValueError("budgeted calls require an explicit output cap")
            reservation = budget.reserve(input_tokens=estimate_tokens(str(messages), model=model),
                                         output_cap=max_output_tokens, model=model)
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                kwargs["max_completion_tokens"] = max_output_tokens
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            # Reasoning-capable endpoints can reject sampling controls when
            # a reasoning effort is explicitly selected.
            if reasoning_effort is not None and reasoning_effort != "none":
                kwargs.pop("temperature", None)
            response = client.chat.completions.create(
                **kwargs,
            )
            if budget is not None:
                budget.finish(reservation, response)
            ledger.append({"attempt": attempt, "status": "completed"})
            try:
                setattr(response, "_rag_benchmark_attempt_count", attempt)
                setattr(response, "_rag_benchmark_attempts", ledger)
            except Exception:
                pass
            return response
        except Exception as exc:
            if budget is not None:
                budget.finish(reservation)
            ledger.append({"attempt": attempt, "status": "failed", "error_type": type(exc).__name__,
                           "http_status": getattr(exc, "status_code", None), "usage_source": "unavailable"})
            setattr(exc, "_rag_benchmark_usage", {"attempt_count": attempt, "attempts": ledger.copy(),
                    "input_tokens": None, "output_tokens": None, "usage_source": "unavailable",
                    "unknown_attempt_usage": True})
            if attempt >= max_attempts or not is_retryable_openai_error(exc):
                raise
            headers = getattr(getattr(exc, "response", None), "headers", {})
            try:
                retry_after = float(headers.get("retry-after", "nan"))
            except (TypeError, ValueError):
                from email.utils import parsedate_to_datetime
                try:
                    retry_after = parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time()
                except (KeyError, TypeError, ValueError, OverflowError):
                    retry_after = float("nan")
            if retry_after > 30:
                raise  # defer rather than retry earlier than the server requested
            time.sleep(max(delay_seconds, retry_after) if retry_after >= 0 else delay_seconds)
            delay_seconds = min(delay_seconds * 2, 15.0)
    raise RuntimeError("unreachable retry loop exit")


def is_retryable_openai_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code") in {"insufficient_quota", "billing_hard_limit_reached"}:
            return False
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (isinstance(status_code, int) and status_code >= 500):
        return True
    error_name = type(exc).__name__.lower()
    return any(fragment in error_name for fragment in ("timeout", "connection", "rate", "server"))


def load_openai_local_config(repo_root: Path) -> dict[str, Any]:
    """Read supported non-secret settings; credentials are environment-only."""
    path = repo_root / "configs" / "openai.local.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configs/openai.local.json must contain an object")
    supported = {"model", "input_cost_per_1m_tokens", "output_cost_per_1m_tokens", "ca_bundle"}
    return {name: value for name, value in payload.items() if name in supported}


def resolve_openai_model(repo_root: Path, model: str | None = None, *,
                         config: dict[str, Any] | None = None) -> str:
    """Resolve an explicit model, then local settings, then OPENAI_MODEL."""
    local = load_openai_local_config(repo_root) if config is None else config
    selected = model or local.get("model") or os.environ.get("OPENAI_MODEL")
    if not isinstance(selected, str) or not selected.strip():
        raise RuntimeError("OpenAI model not set. Use --model, local model settings, or OPENAI_MODEL")
    return selected.strip()


def require_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment")
    return key


class BaseGenerator:
    provider = "base"
    model = "base"

    def generate(
        self,
        *,
        question: str,
        contexts: list[RetrievedChunk],
        pipeline: str,
        task_track: str = "short_answer",
        task_type: str | None = None,
    ) -> GenerationResult:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    provider = "mock"
    model = "mock-v0"

    def generate(
        self, *, question: str, contexts: list[RetrievedChunk], pipeline: str, task_track: str = "short_answer", task_type: str | None = None
    ) -> GenerationResult:
        start = time.perf_counter()
        answer = "I don't know."
        prompt = build_prompt(question=question, contexts=contexts, pipeline=pipeline, task_track=task_track, task_type=task_type)
        return GenerationResult(
            answer=answer,
            generation_ms=(time.perf_counter() - start) * 1000,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(answer),
            estimated_cost=None,
            generator={
                "provider": self.provider,
                "model": self.model,
                "usage_source": "estimated",
                "cost_source": "none",
            },
        )


class ExtractiveGenerator(BaseGenerator):
    provider = "local_extractive"
    model = "first-context-sentence-v0"

    def generate(
        self, *, question: str, contexts: list[RetrievedChunk], pipeline: str, task_track: str = "short_answer", task_type: str | None = None
    ) -> GenerationResult:
        start = time.perf_counter()
        prompt = build_prompt(question=question, contexts=contexts, pipeline=pipeline, task_track=task_track, task_type=task_type)
        if contexts:
            sentences = contexts[0].chunk.sentences
            answer = str(sentences[0]["text"]) if sentences else contexts[0].chunk.text.split(".")[0]
        else:
            answer = "I don't know."
        return GenerationResult(
            answer=answer.strip(),
            generation_ms=(time.perf_counter() - start) * 1000,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(answer),
            estimated_cost=None,
            generator={
                "provider": self.provider,
                "model": self.model,
                "usage_source": "estimated",
                "cost_source": "none",
            },
        )


class OpenAIGenerator(BaseGenerator):
    provider = "openai"

    def __init__(
        self,
        *,
        model: str,
        input_cost_per_1m_tokens: float | None = None,
        output_cost_per_1m_tokens: float | None = None,
        ca_bundle: str | None = None,
        cost_source: str = "none",
        max_output_tokens: int = 256,
    ) -> None:
        from openai import OpenAI

        api_key = require_openai_api_key()
        self.model = model
        self.input_cost_per_1m_tokens = input_cost_per_1m_tokens
        self.output_cost_per_1m_tokens = output_cost_per_1m_tokens
        self.ca_bundle = ca_bundle
        self.cost_source = cost_source
        self.max_output_tokens = max_output_tokens
        if ca_bundle:
            import httpx

            self.client = OpenAI(api_key=api_key, max_retries=0, http_client=httpx.Client(verify=openai_tls_verify(ca_bundle)))
        else:
            self.client = OpenAI(api_key=api_key, max_retries=0)

    def generate(
        self, *, question: str, contexts: list[RetrievedChunk], pipeline: str, task_track: str = "short_answer", task_type: str | None = None
    ) -> GenerationResult:
        prompt = build_prompt(
            question=question,
            contexts=contexts,
            pipeline=pipeline,
            task_track=task_track,
            task_type=task_type,
        )
        system_instruction = task_system_instruction(task_track=task_track, task_type=task_type)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        start = time.perf_counter()
        response = create_chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=messages,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        answer = response.choices[0].message.content or ""
        generation_ms = (time.perf_counter() - start) * 1000
        from rag_benchmark.generation.usage import response_from_api
        measured = response_from_api(response, messages=messages, model=self.model)
        input_tokens, output_tokens = measured.input_tokens, measured.output_tokens
        usage_source = measured.usage_source
        estimated_cost = estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1m_tokens=self.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=self.output_cost_per_1m_tokens,
        )
        return GenerationResult(
            answer=answer.strip(),
            generation_ms=generation_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            generator={
                "provider": self.provider,
                "model": self.model,
                "usage_source": usage_source,
                "output_format": "plain",
                "max_output_tokens": self.max_output_tokens,
                "calls": [measured.usage_record()],
                "unknown_attempt_usage": measured.unknown_attempt_usage,
                "schema_failure": measured.finish_reason in {"length", "content_filter"},
                "finish_reason": measured.finish_reason,
                "cost_source": self.cost_source if estimated_cost is not None else "none",
                "ca_bundle": self.ca_bundle,
                "attempt_count": int(getattr(response, "_rag_benchmark_attempt_count", 1) or 1),
            },
        )


def build_generator(
    kind: str,
    *,
    repo_root: Path,
    model: str | None = None,
    input_cost_per_1m_tokens: float | None = None,
    output_cost_per_1m_tokens: float | None = None,
    max_output_tokens: int = 256,
) -> BaseGenerator:
    if kind == "mock":
        return MockGenerator()
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "openai":
        config = load_openai_local_config(repo_root)
        selected_model = resolve_openai_model(repo_root, model, config=config)
        ca_bundle = resolve_openai_ca_bundle(repo_root=repo_root, config=config)
        input_cost = (
            input_cost_per_1m_tokens
            if input_cost_per_1m_tokens is not None
            else config.get("input_cost_per_1m_tokens")
        )
        output_cost = (
            output_cost_per_1m_tokens
            if output_cost_per_1m_tokens is not None
            else config.get("output_cost_per_1m_tokens")
        )
        if input_cost_per_1m_tokens is not None or output_cost_per_1m_tokens is not None:
            cost_source = "cli_price_config" if input_cost is not None and output_cost is not None else "none"
        else:
            cost_source = "local_price_config" if input_cost is not None and output_cost is not None else "none"
        return OpenAIGenerator(
            model=selected_model,
            input_cost_per_1m_tokens=input_cost,
            output_cost_per_1m_tokens=output_cost,
            ca_bundle=ca_bundle,
            cost_source=cost_source,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"Unknown generator kind: {kind}")


def resolve_openai_ca_bundle(*, repo_root: Path, config: dict[str, Any]) -> str | None:
    explicit_candidates = [
        config.get("ca_bundle"),
        os.environ.get("OPENAI_CA_BUNDLE"),
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
    ]
    for candidate in explicit_candidates:
        if not candidate:
            continue
        path = _resolve_path(repo_root=repo_root, value=str(candidate))
        if not path.exists():
            raise RuntimeError(f"Configured OpenAI CA bundle does not exist: {path}")
        return str(path)

    return None


def openai_tls_verify(ca_bundle):
    """Combine the configured CA with existing Windows trust; retain TLS checks."""
    if os.name != "nt":
        return ca_bundle or True
    import ssl
    context = ssl.create_default_context()
    if ca_bundle:
        context.load_verify_locations(cafile=ca_bundle)
    return context


def _resolve_path(*, repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_prompt(
    *,
    question: str,
    contexts: list[RetrievedChunk],
    pipeline: str,
    task_track: str = "short_answer",
    task_type: str | None = None,
) -> str:
    answer_instruction = task_system_instruction(task_track=task_track, task_type=task_type)
    if pipeline == "closed_book":
        return f"{answer_instruction}\n\nQuestion: {question}\nAnswer:"
    context_blocks = []
    for result in contexts:
        context_blocks.append(
            result.metadata.get("prepared_prompt_block") or
            f"[{result.rank}] Title: {result.chunk.title or 'Untitled'}\n"
            f"Chunk ID: {result.chunk.chunk_id}\n"
            f"Text: {result.chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "Use the provided evidence as the source of truth. "
        "If it does not establish an answer, return exactly UNKNOWN. "
        f"{answer_instruction}\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )


def task_system_instruction(*, task_track: str, task_type: str | None) -> str:
    if task_type == "scientific_claim_verification":
        return (
            "Classify the scientific claim. Return exactly one label: SUPPORT, CONTRADICT, "
            "or NOT_ENOUGH_INFO. Do not add explanation."
        )
    if task_type == "grounding_and_selective_refusal":
        return (
            "Answer only when the supplied evidence establishes the answer and the premise is valid. "
            "For insufficient evidence or a false premise, return exactly UNKNOWN."
        )
    if task_type == "rag_robustness_stress_test":
        return (
            "Treat the supplied context as the source of truth, reconcile relevant evidence, ignore noise, "
            "and return UNKNOWN when the answer is not established."
        )
    if task_type == "list_answer_citation":
        return ("Return a concise comma-separated list containing only supported answer items. "
                "Put an item containing a comma or double quote in CSV double quotes; escape internal quotes twice. "
                "Do not append explanations or reasons to list items.")
    if task_track == "short_answer":
        return "Return only the shortest exact answer span; do not repeat the question or explain."
    return "Write a complete answer, but include only claims supported by the evidence."


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m_tokens: float | None,
    output_cost_per_1m_tokens: float | None,
) -> float | None:
    if input_cost_per_1m_tokens is None or output_cost_per_1m_tokens is None:
        return None
    return (input_tokens / 1_000_000 * input_cost_per_1m_tokens) + (
        output_tokens / 1_000_000 * output_cost_per_1m_tokens
    )
