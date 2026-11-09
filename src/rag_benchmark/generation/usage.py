"""Structured, per-call usage shared by generation, rewriting and verification."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TextResponse:
    text: str
    input_tokens: int
    output_tokens: int
    usage_source: str
    attempts: int = 1
    finish_reason: str | None = None
    # Failed transport attempts do not provide authoritative billing usage.
    unknown_attempt_usage: bool = False
    attempt_records: tuple[dict, ...] = ()

    def usage_record(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "text"}


def response_from_api(response: Any, *, messages: list[dict], model: str) -> TextResponse:
    from rag_benchmark.generation.backends import estimate_tokens
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = getattr(response, "usage", None)
    input_count = getattr(usage, "prompt_tokens", None)
    output_count = getattr(usage, "completion_tokens", None)
    actual = input_count is not None and output_count is not None
    attempts = int(getattr(response, "_rag_benchmark_attempt_count", 1) or 1)
    return TextResponse(text, int(input_count) if actual else estimate_tokens(
        "\n".join(str(m.get("content", "")) for m in messages), model=model),
        int(output_count) if actual else (estimate_tokens(text, model=model) if text else 0),
        "api_response" if actual else "estimated", attempts,
        getattr(choice, "finish_reason", None), attempts > 1,
        tuple(getattr(response, "_rag_benchmark_attempts", ())))


def failure_usage(exc):
    return getattr(exc, "_rag_benchmark_usage",
                   {"usage_source": "unavailable", "unknown_attempt_usage": True})


def call_text_generator(generator: Any, prompt: str) -> TextResponse:
    """Adapt local generators while leaving API usage unset."""
    from rag_benchmark.generation.backends import estimate_tokens
    method = getattr(generator, "generate_text_result", None)
    if callable(method):
        return method(prompt)
    text = generator.generate_text(prompt)
    model = getattr(generator, "model", None)
    return TextResponse(text, estimate_tokens(prompt, model=model),
                        estimate_tokens(text, model=model) if text else 0, "estimated")


def summarize_calls(calls: list[TextResponse], failed_calls=None) -> dict:
    failed_calls = failed_calls or []
    sources = {call.usage_source for call in calls}
    return {"input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "usage_source": next(iter(sources)) if len(sources) == 1 else "mixed" if sources else "unavailable",
            "attempt_count": sum(c.attempts for c in calls) + sum(c.get("attempt_count", 1) for c in failed_calls),
            "unknown_attempt_usage": bool(failed_calls) or any(c.unknown_attempt_usage for c in calls),
            "failed_calls": failed_calls,
            "calls": [c.usage_record() for c in calls]}
