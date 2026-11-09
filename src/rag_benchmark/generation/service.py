"""Citation generation orchestration, independent of datasets, scoring and storage."""
from __future__ import annotations
import json
import time

from rag_benchmark.generation.backends import GenerationResult, estimate_cost
from rag_benchmark.generation.citation import build_citation_prompt, parse_citation_generation
from rag_benchmark.generation.usage import call_text_generator, summarize_calls, failure_usage


def local_citation_json(*, context):
    if not context.sentences:
        return json.dumps({"short_answer": "I don't know.", "long_answer": None, "claims": []})
    first = context.sentences[0]
    return json.dumps({"short_answer": first.text.strip(), "long_answer": first.text.strip(),
        "claims": [{"claim_id": "claim-1", "text": first.text.strip(), "citations": [first.context_id]}]})


def generate_citation_answer(*, question, context, task_track, task_type, generator,
                             model, strict, input_price=None, output_price=None):
    prompt = build_citation_prompt(question=question, context=context, task_track=task_track, task_type=task_type)
    started = time.perf_counter()
    calls = []
    attempt_prompt = prompt
    parse_error = None
    used_fallback = schema_failure = False
    failure_kind = None
    attempts = 3 if generator is not None else 1
    for attempt in range(1, attempts + 1):
        if generator is not None:
            try:
                response = call_text_generator(generator, attempt_prompt)
            except Exception as exc:
                failed = failure_usage(exc)
                known = summarize_calls(calls)
                exc._rag_benchmark_usage = {**known,
                    "attempt_count": known["attempt_count"] + failed.get("attempt_count", 1),
                    "failed_calls": [failed], "unknown_attempt_usage": True,
                    "usage_scope": "known_successes_only_plus_unknown_failed_attempts"}
                raise
            calls.append(response)
            raw = response.text
        else:
            raw = local_citation_json(context=context)
        try:
            if generator is not None and response.finish_reason in {"length", "content_filter"}:
                raise ValueError("incomplete_generation_output")
            parsed = parse_citation_generation(raw, context=context, task_track=task_track, task_type=task_type)
            if parsed.invalid_citations:
                raise ValueError("citation generation used unknown context ids")
            break
        except (ValueError, TypeError, KeyError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            if attempt == attempts or (generator is not None and response.finish_reason in {"length", "content_filter"}):
                schema_failure = True
                failure_kind = ("truncated" if generator is not None and response.finish_reason == "length" else
                                "filtered" if generator is not None and response.finish_reason == "content_filter" else "format")
                raw = json.dumps({"answer": "I don't know.", "claims": []}) if strict else local_citation_json(context=context)
                parsed = parse_citation_generation(raw, context=context, task_track=task_track, task_type=task_type)
                used_fallback = True
                break
            attempt_prompt = (prompt + "\n\nThe previous JSON failed validation: " + parse_error
                + ". Return corrected complete JSON. Use only these context ids: "
                + ", ".join(s.context_id for s in context.sentences) + "\nCorrected JSON:")
    usage = summarize_calls(calls)
    cost = estimate_cost(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                         input_cost_per_1m_tokens=input_price, output_cost_per_1m_tokens=output_price)
    result = GenerationResult(answer=parsed.answer, generation_ms=(time.perf_counter() - started) * 1000,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"], estimated_cost=cost,
        generator={"provider": getattr(generator, "provider", "custom") if generator is not None else "local_citation",
            "model": model or "local-citation-v0", **usage,
            "output_format": "citation_json", "max_output_tokens": getattr(generator, "max_output_tokens", None),
            "cost_source": "configured" if cost is not None else "none", "schema_failure": schema_failure,
            "failure_kind": failure_kind})
    return result, parsed, {"parse_error": parse_error, "used_fallback": used_fallback,
        "schema_failure": schema_failure, "attempt_count": usage["attempt_count"],
        "format_attempt_count": attempt}
