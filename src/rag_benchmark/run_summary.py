"""Summarize observed run usage and answer-format outcomes."""


def summarize_run(predictions):
    calls, formats, outcomes = [], [], {}
    for p in predictions:
        calls.extend(p.metadata.get("execution_budget", {}).get("calls", []))
        if p.generator.get("output_format") == "citation_json":
            runtime = p.metadata.get("citation_generation_runtime", {})
            formats.append({"failed": bool(p.generator.get("schema_failure")) and p.generator.get("failure_kind", "format") == "format",
                            "truncated": p.generator.get("failure_kind") == "truncated",
                            "filtered": p.generator.get("failure_kind") == "filtered",
                            "attempts": runtime.get("format_attempt_count", 1)})
        elif p.answer_bundle and "list_format_status" in p.answer_bundle.metadata:
            formats.append({"failed": p.answer_bundle.metadata.get("list_format_status") == "unavailable",
                            "truncated": p.generator.get("failure_kind") == "truncated",
                            "filtered": p.generator.get("failure_kind") == "filtered", "attempts": 1})
        reason = p.generator.get("finish_reason") or "unspecified"
        outcomes[reason] = outcomes.get(reason, 0) + 1
    complete_lengths = [c["output_tokens"] for c in calls if c.get("output_tokens") is not None and c.get("finish_reason") not in {"length", "content_filter"}]
    return {"examples": len(predictions), "format_required_count": len(formats),
            "format_denominator_scope": "saved_predictions_requiring_citation_json_or_structured_list; API failures tracked separately",
            "first_pass_format_failure_count": sum(r["failed"] or r["attempts"] > 1 for r in formats),
            "first_pass_format_failure_rate": sum(r["failed"] or r["attempts"] > 1 for r in formats)/len(formats) if formats else None,
            "final_format_failure_count": sum(r["failed"] for r in formats),
            "final_format_failure_rate": sum(r["failed"] for r in formats)/len(formats) if formats else None,
            "format_repair_count": sum(max(0,r["attempts"]-1) for r in formats),
            "truncated_answer_count": sum(p.generator.get("failure_kind") == "truncated"
                or p.generator.get("finish_reason") == "length" for p in predictions),
            "filtered_answer_count": sum(r["filtered"] for r in formats),
            "completed_call_output_max": max(complete_lengths) if complete_lengths else None,
            "censored_call_count": sum(c.get("finish_reason") == "length" for c in calls),
            "unknown_call_usage_count": sum(c.get("usage_source") != "api_response" for c in calls),
            "logical_reused_call_count": sum(bool(c.get("reused")) for c in calls),
            "finish_reasons": outcomes}
