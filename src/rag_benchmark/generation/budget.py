"""Format-aware output limits, bounded by the existing experiment cap."""
from contextlib import contextmanager
from contextvars import ContextVar

CURRENT_BUDGET = ContextVar("rag_execution_budget", default=None)


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    """One ledger per answer, including transport retries and format repairs."""
    def __init__(self, *, max_calls=None, max_total_tokens=None):
        self.max_calls, self.max_total_tokens = max_calls, max_total_tokens
        self.calls = []
        self.stage = "unspecified"
        self.exhausted = False

    def reserve(self, *, input_tokens, output_cap, model):
        from uuid import uuid4
        reservation = input_tokens + output_cap
        used = sum(c["charged_tokens"] for c in self.calls)
        if ((self.max_calls is not None and len(self.calls) >= self.max_calls) or
                (self.max_total_tokens is not None and used + reservation > self.max_total_tokens)):
            self.exhausted = True
            error = BudgetExhausted("cumulative call/token budget exhausted before dispatch")
            error._rag_benchmark_usage = {"usage_source": "not_dispatched", "unknown_attempt_usage": False,
                                         "budget_exhausted": True, "budget_ledger": self.snapshot()}
            raise error
        row = {"model": model, "stage": self.stage, "input_tokens_estimate": input_tokens, "output_cap": output_cap,
               "reserved_tokens": reservation, "charged_tokens": reservation, "status": "reserved",
               "logical_call_id": uuid4().hex}
        self.calls.append(row)
        return row

    def finish(self, row, response=None):
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        known = input_tokens is not None and output_tokens is not None
        row.update(status="completed" if response is not None else "failed",
                   input_tokens=input_tokens, output_tokens=output_tokens,
                   usage_source="api_response" if known else "unavailable",
                   finish_reason=getattr((getattr(response, "choices", None) or [None])[0], "finish_reason", None))
        if known:
            row["charged_tokens"] = input_tokens + output_tokens
        # Keep the full reservation while an attempt's usage is unknown.

    def snapshot(self):
        return {"max_calls": self.max_calls, "max_total_tokens": self.max_total_tokens,
                "exhausted": self.exhausted, "calls": [dict(c) for c in self.calls],
                "unknown_usage": any(c.get("usage_source") != "api_response" for c in self.calls),
                "accounting": "actual_when_available_otherwise_conservative_estimated_reservation"}


@contextmanager
def execution_budget(*, max_calls=None, max_total_tokens=None):
    budget = CallBudget(max_calls=max_calls, max_total_tokens=max_total_tokens)
    token = CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        CURRENT_BUDGET.reset(token)


def set_budget_stage(stage):
    budget = CURRENT_BUDGET.get()
    if budget is not None:
        budget.stage = stage


def record_reused_calls(usage, *, source_run_id):
    """Charge a frozen prefix against method limits without dispatching it again."""
    budget = CURRENT_BUDGET.get()
    if budget is None:
        return
    calls = usage.get("calls", [])
    if not calls and isinstance(usage.get("input_tokens"), int):
        calls = [usage]
    for call in calls:
        inp, out = call.get("input_tokens"), call.get("output_tokens")
        if "charged_tokens" in call:
            charged = int(call["charged_tokens"])
        elif inp is None or out is None:
            charged = int(call.get("reserved_tokens", 0))
        else:
            charged = int(inp) + int(out)
        if ((budget.max_calls is not None and len(budget.calls) >= budget.max_calls) or
                (budget.max_total_tokens is not None and
                 sum(c["charged_tokens"] for c in budget.calls) + charged > budget.max_total_tokens)):
            budget.exhausted = True
            raise BudgetExhausted("reused prefix exceeds complete-method budget")
        budget.calls.append({**call, "stage": call.get("stage", budget.stage), "status": "reused",
            "reused": True, "source_run_id": source_run_id, "charged_tokens": charged,
            "usage_source": call.get("usage_source", usage.get("usage_source", "unavailable"))})


def with_shared_output_cap(experiment, cap):
    """Explicit shared output limit; does not mutate saved dataset budgets."""
    if type(cap) is not int or cap < 1:
        raise ValueError("shared output cap must be a positive integer")
    pipeline = experiment.pipeline
    pipeline = pipeline.model_copy(update={
        "query_formulation": {**pipeline.query_formulation, "max_output_tokens": cap},
        "budget": pipeline.budget.model_copy(update={"max_generation_tokens": cap}),
        "task_profile": pipeline.task_profile.model_copy(update={"max_answer_tokens": cap}),
        "generation": {**pipeline.generation, "shared_output_cap": cap, "max_output_tokens": cap,
                       "output_tokens_by_mode": {"plain": cap, "citation_json": cap}}})
    return experiment.model_copy(update={"pipeline": pipeline})


def output_token_limit(pipeline, mode):
    caps = pipeline.generation.get("output_tokens_by_mode", {})
    if not isinstance(caps, dict) or set(caps) - {"plain", "citation_json"}:
        raise ValueError("output_tokens_by_mode supports plain/citation_json only")
    value = caps.get(mode, pipeline.generation.get("max_output_tokens", pipeline.budget.max_generation_tokens))
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("output token limit must be a positive integer")
    if mode in caps and value > pipeline.budget.max_generation_tokens:
        raise ValueError("explicit format output limit exceeds experiment cap; budget approval required")
    return min(value, pipeline.budget.max_generation_tokens)
