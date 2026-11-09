"""Text-prompt backend shared by citation generation, rewriting and verification."""
from __future__ import annotations
from pathlib import Path
from rag_benchmark.generation.backends import (
    create_chat_completion_with_retries, load_openai_local_config, resolve_openai_ca_bundle, openai_tls_verify,
    require_openai_api_key, resolve_openai_model,
)

class OpenAITextPromptGenerator:
    """Raw prompt generator for query rewriting and citation-aware JSON output."""

    provider = "openai"

    def __init__(self, *, repo_root: Path, model: str | None = None, max_output_tokens: int = 512,
                 reasoning_effort: str | None = None, max_attempts: int = 4, timeout_seconds: float | None = None) -> None:
        from openai import OpenAI

        config = load_openai_local_config(repo_root)
        api_key = require_openai_api_key()
        selected_model = resolve_openai_model(repo_root, model, config=config)
        self.model = str(selected_model)
        # None uses the provider's default. Explicit "none" remains a distinct
        # evaluator configuration and retains a different cache identity.
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.max_output_tokens = max_output_tokens
        ca_bundle = resolve_openai_ca_bundle(repo_root=repo_root, config=config)
        if ca_bundle:
            import httpx

            self.client = OpenAI(api_key=api_key, max_retries=0, http_client=httpx.Client(verify=openai_tls_verify(ca_bundle)))
        else:
            self.client = OpenAI(api_key=api_key, max_retries=0)
        if timeout_seconds is not None:
            self.client = self.client.with_options(timeout=timeout_seconds)
        self.last_attempt_count = 1

    def generate_text(self, prompt: str) -> str:
        return self.generate_text_result(prompt).text

    def generate_text_result(self, prompt: str):
        from rag_benchmark.generation.usage import response_from_api
        messages = [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ]
        response = create_chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=messages,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=getattr(self, "reasoning_effort", None),
            max_attempts=getattr(self, "max_attempts", 4),
        )
        self.last_response_model = getattr(response, "model", self.model)
        self.last_attempt_count = int(getattr(response, "_rag_benchmark_attempt_count", 1) or 1)
        return response_from_api(response, messages=messages, model=self.model)

