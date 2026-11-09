"""Exact, experiment-scoped reuse of completed independent evaluations.

The database stores judgments, not prediction rows or verification-controller
decisions. Keys include the actual stage inputs and evaluator implementation and
request settings. INSERT OR IGNORE preserves the first valid result. A shared
in-process key lock also coalesces concurrent worker requests.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from rag_benchmark.stage_cache import canonical_hash


CACHE_PROTOCOL = "independent-evaluation-exact-v1"
_LOCKS_GUARD = Lock()
_LOCKS = {}


def _source_hash(value):
    module = inspect.getmodule(value)
    path = getattr(module, "__file__", None)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


def component_identity(component):
    """Fingerprint request-affecting settings without serializing credentials.

    Source hashes cover the prompt builder and parser even when a protocol name
    was not bumped. The text backend source covers the actual system message and
    model-specific parameter dispatch. Endpoint identity is hashed, not exposed.
    """
    generator = getattr(component, "generator", None)
    settings = None
    if generator is not None:
        from rag_benchmark.generation import backends
        settings = {k: getattr(generator, k, None) for k in (
            "provider", "model", "reasoning_effort", "max_output_tokens", "temperature", "top_p", "seed")}
        client = getattr(generator, "client", None)
        settings.update(implementation=_source_hash(type(generator)),
                        parameter_dispatch=_source_hash(backends),
                        endpoint_hash=canonical_hash(str(getattr(client, "base_url", "default"))))
    module = inspect.getmodule(type(component))
    policies = {name: getattr(module, name) for name in ("SEMANTIC_POLICY", "SEMANTIC_POLICY_VERSION")
                if module is not None and hasattr(module, name)}
    return {"type": f"{type(component).__module__}.{type(component).__qualname__}",
            "declared": getattr(component, "identity", None),
            "implementation": _source_hash(type(component)), "policies": policies,
            "generator": settings}


def exact_key(stage, component, inputs):
    return canonical_hash({"namespace": CACHE_PROTOCOL, "stage": stage,
                           "component": component_identity(component), "inputs": inputs})


def zero_reused_usage(value):
    """Keep logical trace/output while removing reused physical billing usage."""
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in {"usage", "failed_usage"} and isinstance(item, dict):
                value[key] = {"input_tokens": 0, "output_tokens": 0, "attempts": 0,
                              "attempt_count": 0, "usage_source": "independent_evaluation_cache",
                              "unknown_attempt_usage": False}
            else:
                value[key] = zero_reused_usage(item)
    elif isinstance(value, list):
        value = [zero_reused_usage(item) for item in value]
    return value


class ExperimentEvaluationCache:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS evaluation_entries "
                               "(protocol TEXT NOT NULL, stage TEXT NOT NULL, input_key TEXT NOT NULL, "
                               "payload TEXT NOT NULL, origin TEXT NOT NULL, "
                               "PRIMARY KEY(protocol, stage, input_key))")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def guard(self, stage, key):
        with _LOCKS_GUARD:
            return _LOCKS.setdefault((str(self.path), stage, key), Lock())

    def get(self, stage, key):
        with self._connection() as connection:
            row = connection.execute("SELECT payload, origin FROM evaluation_entries "
                                     "WHERE protocol=? AND stage=? AND input_key=?",
                                     (CACHE_PROTOCOL, stage, key)).fetchone()
        return {"payload": json.loads(row[0]), "origin": json.loads(row[1])} if row else None

    def put_first(self, stage, key, payload, origin):
        # Only callers with a validated completed stage may publish. A competing
        # process cannot replace the first result with another model sample.
        with self._connection() as connection:
            cursor = connection.execute("INSERT OR IGNORE INTO evaluation_entries VALUES (?, ?, ?, ?, ?)",
                (CACHE_PROTOCOL, stage, key, json.dumps(payload, ensure_ascii=False),
                 json.dumps(origin, ensure_ascii=False)))
            inserted = cursor.rowcount == 1
        return {**self.get(stage, key), "inserted": inserted}
