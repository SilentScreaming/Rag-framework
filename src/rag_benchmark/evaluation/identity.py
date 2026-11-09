"""Strict sample identities shared by writers, replay and statistical consumers.

Records without a complete identity can still be read, but are excluded from pairing and replay.
Never infer seed/version from the current command or silently use blank keys.
"""
from dataclasses import dataclass, asdict
import json

IDENTITY_PROTOCOL = "evaluation-identity-v1"


@dataclass(frozen=True)
class EvaluationIdentity:
    dataset: str
    data_version: str
    split: str
    seed: int
    context_set_id: str | None = None

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (self.dataset, self.data_version, self.split)):
            raise ValueError("incomplete evaluation identity: dataset/data_version/split required")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("incomplete evaluation identity: integer seed required")
        if self.context_set_id is not None and (not isinstance(self.context_set_id, str) or not self.context_set_id):
            raise ValueError("invalid context_set_id")

    def metadata(self):
        return {**asdict(self), "identity_protocol": IDENTITY_PROTOCOL}

    def key(self, example_id):
        namespace = json.dumps([self.dataset, self.data_version, self.split], separators=(",", ":"))
        return namespace, example_id, self.context_set_id or "", str(self.seed)


def prediction_identity(row):
    m = row.metadata
    if m.get("identity_protocol") != IDENTITY_PROTOCOL or "context_set_id" not in m:
        raise ValueError("incomplete evaluation identity; this record cannot be used for pairing or replay")
    identity = EvaluationIdentity(m.get("dataset"), m.get("data_version"), m.get("split"), m.get("seed"), m["context_set_id"])
    nested = m.get("context_set", {}).get("context_set_id")
    if nested is not None and nested != identity.context_set_id:
        raise ValueError("conflicting context identities")
    condition = m.get("metric_strata", {}).get("context_condition")
    if identity.context_set_id is None and condition not in (None, "", "default"):
        raise ValueError("condition-specific record requires a concrete context_set_id")
    return identity


def validate_example_identity(row, example, *, data_version=None, seed=None, context_set_id=...):
    identity = prediction_identity(row)
    if (row.example_id != example.example_id or row.question != example.question
            or identity.dataset != example.dataset or identity.split != example.split):
        raise ValueError("prediction/example identity mismatch")
    if data_version is not None and identity.data_version != data_version:
        raise ValueError("data version mismatch")
    if seed is not None and identity.seed != seed:
        raise ValueError("seed mismatch")
    if context_set_id is not ... and identity.context_set_id != context_set_id:
        raise ValueError("context set mismatch")
    return identity
