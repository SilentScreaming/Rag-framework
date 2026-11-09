"""Deterministic stage artifacts for replayable benchmark runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StageArtifact:
    stage: str
    example_id: str
    method: str
    input_hash: str
    output_hash: str
    payload: JsonObject
    metadata: JsonObject
    path: Path


class StageArtifactCache:
    """Stores one canonical JSON artifact per example and stage."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, *, example_id: str, stage: str, input_value: Any) -> StageArtifact | None:
        path = self.path_for(example_id=example_id, stage=stage)
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(row, dict) or row.get("schema_version") != "stage-artifact-v1"
                or row.get("stage") != stage or row.get("example_id") != example_id):
            raise ValueError("stage artifact identity or schema does not match the requested cache entry")
        expected_hash = canonical_hash(input_value)
        if row.get("input_hash") != expected_hash:
            return None
        payload = row.get("payload")
        if not isinstance(payload, dict) or canonical_hash(payload) != row.get("output_hash"):
            raise ValueError("stage artifact content does not match its fingerprint")
        return StageArtifact(
            stage=str(row["stage"]),
            example_id=str(row["example_id"]),
            method=str(row["method"]),
            input_hash=str(row["input_hash"]),
            output_hash=str(row["output_hash"]),
            payload=payload,
            metadata=dict(row.get("metadata", {})),
            path=path,
        )

    def save(
        self,
        *,
        example_id: str,
        stage: str,
        method: str,
        input_value: Any,
        payload: JsonObject,
        metadata: JsonObject | None = None,
    ) -> StageArtifact:
        path = self.path_for(example_id=example_id, stage=stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": "stage-artifact-v1",
            "stage": stage,
            "example_id": example_id,
            "method": method,
            "input_hash": canonical_hash(input_value),
            "output_hash": canonical_hash(payload),
            "payload": payload,
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return StageArtifact(
            stage=stage,
            example_id=example_id,
            method=method,
            input_hash=row["input_hash"],
            output_hash=row["output_hash"],
            payload=payload,
            metadata=dict(row["metadata"]),
            path=path,
        )

    def path_for(self, *, example_id: str, stage: str) -> Path:
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", example_id).strip("-")[:48] or "example"
        suffix = hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:10]
        return self.root / stage / f"{readable}-{suffix}.json"
