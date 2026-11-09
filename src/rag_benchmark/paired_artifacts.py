"""Exact, experiment-local realizations for controlled component interventions."""
from pathlib import Path

from rag_benchmark.stage_cache import StageArtifactCache, canonical_hash

PROTOCOL = "paired-prefix-v1"


def validate_prefix_anchors(prediction):
    """A resumed full arm must still carry its original frozen realizations."""
    import json
    for stage, anchor in prediction.metadata.get("paired_stage_artifacts", {}).items():
        path = Path(anchor["artifact_path"])
        if not path.is_file():
            raise ValueError(f"missing paired {stage} artifact; do not regenerate a resumed prefix")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("stage") != stage or row.get("example_id") != prediction.example_id or
                row.get("input_hash") != anchor["input_hash"] or
                row.get("output_hash") != anchor["output_hash"] or
                canonical_hash(row.get("payload")) != anchor["output_hash"]):
            raise ValueError(f"changed paired {stage} artifact; preserve the original comparison")


def validate_detection_source(source_predictions, drafts, *, selected_ids, detection_root=None):
    """Verify the saved detection arm before executing its paired edit policy."""
    import json
    from rag_benchmark.evaluation.identity import prediction_identity
    selected = [p for p in source_predictions if p.example_id in selected_ids]
    by_id = {p.example_id: p for p in selected}
    draft_by_id = {p.example_id: p for p in drafts}
    if len(by_id) != len(selected) or set(by_id) != set(selected_ids):
        raise ValueError("paired detection source requires unique matching selected examples")
    for example_id, source in by_id.items():
        draft = draft_by_id.get(example_id)
        if (draft is None or source.question != draft.question or source.task_track != draft.task_track
                or source.verification is None or source.verification.original_answer != draft.answer
                or source.verification.metadata.get("policy") != "detect_only"):
            raise ValueError("paired detection source does not match the frozen draft")
        if prediction_identity(source) != prediction_identity(draft):
            raise ValueError("paired detection source has a different sample/context/seed identity")
        if "detection_artifacts" not in source.verification.metadata:
            raise ValueError("detection source lacks frozen detection records")
        if source.metadata.get("generation_context", {}).get("content_sha256") != draft.metadata.get("generation_context", {}).get("content_sha256"):
            raise ValueError("paired detection source has different draft evidence")
        anchors = source.verification.metadata.get("detection_artifacts", [])
        if not anchors and source.verification.metadata.get("physical_dispatch_count", 0):
            raise ValueError("paired detection source has no saved detection artifacts")
        for anchor in anchors:
            path = Path(anchor["artifact_path"])
            if detection_root and not path.resolve().is_relative_to(Path(detection_root).resolve()):
                raise ValueError("paired detection source uses a different experiment root")
            if not path.is_file():
                raise ValueError("missing paired detection artifact; do not regenerate it")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("stage") != anchor["stage"] or row.get("example_id") != anchor["cache_example_id"]
                    or row.get("input_hash") != anchor["input_hash"]
                    or row.get("output_hash") != anchor["output_hash"]
                    or canonical_hash(row.get("payload")) != anchor["output_hash"]):
                raise ValueError("changed paired detection artifact")


class PairedArtifacts:
    def __init__(self, root: Path, scope: dict):
        self.root = Path(root)
        self.scope = scope

    def _input(self, stage, inputs, config):
        return {"protocol": PROTOCOL, "scope": self.scope, "stage": stage,
                "inputs": inputs, "config": config}

    def load(self, *, example_id, stage, inputs, config):
        identity = self._input(stage, inputs, config)
        artifact = StageArtifactCache(self.root / canonical_hash(identity)).load(
            example_id=example_id, stage=stage, input_value=identity)
        if artifact and canonical_hash(artifact.payload) != artifact.output_hash:
            raise ValueError("paired artifact content does not match its fingerprint")
        return artifact

    def save(self, *, example_id, stage, inputs, config, payload, source_run_id, latency_ms):
        existing = self.load(example_id=example_id, stage=stage, inputs=inputs, config=config)
        if existing is not None:
            return existing
        identity = self._input(stage, inputs, config)
        return StageArtifactCache(self.root / canonical_hash(identity)).save(
            example_id=example_id, stage=stage, method=stage, input_value=identity,
            payload=payload, metadata={"source_run_id": source_run_id,
                                       "logical_latency_ms": latency_ms})
