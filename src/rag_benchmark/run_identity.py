"""Content fingerprints for run identity and compatible artifact reuse."""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from rag_benchmark.generation.backends import load_openai_local_config, resolve_openai_model
from rag_benchmark.indexing.dense_cache import corpus_fingerprint
from rag_benchmark.retrieval.candidate_cache import FrozenCandidateCache
from rag_benchmark.stage_cache import canonical_hash


def source_fingerprint(repo_root: Path) -> str:
    """Hash the active package, even when configuration lives in another workspace."""
    digest = sha256()
    sources = (("rag_benchmark", Path(__file__).resolve().parent),
               ("scripts", Path(repo_root) / "scripts"))
    for namespace, root in sources:
        for path in sorted(root.rglob("*.py")):
            digest.update(f"{namespace}/{path.relative_to(root).as_posix()}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_identity(*, repo_root, experiment, dataset_slice, settings) -> dict:
    component = asdict(settings.component_settings)
    # Fingerprint effective settings without including credentials.
    local = load_openai_local_config(repo_root) if component["generator_kind"] == "openai" else {}
    if component["generator_kind"] == "openai":
        component["generator_model"] = resolve_openai_model(repo_root, component.get("generator_model"), config=local)
    for key, local_key in (("input_cost_per_1m_tokens", "input_cost_per_1m_tokens"),
                           ("output_cost_per_1m_tokens", "output_cost_per_1m_tokens")):
        if component.get(key) is None:
            component[key] = local.get(local_key)
    if dataset_slice.candidate_cache_path:
        cache = FrozenCandidateCache(dataset_slice.candidate_cache_path)
        try:
            corpus_hash = cache.scope_hash
        finally:
            cache.close()
    else:
        corpus_hash = corpus_fingerprint(dataset_slice.corpus)
    asset_manifests = {}
    roots = [settings.component_settings.dense_index_path, dataset_slice.candidate_cache_path]
    for root in roots:
        if root is not None:
            manifest = Path(root) / "manifest.json"
            asset_manifests[str(manifest)] = sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else "missing"
    generation = {
        "schema_version": "generation-identity-v1",
        "implementation": source_fingerprint(repo_root),
        "experiment": experiment.to_json_dict(), "components": json.loads(json.dumps(component, default=str)),
        "corpus_hash": corpus_hash, "candidate_scope": dataset_slice.candidate_scope,
        "asset_manifests": asset_manifests,
        "examples": [asdict(e) for e in dataset_slice.examples],
        "contexts": {key: [asdict(c) for c in value] for key, value in dataset_slice.context_sets_by_example.items()},
        "split": settings.split_profile, "seed": settings.seed,
        "context_condition": settings.context_condition, "context_metadata": settings.context_metadata,
        "paired_artifacts": ({"protocol": "paired-prefix-v1", "root": str(settings.paired_artifact_root.resolve())}
                             if settings.paired_artifact_root else None),
        "drafts": {key: value.to_json_dict() for key, value in (settings.draft_predictions or {}).items()},
    }
    generation_hash = canonical_hash(generation)
    evaluation_hash = canonical_hash({
        "generation": generation_hash, "evaluator_version": "evaluation-v1",
        "qrels": {key: [asdict(q) for q in value] for key, value in dataset_slice.qrels_by_example.items()},
        "annotations": {key: [asdict(a) for a in value] for key, value in dataset_slice.annotations_by_example.items()},
        "metric_ks": settings.retrieval_metric_ks, "bootstrap_samples": settings.bootstrap_samples,
        "gold_relevance_threshold": settings.gold_relevance_threshold,
    })
    return {"schema_version": "run-identity-v1", "generation": generation_hash,
            "evaluation": evaluation_hash}
