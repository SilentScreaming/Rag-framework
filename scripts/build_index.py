"""Build a cached dense retrieval index for a configured dataset slice."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rag_benchmark.component_factory import HashingEmbedder
from rag_benchmark.indexing.dense_cache import build_cached_dense_index
from rag_benchmark.retrieval.dense import SentenceTransformerEmbedder
from rag_benchmark.runner import (
    build_experiment_with_resolved_pipeline,
    context_index_dir,
    load_formal_dataset_slice,
    parse_context_metadata_filters,
    select_context_set,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="alce_asqa")
    parser.add_argument("--pipeline", default="vanilla_rag")
    parser.add_argument("--split-profile", default="mvp")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-relevance-threshold", type=int, default=3)
    parser.add_argument("--embedder", choices=["config", "hashing", "sentence-transformers"], default="config")
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hash-dimensions", type=int, default=128)
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-example-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--search-backend", choices=["auto", "numpy", "faiss"], default="auto")
    parser.add_argument("--exclude-title", action="store_true")
    parser.add_argument("--context-condition", default=None)
    parser.add_argument("--context-metadata", action="append", default=[], metavar="KEY=JSON_VALUE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = build_experiment_with_resolved_pipeline(
        repo_root=REPO_ROOT,
        dataset_name=args.dataset,
        pipeline_name=args.pipeline,
        use_local_paths=not args.use_example_paths,
    )
    dataset_slice = load_formal_dataset_slice(
        experiment=experiment,
        split_profile=args.split_profile,
        limit=args.limit,
        seed=args.seed,
        relevance_threshold=args.gold_relevance_threshold,
    )
    embedder = build_embedder(args, experiment=experiment)
    index_dir = resolve_index_dir(args, experiment=experiment, dataset_name=dataset_slice.dataset_name)
    if any(dataset_slice.context_sets_by_example.values()):
        result = build_scoped_indexes(
            args=args,
            dataset_slice=dataset_slice,
            embedder=embedder,
            index_dir=index_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if index_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dense index already exists: {index_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(index_dir)

    result = build_cached_dense_index(
        corpus=dataset_slice.corpus,
        embedder=embedder,
        output_dir=index_dir,
        dataset_name=dataset_slice.dataset_name,
        split_name=dataset_slice.split_name,
        normalize=True,
        include_title=not args.exclude_title,
        search_backend=args.search_backend,
        metadata={
            "candidate_scope": dataset_slice.candidate_scope,
            "num_examples": len(dataset_slice.examples),
            "build_limit": args.limit,
        },
    )
    print(
        json.dumps(
            {
                "index_dir": str(result.index_dir),
                "manifest": result.manifest,
                "build_log": result.build_log,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_scoped_indexes(*, args: argparse.Namespace, dataset_slice, embedder, index_dir: Path) -> dict[str, object]:
    corpus_by_id = {chunk.chunk_id: chunk for chunk in dataset_slice.corpus}
    metadata_filters = parse_context_metadata_filters(args.context_metadata)
    entries = []
    for example in dataset_slice.examples:
        context = select_context_set(
            dataset_slice.context_sets_by_example.get(example.example_id, []),
            condition=args.context_condition,
            metadata=metadata_filters,
        )
        missing = [chunk_id for chunk_id in context.chunk_ids if chunk_id not in corpus_by_id]
        if missing:
            raise ValueError(f"Context {context.context_set_id} has unloaded chunks: {missing[:3]}")
        scoped_corpus = [corpus_by_id[chunk_id] for chunk_id in context.chunk_ids]
        scoped_dir = context_index_dir(index_dir, context)
        if scoped_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Scoped dense index already exists: {scoped_dir}. Pass --overwrite to replace it."
                )
            shutil.rmtree(scoped_dir)
        result = build_cached_dense_index(
            corpus=scoped_corpus,
            embedder=embedder,
            output_dir=scoped_dir,
            dataset_name=dataset_slice.dataset_name,
            split_name=dataset_slice.split_name,
            normalize=True,
            include_title=not args.exclude_title,
            search_backend=args.search_backend,
            metadata={
                "candidate_scope": dataset_slice.candidate_scope,
                "example_id": example.example_id,
                "context_set_id": context.context_set_id,
                "context_condition": context.condition,
                "context_metadata": context.metadata,
            },
        )
        entries.append(
            {
                "example_id": example.example_id,
                "context_set_id": context.context_set_id,
                "index_dir": str(result.index_dir),
                "chunk_count": len(scoped_corpus),
                "corpus_hash": result.manifest["corpus_hash"],
            }
        )
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "scoped-dense-index-v1",
        "dataset_name": dataset_slice.dataset_name,
        "split_name": dataset_slice.split_name,
        "embedding_model": embedder.name,
        "context_condition": args.context_condition,
        "context_metadata": metadata_filters,
        "scope_count": len(entries),
        "scopes": entries,
    }
    (index_dir / "scoped_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"index_dir": str(index_dir), "manifest": manifest}


def build_embedder(args: argparse.Namespace, *, experiment):
    if args.embedder == "hashing":
        return HashingEmbedder(dimensions=args.hash_dimensions)
    retrieval_config = experiment.model_refs.get("default_retrieval", {})
    model_config = retrieval_config.get("embedding_model", {})
    model_name = args.model or model_config.get("model") or model_config.get("name") or "BAAI/bge-large-en-v1.5"
    revision = args.revision if args.model else (args.revision or model_config.get("revision"))
    if not revision:
        raise ValueError(
            "Sentence-transformer index builds require an immutable model revision; "
            "set it in the model registry or pass --revision."
        )
    local_snapshot = (
        experiment.paths.models_root / "snapshots" / str(model_name).replace("/", "--") / str(revision)
        if experiment.paths is not None
        else None
    )
    return SentenceTransformerEmbedder(
        model_name=(str(local_snapshot.resolve()) if local_snapshot and local_snapshot.exists() else str(model_name)),
        batch_size=args.batch_size or int(model_config.get("batch_size", 32)),
        device=args.device or model_config.get("device"),
        revision=str(revision),
        query_instruction=str(model_config.get("query_instruction", "")),
        document_instruction=str(model_config.get("document_instruction", "")),
        cache_folder=(
            str(experiment.paths.models_root / "hf-cache") if experiment.paths is not None else None
        ),
        identity_name=str(model_name),
    )


def resolve_index_dir(args: argparse.Namespace, *, experiment, dataset_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if experiment.paths is None:
        raise ValueError("paths config is required when --output-dir is not provided")
    index_name = args.index_name or f"{args.embedder}-{args.split_profile}"
    return (experiment.paths.models_root / "indexes" / dataset_name / index_name).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
