"""Dense index cache construction and validation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.dense import (
    DenseVectorIndex,
    EmbeddingBackend,
    document_text,
    encode_documents,
)


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class DenseIndexBuildResult:
    index_dir: Path
    manifest: JsonObject
    build_log: JsonObject


def build_cached_dense_index(
    *,
    corpus: list[CorpusChunk],
    embedder: EmbeddingBackend,
    output_dir: Path,
    dataset_name: str,
    split_name: str,
    normalize: bool = True,
    include_title: bool = True,
    search_backend: str = "auto",
    metadata: JsonObject | None = None,
) -> DenseIndexBuildResult:
    if not corpus:
        raise ValueError("cannot build dense index for an empty corpus")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = encode_documents(
        embedder,
        [document_text(chunk, include_title=include_title) for chunk in corpus],
    )
    index = DenseVectorIndex(
        chunks=list(corpus),
        vectors=vectors,
        embedding_model=embedder.name,
        normalize=normalize,
        search_backend=search_backend,
    )
    np.save(output_dir / "vectors.npy", index.vectors)
    fingerprint = corpus_fingerprint(corpus)
    manifest = {
        "schema_version": "dense-index-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_name": dataset_name,
        "split_name": split_name,
        "embedding_model": embedder.name,
        "normalize": normalize,
        "include_title": include_title,
        "search_backend": search_backend,
        "chunk_count": len(corpus),
        "vector_shape": list(index.vectors.shape),
        "corpus_hash": fingerprint,
        "chunk_ids": [chunk.chunk_id for chunk in corpus],
        "metadata": metadata or {},
    }
    build_ms = (time.perf_counter() - started) * 1000
    build_log = {
        "event": "dense_index_built",
        "index_dir": str(output_dir),
        "dataset_name": dataset_name,
        "split_name": split_name,
        "embedding_model": embedder.name,
        "chunk_count": len(corpus),
        "build_ms": build_ms,
        "corpus_hash": fingerprint,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "build_log.json", build_log)
    write_chunk_ids(output_dir / "chunk_ids.jsonl", corpus)
    return DenseIndexBuildResult(index_dir=output_dir, manifest=manifest, build_log=build_log)


def load_cached_dense_index(
    *,
    index_dir: Path,
    corpus: list[CorpusChunk],
    expected_embedding_model: str,
    normalize: bool = True,
    include_title: bool = True,
    search_backend: str = "auto",
) -> DenseVectorIndex:
    manifest_path = index_dir / "manifest.json"
    vectors_path = index_dir / "vectors.npy"
    if not manifest_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(f"Dense index cache is incomplete: {index_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = corpus_fingerprint(corpus)
    mismatches = []
    if manifest.get("embedding_model") != expected_embedding_model:
        mismatches.append(
            f"embedding_model {manifest.get('embedding_model')!r} != {expected_embedding_model!r}"
        )
    if bool(manifest.get("normalize", True)) != normalize:
        mismatches.append(f"normalize {manifest.get('normalize')!r} != {normalize!r}")
    if bool(manifest.get("include_title", False)) != include_title:
        mismatches.append(f"include_title {manifest.get('include_title')!r} != {include_title!r}")
    if manifest.get("corpus_hash") != expected_hash:
        mismatches.append("corpus_hash mismatch")
    if int(manifest.get("chunk_count", -1)) != len(corpus):
        mismatches.append("chunk_count mismatch")
    if mismatches:
        raise ValueError(f"Dense index cache does not match current run: {', '.join(mismatches)}")
    corpus_by_id = {chunk.chunk_id: chunk for chunk in corpus}
    index = DenseVectorIndex.load(index_dir, corpus_by_id=corpus_by_id)
    index.search_backend = search_backend
    index.__post_init__()
    return index


def corpus_fingerprint(corpus: list[CorpusChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(corpus, key=lambda item: item.chunk_id):
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "dataset": chunk.dataset,
            "title": chunk.title,
            "text": chunk.text,
            "sentences": chunk.sentences,
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, data: JsonObject) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_chunk_ids(path: Path, corpus: list[CorpusChunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, chunk in enumerate(corpus, start=1):
            handle.write(json.dumps({"rank": rank, "chunk_id": chunk.chunk_id}, sort_keys=True))
            handle.write("\n")
