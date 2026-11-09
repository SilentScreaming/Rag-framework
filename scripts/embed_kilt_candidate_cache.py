"""Freeze document embeddings for a KILT candidate cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_benchmark.config import ConfigLoader
from rag_benchmark.retrieval.candidate_cache import corpus_chunk_from_json
from rag_benchmark.retrieval.dense import SentenceTransformerEmbedder, document_text


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = ConfigLoader(REPO_ROOT)
    paths = loader.load_paths(use_local=True)
    cache_root = (
        Path(args.candidate_cache).resolve()
        if args.candidate_cache
        else paths.models_root / "candidate_caches" / "kilt-main-v1"
    )
    manifest_path = cache_root / "manifest.json"
    database_path = cache_root / "candidates.sqlite"
    vectors_path = cache_root / "vectors.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_config = dict(loader.load_model_dict("default_retrieval").get("embedding_model", {}))
    model_name = str(args.model or model_config.get("model") or model_config.get("name"))
    revision = str(args.revision or model_config.get("revision") or "")
    if not model_name or not revision:
        raise ValueError("Embedding model and immutable revision are required")
    local_snapshot = paths.models_root / "snapshots" / model_name.replace("/", "--") / revision
    load_name = str(local_snapshot.resolve()) if local_snapshot.exists() else model_name
    embedder = SentenceTransformerEmbedder(
        model_name=load_name,
        revision=revision,
        batch_size=args.batch_size or int(model_config.get("batch_size", 32)),
        device=args.device or model_config.get("device"),
        query_instruction=str(model_config.get("query_instruction", "")),
        document_instruction=str(model_config.get("document_instruction", "")),
        cache_folder=str(paths.models_root / "hf-cache"),
        identity_name=model_name,
    )
    started = time.perf_counter()
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        document_count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        assigned_count = int(connection.execute("SELECT COUNT(*) FROM documents WHERE vector_row IS NOT NULL").fetchone()[0])
        if args.overwrite:
            connection.execute("UPDATE documents SET vector_row=NULL")
            connection.execute("DELETE FROM metadata WHERE key LIKE 'embedding_%'")
            connection.commit()
            assigned_count = 0
            vectors_path.unlink(missing_ok=True)
        if assigned_count not in {0, document_count}:
            raise RuntimeError("Candidate cache has a partially assigned vector_row mapping")
        if assigned_count == 0:
            for vector_row, (chunk_id,) in enumerate(
                connection.execute("SELECT chunk_id FROM documents ORDER BY chunk_id")
            ):
                connection.execute(
                    "UPDATE documents SET vector_row=? WHERE chunk_id=?", (vector_row, chunk_id)
                )
                if (vector_row + 1) % 10000 == 0:
                    connection.commit()
            connection.commit()
        completed = int(
            connection.execute("SELECT value FROM metadata WHERE key='embedding_completed_rows'").fetchone()[0]
        ) if connection.execute(
            "SELECT 1 FROM metadata WHERE key='embedding_completed_rows'"
        ).fetchone() else 0
        first_rows = connection.execute(
            "SELECT payload_json FROM documents WHERE vector_row>=? ORDER BY vector_row LIMIT ?",
            (completed, embedder.batch_size),
        ).fetchall()
        if not first_rows and completed == document_count:
            manifest = finalize_manifest(
                manifest=manifest,
                manifest_path=manifest_path,
                vectors_path=vectors_path,
                embedding_model=str(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='embedding_model'"
                    ).fetchone()[0]
                ),
                dimensions=int(
                    connection.execute(
                        "SELECT value FROM metadata WHERE key='embedding_dimensions'"
                    ).fetchone()[0]
                ),
                document_count=document_count,
                embedding_seconds=manifest.get("embedding_seconds"),
            )
            print(
                json.dumps(
                    {"status": "already_complete", "candidate_cache": str(cache_root), **manifest},
                    indent=2,
                )
            )
            return 0
        first_texts = [
            document_text(corpus_chunk_from_json(json.loads(row[0])), include_title=True)
            for row in first_rows
        ]
        first_vectors = embedder.encode_documents(first_texts)
        dimensions = int(first_vectors.shape[1])
        if vectors_path.exists():
            vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
            if tuple(vectors.shape) != (document_count, dimensions):
                raise ValueError("Existing vectors.npy shape does not match candidate cache")
        else:
            vectors = np.lib.format.open_memmap(
                vectors_path,
                mode="w+",
                dtype=np.float16,
                shape=(document_count, dimensions),
            )
        cursor = completed
        pending_first = first_vectors
        while cursor < document_count:
            if pending_first is not None:
                batch_vectors = pending_first
                pending_first = None
            else:
                rows = connection.execute(
                    "SELECT payload_json FROM documents WHERE vector_row>=? ORDER BY vector_row LIMIT ?",
                    (cursor, embedder.batch_size),
                ).fetchall()
                texts = [
                    document_text(corpus_chunk_from_json(json.loads(row[0])), include_title=True)
                    for row in rows
                ]
                batch_vectors = embedder.encode_documents(texts)
            end = cursor + len(batch_vectors)
            vectors[cursor:end] = batch_vectors.astype(np.float16)
            vectors.flush()
            cursor = end
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('embedding_completed_rows', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(cursor),),
            )
            connection.commit()
            if cursor % args.progress_every < len(batch_vectors) or cursor == document_count:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "event": "candidate_embedding_progress",
                            "completed": cursor,
                            "total": document_count,
                            "elapsed_s": round(elapsed, 1),
                            "documents_per_s": round((cursor - completed) / max(elapsed, 0.001), 2),
                        }
                    ),
                    flush=True,
                )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [
                ("embedding_model", embedder.name),
                ("embedding_dimensions", str(dimensions)),
                ("embedding_dtype", "float16"),
                ("embedding_completed_at", datetime.now(UTC).isoformat()),
            ],
        )
        connection.commit()
    manifest = finalize_manifest(
        manifest=manifest,
        manifest_path=manifest_path,
        vectors_path=vectors_path,
        embedding_model=embedder.name,
        dimensions=dimensions,
        document_count=document_count,
        embedding_seconds=time.perf_counter() - started,
    )
    print(json.dumps({"status": "complete", "candidate_cache": str(cache_root), **manifest}, indent=2))
    return 0


def finalize_manifest(
    *,
    manifest: dict,
    manifest_path: Path,
    vectors_path: Path,
    embedding_model: str,
    dimensions: int,
    document_count: int,
    embedding_seconds: float | None,
) -> dict:
    vectors_sha256 = file_sha256(vectors_path)
    frozen_identity = {
        "candidate_cache_hash": manifest.get("cache_hash"),
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
        "embedding_dtype": "float16",
        "embedded_documents": document_count,
        "vectors_sha256": vectors_sha256,
    }
    manifest.update(
        {
            **frozen_identity,
            "vectors_file": vectors_path.name,
            "embedding_completed_at": manifest.get("embedding_completed_at")
            or datetime.now(UTC).isoformat(),
            "embedding_seconds": embedding_seconds,
            "frozen_artifact_hash": hashlib.sha256(
                json.dumps(frozen_identity, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def file_sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()
if __name__ == "__main__":
    raise SystemExit(main())
