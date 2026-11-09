"""Freeze non-gold KILT candidate pools from the full Wikipedia page index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from rag_benchmark.config import ConfigLoader
from rag_benchmark.data import CorpusChunk, load_examples, load_qrels, read_jsonl
from rag_benchmark.retrieval.bm25 import BM25Retriever
from rag_benchmark.retrieval.candidate_cache import (
    FullWikiPageIndex,
    canonical_chunks_from_kilt_page,
    corpus_chunk_to_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fts-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-profile", action="append", required=True,
                        help="Explicit DATASET:SPLIT:LIMIT specification; repeat for multiple splits")
    parser.add_argument("--page-top-k", type=int, default=150)
    parser.add_argument("--seed-page-top-k", type=int, default=30)
    parser.add_argument("--candidate-top-k", type=int, default=1000)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_top_k < 1 or args.candidate_top_k < 1:
        raise ValueError("page and candidate limits must be >= 1")
    loader = ConfigLoader(REPO_ROOT)
    paths = loader.load_paths(use_local=True)
    fts_index = (
        Path(args.fts_index).resolve()
        if args.fts_index
        else paths.models_root / "indexes" / "kilt_fullwiki_2019" / "fts5-pages-v1" / "index.sqlite"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else paths.models_root / "candidate_caches" / "kilt-main-v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "candidates.sqlite"
    if args.overwrite:
        database_path.unlink(missing_ok=True)
        (output_dir / "manifest.json").unlink(missing_ok=True)
        (output_dir / "vectors.npy").unlink(missing_ok=True)
    specs = [parse_spec(value) for value in args.dataset_profile]
    started = time.perf_counter()
    with closing(sqlite3.connect(database_path)) as connection:
        configure_database(connection)
        create_schema(connection)
        completed = {
            str(row[0]) for row in connection.execute("SELECT example_id FROM selections WHERE status='complete'")
        }
        selected_by_dataset: dict[str, list[str]] = defaultdict(list)
        qrels_by_dataset = {}
        tasks = []
        processed = 0
        for dataset, profile, limit in specs:
            root = paths.datasets[dataset].root
            examples = load_examples(root)
            qrels_by_dataset[dataset] = load_qrels(root, relevance_threshold=3)
            split_path = root / "splits" / f"{profile}.jsonl"
            selected_ids = [str(row["example_id"]) for row in read_jsonl(split_path)][:limit]
            selected_by_dataset[dataset].extend(selected_ids)
            for example_id in selected_ids:
                if example_id not in examples:
                    raise KeyError(f"Split references missing example: {example_id}")
                example = examples[example_id]
                if example_id in completed:
                    continue
                tasks.append((dataset, profile, example_id, example.question))
        worker = lambda task: build_candidate_selection(
            task=task,
            fts_index=fts_index,
            args=args,
        )
        if args.workers > 1:
            executor = ThreadPoolExecutor(max_workers=args.workers)
            results = executor.map(worker, tasks)
        else:
            executor = None
            results = map(worker, tasks)
        try:
            for selection in results:
                dataset = str(selection["dataset"])
                profile = str(selection["profile"])
                example_id = str(selection["example_id"])
                question = str(selection["question"])
                ranked = selection["ranked"]
                page_rank_by_doc = selection["page_rank_by_doc"]
                page_score_by_doc = selection["page_score_by_doc"]
                connection.execute("BEGIN")
                connection.execute(
                    "INSERT INTO selections(example_id, dataset, profile, question, status) "
                    "VALUES (?, ?, ?, ?, 'building')",
                    (example_id, dataset, profile, question),
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO documents(chunk_id, payload_json) VALUES (?, ?)",
                    [
                        (
                            result.chunk.chunk_id,
                            json.dumps(corpus_chunk_to_json(result.chunk), ensure_ascii=False, sort_keys=True),
                        )
                        for result in ranked
                    ],
                )
                connection.executemany(
                    "INSERT INTO candidates(example_id, candidate_rank, chunk_id, lexical_score, page_rank, page_score) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            example_id,
                            rank,
                            result.chunk.chunk_id,
                            float(result.score),
                            page_rank_by_doc[result.chunk.doc_id],
                            page_score_by_doc[result.chunk.doc_id],
                        )
                        for rank, result in enumerate(ranked, start=1)
                    ],
                )
                connection.execute(
                    "UPDATE selections SET status='complete' WHERE example_id=?", (example_id,)
                )
                connection.commit()
                processed += 1
                if processed % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "candidate_cache_progress",
                                "new_examples": processed,
                                "elapsed_s": round(elapsed, 1),
                                "examples_per_s": round(processed / max(elapsed, 0.001), 3),
                            }
                        ),
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        audit = audit_candidate_hits(
            connection,
            selected_by_dataset=selected_by_dataset,
            qrels_by_dataset=qrels_by_dataset,
        )
        example_count = int(connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0])
        candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        document_count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        cache_hash = candidate_cache_hash(connection)
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('cache_hash', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (cache_hash,),
        )
        connection.commit()
    manifest = {
        "schema_version": "kilt-candidate-cache-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_scope": "fullwiki_fts_candidate_restricted",
        "gold_injection": False,
        "selection_query": "original_question_only",
        "fts_index": str(fts_index),
        "dataset_profiles": [f"{dataset}:{profile}:{limit}" for dataset, profile, limit in specs],
        "page_top_k": args.page_top_k,
        "seed_page_top_k": args.seed_page_top_k,
        "candidate_top_k": args.candidate_top_k,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "examples": example_count,
        "candidate_rows": candidate_count,
        "unique_documents": document_count,
        "cache_hash": cache_hash,
        "embedding_model": None,
        "vectors_file": None,
        "audit": audit,
        "build_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), **manifest}, indent=2))
    return 0


def parse_spec(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected DATASET:PROFILE:LIMIT, got {value!r}")
    return parts[0], parts[1], int(parts[2])


def build_candidate_selection(*, task: tuple[str, str, str, str], fts_index: Path, args) -> dict:
    dataset, profile, example_id, question = task
    page_index = getattr(THREAD_LOCAL, "page_index", None)
    if page_index is None:
        page_index = FullWikiPageIndex(fts_index)
        THREAD_LOCAL.page_index = page_index
    seed_hits = page_index.search(
        question,
        top_k=min(args.seed_page_top_k, args.page_top_k),
    )
    seed_pages = [(hit, page_index.load_page(hit)) for hit in seed_hits]
    linked_ids = rank_anchor_expansions(
        question=question,
        seed_pages=seed_pages,
        limit=max(0, args.page_top_k - len(seed_pages)),
    )
    linked_hits = page_index.get_by_wikipedia_ids(linked_ids)
    page_hits_with_payload = [
        *seed_pages,
        *((hit, page_index.load_page(hit)) for hit in linked_hits),
    ][: args.page_top_k]
    chunks_by_id: dict[str, CorpusChunk] = {}
    page_rank_by_doc = {}
    page_score_by_doc = {}
    for page_rank, (hit, page) in enumerate(page_hits_with_payload, start=1):
        for chunk in canonical_chunks_from_kilt_page(page):
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
            page_rank_by_doc[chunk.doc_id] = page_rank
            page_score_by_doc[chunk.doc_id] = hit.score
    if not chunks_by_id:
        raise RuntimeError(f"FullWiki FTS returned no chunks for {example_id}")
    retriever = BM25Retriever(
        k1=args.bm25_k1,
        b=args.bm25_b,
        include_title=True,
        title_weight=2,
    )
    retriever.build(list(chunks_by_id.values()))
    ranked = retriever.retrieve(question, top_k=min(args.candidate_top_k, len(chunks_by_id)))
    return {
        "dataset": dataset,
        "profile": profile,
        "example_id": example_id,
        "question": question,
        "ranked": ranked,
        "page_rank_by_doc": page_rank_by_doc,
        "page_score_by_doc": page_score_by_doc,
    }


def rank_anchor_expansions(
    *,
    question: str,
    seed_pages: list[tuple[object, dict]],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    query_tokens = {token.casefold() for token in question.split() if len(token) > 2}
    seen = {
        str(page.get("wikipedia_id") or page.get("_id"))
        for _hit, page in seed_pages
    }
    candidates = []
    for source_rank, (_hit, page) in enumerate(seed_pages, start=1):
        for anchor_index, anchor in enumerate(list(page.get("anchors") or [])):
            wikipedia_id = str(anchor.get("wikipedia_id") or "")
            if not wikipedia_id or wikipedia_id in seen:
                continue
            title = str(anchor.get("wikipedia_title") or "")
            text = str(anchor.get("text") or "")
            overlap = len(query_tokens & {token.casefold() for token in f"{title} {text}".split()})
            candidates.append((-overlap, source_rank, anchor_index, wikipedia_id))
    result = []
    for _negative_overlap, _source_rank, _anchor_index, wikipedia_id in sorted(
        candidates,
        key=lambda row: (row[1], row[0], row[2], row[3]),
    ):
        if wikipedia_id in seen:
            continue
        seen.add(wikipedia_id)
        result.append(wikipedia_id)
        if len(result) >= limit:
            break
    return result


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-1048576")


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS documents("
        "chunk_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, vector_row INTEGER UNIQUE)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS selections("
        "example_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, profile TEXT NOT NULL, "
        "question TEXT NOT NULL, status TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS candidates("
        "example_id TEXT NOT NULL, candidate_rank INTEGER NOT NULL, chunk_id TEXT NOT NULL, "
        "lexical_score REAL NOT NULL, page_rank INTEGER NOT NULL, page_score REAL NOT NULL, "
        "PRIMARY KEY(example_id, candidate_rank), UNIQUE(example_id, chunk_id), "
        "FOREIGN KEY(example_id) REFERENCES selections(example_id), "
        "FOREIGN KEY(chunk_id) REFERENCES documents(chunk_id))"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS candidates_chunk_idx ON candidates(chunk_id)")
    connection.commit()


def audit_candidate_hits(
    connection: sqlite3.Connection,
    *,
    selected_by_dataset: dict[str, list[str]],
    qrels_by_dataset: dict[str, dict[str, list]],
) -> dict[str, dict[str, float | int]]:
    report = {}
    for dataset, example_ids in selected_by_dataset.items():
        any_hits = all_hits = examples_with_gold = 0
        for example_id in example_ids:
            gold_ids = {row.chunk_id for row in qrels_by_dataset[dataset].get(example_id, [])}
            if not gold_ids:
                continue
            examples_with_gold += 1
            candidate_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT chunk_id FROM candidates WHERE example_id=?", (example_id,)
                )
            }
            overlap = gold_ids & candidate_ids
            any_hits += int(bool(overlap))
            all_hits += int(gold_ids <= candidate_ids)
        report[dataset] = {
            "examples_with_gold": examples_with_gold,
            "any_gold_hit_rate": any_hits / examples_with_gold if examples_with_gold else 0.0,
            "all_gold_hit_rate": all_hits / examples_with_gold if examples_with_gold else 0.0,
        }
    return report


def candidate_cache_hash(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for example_id, rank, chunk_id in connection.execute(
        "SELECT example_id, candidate_rank, chunk_id FROM candidates ORDER BY example_id, candidate_rank"
    ):
        digest.update(f"{example_id}\t{rank}\t{chunk_id}\n".encode("utf-8"))
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
