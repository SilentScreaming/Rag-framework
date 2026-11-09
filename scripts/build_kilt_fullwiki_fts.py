"""Build a resumable article-level FTS5 index for the KILT 2019 Wikipedia snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from rag_benchmark.config import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--skip-database-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    loader = ConfigLoader(REPO_ROOT)
    paths = loader.load_paths(use_local=True)
    kilt_root = paths.datasets["kilt_nq"].root.parents[1]
    source = Path(args.source).resolve() if args.source else kilt_root / "raw" / "kilt_knowledgesource.json"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else paths.models_root / "indexes" / "kilt_fullwiki_2019" / "fts5-pages-v1"
    )
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "index.sqlite"
    started = time.perf_counter()
    with closing(sqlite3.connect(database_path)) as connection:
        configure_database(connection)
        create_schema(connection)
        metadata = read_metadata(connection)
        if metadata.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "output_dir": str(output_dir)}, indent=2))
            return 0
        source_offset = int(metadata.get("source_offset", 0))
        page_count = int(metadata.get("page_count", 0))
        next_row_id = int(connection.execute("SELECT COALESCE(MAX(rowid), 0) + 1 FROM pages").fetchone()[0])
        with source.open("rb") as handle:
            handle.seek(source_offset)
            while True:
                page_rows = []
                fts_rows = []
                for _ in range(args.batch_size):
                    byte_offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    if not raw_line.strip():
                        continue
                    page = json.loads(raw_line.decode("utf-8"))
                    wiki_id = str(page.get("wikipedia_id") or page.get("_id"))
                    title = str(page.get("wikipedia_title") or "")
                    body = "\n".join(str(value) for value in list(page.get("text") or []))
                    row_id = next_row_id
                    next_row_id += 1
                    page_rows.append((row_id, wiki_id, title, byte_offset, len(raw_line)))
                    fts_rows.append((row_id, title, body))
                if not page_rows:
                    break
                connection.execute("BEGIN")
                connection.executemany(
                    "INSERT INTO pages(rowid, wikipedia_id, title, byte_offset, byte_length) VALUES (?, ?, ?, ?, ?)",
                    page_rows,
                )
                connection.executemany(
                    "INSERT INTO pages_fts(rowid, title, body) VALUES (?, ?, ?)",
                    fts_rows,
                )
                page_count += len(page_rows)
                source_offset = handle.tell()
                write_metadata(
                    connection,
                    {
                        "schema_version": "kilt-fullwiki-fts-v1",
                        "status": "building",
                        "source_path": str(source),
                        "source_size": str(source.stat().st_size),
                        "source_offset": str(source_offset),
                        "page_count": str(page_count),
                    },
                )
                connection.commit()
                if page_count % args.progress_every < len(page_rows):
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "fts_build_progress",
                                "pages": page_count,
                                "source_gib": round(source_offset / (1024**3), 3),
                                "elapsed_s": round(elapsed, 1),
                                "pages_per_s": round(page_count / max(elapsed, 0.001), 1),
                            }
                        ),
                        flush=True,
                    )
        if not args.skip_optimize:
            connection.execute("INSERT INTO pages_fts(pages_fts) VALUES ('optimize')")
        write_metadata(
            connection,
            {
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "source_offset": str(source.stat().st_size),
                "page_count": str(page_count),
            },
        )
        connection.commit()
        sqlite_pages = int(connection.execute("PRAGMA page_count").fetchone()[0])
        sqlite_page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])

    source_sha256 = expected_source_sha256(kilt_root)
    database_sha256 = None if args.skip_database_hash else sha256_file(database_path)
    manifest = {
        "schema_version": "kilt-fullwiki-fts-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": source_sha256,
        "database_file": database_path.name,
        "database_size": database_path.stat().st_size,
        "database_sha256": database_sha256,
        "pages": page_count,
        "sqlite_page_count": sqlite_pages,
        "sqlite_page_size": sqlite_page_size,
        "tokenizer": "porter unicode61 remove_diacritics 2",
        "title_bm25_weight": 8.0,
        "body_bm25_weight": 1.0,
        "build_seconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), **manifest}, indent=2))
    return 0


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-2097152")
    connection.execute("PRAGMA wal_autocheckpoint=20000")


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS pages("
        "rowid INTEGER PRIMARY KEY, wikipedia_id TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
        "byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5("
        "title, body, content='', tokenize='porter unicode61 remove_diacritics 2')"
    )
    connection.commit()


def read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(key): str(value) for key, value in connection.execute("SELECT key, value FROM metadata")}


def write_metadata(connection: sqlite3.Connection, values: dict[str, str]) -> None:
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        list(values.items()),
    )


def expected_source_sha256(kilt_root: Path) -> str | None:
    manifest_path = kilt_root / "manifests" / "manifest-kilt_nq.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for asset in manifest.get("assets", []):
        if asset.get("name") == "kilt_knowledgesource.json":
            return asset.get("sha256")
    return None


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
