"""Download revision-pinned model snapshots and write a verifiable asset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests

from rag_benchmark.config import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SECTIONS = ("embedding_model", "reranker", "evidence_judge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ref", default="default_retrieval")
    parser.add_argument("--section", action="append", choices=MODEL_SECTIONS, default=[])
    parser.add_argument("--skip-file-hashes", action="store_true")
    return parser.parse_args()


def load_hub_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            'freeze_model_assets needs the model extra: python -m pip install -e ".[models]"'
        ) from exc
    return HfApi


def main() -> int:
    args = parse_args()
    HfApi = load_hub_api()
    loader = ConfigLoader(REPO_ROOT)
    paths = loader.load_paths(use_local=True)
    model_config = loader.load_model_dict(args.model_ref)
    snapshots_root = paths.models_root / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    verify: str | bool = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or True
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    records = []
    for section in args.section or MODEL_SECTIONS:
        config = dict(model_config[section])
        repo_id = str(config.get("model") or config.get("name") or "")
        revision = str(config.get("revision") or "")
        if not repo_id or not revision:
            raise ValueError(f"{section} requires model/name and immutable revision")
        snapshot_path = (snapshots_root / repo_id.replace("/", "--") / revision).resolve()
        snapshot_path.mkdir(parents=True, exist_ok=True)
        repo_files = [
            row
            for row in api.list_repo_tree(repo_id=repo_id, revision=revision, recursive=True)
            if getattr(row, "path", None) and hasattr(row, "size")
        ]
        for row in repo_files:
            destination = snapshot_path / str(row.path)
            expected_size = int(row.size or 0)
            if destination.exists() and (expected_size == 0 or destination.stat().st_size == expected_size):
                continue
            download_file(
                repo_id=repo_id,
                revision=revision,
                filename=str(row.path),
                destination=destination,
                expected_size=expected_size,
                verify=verify,
                token=token,
            )
        files = []
        for path in sorted(item for item in snapshot_path.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(snapshot_path).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": None if args.skip_file_hashes else sha256_file(path),
                }
            )
        records.append(
            {
                "section": section,
                "repo_id": repo_id,
                "revision": revision,
                "snapshot_path": str(snapshot_path),
                "file_count": len(files),
                "size_bytes": sum(int(row["size"]) for row in files),
                "files": files,
            }
        )
        print(json.dumps({"event": "model_frozen", **{key: records[-1][key] for key in ("section", "repo_id", "revision", "snapshot_path", "file_count", "size_bytes")}}), flush=True)
    manifest = {
        "schema_version": "model-assets-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model_ref": args.model_ref,
        "snapshots_root": str(snapshots_root),
        "models": records,
    }
    manifest_dir = paths.models_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_path = manifest_dir / f"{args.model_ref}-assets.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "manifest": str(output_path)}, indent=2))
    return 0


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    expected_size: int,
    verify: str | bool,
    token: str | None,
) -> None:
    from huggingface_hub import hf_hub_url

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300), verify=verify) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    handle.write(block)
    if expected_size and part.stat().st_size != expected_size:
        raise IOError(
            f"Downloaded size mismatch for {repo_id}/{filename}: "
            f"{part.stat().st_size} != {expected_size}"
        )
    part.replace(destination)


if __name__ == "__main__":
    raise SystemExit(main())
