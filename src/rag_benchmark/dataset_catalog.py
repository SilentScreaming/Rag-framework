"""Dataset sources, revisions, and download helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from rag_benchmark.paths import configured_datasets_root


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class RawAsset:
    name: str
    url: str
    split: str = "all"
    archive: str | None = None
    required: bool = True


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    key: str
    provider: str
    dataset: str
    version: str
    task_track: str
    role: str
    source_page: str
    source_revision: str
    license: str
    assets: tuple[RawAsset, ...]
    notes: str = ""

    @property
    def root(self) -> Path:
        return configured_datasets_root() / self.provider / self.dataset / self.version

    @property
    def processed_root(self) -> Path:
        return self.root / "processed" / self.key


KILT_REVISION = "2664322b9a994be686e4c3a9e8f75a0b70927f22"
ALCE_REVISION = "246c476a4edfc564266b7346b6e29ef4861ae937"
ALCE_DATA_REVISION = "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
SCIFACT_REVISION = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
GARAGE_REVISION = "2b7264f5d4e066affbe4b67b7d3b9cffeff98ede"
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RGB_REVISION = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"


def _kilt_asset(name: str, split: str) -> RawAsset:
    return RawAsset(name=name, url=f"https://dl.fbaipublicfiles.com/KILT/{name}", split=split)


DATASET_SPECS: dict[str, BenchmarkDatasetSpec] = {
    "kilt_nq": BenchmarkDatasetSpec(
        key="kilt_nq",
        provider="facebookresearch",
        dataset="kilt",
        version="kilt-2019-08",
        task_track="short_answer",
        role="single_hop_open_domain_qa",
        source_page="https://github.com/facebookresearch/KILT",
        source_revision=KILT_REVISION,
        license="MIT code; source datasets and Wikipedia content retain upstream terms",
        assets=(
            _kilt_asset("nq-train-kilt.jsonl", "train"),
            _kilt_asset("nq-dev-kilt.jsonl", "dev"),
            _kilt_asset("nq-test_without_answers-kilt.jsonl", "test"),
            RawAsset(
                name="kilt_knowledgesource.json",
                url="https://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json",
                split="knowledge_source",
            ),
        ),
        notes="Answer accuracy and retrieval use aligned KILT provenance.",
    ),
    "kilt_hotpotqa": BenchmarkDatasetSpec(
        key="kilt_hotpotqa",
        provider="facebookresearch",
        dataset="kilt",
        version="kilt-2019-08",
        task_track="short_answer",
        role="multi_hop_open_domain_qa",
        source_page="https://github.com/facebookresearch/KILT",
        source_revision=KILT_REVISION,
        license="MIT code; source datasets and Wikipedia content retain upstream terms",
        assets=(
            _kilt_asset("hotpotqa-train-kilt.jsonl", "train"),
            _kilt_asset("hotpotqa-dev-kilt.jsonl", "dev"),
            _kilt_asset("hotpotqa-test_without_answers-kilt.jsonl", "test"),
            RawAsset(
                name="kilt_knowledgesource.json",
                url="https://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json",
                split="knowledge_source",
            ),
        ),
        notes="Uses the shared KILT FullWiki knowledge source rather than per-question contexts.",
    ),
    "alce": BenchmarkDatasetSpec(
        key="alce",
        provider="princeton-nlp",
        dataset="alce",
        version="emnlp2023-v1",
        task_track="long_answer_citation",
        role="long_answer_and_list_citation",
        source_page="https://github.com/princeton-nlp/ALCE",
        source_revision=ALCE_REVISION,
        license="MIT code; bundled datasets retain upstream terms",
        assets=(
            RawAsset(
                name="ALCE-data.tar",
                url="https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/"
                f"{ALCE_DATA_REVISION}/ALCE-data.tar",
                archive="tar",
            ),
        ),
        notes="ASQA and QAMPARI use the official ALCE evaluation bundle and top-100 retrieval results.",
    ),
    "scifact": BenchmarkDatasetSpec(
        key="scifact",
        provider="allenai",
        dataset="scifact",
        version="release-latest",
        task_track="verification_hallucination",
        role="claim_verification",
        source_page="https://github.com/allenai/scifact",
        source_revision=SCIFACT_REVISION,
        license="Claims/annotations: CC-BY-4.0; S2ORC abstracts: ODC-By-1.0; code: Apache-2.0",
        assets=(
            RawAsset(
                name="data.tar.gz",
                url="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
                archive="tar.gz",
            ),
        ),
        notes="Dev labels provide document retrieval, rationale selection, and SUPPORT/CONTRADICT supervision.",
    ),
    "garage": BenchmarkDatasetSpec(
        key="garage",
        provider="amazon-science",
        dataset="garage",
        version="acl2025-v1",
        task_track="verification_hallucination",
        role="grounding_and_selective_refusal",
        source_page="https://github.com/amazon-science/GaRAGe",
        source_revision=GARAGE_REVISION,
        license="CC-BY-NC-4.0",
        assets=(
            RawAsset(
                name="GaRAGe_benchmark.jsonl",
                url=f"https://raw.githubusercontent.com/amazon-science/GaRAGe/{GARAGE_REVISION}/"
                "data/GaRAGe_benchmark.jsonl",
            ),
        ),
    ),
    "ragtruth": BenchmarkDatasetSpec(
        key="ragtruth",
        provider="particlemedia",
        dataset="ragtruth",
        version="2024-02",
        task_track="verification_hallucination",
        role="hallucination_judge_calibration",
        source_page="https://github.com/ParticleMedia/RAGTruth",
        source_revision=RAGTRUTH_REVISION,
        license="MIT",
        assets=(
            RawAsset(
                name="response.jsonl",
                url=f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/"
                "dataset/response.jsonl",
            ),
            RawAsset(
                name="source_info.jsonl",
                url=f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/"
                "dataset/source_info.jsonl",
            ),
        ),
        notes="Only the QA subset is normalized for this project; all raw task types remain preserved.",
    ),
    "rgb": BenchmarkDatasetSpec(
        key="rgb",
        provider="chen700564",
        dataset="rgb",
        version="2024-03",
        task_track="verification_hallucination",
        role="rag_robustness_stress_test",
        source_page="https://github.com/chen700564/RGB",
        source_revision=RGB_REVISION,
        license="CC-BY-NC-SA-4.0",
        assets=tuple(
            RawAsset(
                name=name,
                url=f"https://raw.githubusercontent.com/chen700564/RGB/{RGB_REVISION}/data/{name}",
                split=name.removesuffix(".json"),
            )
            for name in (
                "en.json",
                "en_refine.json",
                "en_int.json",
                "en_fact.json",
                "zh.json",
                "zh_refine.json",
                "zh_int.json",
                "zh_fact.json",
            )
        ),
        notes="All release files are retained and profiled; formal stress tests use the English refined files.",
    ),
}


def ensure_layout(root: Path) -> None:
    for relative in ("raw", "processed", "processed/splits", "manifests"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_ca_bundle(repo_root: Path | None = None) -> str | bool:
    configured = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if configured and Path(configured).exists():
        return configured
    return True


def download_dataset(
    spec: BenchmarkDatasetSpec,
    *,
    repo_root: Path | None = None,
    overwrite: bool = False,
) -> JsonObject:
    ensure_layout(spec.root)
    entries = []
    for asset in spec.assets:
        target = spec.root / "raw" / asset.name
        download_resumable(
            asset.url,
            target,
            verify=resolve_ca_bundle(repo_root),
            overwrite=overwrite,
        )
        extracted = extract_archive(target, spec.root / "raw", archive=asset.archive)
        entries.append(
            {
                "name": asset.name,
                "url": asset.url,
                "split": asset.split,
                "archive": asset.archive,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "extracted_files": [str(path.relative_to(spec.root / "raw")).replace("\\", "/") for path in extracted],
            }
        )
    manifest = {
        "schema_version": "dataset-manifest-v1",
        "dataset": spec.key,
        "provider": spec.provider,
        "version": spec.version,
        "task_track": spec.task_track,
        "role": spec.role,
        "source_page": spec.source_page,
        "source_revision": spec.source_revision,
        "license": spec.license,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "assets": entries,
        "notes": spec.notes,
    }
    manifest_path = spec.root / "manifests" / f"manifest-{spec.key}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def download_resumable(
    url: str,
    target: Path,
    *,
    verify: str | bool,
    overwrite: bool = False,
    retries: int = 5,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return
    part = target.with_name(target.name + ".part")
    if overwrite:
        part.unlink(missing_ok=True)
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "rag-benchmark-dataset-prep/2.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, stream=True, timeout=(30, 300), headers=headers, verify=verify) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with part.open(mode) as handle:
                    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            part.replace(target)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(min(30, attempt * 3))


def extract_archive(path: Path, destination: Path, *, archive: str | None) -> list[Path]:
    if archive is None:
        return []
    marker = destination / f".{path.name}.extracted"
    mode = "r:gz" if archive == "tar.gz" else "r:"
    if marker.exists():
        with tarfile.open(path, mode) as handle:
            return [destination / member.name for member in handle.getmembers() if member.isfile()]
    extracted: list[Path] = []
    destination_resolved = destination.resolve()
    with tarfile.open(path, mode) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)
        extracted = [destination / member.name for member in handle.getmembers() if member.isfile()]
    marker.write_text(datetime.now(UTC).isoformat(), encoding="ascii")
    return extracted
