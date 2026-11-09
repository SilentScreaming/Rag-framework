"""Reproducible sampling, independent of method and score."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from rag_benchmark.file_utils import file_sha256
from rag_benchmark.stage_cache import canonical_hash


def make_sample_manifest(*, dataset, data_version, split, split_path, fraction=0.05, seed=42,
                         purpose="resource_and_format_sample"):
    rows = [json.loads(s) for s in Path(split_path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or not 0 < fraction <= 1:
        raise ValueError("nonempty target split and fraction in (0,1] required")
    key = "response_id" if all(r.get("response_id") for r in rows) else "example_id"
    ids = sorted(str(r[key]) for r in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sampling unit")
    selected = sorted(random.Random(seed).sample(ids, math.ceil(len(ids) * fraction)))
    payload = {"schema_version": "sample-manifest-v1", "dataset": dataset, "data_version": data_version,
               "split": split, "seed": seed, "fraction": fraction, "unit": key,
               "population_count": len(ids), "population_hash": canonical_hash(ids),
               "split_file_sha256": file_sha256(Path(split_path)), "sample_ids": selected,
               "sampling": "sorted_population_random_sample_without_replacement",
                "purpose": purpose, "retrieval_corpus_subsampled": False}
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def load_sample_manifest(path, *, dataset, split_path, seed=None, data_version=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sample-manifest-v1" or payload.get("manifest_hash") != canonical_hash({k:v for k,v in payload.items() if k != "manifest_hash"}):
        raise ValueError("invalid sample manifest")
    if payload["dataset"] != dataset or payload["split"] != Path(split_path).stem or payload["split_file_sha256"] != file_sha256(Path(split_path)):
        raise ValueError("sample manifest dataset/split/source mismatch")
    if seed is not None and payload["seed"] != seed:
        raise ValueError("sample seed mismatch")
    if data_version is not None and payload["data_version"] != data_version:
        raise ValueError("sample data version mismatch")
    expected = make_sample_manifest(dataset=dataset, data_version=payload["data_version"], split=payload["split"],
                                   split_path=split_path, fraction=payload["fraction"], seed=payload["seed"],
                                   purpose=payload.get("purpose", "resource_and_format_sample"))
    if expected != payload:
        raise ValueError("sample list is not the declared random sample")
    return payload
