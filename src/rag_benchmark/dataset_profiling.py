"""Full-file format inventory for downloaded benchmark releases."""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rag_benchmark.dataset_catalog import BenchmarkDatasetSpec, sha256_file


JsonObject = dict[str, Any]


def profile_raw_dataset(spec: BenchmarkDatasetSpec) -> JsonObject:
    raw = spec.root / "raw"
    shared_report_path = (
        spec.root / "manifests" / "raw-format-report-kilt-shared.json"
        if spec.key in {"kilt_nq", "kilt_hotpotqa"}
        else None
    )
    if shared_report_path is not None and shared_report_path.exists():
        report = json.loads(shared_report_path.read_text(encoding="utf-8"))
        report["dataset"] = spec.key
        output = spec.root / "manifests" / f"raw-format-report-{spec.key}.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return report
    manifest_path = spec.root / "manifests" / f"manifest-{spec.key}.json"
    known_assets = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        known_assets = {str(asset["name"]): asset for asset in manifest.get("assets", [])}
    entries = []
    for path in sorted(item for item in raw.rglob("*") if item.is_file()):
        relative = str(path.relative_to(raw)).replace("\\", "/")
        entry: JsonObject = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": checksum_from_manifest(path, relative, known_assets),
        }
        if path.name.startswith(".") or path.name.startswith("._"):
            entry["format"] = "extraction_metadata"
        elif path.name.endswith((".tar", ".tar.gz", ".tgz")):
            entry.update(profile_tar(path))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            entry.update(profile_json_records(path))
        else:
            entry["format"] = "binary_or_unrecognized"
        entries.append(entry)
    report = {
        "schema_version": "raw-format-report-v1",
        "dataset": spec.key,
        "root": str(spec.root),
        "file_count": len(entries),
        "files": entries,
    }
    output = spec.root / "manifests" / f"raw-format-report-{spec.key}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    if shared_report_path is not None:
        shared_report = {**report, "dataset": "kilt_shared_release"}
        shared_report_path.write_text(
            json.dumps(shared_report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return report


def checksum_from_manifest(path: Path, relative: str, known_assets: dict[str, JsonObject]) -> str:
    asset = known_assets.get(relative) or known_assets.get(path.name)
    if asset and int(asset.get("size_bytes", -1)) == path.stat().st_size and asset.get("sha256"):
        return str(asset["sha256"])
    return sha256_file(path)


def profile_tar(path: Path) -> JsonObject:
    mode = "r:gz" if path.name.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(path, mode) as handle:
        members = [
            {"name": member.name, "size_bytes": member.size, "type": "file" if member.isfile() else "directory"}
            for member in handle.getmembers()
        ]
    return {"format": "tar_archive", "member_count": len(members), "members": members}


def profile_json_records(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = next((character for character in iter(lambda: handle.read(1), "") if not character.isspace()), "")
    if first == "[":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        return summarize_records(payload, format_name="json_array")
    if first != "{":
        raise ValueError(f"Unsupported JSON top level in {path}: {first!r}")
    with path.open("r", encoding="utf-8-sig") as handle:
        sample_lines = []
        for line in handle:
            if line.strip():
                sample_lines.append(line)
            if len(sample_lines) == 2:
                break
    if len(sample_lines) == 2:
        try:
            json.loads(sample_lines[0])
            json.loads(sample_lines[1])
            return summarize_jsonl(path)
        except json.JSONDecodeError:
            pass
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        records: Iterable[Any]
        if isinstance(payload, dict):
            records = [payload]
        else:
            records = payload
        return summarize_records(records, format_name="json_object")
    except json.JSONDecodeError:
        return summarize_jsonl(path)


def summarize_jsonl(path: Path) -> JsonObject:
    count = 0
    field_types: dict[str, set[str]] = defaultdict(set)
    nested_fields: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            count += 1
            collect_shape(row, field_types, nested_fields)
    return shape_report("jsonl", count, field_types, nested_fields)


def summarize_records(records: Iterable[Any], *, format_name: str) -> JsonObject:
    count = 0
    field_types: dict[str, set[str]] = defaultdict(set)
    nested_fields: dict[str, set[str]] = defaultdict(set)
    for row in records:
        count += 1
        collect_shape(row, field_types, nested_fields)
    return shape_report(format_name, count, field_types, nested_fields)


def collect_shape(
    row: Any,
    field_types: dict[str, set[str]],
    nested_fields: dict[str, set[str]],
) -> None:
    if not isinstance(row, dict):
        field_types["<record>"].add(type(row).__name__)
        return
    for key, value in row.items():
        field_types[str(key)].add(type(value).__name__)
        if isinstance(value, dict):
            keys = [str(nested_key) for nested_key in value]
            if keys and all(item.isdigit() for item in keys):
                nested_fields[str(key)].add("<dynamic_id_keys>")
            else:
                remaining = max(0, 100 - len(nested_fields[str(key)]))
                nested_fields[str(key)].update(keys[:remaining])
        elif isinstance(value, list):
            for item in value[:10]:
                if isinstance(item, dict):
                    remaining = max(0, 100 - len(nested_fields[str(key)]))
                    nested_fields[str(key)].update(str(nested_key) for nested_key in list(item)[:remaining])


def shape_report(
    format_name: str,
    count: int,
    field_types: dict[str, set[str]],
    nested_fields: dict[str, set[str]],
) -> JsonObject:
    return {
        "format": format_name,
        "record_count": count,
        "fields": {key: sorted(values) for key, values in sorted(field_types.items())},
        "nested_fields": {key: sorted(values) for key, values in sorted(nested_fields.items()) if values},
    }
