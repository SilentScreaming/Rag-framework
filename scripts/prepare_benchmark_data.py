"""Download, inventory, normalize, and validate the benchmark data portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_benchmark.dataset_adapters import process_dataset, validate_canonical_dataset
from rag_benchmark.dataset_catalog import DATASET_SPECS, download_dataset
from rag_benchmark.dataset_profiling import profile_raw_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any((args.download, args.profile, args.process, args.validate)):
        print_plan(args.datasets)
        return 0
    output = {}
    for key in args.datasets:
        spec = DATASET_SPECS[key]
        result = {}
        if args.download:
            result["download"] = download_dataset(spec, repo_root=REPO_ROOT, overwrite=args.overwrite)
        if args.profile:
            result["profile"] = profile_raw_dataset(spec)
        if args.process:
            result["process"] = process_dataset(spec, overwrite=args.overwrite)
        elif args.validate:
            result["validation"] = validate_expected_outputs(key)
        output[key] = result
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def print_plan(keys: list[str]) -> None:
    rows = []
    for key in keys:
        spec = DATASET_SPECS[key]
        rows.append(
            {
                "key": key,
                "root": str(spec.root),
                "task_track": spec.task_track,
                "role": spec.role,
                "source_page": spec.source_page,
                "source_revision": spec.source_revision,
                "license": spec.license,
                "assets": [asset.__dict__ for asset in spec.assets],
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))


def validate_expected_outputs(key: str) -> list[dict[str, object]]:
    spec = DATASET_SPECS[key]
    if key == "alce":
        roots = [spec.root / "processed" / "alce_asqa", spec.root / "processed" / "alce_qampari"]
    else:
        roots = [spec.processed_root]
    return [validate_canonical_dataset(root) for root in roots]


if __name__ == "__main__":
    raise SystemExit(main())
