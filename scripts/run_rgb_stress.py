"""Run configurable RGB noise, integration, and counterfactual comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_benchmark.config import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    ("noise-0.0", "noise_robustness", {"noise_rate": 0.0}),
    ("noise-0.2", "noise_robustness", {"noise_rate": 0.2}),
    ("noise-0.4", "noise_robustness", {"noise_rate": 0.4}),
    ("noise-0.6", "noise_robustness", {"noise_rate": 0.6}),
    ("noise-0.8", "noise_robustness", {"noise_rate": 0.8}),
    ("noise-1.0", "noise_robustness", {"noise_rate": 1.0}),
    ("integration", "information_integration", {}),
    ("factual-control", "factual_control", {}),
    ("counterfactual", "counterfactual_context", {}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-profile", choices=["quality_max", "budget_matched"], default="quality_max")
    parser.add_argument("--condition", action="append", choices=[row[0] for row in CONDITIONS], default=[])
    parser.add_argument("--model", default=None, help="Model name; defaults to local API configuration")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split-profile", default="formal")
    parser.add_argument("--dense-index-path", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ConfigLoader(REPO_ROOT).load_paths(use_local=True)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else paths.outputs_root / "rgb" / args.execution_profile
    )
    output_root.mkdir(parents=True, exist_ok=True)
    scoped_index_root = args.dense_index_path or paths.models_root / "indexes" / "rgb_en" / "scoped"
    selected = [row for row in CONDITIONS if not args.condition or row[0] in set(args.condition)]
    records = []
    for condition_id, condition, metadata in selected:
        suite_id = f"rgb-{args.execution_profile}-{condition_id}"
        summary_path = output_root / "suites" / suite_id / "suite_summary.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                "-m",
                "scripts.run_experiment_suite",
                "--dataset",
                "rgb_en",
                "--split-profile",
                args.split_profile,
                "--limit",
                str(args.limit),
                "--execution-profile",
                args.execution_profile,
                "--generator",
                "openai",
                "--query-rewriter",
                "openai",
                "--dense-index-path",
                str(scoped_index_root),
                "--build-index-if-missing",
                "--context-condition",
                condition,
                "--output-root",
                str(output_root),
                "--suite-id",
                suite_id,
                "--bootstrap-samples",
                str(args.bootstrap_samples),
            ]
            if args.model:
                command.extend(["--model", args.model])
            for key, value in metadata.items():
                command.extend(["--context-metadata", f"{key}={json.dumps(value)}"])
            completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                failure = output_root / f"{suite_id}.failure.log"
                failure.write_text(
                    f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}", encoding="utf-8"
                )
                raise RuntimeError(f"RGB suite failed: {suite_id}. See {failure}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for pipeline, aggregate in payload["aggregates"].items():
            records.append(
                {
                    "condition_id": condition_id,
                    "condition": condition,
                    "noise_rate": metadata.get("noise_rate"),
                    "pipeline": pipeline,
                    **aggregate,
                }
            )
        print(json.dumps({"event": "rgb_condition_complete", "condition_id": condition_id}), flush=True)
    report = {
        "schema_version": "rgb-stress-matrix-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "execution_profile": args.execution_profile,
        "conditions": [row[0] for row in selected],
        "rows": records,
    }
    (output_root / "rgb_stress_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output_root / "rgb_stress_metrics.csv", records)
    print(json.dumps({"status": "complete", "output_root": str(output_root), "rows": len(records)}, indent=2))
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
