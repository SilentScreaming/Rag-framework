"""Write a single machine-readable table and a single browsable metrics report."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any, Iterable

from rag_benchmark.evaluation.metric_catalog import CATALOG_BY_KEY, definitions_for_track


JsonObject = dict[str, Any]


REPORT_COLUMNS = (
    "run_id",
    "dataset",
    "split",
    "pipeline",
    "track",
    "tier",
    "group",
    "metric",
    "value",
    "status",
    "direction",
    "unit",
    "ci95_low",
    "ci95_high",
    "p_value",
    "n",
    "applicability",
    "stratum_dimension",
    "stratum_value",
    "description",
    "protocol_version",
    "n_expected",
    "n_evaluated",
    "n_missing",
    "reason",
)


def write_unified_metrics_report(
    output_dir: Path,
    metric_documents: JsonObject | Iterable[JsonObject],
    *,
    stem: str = "all_metrics",
    comparisons: Iterable[JsonObject] = (),
) -> list[JsonObject]:
    """Export every metric and its availability state to CSV and HTML."""

    documents = [metric_documents] if isinstance(metric_documents, dict) else list(metric_documents)
    rows = [row for document in documents for row in metric_rows(document)]
    rows.extend(comparison_rows(list(comparisons), documents=documents))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{stem}.html").write_text(render_html(rows), encoding="utf-8")
    return rows


def metric_rows(document: JsonObject) -> list[JsonObject]:
    if "metric_results" in document:
        return structured_metric_rows(document)
    track = str(document.get("task_track", "short_answer"))
    dataset = str(document.get("dataset", "unknown"))
    split = str(document.get("split", "unknown"))
    run_id = str(document.get("run_id", "unknown"))
    definitions = {row.key: row for row in definitions_for_track(track)}
    rows: list[JsonObject] = []
    for pipeline, aggregate in document.get("pipelines", {}).items():
        intervals = document.get("confidence_intervals", {}).get(pipeline, {})
        emitted = set()
        for key, definition in definitions.items():
            value = aggregate.get(key)
            interval = intervals.get(key, {})
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split,
                    "pipeline": pipeline,
                    "track": track,
                    "tier": definition.tier,
                    "group": definition.group,
                    "metric": key,
                    "value": value if isinstance(value, (int, float)) else "",
                    "status": availability_status(
                        key=key,
                        value=value,
                        dataset=dataset,
                        applicability=definition.applicability,
                    ),
                    "direction": definition.direction,
                    "unit": definition.unit,
                    "ci95_low": interval.get("ci95_low", ""),
                    "ci95_high": interval.get("ci95_high", ""),
                    "p_value": "",
                    "n": interval.get("n", document.get("num_examples", "")),
                    "applicability": definition.applicability,
                    "stratum_dimension": "",
                    "stratum_value": "",
                    "description": definition.description,
                }
            )
            emitted.add(key)
        for key in sorted(aggregate):
            if key in emitted or not isinstance(aggregate[key], (int, float)):
                continue
            definition = CATALOG_BY_KEY.get(key)
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split,
                    "pipeline": pipeline,
                    "track": track,
                    "tier": definition.tier if definition else "diagnostic",
                    "group": definition.group if definition else infer_group(key),
                    "metric": key,
                    "value": aggregate[key],
                    "status": "computed",
                    "direction": definition.direction if definition else "neutral",
                    "unit": definition.unit if definition else infer_unit(key),
                    "ci95_low": intervals.get(key, {}).get("ci95_low", ""),
                    "ci95_high": intervals.get(key, {}).get("ci95_high", ""),
                    "p_value": "",
                    "n": intervals.get(key, {}).get("n", document.get("num_examples", "")),
                    "applicability": definition.applicability if definition else "run_specific",
                    "stratum_dimension": "",
                    "stratum_value": "",
                    "description": definition.description if definition else "Run-specific diagnostic metric.",
                }
            )
        rows.extend(stratum_rows(document, pipeline=pipeline, aggregate=aggregate))
    return rows


def stratum_rows(document: JsonObject, *, pipeline: str, aggregate: JsonObject) -> list[JsonObject]:
    rows = []
    grouped = document.get("grouped_results", {}).get(pipeline, {})
    for dimension, values in grouped.items():
        for stratum, metrics in values.items():
            for key, value in metrics.items():
                if key == "n" or not isinstance(value, (int, float)):
                    continue
                definition = CATALOG_BY_KEY.get(key)
                rows.append(
                    {
                        "run_id": document.get("run_id", "unknown"),
                        "dataset": document.get("dataset", "unknown"),
                        "split": document.get("split", "unknown"),
                        "pipeline": pipeline,
                        "track": document.get("task_track", "unknown"),
                        "tier": "diagnostic",
                        "group": definition.group if definition else infer_group(key),
                        "metric": key,
                        "value": value,
                        "status": "computed",
                        "direction": definition.direction if definition else "neutral",
                        "unit": definition.unit if definition else infer_unit(key),
                        "ci95_low": "",
                        "ci95_high": "",
                        "p_value": "",
                        "n": metrics.get("n", ""),
                        "applicability": "stratified",
                        "stratum_dimension": dimension,
                        "stratum_value": stratum,
                        "description": f"{key} for {dimension}={stratum}.",
                    }
                )
    return rows


def comparison_rows(comparisons: list[JsonObject], *, documents: list[JsonObject]) -> list[JsonObject]:
    if not comparisons:
        return []
    first = documents[0] if documents else {}
    rows = []
    for comparison in comparisons:
        key = str(comparison["metric"])
        definition = CATALOG_BY_KEY.get(key)
        baseline = str(comparison["baseline"])
        candidate = str(comparison["candidate"])
        rows.append(
            {
                "run_id": "suite_paired_comparison",
                "dataset": first.get("dataset", "unknown"),
                "split": first.get("split", "unknown"),
                "pipeline": f"{candidate} - {baseline}",
                "track": first.get("task_track", "unknown"),
                "tier": definition.tier if definition else "module",
                "group": "paired_comparison",
                "metric": f"delta_{key}",
                "value": comparison["mean_delta"],
                "status": comparison.get("status", "computed"),
                "direction": definition.direction if definition else "neutral",
                "unit": definition.unit if definition else infer_unit(key),
                "ci95_low": comparison.get("ci95_low", ""),
                "ci95_high": comparison.get("ci95_high", ""),
                "p_value": comparison.get("paired_p_value", ""),
                "n": comparison.get("n", ""),
                "applicability": "paired_examples",
                "stratum_dimension": "",
                "stratum_value": "",
                "description": (
                    f"Paired candidate-minus-baseline delta for {key}; "
                    f"test={comparison.get('test', 'paired')}"
                ),
            }
        )
    return rows


def structured_metric_rows(document: JsonObject) -> list[JsonObject]:
    rows = []
    for pipeline, results in document["metric_results"].items():
        intervals = document.get("confidence_intervals", {}).get(pipeline, {})
        for key, result in results.items():
            definition = CATALOG_BY_KEY.get(key)
            ci = intervals.get(key, {})
            row = {column: "" for column in REPORT_COLUMNS}
            row.update(run_id=document.get("run_id", ""), dataset=document.get("dataset", ""),
                       split=document.get("split", ""), pipeline=pipeline, track=document.get("task_track", ""),
                       metric=key, value=result.get("value"), status=result["status"],
                       group=definition.group if definition else infer_group(key),
                       tier=definition.tier if definition else "diagnostic",
                       direction=definition.direction if definition else "neutral",
                       unit=definition.unit if definition else "ratio",
                       ci95_low=ci.get("ci95_low"), ci95_high=ci.get("ci95_high"),
                       n=result.get("n_evaluated"), description=definition.description if definition else key)
            for field in ("protocol_version", "n_expected", "n_evaluated", "n_missing", "reason"):
                row[field] = result.get(field)
            rows.append(row)
    return rows


def availability_status(*, key: str, value: Any, dataset: str, applicability: str) -> str:
    if isinstance(value, (int, float)):
        return "computed"
    if applicability == "all":
        return "not_computed_missing_input"
    dataset_key = dataset.casefold().replace("-", "_")
    aliases = {
        "kilt": "kilt" in dataset_key,
        "hotpotqa": "hotpot" in dataset_key,
        "multi_evidence": "hotpot" in dataset_key,
        "asqa": "asqa" in dataset_key,
        "qampari": "qampari" in dataset_key,
        "scifact": "scifact" in dataset_key,
        "ragtruth": "ragtruth" in dataset_key,
        "garage": "garage" in dataset_key,
        "deflection_labels": "garage" in dataset_key,
        "attribution_labels": "garage" in dataset_key,
        "labeled_verification": "scifact" in dataset_key,
    }
    if applicability in aliases and not aliases[applicability]:
        return "not_applicable_dataset"
    if applicability in {"paired_oracle_run", "reference_distribution", "calibration_labels", "garage_requirement_judge"}:
        return f"requires_{applicability}"
    if applicability == "graded_qrels":
        return "requires_graded_qrels"
    return "not_computed_missing_input"


def render_html(rows: list[JsonObject]) -> str:
    pipelines = sorted({str(row["pipeline"]) for row in rows})
    core = [row for row in rows if row["tier"] == "core" and not row["stratum_dimension"]]
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>RAG benchmark metrics</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#17202a;background:#f7f8fa}",
        "h1,h2{letter-spacing:0} .note{color:#566573} table{border-collapse:collapse;width:100%;background:white;margin:12px 0 28px}",
        "th{background:#1f4e78;color:white;text-align:left;position:sticky;top:0}th,td{padding:7px 9px;border-bottom:1px solid #dfe6e9;font-size:13px}",
        "tr:nth-child(even){background:#f5f9fc}.computed{color:#146b3a;font-weight:600}.missing{color:#9a5b00}",
        ".num{text-align:right;font-variant-numeric:tabular-nums}</style></head><body>",
        "<h1>RAG benchmark unified metrics</h1>",
        "<p class='note'>Core metrics are summarized first. The full table includes module, diagnostic, audit, efficiency, and unavailable metrics. No blended score is constructed.</p>",
        "<h2>Core results</h2>",
        render_table(core, compact=True),
        "<h2>All metrics</h2>",
        render_table(rows, compact=False),
        f"<p class='note'>Pipelines: {html.escape(', '.join(pipelines))}. The CSV beside this file is the machine-readable source.</p>",
        "</body></html>",
    ]
    return "".join(body)


def render_table(rows: list[JsonObject], *, compact: bool) -> str:
    columns = (
        ("pipeline", "Pipeline"),
        ("tier", "Tier"),
        ("group", "Group"),
        ("metric", "Metric"),
        ("value", "Value"),
        ("ci95_low", "CI low"),
        ("ci95_high", "CI high"),
        ("p_value", "Paired p"),
        ("status", "Status"),
    )
    if not compact:
        columns += (("stratum_dimension", "Stratum"), ("stratum_value", "Value group"), ("description", "Description"))
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(label)}</th>" for _, label in columns)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                display = f"{value:.6g}"
            else:
                display = str(value)
            css = "num" if key in {"value", "ci95_low", "ci95_high"} else ""
            if key == "status":
                css = "computed" if value == "computed" else "missing"
            parts.append(f"<td class='{css}'>{html.escape(display)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def infer_group(key: str) -> str:
    if "latency" in key or key.endswith("_ms") or "tokens" in key or "cost" in key:
        return "efficiency"
    if key.startswith(("recall@", "mrr@", "r_precision", "ndcg", "any_gold", "all_gold")):
        return "retrieval"
    if "citation" in key or "claim" in key or "attribution" in key:
        return "evidence_and_citation"
    if "verification" in key or "revision" in key or "correction" in key:
        return "verification_safety"
    return "other"


def infer_unit(key: str) -> str:
    if key.endswith("_ms"):
        return "ms"
    if "tokens" in key:
        return "tokens"
    if "cost" in key:
        return "currency"
    if key.endswith("_count") or key.startswith("total_"):
        return "count"
    return "ratio"
