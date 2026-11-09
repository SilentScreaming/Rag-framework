"""Adapters from official benchmark releases to the canonical RAG schema."""

from __future__ import annotations

import json
import hashlib
import os
import random
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel

from rag_benchmark.dataset_catalog import BenchmarkDatasetSpec, sha256_file
from rag_benchmark.schemas import (
    AnnotationRecord,
    CandidateResponseRecord,
    ContextSetRecord,
    CorpusChunkRecord,
    ExampleRecord,
    QrelRecord,
    SplitMemberRecord,
)


JsonObject = dict[str, Any]
SECTION_RE = re.compile(r"^Section::::(.*?)\.?\s*$", flags=re.DOTALL)
PASSAGE_RE = re.compile(r"(?:^|\n\s*)passage\s+(\d+)\s*:\s*", flags=re.IGNORECASE)
CITATION_RE = re.compile(r"\[cite_(\d+)\]")


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.part = path.with_name(path.name + ".part")
        self.handle = None
        self.count = 0

    def __enter__(self) -> "JsonlSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.part.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, row: BaseModel | JsonObject) -> None:
        if self.handle is None:
            raise RuntimeError("JSONL sink is not open")
        payload = row.model_dump(mode="json", exclude_none=False) if isinstance(row, BaseModel) else row
        self.handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        self.handle.write("\n")
        self.count += 1

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            self.part.replace(self.path)
        else:
            self.part.unlink(missing_ok=True)


@dataclass
class OutputSinks:
    examples: JsonlSink
    corpus: JsonlSink
    qrels: JsonlSink
    responses: JsonlSink
    annotations: JsonlSink
    contexts: JsonlSink


def canonical_sinks(stack: ExitStack, processed: Path, *, include_corpus: bool = True) -> OutputSinks:
    empty_corpus = processed / "corpus.jsonl"
    return OutputSinks(
        examples=stack.enter_context(JsonlSink(processed / "examples.jsonl")),
        corpus=(
            stack.enter_context(JsonlSink(empty_corpus))
            if include_corpus
            else JsonlSink(empty_corpus)
        ),
        qrels=stack.enter_context(JsonlSink(processed / "qrels.jsonl")),
        responses=stack.enter_context(JsonlSink(processed / "candidate_responses.jsonl")),
        annotations=stack.enter_context(JsonlSink(processed / "annotations.jsonl")),
        contexts=stack.enter_context(JsonlSink(processed / "context_sets.jsonl")),
    )


def stable_hash(value: str, length: int = 16) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def iter_jsonl(path: Path) -> Iterator[JsonObject]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unique_texts(values: Iterable[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def process_dataset(spec: BenchmarkDatasetSpec, *, overwrite: bool = False) -> JsonObject:
    if spec.key in {"kilt_nq", "kilt_hotpotqa"}:
        result = process_kilt_task(spec, overwrite=overwrite)
    elif spec.key == "alce":
        result = process_alce(spec)
    elif spec.key == "scifact":
        result = process_scifact(spec)
    elif spec.key == "garage":
        result = process_garage(spec)
    elif spec.key == "ragtruth":
        result = process_ragtruth(spec)
    elif spec.key == "rgb":
        result = process_rgb(spec)
    else:
        raise ValueError(f"Unsupported dataset: {spec.key}")
    reports = []
    for processed in result.pop("processed_roots"):
        reports.append(validate_canonical_dataset(Path(processed)))
    result["validation"] = reports
    return result


def process_kilt_task(spec: BenchmarkDatasetSpec, *, overwrite: bool) -> JsonObject:
    shared = spec.root / "processed" / "kilt_wikipedia"
    corpus_path, map_path, corpus_stats = build_kilt_corpus(spec, shared=shared, overwrite=overwrite)
    processed = spec.processed_root
    processed.mkdir(parents=True, exist_ok=True)
    corpus_link = processed / "corpus.jsonl"
    if overwrite and corpus_link.exists():
        corpus_link.unlink()
    if not corpus_link.exists():
        os.link(corpus_path, corpus_link)

    prefix = "nq" if spec.key == "kilt_nq" else "hotpotqa"
    rows_by_split: dict[str, list[str]] = defaultdict(list)
    eligible_by_split: dict[str, list[str]] = defaultdict(list)
    labels = Counter()
    with sqlite3.connect(map_path) as connection, ExitStack() as stack:
        sinks = canonical_sinks(stack, processed, include_corpus=False)
        for asset in spec.assets:
            if not asset.name.startswith(prefix + "-"):
                continue
            path = spec.root / "raw" / asset.name
            for row in iter_jsonl(path):
                source_id = str(row["id"])
                example_id = f"{spec.key}:{asset.split}:{source_id}"
                outputs = list(row.get("output") or [])
                answers = unique_texts(output.get("answer") for output in outputs if output.get("answer"))
                provenance = [item for output in outputs for item in output.get("provenance", [])]
                qrel_rows: dict[str, JsonObject] = {}
                evidence_rows = []
                for item in provenance:
                    wiki_id = str(item.get("wikipedia_id", ""))
                    start = int(item.get("start_paragraph_id", 0))
                    end = int(item.get("end_paragraph_id", start))
                    for paragraph_id in range(start, end + 1):
                        mapped = connection.execute(
                            "SELECT chunk_id, title FROM paragraph_map WHERE wikipedia_id=? AND paragraph_id=?",
                            (wiki_id, paragraph_id),
                        ).fetchone()
                        if mapped is None:
                            continue
                        chunk_id, title = mapped
                        qrel_rows[chunk_id] = {
                            "example_id": example_id,
                            "chunk_id": chunk_id,
                            "relevance": 3,
                            "support_type": "kilt_provenance",
                            "metadata": {
                                "wikipedia_id": wiki_id,
                                "paragraph_id": paragraph_id,
                                "title": title,
                                "section": item.get("section"),
                                "bleu_score": item.get("bleu_score"),
                            },
                        }
                    evidence_spans = item.get("meta", {}).get("evidence_span", [])
                    for evidence_text in unique_texts(evidence_spans):
                        evidence_rows.append(
                            {
                                "doc_id": f"kilt:wiki:{wiki_id}",
                                "title": item.get("title"),
                                "text": evidence_text,
                                "support_type": "kilt_evidence_span",
                                "metadata": {"wikipedia_id": wiki_id, "section": item.get("section")},
                            }
                        )
                sinks.examples.write(
                    ExampleRecord(
                        example_id=example_id,
                        dataset=spec.key,
                        split=asset.split,
                        task_type=spec.role,
                        question=clean_text(row.get("input")),
                        answers=[{"text": answer, "type": "short"} for answer in answers],
                        gold_evidence=evidence_rows,
                        metadata={
                            "source_id": source_id,
                            "source_file": asset.name,
                            "task_track": spec.task_track,
                            "provenance_count": len(provenance),
                            "has_labels": bool(answers),
                            "candidate_scope": "kilt_full_wikipedia_2019",
                        },
                    )
                )
                for qrel in qrel_rows.values():
                    sinks.qrels.write(QrelRecord(**qrel))
                rows_by_split[asset.split].append(example_id)
                provenance_pages = {str(item.get("wikipedia_id")) for item in provenance if item.get("wikipedia_id")}
                eligible = bool(answers and qrel_rows)
                if spec.key == "kilt_hotpotqa":
                    eligible = eligible and len(provenance_pages) >= 2
                if eligible:
                    eligible_by_split[asset.split].append(example_id)
                labels[asset.split] += 1

    write_standard_splits(
        processed,
        deterministic_order(eligible_by_split.get("dev", []), seed=42),
        dataset=spec.key,
        source_split="dev",
        formal_limit=1000,
    )
    corpus_sha256 = corpus_stats.get("corpus_sha256")
    if not corpus_sha256:
        corpus_sha256 = sha256_file(corpus_path)
        corpus_stats = {
            **corpus_stats,
            "corpus_sha256": corpus_sha256,
            "corpus_size_bytes": corpus_path.stat().st_size,
            "mapping_size_bytes": map_path.stat().st_size,
        }
        (shared / "build_report.json").write_text(
            json.dumps(corpus_stats, indent=2),
            encoding="utf-8",
        )
    (processed / "corpus_reference.json").write_text(
        json.dumps(
            {
                "scope": "kilt_full_wikipedia_2019",
                "path": str(corpus_path),
                "sha256": corpus_sha256,
                "size_bytes": corpus_path.stat().st_size,
                "mapping_db": str(map_path),
                "chunks": corpus_stats["chunks"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "dataset": spec.key,
        "processed_roots": [str(processed)],
        "split_counts": dict(labels),
        "eligible_split_counts": {key: len(value) for key, value in eligible_by_split.items()},
        "corpus": corpus_stats,
    }


def build_kilt_corpus(
    spec: BenchmarkDatasetSpec,
    *,
    shared: Path,
    overwrite: bool,
    max_words: int = 220,
    max_paragraphs: int = 3,
) -> tuple[Path, Path, JsonObject]:
    shared.mkdir(parents=True, exist_ok=True)
    corpus_path = shared / "corpus.jsonl"
    map_path = shared / "paragraph_map.sqlite"
    report_path = shared / "build_report.json"
    if corpus_path.exists() and map_path.exists() and report_path.exists() and not overwrite:
        return corpus_path, map_path, json.loads(report_path.read_text(encoding="utf-8"))
    corpus_path.unlink(missing_ok=True)
    map_path.unlink(missing_ok=True)
    knowledge_path = spec.root / "raw" / "kilt_knowledgesource.json"
    page_count = paragraph_count = chunk_count = 0
    with sqlite3.connect(map_path) as connection, JsonlSink(corpus_path) as sink:
        connection.execute(
            "CREATE TABLE paragraph_map (wikipedia_id TEXT NOT NULL, paragraph_id INTEGER NOT NULL, "
            "chunk_id TEXT NOT NULL, title TEXT, PRIMARY KEY(wikipedia_id, paragraph_id))"
        )
        for page in iter_jsonl(knowledge_path):
            wiki_id = str(page.get("wikipedia_id") or page.get("_id"))
            title = clean_text(page.get("wikipedia_title"))
            paragraphs = list(page.get("text") or [])
            groups: list[list[tuple[int, str, bool]]] = []
            current: list[tuple[int, str, bool]] = []
            current_words = 0
            for paragraph_id, raw_text in enumerate(paragraphs):
                text = clean_text(raw_text)
                if not text:
                    continue
                section = SECTION_RE.match(text)
                if section:
                    text = clean_text(section.group(1)) or "Section"
                words = len(text.split())
                if current and (len(current) >= max_paragraphs or current_words + words > max_words):
                    groups.append(current)
                    current = []
                    current_words = 0
                current.append((paragraph_id, text, bool(section)))
                current_words += words
            if current:
                groups.append(current)
            for group in groups:
                start, end = group[0][0], group[-1][0]
                chunk_id = f"kilt:wiki:{wiki_id}:p{start}-{end}"
                sentence_rows = [
                    {
                        "sentence_id": paragraph_id,
                        "text": text,
                        "metadata": {"is_section_heading": is_heading},
                    }
                    for paragraph_id, text, is_heading in group
                ]
                sink.write(
                    CorpusChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=f"kilt:wiki:{wiki_id}",
                        dataset="kilt_wikipedia_2019",
                        title=title,
                        url=page.get("history", {}).get("url"),
                        text="\n".join(text for _, text, _ in group),
                        sentences=sentence_rows,
                        metadata={
                            "wikipedia_id": wiki_id,
                            "paragraph_start": start,
                            "paragraph_end": end,
                            "revision_id": page.get("history", {}).get("revid"),
                            "revision_timestamp": page.get("history", {}).get("timestamp"),
                            "wikidata_id": page.get("wikidata_info", {}).get("wikidata_id"),
                        },
                    )
                )
                connection.executemany(
                    "INSERT INTO paragraph_map VALUES (?, ?, ?, ?)",
                    [(wiki_id, paragraph_id, chunk_id, title) for paragraph_id, _, _ in group],
                )
                paragraph_count += len(group)
                chunk_count += 1
            page_count += 1
            if page_count % 5000 == 0:
                connection.commit()
        connection.commit()
    manifest = json.loads(
        (spec.root / "manifests" / f"manifest-{spec.key}.json").read_text(encoding="utf-8")
    )
    source_sha256 = next(
        asset["sha256"]
        for asset in manifest["assets"]
        if asset["name"] == "kilt_knowledgesource.json"
    )
    report = {
        "pages": page_count,
        "paragraphs": paragraph_count,
        "chunks": chunk_count,
        "max_words": max_words,
        "max_paragraphs": max_paragraphs,
        "source_sha256": source_sha256,
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_size_bytes": corpus_path.stat().st_size,
        "mapping_size_bytes": map_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return corpus_path, map_path, report


def process_alce(spec: BenchmarkDatasetSpec) -> JsonObject:
    raw = spec.root / "raw" / "ALCE-data"
    results = []
    for subset in ("asqa", "qampari"):
        base_rows = json.loads((raw / f"{subset}_eval_gtr_top100.json").read_text(encoding="utf-8"))
        oracle_rows = json.loads(
            (raw / f"{subset}_eval_gtr_top100_reranked_oracle.json").read_text(encoding="utf-8")
        )
        oracle_by_id = {alce_source_id(subset, row): row for row in oracle_rows}
        processed = spec.root / "processed" / f"alce_{subset}"
        ids = []
        with ExitStack() as stack:
            sinks = canonical_sinks(stack, processed)
            for row in base_rows:
                source_id = alce_source_id(subset, row)
                example_id = f"alce_{subset}:eval:{source_id}"
                ids.append(example_id)
                oracle = oracle_by_id[source_id]
                oracle_docs = {alce_doc_key(doc): doc for doc in oracle.get("docs", [])}
                docs = []
                seen = set()
                for origin, source_docs in (("gtr_top100", row.get("docs", [])), ("reranked_oracle", oracle.get("docs", []))):
                    for rank, doc in enumerate(source_docs, start=1):
                        key = alce_doc_key(doc)
                        if key in seen:
                            continue
                        seen.add(key)
                        merged = {**doc, **oracle_docs.get(key, {})}
                        docs.append((origin, rank, merged))
                answers = alce_answers(subset, row)
                gold_evidence = []
                context_ids = []
                for origin, rank, doc in docs:
                    text = clean_text(doc.get("text"))
                    if not text:
                        continue
                    chunk_id = f"alce_{subset}:chunk:{source_id}:{stable_hash(alce_doc_key(doc))}"
                    context_ids.append(chunk_id)
                    answers_found = [int(value) for value in doc.get("answers_found", [])]
                    sinks.corpus.write(
                        CorpusChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=f"alce:{doc.get('id') or stable_hash(alce_doc_key(doc))}",
                            dataset=f"alce_{subset}",
                            title=clean_text(doc.get("title")) or None,
                            url=doc.get("url"),
                            text=text,
                            metadata={
                                "source_id": source_id,
                                "source_example_id": source_id,
                                "retrieval_origin": origin,
                                "retrieval_rank": rank,
                                "retrieval_score": doc.get("score"),
                                "summary": doc.get("summary"),
                                "extraction": doc.get("extraction"),
                                "answers_found": answers_found,
                            },
                        )
                    )
                    if any(answers_found):
                        sinks.qrels.write(
                            QrelRecord(
                                example_id=example_id,
                                chunk_id=chunk_id,
                                relevance=3,
                                support_type="alce_oracle_answer_document",
                                metadata={"answers_found": answers_found, "oracle_rank": rank},
                            )
                        )
                        gold_evidence.append(
                            {
                                "chunk_id": chunk_id,
                                "title": doc.get("title"),
                                "text": clean_text(doc.get("extraction") or doc.get("summary") or text),
                                "support_type": "alce_oracle_answer_document",
                                "metadata": {"answers_found": answers_found},
                            }
                        )
                sinks.examples.write(
                    ExampleRecord(
                        example_id=example_id,
                        dataset=f"alce_{subset}",
                        split="eval",
                        task_type=("ambiguous_long_answer_citation" if subset == "asqa" else "list_answer_citation"),
                        question=clean_text(row.get("question")),
                        answers=answers,
                        gold_evidence=gold_evidence,
                        metadata={
                            "source_id": source_id,
                            "task_track": spec.task_track,
                            "candidate_scope": "alce_gtr_top100_plus_oracle",
                            "retrieved_document_count": len(docs),
                            "qa_pairs": row.get("qa_pairs", []),
                            "annotations": row.get("annotations", []),
                        },
                    )
                )
                sinks.contexts.write(
                    ContextSetRecord(
                        context_set_id=f"alce_{subset}:context:{source_id}",
                        example_id=example_id,
                        dataset=f"alce_{subset}",
                        condition="official_retrieved_top100_plus_oracle",
                        chunk_ids=context_ids,
                        expected_behavior="answer_with_citations",
                        metadata={"retrieved_document_count": len(context_ids)},
                    )
                )
        shuffled = deterministic_order(ids, seed=42)
        write_standard_splits(processed, shuffled, dataset=f"alce_{subset}", source_split="eval", formal_limit=500)
        results.append({"subset": subset, "processed": str(processed), "examples": len(ids)})
    return {"dataset": spec.key, "processed_roots": [item["processed"] for item in results], "subsets": results}


def alce_source_id(subset: str, row: JsonObject) -> str:
    value = row.get("sample_id") if subset == "asqa" else row.get("id")
    return str(value or stable_hash(clean_text(row.get("question"))))


def alce_doc_key(doc: JsonObject) -> str:
    return str(doc.get("id") or doc.get("url") or (clean_text(doc.get("title")) + "\n" + clean_text(doc.get("text"))))


def alce_answers(subset: str, row: JsonObject) -> list[JsonObject]:
    answers = []
    long_answer = clean_text(row.get("answer"))
    if long_answer:
        answers.append({"text": long_answer, "type": "long" if subset == "asqa" else "list"})
    if subset == "asqa":
        for pair_index, pair in enumerate(row.get("qa_pairs", [])):
            for answer in unique_texts(pair.get("short_answers", [])):
                answers.append(
                    {
                        "text": answer,
                        "type": "short",
                        "metadata": {"qa_pair_index": pair_index, "subquestion": pair.get("question")},
                    }
                )
    else:
        for item_index, aliases in enumerate(row.get("answers", [])):
            for answer in unique_texts(aliases if isinstance(aliases, list) else [aliases]):
                answers.append({"text": answer, "type": "list_item", "metadata": {"item_index": item_index}})
    return answers


def process_scifact(spec: BenchmarkDatasetSpec) -> JsonObject:
    raw = spec.root / "raw" / "data"
    processed = spec.processed_root
    corpus_by_id = {str(row["doc_id"]): row for row in iter_jsonl(raw / "corpus.jsonl")}
    ids_by_split: dict[str, list[str]] = defaultdict(list)
    with ExitStack() as stack:
        sinks = canonical_sinks(stack, processed)
        for doc_id, row in corpus_by_id.items():
            sentences = [clean_text(text) for text in row.get("abstract", []) if clean_text(text)]
            sinks.corpus.write(
                CorpusChunkRecord(
                    chunk_id=f"scifact:doc:{doc_id}",
                    doc_id=f"scifact:doc:{doc_id}",
                    dataset="scifact",
                    title=clean_text(row.get("title")) or None,
                    text=" ".join(sentences),
                    sentences=[{"sentence_id": index, "text": text} for index, text in enumerate(sentences)],
                    metadata={"source_doc_id": doc_id, "structured": bool(row.get("structured"))},
                )
            )
        for split in ("train", "dev", "test"):
            for row in iter_jsonl(raw / f"claims_{split}.jsonl"):
                source_id = str(row["id"])
                example_id = f"scifact:{split}:{source_id}"
                evidence = dict(row.get("evidence") or {})
                labels = unique_texts(
                    evidence_set.get("label")
                    for evidence_sets in evidence.values()
                    for evidence_set in evidence_sets
                )
                if split != "test" and not labels:
                    labels = ["NOT_ENOUGH_INFO"]
                qrels: dict[str, int] = {str(doc_id): 2 for doc_id in row.get("cited_doc_ids", [])}
                gold_evidence = []
                for doc_id, evidence_sets in evidence.items():
                    qrels[str(doc_id)] = 3
                    document = corpus_by_id.get(str(doc_id), {})
                    abstract = list(document.get("abstract", []))
                    for set_index, evidence_set in enumerate(evidence_sets):
                        rationale_ids = [int(value) for value in evidence_set.get("sentences", [])]
                        evidence_chunk = f"scifact:doc:{doc_id}"
                        sinks.annotations.write(
                            AnnotationRecord(
                                annotation_id=f"scifact:{source_id}:{doc_id}:{set_index}",
                                example_id=example_id,
                                target_type="evidence",
                                target_id=evidence_chunk,
                                label=str(evidence_set.get("label", "")),
                                evidence_chunk_ids=[evidence_chunk],
                                metadata={"sentence_ids": rationale_ids, "evidence_set_index": set_index},
                            )
                        )
                        for sentence_id in rationale_ids:
                            if 0 <= sentence_id < len(abstract):
                                gold_evidence.append(
                                    {
                                        "doc_id": evidence_chunk,
                                        "chunk_id": evidence_chunk,
                                        "title": document.get("title"),
                                        "sentence_id": sentence_id,
                                        "text": clean_text(abstract[sentence_id]),
                                        "support_type": "scifact_rationale",
                                        "metadata": {"label": evidence_set.get("label")},
                                    }
                                )
                for doc_id, relevance in qrels.items():
                    sinks.qrels.write(
                        QrelRecord(
                            example_id=example_id,
                            chunk_id=f"scifact:doc:{doc_id}",
                            relevance=relevance,
                            support_type="scifact_rationale" if relevance == 3 else "scifact_cited_document",
                            metadata={"verification_labels": labels},
                        )
                    )
                sinks.examples.write(
                    ExampleRecord(
                        example_id=example_id,
                        dataset="scifact",
                        split=split,
                        task_type="scientific_claim_verification",
                        question=clean_text(row.get("claim")),
                        answers=[{"text": label, "type": "verification_label"} for label in labels],
                        gold_evidence=gold_evidence,
                        metadata={
                            "source_id": source_id,
                            "task_track": spec.task_track,
                            "candidate_scope": "scifact_full_corpus",
                            "has_labels": split != "test",
                        },
                    )
                )
                ids_by_split[split].append(example_id)
    calibration_ids = deterministic_order(ids_by_split["train"], seed=42)[:300]
    formal_ids = deterministic_order(ids_by_split["dev"], seed=42)
    write_explicit_splits(
        processed,
        dataset="scifact",
        mvp_ids=calibration_ids[:100],
        calibration_ids=calibration_ids,
        formal_ids=formal_ids,
        source_splits={
            "mvp": "train",
            "dev_small": "train",
            "formal": "dev",
        },
    )
    return {"dataset": spec.key, "processed_roots": [str(processed)], "split_counts": {k: len(v) for k, v in ids_by_split.items()}}


def process_garage(spec: BenchmarkDatasetSpec) -> JsonObject:
    processed = spec.processed_root
    ids = []
    strata: dict[str, list[str]] = defaultdict(list)
    label_counts = Counter()
    with ExitStack() as stack:
        sinks = canonical_sinks(stack, processed)
        for row in iter_jsonl(spec.root / "raw" / "GaRAGe_benchmark.jsonl"):
            source_id = str(row["sample_id"])
            example_id = f"garage:all:{source_id}"
            ids.append(example_id)
            relevant_labels = [clean_text(value).upper() for value in row.get("evidence_relevant", [])]
            labels = []
            for index, value in enumerate(row.get("evidence_correct", [])):
                label = clean_text(value).upper()
                if not label:
                    is_irrelevant = index < len(relevant_labels) and relevant_labels[index] != "YES"
                    label = "IRRELEVANT" if is_irrelevant else "UNANNOTATED"
                labels.append(label)
            has_answer_evidence = "ANSWER-THE-QUESTION" in labels
            from rag_benchmark.data_semantics import garage_semantics
            semantics = garage_semantics(row)
            expected_behavior = semantics["expected_behavior"]
            strata[f"{expected_behavior}:{row.get('question_type')}"].append(example_id)
            context_ids = []
            gold_evidence = []
            for index, grounding in enumerate(row.get("grounding", []), start=1):
                citation_key = next((key for key in grounding if str(key).startswith("cite_")), None)
                text = clean_text(grounding.get(citation_key)) if citation_key else ""
                if not text:
                    continue
                chunk_id = f"garage:chunk:{source_id}:{index}"
                context_ids.append(chunk_id)
                relevance_label = labels[index - 1] if index <= len(labels) else "UNKNOWN"
                relevant = str(row.get("evidence_relevant", [])[index - 1]).upper() if index <= len(row.get("evidence_relevant", [])) else "NO"
                relevance = garage_relevance_grade(relevant, relevance_label)
                cited = str(row.get("evidence_cited", [])[index - 1]).upper() == "YES" if index <= len(row.get("evidence_cited", [])) else False
                label_counts[relevance_label] += 1
                sinks.corpus.write(
                    CorpusChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=f"garage:doc:{source_id}:{index}",
                        dataset="garage",
                        title=f"GaRAGe evidence {index}",
                        text=text,
                        metadata={
                            "source_id": source_id,
                            "source_example_id": source_id,
                            "citation_marker": citation_key or f"cite_{index}",
                            "provider": grounding.get("provider"),
                            "age": grounding.get("age"),
                            "date": grounding.get("date"),
                            "relevance_label": relevance_label,
                            "evidence_relevant": relevant,
                            "gold_cited": cited,
                        },
                    )
                )
                sinks.qrels.write(
                    QrelRecord(
                        example_id=example_id,
                        chunk_id=chunk_id,
                        relevance=relevance,
                        support_type=f"garage_{relevance_label.lower().replace('-', '_')}",
                        metadata={"citation_marker": citation_key, "gold_cited": cited},
                    )
                )
                sinks.annotations.write(
                    AnnotationRecord(
                        annotation_id=f"garage:{source_id}:evidence:{index}",
                        example_id=example_id,
                        target_type="evidence",
                        target_id=chunk_id,
                        label=relevance_label,
                        evidence_chunk_ids=[chunk_id],
                        metadata={"relevance_grade": relevance, "gold_cited": cited},
                    )
                )
                if relevance == 3:
                    gold_evidence.append(
                        {
                            "doc_id": f"garage:doc:{source_id}:{index}",
                            "chunk_id": chunk_id,
                            "title": f"GaRAGe evidence {index}",
                            "text": text,
                            "support_type": "answer_the_question",
                            "metadata": {"citation_marker": citation_key},
                        }
                    )
            answer = clean_text(row.get("answer_generate") or row.get("answer_related_info"))
            sinks.examples.write(
                ExampleRecord(
                    example_id=example_id,
                    dataset="garage",
                    split="all",
                    task_type="grounded_long_answer_or_deflection",
                    question=clean_text(row.get("question")),
                    answers=[{"text": answer, "type": "long"}] if answer else [],
                    gold_evidence=gold_evidence,
                    metadata={
                        "source_id": source_id,
                        "task_track": spec.task_track,
                        "expected_behavior": expected_behavior,
                        **semantics,
                        "question_date": row.get("question_date"),
                        "question_valid": row.get("question_valid"),
                        "question_false_premise": row.get("question_false_premise"),
                        "question_type": row.get("question_type"),
                        "question_complexity": row.get("question_complexity"),
                        "question_category": row.get("question_category"),
                        "question_popularity": row.get("question_popularity"),
                        "answer_validated": row.get("answer_validate"),
                        "gold_citation_markers": CITATION_RE.findall(answer),
                        "candidate_scope": "per_example_provided_context",
                    },
                )
            )
            sinks.contexts.write(
                ContextSetRecord(
                    context_set_id=f"garage:context:{source_id}",
                    example_id=example_id,
                    dataset="garage",
                    condition="annotated_grounding",
                    chunk_ids=context_ids,
                    expected_behavior=expected_behavior,
                    metadata={"question_date": row.get("question_date")},
                )
            )
    ordered = round_robin_strata(strata, seed=42)
    write_standard_splits(processed, ordered, dataset="garage", source_split="all", formal_limit=1000)
    return {"dataset": spec.key, "processed_roots": [str(processed)], "examples": len(ids), "evidence_labels": dict(label_counts)}


def garage_relevance_grade(relevant: str, label: str) -> int:
    if relevant != "YES" or label == "IRRELEVANT":
        return 0
    if label == "ANSWER-THE-QUESTION":
        return 3
    if label == "RELATED-INFORMATION":
        return 2
    return 1


def process_ragtruth(spec: BenchmarkDatasetSpec) -> JsonObject:
    raw = spec.root / "raw"
    sources = {
        str(row["source_id"]): row
        for row in iter_jsonl(raw / "source_info.jsonl")
        if str(row.get("task_type", "")).upper() == "QA"
    }
    responses_by_source: dict[str, list[JsonObject]] = defaultdict(list)
    for row in iter_jsonl(raw / "response.jsonl"):
        if str(row.get("source_id")) in sources:
            responses_by_source[str(row["source_id"])].append(row)
    processed = spec.processed_root
    response_strata: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with ExitStack() as stack:
        sinks = canonical_sinks(stack, processed)
        for source_id, source in sources.items():
            source_info = dict(source.get("source_info") or {})
            example_id = f"ragtruth:qa:{source_id}"
            source_responses = responses_by_source.get(source_id, [])
            splits = Counter(str(row.get("split", "train")) for row in source_responses)
            split = splits.most_common(1)[0][0] if splits else "train"
            chunk_ids = []
            for passage_index, passage in enumerate(parse_ragtruth_passages(source_info.get("passages")), start=1):
                chunk_id = f"ragtruth:chunk:{source_id}:{passage_index}"
                chunk_ids.append(chunk_id)
                sinks.corpus.write(
                    CorpusChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=f"ragtruth:doc:{source_id}:{passage_index}",
                        dataset="ragtruth_qa",
                        title=f"Passage {passage_index}",
                        text=passage,
                        metadata={"source_id": source_id, "source_example_id": source_id, "passage_index": passage_index},
                    )
                )
                sinks.qrels.write(
                    QrelRecord(
                        example_id=example_id,
                        chunk_id=chunk_id,
                        relevance=2,
                        support_type="provided_rag_context",
                        metadata={"passage_index": passage_index},
                    )
                )
            sinks.examples.write(
                ExampleRecord(
                    example_id=example_id,
                    dataset="ragtruth_qa",
                    split=split,
                    task_type="hallucination_judge_calibration",
                    question=clean_text(source_info.get("question")),
                    metadata={
                        "source_id": source_id,
                        "task_track": spec.task_track,
                        "source": source.get("source"),
                        "candidate_scope": "per_example_provided_context",
                        "candidate_response_count": len(source_responses),
                        "has_gold_answer": False,
                    },
                )
            )
            sinks.contexts.write(
                ContextSetRecord(
                    context_set_id=f"ragtruth:context:{source_id}",
                    example_id=example_id,
                    dataset="ragtruth_qa",
                    condition="provided_rag_context",
                    chunk_ids=chunk_ids,
                    expected_behavior="judge_candidate_response",
                )
            )
            for response in source_responses:
                response_id = f"ragtruth:response:{response['id']}"
                labels = list(response.get("labels") or [])
                response_text = str(response.get("response") or "")
                has_hallucination = bool(labels)
                sinks.responses.write(
                    CandidateResponseRecord(
                        response_id=response_id,
                        example_id=example_id,
                        dataset="ragtruth_qa",
                        split=str(response.get("split", split)),
                        text=response_text,
                        model=str(response.get("model")) if response.get("model") else None,
                        metadata={
                            "temperature": response.get("temperature"),
                            "quality": response.get("quality"),
                            "has_hallucination": has_hallucination,
                            "hallucination_span_count": len(labels),
                            "implicit_true_span_count": sum(
                                bool(label.get("implicit_true", False)) for label in labels
                            ),
                        },
                    )
                )
                sinks.annotations.write(
                    AnnotationRecord(
                        annotation_id=f"{response_id}:overall",
                        example_id=example_id,
                        target_type="response",
                        target_id=response_id,
                        label="hallucinated" if has_hallucination else "grounded",
                        metadata={"quality": response.get("quality")},
                    )
                )
                for label_index, label in enumerate(labels):
                    start_char = int(label["start"]) if label.get("start") is not None else None
                    end_char = int(label["end"]) if label.get("end") is not None else None
                    span_text = str(label.get("text") or "") or None
                    if start_char is not None and end_char is not None and span_text is not None:
                        actual_span = response_text[start_char:end_char]
                        if actual_span != span_text:
                            raise ValueError(
                                f"RAGTruth span mismatch for response {response['id']}: "
                                f"[{start_char}:{end_char}]"
                            )
                    sinks.annotations.write(
                        AnnotationRecord(
                            annotation_id=f"{response_id}:span:{label_index}",
                            example_id=example_id,
                            target_type="span",
                            target_id=response_id,
                            label=str(label.get("label_type") or "hallucination"),
                            start_char=start_char,
                            end_char=end_char,
                            text=span_text,
                            metadata={
                                "meta": label.get("meta"),
                                "due_to_null": label.get("due_to_null"),
                                "implicit_true": label.get("implicit_true", False),
                            },
                        )
                    )
                key = f"{response.get('model')}:{has_hallucination}:{response.get('split')}"
                response_strata[key].append((response_id, example_id, str(response.get("split", split))))
    write_standard_splits(
        processed,
        deterministic_order(list(sources), seed=42, prefix="ragtruth:qa:"),
        dataset="ragtruth_qa",
        source_split="mixed",
        formal_limit=len(sources),
        calibration_limit=0,
    )
    write_ragtruth_judge_splits(
        processed,
        response_strata,
        calibration_limit=1000,
        evaluation_limit=1000,
    )
    return {
        "dataset": spec.key,
        "processed_roots": [str(processed)],
        "qa_sources": len(sources),
        "qa_responses": sum(len(rows) for rows in responses_by_source.values()),
    }


def parse_ragtruth_passages(value: Any) -> list[str]:
    if isinstance(value, list):
        return unique_texts(value)
    text = str(value or "")
    matches = list(PASSAGE_RE.finditer(text))
    if not matches:
        return [clean_text(text)] if clean_text(text) else []
    passages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        passage = clean_text(text[match.end() : end])
        if passage:
            passages.append(passage)
    return passages


def write_ragtruth_judge_splits(
    processed: Path,
    strata: dict[str, list[tuple[str, str, str]]],
    *,
    calibration_limit: int,
    evaluation_limit: int,
) -> None:
    partitions: dict[str, dict[str, list[tuple[str, str, str]]]] = {
        "judge_calibration": defaultdict(list),
        "judge_eval": defaultdict(list),
    }
    for stratum, rows in strata.items():
        for row in rows:
            example_id = row[1]
            bucket = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) % 5
            profile = "judge_calibration" if bucket < 2 else "judge_eval"
            partitions[profile][stratum].append(row)
    limits = {"judge_calibration": calibration_limit, "judge_eval": evaluation_limit}
    for profile, partition in partitions.items():
        selected = stratified_response_sample(partition, limit=limits[profile], seed=42)
        with JsonlSink(processed / "splits" / f"{profile}.jsonl") as sink:
            for response_id, example_id, source_split in selected:
                sink.write(
                    {
                        "schema_version": "response-split-member-v1",
                        "response_id": response_id,
                        "example_id": example_id,
                        "dataset": "ragtruth_qa",
                        "source_split": source_split,
                        "split_profile": profile,
                    }
                )


def stratified_response_sample(
    strata: dict[str, list[tuple[str, str, str]]],
    *,
    limit: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    pools = {key: sorted(rows) for key, rows in strata.items()}
    for rows in pools.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < limit and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(selected) < limit:
                selected.append(pools[key].pop())
    return selected


def process_rgb(spec: BenchmarkDatasetSpec) -> JsonObject:
    processed = spec.processed_root
    raw = spec.root / "raw"
    ids = []
    with ExitStack() as stack:
        sinks = canonical_sinks(stack, processed)
        for filename, scenario in (
            ("en_refine.json", "noise_and_rejection"),
            ("en_int.json", "information_integration"),
            ("en_fact.json", "counterfactual_robustness"),
        ):
            for row in iter_jsonl(raw / filename):
                source_id = f"{scenario}:{row['id']}"
                example_id = f"rgb:{source_id}"
                ids.append(example_id)
                answers = rgb_answers(row)
                positive_ids = write_rgb_documents(sinks, example_id, source_id, "positive", row.get("positive", []), relevance=3)
                negative_ids = write_rgb_documents(sinks, example_id, source_id, "negative", row.get("negative", []), relevance=0)
                wrong_ids = write_rgb_documents(
                    sinks,
                    example_id,
                    source_id,
                    "positive_wrong",
                    row.get("positive_wrong", []),
                    relevance=0,
                )
                sinks.examples.write(
                    ExampleRecord(
                        example_id=example_id,
                        dataset="rgb_en",
                        split=scenario,
                        task_type=scenario,
                        question=clean_text(row.get("query")),
                        answers=answers,
                        metadata={
                            "source_id": source_id,
                            "task_track": spec.task_track,
                            "candidate_scope": "per_example_provided_context",
                            "fake_answer": row.get("fakeanswer"),
                            "answer_part_1": row.get("asnwer1"),
                            "answer_part_2": row.get("answer2"),
                        },
                    )
                )
                if scenario == "noise_and_rejection":
                    write_rgb_noise_contexts(sinks, example_id, source_id, positive_ids, negative_ids)
                elif scenario == "information_integration":
                    sinks.contexts.write(
                        ContextSetRecord(
                            context_set_id=f"rgb:context:{source_id}:integration",
                            example_id=example_id,
                            dataset="rgb_en",
                            condition="information_integration",
                            chunk_ids=positive_ids,
                            expected_behavior="integrate_all_relevant_passages",
                        )
                    )
                else:
                    sinks.contexts.write(
                        ContextSetRecord(
                            context_set_id=f"rgb:context:{source_id}:counterfactual",
                            example_id=example_id,
                            dataset="rgb_en",
                            condition="counterfactual_context",
                            chunk_ids=wrong_ids,
                            expected_behavior="detect_or_correct_false_information",
                            metadata={"fake_answer": row.get("fakeanswer")},
                        )
                    )
                    sinks.contexts.write(
                        ContextSetRecord(
                            context_set_id=f"rgb:context:{source_id}:factual-control",
                            example_id=example_id,
                            dataset="rgb_en",
                            condition="factual_control",
                            chunk_ids=positive_ids,
                            expected_behavior="answer",
                        )
                    )
                    if row.get("fakeanswer"):
                        sinks.annotations.write(
                            AnnotationRecord(
                                annotation_id=f"rgb:{source_id}:fakeanswer",
                                example_id=example_id,
                                target_type="example",
                                label="counterfactual_answer",
                                text=clean_text(row.get("fakeanswer")),
                                evidence_chunk_ids=wrong_ids,
                            )
                        )
    write_standard_splits(
        processed,
        ids,
        dataset="rgb_en",
        source_split="mixed",
        formal_limit=len(ids),
        calibration_limit=0,
    )
    return {"dataset": spec.key, "processed_roots": [str(processed)], "english_examples": len(ids)}


def rgb_answers(row: JsonObject) -> list[JsonObject]:
    raw = row.get("answer", [])
    values = raw if isinstance(raw, list) else [raw]
    answers = [{"text": text, "type": "short"} for text in unique_texts(values)]
    for field, label in (("asnwer1", "integration_part_1"), ("answer2", "integration_part_2")):
        values = row.get(field, [])
        values = values if isinstance(values, list) else [values]
        answers.extend({"text": text, "type": label} for text in unique_texts(values))
    return answers


def write_rgb_documents(
    sinks: OutputSinks,
    example_id: str,
    source_id: str,
    kind: str,
    documents: Iterable[Any],
    *,
    relevance: int,
) -> list[str]:
    from rag_benchmark.rgb_protocol import rgb_display_title
    chunk_ids = []
    for index, document in enumerate(documents, start=1):
        text = clean_text(document)
        if not text:
            continue
        chunk_id = f"rgb:chunk:{stable_hash(source_id)}:{kind}:{index}"
        chunk_ids.append(chunk_id)
        sinks.corpus.write(
            CorpusChunkRecord(
                chunk_id=chunk_id,
                doc_id=chunk_id.replace(":chunk:", ":doc:"),
                dataset="rgb_en",
                title=rgb_display_title(chunk_id),
                text=text,
                metadata={"source_id": source_id, "source_example_id": source_id, "context_kind": kind},
            )
        )
        sinks.qrels.write(
            QrelRecord(
                example_id=example_id,
                chunk_id=chunk_id,
                relevance=relevance,
                support_type=f"rgb_{kind}",
            )
        )
    return chunk_ids


def write_rgb_noise_contexts(
    sinks: OutputSinks,
    example_id: str,
    source_id: str,
    positive_ids: list[str],
    negative_ids: list[str],
    *,
    passage_count: int = 5,
) -> None:
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        positive_count = min(len(positive_ids), round(passage_count * (1.0 - rate)))
        negative_count = min(len(negative_ids), passage_count - positive_count)
        selected = positive_ids[:positive_count] + negative_ids[:negative_count]
        sinks.contexts.write(
            ContextSetRecord(
                context_set_id=f"rgb:context:{source_id}:noise-{rate:.1f}",
                example_id=example_id,
                dataset="rgb_en",
                condition="noise_robustness",
                chunk_ids=selected,
                expected_behavior="abstain" if positive_count == 0 else "answer",
                metadata={"noise_rate": rate, "passage_count": len(selected), "positive_count": positive_count},
            )
        )


def deterministic_order(values: Iterable[str], *, seed: int, prefix: str = "") -> list[str]:
    normalized = [value if not prefix or str(value).startswith(prefix) else prefix + str(value) for value in values]
    normalized.sort()
    random.Random(seed).shuffle(normalized)
    return normalized


def round_robin_strata(strata: dict[str, list[str]], *, seed: int) -> list[str]:
    rng = random.Random(seed)
    for rows in strata.values():
        rows.sort()
        rng.shuffle(rows)
    ordered = []
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                ordered.append(strata[key].pop())
    return ordered


def write_standard_splits(
    processed: Path,
    ordered_ids: list[str],
    *,
    dataset: str,
    source_split: str,
    formal_limit: int,
    calibration_limit: int = 300,
) -> None:
    calibration_ids = ordered_ids[: min(calibration_limit, len(ordered_ids))]
    formal_start = len(calibration_ids)
    formal_ids = ordered_ids[formal_start : formal_start + formal_limit]
    required_formal = min(formal_limit, max(0, len(ordered_ids) - formal_start))
    if len(formal_ids) != required_formal:
        raise ValueError(
            f"Unable to create disjoint split for {dataset}: calibration={len(calibration_ids)}, "
            f"formal={len(formal_ids)}, available={len(ordered_ids)}"
        )
    if formal_limit > len(formal_ids) and calibration_ids:
        raise ValueError(
            f"Not enough disjoint examples for {dataset}: calibration={len(calibration_ids)}, "
            f"formal_requested={formal_limit}, available={len(ordered_ids)}"
        )
    write_explicit_splits(
        processed,
        dataset=dataset,
        mvp_ids=(calibration_ids or formal_ids)[:100],
        calibration_ids=calibration_ids,
        formal_ids=formal_ids,
        source_splits={
            "mvp": source_split,
            "dev_small": source_split,
            "formal": source_split,
        },
    )


def write_explicit_splits(
    processed: Path,
    *,
    dataset: str,
    mvp_ids: list[str],
    calibration_ids: list[str],
    formal_ids: list[str],
    source_splits: dict[str, str],
) -> None:
    profiles = {
        "mvp": mvp_ids,
        "dev_small": calibration_ids,
        "formal": formal_ids,
    }
    for profile, example_ids in profiles.items():
        with JsonlSink(processed / "splits" / f"{profile}.jsonl") as sink:
            for example_id in example_ids:
                sink.write(
                    SplitMemberRecord(
                        example_id=example_id,
                        dataset=dataset,
                        source_split=source_splits[profile],
                        split_profile=profile,
                    )
                )


def validate_canonical_dataset(processed: Path) -> JsonObject:
    schema_by_name = {
        "examples.jsonl": ExampleRecord,
        "corpus.jsonl": CorpusChunkRecord,
        "qrels.jsonl": QrelRecord,
        "candidate_responses.jsonl": CandidateResponseRecord,
        "annotations.jsonl": AnnotationRecord,
        "context_sets.jsonl": ContextSetRecord,
    }
    counts = {}
    corpus_reference_path = processed / "corpus_reference.json"
    corpus_reference = (
        json.loads(corpus_reference_path.read_text(encoding="utf-8"))
        if corpus_reference_path.exists()
        else None
    )
    example_ids = set()
    response_ids = set()
    context_set_ids = set()
    required_chunk_ids = set()
    referenced_qrels: list[tuple[str, int, str]] = []
    for name, schema in schema_by_name.items():
        path = processed / name
        if name == "corpus.jsonl" and corpus_reference is not None:
            referenced_path = Path(corpus_reference["path"])
            if not referenced_path.exists() or not os.path.samefile(path, referenced_path):
                raise ValueError("KILT corpus.jsonl is not the declared shared-corpus hard link")
            if path.stat().st_size != int(corpus_reference["size_bytes"]):
                raise ValueError("KILT shared corpus size no longer matches its build report")
            with path.open("r", encoding="utf-8") as handle:
                first_row = next((json.loads(line) for line in handle if line.strip()), None)
            if first_row is None:
                raise ValueError("referenced KILT corpus is empty")
            schema.model_validate(first_row)
            counts[name] = int(corpus_reference["chunks"])
            continue
        count = 0
        for row in iter_jsonl(path):
            record = schema.model_validate(row)
            count += 1
            if name == "examples.jsonl":
                if record.example_id in example_ids:
                    raise ValueError(f"duplicate example_id: {record.example_id}")
                example_ids.add(record.example_id)
            elif name == "qrels.jsonl":
                if record.example_id not in example_ids:
                    raise ValueError(f"qrel references unknown example: {record.example_id}")
                required_chunk_ids.add(record.chunk_id)
                if corpus_reference is not None:
                    wiki_id = record.metadata.get("wikipedia_id")
                    paragraph_id = record.metadata.get("paragraph_id")
                    if wiki_id is None or paragraph_id is None:
                        raise ValueError(f"KILT qrel lacks provenance key: {record.example_id}")
                    referenced_qrels.append((str(wiki_id), int(paragraph_id), record.chunk_id))
            elif name == "candidate_responses.jsonl":
                if record.example_id not in example_ids:
                    raise ValueError(f"response references unknown example: {record.example_id}")
                if record.response_id in response_ids:
                    raise ValueError(f"duplicate response_id: {record.response_id}")
                response_ids.add(record.response_id)
            elif name == "annotations.jsonl":
                if record.example_id not in example_ids:
                    raise ValueError(f"annotation references unknown example: {record.example_id}")
                if record.target_type in {"response", "span"} and record.target_id not in response_ids:
                    raise ValueError(f"annotation references unknown response: {record.target_id}")
                required_chunk_ids.update(record.evidence_chunk_ids)
            elif name == "context_sets.jsonl":
                if record.example_id not in example_ids:
                    raise ValueError(f"context set references unknown example: {record.example_id}")
                if record.context_set_id in context_set_ids:
                    raise ValueError(f"duplicate context_set_id: {record.context_set_id}")
                context_set_ids.add(record.context_set_id)
                required_chunk_ids.update(record.chunk_ids)
        counts[name] = count
    if corpus_reference is not None:
        with sqlite3.connect(corpus_reference["mapping_db"]) as connection:
            for wiki_id, paragraph_id, expected_chunk_id in referenced_qrels:
                row = connection.execute(
                    "SELECT chunk_id FROM paragraph_map WHERE wikipedia_id=? AND paragraph_id=?",
                    (wiki_id, paragraph_id),
                ).fetchone()
                if row is None or row[0] != expected_chunk_id:
                    raise ValueError(
                        f"KILT qrel mapping mismatch: {wiki_id}:{paragraph_id} -> {expected_chunk_id}"
                    )
        required_chunk_ids.clear()
    missing = set(required_chunk_ids)
    for row in iter_jsonl(processed / "corpus.jsonl"):
        missing.discard(str(row["chunk_id"]))
        if not missing:
            break
    if missing:
        raise ValueError(f"qrels reference {len(missing)} missing chunks; first={sorted(missing)[:3]}")
    report = {"processed_root": str(processed), "counts": counts, "status": "ok"}
    (processed / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
