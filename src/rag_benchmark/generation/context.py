"""One evidence-selection and budget boundary for every generation strategy.

Compression chooses content; budgeting limits its serialized representation.
Neither the citation renderer nor the answer parser may select more evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re

from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk
from rag_benchmark.generation.backends import estimate_tokens
from rag_benchmark.rgb_protocol import rgb_display_title


@dataclass(frozen=True)
class PreparedContext:
    evidence: CompressedContext
    contexts: tuple[RetrievedChunk, ...]
    metadata: dict


def compressed_context_from_retrieved(*, query: str, retrieved: list[RetrievedChunk]) -> CompressedContext:
    """Wrap retrieved chunks as evidence units without changing their text (no compression)."""
    units = []
    for index, result in enumerate(retrieved):
        chunk = result.chunk
        rows = chunk.sentences
        # Only use a sentence decomposition when it represents the full chunk.
        represented = " ".join(str(row.get("text", "")) for row in rows)
        if not rows or " ".join(represented.split()) != " ".join(chunk.text.split()):
            rows = [{"text": chunk.text, "sentence_id": None}]
        for position, row in enumerate(rows):
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            sentence_id = row.get("sentence_id")
            units.append(CompressedSentence(
                context_id=f"C{index + 1}-S{position}-{sha256(chunk.chunk_id.encode()).hexdigest()[:12]}",
                chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                title=rgb_display_title(chunk.chunk_id) if chunk.dataset == "rgb_en" else chunk.title,
                sentence_id=int(sentence_id) if sentence_id is not None else None,
                text=text, parent_rank=result.rank, parent_score=result.score,
                selection_score=result.score,
                metadata={"dataset": chunk.dataset, "parent_metadata": dict(result.metadata)},
            ))
    return CompressedContext(query=query, sentences=units, metadata={"compressor": "none"})


def prepare_context(*, query: str, retrieved: list[RetrievedChunk],
                    compressed: CompressedContext | None, max_tokens: int,
                    model: str | None = None) -> PreparedContext:
    if max_tokens < 0:
        raise ValueError("context token budget cannot be negative")
    source = compressed if compressed is not None else compressed_context_from_retrieved(query=query, retrieved=retrieved)
    selected = []
    truncated = 0
    for position, sentence in enumerate(source.sentences):
        if sentence.metadata.get("dataset") == "rgb_en" or sentence.chunk_id.startswith("rgb:chunk:"):
            from rag_benchmark.compression.sentence_select import context_identifier
            sentence = replace(sentence, title=rgb_display_title(sentence.chunk_id),
                context_id=context_identifier(rank=sentence.parent_rank, chunk_id=sentence.chunk_id,
                    sentence_id=sentence.sentence_id, fallback_index=position))
        sentence = replace(sentence, metadata={**sentence.metadata,
            "source_sentence_id": sentence.metadata.get("source_sentence_id", sentence.sentence_id),
            "fully_visible": sentence.metadata.get("fully_visible", True) and not sentence.metadata.get("budget_truncated", False),
            "visible_char_span": sentence.metadata.get("visible_char_span", [0, len(sentence.text)]),
            "source_text_sha256": sentence.metadata.get("source_text_sha256", sha256(sentence.text.encode()).hexdigest())})
        def serialized(candidate):
            return CompressedContext(query=query, sentences=[*selected, candidate]).to_prompt_context()
        candidate = sentence
        if estimate_tokens(serialized(candidate), model=model) > max_tokens:
            # Prefix truncation preserves provenance; recheck the WHOLE serialized
            # context, including titles, citation IDs and separators.
            words = list(re.finditer(r"\S+", sentence.text))
            while words:
                candidate = replace(sentence, text=sentence.text[:words[-1].end()])
                if estimate_tokens(serialized(candidate), model=model) <= max_tokens:
                    break
                count = estimate_tokens(serialized(candidate), model=model)
                remove = max(1, int(len(words) * (count - max_tokens) / max(count, 1)))
                words = words[:-remove]
            if not words:
                break
            truncated += 1
            candidate = replace(candidate, metadata={**candidate.metadata, "budget_truncated": True,
                "fully_visible": False, "visible_char_span": [sentence.metadata["visible_char_span"][0],
                    sentence.metadata["visible_char_span"][0] + len(candidate.text)],
                "source_text_sha256": sentence.metadata["source_text_sha256"]})
        selected.append(candidate)
        if truncated:
            break
    evidence = CompressedContext(query=query, sentences=selected, metadata=dict(source.metadata))
    rendered = evidence.to_prompt_context()
    used = estimate_tokens(rendered, model=model) if rendered else 0
    if used > max_tokens:
        raise ValueError("prepared context exceeds configured budget")
    contexts = tuple(RetrievedChunk(
        chunk=CorpusChunk(s.chunk_id, s.doc_id, s.metadata.get("dataset", ""), s.title, s.text,
                          [{"sentence_id": s.sentence_id, "text": s.text}], {}),
        rank=s.parent_rank, score=s.parent_score,
        metadata={"context_id": s.context_id, "budget_truncated": bool(s.metadata.get("budget_truncated")),
                  "prepared_prompt_block": s.prompt_block()},
    ) for s in selected)
    metadata = {"schema_version": "prepared-context-v1", "max_tokens": max_tokens,
        "used_tokens": used, "token_count_source": "local_tokenizer_estimate",
        "budget_scope": "serialized_evidence_with_citation_ids",
        "compression": source.metadata.get("compressor", "explicit" if compressed is not None else "none"),
        "chunks_available": len(retrieved), "chunks_used": len({s.chunk_id for s in selected}),
        "truncated_units": truncated, "omitted_units": len(source.sentences) - len(selected),
        "content_sha256": sha256(rendered.encode()).hexdigest(),
        "visible_evidence": [{"context_id": s.context_id, "chunk_id": s.chunk_id,
            "sentence_id": s.sentence_id, "title": s.title, "text": s.text, "metadata": s.metadata} for s in selected]}
    return PreparedContext(evidence=evidence, contexts=contexts, metadata=metadata)
