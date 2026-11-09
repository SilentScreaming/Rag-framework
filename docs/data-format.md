# Canonical data format

Dataset adapters normalize source formats into UTF-8 JSON Lines: one JSON object per nonempty line. The authoritative Python records and loaders are in [`data.py`](../src/rag_benchmark/data.py); task-specific output records are in [`schemas.py`](../src/rag_benchmark/schemas.py).

## Directory and configuration

```text
processed/
  examples.jsonl
  corpus.jsonl
  qrels.jsonl
  splits/
    mvp.jsonl
    formal.jsonl
  context_sets.jsonl          Optional
  candidate_responses.jsonl  Optional
  annotations.jsonl          Optional
```

`configs/datasets/<name>.example.yaml` supplies the dataset name, task track, source version, root key, and processing settings. The root key resolves through `datasets.<root_key>` in the path configuration. A root can point directly to the directory containing `examples.jsonl`, or to its parent containing `processed/`.

For a custom source, start from [`toy.example.yaml`](../configs/datasets/toy.example.yaml) and the [authored toy records](../examples/toy/processed). Keep stable identifiers and a version that changes when the data or interpretation changes.

## Required records

The following examples illustrate field shapes. They are authored synthetic content.

### `examples.jsonl`

```json
{"example_id":"demo-1","dataset":"demo","split":"example","task_type":"short_answer_qa","question":"What color is the sample marker?","answers":[{"text":"blue"}],"gold_evidence":[],"metadata":{"source_id":"demo-1"}}
```

`example_id`, `dataset`, `split`, `task_type`, and `question` are required by the loader. `answers`, `gold_evidence`, and `metadata` default to empty collections. Task evaluators may require structured answer groups, entity aliases, labels, or other metadata; a short-answer record is not a substitute for those structures.

### `corpus.jsonl`

```json
{"chunk_id":"demo-doc-1:0","doc_id":"demo-doc-1","dataset":"demo","title":"Sample card","text":"The sample marker is blue.","sentences":[{"sentence_id":0,"text":"The sample marker is blue."}],"metadata":{"source_example_id":"demo-1"}}
```

Required loader fields are `chunk_id`, `doc_id`, `dataset`, and `text`. Preserve document and sentence identities when a dataset supplies rationale annotations. A sentence list does not itself establish that the sentences are gold evidence.

### `qrels.jsonl`

```json
{"example_id":"demo-1","chunk_id":"demo-doc-1:0","relevance":3,"support_type":"direct","metadata":{}}
```

Qrels link examples to corpus chunks. The standard relevance threshold is 3, with lower grades available where an adapter supplies them. The precise meaning of lower grades is dataset-dependent. Use an empty file when no relevance labels exist; do not fabricate qrels to make retrieval metrics available.

### Split files

```json
{"example_id":"demo-1"}
```

`--split-profile mvp` selects `splits/mvp.jsonl`. Response-level tasks may also include `response_id`. The selected split file is required: the loader raises an error if it is missing. Keep split membership explicit and record the sampling design used to produce it. Standard adapter outputs include separate `dev_small` and `formal` splits; response-judgment datasets can supply their own named splits.

## Corpus scope

- `full_corpus` searches the complete canonical corpus. The toy dataset uses this mode.
- `per_example_pooled_context` pools source-associated chunks for the selected examples, plus required gold chunks. Set each example's `metadata.source_id` and corresponding chunks' `metadata.source_example_id` consistently. This is a restricted candidate universe, not open-corpus retrieval.
- `scifact_full_corpus` retains the SciFact-specific full-corpus mode.
- `kilt_full_wikipedia_2019` requires the KILT full-corpus/candidate workflow for retrieval. The runner refuses to replace missing FullWiki infrastructure with the selected examples' gold chunks. A frozen candidate cache remains candidate-restricted retrieval and must be described that way.

Report corpus scope, candidate construction, chunking, and candidate depth alongside retrieval scores. An oracle configuration deliberately uses gold relevance information and must stay separate from ordinary retrieval comparisons.

## Optional records

| File | Fields and use |
| --- | --- |
| `context_sets.jsonl` | `context_set_id`, `example_id`, `dataset`, `condition`, `chunk_ids`, `expected_behavior`, and optional `metadata`; represents a particular evidence condition. |
| `candidate_responses.jsonl` | `response_id`, `example_id`, `dataset`, `split`, `text`, optional `model` and `metadata`; supports scoring supplied responses rather than generating new ones. |
| `annotations.jsonl` | `annotation_id`, `example_id`, `target_type`, optional `target_id`, `label`, span offsets, `text`, `evidence_chunk_ids`, and `metadata`; supports response spans, rationale labels, and other task annotations. |

Context-sensitive runs must resolve exactly one context set per example using its condition and, where needed, metadata filters. Do not merge different context conditions into one pairing identity. Character offsets must refer to the original target text; changing that text can invalidate span labels.

## Adding or acquiring data

1. Review the upstream source, version, license, and expected files. The preparation command without action flags prints this plan.
2. Download through the supported adapter or obtain the files manually in the planned layout.
3. Normalize records while preserving source identities, answer structure, and annotation provenance.
4. Run the adapter's structural validation before executing a pipeline.
5. Review task semantics separately: structural validity cannot verify that evidence labels, expected behavior, or references are correct.

The source catalog records download and revision information. External sources can change or require manual access; the presence of a catalog entry is not a guarantee that every upstream download remains available. Do not redistribute raw data, extracted passages, models, or derived assets unless their terms permit it.

