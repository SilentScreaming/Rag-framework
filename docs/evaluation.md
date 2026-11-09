# Evaluation semantics

The framework separates task performance, evidence behavior, and resource use. It does not combine them into one overall RAG score. Available metrics depend on the dataset, predictions, reference annotations, and configured evaluation components.

## What each dimension measures

| Dimension | Examples | Interpretation and requirements |
| --- | --- | --- |
| Answer quality | Exact match, token F1, grouped answer recall, list precision/recall | Compares answers with task-specific references. Answer groups and entity aliases matter; substring overlap is not a general correctness judgment. |
| Retrieval | Recall@k, MRR, R-Precision, all-gold-hit, graded nDCG | Requires usable relevance labels and a declared candidate universe. Graded metrics require preserved relevance grades. |
| Evidence support | Independently extracted claim support | Depends on extraction coverage and the specified evidence scope. Support by retrieved text is distinct from external factual correctness. |
| Citation attribution | Support of fact-citation relationships, cited-claim coverage, invalid references | Requires interpretable citations, claim alignment, and evidence judgments. Citation presence alone is not citation quality. |
| Verification | Correction success, harmful revision, preservation, named score deltas | Requires paired initial/final answers and a task-appropriate correctness rule or score. |
| Robustness and supplied-response evaluation | Context-conditioned answer/abstain behavior, response/span hallucination scores | Uses the relevant annotations and task protocol; not interchangeable with newly generated QA accuracy. |
| Efficiency | Latency, model calls, tokens, estimated cost, reuse | Depends on observed events and configured prices. Missing usage remains unknown. |

For the exact active metric names and statuses, inspect [`metric_catalog.py`](../src/rag_benchmark/evaluation/metric_catalog.py), [`answers.py`](../src/rag_benchmark/evaluation/answers.py), and [`tracks.py`](../src/rag_benchmark/evaluation/tracks.py).

## Independent evaluation

Pipeline-time evidence checks are operational components: they may influence which answer is returned. Independent evaluation instead reads the completed answer and produces separate artifacts. Claims selected by the generating or verifying component are not automatically a complete independent inventory of the answer's factual content.

`python -m scripts.evaluate_predictions` accepts one or more prediction files, a canonical processed-data root, and a new output directory. With `--judge none` and no extractor, applicable reference-based scoring can run while unsupported claim/citation judgments remain missing. Configure `--extractor-model` for independent claim extraction/alignment and `--judge-model` with `--judge openai` for API-backed support judgments. Those network-backed options require `--allow-network`.

The local NLI path judges the evidence supplied to it subject to the model's input limit. A truncation, missing model, request failure, malformed judgment, and genuine insufficient evidence are different states. Larger evidence sets can require a suitable long-context judge; increasing a configuration cap cannot enlarge a model's actual context window.

Evaluation writes sidecar records such as `evaluations.jsonl`, `evaluation_manifest.json`, `metrics.json`, diagnostics, and call/cache metadata. It requires a new output directory and does not rewrite the source prediction file. Reusing a previous evaluation cache requires compatible identities and hashes.

## Evidence scopes and missing values

Keep these evidence scopes separate:

- **Reference:** a specified reference pool used for evaluation.
- **Visible context:** evidence actually supplied to answer generation.
- **Cited evidence:** the subset selected by the answer's citations.

Support under one scope does not imply support under another. A statement supported by a counterfactual passage can still be incorrect relative to the reference answer.

Reports retain statuses such as `not_applicable_dataset`, `not_computed_missing_input`, `requires_calibration_labels`, and other metric-specific missing or partial states. Do not replace these with zero, discard them without reporting coverage, or interpret an empty set of extracted facts as perfect factuality. Extraction failure can change coverage without changing the underlying answer.

Model-based extraction, attribution, and requirement judgments need independent task-specific calibration. Synthetic regression tests check software behavior; they do not validate an evaluator against human judgments. A supplied review overlay is user-provided evidence, not an automatic certification of evaluator reliability.

## Task-specific boundaries

- **KILT and Hotpot-style retrieval:** report the actual corpus/candidate scope. Candidate-restricted scores do not establish unrestricted FullWiki retrieval quality. Supporting-fact metrics need the appropriate sentence identifiers.
- **ASQA and QAMPARI:** grouped/aliased reference scoring and stricter unique-entity variants answer different questions. QAMPARI recall@5 uses the task's denominator convention; it should not be described as simply scoring only the first five predicted entities.
- **SciFact:** label and rationale scoring depend on the three-way scientific-claim contract. An insufficient-evidence label is a task label, not a generic API failure or refusal.
- **GaRAGe:** eligibility, requirement satisfaction, answer/abstain behavior, and attribution remain separate. RAF/uRAF require their judge inputs and calibration conditions; answer overlap is not a replacement for requirement satisfaction.
- **RAGTruth:** response and character-span scoring applies to supplied responses and their annotations. It does not directly measure the accuracy of a new QA pipeline.
- **RGB:** context conditions and expected behavior must be explicit. Semantic ambiguity or absent review information should remain unresolved rather than being silently inferred from a condition name.

## Paired comparisons and revision

Comparison inputs must align on immutable identity: dataset/version, example, context instance, and seed, with matching task scope. The evaluation entry point checks pairing before any paid judgment calls. For response-level tasks, preserve response identity as well.

Verification summaries distinguish:

- **Correction success:** initially incorrect answers that become correct, divided by initially incorrect answers.
- **Harmful revision:** initially correct answers that become incorrect, divided by initially correct answers.
- **Preservation:** initially correct answers that remain correct, divided by initially correct answers.
- **Score movement:** the difference in a named score shared by both answers, when a binary correctness judgment is unavailable.

Zero denominators and undefined correctness are not successful revisions. A nonzero overlap score is not automatically a correct answer. Reusing the same initial draft isolates the revision intervention more clearly than comparing separately generated answers.

Bootstrap intervals and paired randomization comparisons describe variation in the evaluated sample under their assumptions. They do not capture all uncertainty in model judges, annotation quality, changing API behavior, or implementation choices. Dataset/example clustering matters when several context conditions share an example. Treat unadjusted multiple comparisons as exploratory.

## Resource accounting and reproducibility

Report end-to-end timing, component time where available, logical calls, observed API attempts, input/output tokens, and reuse separately. A stage retrieved from cache does not incur a fresh call. Unknown usage for an interrupted or failed request is not zero; token-based estimates may differ from provider billing.

Budget profiles constrain configured behavior. They do not guarantee equal observed cost, identical latency, or a strict monetary ceiling when retries and unknown usage are involved. Keep prices and model identifiers explicit and verify them for the actual endpoint used.

For a reproducible study, retain source/configuration versions, data and corpus identities, model revisions, split/sample definitions, prompts, execution conditions, and the relevant artifacts. Pinned assets and manifests help trace a run, but do not guarantee identical outputs from a changing external service. This repository contains no claim that a particular configuration has passed a model-quality or independent-human validation threshold.
