# rag-framework

[English](README.md) | [简体中文](README.zh-CN.md)

Run controlled RAG experiments and see not only which answers a component fixes, but which ones it breaks.

rag-framework runs closed-book, vanilla, enhanced and verification pipelines on the same data slices, and scores answer correctness, retrieval relevance, evidence support and citation quality separately. Verification is compared against its own draft, so corrected answers and newly broken answers are both counted.

- Adapters for KILT (NQ, HotpotQA), ALCE (ASQA, QAMPARI), SciFact, GaRAGe, RAGTruth and RGB
- Components you can swap or remove: query variants, BM25, dense retrieval, rank fusion, cross-encoder reranking, sentence selection, citation-aware generation and verification
- Leave-one-out and cumulative ablations, plus replay of saved drafts under different revision policies
- Missing inputs, unsupported tasks and failed calls are recorded as explicit states, not as zero scores
- Every run saves its resolved configuration, predictions, retrieval and stage traces, metrics, token usage and latency

**Status:** research prototype, version 0.1.0. Command-line options, configuration keys and record formats may change between releases.

## Quick start

The first run uses a three-question synthetic dataset. It needs no API key and downloads nothing.

```sh
python3.11 -m venv .venv
. .venv/bin/activate              # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m scripts.init_local

python -m scripts.run_pipeline --dataset toy --pipeline vanilla_rag --execution-profile smoke --generator extractive --limit 3 --run-id toy-vanilla
```

On Windows, create the environment with `py -3.11 -m venv .venv`. Run all commands from the repository root.

The command prints a JSON summary and writes its files to `outputs/formal/toy-vanilla/`. The headline numbers are under `metrics.track_results.vanilla_rag.primary_metrics`:

```json
"primary_metrics": {
  "all_gold_hit@10": 1.0,
  "em": 0.0,
  "f1": 0.2738095238095238,
  "recall@10": 1.0
}
```

Retrieval found the gold passage for every question. Exact match is 0 because the `extractive` generator answers with the whole evidence sentence ("The Harbor Observatory roof is made of copper.") rather than the short answer "copper"; token F1 gives partial credit.

You can also score the saved predictions separately from the run:

```sh
python -m scripts.evaluate_predictions --predictions outputs/formal/toy-vanilla/predictions.jsonl --processed-root examples/toy/processed --output-dir outputs/evaluation/toy-vanilla --judge none
```

Each run needs a new `--run-id`, and each evaluation a new `--output-dir`; the commands refuse to overwrite existing results.

## Pipelines

| Pipeline | What it does | Use it as |
| --- | --- | --- |
| `closed_book` | Answers with the generator alone, without retrieval. | A no-retrieval baseline. |
| `vanilla_rag` | Retrieves with the original question using a dense retriever and passes the raw passages to the generator. | A standard RAG baseline. |
| `best_practice_rag` | Query variants, hybrid BM25 + dense retrieval, cross-encoder reranking, sentence selection and citation-aware generation. | The enhanced stack. The name is a label for this configuration, not a benchmark result. |
| `rag_verification` | Checks the claims in a `best_practice_rag` draft against the evidence and revises only where the evidence supports a change. | The intervention under test. |
| `oracle_context_rag` | Picks contexts from the gold relevance annotations. | A diagnostic that uses information a real system would not have. |

The experiment suite runs the first four on the same slice and feeds the `best_practice_rag` draft into `rag_verification`, so both are scored on the same initial answers. Answer format, citation requirements and token budgets are adapted to each dataset's task.

## Supported datasets

| Pipeline name | Preparation key | Task | Upstream | License recorded in the catalog |
| --- | --- | --- | --- | --- |
| `kilt_nq` | `kilt_nq` | Single-hop open-domain QA | [KILT](https://github.com/facebookresearch/KILT) | MIT code; source datasets and Wikipedia content keep their own terms |
| `kilt_hotpotqa` | `kilt_hotpotqa` | Multi-hop open-domain QA | [KILT](https://github.com/facebookresearch/KILT) | as above |
| `alce_asqa` | `alce` | Ambiguous questions, long answers with citations | [ALCE](https://github.com/princeton-nlp/ALCE) | MIT code; bundled datasets keep their own terms |
| `alce_qampari` | `alce` | List answers with citations | [ALCE](https://github.com/princeton-nlp/ALCE) | as above |
| `scifact` | `scifact` | Scientific claim verification | [SciFact](https://github.com/allenai/scifact) | Claims/annotations CC-BY-4.0; abstracts ODC-By-1.0; code Apache-2.0 |
| `garage` | `garage` | Grounding and selective refusal | [GaRAGe](https://github.com/amazon-science/GaRAGe) | CC-BY-NC-4.0 |
| `ragtruth_qa` | `ragtruth` | Hallucination detection on given responses | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | MIT |
| `rgb_en` | `rgb` | Robustness to noise, counterfactual evidence and information integration | [RGB](https://github.com/chen700564/RGB) | CC-BY-NC-SA-4.0 |

The framework does not redistribute any dataset. Check each upstream license before you use or share the data; several are non-commercial.

## Running a real benchmark

This walkthrough uses ALCE ASQA from start to finish. It needs model downloads and an OpenAI API key. Start with `--limit 20` and look at `cost_latency.json` before scaling up.

**1. Install the model dependencies.**

```sh
python -m pip install -e ".[models]"
```

Choose a PyTorch build for your CPU or GPU. The default models are `BAAI/bge-large-en-v1.5` (embeddings), `BAAI/bge-reranker-large` (reranking) and `cross-encoder/nli-deberta-v3-base` (NLI), pinned by revision in [`configs/models/default_retrieval.example.yaml`](configs/models/default_retrieval.example.yaml).

**2. Configure the generator.** Set `OPENAI_API_KEY` in your environment; it is the only place the key is read from. Then save the model ID, and optionally your provider's prices in USD per million tokens so runs can report cost:

```sh
python -m scripts.configure_openai_local --model YOUR_MODEL_ID --input-price INPUT_PRICE --output-price OUTPUT_PRICE
```

The settings go to `configs/openai.local.json`, which is git-ignored. TLS certificates, model-selection order and other API details are in [advanced usage](docs/advanced-usage.md#api-generation).

**3. Prepare the data.** The first command only prints the download plan: source, license, revision, target paths and required files.

```sh
python -m scripts.prepare_benchmark_data --datasets alce
python -m scripts.prepare_benchmark_data --datasets alce --download
python -m scripts.prepare_benchmark_data --datasets alce --profile --process
python -m scripts.prepare_benchmark_data --datasets alce --validate
```

The `alce` key produces both `alce_asqa` and `alce_qampari`. To download manually, put the upstream files where the plan says, then run the process and validate steps.

**4. Build a dense index** over the corpus of the slice you will run:

```sh
python -m scripts.build_index --dataset alce_asqa --pipeline vanilla_rag --split-profile mvp --limit 20 --embedder config --search-backend numpy --output-dir models/indexes/asqa-example
```

NumPy search needs no extra package. For `--search-backend faiss`, install a FAISS build that matches your platform.

**5. Compare the four pipelines.**

```sh
python -m scripts.run_experiment_suite --dataset alce_asqa --split-profile mvp --limit 20 --execution-profile quality_max --generator openai --query-rewriter openai --dense-index-path models/indexes/asqa-example --suite-id asqa-example
```

**6. Measure each component's contribution.** Leave-one-out removes one of the five `best_practice_rag` components at a time and compares each variant with the full stack. `--design sequential` instead starts from `vanilla_rag` and adds one component per step, up to verification.

```sh
python -m scripts.run_ablation --dataset alce_asqa --split-profile mvp --limit 20 --execution-profile quality_max --generator openai --query-rewriter openai --dense-index-path models/indexes/asqa-example --design leave-one-out --ablation-id asqa-components
```

**7. Evaluate saved predictions** with `scripts.evaluate_predictions`, as in the quick start. Reference-based metrics need no model. For API-backed claim extraction, set `--extractor-model`; for support judgments, use `--judge local-nli`, or `--judge openai` with `--judge-model`. API-backed steps also need `--allow-network`. Prediction files compared with each other must come from the same dataset, examples, contexts and seed.

## Terms

| Term | Meaning |
| --- | --- |
| `mvp`, `dev_small`, `formal` | Split profiles written by the adapters. `dev_small` is a development pool (up to 300 examples for most datasets), `mvp` is its first 100 examples for quick runs, and `formal` is a held-out evaluation split that does not overlap the development pool. RAGTruth uses `judge_calibration` and `judge_eval` instead. Each run reads the split file for its `--split-profile` and stops with an error if it is missing. |
| `outputs/formal/` | Where saved pipeline runs go. The directory name is fixed and does not depend on the split. |
| `smoke` components | Deterministic stand-ins (no downloads, no API calls) used to test the data flow. Their scores say nothing about model quality. |
| Draft | The `best_practice_rag` answer before verification. Verification results are always paired with the draft they started from. |
| Context set | A fixed set of passages supplied by the dataset for one example under one condition, such as an RGB noise rate. Select one with `--context-condition` and `--context-metadata`; see [advanced usage](docs/advanced-usage.md#context-conditions). |
| Candidate cache | A frozen per-question pool of KILT Wikipedia candidates, used instead of searching the full corpus at run time. |

## Configuration

```text
configs/
  paths.example.json             Data, model and output locations
  datasets/*.example.yaml        Dataset and task definitions
  pipelines/*.example.yaml       Pipeline stages and budgets
  profiles/*.example.yaml        Execution-profile overrides
  models/*.example.yaml          Retrieval, reranking and NLI models
```

`scripts.init_local` creates `configs/paths.local.json` if it does not exist and never overwrites it. Paths are relative to the repository root unless you make them absolute. Files named `*.local.json` are git-ignored.

| Execution profile | Purpose |
| --- | --- |
| `smoke` | Data-flow checks with deterministic stand-ins. |
| `quality_max` | Model-backed components; fails instead of falling back to stand-ins. |
| `budget_matched` | Comparisons under shared token and retrieval limits. |

Budget limits, failed requests and malformed model output are kept in the run records. Installing from a wheel, cache locations, TLS settings and model snapshots are covered in [advanced usage](docs/advanced-usage.md).

## Outputs

| File | Contents |
| --- | --- |
| `config.snapshot.yaml`, `run_manifest.json` | Resolved settings, identities and run metadata |
| `predictions.jsonl` | Answers, task-specific structure and example metadata |
| `retrieval_traces.jsonl`, `stage_traces.jsonl` | Retrieval results and per-stage inputs and outputs |
| `metrics.json`, `all_metrics.csv`, `all_metrics.html` | Metrics with availability states, as JSON, CSV and HTML |
| `cost_latency.json` | Token usage, latency and cost, where prices are known |

`scripts.evaluate_predictions` writes `evaluations.jsonl`, an evaluation manifest and call and cache records to its own directory; it never modifies the prediction file. `scripts.run_diagnostics --run-dir <run>` writes `diagnostics.jsonl` into a timestamped `diagnostics-v1-*` folder inside the run. Outputs can contain source text and absolute local paths, so review them before sharing. The `outputs/` directory is git-ignored.

Metric definitions, pairing and statistics are described in [evaluation semantics](docs/evaluation.md); record formats in [canonical data format](docs/data-format.md).

## Other tools

| Command | Purpose |
| --- | --- |
| `scripts.run_verification_replay` | Re-run verification on saved drafts with a chosen policy (`detect_only`, `constrained_revision`, `abstain_only`) |
| `scripts.run_rgb_stress` | RGB comparisons across noise rates, information integration and counterfactual contexts |
| `scripts.run_ragtruth_benchmark` | Evidence-grounded hallucination detection on RAGTruth QA responses |
| `scripts.run_judge` | Evidence and citation judgments over saved predictions |
| `scripts.run_diagnostics` | Retrieval and generation error diagnostics for a saved run |
| `scripts.build_kilt_fullwiki_fts`, `build_kilt_candidate_cache`, `embed_kilt_candidate_cache` | Full-Wikipedia search index and frozen candidate pools for KILT |
| `scripts.freeze_model_assets` | Download revision-pinned model snapshots and write a manifest |

Each command documents its options under `--help`. Longer examples are in [advanced usage](docs/advanced-usage.md).

## Extending

| Area | Code in `src/rag_benchmark/` |
| --- | --- |
| Configuration and execution | `config.py`, `component_factory.py`, `runner.py` |
| Records and dataset adapters | `data.py`, `schemas.py`, `dataset_adapters.py`, `dataset_catalog.py` |
| Query and context preparation | `query_formulation/`, `retrieval/`, `reranking/`, `compression/` |
| Generation and revision | `generation/`, `verification/` |
| Evaluation and diagnostics | `evaluation/`, `analysis/` |
| Comparisons and artifacts | `ablation.py`, `sampling.py`, `run_summary.py`, `paired_artifacts.py`, `reporting/` |

To add a dataset, write canonical records, a dataset configuration, a paths entry and explicit split files. To add a component, implement its interface and register it in `component_factory.py`. Before comparing a new metric across runs, define its inputs, unit of analysis and missing states.

## Development

```sh
python -m pip install -e ".[dev]"
python -m scripts.check_environment
python -m pytest -q
```

The tests run offline on small synthetic fixtures (see [`tests/fixtures/README.md`](tests/fixtures/README.md)). `check_environment` checks a CPU-only setup by default; add `--require-models` or `--require-cuda` to check the optional dependencies.

Bug reports and pull requests are welcome. Please run the test suite before opening a pull request.

## Limitations

- The toy dataset and smoke components only exercise the interfaces; their scores do not measure model quality.
- Model-based judges need task-specific calibration and human validation. Passing tests do not show that an evaluator is accurate or that a revision helps.
- Evidence support is measured against the chosen evidence and can differ from factual correctness. Report candidate-restricted KILT retrieval with its actual scope.
- Token-based cost estimates can differ from your bill. Caching, retries and external services affect repeatability.
- The repository contains tooling only, no experimental results.

## Citation

If you use rag-framework in your research, please cite the repository:

```bibtex
@software{rag_framework_2026,
  author = {SilentScreaming},
  title  = {rag-framework: controlled experiments and evaluation for retrieval-augmented generation},
  year   = {2026},
  version = {0.1.0},
  url    = {https://github.com/SilentScreaming/Rag-framework}
}
```

## License

The source code is released under the [MIT License](LICENSE). Models and datasets are obtained separately and keep their own licenses.
