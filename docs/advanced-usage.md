# Advanced usage

This page collects the details that most first-time users do not need. Start with the [README](../README.md).

## Installing from a wheel

An editable install (`pip install -e .`) finds its checkout automatically, even when called from another directory. A wheel does not include `configs/`, so Python code that loads configuration needs a workspace: pass `repo_root` explicitly, or set `RAG_FRAMEWORK_ROOT` to a directory that contains `configs/`. The command-line workflows (`python -m scripts...`) always run from a checkout.

## Caches

Importing `rag_benchmark` sets per-process defaults for these cache variables, under `.cache/` in the workspace:

| Variable | Default location |
| --- | --- |
| `XDG_CACHE_HOME` | `.cache/` |
| `HF_HOME` | `.cache/huggingface/` |
| `TORCH_HOME` | `.cache/torch/` |
| `SENTENCE_TRANSFORMERS_HOME` | `.cache/sentence-transformers/` |
| `TIKTOKEN_CACHE_DIR` | `.cache/tiktoken/` |

A variable you have already set is left unchanged. `RAG_FRAMEWORK_CACHE_ROOT` moves the base directory. Without a configured workspace, the base is your user cache directory (`%LOCALAPPDATA%` on Windows, `~/.cache` elsewhere). Setting these defaults does not download anything.

## API generation

The API key is read only from `OPENAI_API_KEY`. The generation model is chosen in this order: `--model`, then `configs/openai.local.json`, then `OPENAI_MODEL`.

`scripts.configure_openai_local` writes `configs/openai.local.json` from the options you pass:

| Option | Effect |
| --- | --- |
| `--model` | Model ID (required) |
| `--input-price`, `--output-price` | USD per million input and output tokens; give both or neither |
| `--ca-bundle` | CA bundle for TLS; relative paths resolve against the repository root |

Each call replaces the file, so repeat any option you want to keep. Prices can also be set for a single run with `--input-cost-per-1m-tokens` and `--output-cost-per-1m-tokens`, which take precedence over the saved values. Without prices, `cost_latency.json` reports token usage and marks cost as unavailable.

**TLS certificates.** The CA bundle is taken from the first of these that is set: `ca_bundle` in `configs/openai.local.json`, `OPENAI_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`. A configured path that does not exist is an error. TLS verification is never turned off. On Windows the bundle is used in addition to the system certificates rather than replacing them.

**Endpoints.** Requests use the Chat Completions API with `max_completion_tokens`, and `reasoning_effort` when you set one. The official `openai` client also reads `OPENAI_BASE_URL`, but other OpenAI-compatible endpoints have not been tested and may reject some of these parameters.

## Model snapshots

`scripts.freeze_model_assets` needs the model extra (`pip install -e ".[models]"`). It downloads the revision-pinned models listed in a model configuration to `<models_root>/snapshots/` and writes a manifest with file sizes and SHA-256 hashes to `<models_root>/manifests/`.

```sh
python -m scripts.freeze_model_assets --model-ref default_retrieval
python -m scripts.freeze_model_assets --section reranker --skip-file-hashes
```

`--section` (repeatable) limits the download to `embedding_model`, `reranker` or `evidence_judge`. Set `HF_TOKEN` for gated or private models. Downloads use `REQUESTS_CA_BUNDLE` or `SSL_CERT_FILE` when set.

## Context conditions

Some datasets supply a fixed set of passages per example under one or more conditions instead of, or in addition to, a searchable corpus:

| Dataset | Conditions |
| --- | --- |
| `rgb_en` | `noise_robustness` (with `noise_rate` 0.0, 0.2, 0.4, 0.6, 0.8 or 1.0), `information_integration`, `counterfactual_context`, `factual_control` |
| `garage` | `annotated_grounding` |
| `ragtruth_qa` | `provided_rag_context` |

Each example must resolve to exactly one context set. Select it with `--context-condition` and, where a condition has variants, `--context-metadata`:

```sh
python -m scripts.run_experiment_suite --dataset rgb_en --split-profile mvp --limit 20 --execution-profile quality_max --generator openai --query-rewriter openai --dense-index-path models/indexes/rgb_en/scoped --build-index-if-missing --context-condition noise_robustness --context-metadata noise_rate=0.4 --suite-id rgb-noise-0.4
```

`--context-metadata` takes `KEY=JSON_VALUE` and can be repeated in `run_pipeline`, `run_experiment_suite`, `run_ablation` and `build_index`. `run_verification_replay` takes a single JSON object instead, for example `--context-metadata '{"noise_rate": 0.4}'`. `scripts.run_rgb_stress` runs the same suite for every RGB condition, or for the ones you pick with `--condition` (for example `--condition noise-0.4`); it defaults to the `formal` split and 100 examples.

The context set is part of each prediction's identity, so predictions made under different conditions are never paired with each other.

## Verification replay

`scripts.run_verification_replay` re-runs only the verification stage on saved drafts, so you can compare policies without regenerating the drafts. The experiment suite saves its drafts under `outputs/formal/<suite-id>-best_practice_rag/`.

```sh
python -m scripts.run_verification_replay --dataset alce_asqa --split-profile mvp --limit 20 --draft-predictions outputs/formal/asqa-example-best_practice_rag/predictions.jsonl --dense-index-path models/indexes/asqa-example --output-root outputs/replay --run-id asqa-replay
```

The drafts must match the dataset, split, examples and seed; the command checks this before running verification. For scientific-claim (`scifact`) and list-answer (`alce_qampari`) tasks, `--policy` selects `detect_only`, `constrained_revision` or `abstain_only`. See `--help` for call limits, output caps and sample manifests.

## KILT full-Wikipedia retrieval

Searching the full 2019 Wikipedia snapshot at run time is expensive, so KILT runs can use a frozen candidate pool instead:

1. `scripts.build_kilt_fullwiki_fts` builds a resumable, article-level SQLite FTS5 index of the snapshot.
2. `scripts.build_kilt_candidate_cache --dataset-profile DATASET:SPLIT:LIMIT` retrieves a candidate pool for each question from that index. Gold passages are not added to the pools.
3. `scripts.embed_kilt_candidate_cache` precomputes dense embeddings for the cached candidates.

Pass the result to runs with `--candidate-cache-path`. Report results from these runs as candidate-restricted retrieval, not full-corpus retrieval. All three tools are long-running; check their `--help` and available disk space first.
