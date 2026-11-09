"""Factories for config-driven pipeline components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag_benchmark.compression.sentence_select import (
    SentenceCompressionConfig,
    SentenceTopNCompressor,
)
from rag_benchmark.config import PipelineConfig
from rag_benchmark.indexing.dense_cache import build_cached_dense_index, load_cached_dense_index
from rag_benchmark.generation.backends import (
    BaseGenerator,
    build_generator,
)
from rag_benchmark.generation.text_backend import OpenAITextPromptGenerator
from rag_benchmark.retrieval.bm25 import BM25Retriever, tokenize
from rag_benchmark.query_formulation import (
    IdentityQueryFormulator,
    LLMQueryRewriter,
    MultiQueryLLMFormulator,
    QueryFormulator,
    RuleBasedQueryRewriter,
)
from rag_benchmark.reranking.cross_encoder import CrossEncoderReranker, RerankerConfig
from rag_benchmark.retrieval.dense import DenseRetriever, DenseVectorIndex
from rag_benchmark.retrieval.hybrid import HybridRetrievalConfig, HybridRetriever
from rag_benchmark.evaluation.evidence import RuleBasedEvidenceJudge, TransformersNLIEvidenceJudge


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ComponentFactorySettings:
    generator_kind: str = "extractive"
    generator_model: str | None = None
    query_rewriter: str = "auto"
    use_real_dense: bool = False
    use_real_reranker: bool = False
    dense_index_path: Path | None = None
    build_index_if_missing: bool = False
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    execution_profile: str = "smoke"
    strict_components: bool = False
    model_configs: dict[str, JsonObject] = field(default_factory=dict)
    model_cache_dir: Path | None = None


class HashingEmbedder:
    """Small deterministic embedding backend for local smoke tests."""

    def __init__(self, *, dimensions: int = 128) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-token-embedder-v0:{self.dimensions}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vectors[row_index, index] += 1.0
        return vectors


class TokenOverlapPairScorer:
    """Deterministic query/chunk scorer used when no cross-encoder is requested."""

    name = "token_overlap_reranker_v0"

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        scores = []
        for text in texts:
            text_tokens = set(tokenize(text))
            if not query_tokens:
                scores.append(0.0)
            else:
                scores.append(len(query_tokens & text_tokens) / len(query_tokens))
        return scores


class JsonEchoQueryGenerator:
    """Local text generator that makes llm_rewrite testable without an API call."""

    provider = "local"
    model = "rule-json-query-v0"

    def __init__(self, rewriter: RuleBasedQueryRewriter | None = None) -> None:
        self.rewriter = rewriter or RuleBasedQueryRewriter()

    def generate_text(self, prompt: str) -> str:
        question = prompt.split("Question:", 1)[-1].split("JSON:", 1)[0].strip()
        result = self.rewriter.formulate(question)
        return json.dumps({"query": result.query, "rationale": "local rule rewrite"})


def build_query_formulator(
    *,
    stage: str,
    repo_root: Path,
    settings: ComponentFactorySettings,
    stage_config: JsonObject | None = None,
) -> QueryFormulator:
    stage_config = stage_config or {}
    if stage in {"none", "original_question", "identity"}:
        return IdentityQueryFormulator()
    if stage == "rule_rewrite":
        return RuleBasedQueryRewriter()
    if stage in {"llm_rewrite", "multi_query_rewrite"}:
        if settings.strict_components and settings.query_rewriter in {"identity", "rule"}:
            raise RuntimeError("Formal query formulation cannot replace an LLM rewrite stage with a stub")
        if settings.query_rewriter == "identity":
            return IdentityQueryFormulator()
        if settings.query_rewriter == "rule":
            return RuleBasedQueryRewriter()
        if settings.query_rewriter == "openai" or (
            settings.query_rewriter == "auto" and settings.generator_kind == "openai"
        ):
            cap = int(stage_config.get("max_output_tokens", 512))
            generator = OpenAITextPromptGenerator(repo_root=repo_root, model=settings.generator_model,
                max_output_tokens=cap)
        else:
            generator = JsonEchoQueryGenerator()
        if stage == "multi_query_rewrite":
            return MultiQueryLLMFormulator(
                generator,
                max_queries=int(stage_config.get("max_queries", 4)),
                input_cost_per_1m_tokens=settings.input_cost_per_1m_tokens,
                output_cost_per_1m_tokens=settings.output_cost_per_1m_tokens,
                fail_on_error=settings.strict_components,
            )
        return LLMQueryRewriter(
            generator,
            input_cost_per_1m_tokens=settings.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=settings.output_cost_per_1m_tokens,
            fail_on_error=settings.strict_components,
        )
    raise ValueError(f"Unsupported query formulation stage: {stage}")


def build_base_generator(
    *,
    repo_root: Path,
    settings: ComponentFactorySettings,
    pipeline: PipelineConfig | None = None,
) -> BaseGenerator:
    from rag_benchmark.generation.budget import output_token_limit
    if settings.strict_components and settings.generator_kind in {"mock", "extractive"}:
        raise RuntimeError("Formal generation cannot use mock or local extractive generators")
    return build_generator(
        settings.generator_kind,
        repo_root=repo_root,
        model=settings.generator_model,
        input_cost_per_1m_tokens=settings.input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=settings.output_cost_per_1m_tokens,
        max_output_tokens=(
            output_token_limit(pipeline, "plain")
            if pipeline is not None
            else 256
        ),
    )


def build_retriever(
    *,
    pipeline: PipelineConfig,
    corpus: list[Any],
    settings: ComponentFactorySettings,
    precomputed_vectors: np.ndarray | None = None,
    precomputed_embedding_model: str | None = None,
) -> Any | None:
    stage = pipeline.stages.retrieval
    if stage == "none":
        return None
    if stage == "oracle":
        return None
    if stage == "bm25":
        retriever = _build_bm25_retriever(pipeline)
        retriever.build(corpus)
        return retriever
    if stage == "dense":
        return _build_dense_retriever(
            pipeline=pipeline,
            corpus=corpus,
            settings=settings,
            precomputed_vectors=precomputed_vectors,
            precomputed_embedding_model=precomputed_embedding_model,
        )
    if stage == "hybrid":
        bm25 = _build_bm25_retriever(pipeline)
        bm25.build(corpus)
        dense = _build_dense_retriever(
            pipeline=pipeline,
            corpus=corpus,
            settings=settings,
            precomputed_vectors=precomputed_vectors,
            precomputed_embedding_model=precomputed_embedding_model,
        )
        retriever = HybridRetriever(
            lexical_retriever=bm25,
            dense_retriever=dense,
            config=HybridRetrievalConfig(
                bm25_top_k=int(pipeline.retrieval.get("bm25_top_k", 50)),
                dense_top_k=int(pipeline.retrieval.get("dense_top_k", 50)),
                final_top_k=int(pipeline.retrieval.get("final_top_k", 12)),
                rrf_k=int(pipeline.retrieval.get("rrf_k", 60)),
                fusion=str(pipeline.retrieval.get("fusion", "rrf")),
                lexical_weight=float(pipeline.retrieval.get("lexical_weight", 0.5)),
                dense_weight=float(pipeline.retrieval.get("dense_weight", 0.5)),
            ),
        )
        return retriever
    raise ValueError(f"Unsupported retrieval stage: {stage}")


def build_reranker(*, pipeline: PipelineConfig, settings: ComponentFactorySettings) -> CrossEncoderReranker | None:
    stage = pipeline.stages.reranking
    if stage == "none":
        return None
    if stage != "cross_encoder":
        raise ValueError(f"Unsupported reranking stage: {stage}")
    use_real = settings.use_real_reranker or settings.execution_profile in {"quality_max", "budget_matched"}
    if settings.strict_components and not use_real:
        raise RuntimeError("Formal reranking cannot use token_overlap_reranker_v0")
    if use_real:
        from rag_benchmark.reranking.cross_encoder import SentenceTransformersCrossEncoderScorer

        model_config = _resolve_model_section(
            pipeline.reranking,
            settings=settings,
            section="reranker",
        )
        model_name = str(
            pipeline.reranking.get("model_name")
            or model_config.get("model")
            or model_config.get("name")
            or "BAAI/bge-reranker-large"
        )
        revision = _resolve_model_revision(
            pipeline.reranking,
            model_config,
            settings=settings,
            component="reranker",
        )
        scorer = SentenceTransformersCrossEncoderScorer(
            model_name=_resolve_local_model_path(
                model_name=model_name,
                revision=revision,
                settings=settings,
            ),
            batch_size=int(model_config.get("batch_size", pipeline.reranking.get("batch_size", 16))),
            device=model_config.get("device") or pipeline.reranking.get("device"),
            revision=revision,
            max_length=int(model_config.get("max_length", pipeline.reranking.get("max_length", 512))),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            cache_folder=(str(settings.model_cache_dir) if settings.model_cache_dir else None),
            identity_name=model_name,
        )
    else:
        scorer = TokenOverlapPairScorer()
    return CrossEncoderReranker(
        scorer,
        config=RerankerConfig(
            top_k=int(pipeline.reranking.get("top_k", 8)),
            max_candidates=(
                int(pipeline.reranking["max_candidates"])
                if pipeline.reranking.get("max_candidates") is not None
                else None
            ),
            include_title=bool(pipeline.reranking.get("include_title", True)),
        ),
    )


def build_compressor(
    pipeline: PipelineConfig,
    *,
    semantic_scorer: Any | None = None,
) -> SentenceTopNCompressor | None:
    stage = pipeline.stages.compression
    if stage == "none":
        return None
    if stage != "sentence_top_n":
        raise ValueError(f"Unsupported compression stage: {stage}")
    return SentenceTopNCompressor(
        SentenceCompressionConfig(
            max_sentences=int(pipeline.compression.get("max_sentences", 16)),
            per_chunk_limit=(
                int(pipeline.compression["per_chunk_limit"])
                if pipeline.compression.get("per_chunk_limit") is not None
                else None
            ),
            max_tokens=(
                int(pipeline.compression["max_tokens"])
                if pipeline.compression.get("max_tokens") is not None
                else None
            ),
            min_chunks=int(pipeline.compression.get("min_chunks", 1)),
            mmr_lambda=float(pipeline.compression.get("mmr_lambda", 0.8)),
            semantic_weight=float(pipeline.compression.get("semantic_weight", 1.0)),
            lexical_weight=float(pipeline.compression.get("lexical_weight", 0.35)),
            rank_weight=float(pipeline.compression.get("rank_weight", 0.15)),
        ),
        semantic_scorer=semantic_scorer if bool(pipeline.compression.get("semantic_scoring", False)) else None,
    )


def pipeline_retrieval_top_k(pipeline: PipelineConfig, *, default: int = 10) -> int:
    if pipeline.stages.retrieval == "hybrid":
        return int(pipeline.retrieval.get("final_top_k", default))
    return int(pipeline.retrieval.get("top_k", default))


def build_evidence_judge(*, pipeline: PipelineConfig, settings: ComponentFactorySettings) -> Any:
    kind = str(pipeline.evidence_judge.get("type", "rule"))
    judge_is_required = (
        pipeline.stages.generation == "citation_aware"
        or pipeline.stages.verification not in {"none", "disabled"}
    )
    if settings.execution_profile == "smoke" and not settings.strict_components:
        kind = "rule"
    if settings.strict_components and judge_is_required and kind == "rule":
        kind = "nli"
    if kind == "rule":
        return RuleBasedEvidenceJudge(
            min_claim_token_recall=float(pipeline.evidence_judge.get("min_claim_token_recall", 0.5))
        )
    if kind != "nli":
        raise ValueError(f"Unsupported evidence judge: {kind}")
    model_config = _resolve_model_section(
        pipeline.evidence_judge,
        settings=settings,
        section="evidence_judge",
    )
    revision = _resolve_model_revision(
        pipeline.evidence_judge,
        model_config,
        settings=settings,
        component="evidence judge",
    )
    model_name = str(
            pipeline.evidence_judge.get("model_name")
            or model_config.get("model")
            or model_config.get("name")
            or "cross-encoder/nli-deberta-v3-base"
        )
    return TransformersNLIEvidenceJudge(
        model_name=_resolve_local_model_path(
            model_name=model_name,
            revision=revision,
            settings=settings,
        ),
        revision=revision,
        device=model_config.get("device") or pipeline.evidence_judge.get("device"),
        max_length=int(model_config.get("max_length", pipeline.evidence_judge.get("max_length", 512))),
        entailment_threshold=float(pipeline.evidence_judge.get("entailment_threshold", 0.65)),
        contradiction_threshold=float(pipeline.evidence_judge.get("contradiction_threshold", 0.65)),
        cache_dir=(str(settings.model_cache_dir) if settings.model_cache_dir else None),
        identity_name=model_name,
    )


def _build_embedder(*, pipeline: PipelineConfig, settings: ComponentFactorySettings) -> Any:
    use_real = settings.use_real_dense or settings.execution_profile in {"quality_max", "budget_matched"}
    if settings.strict_components and not use_real:
        raise RuntimeError("Formal dense retrieval cannot use hashing-token-embedder-v0")
    if use_real:
        from rag_benchmark.retrieval.dense import SentenceTransformerEmbedder

        model_config = _resolve_model_section(
            pipeline.retrieval,
            settings=settings,
            section="embedding_model",
        )
        model_name = str(
            pipeline.retrieval.get("embedding_model")
            or model_config.get("model")
            or model_config.get("name")
            or "BAAI/bge-large-en-v1.5"
        )
        revision = _resolve_model_revision(
            pipeline.retrieval,
            model_config,
            settings=settings,
            component="embedding model",
        )
        return SentenceTransformerEmbedder(
            model_name=_resolve_local_model_path(
                model_name=model_name,
                revision=revision,
                settings=settings,
            ),
            batch_size=int(model_config.get("batch_size", pipeline.retrieval.get("batch_size", 32))),
            device=model_config.get("device") or pipeline.retrieval.get("device"),
            revision=revision,
            query_instruction=str(model_config.get("query_instruction", "")),
            document_instruction=str(model_config.get("document_instruction", "")),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            cache_folder=(str(settings.model_cache_dir) if settings.model_cache_dir else None),
            identity_name=model_name,
        )
    dimensions = int(pipeline.retrieval.get("hash_dimensions", 128))
    return HashingEmbedder(dimensions=dimensions)


def _build_bm25_retriever(pipeline: PipelineConfig) -> BM25Retriever:
    return BM25Retriever(
        k1=float(pipeline.retrieval.get("bm25_k1", 1.5)),
        b=float(pipeline.retrieval.get("bm25_b", 0.75)),
        epsilon=float(pipeline.retrieval.get("bm25_epsilon", 0.25)),
        include_title=bool(pipeline.retrieval.get("include_title", True)),
        title_weight=int(pipeline.retrieval.get("title_weight", 2)),
    )


def _build_dense_retriever(
    *,
    pipeline: PipelineConfig,
    corpus: list[Any],
    settings: ComponentFactorySettings,
    precomputed_vectors: np.ndarray | None = None,
    precomputed_embedding_model: str | None = None,
) -> DenseRetriever:
    embedder = _build_embedder(pipeline=pipeline, settings=settings)
    include_title = bool(pipeline.retrieval.get("include_title", True))
    search_backend = str(pipeline.retrieval.get("search_backend", "auto"))
    retriever = DenseRetriever(
        embedder,
        include_title=include_title,
        search_backend=search_backend,
    )
    if precomputed_vectors is not None:
        if precomputed_embedding_model != embedder.name:
            raise ValueError(
                "Frozen candidate vectors use embedding model "
                f"{precomputed_embedding_model!r}, expected {embedder.name!r}"
            )
        retriever.index = DenseVectorIndex(
            chunks=list(corpus),
            vectors=precomputed_vectors,
            embedding_model=embedder.name,
            normalize=True,
            search_backend=search_backend,
        )
        return retriever
    if settings.strict_components and settings.dense_index_path is None:
        raise RuntimeError(
            "Formal dense retrieval requires --dense-index-path so corpus/model identity is reproducible"
        )
    if settings.dense_index_path is None:
        retriever.build(corpus)
        return retriever

    index_path = settings.dense_index_path
    if index_path.exists():
        retriever.index = load_cached_dense_index(
            index_dir=index_path,
            corpus=corpus,
            expected_embedding_model=embedder.name,
            normalize=True,
            include_title=include_title,
            search_backend=search_backend,
        )
        return retriever
    if not settings.build_index_if_missing:
        raise FileNotFoundError(
            f"Dense index cache not found: {index_path}. Run scripts.build_index first "
            "or pass --build-index-if-missing."
        )
    build_cached_dense_index(
        corpus=corpus,
        embedder=embedder,
        output_dir=index_path,
        dataset_name=str(corpus[0].dataset) if corpus else "unknown",
        split_name="runtime",
        normalize=True,
        include_title=include_title,
        search_backend=search_backend,
        metadata={"source": "component_factory_build_index_if_missing"},
    )
    retriever.index = load_cached_dense_index(
        index_dir=index_path,
        corpus=corpus,
        expected_embedding_model=embedder.name,
        normalize=True,
        include_title=include_title,
        search_backend=search_backend,
    )
    return retriever


def _resolve_model_section(
    stage_config: JsonObject,
    *,
    settings: ComponentFactorySettings,
    section: str,
) -> JsonObject:
    ref = stage_config.get("model_ref")
    if not ref:
        return {}
    config = settings.model_configs.get(str(ref), {})
    nested = config.get(section)
    if isinstance(nested, dict):
        return dict(nested)
    return dict(config)


def _resolve_local_model_path(
    *,
    model_name: str,
    revision: str | None,
    settings: ComponentFactorySettings,
) -> str:
    if revision and settings.model_cache_dir is not None:
        snapshot = (
            settings.model_cache_dir.parent
            / "snapshots"
            / model_name.replace("/", "--")
            / revision
        )
        if snapshot.exists():
            return str(snapshot.resolve())
    return model_name


def _resolve_model_revision(
    stage_config: JsonObject,
    model_config: JsonObject,
    *,
    settings: ComponentFactorySettings,
    component: str,
) -> str | None:
    revision = stage_config.get("revision") or model_config.get("revision")
    if settings.strict_components and not revision:
        raise RuntimeError(
            f"Formal {component} configuration must pin an immutable model revision"
        )
    return str(revision) if revision else None
