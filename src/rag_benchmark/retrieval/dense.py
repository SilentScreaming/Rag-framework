"""Dense vector retrieval over canonical corpus chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from rag_benchmark.data import CorpusChunk
from rag_benchmark.retrieval.bm25 import RetrievedChunk


class EmbeddingBackend(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a 2D float array with one embedding per input text."""


@dataclass
class SentenceTransformerEmbedder:
    """Persistent sentence-transformers wrapper with asymmetric instructions."""

    model_name: str
    batch_size: int = 64
    device: str | None = None
    revision: str | None = None
    query_instruction: str = ""
    document_instruction: str = ""
    trust_remote_code: bool = False
    cache_folder: str | None = None
    identity_name: str | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _MODEL_CACHE: ClassVar[dict[tuple[Any, ...], Any]] = {}

    @property
    def name(self) -> str:
        identity = self.identity_name or self.model_name
        return f"{identity}@{self.revision}" if self.revision else identity

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"{self.query_instruction}{text}" for text in texts])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"{self.document_instruction}{text}" for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        resolved_revision = None if Path(self.model_name).exists() else self.revision
        cache_key = (
            str(Path(self.model_name).resolve()) if Path(self.model_name).exists() else self.model_name,
            self.device,
            resolved_revision,
            self.trust_remote_code,
            self.cache_folder,
        )
        self._model = self._MODEL_CACHE.get(cache_key)
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=resolved_revision,
                trust_remote_code=self.trust_remote_code,
                cache_folder=self.cache_folder,
            )
            self._MODEL_CACHE[cache_key] = self._model
        return self._model


@dataclass
class DenseVectorIndex:
    chunks: list[CorpusChunk]
    vectors: np.ndarray
    embedding_model: str
    normalize: bool = True
    search_backend: str = "auto"
    _faiss_index: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("DenseVectorIndex vectors must be a 2D array")
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError("DenseVectorIndex chunks and vectors length mismatch")
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.normalize:
            self.vectors = l2_normalize(self.vectors)
        if self.search_backend not in {"auto", "numpy", "faiss"}:
            raise ValueError("search_backend must be auto, numpy, or faiss")
        if self.search_backend in {"auto", "faiss"}:
            try:
                import faiss

                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(np.ascontiguousarray(self.vectors))
                self._faiss_index = index
            except ImportError:
                if self.search_backend == "faiss":
                    raise RuntimeError("FAISS search requested but faiss is not installed")

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[RetrievedChunk]:
        return self.search_many(np.asarray(query_vector).reshape(1, -1), top_k=top_k)[0]

    def search_many(
        self,
        query_vectors: np.ndarray,
        *,
        top_k: int,
    ) -> list[list[RetrievedChunk]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("Query embeddings must be a 2D array")
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"Query embedding dimension {queries.shape[1]} does not match index dimension {self.vectors.shape[1]}"
            )
        if not len(queries):
            return []
        if self.normalize:
            queries = l2_normalize(queries)
        limit = min(top_k, len(self.chunks))
        if self._faiss_index is not None:
            ranked_scores, ranked_ids = self._faiss_index.search(
                np.ascontiguousarray(queries), limit
            )
        else:
            scores = np.matmul(queries, self.vectors.T)
            ranked_ids = np.argsort(-scores, axis=1, kind="mergesort")[:, :limit]
            ranked_scores = np.take_along_axis(scores, ranked_ids, axis=1)
        return [
            [
                RetrievedChunk(
                    chunk=self.chunks[int(index)],
                    rank=rank,
                    score=float(score),
                    metadata={
                        "retriever": "dense",
                        "embedding_model": self.embedding_model,
                        "dense_rank": rank,
                        "dense_score": float(score),
                        "search_backend": "faiss" if self._faiss_index is not None else "numpy",
                    },
                )
                for rank, (index, score) in enumerate(zip(indices, scores), start=1)
            ]
            for indices, scores in zip(ranked_ids, ranked_scores)
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        manifest = {
            "embedding_model": self.embedding_model,
            "normalize": self.normalize,
            "search_backend": self.search_backend,
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, corpus_by_id: dict[str, CorpusChunk]) -> "DenseVectorIndex":
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        vectors = np.load(path / "vectors.npy")
        missing = [chunk_id for chunk_id in manifest["chunk_ids"] if chunk_id not in corpus_by_id]
        if missing:
            raise ValueError(f"Dense index references missing chunks: {missing[:3]}")
        chunks = [corpus_by_id[chunk_id] for chunk_id in manifest["chunk_ids"]]
        return cls(
            chunks=chunks,
            vectors=vectors,
            embedding_model=str(manifest["embedding_model"]),
            normalize=bool(manifest.get("normalize", True)),
            search_backend=str(manifest.get("search_backend", "auto")),
        )


class DenseRetriever:
    """Dense retriever with explicit build/retrieve lifecycle."""

    def __init__(
        self,
        embedder: EmbeddingBackend,
        *,
        normalize: bool = True,
        include_title: bool = True,
        search_backend: str = "auto",
    ) -> None:
        self.embedder = embedder
        self.normalize = normalize
        self.include_title = include_title
        self.search_backend = search_backend
        self.index: DenseVectorIndex | None = None

    def build(self, corpus: list[CorpusChunk]) -> None:
        texts = [document_text(chunk, include_title=self.include_title) for chunk in corpus]
        vectors = encode_documents(self.embedder, texts)
        self.index = DenseVectorIndex(
            chunks=list(corpus),
            vectors=vectors,
            embedding_model=self.embedder.name,
            normalize=self.normalize,
            search_backend=self.search_backend,
        )

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("DenseRetriever.build must be called before retrieve")
        query_vector = encode_queries(self.embedder, [query])[0]
        return self.index.search(query_vector, top_k=top_k)

    def retrieve_many(self, queries: list[str], *, top_k: int) -> list[list[RetrievedChunk]]:
        if self.index is None:
            raise RuntimeError("DenseRetriever.build must be called before retrieve")
        if not queries:
            return []
        query_vectors = encode_queries(self.embedder, queries)
        return self.index.search_many(query_vectors, top_k=top_k)

    def save_index(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("DenseRetriever.build must be called before save_index")
        self.index.save(path)

    def load_index(self, path: Path, *, corpus_by_id: dict[str, CorpusChunk]) -> None:
        self.index = DenseVectorIndex.load(path, corpus_by_id=corpus_by_id)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def encode_queries(embedder: EmbeddingBackend, texts: list[str]) -> np.ndarray:
    method = getattr(embedder, "encode_queries", None)
    return method(texts) if callable(method) else embedder.encode(texts)


def encode_documents(embedder: EmbeddingBackend, texts: list[str]) -> np.ndarray:
    method = getattr(embedder, "encode_documents", None)
    return method(texts) if callable(method) else embedder.encode(texts)


def document_text(chunk: CorpusChunk, *, include_title: bool) -> str:
    if include_title and chunk.title:
        return f"{chunk.title}\n{chunk.text}"
    return chunk.text
