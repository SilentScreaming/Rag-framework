from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.component_factory import ComponentFactorySettings, HashingEmbedder
from rag_benchmark.config import (
    DatasetConfig,
    DatasetSourceConfig,
    ExperimentConfig,
    PathConfig,
    PipelineConfig,
)
from rag_benchmark.indexing.dense_cache import build_cached_dense_index, load_cached_dense_index
from rag_benchmark.data import CorpusChunk, DatasetSlice, Example, Qrel
from rag_benchmark.runner import ConfigDrivenPipelineRunner, FormalRunSettings


class DenseIndexCacheTest(unittest.TestCase):
    def test_build_and_load_cached_dense_index(self) -> None:
        corpus = make_corpus()
        embedder = HashingEmbedder(dimensions=32)
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"

            result = build_cached_dense_index(
                corpus=corpus,
                embedder=embedder,
                output_dir=index_dir,
                dataset_name="synthetic",
                split_name="mvp",
            )
            index = load_cached_dense_index(
                index_dir=index_dir,
                corpus=corpus,
                expected_embedding_model=embedder.name,
            )

            self.assertEqual(result.manifest["chunk_count"], 2)
            self.assertEqual(index.embedding_model, embedder.name)
            self.assertTrue((index_dir / "vectors.npy").exists())
            self.assertTrue((index_dir / "chunk_ids.jsonl").exists())

    def test_load_rejects_embedding_model_mismatch(self) -> None:
        corpus = make_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            build_cached_dense_index(
                corpus=corpus,
                embedder=HashingEmbedder(dimensions=16),
                output_dir=index_dir,
                dataset_name="synthetic",
                split_name="mvp",
            )

            with self.assertRaises(ValueError):
                load_cached_dense_index(
                    index_dir=index_dir,
                    corpus=corpus,
                    expected_embedding_model=HashingEmbedder(dimensions=32).name,
                )

    def test_runner_uses_cached_dense_index(self) -> None:
        dataset_slice = make_dataset_slice()
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            build_cached_dense_index(
                corpus=dataset_slice.corpus,
                embedder=HashingEmbedder(dimensions=32),
                output_dir=index_dir,
                dataset_name="synthetic",
                split_name="mvp",
            )
            runner = ConfigDrivenPipelineRunner(
                repo_root=REPO_ROOT,
                experiment=make_experiment(make_vanilla_pipeline(), outputs_root=Path(tmp)),
                dataset_slice=dataset_slice,
                settings=FormalRunSettings(
                    run_id="cached-dense-test",
                    save_outputs=False,
                    component_settings=ComponentFactorySettings(
                        generator_kind="extractive",
                        dense_index_path=index_dir,
                    ),
                ),
            )

            prediction = runner.run().predictions[0]

            self.assertTrue(prediction.retrieved_chunks)
            self.assertIn("hashing-token-embedder-v0", prediction.retrieved_chunks[0].metadata["embedding_model"])


def make_corpus() -> list[CorpusChunk]:
    return [
        CorpusChunk(
            chunk_id="chunk-gold",
            doc_id="doc-1",
            dataset="synthetic",
            title="Gold",
            text="Ada wrote the notes.",
            sentences=[{"sentence_id": 0, "text": "Ada wrote the notes."}],
            metadata={"source_example_id": "source-1"},
        ),
        CorpusChunk(
            chunk_id="chunk-other",
            doc_id="doc-2",
            dataset="synthetic",
            title="Other",
            text="Grace built a compiler.",
            sentences=[{"sentence_id": 0, "text": "Grace built a compiler."}],
            metadata={"source_example_id": "source-1"},
        ),
    ]


def make_dataset_slice() -> DatasetSlice:
    example = Example(
        example_id="ex-1",
        dataset="synthetic",
        split="mvp",
        task_type="short_answer",
        question="Who wrote the notes?",
        answers=[{"text": "Ada", "type": "short"}],
        gold_evidence=[],
        metadata={"source_id": "source-1"},
    )
    qrel = Qrel(
        example_id="ex-1",
        chunk_id="chunk-gold",
        relevance=3,
        support_type="gold",
        metadata={"sentence_id": 0},
    )
    return DatasetSlice(
        dataset_name="synthetic",
        split_name="mvp",
        candidate_scope="synthetic",
        examples=[example],
        corpus=make_corpus(),
        qrels_by_example={"ex-1": [qrel]},
    )


def make_experiment(pipeline: PipelineConfig, *, outputs_root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            name="synthetic",
            role="test",
            split="mvp",
            source=DatasetSourceConfig(provider="synthetic", version="test", root_key="synthetic"),
        ),
        pipeline=pipeline,
        paths=PathConfig(
            datasets_root=(Path(__file__).resolve().parents[1] / "data"),
            models_root=(Path(__file__).resolve().parents[1] / "models"),
            outputs_root=outputs_root,
            datasets={},
        ),
    )


def make_vanilla_pipeline() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "vanilla_rag",
            "stages": {
                "query_formulation": "original_question",
                "retrieval": "dense",
                "reranking": "none",
                "compression": "none",
                "generation": "context_grounded",
                "verification": "none",
            },
            "retrieval": {"top_k": 2, "hash_dimensions": 32},
            "generation": {},
        }
    )


if __name__ == "__main__":
    unittest.main()
