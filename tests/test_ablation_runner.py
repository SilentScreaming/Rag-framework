from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.ablation import (
    build_default_ablation_steps,
    build_profile_ablation_steps,
    run_ablation_suite,
)
from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.config import (
    ConfigLoader,
    DatasetConfig,
    DatasetSourceConfig,
    ExperimentConfig,
    PathConfig,
)
from rag_benchmark.data import CorpusChunk, DatasetSlice, Example, Qrel
from rag_benchmark.runner import FormalRunSettings, resolve_pipeline_with_base


class AblationRunnerTest(unittest.TestCase):
    def test_default_ablation_steps_have_expected_order(self) -> None:
        names = [step.name for step in build_default_ablation_steps()]

        self.assertEqual(
            names,
            [
                "01_vanilla",
                "02_query_rewrite",
                "03_hybrid",
                "04_reranker",
                "05_compression",
                "06_citation",
                "07_verification",
            ],
        )

    def test_run_ablation_suite_writes_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            result = run_ablation_suite(
                repo_root=REPO_ROOT,
                base_experiment=make_experiment(outputs_root=output_root),
                dataset_slice=make_dataset_slice(),
                settings=FormalRunSettings(
                    split_profile="mvp",
                    limit=1,
                    output_root=output_root,
                    save_outputs=True,
                    component_settings=ComponentFactorySettings(generator_kind="extractive", query_rewriter="rule"),
                ),
                ablation_id="test-ablation",
            )

            self.assertEqual(len(result.step_results), 7)
            self.assertEqual(len(result.metrics_rows), 7)
            self.assertEqual(len(result.delta_rows), 6)
            self.assertTrue((result.output_dir / "ablation_metrics.json").exists())
            self.assertTrue((result.output_dir / "ablation_metrics.csv").exists())
            self.assertTrue((result.output_dir / "ablation_delta.csv").exists())

    def test_profile_steps_reuse_resolved_formal_limits(self) -> None:
        loader = ConfigLoader(REPO_ROOT)
        vanilla = resolve_pipeline_with_base(loader, "vanilla_rag")
        best = resolve_pipeline_with_base(loader, "best_practice_rag")
        verified = resolve_pipeline_with_base(loader, "rag_verification")
        steps = build_profile_ablation_steps(
            vanilla=vanilla,
            best_practice=best,
            verification=verified,
        )

        self.assertEqual(steps[1].pipeline.budget.max_queries, best.budget.max_queries)
        self.assertEqual(steps[1].pipeline.query_formulation, best.query_formulation)
        self.assertEqual(steps[2].pipeline.retrieval, best.retrieval)
        self.assertEqual(steps[6].pipeline.verification, verified.verification)
        self.assertEqual(
            steps[6].pipeline.budget.max_verification_questions,
            verified.budget.max_verification_questions,
        )


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
    corpus = [
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
        corpus=corpus,
        qrels_by_example={"ex-1": [qrel]},
    )


def make_experiment(*, outputs_root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            name="synthetic",
            role="test",
            split="mvp",
            source=DatasetSourceConfig(provider="synthetic", version="test", root_key="synthetic"),
        ),
        pipeline=build_default_ablation_steps()[0].pipeline,
        paths=PathConfig(
            datasets_root=(Path(__file__).resolve().parents[1] / "data"),
            models_root=(Path(__file__).resolve().parents[1] / "models"),
            outputs_root=outputs_root,
            datasets={},
        ),
    )


if __name__ == "__main__":
    unittest.main()
