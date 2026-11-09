from __future__ import annotations

import sys
import tempfile
import unittest
import json
import re
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.component_factory import ComponentFactorySettings
from rag_benchmark.config import (  # noqa: E402
    DatasetConfig,
    DatasetSourceConfig,
    ExperimentConfig,
    PathConfig,
    PipelineConfig,
)
from rag_benchmark.data import ContextSet, CorpusChunk, DatasetSlice, Example, Qrel
from rag_benchmark.runner import (
    ConfigDrivenPipelineRunner,
    FormalRunSettings,
    resolve_pipeline_with_base,
)
from rag_benchmark.config import ConfigLoader


class ConfigDrivenRunnerTest(unittest.TestCase):
    def test_strict_citation_generation_retries_unknown_context_ids(self) -> None:
        runner = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_best_practice_pipeline()),
            dataset_slice=make_dataset_slice(),
            settings=FormalRunSettings(
                run_id="test-citation-reference-retry",
                save_outputs=False,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        )
        generator = RepairingCitationGenerator()
        runner.text_generator = generator
        runner.component_settings = replace(runner.component_settings, strict_components=True)

        prediction = runner.run().predictions[0]

        self.assertEqual(prediction.generator["attempt_count"], 2)
        self.assertEqual(prediction.metadata["citation_generation"]["invalid_citation_count"], 0)
        self.assertIn("unknown context ids", generator.prompts[1])
        self.assertFalse(prediction.generator["schema_failure"])

    def test_strict_citation_schema_failure_becomes_auditable_abstention(self) -> None:
        runner = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_best_practice_pipeline()),
            dataset_slice=make_dataset_slice(),
            settings=FormalRunSettings(
                run_id="test-citation-schema-failure",
                save_outputs=False,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        )
        runner.text_generator = EmptyCitationGenerator()
        runner.component_settings = replace(runner.component_settings, strict_components=True)

        prediction = runner.run().predictions[0]

        self.assertEqual(prediction.answer, "I don't know.")
        self.assertTrue(prediction.answer_bundle.abstained)
        self.assertTrue(prediction.generator["schema_failure"])
        self.assertEqual(prediction.generator["attempt_count"], 3)
        self.assertTrue(prediction.metadata["citation_generation_runtime"]["used_fallback"])

    def test_best_practice_pipeline_runs_existing_modules(self) -> None:
        runner = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_best_practice_pipeline()),
            dataset_slice=make_dataset_slice(),
            settings=FormalRunSettings(
                run_id="test-best",
                save_outputs=False,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        )

        result = runner.run()

        self.assertEqual(len(result.predictions), 1)
        prediction = result.predictions[0]
        self.assertEqual(prediction.pipeline, "best_practice_rag")
        self.assertTrue(prediction.citations)
        self.assertEqual(prediction.claim_checks[0].status, "supported")
        self.assertNotIn("citation_support_rate", prediction.metrics)
        self.assertEqual(prediction.metadata["evaluation_coverage"]["full_answer_claim_coverage"], "unknown")
        self.assertIn("query_formulation", prediction.metadata)
        self.assertIn("pipeline_overhead_ms", prediction.metrics)
        self.assertEqual(prediction.metrics["total_ms"], prediction.latency.total_ms)
        self.assertEqual(len(result.retrieval_traces), 1)
        self.assertIn("best_practice_rag", result.metrics["pipelines"])

    def test_rag_verification_pipeline_adds_verification_record(self) -> None:
        runner = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_verification_pipeline()),
            dataset_slice=make_dataset_slice(),
            settings=FormalRunSettings(
                run_id="test-verification",
                save_outputs=False,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        )

        prediction = runner.run().predictions[0]

        self.assertIsNotNone(prediction.verification)
        self.assertEqual(prediction.verification.status, "passed")
        self.assertEqual(prediction.metrics["verification_passed"], 1.0)

    def test_runner_writes_formal_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = ConfigDrivenPipelineRunner(
                repo_root=REPO_ROOT,
                experiment=make_experiment(make_closed_book_pipeline(), outputs_root=Path(tmp)),
                dataset_slice=make_dataset_slice(),
                settings=FormalRunSettings(
                    run_id="test-output",
                    save_outputs=True,
                    output_root=Path(tmp),
                    component_settings=ComponentFactorySettings(generator_kind="mock"),
                ),
            )

            result = runner.run()

            self.assertIsNotNone(result.output_dir)
            for name in (
                "config.snapshot.yaml",
                "run_manifest.json",
                "predictions.jsonl",
                "retrieval_traces.jsonl",
                "stage_traces.jsonl",
                "metrics.json",
                "metrics.csv",
                "all_metrics.csv",
                "all_metrics.html",
                "logs.jsonl",
                "cost_latency.json",
            ):
                self.assertTrue((result.output_dir / name).exists(), name)

    def test_runner_recovers_per_example_checkpoint_and_clears_it_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_slice = make_dataset_slice()
            settings = FormalRunSettings(
                run_id="test-checkpoint",
                save_outputs=True,
                output_root=Path(tmp),
                bootstrap_samples=0,
                component_settings=ComponentFactorySettings(generator_kind="mock"),
            )
            interrupted = ConfigDrivenPipelineRunner(
                repo_root=REPO_ROOT,
                experiment=make_experiment(make_closed_book_pipeline(), outputs_root=Path(tmp)),
                dataset_slice=dataset_slice,
                settings=settings,
            )
            example = dataset_slice.examples[0]
            state = interrupted.run_example(
                example=example,
                qrels=dataset_slice.qrels_by_example[example.example_id],
            )
            interrupted._append_checkpoint(state)
            checkpoint_dir = interrupted._checkpoint_dir()

            self.assertIsNotNone(checkpoint_dir)
            self.assertTrue(checkpoint_dir.exists())

            resumed = ConfigDrivenPipelineRunner(
                repo_root=REPO_ROOT,
                experiment=make_experiment(make_closed_book_pipeline(), outputs_root=Path(tmp)),
                dataset_slice=dataset_slice,
                settings=settings,
            ).run()

            self.assertEqual(len(resumed.predictions), 1)
            self.assertTrue(any(row["event"] == "example_recovered" for row in resumed.logs))
            self.assertFalse(checkpoint_dir.exists())

    def test_verification_runner_reuses_paired_best_practice_draft(self) -> None:
        dataset_slice = make_dataset_slice()
        best = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_best_practice_pipeline()),
            dataset_slice=dataset_slice,
            settings=FormalRunSettings(
                run_id="paired-best",
                save_outputs=False,
                bootstrap_samples=0,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        ).run()
        drafts = {prediction.example_id: prediction for prediction in best.predictions}

        verified = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(make_verification_pipeline()),
            dataset_slice=dataset_slice,
            settings=FormalRunSettings(
                run_id="paired-verify",
                save_outputs=False,
                bootstrap_samples=0,
                draft_predictions=drafts,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        ).run().predictions[0]

        self.assertEqual(verified.metrics["draft_reused"], 1.0)
        self.assertEqual(verified.metadata["draft_reuse"]["draft_run_id"], "paired-best")
        self.assertTrue(any(trace.stage == "draft_reuse" for trace in verified.stage_traces))

    def test_pipeline_base_config_is_resolved(self) -> None:
        pipeline = resolve_pipeline_with_base(ConfigLoader(REPO_ROOT), "rag_verification")

        self.assertEqual(pipeline.name, "rag_verification")
        self.assertEqual(pipeline.stages.retrieval, "hybrid")
        self.assertEqual(pipeline.retrieval["final_top_k"], 100)
        self.assertEqual(pipeline.reranking["top_k"], 12)
        self.assertEqual(pipeline.stages.verification, "factored_cove")

    def test_retrieval_is_limited_to_the_examples_fixed_context_set(self) -> None:
        dataset_slice = make_dataset_slice()
        dataset_slice.context_sets_by_example["ex-1"] = [
            ContextSet(
                context_set_id="context-1",
                example_id="ex-1",
                dataset="synthetic",
                condition="provided_context",
                chunk_ids=["chunk-gold"],
                expected_behavior="answer",
                metadata={},
            )
        ]
        pipeline = PipelineConfig.model_validate(
            {
                "name": "scoped_vanilla",
                "stages": {
                    "query_formulation": "none",
                    "retrieval": "bm25",
                    "reranking": "none",
                    "compression": "none",
                    "generation": "citation_free",
                    "verification": "none",
                },
                "retrieval": {"top_k": 2},
                "generation": {},
            }
        )
        runner = ConfigDrivenPipelineRunner(
            repo_root=REPO_ROOT,
            experiment=make_experiment(pipeline),
            dataset_slice=dataset_slice,
            settings=FormalRunSettings(
                run_id="scoped-context",
                save_outputs=False,
                bootstrap_samples=0,
                component_settings=ComponentFactorySettings(generator_kind="extractive"),
            ),
        )

        result = runner.run()

        self.assertEqual(
            [row.chunk_id for row in result.predictions[0].retrieved_chunks],
            ["chunk-gold"],
        )


def make_dataset_slice() -> DatasetSlice:
    example = Example(
        example_id="ex-1",
        dataset="synthetic",
        split="dev",
        task_type="short_answer",
        question="Who wrote the notes?",
        answers=[{"text": "Ada Lovelace", "type": "short"}],
        gold_evidence=[],
        metadata={"source_id": "source-1"},
    )
    gold = CorpusChunk(
        chunk_id="chunk-gold",
        doc_id="doc-gold",
        dataset="synthetic",
        title="Notes",
        text="Ada Lovelace wrote the notes.",
        sentences=[{"sentence_id": 0, "text": "Ada Lovelace wrote the notes."}],
        metadata={"source_example_id": "source-1"},
    )
    distractor = CorpusChunk(
        chunk_id="chunk-distractor",
        doc_id="doc-distractor",
        dataset="synthetic",
        title="Other",
        text="Charles Babbage designed the engine.",
        sentences=[{"sentence_id": 0, "text": "Charles Babbage designed the engine."}],
        metadata={"source_example_id": "source-1"},
    )
    qrel = Qrel(
        example_id="ex-1",
        chunk_id="chunk-gold",
        relevance=3,
        support_type="supporting_fact",
        metadata={"sentence_id": 0},
    )
    return DatasetSlice(
        dataset_name="synthetic",
        split_name="mvp",
        candidate_scope="synthetic_pooled_context",
        examples=[example],
        corpus=[gold, distractor],
        qrels_by_example={"ex-1": [qrel]},
    )


def make_experiment(pipeline: PipelineConfig, *, outputs_root: Path | None = None) -> ExperimentConfig:
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
            outputs_root=outputs_root or (Path(__file__).resolve().parents[1] / "outputs"),
            datasets={},
        ),
    )


def make_closed_book_pipeline() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "closed_book",
            "stages": {
                "query_formulation": "none",
                "retrieval": "none",
                "reranking": "none",
                "compression": "none",
                "generation": "citation_free",
                "verification": "none",
            },
            "generation": {},
        }
    )


class EmptyCitationGenerator:
    provider = "test"
    model = "empty-citation"
    last_attempt_count = 1

    def generate_text(self, prompt: str) -> str:
        return '{"answer":"","claims":[]}'


class RepairingCitationGenerator:
    provider = "test"
    model = "repairing-citation"
    last_attempt_count = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        context_id = "missing"
        if len(self.prompts) > 1:
            allowed_section = prompt.rsplit("context ids:", 1)[-1]
            matches = re.findall(r"C\d+-S[^\s,]+", allowed_section)
            context_id = matches[0].rstrip(".\"'")
        return json.dumps(
            {
                "short_answer": "Ada Lovelace",
                "long_answer": "Ada Lovelace wrote the notes.",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "text": "Ada Lovelace wrote the notes.",
                        "citations": [context_id],
                    }
                ],
            }
        )


def make_best_practice_pipeline() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "best_practice_rag",
            "stages": {
                "query_formulation": "llm_rewrite",
                "retrieval": "hybrid",
                "reranking": "cross_encoder",
                "compression": "sentence_top_n",
                "generation": "citation_aware",
                "verification": "none",
            },
            "retrieval": {"bm25_top_k": 2, "dense_top_k": 2, "final_top_k": 2},
            "reranking": {"top_k": 2},
            "compression": {"max_sentences": 2},
            "generation": {},
        }
    )


def make_verification_pipeline() -> PipelineConfig:
    data = make_best_practice_pipeline().to_json_dict()
    data["name"] = "rag_verification"
    data["stages"]["verification"] = "cove_style"
    data["verification"] = {"questions_per_answer": 5, "revise_answer": True}
    return PipelineConfig.model_validate(data)


if __name__ == "__main__":
    unittest.main()
