from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_benchmark.runner import load_completed_formal_run
from rag_benchmark.schemas import PredictionRecord


class ExperimentSuiteResumeTest(unittest.TestCase):
    def test_recovers_complete_matching_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run(Path(tmp), example_id="ex-1")

            result = load_completed_formal_run(
                run_dir,
                run_id="suite-vanilla_rag",
                pipeline="vanilla_rag",
                expected_example_ids={"ex-1"},
                expected_identity={"generation": "g1", "evaluation": "e1"},
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.predictions[0].example_id, "ex-1")

    def test_rejects_run_with_different_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run(Path(tmp), example_id="ex-1")

            result = load_completed_formal_run(
                run_dir,
                run_id="suite-vanilla_rag",
                pipeline="vanilla_rag",
                expected_example_ids={"ex-2"},
            )

        self.assertIsNone(result)

    @staticmethod
    def _write_run(root: Path, *, example_id: str) -> Path:
        from test_config_driven_runner import make_closed_book_pipeline, make_experiment
        from rag_benchmark.evaluation.identity import EvaluationIdentity
        from rag_benchmark.artifacts import summarize_cost_latency
        import yaml
        pipeline = make_closed_book_pipeline().model_copy(update={"name": "vanilla_rag"})
        experiment = make_experiment(pipeline)
        run_dir = root / "formal" / "suite-vanilla_rag"
        run_dir.mkdir(parents=True)
        prediction = PredictionRecord(
            run_id="suite-vanilla_rag",
            example_id=example_id,
            pipeline="vanilla_rag",
            question="Question?",
            answer="Answer.",
            metrics={"f1": 1.0},
            metadata=EvaluationIdentity("synthetic", "test", "dev", 42).metadata(),
        )
        payloads = {
            "metrics.json": {
                "run_id": "suite-vanilla_rag",
                "pipelines": {"vanilla_rag": {"f1": 1.0}},
            },
            "cost_latency.json": summarize_cost_latency([prediction]),
            "run_manifest.json": {"schema_version": "test", "run_identity": {"generation": "g1", "evaluation": "e1"}},
        }
        for name, payload in payloads.items():
            (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump({
            "run_id": "suite-vanilla_rag", "experiment": experiment.to_json_dict(), "settings": {"seed": 42}}), encoding="utf-8")
        (run_dir / "predictions.jsonl").write_text(
            json.dumps(prediction.to_json_dict()) + "\n", encoding="utf-8"
        )
        (run_dir / "logs.jsonl").write_text("{}\n", encoding="utf-8")
        for name in ("retrieval_traces.jsonl", "stage_traces.jsonl"):
            (run_dir / name).write_text("", encoding="utf-8")
        from rag_benchmark.artifacts import seal_formal_outputs
        seal_formal_outputs(run_dir, payloads["run_manifest.json"])
        return run_dir


if __name__ == "__main__":
    unittest.main()
