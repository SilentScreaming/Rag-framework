from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.config import ConfigLoader, PathConfig
from rag_benchmark.runner import build_experiment_with_resolved_pipeline


class ConfigLoaderTest(unittest.TestCase):
    def test_load_committed_pipeline_and_dataset_configs(self) -> None:
        loader = ConfigLoader(REPO_ROOT)

        dataset = loader.load_dataset("toy")
        pipeline = loader.load_pipeline("closed_book")

        self.assertEqual(dataset.name, "toy")
        self.assertEqual(pipeline.name, "closed_book")
        self.assertEqual(pipeline.model_refs(), set())

    def test_build_experiment_config_from_committed_examples(self) -> None:
        loader = ConfigLoader(REPO_ROOT)

        config = loader.build_experiment_config(
            dataset_name="toy",
            pipeline_name="closed_book",
            use_local_paths=False,
        )

        self.assertEqual(config.dataset.name, "toy")
        self.assertEqual(config.pipeline.name, "closed_book")
        self.assertEqual(config.model_refs, {})
        self.assertEqual(config.config_files["paths"], "configs/paths.example.json")

    def test_load_paths_requires_local_config_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "configs").mkdir()
            loader = ConfigLoader(repo_root)

            with self.assertRaises(FileNotFoundError):
                loader.load_paths(use_local=True)

    def test_path_config_rejects_relative_roots(self) -> None:
        with self.assertRaises(ValueError):
            PathConfig.model_validate(
                {
                    "schema_version": 1,
                    "datasets_root": "relative/datasets",
                    "models_root": str((Path(__file__).resolve().parents[1] / "models")),
                    "outputs_root": str((Path(__file__).resolve().parents[1] / "outputs")),
                    "datasets": {},
                }
            )

    def test_loader_finds_configs_by_file_stem_when_name_is_absent(self) -> None:
        loader = ConfigLoader(REPO_ROOT)

        retrieval = loader.load_model_dict("default_retrieval")

        self.assertIn("embedding_model", retrieval)
        self.assertEqual(retrieval["hybrid"]["fusion"], "rrf")

    def test_experiment_config_is_json_serializable(self) -> None:
        loader = ConfigLoader(REPO_ROOT)
        config = loader.build_experiment_config(
            dataset_name="toy",
            pipeline_name="closed_book",
            use_local_paths=False,
        )

        json.dumps(config.to_json_dict())

    def test_execution_profile_applies_nested_pipeline_overrides(self) -> None:
        config = build_experiment_with_resolved_pipeline(
            repo_root=REPO_ROOT,
            dataset_name="toy",
            pipeline_name="best_practice_rag",
            use_local_paths=False,
            execution_profile_name="budget_matched",
        )

        self.assertEqual(config.pipeline.budget.profile, "budget_matched")
        self.assertEqual(config.pipeline.query_formulation["max_queries"], 2)
        self.assertEqual(config.pipeline.reranking["top_k"], 5)
        self.assertIn("default_retrieval", config.model_refs)

    def test_formal_model_registry_pins_immutable_revisions(self) -> None:
        loader = ConfigLoader(REPO_ROOT)
        registry = loader.load_model_dict("default_retrieval")

        for section in ("embedding_model", "reranker", "evidence_judge"):
            revision = registry[section].get("revision")
            self.assertRegex(str(revision), r"^[0-9a-f]{40}$", msg=section)


if __name__ == "__main__":
    unittest.main()
