"""Sampling identity, reproducibility, and source validation."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from rag_benchmark.sampling import make_sample_manifest, load_sample_manifest
from rag_benchmark.stage_cache import canonical_hash


class SampleManifestTests(unittest.TestCase):
    def test_random_sample_preserves_purpose_and_original_population(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "formal.jsonl"
            ids = [f"e{i:02}" for i in range(10)]
            split.write_text(
                "".join(json.dumps({"example_id": item}) + "\n" for item in ids),
                encoding="utf-8",
            )
            sample = make_sample_manifest(
                dataset="d", data_version="v1", split="formal", split_path=split,
                fraction=0.5, seed=42, purpose="evaluation_sample",
            )
            path = root / "sample.json"
            path.write_text(json.dumps(sample), encoding="utf-8")
            loaded = load_sample_manifest(path, dataset="d", split_path=split, seed=42, data_version="v1")
            self.assertEqual(loaded["sample_ids"], sorted(random.Random(42).sample(ids, 5)))
            self.assertEqual(loaded["population_count"], 10)
            self.assertFalse(loaded["retrieval_corpus_subsampled"])
            self.assertEqual(loaded["purpose"], "evaluation_sample")
            sample["sample_ids"][0] = "changed"
            sample["manifest_hash"] = canonical_hash(
                {key: value for key, value in sample.items() if key != "manifest_hash"}
            )
            path.write_text(json.dumps(sample), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "random sample"):
                load_sample_manifest(path, dataset="d", split_path=split, seed=42, data_version="v1")

    def test_default_purpose_and_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "formal.jsonl"
            split.write_text('{"example_id":"e1"}\n', encoding="utf-8")
            sample = make_sample_manifest(dataset="d", data_version="v1", split="formal", split_path=split)
            self.assertEqual(sample["purpose"], "resource_and_format_sample")
            path = root / "sample.json"
            path.write_text(json.dumps(sample), encoding="utf-8")
            self.assertEqual(load_sample_manifest(path, dataset="d", split_path=split), sample)


if __name__ == "__main__":
    unittest.main()
