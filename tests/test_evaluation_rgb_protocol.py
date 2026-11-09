"""Independent evaluation trusts canonical RGB reviews, not output metadata."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.evaluate_predictions import main, neutral_rgb_reference_pools
from rag_benchmark.data import CorpusChunk, write_jsonl
from rag_benchmark.evaluation.identity import EvaluationIdentity
from rag_benchmark.rgb_protocol import rgb_review_identity
from rag_benchmark.schemas import PredictionRecord, AnswerBundleRecord


class EvaluationRGBProtocolTest(unittest.TestCase):
    def fixture(self, root):
        chunk = CorpusChunk("rgb-chunk", "rgb-doc", "rgb_en", "positive_wrong", "An unrelated passage.", [], {})
        write_jsonl(root/"corpus.jsonl", [asdict(chunk)])
        write_jsonl(root/"examples.jsonl", [{"example_id":"rgb-fixture", "dataset":"rgb_en", "split":"dev",
            "task_type":"rag_robustness_stress_test", "question":"Where is Paris?",
            "answers":[{"text":"France", "type":"short"}], "gold_evidence":[], "metadata":{}}])
        write_jsonl(root/"context_sets.jsonl", [{"context_set_id":"rgb-context", "example_id":"rgb-fixture",
            "dataset":"rgb_en", "condition":"noise", "chunk_ids":[chunk.chunk_id],
            "expected_behavior":"abstain", "metadata":{"positive_count":0}}])
        write_jsonl(root/"qrels.jsonl", [])
        p = PredictionRecord(example_id="rgb-fixture", pipeline="p", run_id="r", question="Where is Paris?",
            answer="UNKNOWN", answer_bundle=AnswerBundleRecord(long_answer="UNKNOWN", abstained=True),
            metadata={**EvaluationIdentity("rgb_en", "fixture-v1", "dev", 42, "rgb-context").metadata(),
                      "context_set":{"metadata":{"insufficiency_analysis_eligible":True}}})
        write_jsonl(root/"predictions.jsonl", [p.to_json_dict()])
        return chunk

    def args(self, root, name, overlay=None):
        args=["--predictions",str(root/"predictions.jsonl"),"--processed-root",str(root),
              "--output-dir",str(root/name),"--bootstrap-samples","0"]
        return args + (["--rgb-reviews",str(overlay)] if overlay else [])

    def test_no_review_ignores_prediction_eligibility_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            main(self.args(root,"eval"))
            row=json.loads((root/"eval/evaluations.jsonl").read_text())
            self.assertNotEqual(row["metrics"]["eligibility_accuracy"]["status"],"computed")
            self.assertIsNone(row["metrics"]["eligibility_accuracy"]["value"])
            manifest=json.loads((root/"eval/evaluation_manifest.json").read_text())
            self.assertEqual(manifest["rgb_protocol"]["input_display_protocol"],"rgb-neutral-inputs-v1")

    def test_verified_review_reaches_canonical_context_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);chunk=self.fixture(root)
            overlay=root/"review.json"
            overlay.write_text(json.dumps({"schema_version":"fixture-review-v1","reviews":{"rgb-context":{
                "input_hash":rgb_review_identity("Where is Paris?",[chunk]),"decision":"insufficient"}}}),encoding="utf-8")
            main(self.args(root,"eval",overlay))
            row=json.loads((root/"eval/evaluations.jsonl").read_text())
            self.assertEqual(row["metrics"]["eligibility_accuracy"]["value"],1)
            self.assertTrue(row["task"]["insufficiency_analysis_eligible"])
            manifest=json.loads((root/"eval/evaluation_manifest.json").read_text())
            self.assertTrue(manifest["rgb_protocol"]["review_overlay_hash"])
            self.assertTrue(manifest["context_sets_hash"])
            self.assertEqual((root/"eval/evaluation_calls.jsonl").read_text(),"")

    def test_stale_review_hash_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            overlay=root/"review.json"
            overlay.write_text(json.dumps({"reviews":{"rgb-context":{
                "input_hash":"stale","decision":"insufficient"}}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"rgb_review_evidence_mismatch"):
                main(self.args(root,"eval",overlay))
            self.assertFalse((root/"eval").exists())

    def test_reference_display_headers_are_neutral_bodies_preserved(self):
        pools={"e":["Title: positive_wrong\nA statement.","An unlabelled passage."]}
        neutral=neutral_rgb_reference_pools(pools)
        self.assertTrue(neutral["e"][0].startswith("Title: Passage "))
        self.assertTrue(neutral["e"][0].endswith("\nA statement."))
        self.assertEqual(neutral["e"][1],pools["e"][1])
        self.assertIn("positive_wrong",pools["e"][0])
