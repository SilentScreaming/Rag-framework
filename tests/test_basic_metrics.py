from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.evaluation.basic import aggregate_metrics, best_qa_scores, retrieval_scores


class BasicMetricsTest(unittest.TestCase):
    def test_best_qa_scores(self) -> None:
        scores = best_qa_scores("The Arthur's Magazine", [{"text": "Arthur's Magazine"}])
        self.assertEqual(scores["em"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["accuracy"], 1.0)

    def test_retrieval_scores_multi_hop(self) -> None:
        scores = retrieval_scores(["a", "c", "d"], {"a", "b"}, k=3)
        self.assertEqual(scores["any_gold_hit@3"], 1.0)
        self.assertEqual(scores["all_gold_hit@3"], 0.0)
        self.assertEqual(scores["recall@3"], 0.5)
        self.assertEqual(scores["mrr@3"], 1.0)

    def test_aggregate_metrics_includes_cost_and_token_totals(self) -> None:
        metrics = aggregate_metrics(
            [
                {"estimated_cost": 0.25, "input_tokens": 10, "output_tokens": 2},
                {"estimated_cost": 0.75, "input_tokens": 30, "output_tokens": 4},
            ]
        )
        self.assertEqual(metrics["estimated_cost"], 0.5)
        self.assertEqual(metrics["total_estimated_cost"], 1.0)
        self.assertEqual(metrics["input_tokens"], 20.0)
        self.assertEqual(metrics["total_input_tokens"], 40.0)
        self.assertEqual(metrics["output_tokens"], 3.0)
        self.assertEqual(metrics["total_output_tokens"], 6.0)


if __name__ == "__main__":
    unittest.main()
