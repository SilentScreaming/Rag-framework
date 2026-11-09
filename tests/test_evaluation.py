"""Independent answer, evidence, citation, and statistical evaluation contracts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_predictions import main, load_reference_pools, aggregate_results
from rag_benchmark.analysis.diagnostics import diagnose_prediction
from rag_benchmark.data import Example, write_jsonl
from rag_benchmark.evaluation.answers import grouped_answer_metrics, assess_task
from rag_benchmark.evaluation.attribution import align_citations, citation_contributions
from rag_benchmark.evaluation.claims import answer_segments, parse_extraction, TextClaimExtractor, ExtractionResult
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision, TextExactJudge, NLIExactJudge
from rag_benchmark.evaluation.service import EvaluationService, evaluated_prediction
from rag_benchmark.evaluation.statistics import confidence_intervals, paired_comparisons, metric_value
from rag_benchmark.evaluation.tracks import evaluate_prediction_track_metrics, multiclass_macro_f1
from rag_benchmark.reporting.metrics_report import metric_rows, comparison_rows
from rag_benchmark.schemas import PredictionRecord, CitationRecord, ClaimCheckRecord
from rag_benchmark.stage_cache import canonical_hash


def prediction(answer="Paris is in France.", **kwargs):
    from rag_benchmark.evaluation.identity import EvaluationIdentity
    kwargs["metadata"] = {**EvaluationIdentity("fixture", "fixture-v1", "dev", 42).metadata(), **kwargs.get("metadata", {})}
    return PredictionRecord(example_id="e1", pipeline="a", question="Where is Paris?", answer=answer, **kwargs)


def grouped(kind, groups):
    return [{"text": alias, "type": "short" if kind == "asqa" else "list_item",
             "metadata": {"qa_pair_index" if kind == "asqa" else "item_index": i}}
            for i, aliases in enumerate(groups) for alias in aliases]


class FakeExtractor:
    identity = {"model": "fixture", "protocol": "answer-claims-v1"}
    def __init__(self):
        self.calls = 0
    def extract(self, *, question, answer, task_type):
        self.calls += 1
        return parse_extraction({"answer_behavior": {"label": "answered", "reason": "fixture"},
                                "segments": [{"segment_id": s["segment_id"], "kind": "factual", "claims": [s["text"]]}
                                               for s in answer_segments(answer)]}, question=question, answer=answer)


class FakeJudge:
    identity = {"model": "fixture", "protocol": "exact-evidence-v1"}
    def judge_joint(self, claim, texts):
        if "FAIL" in texts:
            return EvidenceDecision("failed")
        supported = claim in texts or (claim == "Joint fact." and set(texts) >= {"part A", "part B"})
        return EvidenceDecision("completed", "supported" if supported else "insufficient")


class EvaluationTest(unittest.TestCase):
    def test_reference_preprocessing_does_not_truncate_strict_evaluation(self):
        scores = grouped_answer_metrics("Paris [1]\nLondon", grouped("asqa", [["Paris"], ["London"]]), kind="asqa")
        self.assertEqual(scores["asqa_str_em_alce"], 0.5)
        self.assertEqual(scores["asqa_group_recall_strict"], 1)

    def test_cache_roundtrip_reuses_successes_without_new_calls(self):
        service = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge())
        p = prediction()
        first = service.evaluate(p, reference_evidence=[p.answer])
        restored = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge())
        restored.restore_cache(json.loads(json.dumps(service.cache_payload())))
        second = restored.evaluate(p, reference_evidence=[p.answer])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["extraction"]["claims"], second["extraction"]["claims"])
        self.assertTrue(all(s["cache_hit"] for s in second["evaluation_reuse"]["stages"]))
        self.assertEqual(restored.calls, [])
        with self.assertRaises(ValueError):
            restored.restore_cache({"schema_version": "old"})

    def test_text_backends_success_and_truncated_response(self):
        from rag_benchmark.generation.usage import TextResponse
        class Backend:
            model = "fixture"
            max_output_tokens = 512
            def __init__(self, text, finish="stop"):
                self.text, self.finish = text, finish
            def generate_text_result(self, prompt):
                self.prompt = prompt
                return TextResponse(self.text, 10, 10, "fixture", finish_reason=self.finish)
        payload = json.dumps({"segments": [{"segment_id": "s1", "kind": "factual", "claims": ["A fact."]}]})
        backend = Backend(payload)
        self.assertEqual(TextClaimExtractor(backend).extract(question="q", answer="A fact.", task_type="short").status, "completed")
        self.assertEqual(TextClaimExtractor(Backend(payload, "length")).extract(question="q", answer="A fact.", task_type="short").status, "failed")
        self.assertEqual(TextExactJudge(Backend('{"label":"supported"}')).judge_joint("fact", ["evidence"]).label, "supported")
        self.assertEqual(TextExactJudge(Backend('{"label":"invalid"}')).judge_joint("fact", ["evidence"]).status, "failed")
        self.assertEqual(TextExactJudge(Backend('{"label":"supported"}', "length")).judge_joint("fact", ["evidence"]).status, "failed")

    def test_hallucination_f1_uses_counts_not_sample_mean(self):
        a = prediction(metrics={"hallucination_span_tp": 1, "hallucination_span_fp": 0, "hallucination_span_fn": 0})
        b = prediction(metrics={"hallucination_span_tp": 0, "hallucination_span_fp": 0, "hallucination_span_fn": 9})
        self.assertAlmostEqual(metric_value([a, b], "hallucination_span_f1"), 2 / 11)
        self.assertIsNone(metric_value([a, prediction()], "hallucination_span_f1"))

    def test_label_separators_are_not_erased(self):
        from rag_benchmark.evaluation.tracks import normalize_verification_label
        for label in ("NOT_ENOUGH_INFO", "not enough info", "INSUFFICIENT_EVIDENCE", "not-enough-info"):
            self.assertEqual(normalize_verification_label(label), "INSUFFICIENT")


    def test_asqa_aliases_are_or_and_long_reference_excluded(self):
        rows = grouped("asqa", [["NYC", "New York City"], ["Paris"]]) + [{"text": "A long answer", "type": "long"}]
        scores = grouped_answer_metrics("NYC and Paris", rows, kind="asqa")
        self.assertEqual(scores["gold_answer_recall"], 1)
        self.assertEqual(scores["gold_answer_count"], 2)

    def test_asqa_reference_and_strict_are_explicitly_different(self):
        scores = grouped_answer_metrics("Yorkshire", grouped("asqa", [["York"]]), kind="asqa")
        self.assertEqual(scores["asqa_str_em_alce"], 1)
        self.assertEqual(scores["asqa_group_recall_strict"], 0)

    def test_asqa_group_order_and_alias_duplication_invariant(self):
        rows = grouped("asqa", [["NYC", "NYC"], ["Paris"]])
        self.assertEqual(grouped_answer_metrics("NYC", rows, kind="asqa"),
                         grouped_answer_metrics("NYC", list(reversed(rows)), kind="asqa"))

    def test_qampari_no_substring_false_hits(self):
        scores = grouped_answer_metrics("York", grouped("qampari", [["New York City"], ["York"]]), kind="qampari")
        self.assertEqual(scores["qampari_precision"], 1)
        self.assertEqual(scores["qampari_recall@5"], 0.5)

    def test_qampari_matching_and_duplicates(self):
        rows = grouped("qampari", [["x"], ["x", "y"]])
        scores = grouped_answer_metrics("x, x", rows, kind="qampari")
        self.assertEqual(scores["qampari_unique_entity_recall"], 0.5)
        self.assertEqual(scores["list_duplicate_rate"], 0.5)
        self.assertEqual(grouped_answer_metrics("x, y", rows, kind="qampari")["qampari_unique_entity_recall"], 1)

    def test_qampari_recall5_not_first_five_predictions(self):
        scores = grouped_answer_metrics("bad, bad2, bad3, bad4, bad5, York", grouped("qampari", [["York"]]), kind="qampari")
        self.assertEqual(scores["qampari_recall@5"], 1)

    def test_missing_grouping_is_not_zero_or_stale_score(self):
        example = Example("e1", "alce_asqa", "dev", "long", "Where is Paris?", [{"text": "Paris"}], [], {})
        p = prediction(task_track="long_answer_citation", metrics={"gold_answer_recall": 1, "citation_f1": 1})
        result = evaluate_prediction_track_metrics(p, example=example, qrels=[])
        self.assertNotIn("gold_answer_recall", result.metrics)
        self.assertNotIn("citation_f1", result.metrics)
        self.assertEqual(result.metadata["answer_metric_status"], "missing_answer_grouping")

    def test_metadata_can_restore_asqa_groups(self):
        result = grouped_answer_metrics("Paris", [], kind="asqa", metadata={"qa_pairs": [{"short_answers": ["Paris"]}]})
        self.assertEqual(result["gold_answer_recall"], 1)

    def test_empty_prediction_is_scored_but_missing_gold_is_not(self):
        self.assertEqual(grouped_answer_metrics("", grouped("qampari", [["York"]]), kind="qampari")["qampari_precision"], 0)
        with self.assertRaises(ValueError):
            grouped_answer_metrics("York", [], kind="qampari")

    def test_scifact_fixed_classes_and_label_alias(self):
        self.assertAlmostEqual(multiclass_macro_f1([("SUPPORTS", "SUPPORTS")]), 1 / 3)
        p = prediction("CONTRADICT", gold_answers=[{"text": "REFUTES", "type": "verification_label"}])
        self.assertTrue(assess_task(p)["task_correct"])
        self.assertEqual(multiclass_macro_f1([("SUPPORTS", "INVALID")]), 0)

    def label_rows(self, answers):
        rows = []
        for i, (gold, answer) in enumerate(answers):
            rows.append(prediction(answer, gold_answers=[{"text": gold, "type": "verification_label"}]).model_copy(
                update={"example_id": str(i), "task_track": "verification_hallucination"}))
        return rows

    def test_nonadditive_ci_and_paired_predictions(self):
        a = self.label_rows([("SUPPORTS", "SUPPORTS"), ("REFUTES", "SUPPORTS"), ("INSUFFICIENT", "INSUFFICIENT")])
        b = self.label_rows([("SUPPORTS", "SUPPORTS"), ("REFUTES", "REFUTES"), ("INSUFFICIENT", "INSUFFICIENT")])
        ci = confidence_intervals(a, keys=["verification_macro_f1"], samples=30)["verification_macro_f1"]
        self.assertEqual(ci["estimate"], metric_value(a, "verification_macro_f1"))
        self.assertIsNotNone(ci["ci95_low"])
        delta = paired_comparisons(a, b, keys=["verification_macro_f1"], samples=30)[0]
        self.assertAlmostEqual(delta["mean_delta"], 1 - ci["estimate"])
        reverse = paired_comparisons(b, a, keys=["verification_macro_f1"], samples=30)[0]
        self.assertAlmostEqual(reverse["mean_delta"], -delta["mean_delta"])

    def test_pairing_rejects_missing_duplicate_and_protocol_mismatch(self):
        a = prediction(metrics={"f1": 1})
        with self.assertRaises(ValueError):
            paired_comparisons([a], [], keys=["f1"])
        with self.assertRaises(ValueError):
            confidence_intervals([a, a], keys=["f1"])
        with self.assertRaises(ValueError):
            paired_comparisons([a], [a.model_copy(update={"metadata": {"evaluation_protocol": "other"}})], keys=["f1"])

    def test_missing_score_is_not_complete_case_mean(self):
        a = prediction(metrics={"f1": 1})
        b = prediction().model_copy(update={"example_id": "e2"})
        ci = confidence_intervals([a, b], keys=["f1"], samples=10)["f1"]
        self.assertIsNone(ci["estimate"])
        self.assertEqual(ci["status"], "not_estimable")

    def test_same_question_conditions_form_one_cluster(self):
        a = prediction(metrics={"f1": 1}, metadata={"context_set_id": "a"})
        b = prediction(metrics={"f1": 0}, metadata={"context_set_id": "b"})
        ci = confidence_intervals([a, b], keys=["f1"], samples=10)["f1"]
        self.assertEqual(ci["n_clusters"], 1)
        self.assertIsNone(ci["ci95_low"])

    def test_ci_reproducible_under_row_permutation(self):
        a = prediction(metrics={"f1": 1})
        b = prediction(metrics={"f1": 0}).model_copy(update={"example_id": "e2"})
        self.assertEqual(confidence_intervals([a, b], keys=["f1"], samples=20),
                         confidence_intervals([b, a], keys=["f1"], samples=20))

    def test_extraction_requires_every_segment(self):
        with self.assertRaises(ValueError):
            parse_extraction({"segments": []}, question="q", answer="A fact.")
        with self.assertRaises(ValueError):
            parse_extraction({"segments": [{"segment_id": "s1", "kind": "factual", "claims": []}]}, question="q", answer="A fact.")

    def test_extraction_preserves_source_offsets(self):
        answer = "  Fact A.\n Fact B!  "
        for row in answer_segments(answer):
            self.assertEqual(answer[row["start"]:row["end"]], row["text"])

    def test_empty_declared_claims_do_not_skip_independent_evaluation(self):
        service = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge())
        p = prediction("Paris is in France. Mars is cheese.")
        original = p.to_json_dict()
        result = service.evaluate(p, reference_evidence=["Paris is in France."])
        self.assertEqual(len(result["extraction"]["claims"]), 2)
        self.assertEqual(result["metrics"]["reference_supported_claim_rate"]["value"], 0.5)
        self.assertEqual(p.to_json_dict(), original)

    def test_extraction_cache_shared_between_methods(self):
        extractor = FakeExtractor()
        service = EvaluationService(extractor=extractor, judge=FakeJudge())
        p = prediction()
        service.evaluate(p, reference_evidence=[p.answer])
        service.evaluate(p.model_copy(update={"pipeline": "b"}), reference_evidence=[p.answer])
        self.assertEqual(extractor.calls, 1)

    def test_judge_failure_is_not_false_fact(self):
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(prediction(), reference_evidence=["FAIL"])
        metric = result["metrics"]["reference_supported_claim_rate"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["status"], "partial")

    def test_absent_visible_evidence_is_not_original_full_chunk(self):
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(prediction())
        self.assertEqual(result["metrics"]["visible_context_supported_claim_rate"]["status"], "missing_input")

    def test_abstention_does_not_get_perfect_support(self):
        class Abstain(FakeExtractor):
            def extract(self, **kwargs):
                return parse_extraction({"segments": [{"segment_id": "s1", "kind": "abstention", "claims": []}]},
                                        question=kwargs["question"], answer=kwargs["answer"])
        result = EvaluationService(extractor=Abstain(), judge=FakeJudge()).evaluate(prediction("UNKNOWN"), reference_evidence=[])
        self.assertIsNone(result["metrics"]["reference_supported_claim_rate"]["value"])
        self.assertEqual(result["metrics"]["reference_supported_claim_rate"]["status"], "not_applicable")

    def cited(self, answer, evidence):
        citations = [CitationRecord(chunk_id=str(i), citation_id=str(i), cited_text=text, claim_id="d1") for i, text in enumerate(evidence)]
        return prediction(answer, citations=citations, claim_checks=[ClaimCheckRecord(claim_id="d1", claim_text=answer, citations=citations)])

    def test_irrelevant_citation_lowers_contribution_precision(self):
        p = self.cited("Paris is in France.", ["Paris is in France.", "Mars is a planet."])
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(p)
        self.assertEqual(result["metrics"]["citation_contribution_precision"]["value"], 0.5)
        self.assertNotIn("citation_f1", result["metrics"])

    def test_joint_evidence_gets_contribution(self):
        p = self.cited("Joint fact.", ["part A", "part B"])
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(p)
        self.assertEqual(result["metrics"]["citation_contribution_precision"]["value"], 1)
        self.assertTrue(all(r["category"] == "joint_contribution" for r in result["attribution"]["claims"][0]["contributions"]))

    def test_duplicates_and_invalid_citations_not_dropped(self):
        p = self.cited("Paris is in France.", ["Paris is in France."])
        p.claim_checks[0].citations.append(p.citations[0])
        p.metadata["citation_generation"] = {"invalid_citation_count": 1}
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(p)
        self.assertEqual(result["attribution"]["duplicate_citations"], 1)
        self.assertEqual(result["metrics"]["citation_contribution_precision"]["value"], 0.5)

    def test_uncited_fact_remains_in_coverage_denominator(self):
        p = self.cited("Paris is in France.", ["Paris is in France."])
        p.answer += " Other fact."
        result = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge()).evaluate(p)
        # No semantic aligner: cannot pretend an unmatched fact was assessed as uncited.
        self.assertEqual(result["metrics"]["supported_cited_claim_coverage"]["denominator"], 2)
        self.assertEqual(result["metrics"]["supported_cited_claim_coverage"]["status"], "partial")

    def test_exact_nli_rejects_truncation_before_prediction(self):
        backend = SimpleNamespace(model_name="fixture", revision="1", max_length=2,
            entailment_threshold=0.65, contradiction_threshold=0.65,
            _load=lambda: (lambda *a, **k: {"input_ids": [1, 2, 3]}, None))
        result = NLIExactJudge(backend).judge_joint("claim", ["evidence"])
        self.assertEqual(result.status, "input_too_long")

    def test_unknown_diagnostic_is_not_retrieval_failure(self):
        result = diagnose_prediction(prediction())
        self.assertIsNone(result.signals["gold_count"])
        self.assertEqual(result.primary_error, "not_evaluated")

    def test_task_correct_can_coexist_with_unsupported_extra_fact(self):
        p = prediction(gold_answers=[{"text": "Paris is in France."}])
        service = EvaluationService(extractor=FakeExtractor(), judge=FakeJudge())
        result = service.evaluate(p, reference_evidence=["other evidence"])
        d = diagnose_prediction(evaluated_prediction(p, result))
        self.assertIn("correct_under_task_rule", [d.primary_error, *d.secondary_errors])
        self.assertIn("reference_support_incomplete", [d.primary_error, *d.secondary_errors])

    def test_sidecar_hash_prevents_wrong_answer_reuse(self):
        p = prediction()
        result = EvaluationService().evaluate(p)
        with self.assertRaises(ValueError):
            evaluated_prediction(p.model_copy(update={"answer": "changed"}), result)

    def test_report_keeps_missing_status_and_denominator(self):
        document = {"metric_results": {"p": {"x": {"value": None, "status": "failed", "n_expected": 3, "n_evaluated": 0}}}}
        row = metric_rows(document)[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["n_expected"], 3)
        self.assertIsNone(row["value"])

    def test_default_cli_does_not_call_models_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "examples.jsonl", [{"example_id": "e1", "dataset": "alce_qampari", "split": "dev",
                "task_type": "list_answer_citation", "question": "Where is Paris?", "answers": grouped("qampari", [["Paris"]]), "gold_evidence": [], "metadata": {}}])
            write_jsonl(root / "qrels.jsonl", [])
            p = prediction("Paris", task_track="long_answer_citation", metadata={"dataset": "alce_qampari"})
            path = root / "predictions.jsonl"
            write_jsonl(path, [p.to_json_dict()])
            before = path.read_bytes()
            args = ["--predictions", str(path), "--processed-root", str(root), "--output-dir", str(root / "eval"),
                    "--bootstrap-samples", "5", "--experiment-cache", str(root / "experiment.sqlite")]
            self.assertEqual(main(args), 0)
            self.assertEqual(path.read_bytes(), before)
            manifest = json.loads((root / "eval" / "evaluation_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertIsNone(manifest["extractor"])
            self.assertEqual(manifest["experiment_cache"]["protocol"], "independent-evaluation-exact-v1")
            self.assertTrue((root / "experiment.sqlite").exists())
            reuse = json.loads((root / "eval" / "evaluation_reuse.json").read_text())
            self.assertEqual(reuse["new_stage_attempts"], 0)
            self.assertTrue((root / "eval" / "all_metrics.csv").exists())
            cached_args = ["--predictions", str(path), "--processed-root", str(root), "--output-dir", str(root / "cached"),
                           "--cache-from", str(root / "eval"), "--bootstrap-samples", "5"]
            self.assertEqual(main(cached_args), 0)
            self.assertEqual((root / "cached" / "evaluation_calls.jsonl").read_text(), "")
            with self.assertRaises(FileExistsError):
                main(args)
            (root / "eval" / "evaluation_cache.json").write_text("{}")
            cached_args[cached_args.index(str(root / "cached"))] = str(root / "invalid")
            with self.assertRaises(ValueError):
                main(cached_args)


if __name__ == "__main__":
    unittest.main()
