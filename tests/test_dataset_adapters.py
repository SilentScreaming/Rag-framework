from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_benchmark.config import ConfigLoader
from rag_benchmark.dataset_adapters import (
    garage_relevance_grade,
    parse_ragtruth_passages,
    write_ragtruth_judge_splits,
    write_standard_splits,
)
from rag_benchmark.dataset_catalog import DATASET_SPECS
from rag_benchmark.dataset_profiling import profile_json_records
from rag_benchmark.data import (
    ContextSet,
    load_all_corpus,
    load_annotations,
    load_candidate_responses,
    load_context_sets,
)
from rag_benchmark.runner import (
    build_normalized_split_slice,
    build_experiment_with_resolved_pipeline,
    parse_context_metadata_filters,
    select_context_set,
)
from rag_benchmark.schemas import AnnotationRecord, CandidateResponseRecord, ContextSetRecord


REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetAdaptersTest(unittest.TestCase):
    def test_standard_splits_keep_development_and_formal_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_standard_splits(
                root,
                [f"e-{index}" for index in range(1000)],
                dataset="synthetic",
                source_split="dev",
                formal_limit=500,
            )
            development = {
                json.loads(line)["example_id"]
                for line in (root / "splits" / "dev_small.jsonl").read_text().splitlines()
            }
            formal = {
                json.loads(line)["example_id"]
                for line in (root / "splits" / "formal.jsonl").read_text().splitlines()
            }

        self.assertEqual(len(development), 300)
        self.assertEqual(len(formal), 500)
        self.assertFalse(development & formal)

    def test_ragtruth_judge_splits_are_source_disjoint(self) -> None:
        strata = {
            "model:false:train": [
                (f"r-{source}-{response}", f"e-{source}", "train")
                for source in range(100)
                for response in range(3)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ragtruth_judge_splits(
                root,
                strata,
                calibration_limit=100,
                evaluation_limit=100,
            )
            calibration = {
                json.loads(line)["example_id"]
                for line in (root / "splits" / "judge_calibration.jsonl").read_text().splitlines()
            }
            evaluation = {
                json.loads(line)["example_id"]
                for line in (root / "splits" / "judge_eval.jsonl").read_text().splitlines()
            }

        self.assertFalse(calibration & evaluation)

    def test_catalog_covers_supported_datasets(self) -> None:
        self.assertEqual(
            {"kilt_nq", "kilt_hotpotqa", "alce", "scifact", "garage", "ragtruth", "rgb"},
            set(DATASET_SPECS),
        )
        self.assertEqual(DATASET_SPECS["garage"].license, "CC-BY-NC-4.0")

    def test_dataset_configs_are_typed(self) -> None:
        loader = ConfigLoader(REPO_ROOT)
        names = (
            "kilt_nq",
            "kilt_hotpotqa",
            "alce_asqa",
            "alce_qampari",
            "scifact",
            "garage",
            "ragtruth_qa",
            "rgb_en",
        )

        configs = [loader.load_dataset(name) for name in names]

        self.assertTrue(all(config.enabled for config in configs))
        self.assertEqual(configs[0].source.root_key, "kilt_nq")
        self.assertEqual(configs[-1].task_track, "verification_hallucination")

    def test_dataset_track_overrides_short_answer_pipeline_budget(self) -> None:
        experiment = build_experiment_with_resolved_pipeline(
            repo_root=REPO_ROOT,
            dataset_name="alce_asqa",
            pipeline_name="vanilla_rag",
            use_local_paths=False,
        )

        self.assertEqual(experiment.pipeline.task_profile.track, "long_answer_citation")
        self.assertEqual(experiment.pipeline.task_profile.answer_mode, "long")
        self.assertEqual(experiment.pipeline.generation["max_output_tokens"], 512)

    def test_ragtruth_passage_parser_preserves_passage_boundaries(self) -> None:
        passages = parse_ragtruth_passages("passage 1: Alpha.\n\n passage 2: Beta.\npassage 3: Gamma.")

        self.assertEqual(passages, ["Alpha.", "Beta.", "Gamma."])

    def test_garage_relevance_grades_are_explicit(self) -> None:
        self.assertEqual(garage_relevance_grade("YES", "ANSWER-THE-QUESTION"), 3)
        self.assertEqual(garage_relevance_grade("YES", "RELATED-INFORMATION"), 2)
        self.assertEqual(garage_relevance_grade("YES", "OUTDATED"), 1)
        self.assertEqual(garage_relevance_grade("NO", "UNANNOTATED"), 0)

    def test_json_extension_with_jsonl_content_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text('{"id": 1, "items": [{"text": "a"}]}\n{"id": 2, "items": []}\n', encoding="utf-8")

            report = profile_json_records(path)

        self.assertEqual(report["format"], "jsonl")
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["nested_fields"]["items"], ["text"])

    def test_auxiliary_records_preserve_offsets_and_context_conditions(self) -> None:
        response = CandidateResponseRecord(
            response_id="r1",
            example_id="e1",
            dataset="ragtruth_qa",
            split="test",
            text="Unsupported claim.",
        )
        annotation = AnnotationRecord(
            annotation_id="a1",
            example_id="e1",
            target_type="span",
            target_id="r1",
            label="Evident Baseless Info",
            start_char=0,
            end_char=11,
            text="Unsupported",
        )
        context = ContextSetRecord(
            context_set_id="c1",
            example_id="e1",
            dataset="rgb_en",
            condition="noise_robustness",
            chunk_ids=["d1"],
            expected_behavior="answer",
        )

        self.assertEqual(response.text[annotation.start_char : annotation.end_char], "Unsupported")
        self.assertEqual(context.condition, "noise_robustness")

    def test_candidate_response_schema_preserves_edge_whitespace_for_offsets(self) -> None:
        response = CandidateResponseRecord(
            response_id="r1",
            example_id="e1",
            dataset="ragtruth_qa",
            split="test",
            text="\nUnsupported claim.\n",
        )

        self.assertEqual(response.text, "\nUnsupported claim.\n")

    def test_auxiliary_files_load_by_selected_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate_responses.jsonl").write_text(
                '{"response_id":"r1","example_id":"e1","dataset":"ragtruth_qa",'
                '"split":"test","text":"Answer one.","model":"m1","metadata":{}}\n'
                '{"response_id":"r2","example_id":"e2","dataset":"ragtruth_qa",'
                '"split":"test","text":"Answer two.","model":"m2","metadata":{}}\n',
                encoding="utf-8",
            )
            (root / "context_sets.jsonl").write_text(
                '{"context_set_id":"c1","example_id":"e1","dataset":"rgb_en",'
                '"condition":"noise_robustness","chunk_ids":["d1"],'
                '"expected_behavior":"answer","metadata":{"noise_rate":0.0}}\n',
                encoding="utf-8",
            )
            (root / "annotations.jsonl").write_text(
                '{"annotation_id":"a1","example_id":"e1","target_type":"response",'
                '"target_id":"r1","label":"hallucinated","start_char":null,"end_char":null,'
                '"text":null,"evidence_chunk_ids":[],"metadata":{}}\n',
                encoding="utf-8",
            )

            responses = load_candidate_responses(root, example_ids={"e1"})
            contexts = load_context_sets(root, example_ids={"e1"})
            annotations = load_annotations(root, example_ids={"e1"})

        self.assertEqual([row.response_id for row in responses["e1"]], ["r1"])
        self.assertEqual(contexts["e1"][0].chunk_ids, ["d1"])
        self.assertEqual(annotations["e1"][0].target_id, "r1")

    def test_full_corpus_loader_does_not_filter_unlinked_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.jsonl").write_text(
                '{"chunk_id":"d1","doc_id":"d1","dataset":"scifact",'
                '"title":null,"text":"First.","sentences":[],"metadata":{}}\n'
                '{"chunk_id":"d2","doc_id":"d2","dataset":"scifact",'
                '"title":null,"text":"Second.","sentences":[],"metadata":{}}\n',
                encoding="utf-8",
            )

            corpus = load_all_corpus(root)

        self.assertEqual(set(corpus), {"d1", "d2"})

    def test_context_selection_distinguishes_rgb_noise_variants(self) -> None:
        contexts = [
            ContextSet(
                context_set_id=f"c-{rate}",
                example_id="e1",
                dataset="rgb_en",
                condition="noise_robustness",
                chunk_ids=["d1"],
                expected_behavior="answer",
                metadata={"noise_rate": rate},
            )
            for rate in (0.0, 0.4, 1.0)
        ]

        selected = select_context_set(
            contexts,
            condition="noise_robustness",
            metadata=parse_context_metadata_filters(["noise_rate=0.4"]),
        )

        self.assertEqual(selected.context_set_id, "c-0.4")

    def test_kilt_loader_refuses_gold_only_fullwiki_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "examples.jsonl").write_text(
                '{"example_id":"e1","dataset":"kilt_nq","split":"dev",'
                '"task_type":"short_answer","question":"Q?","answers":[],'
                '"gold_evidence":[],"metadata":{"source_id":"s1"}}\n',
                encoding="utf-8",
            )
            (root / "qrels.jsonl").write_text(
                '{"example_id":"e1","chunk_id":"gold","relevance":3,'
                '"support_type":"kilt_provenance","metadata":{}}\n',
                encoding="utf-8",
            )
            (root / "splits").mkdir()
            (root / "splits" / "formal.jsonl").write_text(
                '{"example_id":"e1"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "Refusing to substitute"):
                build_normalized_split_slice(
                    processed_root=root,
                    dataset_name="kilt_nq",
                    split_profile="formal",
                    limit=1,
                    relevance_threshold=3,
                    corpus_scope="kilt_full_wikipedia_2019",
                    retrieval_required=True,
                )

            closed_book_slice = build_normalized_split_slice(
                processed_root=root,
                dataset_name="kilt_nq",
                split_profile="formal",
                limit=1,
                relevance_threshold=3,
                corpus_scope="kilt_full_wikipedia_2019",
                retrieval_required=False,
            )

        self.assertEqual(closed_book_slice.corpus, [])


if __name__ == "__main__":
    unittest.main()
