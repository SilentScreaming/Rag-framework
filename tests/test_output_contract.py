"""Task-answer payload regressions; no network or model dependencies."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_benchmark.generation.answer_format import (
    build_answer_bundle, list_items_for_prediction, render_entity_list)
from rag_benchmark.schemas import PredictionRecord, RevisionDecisionRecord
from rag_benchmark.evaluation.claims import parse_extraction, TextClaimExtractor
from rag_benchmark.evaluation.answers import grouped_answer_metrics, assess_task
from rag_benchmark.evaluation.task_registry import evaluate_task
from rag_benchmark.evaluation.tracks import is_abstention, verification_intervention_metrics
from rag_benchmark.verification.cove import FactoredCoVeVerifier, FactoredCoVeConfig


def prediction(answer='Paris', **kwargs):
    return PredictionRecord(example_id='x', pipeline='fixture', question='Which places?', answer=answer, **kwargs)


class OutputContractTests(unittest.TestCase):
    def test_citation_list_payload_survives_prediction_roundtrip(self):
        from rag_benchmark.compression.sentence_select import CompressedContext, CompressedSentence
        from rag_benchmark.generation.citation import parse_citation_generation, apply_citation_generation
        context = CompressedContext(query='Where?', sentences=[CompressedSentence(
            context_id='c1', chunk_id='c', doc_id='d', title='Places', sentence_id=0,
            text='Paris is a city.', parent_rank=1, parent_score=1., selection_score=1.)])
        raw = json.dumps({'short_answer': None, 'long_answer': '"Washington, D.C.", Paris',
            'claims': [{'claim_id': 'fact', 'text': 'Paris is a city.', 'citations': ['c1']}],
            'list_items': [{'item_id': 'i1', 'text': 'Washington, D.C.', 'citation_ids': []},
                           {'item_id': 'i2', 'text': 'Paris', 'citation_ids': ['c1']}]})
        parsed = parse_citation_generation(raw, context=context, task_track='long_answer_citation',
                                           task_type='list_answer_citation')
        p = apply_citation_generation(prediction('draft', task_track='long_answer_citation'), parsed)
        restored = PredictionRecord.model_validate(p.to_json_dict())
        self.assertEqual([i.text for i in list_items_for_prediction(restored)], ['Washington, D.C.', 'Paris'])
        self.assertEqual(restored.answer_bundle.list_items[1].citation_ids, ['c1'])
        broken = json.loads(raw)
        broken['claims'] = []
        with self.assertRaisesRegex(ValueError, 'undeclared claim citations'):
            parse_citation_generation(json.dumps(broken), context=context, task_track='long_answer_citation',
                                      task_type='list_answer_citation')

    def test_quoted_entities_roundtrip_and_current_structured_payload(self):
        text = '"Washington, D.C." [1], "The ""Garden"""'
        bundle = build_answer_bundle(text, 'long_answer_citation', 'list_answer_citation')
        p = prediction(text, answer_bundle=bundle)
        with patch('rag_benchmark.generation.answer_format.parse_entity_list', side_effect=AssertionError('reparsed')):
            items = list_items_for_prediction(p)
        self.assertEqual([i.text for i in items], ['Washington, D.C.', 'The "Garden"'])
        self.assertEqual(render_entity_list(items), text)
        stale = p.model_copy(update={'answer': 'Paris'})
        with self.assertRaisesRegex(ValueError, 'stale_structured_list'):
            list_items_for_prediction(stale)
        altered = bundle.model_copy(update={'list_items': [bundle.list_items[0].model_copy(update={'text': 'London'}),
                                                           bundle.list_items[1]]})
        with self.assertRaisesRegex(ValueError, 'stale_structured_list'):
            list_items_for_prediction(p.model_copy(update={'answer_bundle': altered}))

    def test_malformed_list_is_recorded_without_pipeline_exception(self):
        bundle = build_answer_bundle('Paris; London', 'long_answer_citation', 'list_answer_citation')
        self.assertIsNone(bundle.list_items)
        self.assertEqual(bundle.metadata['list_format_status'], 'unavailable')
        abstained = build_answer_bundle('UNKNOWN', 'long_answer_citation', 'list_answer_citation')
        self.assertTrue(abstained.abstained)
        self.assertEqual(abstained.metadata['list_format_status'], 'computed')
        e = SimpleNamespace(dataset='alce_qampari', metadata={}, answers=[{'text': 'Paris', 'type': 'list_item', 'metadata': {'item_index': 0}}])
        self.assertEqual(assess_task(prediction('Paris; London', answer_bundle=bundle), e)['status'], 'failed')

    def test_structured_unique_metric_and_reference_metric_are_distinct(self):
        text = '"Washington, D.C.", Paris'
        bundle = build_answer_bundle(text, 'long_answer_citation', 'list_answer_citation')
        gold = [{'text': name, 'type': 'list_item', 'metadata': {'item_index': n}}
                for n, name in enumerate(['Washington, D.C.', 'Paris'])]
        scores = grouped_answer_metrics(text, gold, kind='qampari', list_items=bundle.list_items)
        self.assertEqual(scores['qampari_unique_entity_recall'], 1)
        self.assertEqual(scores['list_item_count'], 2)
        self.assertEqual(scores['alce_list_item_count'], 3)

    def test_behavior_preserves_partial_uncertain_and_missing(self):
        answer = 'Paris'
        base = {'segments': [{'segment_id': 's1', 'kind': 'factual', 'claims': ['Paris is the answer.']}]}
        for label, abstained in [('answered', False), ('abstained', True), ('partial', False), ('uncertain', None), ('unknown', None)]:
            result = parse_extraction({**base, 'answer_behavior': {'label': label, 'reason': 'fixture'}}, question='Where?', answer=answer)
            self.assertIs(is_abstention(prediction(answer), answer_behavior=result.answer_behavior), abstained)
            self.assertEqual(len(result.claims), 1)
        missing = parse_extraction(base, question='Where?', answer=answer)
        self.assertEqual(missing.answer_behavior['status'], 'missing_input')
        self.assertIsNone(is_abstention(prediction(answer), answer_behavior=missing.answer_behavior))

    def test_behavior_uses_the_existing_single_extraction_call(self):
        payload = {'segments': [{'segment_id': 's1', 'kind': 'abstention', 'claims': []}],
                   'answer_behavior': {'label': 'abstained', 'reason': 'Declines to supply a value.'}}
        class Generator:
            model = 'fixture-model'
            max_output_tokens = 32768
            reasoning_effort = None
            def __init__(self): self.calls = []
            def generate_text(self, prompt):
                self.calls.append(prompt)
                return json.dumps(payload)
        generator = Generator()
        extractor = TextClaimExtractor(generator)
        result = extractor.extract(question='Where?', answer='No answer can be supplied.', task_type='fixture')
        self.assertEqual(result.answer_behavior['label'], 'abstained')
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0], extractor.build_prompt(question='Where?', answer='No answer can be supplied.', task_type='fixture'))

    def test_uncertain_behavior_does_not_override_content_score_or_become_refusal(self):
        e = SimpleNamespace(dataset='garage', answers=[{'text': 'Paris'}],
                            metadata={'expected_behavior': 'answer'}, task_type='grounding_and_selective_refusal')
        task, metrics = evaluate_task(prediction(), example=e,
            answer_behavior={'status': 'computed', 'label': 'uncertain'})
        self.assertEqual(metrics['f1']['value'], 1.)
        self.assertIsNone(metrics['eligibility_accuracy']['value'])
        self.assertIsNone(task['task_correct'])

    def test_rgb_requires_review_for_semantic_insufficiency(self):
        e = SimpleNamespace(dataset='rgb', answers=[{'text': 'Paris'}],
                            metadata={'expected_behavior': 'abstain'}, task_type='rag_robustness_stress_test')
        task, metrics = evaluate_task(prediction('UNKNOWN'), example=e)
        self.assertIsNone(metrics['eligibility_accuracy']['value'])
        from rag_benchmark.evaluation.tracks import deflection_metrics
        self.assertEqual(deflection_metrics([prediction('UNKNOWN', metadata={'task_assessment': task})],
                                           examples_by_id={'x': e}), {})
        e.metadata.update(insufficiency_analysis_eligible=True, evidence_review_status='insufficient')
        _, metrics = evaluate_task(prediction('UNKNOWN'), example=e)
        self.assertEqual(metrics['eligibility_accuracy']['value'], 1.)


    def test_replay_keeps_named_deltas_and_unknown_behavior(self):
        from rag_benchmark.verification.replay import summarize_replay
        from rag_benchmark.evaluation.identity import EvaluationIdentity
        from rag_benchmark.data import Example
        from rag_benchmark.schemas import VerificationRecord
        before, after, examples = [], [], {}
        for ident, expected in [('a', 'answer'), ('b', 'abstain')]:
            md = EvaluationIdentity('garage', 'fixture-v1', 'dev', 42).metadata()
            md['task_assessment'] = {'answer_behavior': {'status': 'computed', 'label': 'answered'}}
            a = prediction('London', task_track='verification_hallucination', metadata=md).model_copy(update={'example_id': ident})
            b = a.model_copy(update={'answer': 'Paris' if expected == 'answer' else 'UNKNOWN',
                'verification': VerificationRecord(status='revised', original_answer=a.answer),
                'metadata': {**md, 'task_assessment': {'answer_behavior': {'status': 'computed',
                    'label': 'answered' if expected == 'answer' else 'abstained'}}}})
            examples[ident] = Example(ident, 'garage', 'dev', 'grounding_and_selective_refusal', a.question,
                                      [{'text': 'Paris'}], [], {'expected_behavior': expected})
            before.append(a); after.append(b)
        result = summarize_replay(before, after, examples, samples=5)
        self.assertEqual([p['score_delta'] for p in result['paired']], [1., 1.])
        self.assertIsNone(result['mean_task_score_delta'])
        self.assertEqual(result['answer_coverage'], .5)
        after[0] = after[0].model_copy(update={'metadata': {**after[0].metadata, 'task_assessment': {}}})
        result = summarize_replay(before, after, examples, samples=5)
        self.assertIsNone(result['answer_coverage'])
        self.assertEqual(result['answer_behavior_n_evaluated'], 1)

    def test_short_revision_uses_candidate_answer_not_fact_or_reason(self):
        verifier = FactoredCoVeVerifier(judge=None, config=FactoredCoVeConfig(revise_short_answer=True))
        p = prediction('Old', answer_bundle=build_answer_bundle('Old'))
        decision = RevisionDecisionRecord(claim_id='c', original_text='Old wrote the book.',
            action='replace', revised_text='Mira wrote the book because the evidence says so.',
            reason='This is a reason.', metadata={'replacement_answer': 'Mira'})
        self.assertEqual(verifier._apply_decisions(p, [decision]), 'Mira')
        missing = decision.model_copy(update={'metadata': {}})
        self.assertIsNone(verifier._apply_decisions(p, [missing]))

    def test_named_score_movement_requires_one_shared_metric(self):
        def row(name, after_name=None):
            return prediction(metrics={'pre_verification_task_score': .2, 'post_verification_task_score': .7},
                metadata={'pre_verification_task_score_metric': name,
                          'post_verification_task_score_metric': after_name or name})
        self.assertAlmostEqual(verification_intervention_metrics([row('f1')])['net_verification_gain'], .5)
        for rows in [[row('f1', 'eligibility_accuracy')], [row('f1'), row('em')], [row(None)]]:
            self.assertNotIn('net_verification_gain', verification_intervention_metrics(rows))


if __name__ == '__main__':
    unittest.main()
