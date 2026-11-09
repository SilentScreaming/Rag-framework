import unittest
from rag_benchmark.evaluation.attribution import CitationAlignmentError, TextCitationAligner
from rag_benchmark.evaluation.claims import EvaluatedClaim
from rag_benchmark.generation.usage import TextResponse


class CitationAlignmentDiagnosticTests(unittest.TestCase):
    def check_error(self,text,code):
        class Backend:
            model="fixture"
            def generate_text_result(self,prompt):
                return TextResponse(text,11,12,"fixture",finish_reason="stop")
        with self.assertRaises(CitationAlignmentError) as got:
            TextCitationAligner(Backend()).align(answer="A fact.",claims=[EvaluatedClaim("f1","A fact.",(),())],
                declarations=[EvaluatedClaim("d1","A fact.",(),())])
        self.assertEqual(got.exception.trace["raw_response"],text)
        self.assertEqual(got.exception.trace["validation_error"],code)
        self.assertEqual(got.exception.trace["usage"]["output_tokens"],12)

    def test_invalid_json_keeps_raw_output(self):
        self.check_error("not JSON","invalid_alignment_json")

    def test_unknown_identifier_keeps_raw_output(self):
        self.check_error('{"matches":[{"claim_id":"invented","declaration_ids":["d1"],"status":"matched"}]}',
                         "invalid_alignment_identifiers")


if __name__=="__main__":unittest.main()
