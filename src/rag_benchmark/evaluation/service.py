"""Independent evaluation of saved predictions with separate evaluation records."""
from __future__ import annotations

from copy import deepcopy
from threading import Lock, local

from rag_benchmark.stage_cache import canonical_hash
from rag_benchmark.evaluation.answers import assess_task
from rag_benchmark.evaluation.attribution import align_citations, citation_contributions
from rag_benchmark.evaluation.claims import ExtractionResult, EvaluatedClaim
from rag_benchmark.evaluation.contracts import EVALUATION_VERSION, MetricResult, ratio_result
from rag_benchmark.evaluation.exact_evidence import EvidenceDecision
from rag_benchmark.evaluation.cache import CACHE_PROTOCOL, exact_key, zero_reused_usage
from rag_benchmark.generation.usage import failure_usage


class EvaluationService:
    def __init__(self, *, extractor=None, judge=None, aligner=None, garage_evaluator=None,
                 experiment_cache=None):
        self.extractor, self.judge, self.aligner = extractor, judge, aligner
        self.garage_evaluator = garage_evaluator
        self.experiment_cache = experiment_cache
        self._stage_cache = {stage: {} for stage in ("extraction", "judge", "alignment", "garage")}
        self.extraction_cache = self._stage_cache["extraction"]
        self.judgment_cache = self._stage_cache["judge"]
        self._origins = {}
        self._context = local()
        self.calls = []
        self._locks_guard = Lock()
        self._cache_locks = {}

    def _cache_guard(self, kind, key):
        if self.experiment_cache is not None:
            return self.experiment_cache.guard(kind, key)
        with self._locks_guard:
            return self._cache_locks.setdefault((kind,key),Lock())

    def cache_payload(self):
        return {"schema_version": EVALUATION_VERSION, "cache_protocol": CACHE_PROTOCOL,
                "extractions": self.extraction_cache, "judgments": self.judgment_cache,
                "alignments": self._stage_cache["alignment"], "garage": self._stage_cache["garage"],
                "origins": {stage: {key: self._origins.get((stage, key), {}) for key in rows}
                            for stage, rows in self._stage_cache.items()}}

    def restore_cache(self, payload):
        if payload.get("schema_version") != EVALUATION_VERSION:
            raise ValueError("incompatible evaluation cache")
        # Keys in any other format cannot match the namespaced exact keys. Such
        # cache files stay readable, but their entries are never reused.
        for stage, field in (("extraction", "extractions"), ("judge", "judgments"),
                             ("alignment", "alignments"), ("garage", "garage")):
            for key, row in payload.get(field, {}).items():
                if self._completed(stage, row):
                    self._stage_cache[stage][key] = row
                    self._origins[stage, key] = payload.get("origins", {}).get(stage, {}).get(
                        key, {"source": "restored_evaluation_cache"})

    @staticmethod
    def _completed(stage, row):
        if stage == "alignment":
            return row.get("status") == "completed"
        if stage == "garage":
            # Uncalibrated is a completed model judgment, not a validated metric.
            # Its status is retained verbatim; calibration participates in key.
            return row.get("status") in {"computed", "uncalibrated"}
        if stage == "extraction" and row.get("answer_behavior", {}).get("status") in {"failed", "missing_input"}:
            return False
        return row.get("status") == "completed"

    def _reuse(self, stage, component, inputs, compute):
        key = exact_key(stage, component, inputs)
        with self._cache_guard(stage, key):
            memory = self._stage_cache[stage]
            entry = {"payload": memory[key], "origin": self._origins.get((stage,key), {})} if key in memory else None
            if entry is None and self.experiment_cache is not None:
                entry = self.experiment_cache.get(stage, key)
            hit = entry is not None
            if entry is None:
                value = compute()
                if self._completed(stage, value):
                    origin = deepcopy(getattr(self._context, "origin", {}))
                    entry = {"payload": value, "origin": origin}
                    if self.experiment_cache is not None:
                        stored = self.experiment_cache.put_first(stage, key, value, origin)
                        # First-valid wins even if a separate process raced us.
                        hit = not stored["inserted"]
                        entry = stored
                else:
                    return value
            memory[key] = deepcopy(entry["payload"])
            self._origins[stage,key] = deepcopy(entry["origin"])
            events = getattr(self._context, "reuse", None)
            event = {"stage": stage, "input_key": key, "cache_hit": hit,
                     "origin": entry["origin"], "source_result_hash": canonical_hash(entry["payload"])}
            if events is not None:
                events.append(event)
            value = zero_reused_usage(entry["payload"]) if hit else deepcopy(entry["payload"])
            if hit:
                value.setdefault("trace", {})["independent_evaluation_cache"] = {
                    "protocol": CACHE_PROTOCOL, "input_key": key, "origin": entry["origin"],
                    "physical_calls": 0, "source_trace_hash": canonical_hash(entry["payload"].get("trace", {}))}
            return value

    def _judge(self, claim, texts, *, question=None):
        from rag_benchmark.generation.budget import set_budget_stage
        set_budget_stage("evaluation_support")
        if self.judge is None:
            return EvidenceDecision("not_run", reason="judge_not_configured")
        uses_question = getattr(self.judge, "uses_question_context", False)
        inputs = {"claim": claim, "texts": texts, "question": question if uses_question else None}
        def compute():
            try:
                decision = (self.judge.judge_joint(claim, texts, question=question) if uses_question
                            else self.judge.judge_joint(claim, texts))
            except Exception as exc:
                decision = EvidenceDecision("failed", reason=type(exc).__name__)
            self.calls.append({"kind": "judge", "key": exact_key("judge", self.judge, inputs),
                               "trace": decision.trace, "status": decision.status})
            return decision.to_dict()
        return EvidenceDecision(**self._reuse("judge", self.judge, inputs, compute))

    def _extract(self, prediction, task_type):
        from rag_benchmark.generation.budget import set_budget_stage
        set_budget_stage("evaluation_extraction")
        if prediction.generator.get("schema_failure"):
            return ExtractionResult("failed", reason="generation_failed")
        if task_type == "scientific_claim_verification":
            from rag_benchmark.evaluation.tracks import normalize_verification_label
            if normalize_verification_label(prediction.answer) in {"SUPPORTS", "REFUTES", "INSUFFICIENT"}:
                return ExtractionResult("not_applicable", reason="label_only_answer")
        if self.extractor is None:
            return ExtractionResult("not_run", reason="independent_extractor_not_configured")
        inputs = {"question": prediction.question, "answer": prediction.answer, "task_type": task_type}
        builder = getattr(self.extractor, "build_prompt", None)
        if callable(builder):
            inputs["actual_prompt"] = builder(**inputs)
        def compute():
            try:
                extracted = self.extractor.extract(question=prediction.question, answer=prediction.answer, task_type=task_type)
            except Exception as exc:
                extracted = ExtractionResult("failed", reason=type(exc).__name__)
            self.calls.append({"kind": "extraction", "key": exact_key("extraction", self.extractor, inputs),
                               "trace": extracted.trace, "status": extracted.status})
            return extracted.to_dict()
        row = self._reuse("extraction", self.extractor, inputs, compute)
        claims = tuple(EvaluatedClaim(c["claim_id"], c["claim_text"], tuple(tuple(s) for s in c["answer_spans"]),
                                      tuple(c["segment_ids"]), c["claim_type"]) for c in row["claims"])
        return ExtractionResult(row["status"], claims, tuple(row["segments"]), row.get("reason"), row.get("trace", {}),
                                row.get("answer_behavior", {"status": "not_run", "label": "unknown", "reason": "behavior_not_assessed"}))

    def _alignment(self, prediction, claims):
        if self.aligner is None:
            return align_citations(prediction, claims)
        service = self
        class CachedAligner:
            def align(self, *, answer, claims, declarations):
                inputs = {"answer": answer,
                          "claims": [{"claim_id": c.claim_id, "text": c.claim_text} for c in claims],
                          "declarations": [{"id": c.claim_id, "text": c.claim_text} for c in declarations]}
                def compute():
                    try:
                        matches, trace = service.aligner.align(answer=answer, claims=claims, declarations=declarations)
                    except Exception as exc:
                        trace = getattr(exc, "trace", {"usage": failure_usage(exc)})
                        service.calls.append({"kind": "alignment", "key": exact_key("alignment", service.aligner, inputs),
                                              "trace": trace, "status": "failed"})
                        raise
                    service.calls.append({"kind": "alignment", "key": exact_key("alignment", service.aligner, inputs),
                                          "trace": trace, "status": "completed"})
                    return {"status": "completed", "matches": matches, "trace": trace}
                row = service._reuse("alignment", service.aligner, inputs, compute)
                return row["matches"], row["trace"]
        return align_citations(prediction, claims, aligner=CachedAligner())

    def evaluate(self, prediction, *, example=None, reference_evidence=None, gold_chunk_ids=None,
                 qrels=None, annotations=None, context_set=None, garage_relevant_evidence=None):
        task_type = example.task_type if example else prediction.task_track
        self._context.origin = {"run_id": prediction.run_id, "pipeline": prediction.pipeline,
                                "example_id": prediction.example_id,
                                "prediction_hash": canonical_hash(prediction.to_json_dict())}
        self._context.reuse = []
        from rag_benchmark.evaluation.task_registry import evaluate_task, scoped_example
        extraction = self._extract(prediction, task_type)
        task, results = evaluate_task(prediction, example=example, qrels=qrels,
                                      annotations=annotations, context_set=context_set,
                                      answer_behavior=extraction.answer_behavior)
        scopes = {}
        visible = prediction.metadata.get("generation_context", {}).get("visible_evidence")
        evidence_scopes = {"reference": reference_evidence,
                           "visible_context": [(f"Title: {v['title']}\n" if v.get("title") else "") + v["text"]
                                               for v in visible] if visible is not None else None}
        for scope, texts in evidence_scopes.items():
            key = f"{scope}_supported_claim_rate"
            if extraction.status != "completed" or not extraction.claims:
                status = "not_applicable" if extraction.status == "completed" or extraction.status == "not_applicable" else extraction.status
                results[key] = MetricResult(None, status=status, reason=extraction.reason or "no_factual_claims").to_dict()
                continue
            if texts is None:
                results[key] = MetricResult(None, status="missing_input", n_expected=len(extraction.claims),
                                            reason="evidence_scope_unavailable").to_dict()
                continue
            judgments = [{"claim_id": c.claim_id, **self._judge(c.claim_text, texts, question=prediction.question).to_dict()} for c in extraction.claims]
            scopes[scope] = judgments
            missing = sum(j["status"] != "completed" for j in judgments)
            for label in ("supported", "contradicted", "insufficient", "conflicting"):
                results[f"{scope}_{label}_claim_rate"] = ratio_result(
                    sum(j["label"] == label for j in judgments), len(judgments), missing=missing).to_dict()
        attribution = None
        if extraction.status == "completed" and extraction.claims:
            try:
                from rag_benchmark.generation.budget import set_budget_stage
                set_budget_stage("evaluation_citation_alignment")
                links, trace = self._alignment(prediction, extraction.claims)
                attribution = citation_contributions(prediction, extraction.claims,
                                                       judge=lambda claim,texts:self._judge(claim,texts,question=prediction.question),
                                                      links=links, gold_chunk_ids=gold_chunk_ids)
                attribution["alignment_trace"] = trace
                results.update(attribution["metrics"])
            except Exception as exc:
                results["citation_contribution_precision"] = MetricResult(None, status="failed", reason=type(exc).__name__).to_dict()
        else:
            for key in ("citation_contribution_precision", "supported_cited_claim_coverage"):
                results[key] = MetricResult(None, status="not_applicable" if extraction.status in {"completed", "not_applicable"} else extraction.status,
                                            reason=extraction.reason or "no_factual_claims").to_dict()
        # Re-evaluate both answers with the same task protocol when task scores are absent.
        intervention = None
        if prediction.verification and prediction.verification.original_answer is not None:
            original = prediction.model_copy(update={"answer": prediction.verification.original_answer, "answer_bundle": None})
            before = assess_task(original, scoped_example(prediction, example, context_set),
                answer_behavior={"status": "not_run", "label": "unknown", "reason": "original_behavior_not_independently_evaluated"})
            intervention = {"before": before, "after": task}
        garage = None
        if example and example.dataset == "garage" and self.garage_evaluator is not None and not prediction.generator.get("schema_failure"):
            garage_inputs = {"question": example.question, "answer": prediction.answer,
                             "reference_answers": [a["text"] for a in example.answers],
                             "relevant_evidence": garage_relevant_evidence,
                             "calibration": getattr(self.garage_evaluator, "calibration", {})}
            def compute_garage():
                row = self.garage_evaluator.evaluate(**{k:v for k,v in garage_inputs.items() if k != "calibration"})
                self.calls.append({"kind": "garage", "trace": row["trace"], "status": row["status"]})
                return row
            garage = self._reuse("garage", self.garage_evaluator, garage_inputs, compute_garage)
            results.update(garage["metrics"])
        return {"schema_version": EVALUATION_VERSION, "prediction_hash": canonical_hash(prediction.to_json_dict()),
                "garage": garage,
                "example_id": prediction.example_id, "pipeline": prediction.pipeline, "run_id": prediction.run_id,
                "answer_hash": canonical_hash(prediction.answer), "task": task, "extraction": extraction.to_dict(),
                "scope_judgments": scopes, "attribution": attribution, "metrics": results, "intervention": intervention,
                "reference_evidence_hash": canonical_hash(reference_evidence) if reference_evidence is not None else None,
                "human_validation": "not_performed", "full_answer_fact_coverage": "unknown",
                "evaluation_reuse": {"protocol": CACHE_PROTOCOL, "stages": self._context.reuse,
                                     "usage_scope": "cached_trace_usage_zero_new_calls_only_in_execution_budget"}}


def evaluated_prediction(prediction, result):
    """Create a reporting view while preserving the original prediction."""
    if canonical_hash(prediction.to_json_dict()) != result["prediction_hash"]:
        raise ValueError("evaluation sidecar does not match prediction")
    metrics = {key: value["value"] for key, value in result["metrics"].items() if value["status"] == "computed"}
    intervention_names = {}
    if result.get("intervention"):
        for prefix, stage in (("pre", "before"), ("post", "after")):
            task = result["intervention"][stage]
            if task["task_correct"] is not None:
                metrics[f"{prefix}_verification_task_correct"] = int(task["task_correct"])
            if task["score"] is not None:
                metrics[f"{prefix}_verification_task_score"] = task["score"]
                intervention_names[f"{prefix}_verification_task_score_metric"] = task.get("score_metric")
    return prediction.model_copy(update={"metrics": metrics, "metadata": {**prediction.metadata,
        "evaluation_protocol": EVALUATION_VERSION, "answer_metric_protocol": result["task"]["protocol_version"],
        "task_assessment": result["task"], "independent_evaluation": result, **intervention_names}})
