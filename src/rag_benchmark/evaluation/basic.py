"""Basic answer and retrieval metrics shared by benchmark tracks."""

from __future__ import annotations

import re
import string
from collections import Counter
from math import log2
from statistics import mean
from typing import Any


SUM_AGGREGATE_KEYS = {"estimated_cost", "input_tokens", "output_tokens"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    return token_prf(prediction, gold)["f1"]


def token_prf(prediction: str, gold: str) -> dict[str, float]:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def best_qa_prf(prediction: str, answers: list[dict[str, Any]]) -> dict[str, float]:
    gold_texts = [str(answer.get("text", "")) for answer in answers if str(answer.get("text", "")).strip()]
    if not gold_texts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    candidates = [token_prf(prediction, gold) for gold in gold_texts]
    return max(candidates, key=lambda row: (row["f1"], row["precision"], row["recall"]))


def best_qa_scores(prediction: str, answers: list[dict[str, Any]]) -> dict[str, float]:
    gold_texts = [str(answer.get("text", "")) for answer in answers if str(answer.get("text", "")).strip()]
    if not gold_texts:
        return {"em": 0.0, "f1": 0.0, "accuracy": 0.0}
    em = max(exact_match(prediction, gold) for gold in gold_texts)
    return {
        "em": em,
        "f1": max(token_f1(prediction, gold) for gold in gold_texts),
        "accuracy": em,
    }


def retrieval_scores(retrieved_chunk_ids: list[str], gold_chunk_ids: set[str], *, k: int) -> dict[str, float | None]:
    if not gold_chunk_ids:
        return {
            f"any_gold_hit@{k}": None,
            f"all_gold_hit@{k}": None,
            f"recall@{k}": None,
            f"mrr@{k}": None,
        }
    top_ids = retrieved_chunk_ids[:k]
    hit_ids = set(top_ids) & gold_chunk_ids
    first_hit_rank = None
    for index, chunk_id in enumerate(top_ids, start=1):
        if chunk_id in gold_chunk_ids:
            first_hit_rank = index
            break
    return {
        f"any_gold_hit@{k}": float(bool(hit_ids)),
        f"all_gold_hit@{k}": float(gold_chunk_ids.issubset(set(top_ids))),
        f"recall@{k}": len(hit_ids) / len(gold_chunk_ids),
        f"mrr@{k}": 1 / first_hit_rank if first_hit_rank is not None else 0.0,
    }


def r_precision(retrieved_chunk_ids: list[str], gold_chunk_ids: set[str]) -> float | None:
    if not gold_chunk_ids:
        return None
    top_r = retrieved_chunk_ids[: len(gold_chunk_ids)]
    return len(set(top_r) & gold_chunk_ids) / len(gold_chunk_ids)


def ndcg_at_k(
    retrieved_chunk_ids: list[str],
    relevance_by_chunk: dict[str, int],
    *,
    k: int,
) -> float | None:
    if not relevance_by_chunk or len(set(relevance_by_chunk.values())) < 2:
        return None

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    actual = [int(relevance_by_chunk.get(chunk_id, 0)) for chunk_id in retrieved_chunk_ids[:k]]
    ideal = sorted((int(value) for value in relevance_by_chunk.values()), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0.0


def set_prf(predicted: set[Any], gold: set[Any]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "em": 1.0}
    hits = len(predicted & gold)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": float(predicted == gold),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    aggregate = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            aggregate[key] = mean(values)
            if key in SUM_AGGREGATE_KEYS:
                aggregate[f"total_{key}"] = sum(values)
    return aggregate
