import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from app.evaluation.external import QasperCase


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not overlap or not predicted or not expected:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evidence_f1(prediction: Iterable[str], reference: Iterable[str]) -> float:
    predicted = set(prediction)
    expected = set(reference)
    if not predicted and not expected:
        return 1.0
    overlap = len(predicted & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def qasper_metrics(
    cases: Iterable[QasperCase], predictions: Mapping[str, Mapping[str, Any]]
) -> dict[str, float | int]:
    answer_scores = []
    evidence_scores = []
    missing = 0
    for case in cases:
        prediction = predictions.get(case.question_id)
        if prediction is None:
            missing += 1
            answer_scores.append(0.0)
            evidence_scores.append(0.0)
            continue
        answer = str(prediction.get("answer", ""))
        evidence = prediction.get("evidence", [])
        if not isinstance(evidence, list):
            raise TypeError(f"QASPER prediction {case.question_id} evidence must be a list.")
        answer_scores.append(max(token_f1(answer, ref.answer) for ref in case.references))
        evidence_scores.append(
            max(evidence_f1(evidence, ref.evidence) for ref in case.references)
        )
    count = len(answer_scores)
    return {
        "answer_f1": sum(answer_scores) / count if count else 0.0,
        "evidence_f1": sum(evidence_scores) / count if count else 0.0,
        "missing_predictions": missing,
        "eligible_cases": count,
    }


def classification_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("Gold and predicted labels must have equal length.")
    labels = sorted(set(gold) | set(predicted))
    per_label = {}
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        support = sum(g == label for g in gold)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    count = len(gold)
    return {
        "accuracy": sum(g == p for g, p in zip(gold, predicted, strict=True)) / count
        if count
        else 0.0,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels)
        if labels
        else 0.0,
        "eligible_cases": count,
        "per_label": per_label,
    }


def binary_sufficiency_metrics(gold: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    if len(gold) != len(predicted):
        raise ValueError("Gold and predicted decisions must have equal length.")
    true_positive = sum(g and p for g, p in zip(gold, predicted, strict=True))
    false_positive = sum(not g and p for g, p in zip(gold, predicted, strict=True))
    false_negative = sum(g and not p for g, p in zip(gold, predicted, strict=True))
    true_negative = sum(not g and not p for g, p in zip(gold, predicted, strict=True))
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    count = len(gold)
    return {
        "accuracy": (true_positive + true_negative) / count if count else 0.0,
        "false_positive_rate": false_positive / negatives if negatives else 0.0,
        "false_negative_rate": false_negative / positives if positives else 0.0,
        "positive_cases": positives,
        "negative_cases": negatives,
        "eligible_cases": count,
    }
