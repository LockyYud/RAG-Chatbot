from __future__ import annotations

from collections import Counter

from ragbench.core.text import tokenize


def exact_match(prediction: str, ground_truth: str) -> float:
    """SQuAD-style exact match after Vietnamese-aware normalization.

    ``tokenize()`` already lowercases, drops punctuation (its word-boundary
    regex only keeps ``[\\wÀ-ỹ]`` runs), and collapses whitespace via
    ``normalize_text`` — exactly the normalization EM/F1 need — so comparing
    token sequences reuses it instead of duplicating a separate normalizer.
    """
    return 1.0 if tokenize(prediction) == tokenize(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-overlap F1 after the same normalization."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
