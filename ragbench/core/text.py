from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

import numpy as np

WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(normalize_text(text))]


def token_count(text: str) -> int:
    return len(tokenize(text))


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def dense_cosine(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity, numpy-native so a memory-mapped embedding row (a
    view, not a Python list) never gets boxed into per-float objects just to
    be compared — see ``IndexedNode.embedding`` / ``load_nodes()``."""
    left_arr = np.asarray(left, dtype="float64")
    right_arr = np.asarray(right, dtype="float64")
    if left_arr.size == 0 or right_arr.size == 0 or left_arr.shape != right_arr.shape:
        return 0.0
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))


def mean_dense_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("Cannot average dense vectors with different dimensions")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def term_vector(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = sum(counts.values()) or 1
    return {term: count / total for term, count in counts.items()}


def first_relevant_sentence(text: str, query: str) -> str:
    query_terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        terms = set(tokenize(sentence))
        score = len(query_terms & terms)
        if score > best_score:
            best_score = score
            best_sentence = sentence.strip()
    return best_sentence or text.strip()
