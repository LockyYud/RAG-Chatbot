from __future__ import annotations

from evaluation.judge import parse_json_object
from raglab.datasets.synthetic import parse_json_array


def test_parse_json_array_extracts_embedded_array() -> None:
    rows = parse_json_array('prefix [{"question":"Q","ground_truth_answer":"A"}] suffix')
    assert rows == [{"question": "Q", "ground_truth_answer": "A"}]


def test_parse_json_object_extracts_embedded_object() -> None:
    payload = parse_json_object('```json\n{"faithfulness": 1}\n```')
    assert payload["faithfulness"] == 1
