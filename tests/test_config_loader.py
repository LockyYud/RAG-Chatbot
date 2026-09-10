from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.core.config import ConfigError, _parse_minimal_yaml, _parse_scalar, _resolve_env, load_config


def test_load_config_parses_json_regardless_of_extension(tmp_path: Path) -> None:
    path = tmp_path / "technique.yaml"
    path.write_text('{"id": "naive_rag", "tags": ["baseline"]}', encoding="utf-8")
    assert load_config(path) == {"id": "naive_rag", "tags": ["baseline"]}


def test_load_config_resolves_env_substitution_with_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    path = tmp_path / "technique.yaml"
    path.write_text('{"model": "${SOME_UNSET_VAR:-gpt-4.1-mini}"}', encoding="utf-8")
    assert load_config(path) == {"model": "gpt-4.1-mini"}


def test_load_config_resolves_env_substitution_from_actual_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_SET_VAR", "claude-haiku")
    path = tmp_path / "technique.yaml"
    path.write_text('{"model": "${SOME_SET_VAR:-gpt-4.1-mini}"}', encoding="utf-8")
    assert load_config(path) == {"model": "claude-haiku"}


def test_load_config_falls_back_to_yaml_when_not_json(tmp_path: Path) -> None:
    path = tmp_path / "technique.yaml"
    path.write_text("id: naive_rag\ntags:\n  - baseline\n  - smoke\n", encoding="utf-8")
    assert load_config(path) == {"id": "naive_rag", "tags": ["baseline", "smoke"]}


def test_parse_scalar_booleans_null_and_numbers() -> None:
    assert _parse_scalar("true") is True
    assert _parse_scalar("False") is False
    assert _parse_scalar("null") is None
    assert _parse_scalar("~") is None
    assert _parse_scalar("42") == 42
    assert _parse_scalar("3.14") == 3.14
    assert _parse_scalar('"quoted"') == "quoted"
    assert _parse_scalar("'single'") == "single"
    assert _parse_scalar("plain text") == "plain text"
    assert _parse_scalar("") == {}


def test_parse_minimal_yaml_nested_mapping_and_list() -> None:
    """Regression: a key whose block turned out to be a nested *mapping* (not
    a list) used to raise ConfigError — the parser eagerly assumed every
    empty-value key opened a list, only correct when the next line happens to
    start with "- ". A one-line lookahead now decides list vs. mapping from
    the actual next line instead of guessing."""
    text = "\n".join(
        [
            "id: naive_rag",
            "implementation:",
            "  level: baseline",
            "  status: runnable",
            "tags:",
            "  - smoke_tests",
            "  - baseline",
            "empty_list:",
        ]
    )
    parsed = _parse_minimal_yaml(text)
    assert parsed == {
        "id": "naive_rag",
        "implementation": {"level": "baseline", "status": "runnable"},
        "tags": ["smoke_tests", "baseline"],
        "empty_list": {},
    }


def test_parse_minimal_yaml_list_of_mappings() -> None:
    text = "\n".join(
        [
            "items:",
            "  - id: a",
            "    weight: 1",
            "  - id: b",
            "    weight: 2",
        ]
    )
    parsed = _parse_minimal_yaml(text)
    assert parsed["items"] == [{"id": "a", "weight": 1}, {"id": "b", "weight": 2}]


def test_parse_minimal_yaml_skips_comments_and_blank_lines() -> None:
    text = "\n# a top-level comment\nid: naive_rag\n\n# another comment\nstatus: runnable\n"
    assert _parse_minimal_yaml(text) == {"id": "naive_rag", "status": "runnable"}


def test_parse_minimal_yaml_rejects_list_item_without_list_parent() -> None:
    with pytest.raises(ConfigError, match="List item without list parent"):
        _parse_minimal_yaml("id: naive_rag\n- oops\n")


def test_parse_minimal_yaml_rejects_a_line_with_no_colon_or_dash() -> None:
    with pytest.raises(ConfigError, match="Unsupported config line"):
        _parse_minimal_yaml("this line has no colon at all\n")


def test_resolve_env_recurses_through_nested_structures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NESTED_VAR", "resolved")
    value = {"a": ["${NESTED_VAR:-fallback}", {"b": "${UNSET_NESTED:-still_fallback}"}]}
    assert _resolve_env(value) == {"a": ["resolved", {"b": "still_fallback"}]}


def test_resolve_env_leaves_non_placeholder_strings_untouched() -> None:
    assert _resolve_env("just a plain string") == "just a plain string"
    assert _resolve_env(42) == 42
    assert _resolve_env(None) is None
