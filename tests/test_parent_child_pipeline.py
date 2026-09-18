from __future__ import annotations

import json
from pathlib import Path

from ragbench.core.base import load_pipeline
from ragbench.core.schema import RetrievalResult
from ragbench.techniques.parent_child.pipeline import ParentChildPipeline


def _result(chunk_id: str, parent_id: str, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        node_id=chunk_id,
        chunk_id=chunk_id,
        doc_id="doc1",
        text=f"child text for {chunk_id}",
        score=score,
        rank=rank,
        metadata={"parent_id": parent_id},
    )


def test_dedupe_by_parent_keeps_only_the_highest_ranked_child_per_parent() -> None:
    """Several children of the same parent commonly rank near the top
    together — without deduping, they'd fill multiple of the final
    rerank_top_k slots with, after parent-text resolution, the exact same
    section text."""
    results = [
        _result("c1", "p1", rank=1, score=0.9),
        _result("c2", "p1", rank=2, score=0.8),  # same parent as c1 — must be dropped
        _result("c3", "p2", rank=3, score=0.7),
        _result("c4", "p1", rank=4, score=0.6),  # same parent again — must be dropped
        _result("c5", "p3", rank=5, score=0.5),
    ]

    deduped = ParentChildPipeline._dedupe_by_parent(results, keep=3)

    assert [result.chunk_id for result in deduped] == ["c1", "c3", "c5"]
    assert [result.metadata["parent_id"] for result in deduped] == ["p1", "p2", "p3"]
    # Ranks are renumbered 1..N over the deduped list, not the original ranks.
    assert [result.rank for result in deduped] == [1, 2, 3]


def test_dedupe_by_parent_falls_back_to_chunk_id_when_parent_id_missing() -> None:
    no_parent = RetrievalResult(
        node_id="c9", chunk_id="c9", doc_id="doc1", text="x", score=0.1, rank=1, metadata={}
    )
    deduped = ParentChildPipeline._dedupe_by_parent([no_parent, no_parent], keep=5)
    assert len(deduped) == 1  # two identical chunk_id-keyed entries still collapse to one


def test_ingest_stores_parent_text_once_not_duplicated_per_child(tmp_path: Path) -> None:
    """Regression for the OOM-causing duplication: every child used to carry
    a full copy of its parent's text in both ``metadata.parent_text`` and
    ``text_for_generation``. Now each child carries only a small
    ``parent_id`` pointer, and the canonical parent text lives once in the
    ``parents.json`` sidecar.
    """
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    artifact = tmp_path / "artifact"
    pipeline.ingest("datasets/sample/docs", str(artifact))

    nodes = json.loads((artifact / "nodes.json").read_text(encoding="utf-8"))
    assert nodes, "expected at least one child node"
    for node in nodes:
        assert "parent_text" not in node["metadata"]
        assert "parent_id" in node["metadata"]

    parents = json.loads((artifact / "parents.json").read_text(encoding="utf-8"))
    referenced_parent_ids = {node["metadata"]["parent_id"] for node in nodes}
    assert referenced_parent_ids == set(parents.keys())
    for parent_id, parent in parents.items():
        assert parent["text"], f"parent {parent_id} must carry its section text exactly once"

    # Every child's generation text must be a (stripped) prefix window of its
    # own parent's text, not the parent's full text — proves no duplication
    # of the whole section into every child.
    multi_child_parents = {
        parent_id for parent_id in parents if sum(1 for n in nodes if n["metadata"]["parent_id"] == parent_id) > 1
    }
    if multi_child_parents:
        sample_parent_id = next(iter(multi_child_parents))
        children = [n for n in nodes if n["metadata"]["parent_id"] == sample_parent_id]
        for child in children:
            assert child["text_for_generation"] != parents[sample_parent_id]["text"]


def test_query_resolves_child_result_to_full_parent_text(tmp_path: Path) -> None:
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    artifact = tmp_path / "artifact"
    pipeline.ingest("datasets/sample/docs", str(artifact))
    pipeline.load(str(artifact))

    parents = json.loads((artifact / "parents.json").read_text(encoding="utf-8"))

    answer = pipeline.query("Chính sách nghỉ phép của nhân viên", mode="retrieval_only")
    assert answer.contexts, "expected at least one retrieved context"
    for context in answer.contexts:
        parent_id = context.metadata.get("parent_id")
        assert parent_id in parents
        # The context handed to generation is the *parent's* full section
        # text, resolved from the sidecar — not the short child snippet BM25
        # actually matched on.
        assert context.text == parents[parent_id]["text"]

    # Dedup contract: no two contexts in the same answer resolve to the same
    # parent section.
    parent_ids = [context.metadata.get("parent_id") for context in answer.contexts]
    assert len(parent_ids) == len(set(parent_ids))
