from __future__ import annotations

from raglab.core.schema import RetrievalResult
from raglab.inference.controllers.agentic import (
    AgentAction,
    AgenticRetrievalController,
    AgentObservation,
    parse_action,
)


def _result(node_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        node_id=node_id,
        chunk_id=node_id,
        doc_id="doc",
        text=f"text {node_id}",
        score=1.0 / rank,
        rank=rank,
    )


def _tool(node_ids: list[str]):
    def run(query: str, top_k: int) -> list[RetrievalResult]:
        return [_result(nid, i + 1) for i, nid in enumerate(node_ids[:top_k])]

    return run


def test_loop_searches_then_answers_and_dedupes() -> None:
    tools = {
        "keyword": _tool(["a", "b"]),
        "semantic": _tool(["b", "c"]),  # "b" overlaps -> must dedupe
    }

    # Scripted policy: search keyword, then semantic, then answer.
    script = [
        AgentAction(kind="search", tool="keyword", query="q1"),
        AgentAction(kind="search", tool="semantic", query="q2"),
        AgentAction(kind="answer", answer="final [C1]"),
    ]
    calls = {"n": 0}

    def policy(obs: AgentObservation) -> AgentAction:
        action = script[calls["n"]]
        calls["n"] += 1
        return action

    controller = AgenticRetrievalController(tools, policy, max_steps=5, per_tool_top_k=5)
    result = controller.run("question")

    assert result.answer == "final [C1]"
    assert result.stopped_reason == "answered"
    assert result.tool_calls == 2
    assert [r.node_id for r in result.evidence] == ["a", "b", "c"]  # deduped, discovery order


def test_well_behaved_policy_answers_on_last_step() -> None:
    tools = {"hybrid": _tool(["a", "b", "c"])}

    # Respects the must_answer flag the controller sets on the final step.
    def behaved(obs: AgentObservation) -> AgentAction:
        if obs.must_answer:
            return AgentAction(kind="answer", answer="answered now")
        return AgentAction(kind="search", tool="hybrid", query="more")

    result = AgenticRetrievalController(tools, behaved, max_steps=3).run("question")

    assert result.stopped_reason == "answered"
    assert result.answer == "answered now"
    assert result.tool_calls == 2  # searched steps 0,1 then answered on step 2


def test_max_steps_guard_stops_a_runaway_policy() -> None:
    tools = {"hybrid": _tool(["a", "b", "c"])}

    # A broken policy that never answers — the guard must still terminate the run.
    def runaway(obs: AgentObservation) -> AgentAction:
        return AgentAction(kind="search", tool="hybrid", query="more")

    result = AgenticRetrievalController(tools, runaway, max_steps=3).run("question")

    assert result.stopped_reason == "max_steps"
    assert result.answer == ""  # never produced one
    assert result.tool_calls == 3  # exactly max_steps searches, no infinite loop


def test_unknown_tool_falls_back_to_first_tool() -> None:
    tools = {"hybrid": _tool(["a"]), "keyword": _tool(["z"])}
    script = [AgentAction(kind="search", tool="does_not_exist", query="q"), AgentAction(kind="answer", answer="x")]
    calls = {"n": 0}

    def policy(obs: AgentObservation) -> AgentAction:
        action = script[calls["n"]]
        calls["n"] += 1
        return action

    result = AgenticRetrievalController(tools, policy, max_steps=5).run("question")
    # Fell back to the first registered tool ("hybrid") -> evidence "a".
    assert [r.node_id for r in result.evidence] == ["a"]


def test_parse_action_variants() -> None:
    search = parse_action('{"thought":"t","action":"search","tool":"keyword","query":"foo"}')
    assert search.kind == "search" and search.tool == "keyword" and search.query == "foo"

    answer = parse_action('Here you go: {"action":"answer","answer":"42"} trailing')
    assert answer.kind == "answer" and answer.answer == "42"

    # Unparseable + must_answer -> raw text becomes the answer.
    forced = parse_action("no json here", must_answer=True)
    assert forced.kind == "answer" and forced.answer == "no json here"

    # Unparseable without must_answer -> default to a search action.
    fallback = parse_action("garbage")
    assert fallback.kind == "search"
