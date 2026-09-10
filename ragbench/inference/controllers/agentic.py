"""Agentic retrieval controller — the inference-time core of A-RAG-style RAG.

The 2026 shift in RAG is one of *control flow*: instead of a fixed
``retrieve → generate`` pipeline, the LLM acts as an agent that decides, step by
step, *whether* to retrieve, *which* tool to use, and *what* to query — then
judges when it has enough evidence to answer.  (See A-RAG, arXiv:2602.03442, and
the System-1/System-2 reasoning-RAG survey, arXiv:2506.10408.)

This module is the reusable engine for that loop.  It is deliberately
*training-free*: the decision policy is just an LLM prompted to emit a JSON
action, so it runs on any chat model with no RL.  A policy is injectable, which
keeps the loop unit-testable offline with a scripted policy.

Design
------
- ``tools``: name → ``Callable[[query, top_k], list[RetrievalResult]]``.  A
  technique wires in whatever retrievers it has (BM25, dense, RRF hybrid, a
  chunk-reader, …).
- ``policy``: ``Callable[[AgentObservation], AgentAction]`` — picks the next
  action from the running state.  The default LLM policy lives in
  :func:`make_llm_policy`.
- Guards: ``max_steps`` and an optional ``max_tool_calls`` bound the loop so a
  confused agent can't spin forever or run up cost.
- Output: an :class:`AgentResult` carrying the answer, the de-duplicated
  evidence (in discovery order), and a structured ``trace`` for failure
  analysis — exactly what the roadmap's multi-hop criteria require.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragbench.core.schema import RetrievalResult

# A retrieval tool: (query, top_k) -> ranked results.
Tool = Callable[[str, int], list[RetrievalResult]]


@dataclass(slots=True)
class AgentAction:
    """One decision from the policy: search with a tool, or answer/stop."""

    kind: str  # "search" | "answer"
    tool: str | None = None
    query: str | None = None
    answer: str | None = None
    thought: str = ""


@dataclass(slots=True)
class AgentObservation:
    """State handed to the policy before each decision."""

    question: str
    step_index: int
    max_steps: int
    must_answer: bool
    tool_names: list[str]
    trace: list[dict[str, Any]]
    evidence: list[RetrievalResult]


@dataclass(slots=True)
class AgentResult:
    answer: str
    evidence: list[RetrievalResult]
    trace: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = ""
    tool_calls: int = 0


Policy = Callable[[AgentObservation], AgentAction]


class AgenticRetrievalController:
    """Run an LLM-driven, multi-step retrieval loop over a set of tools.

    Parameters
    ----------
    tools:
        Mapping of tool name to a retrieval callable.
    policy:
        Decision function.  Use :func:`make_llm_policy` for the real LLM policy,
        or inject a scripted one in tests.
    max_steps:
        Hard cap on reasoning/retrieval steps.
    per_tool_top_k:
        ``top_k`` passed to each tool call.
    max_evidence:
        Cap on accumulated evidence kept (discovery order preserved).
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        policy: Policy,
        *,
        max_steps: int = 4,
        per_tool_top_k: int = 5,
        max_evidence: int = 30,
    ) -> None:
        if not tools:
            raise ValueError("AgenticRetrievalController needs at least one tool")
        self.tools = tools
        self.policy = policy
        self.max_steps = max_steps
        self.per_tool_top_k = per_tool_top_k
        self.max_evidence = max_evidence

    def run(self, question: str) -> AgentResult:
        evidence: list[RetrievalResult] = []
        seen: set[str] = set()
        trace: list[dict[str, Any]] = []
        tool_calls = 0

        for step in range(self.max_steps):
            must_answer = step == self.max_steps - 1
            action = self.policy(
                AgentObservation(
                    question=question,
                    step_index=step,
                    max_steps=self.max_steps,
                    must_answer=must_answer,
                    tool_names=list(self.tools),
                    trace=trace,
                    evidence=evidence,
                )
            )

            if action.kind == "answer":
                trace.append({"step": step, "action": "answer", "thought": action.thought})
                return AgentResult(
                    answer=action.answer or "",
                    evidence=evidence,
                    trace=trace,
                    stopped_reason="answered",
                    tool_calls=tool_calls,
                )

            # action.kind == "search"
            tool_name = action.tool if action.tool in self.tools else next(iter(self.tools))
            query = action.query or question
            results = self.tools[tool_name](query, self.per_tool_top_k)
            tool_calls += 1

            new = [r for r in results if r.node_id not in seen]
            for result in new:
                if len(evidence) >= self.max_evidence:
                    break
                seen.add(result.node_id)
                evidence.append(result)

            trace.append(
                {
                    "step": step,
                    "action": "search",
                    "tool": tool_name,
                    "query": query,
                    "thought": action.thought,
                    "new_evidence": len(new),
                    "total_evidence": len(evidence),
                }
            )

        # Ran out of steps without an explicit answer — make one final attempt.
        final = self.policy(
            AgentObservation(
                question=question,
                step_index=self.max_steps,
                max_steps=self.max_steps,
                must_answer=True,
                tool_names=list(self.tools),
                trace=trace,
                evidence=evidence,
            )
        )
        answer = final.answer or "" if final.kind == "answer" else ""
        trace.append({"step": self.max_steps, "action": "forced_answer", "thought": final.thought})
        return AgentResult(
            answer=answer,
            evidence=evidence,
            trace=trace,
            stopped_reason="max_steps",
            tool_calls=tool_calls,
        )


# ─── LLM policy ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are an agentic RAG controller. You answer a question by deciding, step "
    "by step, whether to RETRIEVE more evidence or to ANSWER now.\n"
    "Available retrieval tools: {tools}.\n"
    " - keyword: lexical/BM25 search, best for exact terms, codes, names.\n"
    " - semantic: dense embedding search, best for paraphrase/meaning.\n"
    " - hybrid: fused lexical+semantic, a strong default.\n"
    " - chunk_read: expand a known result to its full parent section; pass the "
    "node_id as the query.\n"
    "Reply with ONE JSON object and nothing else:\n"
    '  {{"thought": "...", "action": "search", "tool": "<name>", "query": "..."}}\n'
    "  or\n"
    '  {{"thought": "...", "action": "answer", "answer": "...[C1][C2]..."}}\n'
    "Search only when more evidence would help. Answer as soon as the evidence is "
    "sufficient. Ground every claim in the evidence with [C#] citations."
)


def make_llm_policy(
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 500,
    client: Any | None = None,
) -> Policy:
    """Build a policy that asks an LLM for the next :class:`AgentAction`."""
    from ragbench.providers.llm_client import LLMClient

    llm = client or LLMClient()
    system = SYSTEM_PROMPT.format(tools="keyword, semantic, hybrid, chunk_read")
    runtime: dict[str, Any] = {
        "calls": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "estimated_cost": 0.0,
        "latency_ms": 0.0,
    }

    def policy(obs: AgentObservation) -> AgentAction:
        completion = llm.create_chat_completion(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _render_user_prompt(obs)},
            ],
        )
        runtime["calls"] += 1
        for key in runtime["usage"]:
            runtime["usage"][key] += int(completion.usage.get(key, 0))
        runtime["estimated_cost"] = round(float(runtime["estimated_cost"]) + completion.estimated_cost, 8)
        runtime["latency_ms"] = round(float(runtime["latency_ms"]) + completion.latency_ms, 3)
        return parse_action(completion.text, must_answer=obs.must_answer)

    # Callable functions can carry runtime state while retaining the simple
    # Policy protocol used by the controller's offline unit tests.
    policy.runtime = runtime  # type: ignore[attr-defined]
    return policy


def _render_user_prompt(obs: AgentObservation) -> str:
    lines = [f"QUESTION: {obs.question}", ""]
    if obs.evidence:
        lines.append("EVIDENCE SO FAR:")
        for index, result in enumerate(obs.evidence, start=1):
            snippet = result.text.strip().replace("\n", " ")
            lines.append(f"[C{index}] (id={result.node_id}) {snippet[:300]}")
    else:
        lines.append("EVIDENCE SO FAR: (none yet)")
    lines.append("")
    lines.append(f"Step {obs.step_index + 1} of {obs.max_steps}. Tools: {', '.join(obs.tool_names)}.")
    if obs.must_answer:
        lines.append("This is the LAST step — you MUST answer now using the evidence above.")
    return "\n".join(lines)


def parse_action(text: str, *, must_answer: bool = False) -> AgentAction:
    """Parse the LLM's JSON action, degrading gracefully to a sensible default."""
    payload = _extract_json(text)
    if payload is None:
        # Unparseable — treat the raw text as an answer if we must, else search.
        if must_answer:
            return AgentAction(kind="answer", answer=text.strip(), thought="unparsed")
        return AgentAction(kind="search", tool=None, query=None, thought="unparsed")

    action = str(payload.get("action", "")).lower()
    thought = str(payload.get("thought", ""))
    if action == "answer" or must_answer:
        return AgentAction(kind="answer", answer=str(payload.get("answer", "")).strip(), thought=thought)
    return AgentAction(
        kind="search",
        tool=payload.get("tool"),
        query=payload.get("query"),
        thought=thought,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
