from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ragbench.core.measure import canonical_fingerprint
from ragbench.core.schema import EvalItem, RAGAnswer
from ragbench.providers.llm_client import LLMClient


@dataclass(slots=True)
class JudgeResult:
    status: str
    answer_correctness: float
    faithfulness: float
    citation_support: float
    abstention_correctness: float
    notes: list[str]
    usage: dict[str, int]
    latency_ms: float
    estimated_cost: float
    model: str
    temperature: float
    prompt_fingerprint: str
    # Per-sub-judge status — "ok" only if *both* need to be "ok" for the
    # blanket ``status`` above (see LLMJudge.judge's docstring); kept
    # separately so a failure in one half doesn't obscure which one, even
    # though today's mean-metric aggregation (evaluation.metrics) still keys
    # off the blanket status for both judges' fields together.
    correctness_status: str = "ok"
    faithfulness_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer_correctness": self.answer_correctness,
            "faithfulness": self.faithfulness,
            "citation_support": self.citation_support,
            "abstention_correctness": self.abstention_correctness,
            "notes": self.notes,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
            "model": self.model,
            "temperature": self.temperature,
            "prompt_fingerprint": self.prompt_fingerprint,
            "correctness_status": self.correctness_status,
            "faithfulness_status": self.faithfulness_status,
        }


_CORRECTNESS_SYSTEM_PROMPT = (
    "You are a strict RAG evaluation judge assessing answer correctness. Return only JSON with numeric keys "
    "answer_correctness, abstention_correctness in [0,1], plus notes as an array of short strings. "
    "abstention_correctness scores whether the system correctly answered when it should have, or correctly "
    "abstained when the ground truth indicates the question is unanswerable."
)
_FAITHFULNESS_SYSTEM_PROMPT = (
    "You are a strict RAG evaluation judge assessing groundedness. You are NOT given the correct answer — judge "
    "only whether the prediction is supported by the provided evidence. Return only JSON with numeric keys "
    "faithfulness, citation_support in [0,1], plus notes as an array of short strings. faithfulness scores whether "
    "every claim in the prediction is supported by the evidence (not by outside knowledge). citation_support "
    "scores whether the cited spans actually support the claims they're attached to."
)

_CORRECTNESS_FIELDS = ("answer_correctness", "abstention_correctness")
_FAITHFULNESS_FIELDS = ("faithfulness", "citation_support")


class LLMJudge:
    """Two independent judge calls instead of one prompt scoring all four fields.

    The original single-prompt design showed a faithfulness/citation-support
    judge the ground truth answer and the expected doc ids — evaluator
    coupling: a judge that is only supposed to check "is this grounded in the
    retrieved evidence" could instead reward/punish based on matching the
    reference answer, or leak expected-doc identity into a support judgment
    it should reach independently.

    ``CorrectnessJudge`` (question + ground truth + prediction) and
    ``FaithfulnessJudge`` (prediction + cited evidence only, no ground truth,
    no expected_doc_ids) are separate calls so the second one is structurally
    unable to see what the coupling bug depended on.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 450,
        **_: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = LLMClient()

    def judge(self, item: EvalItem, prediction: RAGAnswer) -> JudgeResult:
        correctness_messages = self._correctness_messages(item, prediction)
        correctness_payload, correctness_completion = self._call(correctness_messages)
        correctness_status, correctness_notes = _judge_status(correctness_payload, _CORRECTNESS_FIELDS)

        faithfulness_messages = self._faithfulness_messages(prediction)
        faithfulness_payload, faithfulness_completion = self._call(faithfulness_messages)
        faithfulness_status, faithfulness_notes = _judge_status(faithfulness_payload, _FAITHFULNESS_FIELDS)

        overall_status = correctness_status if correctness_status != "ok" else faithfulness_status
        usage = {
            key: correctness_completion.usage.get(key, 0) + faithfulness_completion.usage.get(key, 0)
            for key in {*correctness_completion.usage, *faithfulness_completion.usage}
        }
        return JudgeResult(
            status=overall_status,
            answer_correctness=_score(correctness_payload.get("answer_correctness")),
            faithfulness=_score(faithfulness_payload.get("faithfulness")),
            citation_support=_score(faithfulness_payload.get("citation_support")),
            abstention_correctness=_score(correctness_payload.get("abstention_correctness")),
            notes=[*correctness_notes, *faithfulness_notes],
            usage=usage,
            latency_ms=round(correctness_completion.latency_ms + faithfulness_completion.latency_ms, 3),
            estimated_cost=round(correctness_completion.estimated_cost + faithfulness_completion.estimated_cost, 8),
            model=self.model,
            temperature=self.temperature,
            prompt_fingerprint=canonical_fingerprint(
                {
                    "version": "ragbench_llm_judge_v3_split",
                    "correctness_system_prompt": correctness_messages[0]["content"],
                    "correctness_user_template_fields": ["question", "ground_truth_answer", "predicted_answer"],
                    "faithfulness_system_prompt": faithfulness_messages[0]["content"],
                    "faithfulness_user_template_fields": [
                        "predicted_answer",
                        "predicted_citations",
                        "retrieved_context",
                    ],
                    "max_tokens": self.max_tokens,
                }
            ),
            correctness_status=correctness_status,
            faithfulness_status=faithfulness_status,
        )

    def _call(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], Any]:
        completion = self.client.create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return parse_json_object(completion.text), completion

    def _correctness_messages(self, item: EvalItem, prediction: RAGAnswer) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _CORRECTNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{item.question}\n\n"
                    f"GROUND_TRUTH:\n{item.ground_truth_answer or ''}\n\n"
                    f"PREDICTED_ANSWER:\n{prediction.answer}"
                ),
            },
        ]

    def _faithfulness_messages(self, prediction: RAGAnswer) -> list[dict[str, str]]:
        context = "\n\n".join(context.text for context in prediction.contexts)
        return [
            {"role": "system", "content": _FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PREDICTED_ANSWER:\n{prediction.answer}\n\n"
                    f"PREDICTED_CITATIONS:\n"
                    f"{json.dumps([c.doc_id for c in prediction.citations], ensure_ascii=False)}\n\n"
                    f"RETRIEVED_CONTEXT (the only evidence you may use):\n{context}"
                ),
            },
        ]


def create_judge(spec: dict[str, Any] | None) -> LLMJudge | None:
    if not spec:
        return None
    if spec.get("type", "openai") not in {"openai", "openai_judge"}:
        raise KeyError("evaluation.judge.type must be openai")
    return LLMJudge(**dict(spec.get("params", {})))


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):  # bool passes float() but is never a legitimate judge score
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _judge_status(payload: dict[str, Any], required_fields: tuple[str, ...]) -> tuple[str, list[str]]:
    """Classify a parsed judge payload — never silently downgrade a schema
    violation into a default 0.0 score reported as ``status="ok"``.

    Valid JSON that is merely missing (or has a non-numeric) required field
    used to fall through to ``_score(None) == 0.0`` while still being counted
    as a successful judgment — a provider that drifts off-schema would then
    quietly drag every mean judge metric toward 0 instead of surfacing as a
    failure rate. ``schema_failure`` is deliberately distinct from
    ``parse_failure`` (invalid JSON) so the two causes stay distinguishable.
    """
    if "parse_error" in payload:
        return "parse_failure", []
    missing = [field for field in required_fields if not _is_numeric(payload.get(field))]
    if missing:
        return "schema_failure", [f"judge JSON is missing or has a non-numeric value for: {', '.join(missing)}"]
    return "ok", []


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"parse_error": True, "notes": ["judge did not return parseable JSON"]}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {"parse_error": True, "notes": ["judge returned malformed JSON"]}


def _score(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except (TypeError, ValueError):
        return 0.0
