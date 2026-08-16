from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import EvalItem, RAGAnswer
from raglab.providers.llm_client import LLMClient


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
        }


class LLMJudge:
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
        context = "\n\n".join(context.text for context in prediction.contexts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluation judge. Return only JSON with numeric keys "
                    "answer_correctness, faithfulness, citation_support, abstention_correctness in [0,1], "
                    "plus notes as an array of short strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{item.question}\n\n"
                    f"GROUND_TRUTH:\n{item.ground_truth_answer or ''}\n\n"
                    f"EXPECTED_DOC_IDS:\n{json.dumps(item.expected_doc_ids, ensure_ascii=False)}\n\n"
                    f"PREDICTED_ANSWER:\n{prediction.answer}\n\n"
                    f"PREDICTED_CITATIONS:\n{json.dumps(prediction.citations, ensure_ascii=False)}\n\n"
                    f"RETRIEVED_CONTEXT:\n{context}"
                ),
            },
        ]
        completion = self.client.create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        payload = parse_json_object(completion.text)
        status = "ok" if "parse_error" not in payload else "parse_failure"
        return JudgeResult(
            status=status,
            answer_correctness=_score(payload.get("answer_correctness")),
            faithfulness=_score(payload.get("faithfulness")),
            citation_support=_score(payload.get("citation_support")),
            abstention_correctness=_score(payload.get("abstention_correctness")),
            notes=[str(note) for note in payload.get("notes", [])],
            usage=completion.usage,
            latency_ms=completion.latency_ms,
            estimated_cost=completion.estimated_cost,
            model=self.model,
            temperature=self.temperature,
            prompt_fingerprint=canonical_fingerprint(
                {
                    "version": "raglab_llm_judge_v1",
                    "system_prompt": messages[0]["content"],
                    "user_template_fields": [
                        "question",
                        "ground_truth_answer",
                        "expected_doc_ids",
                        "predicted_answer",
                        "predicted_citations",
                        "retrieved_context",
                    ],
                    "max_tokens": self.max_tokens,
                }
            ),
        )


def create_judge(spec: dict[str, Any] | None) -> LLMJudge | None:
    if not spec:
        return None
    if spec.get("type", "openai") not in {"openai", "openai_judge"}:
        raise KeyError("evaluation.judge.type must be openai")
    return LLMJudge(**dict(spec.get("params", {})))


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
