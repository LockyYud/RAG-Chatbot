from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from raglab.core.text import token_count
from raglab.providers.env import env_float, env_str, load_dotenv


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    usage: dict[str, int]
    latency_ms: float
    estimated_cost: float


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        load_dotenv()
        self.api_key = env_str(api_key_env)
        self.base_url = (base_url or env_str(base_url_env, "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    def create_embeddings(self, model: str, inputs: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            payload = {"model": model, "input": batch}
            response = self._post("/embeddings", payload)
            vectors.extend(item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"]))
        return vectors

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> ChatCompletionResult:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._post("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        text = response["choices"][0]["message"]["content"]
        usage = _usage(response.get("usage", {}), messages, text)
        return ChatCompletionResult(
            text=text.strip(),
            usage=usage,
            latency_ms=round(latency_ms, 3),
            estimated_cost=_estimate_chat_cost(usage),
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API request failed: {exc.code} {body}") from exc


def _usage(raw_usage: dict[str, Any], messages: list[dict[str, str]], output: str) -> dict[str, int]:
    prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
    if prompt_tokens == 0:
        prompt_tokens = sum(token_count(message.get("content", "")) for message in messages)
    if completion_tokens == 0:
        completion_tokens = token_count(output)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(raw_usage.get("total_tokens") or prompt_tokens + completion_tokens),
    }


def _estimate_chat_cost(usage: dict[str, int]) -> float:
    input_cost = env_float("OPENAI_CHAT_INPUT_COST_PER_1K", 0.0)
    output_cost = env_float("OPENAI_CHAT_OUTPUT_COST_PER_1K", 0.0)
    return round(
        usage["prompt_tokens"] / 1000 * input_cost + usage["completion_tokens"] / 1000 * output_cost,
        8,
    )
