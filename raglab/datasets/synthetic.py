from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from raglab.core.io import write_jsonl
from raglab.core.text import token_count
from raglab.providers.llm_client import LLMClient

QUESTION_TYPES = ["factual", "citation_sensitive", "multi_section", "unanswerable"]


class SyntheticQAGenerator:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        questions_per_chunk: int = 2,
        max_context_tokens: int = 700,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> None:
        self.model = model
        self.questions_per_chunk = questions_per_chunk
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = LLMClient()

    def generate(self, docs_path: str | Path, output_path: str | Path, limit: int = 50) -> list[dict[str, Any]]:
        chunks = _load_text_chunks(docs_path, self.max_context_tokens)
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if len(rows) >= limit:
                break
            completion = self.client.create_chat_completion(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate Vietnamese RAG evaluation questions from the context. Return only a JSON array. "
                            "Each item must contain question, ground_truth_answer, question_type, difficulty. "
                            "Use question_type values from: factual, citation_sensitive, multi_section, unanswerable. "
                            "For unanswerable questions, the ground_truth_answer must say the context has "
                            "insufficient evidence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"DOC_ID: {chunk['doc_id']}\n"
                            f"CITATION: {chunk['citation']}\n"
                            f"NUMBER_OF_QUESTIONS: {self.questions_per_chunk}\n\n"
                            f"CONTEXT:\n{chunk['text']}"
                        ),
                    },
                ],
            )
            for item in parse_json_array(completion.text):
                rows.append(_row(len(rows) + 1, chunk, item))
                if len(rows) >= limit:
                    break
        write_jsonl(output_path, rows)
        return rows


def parse_json_array(text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _load_text_chunks(docs_path: str | Path, max_tokens: int) -> list[dict[str, str]]:
    source = Path(docs_path)
    files = [source] if source.is_file() else sorted(source.rglob("*"))
    chunks: list[dict[str, str]] = []
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            continue
        doc_id = path.stem
        parts = [part.strip() for part in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if part.strip()]
        buffer: list[str] = []
        for part in parts:
            candidate = "\n\n".join([*buffer, part])
            if buffer and token_count(candidate) > max_tokens:
                chunks.append(_chunk(doc_id, len(chunks) + 1, "\n\n".join(buffer)))
                buffer = [part]
            else:
                buffer.append(part)
        if buffer:
            chunks.append(_chunk(doc_id, len(chunks) + 1, "\n\n".join(buffer)))
    return chunks


def _chunk(doc_id: str, index: int, text: str) -> dict[str, str]:
    chunk_id = f"{doc_id}:synthetic:{index:04d}"
    return {"doc_id": doc_id, "chunk_id": chunk_id, "citation": chunk_id, "text": text}


def _row(index: int, chunk: dict[str, str], item: dict[str, Any]) -> dict[str, Any]:
    question_type = str(item.get("question_type", "factual"))
    if question_type not in QUESTION_TYPES:
        question_type = "factual"
    return {
        "question_id": f"syn_{index:04d}",
        "question": str(item.get("question", "")).strip(),
        "ground_truth_answer": str(item.get("ground_truth_answer", "")).strip(),
        "expected_doc_ids": [] if question_type == "unanswerable" else [chunk["doc_id"]],
        "expected_chunk_ids": [] if question_type == "unanswerable" else [chunk["chunk_id"]],
        "expected_citations": [] if question_type == "unanswerable" else [chunk["citation"]],
        "metadata": {
            "question_type": question_type,
            "difficulty": str(item.get("difficulty", "medium")),
            "source_chunk_id": chunk["chunk_id"],
            "generated": True,
        },
    }
