"""Synthetic benchmark generation, v2 — see ``docs/synthetic_benchmark_v2.md``.

Turns a folder of raw text/Markdown into an evaluation set that can score
techniques without anyone hand-labelling qrels. The design premise is *not*
that a good enough generator produces a trustworthy benchmark; it is that every
stage must emit a number, so the resulting benchmark can say how far it should
be trusted (see :mod:`ragbench.evaluation.trust`).

Three properties matter more than the prompt wording:

**Labels are document-level, deliberately.** v1 wrote its own chunk ids
(``doc:synthetic:0001``) into ``expected_chunk_ids`` while pipelines emit
``doc:c1:<sha1>`` (``processing.chunkers.common.chunk_id``). The two families
can never intersect, and because
:func:`ragbench.evaluation.metrics.evaluate_prediction_rows` prefers chunk
labels whenever they are present, *every* retrieval metric silently evaluated
to 0 for *every* technique while still reporting ``retrieval_evaluated: True``.
Emitting real pipeline chunk ids would not fix it either: chunk ids depend on
the chunker, so a label built with one technique's chunker is meaningless for a
technique that chunks differently (``parent_child`` vs ``naive_rag``). Document
ids are the only identifier that survives across techniques, so that is what a
label is. The source chunk is kept in metadata as provenance, never as a label.

**Relevance is pooled, not assumed to be the source chunk alone.** A question
written from one passage is very often answerable from other passages too, and
a single-label benchmark scores the retriever that found an equally valid
passage as a miss — which penalises *better* retrievers hardest. Stage 5 pools
candidates from BM25 (plus dense retrieval when an embedding model is
configured) and asks the verifier to grade them, so the label set reflects the
corpus rather than the accident of which chunk seeded the question.

**Roles are separated.** The model that writes questions must not be the model
that checks them, and neither should be the judge that later scores answers.
:func:`role_collisions` is what the CLI enforces this with.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ragbench.core.io import iter_input_files, read_json, read_jsonl, write_json, write_jsonl
from ragbench.core.measure import canonical_fingerprint
from ragbench.core.schema import Chunk, IndexedNode
from ragbench.core.text import cosine, term_vector, tokenize
from ragbench.processing.chunkers.fixed_size import FixedSizeChunker
from ragbench.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from ragbench.processing.parsers.text_parser import TextParser

GENERATOR_VERSION = "v2"

QUESTION_TYPES = ("factual", "citation_sensitive", "multi_section", "unanswerable")

#: Every model-shaped role in the protocol, and the env var that names it.
#: Kept here (not in the provider layer) because the *separation* of these
#: roles is a benchmark-validity rule, not a provider concern.
ROLE_ENV_VARS = {
    "generator": "RAGBENCH_GENERATOR_MODEL",
    "verifier": "RAGBENCH_VERIFIER_MODEL",
    "judge": "RAGBENCH_JUDGE_MODEL",
}


# ─── Roles ───────────────────────────────────────────────────────────────────


def resolve_role_model(role: str, override: str | None = None) -> str:
    """Resolve the model for *role*: explicit override, else env, else the default chat model."""
    if role not in ROLE_ENV_VARS:
        raise ValueError(f"Unknown model role {role!r}; expected one of {', '.join(sorted(ROLE_ENV_VARS))}")
    if override:
        return override
    from ragbench.providers.env import load_dotenv
    from ragbench.providers.llm_client import default_chat_model

    load_dotenv()
    configured = os.getenv(ROLE_ENV_VARS[role])
    return configured if configured else default_chat_model()


def role_collisions(models: dict[str, str]) -> list[str]:
    """Return ``"a==b"`` labels for every pair of roles sharing one model.

    A shared model is never a hard error here — the caller decides whether to
    refuse or to record it and downgrade the trust verdict — but it is always
    reported, because a generator that also verifies is grading its own work
    and a judge that also generated the questions inflates every answer score.
    """
    collisions: list[str] = []
    names = sorted(models)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if models[left] and models[left] == models[right]:
                collisions.append(f"{left}=={right}")
    return collisions


# ─── Parsing helpers ─────────────────────────────────────────────────────────


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


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


# ─── Text measures ───────────────────────────────────────────────────────────


def token_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether *needle* appears as a contiguous token run inside *haystack*."""
    length = len(needle)
    if length == 0 or length > len(haystack):
        return False
    for start in range(len(haystack) - length + 1):
        if haystack[start : start + length] == needle:
            return True
    return False


def span_in_text(span: str, text: str) -> bool:
    """Whether *span* was really copied out of *text*.

    Compared on the token stream rather than the raw string: chunk text has
    already been through ``split_tokens`` (lowercased, punctuation dropped,
    re-joined on single spaces), so a substring test against the original
    document wording would reject spans that were in fact copied verbatim.
    """
    return token_subsequence(tokenize(span), tokenize(text))


def lexical_overlap(question: str, text: str) -> float:
    """Fraction of the question's tokens that also occur in *text*.

    Containment, not Jaccard: the question is short and the chunk is long, so
    Jaccard would mostly measure chunk length. What matters for bias is how
    much of the question was *lifted from* the passage, which is exactly the
    containment direction. 1.0 means every word of the question appears in the
    passage it was written from, i.e. the retrieval task is close to string
    matching and lexical retrievers are being handed the answer.
    """
    question_tokens = set(tokenize(question))
    if not question_tokens:
        return 0.0
    return len(question_tokens & set(tokenize(text))) / len(question_tokens)


def rouge_l_recall(reference: str, candidate: str) -> float:
    """Longest-common-subsequence recall of *reference* covered by *candidate*."""
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    previous = [0] * (len(candidate_tokens) + 1)
    for ref_token in reference_tokens:
        current = [0]
        for index, cand_token in enumerate(candidate_tokens):
            if ref_token == cand_token:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1] / len(reference_tokens)


# ─── Config and counters ─────────────────────────────────────────────────────


@dataclass(slots=True)
class GenerationConfig:
    generator_model: str
    verifier_model: str
    questions_per_chunk: int = 2
    max_chunks: int | None = None
    seed: int = 42
    chunk_size: int = 250
    chunk_overlap: int = 40
    overlap_threshold: float = 0.6
    dedup_threshold: float = 0.92
    pool_k: int = 20
    pool_embedding_model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 900
    allow_same_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "questions_per_chunk": self.questions_per_chunk,
            "max_chunks": self.max_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "overlap_threshold": self.overlap_threshold,
            "dedup_threshold": self.dedup_threshold,
            "pool_k": self.pool_k,
            "temperature": self.temperature,
        }


@dataclass
class StageCounters:
    generated: int = 0
    span_reject: int = 0
    answerability_reject: int = 0
    rewrite: int = 0
    rewrite_reject: int = 0
    pool_calls: int = 0
    pool_size_total: int = 0
    extra_relevant_total: int = 0
    relabelled_from_unanswerable: int = 0
    dedup_removed: int = 0
    overlaps: list[float] = field(default_factory=list)

    def to_dict(self, final_rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable_after_generation = max(self.generated - self.span_reject, 1)
        return {
            "generated": self.generated,
            "span_reject": self.span_reject,
            "answerability_reject": self.answerability_reject,
            "answerability_reject_rate": round(self.answerability_reject / answerable_after_generation, 6),
            "rewrite": self.rewrite,
            "rewrite_reject": self.rewrite_reject,
            "lexical_overlap_p50": _percentile(self.overlaps, 50),
            "lexical_overlap_p90": _percentile(self.overlaps, 90),
            "pool_size_mean": round(self.pool_size_total / self.pool_calls, 4) if self.pool_calls else 0.0,
            "extra_relevant_mean": round(self.extra_relevant_total / self.pool_calls, 4) if self.pool_calls else 0.0,
            "relabelled_from_unanswerable": self.relabelled_from_unanswerable,
            "dedup_removed": self.dedup_removed,
            "final": len(final_rows),
            "final_unanswerable": sum(1 for row in final_rows if not row["metadata"]["is_answerable"]),
        }


class ChatClient(Protocol):
    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> Any: ...


# ─── Corpus chunking ─────────────────────────────────────────────────────────


def chunk_corpus(docs_path: str | Path, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk a raw corpus with the same parser/cleaner/chunker a pipeline uses.

    Generation must see the passages retrieval will actually return, otherwise
    a question can be written from text no chunk ever contains.
    """
    parser = TextParser()
    blocks = []
    for path in iter_input_files(docs_path):
        blocks.extend(parser.parse(str(path), root=docs_path))
    blocks = VietnameseNormalizer().clean(blocks)
    blocks = WhitespaceCleaner().clean(blocks)
    return FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap).chunk(blocks)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    return canonical_fingerprint(
        sorted(({"doc_id": chunk.doc_id, "text": chunk.text} for chunk in chunks), key=lambda row: row["text"])
    )


# ─── Builder ─────────────────────────────────────────────────────────────────


class SyntheticBenchmarkBuilder:
    """Run the seven-stage generation pipeline over one corpus."""

    def __init__(self, config: GenerationConfig, client: ChatClient | None = None) -> None:
        self.config = config
        if client is None:
            from ragbench.providers.llm_client import LLMClient

            client = LLMClient()
        self.client = client
        self.counters = StageCounters()

    # -- stage 2 ----------------------------------------------------------

    def generate_for_chunk(self, chunk: Chunk) -> list[dict[str, Any]]:
        completion = self.client.create_chat_completion(
            model=self.config.generator_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate Vietnamese RAG evaluation questions from the context. Return only a JSON array. "
                        "Each item must contain question, answer_span, ground_truth_answer, question_type, "
                        "difficulty. answer_span MUST be copied verbatim from the context and must contain the "
                        "answer. Questions must stand alone: never write 'theo đoạn trên', 'trong tài liệu này' "
                        "or any reference to the context itself, because the reader will not see it. "
                        f"question_type is one of: {', '.join(QUESTION_TYPES)}. "
                        "For unanswerable questions omit answer_span and say the context lacks the evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DOC_ID: {chunk.doc_id}\n"
                        f"NUMBER_OF_QUESTIONS: {self.config.questions_per_chunk}\n\n"
                        f"CONTEXT:\n{chunk.text}"
                    ),
                },
            ],
        )
        rows: list[dict[str, Any]] = []
        for item in parse_json_array(completion.text):
            self.counters.generated += 1
            question = str(item.get("question", "")).strip()
            if not question:
                self.counters.span_reject += 1
                continue
            question_type = str(item.get("question_type", "factual"))
            if question_type not in QUESTION_TYPES:
                question_type = "factual"
            answer_span = str(item.get("answer_span", "")).strip()
            if question_type != "unanswerable":
                # The cheapest possible guard against a weak generator inventing
                # an answer: the span it claims to have copied has to actually
                # be in the passage. No API call, no judgement involved.
                if not answer_span or not span_in_text(answer_span, chunk.text):
                    self.counters.span_reject += 1
                    continue
            rows.append(
                {
                    "question": question,
                    "answer_span": answer_span,
                    "ground_truth_answer": str(item.get("ground_truth_answer", "")).strip(),
                    "question_type": question_type,
                    "difficulty": str(item.get("difficulty", "medium")),
                    "source_chunk_id": chunk.chunk_id,
                    "source_doc_id": chunk.doc_id,
                }
            )
        return rows

    # -- stage 3 ----------------------------------------------------------

    def verify_answerable(self, question: str, chunk_text: str, answer_span: str) -> bool:
        """Ask the verifier to answer from the passage alone, then check it landed on the span."""
        completion = self.client.create_chat_completion(
            model=self.config.verifier_model,
            temperature=0.0,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using ONLY the context. Return one JSON object with keys "
                        "answerable (boolean) and answer (string). Set answerable to false when the context "
                        "does not contain the answer."
                    ),
                },
                {"role": "user", "content": f"CONTEXT:\n{chunk_text}\n\nQUESTION: {question}"},
            ],
        )
        payload = parse_json_object(completion.text)
        if not bool(payload.get("answerable", False)):
            return False
        # Both signals must agree: a verifier that says "answerable" but writes
        # something unrelated to the span has not confirmed the label, it has
        # produced a second guess.
        return rouge_l_recall(answer_span, str(payload.get("answer", ""))) >= 0.5

    # -- stage 4 ----------------------------------------------------------

    def rewrite_question(self, question: str, chunk_text: str) -> str:
        completion = self.client.create_chat_completion(
            model=self.config.generator_model,
            temperature=0.3,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the question so it asks for the same fact without reusing the source wording. "
                        "Use synonyms and a different sentence shape. Keep it standalone and answerable. "
                        "Return one JSON object with a single key: question."
                    ),
                },
                {"role": "user", "content": f"SOURCE PASSAGE:\n{chunk_text}\n\nQUESTION: {question}"},
            ],
        )
        rewritten = str(parse_json_object(completion.text).get("question", "")).strip()
        return rewritten or question

    # -- stage 5 ----------------------------------------------------------

    def pool_relevance(self, question: str, candidates: list[IndexedNode], source_doc_id: str | None) -> dict[str, int]:
        """Grade pooled candidates, returning ``{doc_id: grade}`` for grade > 0."""
        if not candidates:
            # A corpus with nothing else to pool still has the passage the
            # question was written from; dropping it here would emit an
            # answerable question with no label at all.
            return {source_doc_id: 2} if source_doc_id else {}
        listing = "\n\n".join(
            f"[{index}] doc_id={node.doc_id}\n{node.text_for_embedding}" for index, node in enumerate(candidates)
        )
        completion = self.client.create_chat_completion(
            model=self.config.verifier_model,
            temperature=0.0,
            max_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Grade how well each passage answers the question. Return only a JSON array of objects "
                        "with keys index (integer) and grade (2 = fully answers it, 1 = partially, 0 = does not). "
                        "Include every passage you grade 1 or 2; omit the rest."
                    ),
                },
                {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n{listing}"},
            ],
        )
        grades: dict[str, int] = {}
        for item in parse_json_array(completion.text):
            try:
                index = int(item.get("index", -1))
                grade = int(item.get("grade", 0))
            except (TypeError, ValueError):
                continue
            if grade <= 0 or not 0 <= index < len(candidates):
                continue
            doc_id = candidates[index].doc_id
            grades[doc_id] = max(grades.get(doc_id, 0), min(grade, 2))
        if source_doc_id is not None:
            grades[source_doc_id] = 2
        return grades

    # -- orchestration ----------------------------------------------------

    def build(self, docs_path: str | Path, output_dir: str | Path, *, dataset_id: str | None = None) -> dict[str, Any]:
        config = self.config
        collisions = role_collisions({"generator": config.generator_model, "verifier": config.verifier_model})
        if collisions and not config.allow_same_model:
            raise ValueError(
                f"Model roles must differ: {', '.join(collisions)}. The generator would be grading its own "
                f"questions. Set {ROLE_ENV_VARS['verifier']} to a different model, or pass --allow-same-model "
                "to proceed with a recorded, trust-downgrading exception."
            )

        chunks = chunk_corpus(docs_path, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        if not chunks:
            raise ValueError(f"No chunks produced from {docs_path}")
        selected = chunks if config.max_chunks is None else chunks[: config.max_chunks]
        nodes = _nodes_from_chunks(chunks)
        pooler = _CandidatePool(nodes, pool_k=config.pool_k, embedding_model=config.pool_embedding_model)
        chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}

        rows: list[dict[str, Any]] = []
        for chunk in selected:
            for draft in self.generate_for_chunk(chunk):
                row = self._finish_row(draft, chunk_text_by_id, pooler)
                if row is not None:
                    rows.append(row)

        rows = self._dedup(rows)
        for index, row in enumerate(rows, start=1):
            row["question_id"] = f"syn_{index:04d}"

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        write_jsonl(target / "qa.jsonl", rows)
        manifest = {
            "dataset_id": dataset_id or target.name,
            "generator_version": GENERATOR_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "corpus": {
                "path": str(docs_path),
                "fingerprint": corpus_fingerprint(chunks),
                "documents": len({chunk.doc_id for chunk in chunks}),
                "chunks": len(chunks),
                "chunks_used": len(selected),
                "chunker_id": "fixed_size",
                "chunk_config": {"chunk_size": config.chunk_size, "overlap": config.chunk_overlap},
            },
            "models": {
                "generator": config.generator_model,
                "verifier": config.verifier_model,
                "pool_embedding": config.pool_embedding_model,
            },
            "same_model_roles": collisions,
            "pool_retrievers": pooler.retriever_ids,
            "params": config.to_dict(),
            "stages": self.counters.to_dict(rows),
            "qa_path": str(target / "qa.jsonl"),
            "metadata": {
                # A synthetic set is a tuning set by construction: its labels came
                # from a model, so no empirical claim can rest on it. Not a flag —
                # there is deliberately no way to write "test" here.
                "protocol_split": "dev",
                "generated": True,
            },
        }
        write_json(target / "manifest.json", manifest)
        return manifest

    def _finish_row(
        self, draft: dict[str, Any], chunk_text_by_id: dict[str, str], pooler: _CandidatePool
    ) -> dict[str, Any] | None:
        config = self.config
        chunk_text = chunk_text_by_id[draft["source_chunk_id"]]
        question = draft["question"]
        answerable = draft["question_type"] != "unanswerable"

        if answerable and not self.verify_answerable(question, chunk_text, draft["answer_span"]):
            self.counters.answerability_reject += 1
            return None

        overlap = lexical_overlap(question, chunk_text)
        rewritten = False
        if answerable and overlap > config.overlap_threshold:
            candidate = self.rewrite_question(question, chunk_text)
            self.counters.rewrite += 1
            if candidate != question and self.verify_answerable(candidate, chunk_text, draft["answer_span"]):
                question = candidate
                overlap = lexical_overlap(question, chunk_text)
                rewritten = True
            else:
                # Keep the original rather than ship a paraphrase that no longer
                # asks the same thing; the flag is what tells the reader the bias
                # is still there.
                self.counters.rewrite_reject += 1
        self.counters.overlaps.append(overlap)

        candidates = pooler.candidates(question, exclude_chunk_id=draft["source_chunk_id"])
        self.counters.pool_calls += 1
        self.counters.pool_size_total += len(candidates)
        grades = self.pool_relevance(question, candidates, draft["source_doc_id"] if answerable else None)
        relabelled = False
        if not answerable and grades:
            # The generator only ever saw one passage, so "unanswerable" meant
            # "unanswerable from that passage". The pool just proved the corpus
            # answers it, which makes the original label wrong, not the question.
            answerable = True
            relabelled = True
            self.counters.relabelled_from_unanswerable += 1
        if answerable:
            self.counters.extra_relevant_total += max(len(grades) - 1, 0)

        expected_doc_ids = sorted(grades) if answerable else []
        return {
            "question_id": "",
            "question": question,
            "ground_truth_answer": draft["ground_truth_answer"],
            "expected_doc_ids": expected_doc_ids,
            # Deliberately empty — see the module docstring. Chunk ids are
            # chunker-specific, so they cannot label a benchmark that several
            # differently-chunking techniques all run against.
            "expected_chunk_ids": [],
            "expected_citations": [draft["source_doc_id"]] if answerable else [],
            "metadata": {
                "question_type": draft["question_type"],
                "difficulty": draft["difficulty"],
                "is_answerable": answerable,
                "generated": True,
                "generator_version": GENERATOR_VERSION,
                "source_chunk_id": draft["source_chunk_id"],
                "answer_span": draft["answer_span"],
                "lexical_overlap": round(overlap, 6),
                "high_overlap": overlap > config.overlap_threshold,
                "rewritten": rewritten,
                "relevance_by_doc_id": {doc_id: grades[doc_id] for doc_id in expected_doc_ids},
                "pool_candidates": len(candidates),
                "relabelled_from_unanswerable": relabelled,
            },
        }

    def _dedup(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop near-identical questions, keeping the one that leaks less wording.

        Lexical rather than embedding cosine: two questions this similar share
        almost all their tokens, so the cheap measure separates them just as
        well and keeps generation free of an embedding provider.
        """
        kept: list[dict[str, Any]] = []
        vectors: list[dict[str, float]] = []
        for row in sorted(rows, key=lambda item: item["metadata"]["lexical_overlap"]):
            vector = term_vector(row["question"])
            if any(cosine(vector, seen) >= self.config.dedup_threshold for seen in vectors):
                self.counters.dedup_removed += 1
                continue
            kept.append(row)
            vectors.append(vector)
        return kept


class _CandidatePool:
    """Pools retrieval candidates so relevance labels are not single-passage."""

    def __init__(self, nodes: list[IndexedNode], *, pool_k: int, embedding_model: str | None) -> None:
        from ragbench.indexing.retrievers import BM25Retriever

        self.nodes = nodes
        self.pool_k = pool_k
        self.bm25 = BM25Retriever(nodes)
        self.retriever_ids = ["bm25"]
        self.dense = None
        if embedding_model:
            from ragbench.indexing.embeddings import Embedder
            from ragbench.indexing.retrievers import DenseRetriever

            embedded = Embedder(model=embedding_model).embed_nodes(nodes)
            self.dense = DenseRetriever(nodes=embedded, embedding_model=embedding_model)
            self.retriever_ids.append("dense")

    def candidates(self, question: str, *, exclude_chunk_id: str | None) -> list[IndexedNode]:
        by_id = {node.chunk_id: node for node in self.nodes}
        ordered: list[IndexedNode] = []
        seen: set[str] = set()
        result_lists = [self.bm25.retrieve(question, self.pool_k)]
        if self.dense is not None:
            result_lists.append(self.dense.retrieve(question, self.pool_k))
        for results in result_lists:
            for result in results:
                if result.chunk_id == exclude_chunk_id or result.chunk_id in seen:
                    continue
                node = by_id.get(result.chunk_id)
                if node is not None:
                    seen.add(result.chunk_id)
                    ordered.append(node)
        return ordered


def _nodes_from_chunks(chunks: list[Chunk]) -> list[IndexedNode]:
    return [
        IndexedNode(
            node_id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text_for_embedding=chunk.text,
            text_for_generation=chunk.text,
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return round(ordered[lower], 6)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)


# ─── Validation ──────────────────────────────────────────────────────────────


def load_synthetic_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    """Return the generation manifest for *dataset_path*, or ``None`` if not synthetic.

    Accepts either the dataset directory or its ``qa.jsonl``, so callers can
    pass whatever the user typed on the command line.
    """
    source = Path(dataset_path)
    directory = source.parent if source.is_file() else source
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return None
    metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
    if isinstance(metadata, dict) and metadata.get("generated") is True:
        return manifest
    return None


def validate_synthetic_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Check a generated set for the failure modes v1 shipped with."""
    source = Path(dataset_dir)
    qa_path = source / "qa.jsonl" if source.is_dir() else source
    if not qa_path.exists():
        raise ValueError(f"Synthetic dataset is missing qa.jsonl: {source}")
    manifest = load_synthetic_manifest(qa_path)
    if manifest is None:
        raise ValueError(f"{source} has no manifest.json declaring metadata.generated = true")
    rows = read_jsonl(qa_path)
    if not rows:
        raise ValueError("qa.jsonl is empty")
    errors: list[str] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id", ""))
        if not question_id:
            errors.append(f"row {index}: missing question_id")
        elif question_id in ids:
            errors.append(f"row {index}: duplicate question_id {question_id!r}")
        ids.add(question_id)
        metadata = row.get("metadata", {})
        answerable = bool(metadata.get("is_answerable", True))
        if row.get("expected_chunk_ids"):
            # The exact v1 defect: chunk labels that no pipeline can ever match
            # make every retrieval metric silently zero.
            errors.append(f"row {index}: expected_chunk_ids must be empty; chunk ids are chunker-specific")
        if answerable and not row.get("expected_doc_ids"):
            errors.append(f"row {index}: answerable question has no expected_doc_ids")
        if not answerable and row.get("expected_doc_ids"):
            errors.append(f"row {index}: unanswerable question carries expected_doc_ids")
    if manifest.get("metadata", {}).get("protocol_split") != "dev":
        errors.append("a generated dataset must be labelled protocol_split=dev")
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return {
        "dataset_dir": str(source),
        "questions": len(rows),
        "answerable": sum(1 for row in rows if row.get("metadata", {}).get("is_answerable", True)),
        "generator_version": manifest.get("generator_version"),
        "same_model_roles": manifest.get("same_model_roles", []),
        "stages": manifest.get("stages", {}),
    }
