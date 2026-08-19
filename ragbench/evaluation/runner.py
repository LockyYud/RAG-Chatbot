from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragbench import __version__
from ragbench.core.base import get_pipeline_spec
from ragbench.core.io import read_json, read_jsonl, write_json
from ragbench.core.measure import canonical_fingerprint
from ragbench.core.schema import Citation, EvalItem, RAGAnswer, RetrievalResult
from ragbench.datasets.schema import resolve_eval_dataset_path, validate_processed_dataset
from ragbench.evaluation.judge import create_judge
from ragbench.evaluation.metrics import _percentile, evaluate_prediction_rows, evaluate_predictions
from ragbench.evaluation.profiles import resolve_profile, validate_profile
from ragbench.indexing.artifacts import load_manifest
from ragbench.providers.llm_client import capture_provider_usage, generation_seed

if TYPE_CHECKING:
    from ragbench.core.base import BasePipeline

REPORT_SCHEMA_VERSION = "2"


class BudgetExceededError(RuntimeError):
    """Raised when a run's estimated pipeline+judge cost exceeds ``max_estimated_cost_usd``.

    Completed predictions up to the point of failure are already durably
    checkpointed, so the run can be inspected or resumed with ``--resume``
    rather than losing the spend that already happened.
    """


def run_eval(
    pipeline: BasePipeline,
    artifact_path: str,
    dataset_path: str,
    output_path: str,
    top_k: int = 5,
    mode: str = "full_rag",
    judge_spec: dict | None = None,
    profile: str = "auto",
    seed: int | None = None,
    cutoffs: list[int] | None = None,
    suite_metadata: dict[str, str] | None = None,
    warmup_queries: int = 0,
    latency_repetitions: int = 1,
    max_estimated_cost_usd: float | None = None,
    concurrency: int = 1,
    latency_sample_size: int = 5,
    profile_coverage: dict[str, Any] | None = None,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if warmup_queries < 0 or latency_repetitions < 1:
        raise ValueError("warmup_queries must be non-negative and latency_repetitions at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if latency_sample_size < 0:
        raise ValueError("latency_sample_size must be non-negative")
    # Evaluation is always strict, even when the artifact was created by an
    # interactive demo that opted into component fallback.
    if hasattr(pipeline, "allow_fallback"):
        pipeline.allow_fallback = False
    cutoffs = sorted(set(cutoffs or [top_k]))
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError("cutoffs must contain positive integers")
    if top_k not in cutoffs:
        cutoffs.append(top_k)
        cutoffs.sort()
    # Mutating query-depth overrides must happen before load() builds
    # retrievers/rerankers bound to the (now corrected) attribute values.
    evaluation_overrides = _ensure_evaluation_depth(pipeline, max(cutoffs))
    if evaluation_overrides:
        # This can change more than the requested metric cutoff — e.g.
        # bumping rerank_top_k also changes how much gets reranked, not just
        # how many results are reported — so it must not be a silent JSON-only
        # detail (report.run_metadata.evaluation_query_overrides) that a
        # human running this interactively would never see before the run
        # already happened under the adjusted config.
        print(
            f"[{pipeline.id}] increased query-depth field(s) to satisfy cutoffs={cutoffs}: "
            f"{evaluation_overrides} — this changes pipeline behavior, not just what gets reported. "
            "See run_metadata.evaluation_query_overrides in the report.",
            file=sys.stderr,
        )
    artifact_manifest = load_manifest(artifact_path)
    # Load the artifact and build query-time state exactly once. Every
    # question below reuses it — this is what keeps a multi-thousand-query
    # eval from re-parsing nodes.json (and reloading any learned reranker
    # model) once per question, and keeps latency measurements below scoped
    # to query() alone instead of including this one-time cost.
    pipeline.load(artifact_path)
    if Path(dataset_path).is_dir():
        validate_processed_dataset(dataset_path)
    resolved_dataset_path = resolve_eval_dataset_path(dataset_path)
    rows = read_jsonl(resolved_dataset_path)
    items = [EvalItem.from_dict(row) for row in rows]
    if profile == "auto" and mode == "full_rag":
        profile = "citation_rag" if any(item.expected_citations for item in items) else "single_hop_rag"
    else:
        profile = resolve_profile(profile, mode)
    supported_profiles = get_pipeline_spec(pipeline.id).evaluation_profiles
    if profile not in supported_profiles:
        raise ValueError(f"Technique '{pipeline.id}' does not declare support for evaluation profile '{profile}'")
    validate_profile(profile, items, coverage=profile_coverage)
    judge = create_judge(judge_spec) if mode == "full_rag" else None
    dataset_fingerprint = _dataset_fingerprint(resolved_dataset_path, rows)
    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    progress_path = Path(f"{output_path}.progress.log")
    # Fixed against the full ``items`` list, before resume removes anything —
    # a resumed run must measure latency on the *same* questions as the run
    # it continues, not "whichever ones happen to still be fresh." Recorded
    # in the checkpoint header (not just recomputed) so this is auditable and
    # so header equality itself guards against a same-config-different-sample
    # mismatch across resumes.
    latency_sample_question_ids = _select_latency_sample(items, latency_sample_size)
    latency_sample_id_set = set(latency_sample_question_ids)
    header = _checkpoint_header(
        pipeline=pipeline,
        artifact_manifest=dict(artifact_manifest),
        dataset_fingerprint=dataset_fingerprint,
        mode=mode,
        top_k=top_k,
        cutoffs=cutoffs,
        profile=profile,
        seed=seed,
        suite_metadata=suite_metadata,
        judge_enabled=judge is not None,
        warmup_queries=warmup_queries,
        latency_repetitions=latency_repetitions,
        concurrency=concurrency,
        latency_sample_size=latency_sample_size,
        latency_sample_question_ids=latency_sample_question_ids,
    )
    resumed, resumed_warmup, checkpoint_handle = _open_checkpoint(checkpoint_path, header)
    if resumed:
        print(
            f"[{pipeline.id}] resuming {len(resumed)}/{len(items)} already-completed queries from {checkpoint_path}",
            file=sys.stderr,
        )
    predictions_by_id: dict[str, RAGAnswer] = {}
    sequential_latencies_ms: list[float] = []
    quality_pass_count = 0
    quality_pass_elapsed = 0.0
    budget_error: BudgetExceededError | None = None
    # Seeded from any warm-up calls a prior (interrupted) attempt already
    # paid for and checkpointed — a resume must not lose that spend from
    # total_spend, and must not pay for those same calls again (see the loop
    # below, which skips any index already in resumed_warmup).
    warmup_cost_total = sum(float(usage.get("estimated_cost", 0.0)) for usage in resumed_warmup.values())
    warmup_cost_trustworthy = all(usage.get("cost_status") == "estimated" for usage in resumed_warmup.values())
    try:
        # Checked before touching anything else: a resumed run whose
        # checkpointed warm-up calls *alone* already exceed the cap must
        # abort right here. Without this, an index already in
        # resumed_warmup is skipped with no check at all (see the loop
        # below), and a not-yet-resumed one would still run its real API
        # call — and get checkpointed as a new "warmup" record — before the
        # per-iteration check below ever re-examined the resumed total. Both
        # let the run spend at least one more query's worth past the cap
        # that a prior attempt had already tripped.
        if (
            max_estimated_cost_usd is not None
            and warmup_cost_trustworthy
            and warmup_cost_total > max_estimated_cost_usd
        ):
            raise BudgetExceededError(
                f"[{pipeline.id}] estimated warm-up spend ${warmup_cost_total:.4f} recovered from a prior "
                f"checkpointed attempt already exceeded max_estimated_cost_usd={max_estimated_cost_usd}. "
                "No new API calls were made this attempt; rerun with a higher cap (or without --resume, "
                "to start over) to proceed."
            )
        for warmup_index in range(warmup_queries):
            if warmup_index in resumed_warmup:
                continue
            item = items[warmup_index]
            # Explicit warm-ups prevent initialization latency from contaminating
            # scored requests — excluded from quality/latency metrics — but the
            # API calls they make are real spend, so they still count toward
            # the budget guard and the report's total_spend (see _cost_summary).
            # Checkpointed like a prediction (its own record type) so a crash
            # mid-warmup neither loses that spend from total_spend on resume
            # nor pays for the same warm-up call twice.
            with capture_provider_usage() as warmup_usage:
                pipeline.query(item.question, mode=mode)
            warmup_usage_dict = warmup_usage.to_dict()
            _append_checkpoint_line(
                checkpoint_handle, {"type": "warmup", "index": warmup_index, "usage": warmup_usage_dict}
            )
            warmup_cost_total += float(warmup_usage_dict.get("estimated_cost", 0.0))
            if warmup_usage_dict.get("cost_status") != "estimated":
                warmup_cost_trustworthy = False
            # Checked immediately, not deferred to the first real prediction:
            # warm-ups are sequential (no concurrency, no in-flight futures to
            # reconcile), so there is no reason a budget already exceeded by
            # warm-up spend alone should be allowed to keep spending through
            # every remaining warm-up call first.
            if (
                max_estimated_cost_usd is not None
                and warmup_cost_trustworthy
                and warmup_cost_total > max_estimated_cost_usd
            ):
                raise BudgetExceededError(
                    f"[{pipeline.id}] estimated warm-up spend ${warmup_cost_total:.4f} alone exceeded "
                    f"max_estimated_cost_usd={max_estimated_cost_usd} after {warmup_index + 1}/{warmup_queries} "
                    f"warm-up quer{'y' if warmup_index == 0 else 'ies'}. Warm-up spend is checkpointed — "
                    "rerun with --resume to continue without re-paying for completed warm-up calls."
                )

        started_run = time.perf_counter()
        retries_total = 0
        pipeline_cost_total = 0.0
        measurement_cost_total = 0.0
        judge_cost_total = 0.0
        # Every cost seen so far must actually be "estimated" (not "unknown")
        # before the running total means anything. One priced call type
        # (e.g. embeddings) alongside an unpriced one (e.g. chat) still yields
        # a positive-but-incomplete total — that must not trip the guard.
        cost_trustworthy_so_far = warmup_cost_trustworthy
        completed = 0
        # Guards every mutation below. At concurrency=1 nothing ever contends
        # for it (record() is only ever called from the main thread, one item
        # at a time, in items order) — this is what keeps that path's
        # behavior byte-for-byte identical to before concurrency existed.
        lock = threading.Lock()

        def record(item: EvalItem, prediction: RAGAnswer, *, already_checkpointed: bool) -> None:
            nonlocal retries_total, pipeline_cost_total, measurement_cost_total, judge_cost_total
            nonlocal cost_trustworthy_so_far, completed, budget_error
            with lock:
                predictions_by_id[item.question_id] = prediction
                if not already_checkpointed:
                    _append_checkpoint_line(
                        checkpoint_handle,
                        {"type": "prediction", "question_id": item.question_id, "prediction": prediction.to_dict()},
                    )
                completed += 1
                retries_total += int(prediction.metadata.get("provider_usage", {}).get("retries", 0))
                pipeline_cost = prediction.metadata.get("cost_estimate", {})
                pipeline_cost_total += float(pipeline_cost.get("amount", 0.0))
                if pipeline_cost.get("status") != "estimated":
                    cost_trustworthy_so_far = False
                # Extra latency_repetitions beyond the first are real API calls
                # too (see _run_single_query) — they must count toward spend
                # even though they are excluded from technique-quality cost. A
                # prediction with no measurement_cost_estimate at all (e.g. a
                # test double, or a pipeline bypassing _run_single_query) is
                # treated as zero/trustworthy rather than penalized — only a
                # key that is *present* with a non-"estimated" status makes
                # the running total untrustworthy.
                measurement_cost = prediction.metadata.get("measurement_cost_estimate")
                if measurement_cost:
                    measurement_cost_total += float(measurement_cost.get("amount", 0.0))
                    if measurement_cost.get("status") not in ("estimated", "not_applicable"):
                        cost_trustworthy_so_far = False
                if judge is not None:
                    judge_cost = prediction.metadata.get("evaluation_cost_estimate", {})
                    judge_cost_total += float(judge_cost.get("amount", 0.0))
                    if judge_cost.get("status") != "estimated":
                        cost_trustworthy_so_far = False
                    retries_total += int(
                        prediction.metadata.get("evaluation_provider_usage", {}).get("retries", 0)
                    )
                _emit_progress(
                    progress_path,
                    pipeline_id=pipeline.id,
                    completed=completed,
                    total=len(items),
                    elapsed_s=time.perf_counter() - started_run,
                    retries=retries_total,
                    pipeline_cost=pipeline_cost_total,
                    judge_cost=judge_cost_total,
                )
                # Checked after the query, not before: cost is only known once
                # the call has actually happened, so actual spend can exceed
                # the cap by up to one query's cost (or, under concurrency, by
                # up to `concurrency` queries' worth of in-flight work).
                # Genuinely a no-op unless every cost seen so far is
                # "estimated" — with pricing unconfigured (or only partially
                # configured) the guard never fires, however large the
                # partial total looks. Includes warm-up + repeated-measurement
                # spend (total_spend), not just pipeline_cost_total +
                # judge_cost_total — those two alone undercount real spend
                # whenever warmup_queries or latency_repetitions > 1.
                total_cost_so_far = warmup_cost_total + pipeline_cost_total + measurement_cost_total + judge_cost_total
                if (
                    budget_error is None
                    and max_estimated_cost_usd is not None
                    and cost_trustworthy_so_far
                    and total_cost_so_far > max_estimated_cost_usd
                ):
                    budget_error = BudgetExceededError(
                        f"[{pipeline.id}] total estimated spend ${total_cost_so_far:.4f} (technique + "
                        f"measurement + warmup + judge) exceeded max_estimated_cost_usd={max_estimated_cost_usd} "
                        f"after {completed}/{len(items)} queries. "
                        f"{completed} completed prediction(s) are saved in {checkpoint_path} — rerun with "
                        "--resume to continue from here."
                    )

        # Resumed items never re-run; account for them up front, in items
        # order, exactly like the pre-concurrency loop did (they still count
        # toward cost/retry totals and progress, just not re-checkpointed).
        for item in items:
            if item.question_id in resumed:
                record(item, resumed[item.question_id], already_checkpointed=True)

        fresh_items = [item for item in items if item.question_id not in resumed]
        if concurrency <= 1:
            # No pass separation at all — identical to the pre-concurrency loop.
            sequential_items, concurrent_items = fresh_items, []
        else:
            # Latency pass: a sequential, uncontended prefix — real predictions
            # (not re-run later), and the only honest source of per-request
            # latency once the remainder starts contending for resources.
            # Partitioned by the *frozen* sample (fixed above, before resume),
            # not by position in ``fresh_items`` — a question already measured
            # in a prior (interrupted) run is in ``resumed``/``predictions_by_id``
            # already and is not re-run, but it still counts toward the sample.
            sequential_items = [item for item in fresh_items if item.question_id in latency_sample_id_set]
            concurrent_items = [item for item in fresh_items if item.question_id not in latency_sample_id_set]

        for item in sequential_items:
            if budget_error is not None:
                break
            prediction = _run_single_query(
                pipeline, item, mode=mode, latency_repetitions=latency_repetitions, judge=judge, seed=seed
            )
            record(item, prediction, already_checkpointed=False)
        if concurrency > 1:
            # Only needed for the report's latency_pass section below — and
            # only actually meaningful when a pass separation happened at all.
            # Sourced from the frozen sample against every item (not just
            # ``sequential_items``) so a question resumed from a prior run's
            # checkpoint still contributes its already-measured latency.
            sequential_latencies_ms = [
                float(predictions_by_id[item.question_id].metadata.get("latency_ms", 0.0))
                for item in items
                if item.question_id in latency_sample_id_set and item.question_id in predictions_by_id
            ]

        # Quality pass: the concurrent remainder, run for throughput. Ranking/
        # correctness of each answer doesn't depend on wall-clock timing, so
        # running these concurrently is safe; only their *individual* latency
        # is untrustworthy (contention), which is exactly why it was already
        # measured above instead of here.
        quality_pass_started = time.perf_counter()
        if budget_error is None and concurrent_items:

            def _submit(executor: ThreadPoolExecutor, item: EvalItem) -> Any:
                return executor.submit(
                    _run_single_query,
                    pipeline,
                    item,
                    mode=mode,
                    latency_repetitions=latency_repetitions,
                    judge=judge,
                    seed=seed,
                )

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending_items = iter(concurrent_items)
                # Bounded sliding window: at most `concurrency` futures exist at
                # once, front-loaded only up to that many. Submitting every
                # remaining item up front (e.g. via a single dict/executor.map
                # call) would let the budget guard trip only after most of the
                # dataset had already been dispatched — overshoot would then be
                # bounded by "how many queries fit under the cap," not by
                # `concurrency`. Replenishing one-in-one-out, and only while
                # budget_error is still None, is what actually bounds overshoot
                # to the (at most `concurrency`) queries already in flight at
                # the moment the cap trips: once tripped, `pending_items` is
                # simply never advanced again, so no new work is ever dispatched.
                in_flight: dict[Any, EvalItem] = {}
                for item in pending_items:
                    in_flight[_submit(executor, item)] = item
                    if len(in_flight) >= concurrency:
                        break
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = in_flight.pop(future)
                        prediction = future.result()
                        record(item, prediction, already_checkpointed=False)
                        quality_pass_count += 1
                        if budget_error is None:
                            next_item = next(pending_items, None)
                            if next_item is not None:
                                in_flight[_submit(executor, next_item)] = next_item
        quality_pass_elapsed = time.perf_counter() - quality_pass_started
        if budget_error is not None:
            raise budget_error
    finally:
        checkpoint_handle.close()

    predictions = [predictions_by_id[item.question_id] for item in items]
    evaluate_citations = profile == "citation_rag"
    metrics_by_cutoff = {
        str(cutoff): evaluate_predictions(items, predictions, k=cutoff, include_citation_metrics=evaluate_citations)
        for cutoff in cutoffs
    }
    query_metrics_by_cutoff = {
        str(cutoff): evaluate_prediction_rows(items, predictions, k=cutoff, include_citation_metrics=evaluate_citations)
        for cutoff in cutoffs
    }
    metrics = metrics_by_cutoff[str(top_k)]
    query_metrics = query_metrics_by_cutoff[str(top_k)]
    if concurrency > 1:
        # evaluate_predictions() computed latency_ms_avg/p50/p95 from *every*
        # prediction, including the concurrent quality pass's contended
        # timings — those are not honest per-request latency and must not
        # stand as the report's headline latency claim (claim_eligibility()
        # reads exactly this field). Replace with the trustworthy,
        # uncontended latency-pass sample instead. Every cutoff bucket shares
        # the same substitute since latency doesn't depend on top_k.
        trustworthy_latency_ms_avg = (
            round(statistics.fmean(sequential_latencies_ms), 6) if sequential_latencies_ms else 0.0
        )
        trustworthy_latency_ms_p50 = _percentile(sequential_latencies_ms, 50)
        trustworthy_latency_ms_p95 = _percentile(sequential_latencies_ms, 95)
        for bucket in metrics_by_cutoff.values():
            bucket["latency_ms_avg"] = trustworthy_latency_ms_avg
            bucket["latency_ms_p50"] = trustworthy_latency_ms_p50
            bucket["latency_ms_p95"] = trustworthy_latency_ms_p95
    effective_components = sorted(
        {
            canonical_fingerprint(prediction.metadata.get("components", {})): prediction.metadata.get("components", {})
            for prediction in predictions
        }.values(),
        key=lambda value: str(value),
    )
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_metadata": _run_metadata(
            pipeline,
            artifact_path,
            dataset_path,
            resolved_dataset_path,
            rows,
            dict(artifact_manifest),
            mode,
            top_k,
            judge is not None,
            evaluation_overrides,
            profile,
            seed,
            suite_metadata,
            warmup_queries,
            latency_repetitions,
            concurrency,
            latency_sample_size,
        ),
        "metrics": metrics,
        "metrics_by_cutoff": metrics_by_cutoff,
        "query_metrics": query_metrics,
        "query_metrics_by_cutoff": query_metrics_by_cutoff,
        "effective_components": effective_components,
        "cost_summary": _cost_summary(predictions, warmup_cost_total=warmup_cost_total),
        "failures": _failures(items, predictions, top_k, check_citations=evaluate_citations),
        "predictions": [
            {
                "question_id": item.question_id,
                "question": item.question,
                "answer": prediction.answer,
                "abstained": prediction.abstained,
                "citations": [citation.to_dict() for citation in prediction.citations],
                "contexts": [context.to_dict() for context in prediction.contexts],
                "metadata": prediction.metadata,
            }
            for item, prediction in zip(items, predictions, strict=True)
        ],
    }
    if concurrency > 1:
        # Only present when a pass separation actually happened — at
        # concurrency=1 there is exactly one pass and metrics.latency_ms_p50/
        # p95 (over every query) already covers it; adding this key there
        # would just duplicate that number under a different name.
        report["performance"] = {
            "latency_pass": {
                "mode": "sequential",
                "sampled_queries": len(sequential_latencies_ms),
                "question_ids": latency_sample_question_ids,
                "latency_ms_p50": _percentile(sequential_latencies_ms, 50),
                "latency_ms_p95": _percentile(sequential_latencies_ms, 95),
            },
            "quality_pass": {
                "mode": "concurrent",
                "workers": concurrency,
                "queries": quality_pass_count,
                "elapsed_s": round(quality_pass_elapsed, 3),
                "throughput_qps": round(quality_pass_count / quality_pass_elapsed, 3)
                if quality_pass_elapsed > 0
                else 0.0,
            },
        }
    report["result_fingerprint"] = _result_fingerprint(items, predictions, metrics)
    write_json(output_path, report)
    return report


def _run_single_query(
    pipeline: BasePipeline,
    item: EvalItem,
    *,
    mode: str,
    latency_repetitions: int,
    judge: Any,
    seed: int | None = None,
) -> RAGAnswer:
    """Run one (not-yet-checkpointed) question and attach latency/cost/judge metadata."""
    timings: list[float] = []
    repeated_usage: list[dict[str, Any]] = []
    prediction: RAGAnswer | None = None
    pipeline_provider: dict[str, Any] | None = None
    for repetition in range(latency_repetitions):
        with capture_provider_usage() as provider_usage, generation_seed(seed):
            started = time.perf_counter()
            result = pipeline.query(item.question, mode=mode)
            timings.append((time.perf_counter() - started) * 1000)
        usage = provider_usage.to_dict()
        if repetition == 0:
            prediction, pipeline_provider = result, usage
        else:
            repeated_usage.append(usage)
    assert prediction is not None and pipeline_provider is not None
    prediction.metadata["latency_ms"] = round(statistics.median(timings), 3)
    prediction.metadata["latency_measurements_ms"] = [round(value, 3) for value in timings]
    prediction.metadata["provider_usage"] = pipeline_provider
    prediction.metadata["pipeline_provider_usage"] = pipeline_provider
    # Requested, not guaranteed — see providers.llm_client.generation_seed.
    # More than one fingerprint here means the provider's serving snapshot
    # itself changed mid-run, independent of whatever seed was requested.
    prediction.metadata["generation_seed_requested"] = seed
    prediction.metadata["generation_system_fingerprints"] = pipeline_provider.get("system_fingerprints", [])
    if repeated_usage:
        prediction.metadata["latency_measurement_usage"] = repeated_usage
    prediction.metadata["cost_estimate"] = {
        "currency": "USD",
        "amount": pipeline_provider["estimated_cost"],
        "embedding_cost": pipeline_provider["embedding_cost"],
        "chat_cost": pipeline_provider["chat_cost"],
        "rerank_cost": pipeline_provider["rerank_cost"],
        "basis": "one pipeline request (technique_cost); excludes judge and repeated latency measurements",
        "status": pipeline_provider["cost_status"],
    }
    # The extra latency_repetitions calls (beyond the first, already counted
    # above as technique cost) are real API spend that was previously
    # dropped entirely — see run_eval's budget guard, which folds this into
    # total_spend rather than letting max_estimated_cost_usd undercount it.
    if repeated_usage:
        measurement_amount = sum(usage["estimated_cost"] for usage in repeated_usage)
        measurement_statuses = {usage["cost_status"] for usage in repeated_usage}
        prediction.metadata["measurement_cost_estimate"] = {
            "currency": "USD",
            "amount": measurement_amount,
            "embedding_cost": sum(usage["embedding_cost"] for usage in repeated_usage),
            "chat_cost": sum(usage["chat_cost"] for usage in repeated_usage),
            "rerank_cost": sum(usage["rerank_cost"] for usage in repeated_usage),
            "basis": f"{len(repeated_usage)} extra latency_repetitions call(s), excluded from technique_cost",
            "status": "estimated" if measurement_statuses == {"estimated"} else "unknown",
        }
    else:
        prediction.metadata["measurement_cost_estimate"] = {
            "currency": "USD",
            "amount": 0.0,
            "embedding_cost": 0.0,
            "chat_cost": 0.0,
            "rerank_cost": 0.0,
            "basis": "latency_repetitions=1; no extra measurement calls",
            "status": "not_applicable",
        }
    if judge is not None:
        with capture_provider_usage() as judge_usage:
            try:
                judge_result = judge.judge(item, prediction)
                prediction.metadata["judge"] = judge_result.to_dict()
            except Exception as exc:  # evaluator failure must not become a zero-quality answer
                prediction.metadata["judge"] = {
                    "status": "provider_failure",
                    "error_type": type(exc).__name__,
                    "notes": [str(exc)],
                }
        evaluation_provider = judge_usage.to_dict()
        prediction.metadata["evaluation_provider_usage"] = evaluation_provider
        prediction.metadata["evaluation_cost_estimate"] = {
            "currency": "USD",
            "amount": evaluation_provider["estimated_cost"],
            "embedding_cost": evaluation_provider["embedding_cost"],
            "chat_cost": evaluation_provider["chat_cost"],
            "rerank_cost": evaluation_provider["rerank_cost"],
            "basis": "LLM judge only; excluded from technique cost",
            "status": evaluation_provider["cost_status"],
        }
    return prediction


def _dataset_fingerprint(resolved_dataset_path: str, rows: list[dict[str, Any]]) -> str:
    """Same formula ``_run_metadata`` uses, computed early for the checkpoint header."""
    dataset_manifest_path = Path(resolved_dataset_path).parent / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    return str(dataset_manifest.get("fingerprint") or canonical_fingerprint(rows))


def _select_latency_sample(items: list[EvalItem], sample_size: int) -> list[str]:
    """Deterministically pick which questions get sequential (uncontended) latency measurement.

    The first ``sample_size`` questions in dataset order — same rule a fresh
    run always used — but computed once against the *full* item list and
    reused verbatim across a resume. See the note in ``run_eval`` on why
    recomputing this against ``fresh_items`` (post-resume) let the sampled
    question set drift between an interrupted run and its continuation.
    """
    return [item.question_id for item in items[: max(sample_size, 0)]]


def _checkpoint_header(
    *,
    pipeline: BasePipeline,
    artifact_manifest: dict[str, Any],
    dataset_fingerprint: str,
    mode: str,
    top_k: int,
    cutoffs: list[int],
    profile: str,
    seed: int | None,
    suite_metadata: dict[str, str] | None,
    judge_enabled: bool,
    warmup_queries: int,
    latency_repetitions: int,
    concurrency: int,
    latency_sample_size: int,
    latency_sample_question_ids: list[str],
) -> dict[str, Any]:
    """Identify "the exact same evaluation run" so resume only reuses matching results.

    Any field that changes what a query would produce (mode, top_k, cutoffs,
    profile, artifact/config fingerprint, judge on/off, warm-up protocol) must
    be here — resume compares this dict for exact equality before trusting a
    prior checkpoint's predictions. ``concurrency``/``latency_sample_size``
    don't change a prediction's *content*, but they do change how its recorded
    latency must be interpreted (sequential-sample vs. contended quality-pass)
    — mixing predictions checkpointed under one protocol into a resumed run
    under a different one would silently corrupt that interpretation, so a
    protocol change here also starts a fresh checkpoint. ``latency_sample_question_ids``
    is included (not just ``latency_sample_size``) so the *specific* frozen
    sample is part of the equality check, not only its size.
    """
    return {
        "pipeline_id": pipeline.id,
        "artifact_fingerprint": artifact_manifest["corpus"]["fingerprint"],
        "pipeline_config_fingerprint": artifact_manifest["pipeline"]["config_fingerprint"],
        "dataset_fingerprint": dataset_fingerprint,
        "mode": mode,
        "top_k": top_k,
        "cutoffs": cutoffs,
        "evaluation_profile": profile,
        "seed": seed,
        "suite": suite_metadata,
        "judge_enabled": judge_enabled,
        "warmup_queries": warmup_queries,
        "latency_repetitions": latency_repetitions,
        "concurrency": concurrency,
        "latency_sample_size": latency_sample_size,
        "latency_sample_question_ids": latency_sample_question_ids,
    }


def _open_checkpoint(
    checkpoint_path: Path, header: dict[str, Any]
) -> tuple[dict[str, RAGAnswer], dict[int, dict[str, Any]], Any]:
    """Return (completed predictions keyed by question_id, completed warm-up
    usage keyed by warm-up index, an open append/write handle).

    Reuses a checkpoint file only if its header matches *exactly* — any
    parameter drift starts a fresh checkpoint rather than silently mixing
    predictions from a different run configuration.

    A record is written with ``flush`` + ``fsync`` (see :func:`_append_checkpoint_line`),
    but a hard kill can still land mid-``write()``, leaving a torn trailing
    line. Only the *physical last* line is ever treated as a torn write and
    dropped; malformed JSON anywhere else means the file is genuinely
    corrupted and must fail loudly rather than silently lose or mix results.

    Repairing a torn trailing line truncates the file in place to the exact
    byte offset of the last valid line, then fsyncs — it does not rewrite the
    file's full content. A crash mid-repair then either leaves the original
    (still fully valid up to that point) file untouched or completes the
    truncate; there is no window where a second power loss could turn an
    already-durable checkpoint into an empty or half-written file.
    """
    resumed: dict[str, RAGAnswer] = {}
    resumed_warmup: dict[int, dict[str, Any]] = {}
    header_matches = False
    raw_text = checkpoint_path.read_bytes().decode("utf-8") if checkpoint_path.exists() else ""
    # Split on the literal "\n" our writer uses, not str.splitlines() — that
    # also breaks on unicode line separators (e.g. U+2028) that can appear
    # un-escaped inside a prediction's text since records are written with
    # ensure_ascii=False. The byte-offset truncate below must match exactly
    # where our own writes put each newline, not Python's broader notion of
    # a "line".
    raw_lines = raw_text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()  # drop the split artifact of the file's trailing "\n"
    if raw_lines and raw_lines[0].strip():
        try:
            first = json.loads(raw_lines[0])
        except json.JSONDecodeError:
            first = {}
        header_matches = first.get("type") == "header" and first.get("header") == header
    if header_matches:
        body = raw_lines[1:]
        valid_line_count = 1  # the header line
        for index, line in enumerate(body):
            if not line.strip():
                valid_line_count += 1
                continue
            is_last_line = index == len(body) - 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if is_last_line:
                    break  # torn write from a mid-record kill — drop it, not a resume blocker
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} has malformed JSON before its last line — "
                    "this is not a torn write and the file cannot be trusted. Delete it to start over "
                    "(this also discards its completed predictions)."
                ) from None
            if record.get("type") == "prediction":
                resumed[record["question_id"]] = _prediction_from_checkpoint_record(record["prediction"])
            elif record.get("type") == "warmup":
                resumed_warmup[int(record["index"])] = dict(record["usage"])
            valid_line_count += 1
        if valid_line_count < len(raw_lines):
            # A torn trailing record was dropped — repair durably in place.
            valid_byte_length = sum(len(line.encode("utf-8")) + 1 for line in raw_lines[:valid_line_count])
            with checkpoint_path.open("r+b") as raw_handle:
                raw_handle.truncate(valid_byte_length)
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
        handle = checkpoint_path.open("a", encoding="utf-8")
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        handle = checkpoint_path.open("w", encoding="utf-8")
        _append_checkpoint_line(handle, {"type": "header", "header": header})
    return resumed, resumed_warmup, handle


def _append_checkpoint_line(handle: Any, payload: dict[str, Any]) -> None:
    """Write one durable JSONL record: flush to the OS, then fsync to disk.

    ``flush()`` alone only hands the bytes to the OS page cache — a power
    loss (not just a killed process) can still lose them. fsync is what
    makes "the record is in the checkpoint" actually true on disk.
    """
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _prediction_from_checkpoint_record(record: dict[str, Any]) -> RAGAnswer:
    contexts = [
        RetrievalResult(
            node_id=context["node_id"],
            chunk_id=context["chunk_id"],
            doc_id=context["doc_id"],
            text=context["text"],
            score=context["score"],
            rank=context["rank"],
            metadata=dict(context.get("metadata", {})),
        )
        for context in record.get("contexts", [])
    ]
    citations = [
        Citation(
            citation_id=citation["citation_id"],
            doc_id=citation["doc_id"],
            chunk_id=citation["chunk_id"],
            start_char=citation.get("start_char"),
            end_char=citation.get("end_char"),
        )
        for citation in record.get("citations", [])
    ]
    return RAGAnswer(
        query=record["query"],
        answer=record["answer"],
        contexts=contexts,
        citations=citations,
        abstained=bool(record.get("abstained", False)),
        metadata=dict(record.get("metadata", {})),
    )


def _emit_progress(
    progress_path: Path,
    *,
    pipeline_id: str,
    completed: int,
    total: int,
    elapsed_s: float,
    retries: int,
    pipeline_cost: float,
    judge_cost: float,
) -> None:
    rate = completed / elapsed_s if elapsed_s > 0 else 0.0
    eta_s = (total - completed) / rate if rate > 0 else None
    payload = {
        "pipeline_id": pipeline_id,
        "completed": completed,
        "total": total,
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": round(eta_s, 1) if eta_s is not None else None,
        "retries": retries,
        "pipeline_cost_usd": round(pipeline_cost, 6),
        "judge_cost_usd": round(judge_cost, 6),
    }
    print(
        f"[{pipeline_id}] {completed}/{total} elapsed={payload['elapsed_s']}s eta={payload['eta_s']}s "
        f"retries={retries} cost=${pipeline_cost:.4f} judge=${judge_cost:.4f}",
        file=sys.stderr,
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _ensure_evaluation_depth(pipeline: BasePipeline, top_k: int) -> dict[str, int]:
    """Ensure a technique returns enough ranked evidence for Recall@K/MRR@K.

    Pipeline parameters in this list are explicitly query-time overrides.  We
    only increase them, preserving a technique's wider candidate pool while
    preventing a requested metric cutoff from silently exceeding the number of
    results produced by retrieval or reranking.
    """
    adjusted: dict[str, int] = {}
    for name in ("top_k", "candidate_k", "rerank_top_k", "per_tool_top_k", "fusion_per_query_top_k"):
        if name not in pipeline.query_override_fields or not hasattr(pipeline, name):
            continue
        current = getattr(pipeline, name)
        if isinstance(current, int) and current < top_k:
            setattr(pipeline, name, top_k)
            adjusted[name] = top_k
    return adjusted


def _result_fingerprint(items: list[EvalItem], predictions: list[RAGAnswer], metrics: dict[str, Any]) -> str:
    """Fingerprint deterministic outputs, excluding wall-clock measurements."""
    stable_metrics = {key: value for key, value in metrics.items() if not key.startswith("latency_ms_")}
    stable_predictions = []
    for item, prediction in zip(items, predictions, strict=True):
        stable_predictions.append(
            {
                "question_id": item.question_id,
                "answer": prediction.answer,
                "abstained": prediction.abstained,
                "citations": [citation.to_dict() for citation in prediction.citations],
                "contexts": [
                    {
                        "node_id": context.node_id,
                        "chunk_id": context.chunk_id,
                        "doc_id": context.doc_id,
                        "score": context.score,
                        "rank": context.rank,
                    }
                    for context in prediction.contexts
                ],
                "components": prediction.metadata.get("components", {}),
                "verification": prediction.metadata.get("verification", {}),
            }
        )
    return canonical_fingerprint({"metrics": stable_metrics, "predictions": stable_predictions})


def _run_metadata(
    pipeline: BasePipeline,
    artifact_path: str,
    dataset_input_path: str,
    dataset_path: str,
    dataset_rows: list[dict[str, Any]],
    artifact_manifest: dict[str, Any],
    mode: str,
    top_k: int,
    judge_enabled: bool,
    evaluation_overrides: dict[str, int],
    profile: str,
    seed: int | None,
    suite_metadata: dict[str, str] | None,
    warmup_queries: int,
    latency_repetitions: int,
    concurrency: int,
    latency_sample_size: int,
) -> dict[str, Any]:
    dataset_manifest_path = Path(dataset_path).parent / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    return {
        "pipeline_id": pipeline.id,
        "pipeline_name": getattr(pipeline, "name", pipeline.id),
        "artifact_path": artifact_path,
        "artifact_fingerprint": artifact_manifest["corpus"]["fingerprint"],
        "pipeline_config_fingerprint": artifact_manifest["pipeline"]["config_fingerprint"],
        "dataset_path": dataset_path,
        "dataset_input_path": dataset_input_path,
        "dataset_fingerprint": dataset_manifest.get("fingerprint") or canonical_fingerprint(dataset_rows),
        # "unspecified" for the many datasets that haven't adopted the
        # dev/test split convention yet (see suites.claim_eligibility, which
        # rejects split="dev" for claim_eligible-tier runs) — never silently
        # None, so a report reader can tell "not marked" from "marked test".
        "dataset_split": dataset_manifest.get("metadata", {}).get("split", "unspecified"),
        "seed": seed if seed is not None else dataset_manifest.get("metadata", {}).get("seed"),
        # `seed` is requested from the LLM provider on every chat completion
        # (see providers.llm_client.generation_seed) but is NOT a determinism
        # guarantee — see that function's docstring. Multiple runs at the same
        # seed are replicates to average over, never an exact reproduction;
        # check each prediction's metadata.generation_system_fingerprints for
        # whether the provider's serving snapshot itself stayed constant.
        "seed_semantics": "generation_seed_requested_not_guaranteed_deterministic",
        "suite": suite_metadata,
        "mode": mode,
        "evaluation_profile": profile,
        "top_k": top_k,
        "evaluation_query_overrides": evaluation_overrides,
        "judge_enabled": judge_enabled,
        "latency_protocol": {
            "warmup_queries": warmup_queries,
            "repetitions_per_query": latency_repetitions,
            "statistic": "median",
            "concurrency": concurrency,
            "latency_sample_size": latency_sample_size,
        },
        "package_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "git": _git_metadata(),
        "environment": _environment_metadata(),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("litellm", "numpy", "faiss-cpu", "sentence-transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment_metadata() -> dict[str, Any]:
    """Best-effort hardware/software fingerprint for cross-machine benchmark comparability.

    A latency/index-size claim only means something if a reader can tell
    whether two runs happened on comparable hardware (see suites.
    claim_eligibility's production_reasons). Every field degrades to
    ``None`` rather than raising — a missing optional dependency (torch), an
    unavailable command (nvidia-smi), or a numpy build that doesn't expose
    BLAS info must not break a benchmark run just to report metadata about
    it. Deliberately stdlib-first: no new hard dependency was added to get
    this, so it works even in the base install (no ``research``/``vector``
    extras).
    """
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "total_ram_bytes": _total_ram_bytes(),
        "gpu": _gpu_info(),
        "torch_device": _torch_device(),
        "numpy_blas_backend": _numpy_blas_backend(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "chat_model": os.environ.get("CHAT_MODEL"),
        "embed_model": os.environ.get("EMBED_MODEL"),
    }


def _cpu_model() -> str | None:
    # On Linux, platform.processor() typically just echoes the machine arch
    # (e.g. "x86_64") — /proc/cpuinfo's "model name" is the actually useful
    # value there and is preferred when available.
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None  # not POSIX (Windows), or the sysconf names aren't recognized


def _gpu_info() -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        name, memory_total, driver_version = (part.strip() for part in first_line.split(","))
        return {"name": name, "memory_total": memory_total, "driver_version": driver_version}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None  # no NVIDIA GPU, or nvidia-smi isn't on PATH


def _torch_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None  # torch is an optional dependency (see the `rerank` extra)
    try:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except Exception:  # torch internals vary enough across versions/builds that this must not crash a run
        return None
    return "cpu"


def _numpy_blas_backend() -> str | None:
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        if not isinstance(config, dict):
            return None
        return config.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # np.show_config's shape has changed across numpy versions; never let this crash a run
        return None


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True, timeout=2
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, check=True, text=True, timeout=2
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _cost_summary(predictions: list[RAGAnswer], *, warmup_cost_total: float = 0.0) -> dict[str, Any]:
    pipeline_costs: list[float] = []
    pipeline_embedding_costs: list[float] = []
    pipeline_chat_costs: list[float] = []
    pipeline_rerank_costs: list[float] = []
    measurement_costs: list[float] = []
    evaluation_costs: list[float] = []
    evaluation_embedding_costs: list[float] = []
    evaluation_chat_costs: list[float] = []
    evaluation_rerank_costs: list[float] = []
    for prediction in predictions:
        cost = prediction.metadata.get("cost_estimate", {})
        cost = cost if isinstance(cost, dict) else {}
        pipeline_costs.append(float(cost.get("amount", 0.0)))
        pipeline_embedding_costs.append(float(cost.get("embedding_cost", 0.0)))
        pipeline_chat_costs.append(float(cost.get("chat_cost", 0.0)))
        pipeline_rerank_costs.append(float(cost.get("rerank_cost", 0.0)))
        measurement_cost = prediction.metadata.get("measurement_cost_estimate", {})
        measurement_cost = measurement_cost if isinstance(measurement_cost, dict) else {}
        measurement_costs.append(float(measurement_cost.get("amount", 0.0)))
        evaluation_cost = prediction.metadata.get("evaluation_cost_estimate", {})
        evaluation_cost = evaluation_cost if isinstance(evaluation_cost, dict) else {}
        evaluation_costs.append(float(evaluation_cost.get("amount", 0.0)))
        evaluation_embedding_costs.append(float(evaluation_cost.get("embedding_cost", 0.0)))
        evaluation_chat_costs.append(float(evaluation_cost.get("chat_cost", 0.0)))
        evaluation_rerank_costs.append(float(evaluation_cost.get("rerank_cost", 0.0)))
    pipeline_total = sum(pipeline_costs)
    measurement_total = sum(measurement_costs)
    evaluation_total = sum(evaluation_costs)
    return {
        "currency": "USD",
        # technique_cost (pipeline_cost below) / measurement_cost / warmup_cost
        # / judge_cost are kept apart so a technique's per-query cost claim
        # (technique_cost) is never inflated by warm-ups or extra
        # latency_repetitions, while total_spend — what the budget guard in
        # run_eval actually enforces — is never an undercount of real spend.
        "pipeline_cost": {
            "total": round(pipeline_total, 8),
            "avg": round(pipeline_total / len(pipeline_costs), 8) if pipeline_costs else 0.0,
            "embedding_cost_total": round(sum(pipeline_embedding_costs), 8),
            "chat_cost_total": round(sum(pipeline_chat_costs), 8),
            "rerank_cost_total": round(sum(pipeline_rerank_costs), 8),
        },
        "measurement_cost": {
            "total": round(measurement_total, 8),
            "basis": "extra latency_repetitions calls beyond the first, per query",
        },
        "warmup_cost": {
            "total": round(warmup_cost_total, 8),
            "basis": "warmup_queries calls, excluded from every quality/latency metric",
        },
        "judge_cost": {
            "total": round(evaluation_total, 8),
            "avg": round(evaluation_total / len(evaluation_costs), 8) if evaluation_costs else 0.0,
            "embedding_cost_total": round(sum(evaluation_embedding_costs), 8),
            "chat_cost_total": round(sum(evaluation_chat_costs), 8),
            "rerank_cost_total": round(sum(evaluation_rerank_costs), 8),
        },
        "total_spend": round(pipeline_total + measurement_total + warmup_cost_total + evaluation_total, 8),
        # Back-compatible flat aliases for existing readers. NOT what the
        # budget guard enforces (that's total_spend, above) — these
        # intentionally preserve their pre-existing "technique cost only" /
        # "judge cost only" meaning.
        "total_estimated_cost": round(pipeline_total, 8),
        "avg_estimated_cost": round(pipeline_total / len(pipeline_costs), 8) if pipeline_costs else 0.0,
        "evaluation_total_estimated_cost": round(evaluation_total, 8),
        "evaluation_avg_estimated_cost": round(evaluation_total / len(evaluation_costs), 8)
        if evaluation_costs
        else 0.0,
    }


def _failures(
    items: list[EvalItem],
    predictions: list[RAGAnswer],
    top_k: int,
    check_citations: bool = True,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item, prediction in zip(items, predictions, strict=True):
        contexts = prediction.contexts[:top_k]
        retrieved_docs = {context.doc_id for context in contexts}
        retrieved_chunks = {context.chunk_id for context in contexts}
        missing_docs = sorted(set(item.expected_doc_ids) - retrieved_docs)
        missing_chunks = sorted(set(item.expected_chunk_ids) - retrieved_chunks)
        expected_citations = set(item.expected_citations)
        predicted_citations = {citation.doc_id for citation in prediction.citations}
        types: list[str] = []
        is_answerable = bool(item.metadata.get("is_answerable", True))
        if missing_docs or missing_chunks:
            types.append("retrieval_miss")
        if check_citations and is_answerable and expected_citations:
            if not predicted_citations:
                types.append("citation_missing")
            elif not expected_citations & predicted_citations:
                types.append("citation_wrong_document")
        if is_answerable and prediction.abstained:
            types.append("incorrect_abstention")
        if not is_answerable and not prediction.abstained:
            types.append("missed_abstention")
        verification = prediction.metadata.get("verification", {})
        if check_citations and isinstance(verification, dict) and verification.get("status") == "run":
            if not verification.get("grounded", False) and prediction.answer and not prediction.abstained:
                types.append("unsupported_answer")
        if not types:
            continue
        failures.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "types": sorted(set(types)),
                "severity": _severity(types),
                "expected": {
                    "doc_ids": item.expected_doc_ids,
                    "chunk_ids": item.expected_chunk_ids,
                    "citations": item.expected_citations,
                    "is_answerable": is_answerable,
                },
                "predicted": {
                    "doc_ids": sorted(retrieved_docs),
                    "chunk_ids": sorted(retrieved_chunks),
                    "citations": sorted(predicted_citations),
                    "abstained": prediction.abstained,
                },
                "components": prediction.metadata.get("components", {}),
            }
        )
    return sorted(failures, key=lambda row: row["severity"], reverse=True)


def _severity(types: list[str]) -> int:
    weights = {
        "provider_or_component_failure": 5,
        "retrieval_miss": 4,
        "unsupported_answer": 4,
        "missed_abstention": 4,
        "incorrect_abstention": 3,
        "citation_wrong_document": 3,
        "citation_missing": 2,
    }
    return max((weights.get(item, 1) for item in types), default=0)
