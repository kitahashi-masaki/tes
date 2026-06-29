from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        AlignmentBlock,
        AlignmentResult,
        AppleSentenceUnit,
        candidate_agreement_score,
        compute_risk_flags,
        is_large_span_drift,
        ensure_dir,
        classify_qwen_apple_difference,
        raw_span_to_norm_span,
        refine_local_asr_span,
        save_jsonl,
        summarize_quality,
    )
    from tools.asr_alignment.conversation_boundary_hint_builder import build_conversation_boundary_hints  # type: ignore
else:
    from ._core import (
        AlignmentBlock,
        AlignmentResult,
        AppleSentenceUnit,
        candidate_agreement_score,
        compute_risk_flags,
        is_large_span_drift,
        ensure_dir,
        classify_qwen_apple_difference,
        raw_span_to_norm_span,
        refine_local_asr_span,
        save_jsonl,
        summarize_quality,
    )
    from .conversation_boundary_hint_builder import build_conversation_boundary_hints


BOUNDARY_HINT_RULE_VERSION = "v1"
_BOUNDARY_HINT_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_BOUNDARY_HINT_CACHE_LOCK = Lock()


def _boundary_hint_key(source: str, text: str) -> tuple[str, str, str]:
    return source, BOUNDARY_HINT_RULE_VERSION, text


def _build_boundary_layer_cached(
    *,
    source: str,
    text: str,
    stats: dict[str, Any],
    stats_lock: Lock | None = None,
) -> dict[str, Any]:
    key = _boundary_hint_key(source, text)
    t0 = time.time()
    with _BOUNDARY_HINT_CACHE_LOCK:
        cached_layer = _BOUNDARY_HINT_CACHE.get(key)
    if cached_layer is not None:
        if stats_lock is not None:
            with stats_lock:
                stats["boundary_hint_cache_hit_count_by_source"][source] += 1
        else:
            stats["boundary_hint_cache_hit_count_by_source"][source] += 1
        cached = dict(cached_layer)
        cached["cache_hit"] = True
        cached["cache_key"] = key
        return cached
    result = build_conversation_boundary_hints([{"text": text}], text_attr="text")[0]
    layer = dict(result)
    layer["cache_hit"] = False
    layer["cache_key"] = key
    with _BOUNDARY_HINT_CACHE_LOCK:
        _BOUNDARY_HINT_CACHE[key] = dict(layer)
    if stats_lock is not None:
        with stats_lock:
            stats["boundary_hint_cache_miss_count_by_source"][source] += 1
            stats["boundary_hint_build_count_by_source"][source] += 1
            stats["boundary_hint_build_total_sec"] += time.time() - t0
    else:
        stats["boundary_hint_cache_miss_count_by_source"][source] += 1
        stats["boundary_hint_build_count_by_source"][source] += 1
        stats["boundary_hint_build_total_sec"] += time.time() - t0
    return layer


def _extract_candidate(
    alignment: AlignmentResult,
    apple_artifact,
    apple_char_start: int,
    apple_char_end: int,
    engine_artifact,
    *,
    source: str,
) -> dict[str, Any]:
    apple_span = max(apple_char_end - apple_char_start, 1)
    search_radius = max(32, min(96, apple_span // 2 + 16))
    apple_norm_start, apple_norm_end = raw_span_to_norm_span(apple_artifact, apple_char_start, apple_char_end)
    if alignment.apple_to_asr_map:
        projected_start = alignment.apple_to_asr_map[min(apple_norm_start, len(alignment.apple_to_asr_map) - 1)]
        projected_end = alignment.apple_to_asr_map[min(max(apple_norm_end - 1, 0), len(alignment.apple_to_asr_map) - 1)]
    else:
        projected_start = 0.0
        projected_end = float(max(apple_span, 1))
    projected_start = int(max(0, round(projected_start)))
    projected_end = int(max(projected_start + 1, round(projected_end) + 1))
    refined = refine_local_asr_span(
        apple_artifact=apple_artifact,
        asr_artifact=engine_artifact,
        apple_raw_start=apple_char_start,
        apple_raw_end=apple_char_end,
        projected_norm_start=projected_start,
        projected_norm_end=projected_end,
        search_radius=search_radius,
    )
    text = refined["refined_text"].strip() or engine_artifact.raw_text_from_norm_range(projected_start, projected_end).strip()
    if not text:
        text = engine_artifact.raw_text[: min(len(engine_artifact.raw_text), max(2, apple_span))]
    alignment_score = float(refined["local_alignment_score"])
    return {
        "text": text,
        "initial_char_start": refined["initial_char_start"],
        "initial_char_end": refined["initial_char_end"],
        "refined_char_start": refined["refined_char_start"],
        "refined_char_end": refined["refined_char_end"],
        "char_start": refined["refined_char_start"],
        "char_end": refined["refined_char_end"],
        "alignment_score": alignment_score,
        "local_alignment_score": alignment_score,
        "span_is_estimated": not refined["usable_for_agreement"],
        "span_refined": refined["span_refined"],
        "span_drift_start": refined["span_drift_start"],
        "span_drift_end": refined["span_drift_end"],
        "boundary_contamination": refined["boundary_contamination"],
        "usable_for_agreement": refined["usable_for_agreement"],
        "unusable_reason": refined["unusable_reason"],
        "search_radius": search_radius,
        "source": source,
    }


def _build_placeholder_candidate(*, source: str, reason: str, boundary_hints_available: bool) -> dict[str, Any]:
    return {
        "text": "",
        "raw_text": "",
        "alignment_text": "",
        "boundary_text": "",
        "display_text": "",
        "boundary_hints": [],
        "skipped": True,
        "skip_reason": reason,
        "boundary_hints_available": boundary_hints_available,
        "span_refined": False,
        "span_is_estimated": False,
        "span_drift_start": 0,
        "span_drift_end": 0,
        "boundary_contamination": False,
        "usable_for_agreement": False,
        "unusable_reason": ["skipped"],
        "search_radius": 0,
        "source": source,
        "alignment_score": 0.0,
        "local_alignment_score": 0.0,
        "search_radius_initial": 0,
        "search_radius_final": 0,
        "fallback_expanded": False,
        "fallback_expand_reason": [],
        "early_exit": False,
        "early_exit_reason": "",
        "boundary_hint_used_for_boundary_eval": False,
        "boundary_warning": False,
        "boundary_warning_reason": [],
    }


def _build_block_payload(
    idx: int,
    block: AlignmentBlock,
    *,
    episode_id: str,
    sentence_units: list[AppleSentenceUnit],
    apple_artifact,
    alignments: dict[str, AlignmentResult],
    candidate_build_mode: str = "staged",
    build_stats: dict[str, Any] | None = None,
    build_stats_lock: Lock | None = None,
) -> tuple[int, dict[str, Any], str]:
    build_stats = build_stats or {}
    t_block = time.perf_counter()
    stage_secs = defaultdict(float)
    sentence_map = {u.sentence_id: u for u in sentence_units}
    t_stage = time.perf_counter()
    boundary_hints: list[dict[str, Any]] = []
    for sid in block.sentence_ids:
        unit = sentence_map.get(sid)
        if unit is not None:
            boundary_hints.extend(list(getattr(unit, "boundary_hints", []) or []))
    stage_secs["sentence_map_and_hints"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    apple_hint_row = _build_boundary_layer_cached(source="apple", text=block.text, stats=build_stats, stats_lock=build_stats_lock)
    stage_secs["apple_boundary_layer"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    payload: dict[str, Any] = {
        "episode_id": episode_id,
        "block_id": block.block_id,
        "sentence_ids": list(block.sentence_ids),
        "sentence_count": block.sentence_count,
        "block_split_reason": block.block_split_reason,
        "parent_sentence_ids": list(block.parent_sentence_ids or block.sentence_ids),
        "is_sub_block": bool(block.is_sub_block),
        "time": {"start_sec": block.start_sec, "end_sec": block.end_sec},
        "apple": {
            "text": block.text,
            "stability_score": block.stability_score,
            "char_start": block.char_start,
            "char_end": block.char_end,
            "raw_text": apple_hint_row["raw_text"],
            "alignment_text": apple_hint_row["alignment_text"],
            "boundary_text": apple_hint_row["boundary_text"],
            "display_text": apple_hint_row["display_text"],
            "boundary_hints": apple_hint_row["boundary_hints"],
            "alignment_score": block.stability_score,
            "local_alignment_score": block.stability_score,
            "usable_for_agreement": True,
            "unusable_reason": [],
            "boundary_contamination": False,
            "span_is_estimated": False,
            "span_refined": False,
            "span_drift_start": 0,
            "span_drift_end": 0,
            "search_radius": 0,
            "source": "apple",
        },
        "apple_boundary_hints": boundary_hints,
    }
    stage_secs["apple_payload_init"] += time.perf_counter() - t_stage
    candidate_rows: dict[str, dict[str, Any]] = {}
    t_stage = time.perf_counter()
    qwen_alignment = alignments["qwen"]
    qwen_candidate = _extract_candidate(
        qwen_alignment,
        apple_artifact,
        block.char_start,
        block.char_end,
        qwen_alignment.asr_artifact,
        source="qwen",
    )
    stage_secs["qwen_refinement"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    qwen_layer = _build_boundary_layer_cached(source="qwen", text=qwen_candidate["text"], stats=build_stats, stats_lock=build_stats_lock)
    qwen_layer.update(qwen_candidate)
    candidate_rows["qwen"] = qwen_layer
    stage_secs["qwen_boundary_layer"] += time.perf_counter() - t_stage

    qwen_high_confidence = (
        qwen_layer.get("local_alignment_score", 0.0) >= 0.92
        and qwen_layer.get("boundary_contamination") is False
        and qwen_layer.get("usable_for_agreement") is True
    )
    skip_support = candidate_build_mode in {"staged", "qwen-only"} and qwen_high_confidence
    if candidate_build_mode == "qwen-only":
        skip_support = True

    t_stage = time.perf_counter()
    for engine in ("nemotron", "whisper"):
        if skip_support:
            candidate_rows[engine] = _build_placeholder_candidate(
                source=engine,
                reason="qwen_high_confidence" if candidate_build_mode != "qwen-only" else "qwen_only_mode",
                boundary_hints_available=bool(boundary_hints),
            )
            continue
        alignment = alignments[engine]
        t_engine = time.perf_counter()
        candidate = _extract_candidate(
            alignment,
            apple_artifact,
            block.char_start,
            block.char_end,
            alignment.asr_artifact,
            source=engine,
        )
        stage_secs[f"{engine}_refinement"] += time.perf_counter() - t_engine
        t_engine = time.perf_counter()
        candidate_rows[engine] = _build_boundary_layer_cached(source=engine, text=candidate["text"], stats=build_stats, stats_lock=build_stats_lock)
        candidate_rows[engine].update(candidate)
        stage_secs[f"{engine}_boundary_layer"] += time.perf_counter() - t_engine
    stage_secs["support_candidates"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    payload.update(candidate_rows)
    usable_asr_candidates = {source: row for source, row in candidate_rows.items() if row.get("usable_for_agreement")}
    candidate_texts = {source: row.get("text", "") for source, row in usable_asr_candidates.items()}
    agreement = candidate_agreement_score(candidate_texts)
    usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in usable_asr_candidates.values()]
    if not usable_scores:
        usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in candidate_rows.values()]
    qwen_diff_type, qwen_similarity, qwen_critical = classify_qwen_apple_difference(block.text, candidate_rows["qwen"]["text"])
    stage_secs["agreement_and_difference"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    apple_layer = apple_hint_row
    alignment_quality = summarize_quality(
        min(usable_scores) if usable_scores else 0.0,
        agreement,
        False,
        not bool(usable_asr_candidates),
    )
    payload["candidate_agreement_score"] = agreement
    payload["apple_raw_text"] = block.text
    payload["apple_alignment_text"] = apple_layer["alignment_text"]
    payload["apple_boundary_text"] = apple_layer["boundary_text"]
    payload["apple_display_text"] = apple_layer["display_text"]
    payload["apple_boundary_hints"] = apple_layer["boundary_hints"]
    payload["qwen_apple_difference_type"] = qwen_diff_type
    payload["qwen_apple_similarity"] = qwen_similarity
    payload["important_term_disagreement"] = qwen_diff_type in {"critical", "semantic"}
    payload["critical_term_disagreement"] = qwen_critical
    payload["soft_domain_difference"] = qwen_diff_type == "soft_domain"
    payload["usable_asr_candidates"] = usable_asr_candidates
    payload["alignment_quality"] = alignment_quality
    payload["risk_flags"] = compute_risk_flags(payload)
    stage_secs["risk_flag_classification"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    large_span_drift = any(is_large_span_drift(row) for row in candidate_rows.values())
    payload["large_span_drift"] = large_span_drift
    qwen_local_alignment = float(candidate_rows["qwen"].get("local_alignment_score", 0.0))
    auto_accept = (
        alignment_quality in {"A", "B"}
        and qwen_local_alignment >= 0.82
        and agreement >= 0.70
        and "numeric_disagreement" not in payload["risk_flags"]
        and not qwen_critical
        and qwen_diff_type in {"none", "surface", "soft_domain"}
        and (qwen_diff_type != "soft_domain" or qwen_similarity >= 0.88)
        and "span_too_long" not in payload["risk_flags"]
        and "all_models_disagree" not in payload["risk_flags"]
        and (not large_span_drift or qwen_local_alignment >= 0.90)
    )
    severe_flags = {"all_models_disagree", "numeric_disagreement", "span_too_short", "span_too_long", "qwen_alignment_low", "apple_unstable", "critical_term_disagreement"}
    payload["needs_review"] = not auto_accept and (
        alignment_quality == "E" or (
            alignment_quality in {"C", "D"} and any(flag in severe_flags for flag in payload["risk_flags"])
        ) or qwen_diff_type in {"critical", "semantic"} or qwen_critical or (large_span_drift and qwen_local_alignment < 0.90)
    )
    payload["auto_accepted"] = not payload["needs_review"]
    payload["candidate_build_mode"] = candidate_build_mode
    payload["boundary_hint_used_for_boundary_eval"] = bool(boundary_hints)
    payload["boundary_warning"] = False
    payload["boundary_warning_reason"] = []
    payload["qwen"]["search_radius_initial"] = payload["qwen"].get("search_radius", 0)
    payload["qwen"]["search_radius_final"] = payload["qwen"].get("search_radius", 0)
    payload["qwen"]["fallback_expanded"] = False
    payload["qwen"]["fallback_expand_reason"] = []
    payload["qwen"]["early_exit"] = False
    payload["qwen"]["early_exit_reason"] = ""
    payload["qwen"]["boundary_hint_used_for_boundary_eval"] = bool(boundary_hints)
    payload["qwen"]["boundary_warning"] = False
    payload["qwen"]["boundary_warning_reason"] = []
    stage_secs["final_payload_finalize"] += time.perf_counter() - t_stage
    payload["block_build_sec_by_stage"] = dict(stage_secs)
    payload["block_build_total_sec"] = time.perf_counter() - t_block
    return idx, payload, alignment_quality


def build_block_candidates(
    *,
    episode_id: str,
    alignment_blocks: list[AlignmentBlock],
    sentence_units: list[AppleSentenceUnit],
    apple_artifact,
    alignments: dict[str, AlignmentResult],
    output_dir: Path | None = None,
    candidate_build_mode: str = "staged",
    workers: int | None = None,
    no_parallel: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    print(f"[block] start episode={episode_id} count={len(alignment_blocks)}", flush=True)
    rows: list[dict[str, Any]] = [None] * len(alignment_blocks)  # type: ignore[list-item]
    quality_counts: dict[str, int] = {}
    max_workers = 1 if no_parallel else (workers if workers and workers > 0 else min(8, max(1, (os.cpu_count() or 1))))
    executor_kind = "thread"
    print(f"[block] parallel workers={max_workers} mode={executor_kind} build_mode={candidate_build_mode}", flush=True)
    stage_secs = defaultdict(float)
    engine_secs = defaultdict(float)
    engine_exec = Counter()
    engine_skip = Counter()
    engine_cache_hit = Counter()
    engine_early_exit = Counter()
    fallback_expanded = Counter()
    search_radius_sum = Counter()
    search_radius_max = Counter()
    stage_secs = defaultdict(float)
    build_stats = {
        "boundary_hint_build_total_sec": 0.0,
        "boundary_hint_build_count_by_source": Counter(),
        "boundary_hint_cache_hit_count_by_source": Counter(),
        "boundary_hint_cache_miss_count_by_source": Counter(),
    }
    build_stats_lock = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _build_block_payload,
                idx,
                block,
                episode_id=episode_id,
                sentence_units=sentence_units,
                apple_artifact=apple_artifact,
                alignments=alignments,
                candidate_build_mode=candidate_build_mode,
                build_stats=build_stats,
                build_stats_lock=build_stats_lock,
            ): idx
            for idx, block in enumerate(alignment_blocks, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            idx, payload, alignment_quality = future.result()
            rows[idx - 1] = payload
            quality_counts[alignment_quality] = quality_counts.get(alignment_quality, 0) + 1
            for stage_name, elapsed in (payload.get("block_build_sec_by_stage") or {}).items():
                stage_secs[stage_name] += float(elapsed)
            for engine in ("qwen", "nemotron", "whisper"):
                cand = payload.get(engine, {})
                if cand.get("skipped"):
                    engine_skip[engine] += 1
                else:
                    engine_exec[engine] += 1
                    sr = int(cand.get("search_radius", 0) or 0)
                    search_radius_sum[engine] += sr
                    search_radius_max[engine] = max(search_radius_max[engine], sr)
                    if cand.get("early_exit"):
                        engine_early_exit[engine] += 1
                    if cand.get("fallback_expanded"):
                        fallback_expanded[engine] += 1
            completed += 1
            if completed == 1 or completed % 5 == 0 or completed == len(alignment_blocks):
                print(
                    f"[block] progress episode={episode_id} {completed}/{len(alignment_blocks)} block_id={payload['block_id']}",
                    flush=True,
                )
    total_sec = time.time() - t0

    summary = {
        "episode_id": episode_id,
        "block_candidate_count": len(rows),
        "quality_counts": quality_counts,
        "needs_review_count": sum(1 for row in rows if row["needs_review"]),
        "candidate_build_total_sec": total_sec,
        "candidate_build_sec_by_stage": {
            "sentence_map_and_hints": round(stage_secs["sentence_map_and_hints"], 4),
            "apple_boundary_layer": round(stage_secs["apple_boundary_layer"], 4),
            "apple_payload_init": round(stage_secs["apple_payload_init"], 4),
            "qwen_refinement": round(stage_secs["qwen_refinement"], 4),
            "qwen_boundary_layer": round(stage_secs["qwen_boundary_layer"], 4),
            "nemotron_refinement": round(stage_secs["nemotron_refinement"], 4),
            "nemotron_boundary_layer": round(stage_secs["nemotron_boundary_layer"], 4),
            "whisper_refinement": round(stage_secs["whisper_refinement"], 4),
            "whisper_boundary_layer": round(stage_secs["whisper_boundary_layer"], 4),
            "support_candidates": round(stage_secs["support_candidates"], 4),
            "agreement_and_difference": round(stage_secs["agreement_and_difference"], 4),
            "risk_flag_classification": round(stage_secs["risk_flag_classification"], 4),
            "final_payload_finalize": round(stage_secs["final_payload_finalize"], 4),
        },
        "candidate_build_sec_by_engine": dict(engine_secs),
        "candidate_build_sec_by_stage_mean": {
            stage_name: round(elapsed / max(len(rows), 1), 4) for stage_name, elapsed in stage_secs.items()
        },
        "candidate_refinement_executed_count_by_engine": dict(engine_exec),
        "candidate_refinement_skipped_count_by_engine": dict(engine_skip),
        "candidate_refinement_cache_hit_count_by_engine": dict(engine_cache_hit),
        "candidate_refinement_early_exit_count_by_engine": dict(engine_early_exit),
        "fallback_expanded_count_by_engine": dict(fallback_expanded),
        "average_search_radius_by_engine": {
            engine: round(search_radius_sum[engine] / max(engine_exec[engine], 1), 2) for engine in ("qwen", "nemotron", "whisper")
        },
        "max_search_radius_by_engine": dict(search_radius_max),
        "candidate_build_mode": candidate_build_mode,
        "workers": max_workers,
        "boundary_hint_build_total_sec": build_stats["boundary_hint_build_total_sec"],
        "boundary_hint_build_count_by_source": dict(build_stats["boundary_hint_build_count_by_source"]),
        "boundary_hint_cache_hit_count_by_source": dict(build_stats["boundary_hint_cache_hit_count_by_source"]),
        "boundary_hint_cache_miss_count_by_source": dict(build_stats["boundary_hint_cache_miss_count_by_source"]),
        "candidate_build_executor_kind": executor_kind,
        "parallel_enabled": not no_parallel,
    }

    if output_dir is not None:
        ensure_dir(output_dir / "aligned_segments")
        save_jsonl(output_dir / "aligned_segments" / f"{episode_id}.block_candidates.jsonl", rows)
        save_jsonl(output_dir / "aligned_segments" / f"{episode_id}.segment_candidates.jsonl", rows)
        (output_dir / "normalized" / f"{episode_id}.block_candidates.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[block] end episode={episode_id} blocks={len(rows)} needs_review={summary['needs_review_count']} "
        f"seconds={time.time() - t0:.2f}",
        flush=True,
    )
    return rows, summary
