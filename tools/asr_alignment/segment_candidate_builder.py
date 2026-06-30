from __future__ import annotations

import json
import os
import re
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
        _normalize_span_text,
        _match_span_similarity,
        _span_similarity,
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
        _normalize_span_text,
        _match_span_similarity,
        _span_similarity,
        refine_local_asr_span,
        save_jsonl,
        summarize_quality,
    )
    from .conversation_boundary_hint_builder import build_conversation_boundary_hints


BOUNDARY_HINT_RULE_VERSION = "v1"
SUPPORT_GLOBAL_ALIGNMENT_SKIP_THRESHOLD = 0.70
_BOUNDARY_HINT_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_BOUNDARY_HINT_CACHE_LOCK = Lock()
_BOUNDARY_CONTAMINATION_PREFIXES = ("ね。", "うことです", "のだから")
_DOMAIN_ERROR_PHRASES = ("排水の陣", "各家族", "社会行動", "整形立てて")


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


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"[0-9０-９]+", text or ""))


def _has_numeric_disagreement(left: str, right: str) -> bool:
    left_tokens = _numeric_tokens(left)
    right_tokens = _numeric_tokens(right)
    return bool(left_tokens or right_tokens) and left_tokens != right_tokens


def _cheap_local_span(
    *,
    apple_target: str,
    asr_norm_text: str,
    projected_start: int,
    projected_end: int,
    radius: int = 24,
) -> dict[str, Any]:
    asr_len = len(asr_norm_text)
    if asr_len <= 0:
        return {"start": projected_start, "end": projected_end, "score": 0.0}
    target_len = max(len(_normalize_span_text(apple_target)), 1)
    window_left = max(0, projected_start - radius)
    window_right = min(asr_len, projected_end + radius)
    min_len = max(1, int(target_len * 0.75))
    max_len = max(min_len, int(target_len * 1.35))
    best = {"start": projected_start, "end": projected_end, "score": -1.0, "drift": 10**9}
    for start in range(window_left, min(window_right, asr_len - 1) + 1, 2):
        for length in range(min_len, max_len + 1, 2):
            end = min(asr_len, start + length)
            if end <= start or end > window_right:
                continue
            candidate = asr_norm_text[start:end]
            text_sim = _match_span_similarity(apple_target, candidate)
            len_score = 1.0 - min(1.0, abs(len(candidate) - target_len) / max(target_len, len(candidate), 1))
            score = max(0.0, min(1.0, 0.82 * text_sim + 0.18 * len_score))
            drift = abs(start - projected_start) + abs(end - projected_end)
            if score > best["score"] or (score == best["score"] and drift < best["drift"]):
                best = {"start": start, "end": end, "score": score, "drift": drift}
    if best["score"] < 0:
        best["score"] = 0.0
    return best


def _has_boundary_contamination_suspect(text: str) -> bool:
    stripped = str(text or "").strip()
    return any(stripped.startswith(prefix) for prefix in _BOUNDARY_CONTAMINATION_PREFIXES)


def _domain_error_phrase_counts(texts: list[str]) -> dict[str, int]:
    joined = "\n".join(str(text or "") for text in texts)
    return {phrase: joined.count(phrase) for phrase in _DOMAIN_ERROR_PHRASES if phrase in joined}


def _extract_candidate(
    alignment: AlignmentResult,
    apple_artifact,
    apple_char_start: int,
    apple_char_end: int,
    engine_artifact,
    *,
    source: str,
    allow_cheap_accept: bool = False,
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

    apple_target = apple_artifact.match_norm_text[apple_norm_start:apple_norm_end]
    apple_raw_text = apple_artifact.raw_text[apple_char_start:apple_char_end]
    cheap_span = {"start": projected_start, "end": projected_end, "score": 0.0}
    if allow_cheap_accept and source == "qwen":
        cheap_span = _cheap_local_span(
            apple_target=apple_target,
            asr_norm_text=engine_artifact.match_norm_text,
            projected_start=projected_start,
            projected_end=projected_end,
        )
        projected_start = int(cheap_span["start"])
        projected_end = int(cheap_span["end"])

    initial_char_start, initial_char_end = engine_artifact.raw_span_from_norm_range(projected_start, projected_end)
    if initial_char_end <= initial_char_start:
        fallback_start = engine_artifact.norm_index_to_raw_index(projected_start)
        initial_char_start = int(fallback_start or 0)
        initial_char_end = min(
            len(engine_artifact.raw_text),
            max(initial_char_start + max(len(apple_artifact.raw_text[apple_char_start:apple_char_end]), 1), initial_char_start + 1),
        )
    initial_text = engine_artifact.raw_text[initial_char_start:initial_char_end].strip() or engine_artifact.raw_text_from_norm_range(projected_start, projected_end).strip()
    if not initial_text:
        initial_text = engine_artifact.raw_text[: min(len(engine_artifact.raw_text), max(2, apple_span))]
    initial_local_alignment_score = max(float(cheap_span.get("score", 0.0)), max(0.0, min(1.0, _span_similarity(apple_target, initial_text))))
    span_length_ratio = len(_normalize_span_text(initial_text)) / max(len(_normalize_span_text(apple_target)), 1)
    initial_diff_type, _, initial_critical = classify_qwen_apple_difference(apple_raw_text, initial_text)
    initial_numeric_disagreement = _has_numeric_disagreement(apple_raw_text, initial_text)
    boundary_contamination = (
        not initial_text.strip()
        or initial_text[:1].isspace()
        or initial_text[-1:].isspace()
        or _has_boundary_contamination_suspect(initial_text)
    )
    qwen_fast_accept = (
        allow_cheap_accept
        and source == "qwen"
        and initial_local_alignment_score >= 0.94
        and 0.75 <= span_length_ratio <= 1.35
        and not boundary_contamination
        and not initial_numeric_disagreement
        and not initial_critical
        and initial_diff_type != "critical"
        and not alignment.asr_artifact.raw_text[:1].isspace()
    )
    if qwen_fast_accept:
        return {
            "text": initial_text,
            "raw_text": initial_text,
            "alignment_text": initial_text,
            "boundary_text": initial_text,
            "display_text": initial_text,
            "boundary_hints": [],
            "initial_char_start": initial_char_start,
            "initial_char_end": initial_char_end,
            "refined_char_start": initial_char_start,
            "refined_char_end": initial_char_end,
            "char_start": initial_char_start,
            "char_end": initial_char_end,
            "alignment_score": initial_local_alignment_score,
            "local_alignment_score": initial_local_alignment_score,
            "span_is_estimated": False,
            "span_refined": False,
            "span_drift_start": 0,
            "span_drift_end": 0,
            "boundary_contamination": False,
            "usable_for_agreement": len(_normalize_span_text(initial_text)) > 2,
            "unusable_reason": "" if len(_normalize_span_text(initial_text)) > 2 else "too_short",
            "search_radius": search_radius,
            "source": source,
            "search_radius_initial": search_radius,
            "search_radius_final": search_radius,
            "fallback_expanded": False,
            "fallback_expand_reason": [],
            "early_exit": True,
            "early_exit_reason": "high_local_alignment",
            "cheap_span_accept": True,
            "cheap_span_accept_reason": "high_initial_alignment",
            "heavy_refinement_skipped": True,
            "refinement_search_profile": "cheap_accept",
            "refinement_candidate_eval_count": 0,
            "refinement_candidate_pruned_count": 0,
            "refinement_window_span": 0,
            "refinement_target_len": len(_normalize_span_text(apple_target)),
            "boundary_hint_used_for_boundary_eval": False,
            "boundary_warning": False,
            "boundary_warning_reason": [],
            "numeric_disagreement": False,
            "critical_term_disagreement": False,
        }
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
        "search_radius_initial": search_radius,
        "search_radius_final": search_radius,
        "fallback_expanded": False,
        "fallback_expand_reason": [],
        "early_exit": bool(refined.get("early_exit")),
        "early_exit_reason": refined.get("early_exit_reason", ""),
        "cheap_span_accept": bool(refined.get("cheap_span_accept")),
        "cheap_span_accept_reason": refined.get("cheap_span_accept_reason", ""),
        "heavy_refinement_skipped": bool(refined.get("heavy_refinement_skipped")),
        "refinement_search_profile": refined.get("refinement_search_profile", "full"),
        "refinement_candidate_eval_count": int(refined.get("refinement_candidate_eval_count", 0) or 0),
        "refinement_candidate_pruned_count": int(refined.get("refinement_candidate_pruned_count", 0) or 0),
        "refinement_window_span": int(refined.get("refinement_window_span", 0) or 0),
        "refinement_target_len": int(refined.get("refinement_target_len", 0) or 0),
        "refinement_estimated_full_grid_count": int(refined.get("refinement_estimated_full_grid_count", 0) or 0),
        "refinement_start_candidate_count": int(refined.get("refinement_start_candidate_count", 0) or 0),
        "refinement_end_candidate_count": int(refined.get("refinement_end_candidate_count", 0) or 0),
        "boundary_hint_used_for_boundary_eval": False,
        "boundary_warning": False,
        "boundary_warning_reason": [],
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
        "refinement_search_profile": "skipped",
        "refinement_candidate_eval_count": 0,
        "refinement_candidate_pruned_count": 0,
        "refinement_window_span": 0,
        "refinement_target_len": 0,
        "refinement_estimated_full_grid_count": 0,
        "refinement_start_candidate_count": 0,
        "refinement_end_candidate_count": 0,
        "boundary_hint_used_for_boundary_eval": False,
        "boundary_warning": False,
        "boundary_warning_reason": [],
    }


def _build_block_payload(
    idx: int,
    block: AlignmentBlock,
    *,
    episode_id: str,
    block_boundary_hints: dict[str, list[dict[str, Any]]],
    apple_artifact,
    alignments: dict[str, AlignmentResult],
    candidate_build_mode: str = "staged",
    build_stats: dict[str, Any] | None = None,
    build_stats_lock: Lock | None = None,
) -> tuple[int, dict[str, Any], str]:
    build_stats = build_stats or {}
    t_block = time.perf_counter()
    stage_secs = defaultdict(float)
    t_stage = time.perf_counter()
    boundary_hints: list[dict[str, Any]] = list(block_boundary_hints.get(block.block_id, []))
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
        allow_cheap_accept=True,
    )
    stage_secs["qwen_refinement"] += time.perf_counter() - t_stage
    t_stage = time.perf_counter()
    qwen_layer = _build_boundary_layer_cached(source="qwen", text=qwen_candidate["text"], stats=build_stats, stats_lock=build_stats_lock)
    qwen_layer.update(qwen_candidate)
    candidate_rows["qwen"] = qwen_layer
    stage_secs["qwen_boundary_layer"] += time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    qwen_diff_type, qwen_similarity, qwen_critical = classify_qwen_apple_difference(block.text, qwen_layer["text"])
    qwen_numeric_disagreement = _has_numeric_disagreement(block.text, qwen_layer["text"])
    qwen_domain_error = bool(_domain_error_phrase_counts([qwen_layer.get("text", "")]))
    preliminary_agreement = candidate_agreement_score({"apple": block.text, "qwen": qwen_layer.get("text", "")})
    preliminary_quality = summarize_quality(
        min(float(qwen_layer.get("local_alignment_score", 0.0)), float(block.stability_score)),
        preliminary_agreement,
        False,
        False,
    )
    qwen_high_confidence = (
        qwen_layer.get("local_alignment_score", 0.0) >= 0.90
        and qwen_similarity >= 0.88
        and qwen_diff_type in {"none", "surface", "soft_domain"}
        and not qwen_numeric_disagreement
        and not qwen_critical
        and not qwen_domain_error
        and preliminary_quality in {"A", "B"}
        and qwen_layer.get("boundary_contamination") is False
        and qwen_layer.get("usable_for_agreement") is True
    )
    qwen_high_confidence_reject_reasons: list[str] = []
    if not qwen_high_confidence:
        if qwen_layer.get("local_alignment_score", 0.0) < 0.90:
            qwen_high_confidence_reject_reasons.append("qwen_alignment_lt_0_90")
        if qwen_similarity < 0.88:
            qwen_high_confidence_reject_reasons.append("qwen_apple_similarity_lt_0_88")
        if qwen_diff_type not in {"none", "surface", "soft_domain"}:
            qwen_high_confidence_reject_reasons.append("qwen_difference_not_safe")
        if qwen_numeric_disagreement:
            qwen_high_confidence_reject_reasons.append("qwen_numeric_disagreement")
        if qwen_critical:
            qwen_high_confidence_reject_reasons.append("qwen_critical_term_disagreement")
        if qwen_domain_error:
            qwen_high_confidence_reject_reasons.append("qwen_domain_error")
        if preliminary_quality not in {"A", "B"}:
            qwen_high_confidence_reject_reasons.append("preliminary_quality_not_a_or_b")
        if qwen_layer.get("boundary_contamination") is not False:
            qwen_high_confidence_reject_reasons.append("qwen_boundary_contamination")
        if qwen_layer.get("usable_for_agreement") is not True:
            qwen_high_confidence_reject_reasons.append("qwen_not_usable_for_agreement")
    skip_support = candidate_build_mode in {"staged", "qwen-only"} and qwen_high_confidence
    if candidate_build_mode == "qwen-only":
        skip_support = True
    stage_secs["support_candidate_decision_sec"] += time.perf_counter() - t_stage

    for engine in ("nemotron", "whisper"):
        if skip_support:
            t_skip = time.perf_counter()
            reason = "qwen_high_confidence" if candidate_build_mode != "qwen-only" else "qwen_only_mode"
            candidate_rows[engine] = _build_placeholder_candidate(
                source=engine,
                reason=reason,
                boundary_hints_available=bool(boundary_hints),
            )
            stage_secs["support_candidate_skip_sec"] += time.perf_counter() - t_skip
            continue
        alignment = alignments[engine]
        if float(alignment.global_alignment_score) < SUPPORT_GLOBAL_ALIGNMENT_SKIP_THRESHOLD:
            t_skip = time.perf_counter()
            candidate_rows[engine] = _build_placeholder_candidate(
                source=engine,
                reason="global_alignment_low",
                boundary_hints_available=bool(boundary_hints),
            )
            candidate_rows[engine]["global_alignment_score"] = float(alignment.global_alignment_score)
            stage_secs["support_candidate_skip_sec"] += time.perf_counter() - t_skip
            continue
        t_refine = time.perf_counter()
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
        stage_secs["support_candidate_refinement_sec"] += time.perf_counter() - t_refine
        t_engine = time.perf_counter()
        candidate_rows[engine] = _build_boundary_layer_cached(source=engine, text=candidate["text"], stats=build_stats, stats_lock=build_stats_lock)
        candidate_rows[engine].update(candidate)
        stage_secs[f"{engine}_boundary_layer"] += time.perf_counter() - t_engine
        stage_secs["support_candidate_payload_sec"] += time.perf_counter() - t_engine
    stage_secs["support_candidates"] += (
        stage_secs["support_candidate_decision_sec"]
        + stage_secs["support_candidate_payload_sec"]
        + stage_secs["support_candidate_skip_sec"]
    )
    t_stage = time.perf_counter()
    payload.update(candidate_rows)
    usable_asr_candidates = {source: row for source, row in candidate_rows.items() if row.get("usable_for_agreement")}
    candidate_texts = {"apple": block.text}
    candidate_texts.update({source: row.get("text", "") for source, row in usable_asr_candidates.items()})
    agreement = candidate_agreement_score(candidate_texts)
    usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in usable_asr_candidates.values()]
    if not usable_scores:
        usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in candidate_rows.values()]
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
    payload["qwen_high_confidence"] = qwen_high_confidence
    payload["qwen_high_confidence_reject_reasons"] = qwen_high_confidence_reject_reasons
    payload["qwen_domain_error"] = qwen_domain_error
    payload["important_term_disagreement"] = qwen_diff_type in {"critical", "semantic"}
    payload["critical_term_disagreement"] = qwen_critical
    payload["soft_domain_difference"] = qwen_diff_type == "soft_domain"
    payload["usable_asr_candidates"] = usable_asr_candidates
    payload["alignment_quality"] = alignment_quality
    payload["risk_flags"] = compute_risk_flags(payload)
    if skip_support:
        payload["risk_flags"] = [
            flag for flag in payload["risk_flags"]
            if flag not in {"nemotron_alignment_low", "whisper_alignment_low", "all_models_disagree"}
        ]
    if _has_boundary_contamination_suspect(candidate_rows["qwen"].get("text", "")) or _has_boundary_contamination_suspect(block.text):
        if "boundary_contamination_suspected" not in payload["risk_flags"]:
            payload["risk_flags"].append("boundary_contamination_suspected")
    domain_error_counts = _domain_error_phrase_counts([block.text] + [row.get("text", "") for row in candidate_rows.values()])
    if domain_error_counts and "domain_error_phrase" not in payload["risk_flags"]:
        payload["risk_flags"].append("domain_error_phrase")
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
    severe_flags = {
        "all_models_disagree",
        "numeric_disagreement",
        "span_too_short",
        "span_too_long",
        "qwen_alignment_low",
        "apple_unstable",
        "critical_term_disagreement",
        "boundary_contamination_suspected",
        "domain_error_phrase",
    }
    payload["needs_review"] = not auto_accept and (
        alignment_quality == "E" or (
            alignment_quality in {"C", "D"} and any(flag in severe_flags for flag in payload["risk_flags"])
        ) or qwen_diff_type in {"critical", "semantic"} or qwen_critical or (large_span_drift and qwen_local_alignment < 0.90)
    )
    payload["auto_accepted"] = not payload["needs_review"]
    payload["candidate_build_mode"] = candidate_build_mode
    payload["support_candidates_skipped"] = bool(skip_support)
    payload["support_candidates_skip_reason"] = "qwen_high_confidence" if skip_support and candidate_build_mode != "qwen-only" else ("qwen_only_mode" if skip_support else "")
    payload["boundary_contamination_suspected"] = "boundary_contamination_suspected" in payload["risk_flags"]
    payload["domain_error_phrase_counts"] = domain_error_counts
    payload["boundary_hint_used_for_boundary_eval"] = bool(boundary_hints)
    payload["boundary_warning"] = False
    payload["boundary_warning_reason"] = []
    payload["qwen"]["search_radius_initial"] = payload["qwen"].get("search_radius", 0)
    payload["qwen"]["search_radius_final"] = payload["qwen"].get("search_radius", 0)
    payload["qwen"]["fallback_expanded"] = False
    payload["qwen"]["fallback_expand_reason"] = []
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
    t_precompute = time.perf_counter()
    sentence_map = {unit.sentence_id: unit for unit in sentence_units}
    block_boundary_hints: dict[str, list[dict[str, Any]]] = {}
    for block in alignment_blocks:
        hints: list[dict[str, Any]] = []
        for sid in block.sentence_ids:
            unit = sentence_map.get(sid)
            if unit is not None:
                hints.extend(list(getattr(unit, "boundary_hints", []) or []))
        block_boundary_hints[block.block_id] = hints
    block_hint_precompute_sec = time.perf_counter() - t_precompute
    stage_secs = defaultdict(float)
    engine_secs = defaultdict(float)
    engine_exec = Counter()
    engine_skip = Counter()
    engine_cache_hit = Counter()
    engine_early_exit = Counter()
    engine_cheap_accept = Counter()
    engine_heavy_skipped = Counter()
    engine_early_exit_ids = defaultdict(list)
    skip_reason_counts = Counter()
    search_profile_counts = Counter()
    candidate_eval_sum = Counter()
    candidate_eval_max = Counter()
    candidate_pruned_sum = Counter()
    candidate_pruned_max = Counter()
    start_candidate_sum = Counter()
    start_candidate_max = Counter()
    end_candidate_sum = Counter()
    end_candidate_max = Counter()
    estimated_grid_sum = Counter()
    estimated_grid_max = Counter()
    qwen_high_confidence_reject_reason_counts = Counter()
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
                block_boundary_hints=block_boundary_hints,
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
            qwen_high_confidence_reject_reason_counts.update(payload.get("qwen_high_confidence_reject_reasons", []) or [])
            for engine in ("qwen", "nemotron", "whisper"):
                cand = payload.get(engine, {})
                if cand.get("skipped"):
                    engine_skip[engine] += 1
                    reason = str(cand.get("skip_reason") or payload.get("support_candidate_skip_reasons", {}).get(engine) or "unknown")
                    skip_reason_counts[reason] += 1
                else:
                    engine_exec[engine] += 1
                    sr = int(cand.get("search_radius", 0) or 0)
                    search_radius_sum[engine] += sr
                    search_radius_max[engine] = max(search_radius_max[engine], sr)
                    if cand.get("early_exit"):
                        engine_early_exit[engine] += 1
                        engine_early_exit_ids[engine].append(payload["block_id"])
                    if cand.get("cheap_span_accept"):
                        engine_cheap_accept[engine] += 1
                    if cand.get("heavy_refinement_skipped"):
                        engine_heavy_skipped[engine] += 1
                    if cand.get("fallback_expanded"):
                        fallback_expanded[engine] += 1
                    search_profile = str(cand.get("refinement_search_profile") or "unknown")
                    search_profile_counts[f"{engine}:{search_profile}"] += 1
                    eval_count = int(cand.get("refinement_candidate_eval_count", 0) or 0)
                    candidate_eval_sum[engine] += eval_count
                    candidate_eval_max[engine] = max(candidate_eval_max[engine], eval_count)
                    pruned_count = int(cand.get("refinement_candidate_pruned_count", 0) or 0)
                    candidate_pruned_sum[engine] += pruned_count
                    candidate_pruned_max[engine] = max(candidate_pruned_max[engine], pruned_count)
                    start_count = int(cand.get("refinement_start_candidate_count", 0) or 0)
                    start_candidate_sum[engine] += start_count
                    start_candidate_max[engine] = max(start_candidate_max[engine], start_count)
                    end_count = int(cand.get("refinement_end_candidate_count", 0) or 0)
                    end_candidate_sum[engine] += end_count
                    end_candidate_max[engine] = max(end_candidate_max[engine], end_count)
                    estimated_grid_count = int(cand.get("refinement_estimated_full_grid_count", 0) or 0)
                    estimated_grid_sum[engine] += estimated_grid_count
                    estimated_grid_max[engine] = max(estimated_grid_max[engine], estimated_grid_count)
            completed += 1
            if completed == 1 or completed % 5 == 0 or completed == len(alignment_blocks):
                print(
                    f"[block] progress episode={episode_id} {completed}/{len(alignment_blocks)} block_id={payload['block_id']}",
                    flush=True,
                )
    total_sec = time.time() - t0
    reject_reason_priority = (
        "qwen_difference_not_safe",
        "qwen_critical_term_disagreement",
        "qwen_numeric_disagreement",
        "qwen_alignment_lt_0_90",
        "qwen_apple_similarity_lt_0_88",
        "qwen_not_usable_for_agreement",
        "preliminary_quality_not_a_or_b",
        "qwen_domain_error",
        "qwen_boundary_contamination",
    )
    qwen_high_confidence_primary_reject_reason_counts = Counter()
    for row in rows:
        reasons = set(row.get("qwen_high_confidence_reject_reasons", []) or [])
        for reason in reject_reason_priority:
            if reason in reasons:
                qwen_high_confidence_primary_reject_reason_counts[reason] += 1
                break
    slowest_blocks = sorted(
        (
            {
                "block_id": row.get("block_id"),
                "block_build_total_sec": round(float(row.get("block_build_total_sec", 0.0) or 0.0), 4),
                "alignment_quality": row.get("alignment_quality"),
                "needs_review": row.get("needs_review"),
                "qwen_local_alignment_score": row.get("qwen", {}).get("local_alignment_score") if isinstance(row.get("qwen"), dict) else None,
                "qwen_apple_difference_type": row.get("qwen_apple_difference_type"),
                "risk_flags": row.get("risk_flags", []),
                "refinement_search_profiles": {
                    engine: row.get(engine, {}).get("refinement_search_profile")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "refinement_candidate_eval_counts": {
                    engine: row.get(engine, {}).get("refinement_candidate_eval_count")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "refinement_candidate_pruned_counts": {
                    engine: row.get(engine, {}).get("refinement_candidate_pruned_count")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "refinement_start_candidate_counts": {
                    engine: row.get(engine, {}).get("refinement_start_candidate_count")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "refinement_end_candidate_counts": {
                    engine: row.get(engine, {}).get("refinement_end_candidate_count")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "refinement_estimated_full_grid_counts": {
                    engine: row.get(engine, {}).get("refinement_estimated_full_grid_count")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
                "stage_sec": {
                    stage_name: round(float(elapsed), 4)
                    for stage_name, elapsed in (row.get("block_build_sec_by_stage") or {}).items()
                    if float(elapsed or 0.0) >= 0.01
                },
            }
            for row in rows
        ),
        key=lambda item: item["block_build_total_sec"],
        reverse=True,
    )[:10]
    slowest_dominant_stage_counts = Counter()
    for row in slowest_blocks:
        stage_sec = row.get("stage_sec", {})
        specific_stage_sec = {
            name: elapsed
            for name, elapsed in stage_sec.items()
            if name not in {"support_candidate_refinement_sec", "support_candidates"}
        }
        if specific_stage_sec:
            slowest_dominant_stage_counts[max(specific_stage_sec, key=specific_stage_sec.get)] += 1
    block_build_time_sum = sum(float(row.get("block_build_total_sec", 0.0) or 0.0) for row in rows)
    slowest_block_time_sum = sum(float(row.get("block_build_total_sec", 0.0) or 0.0) for row in slowest_blocks)
    slow_block_threshold_sec = 30.0
    slow_block_count = sum(1 for row in rows if float(row.get("block_build_total_sec", 0.0) or 0.0) >= slow_block_threshold_sec)

    summary = {
        "episode_id": episode_id,
        "block_candidate_count": len(rows),
        "quality_counts": quality_counts,
        "needs_review_count": sum(1 for row in rows if row["needs_review"]),
        "candidate_build_total_sec": total_sec,
        "candidate_build_wall_sec": total_sec,
        "candidate_build_cumulative_stage_sec": round(sum(stage_secs.values()), 4),
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
            "support_candidate_decision_sec": round(stage_secs["support_candidate_decision_sec"], 4),
            "support_candidate_payload_sec": round(stage_secs["support_candidate_payload_sec"], 4),
            "support_candidate_skip_sec": round(stage_secs["support_candidate_skip_sec"], 4),
            "support_candidate_refinement_sec": round(stage_secs["support_candidate_refinement_sec"], 4),
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
        "candidate_refinement_skipped_reason_counts": dict(skip_reason_counts),
        "refinement_search_profile_counts": dict(search_profile_counts),
        "refinement_candidate_eval_sum_by_engine": dict(candidate_eval_sum),
        "refinement_candidate_eval_max_by_engine": dict(candidate_eval_max),
        "refinement_candidate_pruned_sum_by_engine": dict(candidate_pruned_sum),
        "refinement_candidate_pruned_max_by_engine": dict(candidate_pruned_max),
        "refinement_start_candidate_sum_by_engine": dict(start_candidate_sum),
        "refinement_start_candidate_max_by_engine": dict(start_candidate_max),
        "refinement_end_candidate_sum_by_engine": dict(end_candidate_sum),
        "refinement_end_candidate_max_by_engine": dict(end_candidate_max),
        "refinement_estimated_full_grid_sum_by_engine": dict(estimated_grid_sum),
        "refinement_estimated_full_grid_max_by_engine": dict(estimated_grid_max),
        "qwen_high_confidence_reject_reason_counts": dict(qwen_high_confidence_reject_reason_counts),
        "qwen_high_confidence_primary_reject_reason_counts": dict(qwen_high_confidence_primary_reject_reason_counts),
        "candidate_refinement_cache_hit_count_by_engine": dict(engine_cache_hit),
        "candidate_refinement_early_exit_count_by_engine": dict(engine_early_exit),
        "candidate_refinement_early_exit_block_ids_by_engine": dict(engine_early_exit_ids),
        "cheap_span_accept_count_by_engine": dict(engine_cheap_accept),
        "heavy_refinement_skipped_count_by_engine": dict(engine_heavy_skipped),
        "fallback_expanded_count_by_engine": dict(fallback_expanded),
        "average_search_radius_by_engine": {
            engine: round(search_radius_sum[engine] / max(engine_exec[engine], 1), 2) for engine in ("qwen", "nemotron", "whisper")
        },
        "max_search_radius_by_engine": dict(search_radius_max),
        "candidate_build_mode": candidate_build_mode,
        "support_global_alignment_skip_threshold": SUPPORT_GLOBAL_ALIGNMENT_SKIP_THRESHOLD,
        "candidate_build_slowest_blocks": slowest_blocks,
        "candidate_build_slowest_block_ids": [row["block_id"] for row in slowest_blocks],
        "candidate_build_slowest_dominant_stage_counts": dict(slowest_dominant_stage_counts),
        "candidate_build_block_time_sum": round(block_build_time_sum, 4),
        "candidate_build_slowest_block_time_sum": round(slowest_block_time_sum, 4),
        "candidate_build_slowest_block_time_ratio": round(slowest_block_time_sum / max(block_build_time_sum, 1e-9), 4),
        "candidate_build_slow_block_threshold_sec": slow_block_threshold_sec,
        "candidate_build_slow_block_count": slow_block_count,
        "workers": max_workers,
        "boundary_hint_build_total_sec": build_stats["boundary_hint_build_total_sec"],
        "boundary_hint_build_count_by_source": dict(build_stats["boundary_hint_build_count_by_source"]),
        "boundary_hint_cache_hit_count_by_source": dict(build_stats["boundary_hint_cache_hit_count_by_source"]),
        "boundary_hint_cache_miss_count_by_source": dict(build_stats["boundary_hint_cache_miss_count_by_source"]),
        "candidate_build_executor_kind": executor_kind,
        "parallel_enabled": not no_parallel,
        "block_boundary_hint_precompute_sec": round(block_hint_precompute_sec, 4),
        "block_boundary_hint_precompute_enabled": True,
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
