from __future__ import annotations

import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        LLMDecision,
        choose_best_candidate,
        ensure_dir,
        merge_review_reasons,
        normalize_source_choice,
        save_jsonl,
        segment_similarity_score,
        sequence_similarity,
    )
    from tools.asr_alignment.conversation_punctuation_normalizer import normalize_conversation_punctuation  # type: ignore
else:
    from ._core import (
        LLMDecision,
        choose_best_candidate,
        ensure_dir,
        merge_review_reasons,
        normalize_source_choice,
        save_jsonl,
        segment_similarity_score,
        sequence_similarity,
    )
    from .conversation_punctuation_normalizer import normalize_conversation_punctuation


def _combined_score(candidate: dict[str, Any], apple_segment: dict[str, Any], source: str) -> float:
    alignment = float(candidate.get("local_alignment_score", candidate.get("alignment_score", 0.0)))
    agreement = float(apple_segment.get("candidate_agreement_score", 0.0))
    apple_stability = float(apple_segment.get("apple", {}).get("stability_score", 0.0))
    source_bonus = {"qwen": 0.08, "apple": 0.05, "nemotron": 0.02, "whisper": 0.0}.get(source, 0.0)
    usable_bonus = 0.04 if candidate.get("usable_for_agreement") else -0.05
    risk_penalty = 0.0
    candidate_text = str(candidate.get("text", "") or "")
    qwen_similarity = float(apple_segment.get("qwen_apple_similarity", 0.0) or 0.0) if source == "qwen" else 0.0
    if source != "apple":
        boundary_suspected = bool(candidate.get("boundary_contamination") or _has_boundary_contamination_suspect(candidate_text))
        if source != "qwen" and alignment < 0.70:
            risk_penalty += 0.18
        elif source != "qwen" and alignment < 0.82:
            risk_penalty += 0.08
        elif source == "qwen" and alignment < 0.70 and qwen_similarity < 0.75:
            risk_penalty += 0.18
        elif source == "qwen" and alignment < 0.82 and qwen_similarity < 0.75:
            risk_penalty += 0.08
        if boundary_suspected:
            risk_penalty += 0.10
        if candidate.get("span_too_long") or candidate.get("span_too_short"):
            risk_penalty += 0.06
    if source == "qwen":
        diff_type = str(apple_segment.get("qwen_apple_difference_type") or "")
        if diff_type in {"critical", "semantic"} and qwen_similarity < 0.50:
            risk_penalty += 0.20
        elif diff_type in {"critical", "semantic"} and qwen_similarity < 0.75:
            risk_penalty += 0.08
    score = 0.56 * alignment + 0.24 * agreement + 0.12 * apple_stability + source_bonus + usable_bonus - risk_penalty
    return max(0.0, min(1.0, score))


def _best_source(candidate_rows: dict[str, dict[str, Any]], apple_segment: dict[str, Any]) -> str:
    scored = {}
    for source, row in candidate_rows.items():
        scored[source] = dict(row)
        scored[source]["combined_score"] = _combined_score(row, apple_segment, source)
    return choose_best_candidate(scored)


def _deterministic_selection(candidate_rows: dict[str, dict[str, Any]], apple_segment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    best_source = _best_source(candidate_rows, apple_segment)
    selected = candidate_rows[best_source]
    selected_text = selected.get("text", "")
    return best_source, {
        "selected_source": best_source,
        "final_text": selected_text,
        "confidence": float(selected.get("combined_score", selected.get("alignment_score", 0.0))),
        "selection_method": "deterministic_ranked",
    }


def _maybe_clamp_time(start: float, end: float, previous_end: float | None) -> tuple[float, float]:
    if previous_end is None:
        return start, end
    start = max(start, previous_end)
    if end <= start:
        end = start + 0.1
    return start, end


def _normalize_review_reason(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _sentence_display_text(text: str, *, short_response_period: bool = True) -> str:
    normalized = _normalize_display_text(text, short_response_period=short_response_period)
    return str(normalized.get("display_text", text))


_PROTECTED_PREFIXES = (
    "さあ",
    "それ",
    "そう",
    "うん",
    "いや",
    "なるほど",
    "という",
    "というのも",
    "それに対して",
    "これ",
    "だから",
    "で、",
    "はい",
)
_LEADING_FRAGMENT_RE = re.compile(r"^(?:[ぁ-ゖァ-ヶ]{1,2}|[一-龯]{1,2}|[ぁ-ゖァ-ヶ]{1,2}。|[一-龯]{1,2}。|ね。|よ。|うん。|あ。|え。|はい。|そう。|そうす|ですよ。|ですね、)$")
_TRAILING_FRAGMENT_RE = re.compile(r"(?:[ぁ-ゖァ-ヶ]{1,2}|[一-龯]{1,2}|[ぁ-ゖァ-ヶ]{1,2}。|[一-龯]{1,2}。|建|実|な|ポイ|入れ|そうす)$")
_LEAN_VALIDATION_PREFIXES = ("、", "。", "に対して", "を言うと", "の陣", "ほど。", "とです", "うのも", "は家")
_LEAN_VALIDATION_SUFFIXES = ("ポイ", "簡", "入れ", "そ", "あ", "てる", "中")
_BOUNDARY_CONTAMINATION_SUSPECT_PREFIXES = ("ね。", "うことです", "のだから", "ですよね。", "したね。", "ということです")
_BOUNDARY_CONTAMINATION_SUSPECT_SUFFIXES = ("あるん", "位置づ")
_DOMAIN_ERROR_PHRASES = ("排水の陣", "各家族", "社会行動", "整形立てて")
_DOMAIN_TEXT_REPLACEMENTS = (
    ("排水の陣", "背水の陣"),
    ("各家族", "核家族"),
    ("社会行動", "社会構造"),
    ("整形立て", "生計立て"),
    ("DMA", "DNA"),
    ("ＤＭＡ", "DNA"),
    ("大加速性", "大家族性"),
    ("太路", "退路"),
    ("退路立った", "退路を断った"),
    ("ワガコ", "我が子"),
    ("食べていけれる", "食べていける"),
    ("四十五十", "四十、五十"),
    ("奪わって", "奪って"),
    ("狭巻", "狭間"),
    ("山さん", "矢野さん"),
    ("大事辞典", "大事典"),
    ("そそれ", "それ"),
    ("今今の日本", "今の日本"),
    ("お伝えましてありました", "お伝えしてありました"),
)
_TRAILING_FRAGMENT_SUFFIXES = ("、よ", "。聞", "、し", "。し", "でそうす", "とこ", "依")


def _has_boundary_contamination_suspect(text: str) -> bool:
    stripped = str(text or "").strip()
    return (
        any(stripped.startswith(prefix) for prefix in _BOUNDARY_CONTAMINATION_SUSPECT_PREFIXES)
        or any(stripped.endswith(suffix) for suffix in _BOUNDARY_CONTAMINATION_SUSPECT_SUFFIXES)
    )


def _contains_domain_error(text: str) -> bool:
    return any(phrase in str(text or "") for phrase in _DOMAIN_ERROR_PHRASES)


def _apply_domain_text_corrections(text: str) -> tuple[str, list[dict[str, str]]]:
    corrected = str(text or "")
    applied: list[dict[str, str]] = []
    for wrong, right in _DOMAIN_TEXT_REPLACEMENTS:
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
            applied.append({"from": wrong, "to": right})
    return corrected, applied


def _without_leading_fragment_suggestion(text: str, apple_text: str) -> tuple[str, list[str]]:
    original = str(text or "")
    apple_text = str(apple_text or "")
    suggested = original
    reasons: list[str] = []
    for prefix in ("ね。", "よ。", "よ、"):
        if suggested.startswith(prefix) and not apple_text.startswith(prefix):
            candidate = suggested[len(prefix):].lstrip()
            if len(candidate) >= 3 and not any(candidate.startswith(bad) for bad in _LEAN_VALIDATION_PREFIXES):
                suggested = candidate
                reasons.append("leading_boundary_fragment_suggested")
            break
    for suffix in _TRAILING_FRAGMENT_SUFFIXES:
        if suggested.endswith(suffix):
            candidate = suggested[: -len(suffix)].rstrip()
            if len(candidate) >= 3 and not candidate.endswith(_LEAN_VALIDATION_SUFFIXES):
                suggested = candidate
                reasons.append("trailing_boundary_fragment_suggested")
            break
    return suggested, reasons


def _domain_preferred_source(
    candidate_rows: dict[str, dict[str, Any]],
    selected_source: str,
    selected_text: str,
) -> tuple[str, str, list[str]]:
    if not _contains_domain_error(selected_text):
        return selected_source, selected_text, []
    selected_len = max(len(selected_text), 1)
    scored: list[tuple[float, str, str]] = []
    for source in ("qwen", "nemotron", "apple", "whisper"):
        row = candidate_rows.get(source, {})
        candidate_text = str(row.get("text", "") if isinstance(row, dict) else "")
        if not candidate_text or _contains_domain_error(candidate_text):
            continue
        length_ratio = len(candidate_text) / selected_len
        if not 0.65 <= length_ratio <= 1.45:
            continue
        similarity = segment_similarity_score(selected_text, candidate_text)
        local_score = float(row.get("local_alignment_score", row.get("alignment_score", 0.0)) or 0.0) if isinstance(row, dict) else 0.0
        if similarity >= 0.78 and local_score >= 0.70:
            scored.append((0.75 * similarity + 0.25 * local_score, source, candidate_text))
    if not scored:
        return selected_source, selected_text, []
    scored.sort(reverse=True)
    _, source, text = scored[0]
    return source, text, ["domain_error_avoided_by_candidate_switch"]


def _classify_large_span_drift(risk_flags: list[str], qwen_alignment: float) -> list[str]:
    if "large_span_drift" not in risk_flags:
        return risk_flags
    if qwen_alignment < 0.90 or "span_too_long" in risk_flags:
        return risk_flags
    return ["large_span_drift_warning" if flag == "large_span_drift" else flag for flag in risk_flags]


def _deterministic_needs_review(segment: dict[str, Any], risk_flags: list[str], *, cleanup_needs_review: bool) -> bool:
    diff_type = str(segment.get("qwen_apple_difference_type") or "semantic")
    qwen = segment.get("qwen", {}) if isinstance(segment.get("qwen"), dict) else {}
    qwen_alignment = float(qwen.get("local_alignment_score", qwen.get("alignment_score", 0.0)) or 0.0)
    qwen_similarity = float(segment.get("qwen_apple_similarity", 0.0) or 0.0)
    quality = str(segment.get("alignment_quality") or "E")
    severe_flags = {
        "critical_term_disagreement",
        "numeric_disagreement",
        "boundary_contamination_suspected",
        "domain_error_phrase",
        "span_too_long",
    }
    if diff_type in {"critical", "semantic"}:
        return True
    if quality == "E":
        return True
    if qwen_alignment < 0.82:
        return True
    if any(flag in risk_flags for flag in severe_flags):
        return True
    if "large_span_drift" in risk_flags and qwen_alignment < 0.90:
        return True
    if cleanup_needs_review and any(flag in risk_flags for flag in {"boundary_contamination_suspected", "span_too_long"}):
        return True
    if (
        quality in {"A", "B"}
        and qwen_alignment >= 0.82
        and qwen_similarity >= 0.88
        and diff_type in {"none", "surface", "soft_domain"}
    ):
        return False
    return bool(segment.get("needs_review"))


def _review_reasons_for_flags(risk_flags: list[str]) -> list[str]:
    ignored_when_auto_safe = {"surface_difference", "boundary_cleanup_needed", "large_span_drift_warning", "domain_error_avoided"}
    return [flag for flag in risk_flags if flag not in ignored_when_auto_safe]


def _cleanup_boundary_fragments(
    final_text: str,
    apple_text: str,
    selected_source: str,
    boundary_hints: list[dict[str, Any]] | None = None,
    previous_final_text: str = "",
    previous_selected_text: str = "",
    next_final_text: str = "",
    next_selected_text: str = "",
) -> dict[str, Any]:
    original_text = str(final_text or "")
    apple_text = str(apple_text or "")
    selected_source = str(selected_source or "")
    previous_final_text = str(previous_final_text or "")
    previous_selected_text = str(previous_selected_text or "")
    next_final_text = str(next_final_text or "")
    next_selected_text = str(next_selected_text or "")
    boundary_hints = list(boundary_hints or [])
    result_text = original_text
    reasons: list[str] = []
    applied = False
    attempted = False
    protected_prefix_prevented = False
    reverted = False

    if not result_text or selected_source == "apple":
        return {
            "final_text_before_cleanup": original_text,
            "final_text_after_cleanup": result_text,
            "boundary_cleanup_applied": False,
            "boundary_cleanup_reason": [],
            "boundary_cleanup_attempted": False,
            "boundary_cleanup_reverted": False,
            "cleanup_validation_failed": False,
            "protected_prefix_prevented_cleanup": False,
        }

    def _startswith_protected_prefix(text: str) -> bool:
        return any(text.startswith(prefix) for prefix in _PROTECTED_PREFIXES)

    def _trim_left_if_overlapping(text: str) -> tuple[str, str] | None:
        nonlocal attempted, protected_prefix_prevented
        if _startswith_protected_prefix(text):
            protected_prefix_prevented = True
            return None
        if len(text) < 3:
            return None
        max_cut = min(6, len(text) - 1)
        for cut in range(1, max_cut + 1):
            fragment = text[:cut]
            trimmed = text[cut:]
            if len(trimmed) < 3:
                continue
            if _startswith_protected_prefix(trimmed):
                protected_prefix_prevented = True
                continue
            overlap_ok = False
            hint_ok = any(hint.get("type") == "short_response_period" for hint in boundary_hints)
            if previous_final_text.endswith(fragment) or previous_selected_text.endswith(fragment):
                overlap_ok = True
            elif hint_ok and _LEADING_FRAGMENT_RE.match(fragment) and len(fragment) <= 3:
                overlap_ok = True
            if overlap_ok:
                attempted = True
                return trimmed, "leading_fragment_removed"
        return None

    def _trim_right_if_overlapping(text: str) -> tuple[str, str] | None:
        nonlocal attempted, protected_prefix_prevented
        if len(text) < 3:
            return None
        max_cut = min(6, len(text) - 1)
        for cut in range(1, max_cut + 1):
            fragment = text[-cut:]
            trimmed = text[:-cut]
            if len(trimmed) < 3:
                continue
            overlap_ok = False
            hint_ok = any(hint.get("type") == "short_response_period" for hint in boundary_hints)
            if next_final_text.startswith(fragment) or next_selected_text.startswith(fragment):
                overlap_ok = True
            elif hint_ok and _TRAILING_FRAGMENT_RE.search(fragment) and len(fragment) <= 3:
                overlap_ok = True
            if overlap_ok:
                attempted = True
                return trimmed, "trailing_fragment_removed"
        return None

    leading = _trim_left_if_overlapping(result_text)
    if leading is not None:
        result_text, reason = leading
        reasons.append(reason)
        applied = True
    trailing = _trim_right_if_overlapping(result_text)
    if trailing is not None:
        result_text, reason = trailing
        reasons.append(reason)
        applied = True

    if not result_text:
        result_text = original_text
        reasons = []
        applied = False
        reverted = True

    validation_failed = False
    if result_text:
        if any(result_text.startswith(prefix) for prefix in _LEAN_VALIDATION_PREFIXES):
            validation_failed = True
        if any(result_text.endswith(suffix) for suffix in _LEAN_VALIDATION_SUFFIXES):
            validation_failed = True
        if _LEADING_FRAGMENT_RE.match(result_text[:4]):
            validation_failed = True
        if _TRAILING_FRAGMENT_RE.search(result_text[-4:]):
            validation_failed = True

    if validation_failed:
        result_text = original_text
        reasons = []
        applied = False
        reverted = True

    return {
        "final_text_before_cleanup": original_text,
        "final_text_after_cleanup": result_text,
        "boundary_cleanup_applied": applied,
        "boundary_cleanup_reason": reasons,
        "boundary_cleanup_attempted": attempted,
        "boundary_cleanup_reverted": reverted,
        "cleanup_validation_failed": validation_failed,
        "protected_prefix_prevented_cleanup": protected_prefix_prevented,
        "boundary_cleanup_needs_review": validation_failed,
    }


def _normalize_display_text(text: str, *, short_response_period: bool = True) -> dict[str, Any]:
    result = normalize_conversation_punctuation(text, short_response_period=short_response_period)
    return {
        "display_text": result.display_text,
        "punctuation_hints": result.punctuation_hints,
        "punctuation_inserted_period_count": result.inserted_period_count,
        "punctuation_inserted_comma_count": result.inserted_comma_count,
        "short_response_period_count": result.short_response_period_count,
        "possible_speaker_change_period_count": result.possible_speaker_change_period_count,
        "boundary_hint_used_count": result.boundary_hint_used_count,
    }


def _select_display_source_text(candidate_rows: dict[str, dict[str, Any]], selected_source: str, apple_display_text: str) -> str:
    row = candidate_rows.get(selected_source, {})
    if isinstance(row, dict):
        for key in ("display_text", "boundary_text", "raw_text", "text"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if apple_display_text.strip():
        return apple_display_text
    return candidate_rows.get("apple", {}).get("text", "")


def select_final_candidates(
    *,
    episode_id: str,
    block_rows: list[dict[str, Any]],
    sentence_units: list[Any],
    llm_client: Any | None = None,
    use_llm: bool = False,
    llm_only_risky: bool = False,
    llm_max_segments: int = 200,
    output_dir: Path | None = None,
    conversation_punctuation: bool = False,
    short_response_period: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    final_blocks: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    stats = {
        "llm_used": False,
        "llm_call_count": 0,
        "llm_success_count": 0,
        "llm_failure_count": 0,
        "llm_cache_hit_count": 0,
        "llm_changed_final_text_count": 0,
        "llm_changed_needs_review_true_count": 0,
        "llm_changed_needs_review_false_count": 0,
        "punctuation_normalized_count": 0,
        "punctuation_inserted_period_count": 0,
        "punctuation_inserted_comma_count": 0,
        "short_response_period_count": 0,
        "possible_speaker_change_period_count": 0,
        "boundary_hint_used_count": 0,
        "cleanup_reverted_by_punctuation_hint_count": 0,
        "domain_candidate_switch_count": 0,
        "domain_error_avoided_count": 0,
        "domain_text_correction_count": 0,
        "domain_text_corrected_block_count": 0,
        "suggested_final_text_count": 0,
        "boundary_suggestion_count": 0,
        "trailing_boundary_suggestion_count": 0,
        "large_span_drift_warning_count": 0,
    }
    previous_end: float | None = None
    risky_calls = 0
    unit_map = {u.sentence_id: u for u in sentence_units}
    for idx, segment in enumerate(block_rows):
        candidate_rows = {source: segment[source] for source in ("qwen", "apple", "nemotron", "whisper")}
        best_source, deterministic = _deterministic_selection(candidate_rows, segment)
        selected_source = deterministic["selected_source"]
        final_text_raw = deterministic["final_text"]
        final_text = final_text_raw
        confidence = deterministic["confidence"]
        selection_method = deterministic["selection_method"]
        needs_review = bool(segment.get("needs_review"))
        review_reason = list(segment.get("risk_flags") or [])
        llm_decision: LLMDecision | None = None
        should_call = False
        if use_llm and llm_client is not None:
            should_call = llm_client.should_call(segment, only_risky=llm_only_risky)
            if should_call and risky_calls >= llm_max_segments:
                should_call = False
        if should_call and llm_client is not None:
            stats["llm_used"] = True
            stats["llm_call_count"] += 1
            risky_calls += 1
            print(
                f"[llm] start episode={episode_id} block_id={segment.get('block_id')} "
                f"segment_id={segment.get('segment_id')} quality={segment.get('alignment_quality')} "
                f"diff={segment.get('qwen_apple_difference_type')} flags={segment.get('risk_flags', [])}",
                flush=True,
            )
            t0 = time.time()
            llm_decision = llm_client.choose(segment, episode_id=episode_id)
            dt = time.time() - t0
            print(
                f"[llm] end episode={episode_id} block_id={segment.get('block_id')} "
                f"segment_id={segment.get('segment_id')} success={llm_decision.success} "
                f"cached={llm_decision.cached} seconds={dt:.2f} "
                f"error={llm_decision.error or ''}",
                flush=True,
            )
            if llm_decision.cached:
                stats["llm_cache_hit_count"] += 1
            if llm_decision.success:
                stats["llm_success_count"] += 1
                llm_source = normalize_source_choice(llm_decision.selected_source)
                if llm_source in candidate_rows:
                    selected_source = llm_source
                    final_text_raw = candidate_rows[llm_source].get("text", final_text_raw)
                    final_text = final_text_raw
                    confidence = llm_decision.confidence if llm_decision.confidence is not None else confidence
                    selection_method = "llm_candidate_selection"
                if llm_decision.final_text:
                    llm_text = str(llm_decision.final_text).strip()
                    if llm_text and segment_similarity_score(llm_text, final_text) < 0.95:
                        stats["llm_changed_final_text_count"] += 1
                    if llm_text:
                        final_text_raw = llm_text
                        final_text = llm_text
                if llm_decision.needs_review is not None:
                    if llm_decision.needs_review and not needs_review:
                        stats["llm_changed_needs_review_true_count"] += 1
                    if not llm_decision.needs_review and needs_review:
                        stats["llm_changed_needs_review_false_count"] += 1
                    needs_review = bool(llm_decision.needs_review)
                review_reason = merge_review_reasons(review_reason, _normalize_review_reason(llm_decision.review_reason), [llm_decision.notes] if llm_decision.notes else [])
            else:
                stats["llm_failure_count"] += 1
                failure_reason = llm_decision.error or "llm_failed"
                review_reason = merge_review_reasons(review_reason, [failure_reason])

        candidate_summary = {source: row.get("text", "") for source, row in candidate_rows.items()}
        preferred_source, preferred_text, preferred_reasons = _domain_preferred_source(candidate_rows, selected_source, final_text)
        domain_candidate_switched = False
        if preferred_source != selected_source:
            selected_source = preferred_source
            final_text_raw = preferred_text
            final_text = preferred_text
            selection_method = "deterministic_domain_error_avoidance"
            review_reason = merge_review_reasons(review_reason, preferred_reasons)
            domain_candidate_switched = True
            stats["domain_candidate_switch_count"] += 1
        segment_units = [unit_map[sid] for sid in segment["sentence_ids"] if sid in unit_map]
        segment_boundary_hints: list[dict[str, Any]] = []
        for unit in segment_units:
            segment_boundary_hints.extend(list(getattr(unit, "boundary_hints", []) or []))
        previous_final_text = final_blocks[-1]["final_text_after_cleanup"] if final_blocks else ""
        previous_selected_text = final_blocks[-1]["candidate_summary"].get(final_blocks[-1]["selected_source"], "") if final_blocks else ""
        next_segment = block_rows[idx + 1] if idx + 1 < len(block_rows) else None
        next_selected_text = ""
        next_final_text = ""
        if next_segment is not None:
            next_candidate_rows = {source: next_segment[source] for source in ("qwen", "apple", "nemotron", "whisper")}
            next_source, _ = _deterministic_selection(next_candidate_rows, next_segment)
            next_selected_text = next_candidate_rows.get(next_source, {}).get("text", "")
            next_final_text = next_selected_text
        cleanup = _cleanup_boundary_fragments(
            final_text,
            candidate_summary.get("apple", ""),
            selected_source,
            boundary_hints=segment_boundary_hints,
            previous_final_text=previous_final_text,
            previous_selected_text=previous_selected_text,
            next_final_text=next_final_text,
            next_selected_text=next_selected_text,
        )
        final_text_before_cleanup = cleanup["final_text_before_cleanup"]
        final_text = cleanup["final_text_after_cleanup"]
        if cleanup["boundary_cleanup_applied"]:
            review_reason = merge_review_reasons(review_reason, cleanup["boundary_cleanup_reason"])
        cleanup_needs_review = bool(cleanup.get("boundary_cleanup_needs_review"))
        final_text_before_domain_correction = final_text
        final_text, domain_text_corrections = _apply_domain_text_corrections(final_text)
        domain_text_corrected = bool(domain_text_corrections)
        if domain_text_corrected:
            stats["domain_text_corrected_block_count"] += 1
            stats["domain_text_correction_count"] += len(domain_text_corrections)
            review_reason = merge_review_reasons(review_reason, ["domain_text_corrected"])

        display_source_text = final_text if domain_text_corrected else _select_display_source_text(candidate_rows, selected_source, candidate_summary.get("apple", ""))
        punctuation = {
            "display_text": display_source_text,
            "punctuation_hints": [],
            "punctuation_inserted_period_count": 0,
            "punctuation_inserted_comma_count": 0,
            "short_response_period_count": 0,
            "possible_speaker_change_period_count": 0,
            "boundary_hint_used_count": 0,
        }
        if conversation_punctuation:
            punctuation = _normalize_display_text(display_source_text, short_response_period=short_response_period)
            if punctuation["punctuation_hints"]:
                stats["punctuation_normalized_count"] += 1
                stats["punctuation_inserted_period_count"] += punctuation["punctuation_inserted_period_count"]
                stats["punctuation_inserted_comma_count"] += punctuation["punctuation_inserted_comma_count"]
                stats["short_response_period_count"] += punctuation["short_response_period_count"]
                stats["possible_speaker_change_period_count"] += punctuation["possible_speaker_change_period_count"]
                stats["boundary_hint_used_count"] += punctuation["boundary_hint_used_count"]
                if cleanup["boundary_cleanup_applied"] and punctuation["punctuation_hints"]:
                    stats["cleanup_reverted_by_punctuation_hint_count"] += 0

        final_risk_flags = list(segment.get("risk_flags", []) or [])
        qwen_for_review = segment.get("qwen", {}) if isinstance(segment.get("qwen"), dict) else {}
        qwen_alignment_for_review = float(qwen_for_review.get("local_alignment_score", qwen_for_review.get("alignment_score", 0.0)) or 0.0)
        final_risk_flags = _classify_large_span_drift(final_risk_flags, qwen_alignment_for_review)
        if "domain_error_phrase" in final_risk_flags and not _contains_domain_error(final_text) and not _contains_domain_error(punctuation["display_text"]):
            final_risk_flags = [flag for flag in final_risk_flags if flag != "domain_error_phrase"]
            final_risk_flags.append("domain_error_avoided")
            stats["domain_error_avoided_count"] += 1
        if domain_text_corrected and "domain_text_corrected" not in final_risk_flags:
            final_risk_flags.append("domain_text_corrected")
        if "large_span_drift_warning" in final_risk_flags:
            stats["large_span_drift_warning_count"] += 1
        if _has_boundary_contamination_suspect(final_text) or _has_boundary_contamination_suspect(punctuation["display_text"]):
            needs_review = True
            review_reason = merge_review_reasons(review_reason, ["boundary_contamination_suspected"])
            if "boundary_contamination_suspected" not in final_risk_flags:
                final_risk_flags.append("boundary_contamination_suspected")
        if cleanup_needs_review and "boundary_cleanup_needed" not in final_risk_flags:
            final_risk_flags.append("boundary_cleanup_needed")
        llm_explicit_review = llm_decision is not None and llm_decision.needs_review is not None
        if not llm_explicit_review:
            needs_review = _deterministic_needs_review(segment, final_risk_flags, cleanup_needs_review=cleanup_needs_review)
            review_reason = _review_reasons_for_flags(final_risk_flags) if needs_review else []
        suggested_final_text, suggestion_reasons = _without_leading_fragment_suggestion(final_text, candidate_summary.get("apple", ""))
        if suggested_final_text != final_text:
            stats["suggested_final_text_count"] += 1
            if "leading_boundary_fragment_suggested" in suggestion_reasons:
                stats["boundary_suggestion_count"] += 1
            if "trailing_boundary_fragment_suggested" in suggestion_reasons:
                stats["trailing_boundary_suggestion_count"] += 1
            if "boundary_suggestion_available" not in final_risk_flags:
                final_risk_flags.append("boundary_suggestion_available")
            needs_review = True
            review_reason = merge_review_reasons(review_reason, ["boundary_suggestion_available"])
        elif domain_candidate_switched:
            suggested_final_text = final_text
            suggestion_reasons = preferred_reasons
            stats["suggested_final_text_count"] += 1

        segment_time = dict(segment["time"])
        segment_time["start_sec"], segment_time["end_sec"] = _maybe_clamp_time(
            float(segment_time["start_sec"]),
            float(segment_time["end_sec"]),
            previous_end,
        )
        previous_end = segment_time["end_sec"]
        block_final = {
            "episode_id": episode_id,
            "block_id": segment["block_id"],
            "sentence_ids": segment["sentence_ids"],
            "time": segment_time,
            "final_text": final_text,
            "final_text_raw": final_text_before_cleanup,
            "final_text_after_boundary_cleanup": final_text,
            "final_text_display": punctuation["display_text"],
            "selected_source": selected_source,
            "selection_method": selection_method,
            "confidence": float(confidence),
            "needs_review": needs_review,
            "review_reason": review_reason,
            "risk_flags": final_risk_flags,
            "suggested_final_text": suggested_final_text,
            "suggested_final_text_reason": suggestion_reasons,
            "domain_candidate_switched": domain_candidate_switched,
            "domain_text_corrected": domain_text_corrected,
            "domain_text_corrections": domain_text_corrections,
            "candidate_summary": candidate_summary,
            "apple_display_text": candidate_summary.get("apple", ""),
            "apple_boundary_hints": segment_boundary_hints,
            "punctuation_hint_applied": bool(punctuation["punctuation_hints"]),
            "punctuation_hints": punctuation["punctuation_hints"],
            "punctuation_inserted_period_count": punctuation["punctuation_inserted_period_count"],
            "punctuation_inserted_comma_count": punctuation["punctuation_inserted_comma_count"],
            "short_response_period_count": punctuation["short_response_period_count"],
            "possible_speaker_change_period_count": punctuation["possible_speaker_change_period_count"],
            "boundary_hint_used_count": punctuation["boundary_hint_used_count"],
            "final_text_before_cleanup": final_text_before_cleanup,
            "final_text_after_cleanup": final_text,
            "final_text_before_domain_correction": final_text_before_domain_correction,
            "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
            "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
            "boundary_cleanup_attempted": bool(cleanup.get("boundary_cleanup_attempted")),
            "boundary_cleanup_reverted": bool(cleanup.get("boundary_cleanup_reverted")),
            "cleanup_validation_failed": bool(cleanup.get("cleanup_validation_failed")),
            "protected_prefix_prevented_cleanup": bool(cleanup.get("protected_prefix_prevented_cleanup")),
            "llm_used": bool(should_call),
            "llm_cached": bool(llm_decision.cached) if llm_decision is not None else False,
            "selection_notes": llm_decision.notes if llm_decision is not None else "",
            "llm_error": llm_decision.error if llm_decision is not None else "",
        }
        final_blocks.append(block_final)
        for sentence_id in segment["sentence_ids"]:
            unit = unit_map[sentence_id]
            sentence_display_text = _sentence_display_text(unit.text, short_response_period=short_response_period)
            final_rows.append(
                {
                    "episode_id": episode_id,
                    "sentence_id": sentence_id,
                    "block_id": segment["block_id"],
                    "sentence_ids": segment["sentence_ids"],
                    "time": {"start_sec": unit.start_sec, "end_sec": unit.end_sec},
                    "apple_text": unit.text,
                    "final_text": final_text,
                    "final_text_raw": final_text_before_cleanup,
                    "final_text_display": punctuation["display_text"],
                    "sentence_display_text": sentence_display_text,
                    "punctuation_hints": punctuation["punctuation_hints"],
                    "selected_source": selected_source,
                    "selection_method": selection_method,
                    "confidence": float(confidence),
                    "needs_review": needs_review,
                    "review_reason": review_reason,
                    "risk_flags": final_risk_flags,
                    "suggested_final_text": suggested_final_text,
                    "suggested_final_text_reason": suggestion_reasons,
                    "domain_candidate_switched": domain_candidate_switched,
                    "domain_text_corrected": domain_text_corrected,
                    "domain_text_corrections": domain_text_corrections,
                    "candidate_summary": candidate_summary,
                    "apple_display_text": candidate_summary.get("apple", ""),
                    "apple_boundary_hints": segment_boundary_hints,
                    "punctuation_hint_applied": bool(punctuation["punctuation_hints"]),
                    "punctuation_inserted_period_count": punctuation["punctuation_inserted_period_count"],
                    "punctuation_inserted_comma_count": punctuation["punctuation_inserted_comma_count"],
                    "short_response_period_count": punctuation["short_response_period_count"],
                    "possible_speaker_change_period_count": punctuation["possible_speaker_change_period_count"],
                    "boundary_hint_used_count": punctuation["boundary_hint_used_count"],
                    "final_text_before_cleanup": final_text_before_cleanup,
                    "final_text_after_cleanup": final_text,
                    "final_text_before_domain_correction": final_text_before_domain_correction,
                    "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
                    "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
                    "boundary_cleanup_attempted": bool(cleanup.get("boundary_cleanup_attempted")),
                    "boundary_cleanup_reverted": bool(cleanup.get("boundary_cleanup_reverted")),
                    "cleanup_validation_failed": bool(cleanup.get("cleanup_validation_failed")),
                    "protected_prefix_prevented_cleanup": bool(cleanup.get("protected_prefix_prevented_cleanup")),
                    "llm_used": bool(should_call),
                    "llm_cached": bool(llm_decision.cached) if llm_decision is not None else False,
                    "selection_notes": llm_decision.notes if llm_decision is not None else "",
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                }
            )
        if needs_review:
            review_rows.append(
                {
                    "episode_id": episode_id,
                    "block_id": segment["block_id"],
                    "sentence_ids": segment["sentence_ids"],
                    "time": segment_time,
                    "needs_review": needs_review,
                    "review_reason": review_reason,
                    "alignment_quality": segment.get("alignment_quality"),
                    "risk_flags": final_risk_flags,
                    "candidate_summary": candidate_summary,
                    "final_text": final_text,
                    "suggested_final_text": suggested_final_text,
                    "suggested_final_text_reason": suggestion_reasons,
                    "domain_candidate_switched": domain_candidate_switched,
                    "domain_text_corrected": domain_text_corrected,
                    "domain_text_corrections": domain_text_corrections,
                    "selected_source": selected_source,
                    "confidence": float(confidence),
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                    "apple_display_text": candidate_summary.get("apple", ""),
                    "apple_boundary_hints": segment_boundary_hints,
                    "final_text_raw": final_text_before_cleanup,
                    "final_text_display": punctuation["display_text"],
                    "punctuation_hints": punctuation["punctuation_hints"],
                    "punctuation_inserted_period_count": punctuation["punctuation_inserted_period_count"],
                    "punctuation_inserted_comma_count": punctuation["punctuation_inserted_comma_count"],
                    "short_response_period_count": punctuation["short_response_period_count"],
                    "possible_speaker_change_period_count": punctuation["possible_speaker_change_period_count"],
                    "boundary_hint_used_count": punctuation["boundary_hint_used_count"],
                    "final_text_before_cleanup": final_text_before_cleanup,
                    "final_text_after_cleanup": final_text,
                    "final_text_before_domain_correction": final_text_before_domain_correction,
                    "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
                    "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
                    "boundary_cleanup_attempted": bool(cleanup.get("boundary_cleanup_attempted")),
                    "boundary_cleanup_reverted": bool(cleanup.get("boundary_cleanup_reverted")),
                    "cleanup_validation_failed": bool(cleanup.get("cleanup_validation_failed")),
                    "protected_prefix_prevented_cleanup": bool(cleanup.get("protected_prefix_prevented_cleanup")),
                }
            )

    if output_dir is not None:
        ensure_dir(output_dir / "fusion")
        save_jsonl(output_dir / "fusion" / f"{episode_id}.final_blocks.jsonl", final_blocks)
        save_jsonl(output_dir / "fusion" / f"{episode_id}.final_segments.jsonl", final_rows)
        save_jsonl(output_dir / "fusion" / f"{episode_id}.review_queue.jsonl", review_rows)
        transcript_lines = [f"# {episode_id}", ""]
        for row in final_blocks:
            start = row["time"]["start_sec"]
            end = row["time"]["end_sec"]
            transcript_lines.append(f"- [{start:.2f}-{end:.2f}] {row.get('final_text_display', row['final_text'])}")
        (output_dir / "fusion" / f"{episode_id}.final_transcript.md").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        sentence_timeline_lines = []
        for row in final_rows:
            sentence_timeline_lines.append(
                json.dumps(
                    {
                        "episode_id": row["episode_id"],
                        "sentence_id": row["sentence_id"],
                        "block_id": row["block_id"],
                        "time": row["time"],
                        "raw_text": row["apple_text"],
                        "display_text": row.get("sentence_display_text", row.get("final_text_display", row["final_text"])),
                        "punctuation_hints": row.get("punctuation_hints", []),
                    },
                    ensure_ascii=False,
                )
            )
        (output_dir / "fusion" / f"{episode_id}.sentence_timeline.jsonl").write_text("\n".join(sentence_timeline_lines) + ("\n" if sentence_timeline_lines else ""), encoding="utf-8")

    return final_blocks, final_rows, review_rows, stats
