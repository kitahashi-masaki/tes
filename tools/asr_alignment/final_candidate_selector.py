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
        boundary_anchor_suspected = _has_candidate_boundary_anchor_mismatch(
            candidate_text,
            str(apple_segment.get("apple", {}).get("text", "") or ""),
        )
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
        if boundary_anchor_suspected:
            risk_penalty += 0.16
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
    ("退路を立った", "退路を断った"),
    ("先進で家出ろよ", "自立して家出ろよ"),
    ("自律家出ろよ", "自立して家出ろよ"),
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
    ("つこの8050", "この8050"),
    ("るということです。ただこれ", "ただこれ"),
    ("ということです。た", "ということです"),
    ("ポイントなんですよ。は", "ポイントなんですよ"),
)
_TRAILING_FRAGMENT_SUFFIXES = ("、よ", "。聞", "、し", "。し", "でそうす", "とこ", "依")


def _boundary_anchor_norm(text: str) -> str:
    return re.sub(r"[\s、。，．,.!?！？]+", "", str(text or ""))


def _has_candidate_boundary_anchor_mismatch(candidate_text: str, apple_text: str) -> bool:
    candidate = _boundary_anchor_norm(candidate_text)
    apple = _boundary_anchor_norm(apple_text)
    if len(candidate) < 12 or len(apple) < 12:
        return False
    anchor_len = min(10, max(5, len(apple) // 8))
    prefix = apple[:anchor_len]
    suffix = apple[-anchor_len:]
    head = candidate[: max(24, anchor_len * 4)]
    tail = candidate[-max(24, anchor_len * 4):]
    prefix_pos = head.find(prefix)
    suffix_pos = tail.rfind(suffix)
    if prefix_pos > 3:
        return True
    if suffix_pos >= 0 and len(tail) - (suffix_pos + len(suffix)) > 3:
        return True
    return False


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


_UNUSUAL_FINAL_TEXT_PATTERN_HUMAN_REQUIRED = (
    "19080年代",
    "ハウスカーなんです",
    "火復化した家づくり",
    "倒ブレット",
    "とろびていない",
    "遠藤和樹",
    "円道一樹",
    "遠藤勝樹",
    "矢野圭三",
    "ヤノウ",
)
_UNUSUAL_FINAL_TEXT_PATTERN_MACHINE_NOTE = (
    "うのも",
    "に対して",
    "、いや",
    "ほど。はい",
    "とです",
    "を言うと",
    "の陣",
    "は家",
    "そうしないと思う",
    "ね。そうですね",
    "うことです",
    "のだから",
    "野郎",
)
_UNUSUAL_FINAL_TEXT_PATTERN_CONTEXT_WINDOW = 28


def _find_unusual_final_text_patterns(text: str) -> list[dict[str, str]]:
    value = str(text or "")
    patterns: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not value.strip():
        return patterns
    ordered_patterns = []
    for pattern in _UNUSUAL_FINAL_TEXT_PATTERN_HUMAN_REQUIRED:
        ordered_patterns.append((pattern, "human_required"))
    for pattern in _UNUSUAL_FINAL_TEXT_PATTERN_MACHINE_NOTE:
        ordered_patterns.append((pattern, "machine_note"))
    for pattern, severity in ordered_patterns:
        start = value.find(pattern)
        if start < 0:
            continue
        end = start + len(pattern)
        key = (pattern, severity)
        if key in seen:
            continue
        seen.add(key)
        patterns.append(
            {
                "pattern_id": pattern,
                "matched_text": pattern,
                "context": value[max(0, start - _UNUSUAL_FINAL_TEXT_PATTERN_CONTEXT_WINDOW) : min(len(value), end + _UNUSUAL_FINAL_TEXT_PATTERN_CONTEXT_WINDOW)],
                "severity": severity,
            }
        )
    return patterns


def _unusual_final_text_pattern_level(text: str) -> str | None:
    value = str(text or "").strip()
    if not value:
        return "human_required"
    if any(pattern in value for pattern in _UNUSUAL_FINAL_TEXT_PATTERN_HUMAN_REQUIRED):
        return "human_required"
    if any(pattern in value for pattern in _UNUSUAL_FINAL_TEXT_PATTERN_MACHINE_NOTE):
        return "machine_note"
    if value.startswith(("、", "。")):
        return "machine_note"
    if value.endswith(("ポイ", "簡", "入れ", "そ", "あ", "てる", "中")):
        return "machine_note"
    return None


def _has_unusual_final_text_pattern(text: str) -> bool:
    return _unusual_final_text_pattern_level(text) is not None


def _normalized_review_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    if text:
        return text
    fallback_text = str(fallback or "").strip()
    return fallback_text if fallback_text else "(missing)"


def _candidate_texts_for_block(block: dict[str, Any]) -> dict[str, str]:
    final_text = _normalized_review_text(block.get("final_text_display") or block.get("final_text") or block.get("final_text_raw"))
    return {
        "apple": _normalized_review_text(block.get("apple_text"), final_text),
        "qwen": _normalized_review_text(block.get("qwen_text"), final_text),
        "nemotron": _normalized_review_text(block.get("nemotron_text"), final_text),
        "whisper": _normalized_review_text(block.get("whisper_text"), final_text),
    }


def _review_output_row(block: dict[str, Any], *, review_level: str, human_review_required: bool, machine_review_note: bool) -> dict[str, Any]:
    final_text = _normalized_review_text(block.get("final_text"), block.get("final_text_display"))
    final_text_display = _normalized_review_text(block.get("final_text_display"), final_text)
    apple_text = _normalized_review_text(block.get("apple_text"), final_text_display)
    qwen_text = _normalized_review_text(block.get("qwen_text"), final_text_display)
    nemotron_text = _normalized_review_text(block.get("nemotron_text"), final_text_display)
    whisper_text = _normalized_review_text(block.get("whisper_text"), final_text_display)
    candidate_texts = _candidate_texts_for_block(block)
    final_risk_flags = list(block.get("final_risk_flags") or block.get("risk_flags") or [])
    unusual_patterns = _find_unusual_final_text_patterns(final_text_display)
    row = {
        "episode_id": block.get("episode_id", ""),
        "block_id": block.get("block_id", ""),
        "sentence_ids": list(block.get("sentence_ids") or []),
        "time": {
            "start_sec": float((block.get("time") or {}).get("start_sec", 0.0) or 0.0),
            "end_sec": float((block.get("time") or {}).get("end_sec", 0.0) or 0.0),
        },
        "review_level": review_level,
        "review_priority": _normalized_review_text(block.get("review_priority")),
        "human_review_required": human_review_required,
        "machine_review_note": machine_review_note,
        "needs_review": bool(block.get("needs_review")),
        "pre_llm_needs_review": bool(block.get("pre_llm_needs_review")),
        "normalized_needs_review": bool(block.get("normalized_needs_review")),
        "llm_called": bool(block.get("llm_called", block.get("llm_used"))),
        "llm_selected": bool(block.get("llm_selected")),
        "llm_resolved": bool(block.get("llm_resolved")),
        "selected_source": _normalized_review_text(block.get("selected_source")),
        "selection_method": _normalized_review_text(block.get("selection_method")),
        "alignment_quality": _normalized_review_text(block.get("alignment_quality")),
        "qwen_apple_difference_type": _normalized_review_text(block.get("qwen_apple_difference_type"), "unknown"),
        "qwen_apple_similarity": float(block.get("qwen_apple_similarity") if block.get("qwen_apple_similarity") is not None else 0.0),
        "final_text_raw": final_text,
        "final_text_display": final_text_display,
        "apple_text": apple_text,
        "qwen_text": qwen_text,
        "nemotron_text": nemotron_text,
        "whisper_text": whisper_text,
        "candidate_texts": candidate_texts,
        "risk_flags": list(block.get("risk_flags") or []),
        "final_risk_flags": final_risk_flags,
        "review_gate_reasons": list(block.get("review_gate_reasons") or []),
        "machine_note_reasons": list(block.get("machine_note_reasons") or []),
        "unusual_final_text_patterns": unusual_patterns,
        "review_reason": list(block.get("review_reason") or []),
    }
    row["candidate_summary"] = {key: _normalized_review_text(value, final_text_display) for key, value in (block.get("candidate_summary") or {}).items()}
    row["review_reason"] = list(row["review_reason"]) if isinstance(row["review_reason"], list) else [str(row["review_reason"])]
    row["review_reason"] = [str(reason) for reason in row["review_reason"] if str(reason)]
    row["candidate_texts"] = {key: _normalized_review_text(value, final_text_display) for key, value in candidate_texts.items()}
    return row


def _review_queue_row(block: dict[str, Any]) -> dict[str, Any]:
    return _review_output_row(block, review_level=str(block.get("review_level") or "human_required"), human_review_required=True, machine_review_note=False)


def _machine_review_note_row(block: dict[str, Any]) -> dict[str, Any]:
    return _review_output_row(block, review_level="machine_note", human_review_required=False, machine_review_note=True)


def _classify_review_level(
    segment: dict[str, Any],
    final_risk_flags: list[str],
    final_text: str,
    *,
    selected_source: str,
) -> dict[str, Any]:
    diff_type = str(segment.get("qwen_apple_difference_type") or "semantic")
    qwen = segment.get("qwen", {}) if isinstance(segment.get("qwen"), dict) else {}
    qwen_alignment = float(qwen.get("local_alignment_score", qwen.get("alignment_score", 0.0)) or 0.0)
    reasons = set(final_risk_flags)
    gate_reasons: list[str] = []
    note_reasons: list[str] = []

    severe_gate_flags = {
        "numeric_disagreement",
        "critical_term_disagreement",
        "boundary_contamination_suspected",
        "domain_error_phrase",
    }
    if any(flag in reasons for flag in severe_gate_flags):
        gate_reasons.extend([flag for flag in final_risk_flags if flag in severe_gate_flags])
    if diff_type in {"critical", "semantic"} and "qwen_apple_disagreement" in reasons:
        if any(flag in reasons for flag in {"numeric_disagreement", "critical_term_disagreement", "domain_error_phrase", "unusual_final_text_pattern", "boundary_contamination_suspected"}):
            gate_reasons.append("qwen_apple_disagreement")
        else:
            note_reasons.append("qwen_apple_disagreement")
    elif "qwen_apple_disagreement" in reasons:
        note_reasons.append("qwen_apple_disagreement")

    if qwen_alignment < 0.82:
        if selected_source == "qwen":
            gate_reasons.append("qwen_alignment_low")
        elif any(flag in reasons for flag in severe_gate_flags | {"boundary_contamination_suspected"}):
            gate_reasons.append("qwen_alignment_low")
        else:
            note_reasons.append("qwen_alignment_low")

    unusual_level = _unusual_final_text_pattern_level(final_text)
    if unusual_level == "human_required":
        gate_reasons.append("unusual_final_text_pattern")
    elif unusual_level == "machine_note":
        note_reasons.append("unusual_final_text_pattern")

    if "surface_difference" in reasons:
        note_reasons.append("surface_difference")
    if "large_span_drift_warning" in reasons or "large_span_drift" in reasons:
        note_reasons.append("large_span_drift")
    if "boundary_cleanup_needed" in reasons:
        note_reasons.append("boundary_cleanup_needed")
    if "boundary_suggestion_available" in reasons:
        note_reasons.append("boundary_suggestion_available")

    if diff_type in {"critical", "semantic"}:
        note_reasons.append("qwen_apple_disagreement")

    human_required = bool(gate_reasons)
    machine_note = bool(note_reasons) and not human_required
    if human_required:
        review_level = "human_required"
        priority = "high"
    elif machine_note:
        review_level = "machine_note"
        priority = "low" if len(note_reasons) <= 2 else "medium"
    else:
        review_level = "auto_accept"
        priority = "none"
    return {
        "review_level": review_level,
        "human_review_required": human_required,
        "machine_review_note": machine_note,
        "review_priority": priority,
        "review_gate_reasons": sorted(dict.fromkeys(gate_reasons)),
        "machine_note_reasons": sorted(dict.fromkeys(note_reasons)),
    }


def _human_review_required(
    segment: dict[str, Any],
    final_risk_flags: list[str],
    final_text: str,
    *,
    selected_source: str = "",
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    diff_type = str(segment.get("qwen_apple_difference_type") or "semantic")
    qwen = segment.get("qwen", {}) if isinstance(segment.get("qwen"), dict) else {}
    qwen_alignment = float(qwen.get("local_alignment_score", qwen.get("alignment_score", 0.0)) or 0.0)
    severe_flags = {
        "critical_term_disagreement",
        "numeric_disagreement",
        "boundary_contamination_suspected",
        "span_too_long",
        "domain_error_phrase",
    }
    if any(flag in final_risk_flags for flag in severe_flags):
        reasons.extend([flag for flag in final_risk_flags if flag in severe_flags])
    if diff_type in {"critical", "semantic"} and "qwen_apple_disagreement" in final_risk_flags:
        reasons.append("qwen_apple_disagreement")
    if diff_type in {"critical", "semantic"} and "qwen_apple_disagreement" not in final_risk_flags:
        reasons.append("qwen_apple_disagreement")
    if qwen_alignment < 0.82 and (selected_source == "qwen" or any(flag in final_risk_flags for flag in severe_flags)):
        reasons.append("qwen_alignment_low")
    if _has_unusual_final_text_pattern(final_text):
        reasons.append("unusual_final_text_pattern")
    human_review_required = bool(reasons)
    return human_review_required, sorted(dict.fromkeys(reasons))


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
    machine_review_rows: list[dict[str, Any]] = []
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
        "llm_selected_count": 0,
        "llm_resolved_count": 0,
        "review_level_counts": {"auto_accept": 0, "machine_note": 0, "human_required": 0},
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
        pre_llm_needs_review = bool(segment.get("needs_review"))
        normalized_needs_review = bool(segment.get("needs_review"))
        llm_selected = False
        llm_resolved = False
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
                llm_selected = True
                stats["llm_selected_count"] += 1
                if llm_decision.needs_review is not None:
                    if llm_decision.needs_review and not pre_llm_needs_review:
                        stats["llm_changed_needs_review_true_count"] += 1
                    if not llm_decision.needs_review and pre_llm_needs_review:
                        stats["llm_changed_needs_review_false_count"] += 1
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

        final_risk_flags = list(segment.get("risk_flags", []) or [])
        qwen_for_review = segment.get("qwen", {}) if isinstance(segment.get("qwen"), dict) else {}
        qwen_alignment_for_review = float(qwen_for_review.get("local_alignment_score", qwen_for_review.get("alignment_score", 0.0)) or 0.0)
        final_risk_flags = _classify_large_span_drift(final_risk_flags, qwen_alignment_for_review)

        candidate_text_similarity = 0.0
        candidate_texts_for_mix = [str(row.get("text", "") or "") for row in candidate_rows.values() if str(row.get("text", "") or "").strip()]
        if candidate_texts_for_mix:
            candidate_text_similarity = max(segment_similarity_score(final_text, candidate_text) for candidate_text in candidate_texts_for_mix)
        if llm_selected and llm_decision is not None and llm_decision.final_text and candidate_text_similarity < 0.92:
            final_risk_flags.append("llm_candidate_mix_suspected")
        unusual_final_text_patterns = _find_unusual_final_text_patterns(final_text)
        if unusual_final_text_patterns and "unusual_final_text_pattern" not in final_risk_flags:
            final_risk_flags.append("unusual_final_text_pattern")

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

        if "domain_error_phrase" in final_risk_flags and not _contains_domain_error(final_text) and not _contains_domain_error(punctuation["display_text"]):
            final_risk_flags.append("domain_error_avoided")
            stats["domain_error_avoided_count"] += 1
        if domain_text_corrected and "domain_text_corrected" not in final_risk_flags:
            final_risk_flags.append("domain_text_corrected")
        if "large_span_drift_warning" in final_risk_flags:
            stats["large_span_drift_warning_count"] += 1
        if _has_boundary_contamination_suspect(final_text) or _has_boundary_contamination_suspect(punctuation["display_text"]):
            if "boundary_contamination_suspected" not in final_risk_flags:
                final_risk_flags.append("boundary_contamination_suspected")
        if cleanup_needs_review and "boundary_cleanup_needed" not in final_risk_flags:
            final_risk_flags.append("boundary_cleanup_needed")
        review_classification = _classify_review_level(
            segment,
            final_risk_flags,
            final_text,
            selected_source=selected_source,
        )
        human_review_required = bool(review_classification["human_review_required"])
        machine_review_note = bool(review_classification["machine_review_note"])
        review_level = str(review_classification["review_level"])
        review_priority = str(review_classification["review_priority"])
        review_gate_reasons = list(review_classification["review_gate_reasons"])
        machine_note_reasons = list(review_classification["machine_note_reasons"])
        if llm_selected and selected_source != deterministic["selected_source"]:
            llm_resolved = True
            stats["llm_resolved_count"] += 1
        suggested_final_text, suggestion_reasons = _without_leading_fragment_suggestion(final_text, candidate_summary.get("apple", ""))
        if suggested_final_text != final_text:
            stats["suggested_final_text_count"] += 1
            if "leading_boundary_fragment_suggested" in suggestion_reasons:
                stats["boundary_suggestion_count"] += 1
            if "trailing_boundary_fragment_suggested" in suggestion_reasons:
                stats["trailing_boundary_suggestion_count"] += 1
            if "boundary_suggestion_available" not in final_risk_flags:
                final_risk_flags.append("boundary_suggestion_available")
        elif domain_candidate_switched:
            suggested_final_text = final_text
            suggestion_reasons = preferred_reasons
            stats["suggested_final_text_count"] += 1

        if suggested_final_text != final_text:
            review_gate_reasons = merge_review_reasons(review_gate_reasons, ["boundary_suggestion_available"])
            if not human_review_required:
                machine_review_note = True
                review_level = "machine_note"
                review_priority = "low"
                machine_note_reasons = merge_review_reasons(machine_note_reasons, ["boundary_suggestion_available"])
        human_review_reason = review_gate_reasons
        if human_review_required:
            review_reason = human_review_reason
        elif machine_review_note:
            review_reason = machine_note_reasons
        else:
            review_reason = []
        needs_review = human_review_required
        stats["review_level_counts"][review_level] += 1

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
            "alignment_quality": segment.get("alignment_quality"),
            "qwen_apple_difference_type": segment.get("qwen_apple_difference_type"),
            "qwen_apple_similarity": float(segment.get("qwen_apple_similarity", 0.0) or 0.0),
            "final_text": final_text,
            "final_text_raw": final_text_before_cleanup,
            "final_text_after_boundary_cleanup": final_text,
            "final_text_display": punctuation["display_text"],
            "selected_source": selected_source,
            "selection_method": selection_method,
            "confidence": float(confidence),
            "needs_review": needs_review,
            "pre_llm_needs_review": pre_llm_needs_review,
            "normalized_needs_review": normalized_needs_review,
            "llm_called": bool(should_call),
            "human_review_required": human_review_required,
            "machine_review_note": machine_review_note,
            "review_level": review_level,
            "review_priority": review_priority,
            "review_gate_reasons": review_gate_reasons,
            "machine_note_reasons": machine_note_reasons,
            "human_review_reason": human_review_reason,
            "llm_selected": llm_selected,
            "llm_resolved": llm_resolved,
            "review_reason": review_reason,
            "risk_flags": final_risk_flags,
            "final_risk_flags": final_risk_flags,
            "suggested_final_text": suggested_final_text,
            "suggested_final_text_reason": suggestion_reasons,
            "domain_candidate_switched": domain_candidate_switched,
            "domain_text_corrected": domain_text_corrected,
            "domain_text_corrections": domain_text_corrections,
            "candidate_summary": candidate_summary,
            "candidate_texts": {
                "apple": candidate_summary.get("apple", ""),
                "qwen": candidate_summary.get("qwen", ""),
                "nemotron": candidate_summary.get("nemotron", ""),
                "whisper": candidate_summary.get("whisper", ""),
            },
            "apple_display_text": candidate_summary.get("apple", ""),
            "apple_text": candidate_summary.get("apple", ""),
            "qwen_text": candidate_summary.get("qwen", ""),
            "nemotron_text": candidate_summary.get("nemotron", ""),
            "whisper_text": candidate_summary.get("whisper", ""),
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
            "unusual_final_text_patterns": unusual_final_text_patterns,
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
                    "pre_llm_needs_review": pre_llm_needs_review,
                    "normalized_needs_review": normalized_needs_review,
                    "human_review_required": human_review_required,
                    "machine_review_note": machine_review_note,
                    "review_level": review_level,
                    "review_priority": review_priority,
                    "review_gate_reasons": review_gate_reasons,
                    "machine_note_reasons": machine_note_reasons,
                    "human_review_reason": human_review_reason,
                    "llm_selected": llm_selected,
                    "llm_resolved": llm_resolved,
                    "review_reason": review_reason,
                    "risk_flags": final_risk_flags,
                    "suggested_final_text": suggested_final_text,
                    "suggested_final_text_reason": suggestion_reasons,
                    "domain_candidate_switched": domain_candidate_switched,
                    "domain_text_corrected": domain_text_corrected,
                    "domain_text_corrections": domain_text_corrections,
                    "candidate_summary": candidate_summary,
                    "apple_display_text": candidate_summary.get("apple", ""),
                    "apple_text": candidate_summary.get("apple", ""),
                    "qwen_text": candidate_summary.get("qwen", ""),
                    "nemotron_text": candidate_summary.get("nemotron", ""),
                    "whisper_text": candidate_summary.get("whisper", ""),
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
        if human_review_required:
            review_rows.append(
                {
                    "episode_id": episode_id,
                    "block_id": segment["block_id"],
                    "sentence_ids": segment["sentence_ids"],
                    "time": segment_time,
                    "segment_id": segment.get("segment_id"),
                    "needs_review": needs_review,
                    "human_review_required": human_review_required,
                    "machine_review_note": machine_review_note,
                    "review_level": review_level,
                    "review_priority": review_priority,
                    "review_reason": review_reason,
                    "alignment_quality": segment.get("alignment_quality"),
                    "risk_flags": list(segment.get("risk_flags", []) or []),
                    "final_risk_flags": final_risk_flags,
                    "candidate_summary": candidate_summary,
                    "candidate_texts": candidate_summary,
                    "final_text": final_text,
                    "suggested_final_text": suggested_final_text,
                    "suggested_final_text_reason": suggestion_reasons,
                    "domain_candidate_switched": domain_candidate_switched,
                    "domain_text_corrected": domain_text_corrected,
                    "domain_text_corrections": domain_text_corrections,
                    "selected_source": selected_source,
                    "selection_method": selection_method,
                    "confidence": float(confidence),
                    "pre_llm_needs_review": pre_llm_needs_review,
                    "normalized_needs_review": normalized_needs_review,
                    "llm_called": bool(should_call),
                    "llm_selected": llm_selected,
                    "llm_resolved": llm_resolved,
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                    "apple_display_text": candidate_summary.get("apple", ""),
                    "apple_text": candidate_summary.get("apple", ""),
                    "qwen_text": candidate_summary.get("qwen", ""),
                    "nemotron_text": candidate_summary.get("nemotron", ""),
                    "whisper_text": candidate_summary.get("whisper", ""),
                    "candidate_texts": candidate_summary,
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
                    "review_level": review_level,
                    "review_priority": review_priority,
                    "review_gate_reasons": review_gate_reasons,
                    "machine_note_reasons": machine_note_reasons,
                    "unusual_final_text_patterns": _find_unusual_final_text_patterns(final_text),
                    "llm_called": bool(should_call),
                    "llm_selected": llm_selected,
                    "llm_resolved": llm_resolved,
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                }
            )
        elif machine_review_note:
            machine_review_rows.append(
                {
                    "episode_id": episode_id,
                    "block_id": segment["block_id"],
                    "sentence_ids": segment["sentence_ids"],
                    "time": segment_time,
                    "segment_id": segment.get("segment_id"),
                    "review_level": review_level,
                    "review_priority": review_priority,
                    "machine_review_note": machine_review_note,
                    "machine_note_reasons": machine_note_reasons,
                    "review_gate_reasons": review_gate_reasons,
                    "needs_review": False,
                    "human_review_required": False,
                    "normalized_needs_review": normalized_needs_review,
                    "pre_llm_needs_review": pre_llm_needs_review,
                    "selected_source": selected_source,
                    "selection_method": selection_method,
                    "confidence": float(confidence),
                    "risk_flags": list(segment.get("risk_flags", []) or []),
                    "final_risk_flags": final_risk_flags,
                    "candidate_summary": candidate_summary,
                    "final_text": final_text,
                    "final_text_display": punctuation["display_text"],
                    "final_text_raw": final_text_before_cleanup,
                    "apple_text": candidate_summary.get("apple", ""),
                    "qwen_text": candidate_summary.get("qwen", ""),
                    "nemotron_text": candidate_summary.get("nemotron", ""),
                    "whisper_text": candidate_summary.get("whisper", ""),
                    "candidate_texts": candidate_summary,
                    "unusual_final_text_patterns": _find_unusual_final_text_patterns(final_text),
                    "llm_called": bool(should_call),
                    "llm_selected": llm_selected,
                    "llm_resolved": llm_resolved,
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                }
            )

    if output_dir is not None:
        review_rows = [_review_queue_row(row) for row in final_blocks if row.get("human_review_required")]
        machine_review_rows = [_machine_review_note_row(row) for row in final_blocks if row.get("machine_review_note")]
        ensure_dir(output_dir / "fusion")
        save_jsonl(output_dir / "fusion" / f"{episode_id}.final_blocks.jsonl", final_blocks)
        save_jsonl(output_dir / "fusion" / f"{episode_id}.final_segments.jsonl", final_rows)
        save_jsonl(output_dir / "fusion" / f"{episode_id}.review_queue.jsonl", review_rows)
        save_jsonl(output_dir / "fusion" / f"{episode_id}.machine_review_notes.jsonl", machine_review_rows)
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

    review_rows = [_review_queue_row(row) for row in final_blocks if row.get("human_review_required")]
    machine_review_rows = [_machine_review_note_row(row) for row in final_blocks if row.get("machine_review_note")]
    stats["machine_review_note_count"] = len(machine_review_rows)
    stats["human_review_required_count"] = len(review_rows)
    stats["auto_accept_final_count"] = sum(1 for row in final_blocks if row.get("review_level") == "auto_accept")
    return final_blocks, final_rows, review_rows, stats
