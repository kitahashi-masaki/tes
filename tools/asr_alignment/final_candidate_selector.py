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


def _combined_score(candidate: dict[str, Any], apple_segment: dict[str, Any], source: str) -> float:
    alignment = float(candidate.get("local_alignment_score", candidate.get("alignment_score", 0.0)))
    agreement = float(apple_segment.get("candidate_agreement_score", 0.0))
    apple_stability = float(apple_segment.get("apple", {}).get("stability_score", 0.0))
    source_bonus = {"qwen": 0.08, "apple": 0.05, "nemotron": 0.02, "whisper": 0.0}.get(source, 0.0)
    usable_bonus = 0.04 if candidate.get("usable_for_agreement") else -0.05
    return min(1.0, 0.56 * alignment + 0.24 * agreement + 0.12 * apple_stability + source_bonus + usable_bonus)


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


_LEADING_FRAGMENT_RE = re.compile(r"^(?:[ぁ-ゖァ-ヶ]{1,2}|[一-龯]{1,2}|[ぁ-ゖァ-ヶ]{1,2}。|[一-龯]{1,2}。|ね。|よ。|うん。|あ。|え。|はい。|そう。|そうす|ですよ。|ですね、)$")
_TRAILING_FRAGMENT_RE = re.compile(r"(?:[ぁ-ゖァ-ヶ]{1,2}|[一-龯]{1,2}|[ぁ-ゖァ-ヶ]{1,2}。|[一-龯]{1,2}。|建|実|な|ポイ|入れ|そうす)$")


def _cleanup_boundary_fragments(final_text: str, apple_text: str, selected_source: str) -> dict[str, Any]:
    original_text = str(final_text or "")
    apple_text = str(apple_text or "")
    selected_source = str(selected_source or "")
    result_text = original_text
    reasons: list[str] = []
    applied = False

    if not result_text or selected_source == "apple":
        return {
            "final_text_before_cleanup": original_text,
            "final_text_after_cleanup": result_text,
            "boundary_cleanup_applied": False,
            "boundary_cleanup_reason": [],
        }

    def _score(text: str) -> float:
        return sequence_similarity(text, apple_text)

    baseline = _score(result_text)

    def _choose_best_trim(is_leading: bool) -> tuple[str, str] | None:
        best_text = result_text
        best_reason = ""
        best_score = baseline
        candidates: list[tuple[int, str, str]] = []
        limit = min(6, len(result_text) - 1)
        for cut in range(1, limit + 1):
            trimmed = result_text[cut:] if is_leading else result_text[:-cut]
            if len(trimmed) < 3:
                continue
            fragment = result_text[:cut] if is_leading else result_text[-cut:]
            if not fragment.strip():
                continue
            if not (
                _LEADING_FRAGMENT_RE.match(fragment) if is_leading else _TRAILING_FRAGMENT_RE.search(fragment)
            ):
                continue
            candidates.append((cut, trimmed, fragment))
        for cut, trimmed, fragment in candidates:
            score = _score(trimmed)
            if score > best_score + 0.005 or (abs(score - best_score) <= 0.005 and len(trimmed) > 0 and len(trimmed) < len(best_text)):
                best_text = trimmed
                best_score = score
                best_reason = "leading_fragment_removed" if is_leading else "trailing_fragment_removed"
        if best_reason:
            return best_text, best_reason
        return None

    leading = _choose_best_trim(True)
    if leading is not None:
        result_text, reason = leading
        reasons.append(reason)
        applied = True
    trailing = _choose_best_trim(False)
    if trailing is not None:
        result_text, reason = trailing
        reasons.append(reason)
        applied = True

    if not result_text:
        result_text = original_text
        reasons = []
        applied = False

    unnatural = False
    if result_text:
        if len(result_text) <= 2:
            unnatural = True
        if result_text[-1] in {"な", "で", "と", "そ", "よ", "ね"} and len(result_text) <= 10:
            unnatural = True
        if _LEADING_FRAGMENT_RE.match(result_text[:4]):
            unnatural = True
        if _TRAILING_FRAGMENT_RE.search(result_text[-4:]):
            unnatural = True

    return {
        "final_text_before_cleanup": original_text,
        "final_text_after_cleanup": result_text,
        "boundary_cleanup_applied": applied,
        "boundary_cleanup_reason": reasons,
        "boundary_cleanup_needs_review": unnatural,
    }


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
    }
    previous_end: float | None = None
    risky_calls = 0
    unit_map = {u.sentence_id: u for u in sentence_units}
    for segment in block_rows:
        candidate_rows = {source: segment[source] for source in ("qwen", "apple", "nemotron", "whisper")}
        best_source, deterministic = _deterministic_selection(candidate_rows, segment)
        selected_source = deterministic["selected_source"]
        final_text = deterministic["final_text"]
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
                    final_text = candidate_rows[llm_source].get("text", final_text)
                    confidence = llm_decision.confidence if llm_decision.confidence is not None else confidence
                    selection_method = "llm_candidate_selection"
                if llm_decision.final_text:
                    llm_text = str(llm_decision.final_text).strip()
                    if llm_text and segment_similarity_score(llm_text, final_text) < 0.95:
                        stats["llm_changed_final_text_count"] += 1
                    if llm_text:
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
        cleanup = _cleanup_boundary_fragments(final_text, candidate_summary.get("apple", ""), selected_source)
        final_text_before_cleanup = cleanup["final_text_before_cleanup"]
        final_text = cleanup["final_text_after_cleanup"]
        if cleanup["boundary_cleanup_applied"]:
            review_reason = merge_review_reasons(review_reason, cleanup["boundary_cleanup_reason"])
        if cleanup.get("boundary_cleanup_needs_review"):
            needs_review = True
            review_reason = merge_review_reasons(review_reason, ["boundary_cleanup_needed"])

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
            "selected_source": selected_source,
            "selection_method": selection_method,
            "confidence": float(confidence),
            "needs_review": needs_review,
            "review_reason": review_reason,
            "candidate_summary": candidate_summary,
            "final_text_before_cleanup": final_text_before_cleanup,
            "final_text_after_cleanup": final_text,
            "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
            "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
            "llm_used": bool(should_call),
            "llm_cached": bool(llm_decision.cached) if llm_decision is not None else False,
            "selection_notes": llm_decision.notes if llm_decision is not None else "",
            "llm_error": llm_decision.error if llm_decision is not None else "",
        }
        final_blocks.append(block_final)
        for sentence_id in segment["sentence_ids"]:
            unit = unit_map[sentence_id]
            final_rows.append(
                {
                    "episode_id": episode_id,
                    "sentence_id": sentence_id,
                    "block_id": segment["block_id"],
                    "sentence_ids": segment["sentence_ids"],
                    "time": {"start_sec": unit.start_sec, "end_sec": unit.end_sec},
                    "apple_text": unit.text,
                    "final_text": final_text,
                    "selected_source": selected_source,
                    "selection_method": selection_method,
                    "confidence": float(confidence),
                    "needs_review": needs_review,
                    "review_reason": review_reason,
                    "candidate_summary": candidate_summary,
                    "final_text_before_cleanup": final_text_before_cleanup,
                    "final_text_after_cleanup": final_text,
                    "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
                    "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
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
                    "risk_flags": segment.get("risk_flags", []),
                    "candidate_summary": candidate_summary,
                    "final_text": final_text,
                    "selected_source": selected_source,
                    "confidence": float(confidence),
                    "llm_error": llm_decision.error if llm_decision is not None else "",
                    "final_text_before_cleanup": final_text_before_cleanup,
                    "final_text_after_cleanup": final_text,
                    "boundary_cleanup_applied": bool(cleanup["boundary_cleanup_applied"]),
                    "boundary_cleanup_reason": cleanup["boundary_cleanup_reason"],
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
            transcript_lines.append(f"- [{start:.2f}-{end:.2f}] {row['final_text']}")
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
                        "apple_text": row["apple_text"],
                    },
                    ensure_ascii=False,
                )
            )
        (output_dir / "fusion" / f"{episode_id}.sentence_timeline.jsonl").write_text("\n".join(sentence_timeline_lines) + ("\n" if sentence_timeline_lines else ""), encoding="utf-8")

    return final_blocks, final_rows, review_rows, stats
