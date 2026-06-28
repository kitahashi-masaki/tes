from __future__ import annotations

import json
import sys
from pathlib import Path
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
        search_radius=96,
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
        "source": source,
    }


def build_block_candidates(
    *,
    episode_id: str,
    alignment_blocks: list[AlignmentBlock],
    sentence_units: list[AppleSentenceUnit],
    apple_artifact,
    alignments: dict[str, AlignmentResult],
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quality_counts: dict[str, int] = {}
    for block in alignment_blocks:
        apple_text = block.text
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
                "text": apple_text,
                "stability_score": block.stability_score,
                "char_start": block.char_start,
                "char_end": block.char_end,
            },
        }
        candidate_rows: dict[str, dict[str, Any]] = {}
        for engine in ("qwen", "nemotron", "whisper"):
            alignment = alignments[engine]
            candidate_rows[engine] = _extract_candidate(
                alignment,
                apple_artifact,
                block.char_start,
                block.char_end,
                alignment.asr_artifact,
                source=engine,
            )
        payload.update(candidate_rows)
        usable_asr_candidates = {source: row for source, row in candidate_rows.items() if row.get("usable_for_agreement")}
        candidate_texts = {source: row.get("text", "") for source, row in usable_asr_candidates.items()}
        agreement = candidate_agreement_score(candidate_texts)
        usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in usable_asr_candidates.values()]
        if not usable_scores:
            usable_scores = [float(row.get("local_alignment_score", row.get("alignment_score", 0.0))) for row in candidate_rows.values()]
        qwen_diff_type, qwen_similarity, qwen_critical = classify_qwen_apple_difference(apple_text, candidate_rows["qwen"]["text"])
        alignment_quality = summarize_quality(
            min(usable_scores) if usable_scores else 0.0,
            agreement,
            False,
            not bool(usable_asr_candidates),
        )
        payload["candidate_agreement_score"] = agreement
        payload["qwen_apple_difference_type"] = qwen_diff_type
        payload["qwen_apple_similarity"] = qwen_similarity
        payload["important_term_disagreement"] = qwen_diff_type in {"critical", "semantic"}
        payload["critical_term_disagreement"] = qwen_critical
        payload["soft_domain_difference"] = qwen_diff_type == "soft_domain"
        payload["usable_asr_candidates"] = usable_asr_candidates
        payload["alignment_quality"] = alignment_quality
        payload["risk_flags"] = compute_risk_flags(payload)
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
        rows.append(payload)
        quality_counts[alignment_quality] = quality_counts.get(alignment_quality, 0) + 1

    summary = {
        "episode_id": episode_id,
        "block_candidate_count": len(rows),
        "quality_counts": quality_counts,
        "needs_review_count": sum(1 for row in rows if row["needs_review"]),
    }

    if output_dir is not None:
        ensure_dir(output_dir / "aligned_segments")
        save_jsonl(output_dir / "aligned_segments" / f"{episode_id}.block_candidates.jsonl", rows)
        save_jsonl(output_dir / "aligned_segments" / f"{episode_id}.segment_candidates.jsonl", rows)
        (output_dir / "normalized" / f"{episode_id}.block_candidates.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, summary
