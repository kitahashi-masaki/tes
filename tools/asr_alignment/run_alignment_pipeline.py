from __future__ import annotations

import argparse
import builtins
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        AppleSentenceUnit,
        AlignmentBlock,
        AlignmentResult,
        NormEntry,
        TextArtifact,
        ensure_dir,
        load_episode_files,
        load_jsonl,
        nfc_text,
        read_json_from_path,
        save_jsonl,
        build_text_artifact,
    )
    from tools.asr_alignment.apple_timeline_builder import build_apple_timeline  # type: ignore
    from tools.asr_alignment.flat_text_aligner import align_engine_to_apple  # type: ignore
    from tools.asr_alignment.final_candidate_selector import _machine_note_demotion_debug, select_final_candidates  # type: ignore
    from tools.asr_alignment.llm_client import LLMClient  # type: ignore
    from tools.asr_alignment.normalize_fusion_flags import normalize_file  # type: ignore
    from tools.asr_alignment.segment_candidate_builder import build_block_candidates  # type: ignore
else:
    from ._core import AppleSentenceUnit, AlignmentBlock, AlignmentResult, NormEntry, TextArtifact, ensure_dir, load_episode_files, load_jsonl, nfc_text, read_json_from_path, save_jsonl, build_text_artifact
    from .apple_timeline_builder import build_apple_timeline
    from .flat_text_aligner import align_engine_to_apple
    from .final_candidate_selector import _machine_note_demotion_debug, select_final_candidates
    from .llm_client import LLMClient
    from .normalize_fusion_flags import normalize_file
    from .segment_candidate_builder import build_block_candidates


def _find_engine_file(files: dict[str, Path], suffix: str) -> Path:
    for name, path in files.items():
        if nfc_text(name).endswith(suffix):
            return path
    raise FileNotFoundError(f"missing input file: {suffix}")


def _discover_episode_prefix(input_dir: Path, episode_prefix: str | None) -> str:
    if episode_prefix:
        return nfc_text(episode_prefix)
    files = load_episode_files(input_dir)
    stems = []
    for name in files:
        if ".MacOS-SpeechAnalyzer" in name:
            stems.append(name.split(".MacOS-SpeechAnalyzer", 1)[0])
        elif ".mlx-nemotron-3.5-asr-0.6b" in name:
            stems.append(name.split(".mlx-nemotron-3.5-asr-0.6b", 1)[0])
        elif ".qwen3-asr-1.7b" in name:
            stems.append(name.split(".qwen3-asr-1.7b", 1)[0])
        elif ".whisper-small" in name:
            stems.append(name.split(".whisper-small", 1)[0])
    stems = sorted(set(stems))
    if len(stems) != 1:
        raise ValueError(f"unable to infer a single episode prefix: {stems}")
    return nfc_text(stems[0])


def _force_clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _jsonl_is_valid(path: Path) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    count = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    except Exception:
        return False, count
    return True, count


def _phrase_counts(texts: list[str], phrases: tuple[str, ...]) -> dict[str, int]:
    joined = "\n".join(str(text or "") for text in texts)
    return {phrase: joined.count(phrase) for phrase in phrases if phrase in joined}


def _demote_debug_rows(final_blocks: list[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in final_blocks:
        if not block.get("human_review_required"):
            continue
        candidate_texts = block.get("candidate_texts")
        if not isinstance(candidate_texts, dict):
            candidate_texts = {}
        deterministic_resolution = {
            "deterministic_resolution_available": bool(block.get("deterministic_resolution_available")),
            "deterministic_resolution_reason": list(block.get("deterministic_resolution_reason") or []),
            "apple_whisper_agreement_high": bool(block.get("apple_whisper_agreement_high")),
            "apple_nemotron_whisper_agreement_high": bool(block.get("apple_nemotron_whisper_agreement_high")),
        }
        final_text = str(block.get("final_text_display") or block.get("final_text") or "")
        rows.append(
            _machine_note_demotion_debug(
                episode_id=episode_id,
                block=block,
                final_text=final_text,
                final_risk_flags=list(block.get("final_risk_flags") or block.get("risk_flags") or []),
                candidate_texts={k: str(v or "") for k, v in candidate_texts.items()},
                deterministic_resolution=deterministic_resolution,
            )
        )
    return rows


def _install_timed_print() -> None:
    started_at = time.perf_counter()
    original_print = builtins.print

    def timed_print(*args: Any, **kwargs: Any) -> None:
        elapsed = max(0, int(time.perf_counter() - started_at))
        prefix = f"[{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}]"
        if args and isinstance(args[0], str):
            args = (f"{prefix} {args[0]}",) + args[1:]
        else:
            args = (prefix, *args)
        original_print(*args, **kwargs)

    builtins.print = timed_print


def _load_alignment_result_from_files(output_dir: Path, episode_id: str, engine: str) -> AlignmentResult:
    summary = read_json_from_path(output_dir / "alignment" / f"{episode_id}.{engine}_alignment.json")
    normalized = read_json_from_path(output_dir / "normalized" / f"{episode_id}.{engine}.normalized.json")
    char_map = [
        NormEntry(
            norm_index=int(entry["norm_index"]),
            raw_index=int(entry["raw_index"]),
            raw_char=str(entry["raw_char"]),
            norm_char=str(entry["norm_char"]),
        )
        for entry in normalized.get("char_map", [])
    ]
    asr_artifact = TextArtifact(
        raw_text=str(normalized.get("raw_text", "")),
        display_norm_text=str(normalized.get("display_norm_text", "")),
        match_norm_text=str(normalized.get("match_norm_text", "")),
        entries=char_map,
    )
    apple_to_asr_map = list(summary.get("apple_to_asr_map") or normalized.get("apple_to_asr_map") or [])
    asr_to_apple_map = list(summary.get("asr_to_apple_map") or normalized.get("asr_to_apple_map") or [])
    # The saved alignment JSON does not contain artifacts, so rebuild them from the current outputs.
    return AlignmentResult(
        episode_id=summary["episode_id"],
        engine=summary["engine"],
        source_text_file=str(summary["source_text_file"]),
        apple_text_length=int(summary["apple_text_length"]),
        asr_text_length=int(summary["asr_text_length"]),
        normalized_apple_text_length=int(summary["normalized_apple_text_length"]),
        normalized_asr_text_length=int(summary["normalized_asr_text_length"]),
        global_alignment_score=float(summary["global_alignment_score"]),
        coverage_ratio=float(summary["coverage_ratio"]),
        anchors=summary.get("anchors") or [],
        apple_to_asr_map=[float(x) for x in apple_to_asr_map],
        asr_to_apple_map=[float(x) for x in asr_to_apple_map],
        apple_artifact=build_text_artifact(read_json_from_path(output_dir / "normalized" / f"{episode_id}.apple_timeline.json")["apple_stable_full_text"]),
        asr_artifact=asr_artifact,
    )


def _load_resume_bundle(output_dir: Path) -> tuple[str, dict[str, Any], list[AppleSentenceUnit], list[AlignmentBlock], dict[str, AlignmentResult], list[dict[str, Any]], dict[str, Any]]:
    manifest_paths = sorted((output_dir / "normalized").glob("*.manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"missing resume manifest in {output_dir / 'normalized'}")
    normalized_manifest = read_json_from_path(manifest_paths[0])
    episode_id = str(normalized_manifest["episode_id"])
    apple_timeline = read_json_from_path(output_dir / "normalized" / f"{episode_id}.apple_timeline.json")
    sentence_units = [AppleSentenceUnit(**row) for row in apple_timeline["apple_sentence_units"]]
    blocks = [AlignmentBlock(**row) for row in apple_timeline["alignment_blocks"]]
    alignments = {
        engine: _load_alignment_result_from_files(output_dir, episode_id, engine)
        for engine in ("qwen", "nemotron", "whisper")
    }
    block_rows = load_jsonl(output_dir / "fusion" / f"{episode_id}.normalized_block_candidates.jsonl")
    validation = {
        "sentence": __import__("tools.asr_alignment._core", fromlist=["validate_sentence_units"]).validate_sentence_units(
            apple_timeline["apple_stable_full_text"], sentence_units
        ),
        "block": __import__("tools.asr_alignment._core", fromlist=["validate_alignment_blocks"]).validate_alignment_blocks(
            apple_timeline["apple_stable_full_text"], sentence_units, blocks
        ),
    }
    return episode_id, apple_timeline, sentence_units, blocks, alignments, block_rows, validation


def _build_summary(
    *,
    episode_id: str,
    input_dir: Path,
    source_files: dict[str, Path],
    apple_timeline: dict[str, Any],
    alignments: dict[str, Any],
    block_rows: list[dict[str, Any]],
    block_summary: dict[str, Any],
    final_blocks: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    llm_stats: dict[str, Any],
    validation: dict[str, Any],
    output_dir: Path,
    conversation_punctuation: bool = False,
    short_response_period: bool = False,
) -> dict[str, Any]:
    quality_counts = Counter(row["alignment_quality"] for row in block_rows)
    risk_counts = Counter()
    review_reason_counts = Counter()
    qwen_diff_counts = Counter()
    auto_accepted_count = 0
    important_term_disagreement_count = 0
    surface_difference_only_count = 0
    soft_domain_difference_count = 0
    critical_term_disagreement_count = 0
    large_span_drift_count_by_engine = Counter()
    large_span_drift_review_count = 0
    usable_distribution = Counter()
    usable_counts = Counter()
    local_scores = Counter()
    drift_abs = Counter()
    refined_counts = Counter()
    unusable_counts = Counter()
    contamination_counts = Counter()
    selected_source_counts = Counter()
    selection_method_counts = Counter()
    final_text_equals_apple_text_count = 0
    selected_source_not_apple_final_equals_apple_count = 0
    final_text_raw_equals_display_count = 0
    final_text_display_changed_count = 0
    punctuation_normalized_count = 0
    punctuation_inserted_period_count = 0
    punctuation_inserted_comma_count = 0
    short_response_period_count = 0
    possible_speaker_change_period_count = 0
    boundary_hint_used_count = 0
    cleanup_reverted_by_punctuation_hint_count = 0
    boundary_cleanup_attempted_count = 0
    boundary_cleanup_applied_count = 0
    boundary_cleanup_reverted_count = 0
    cleanup_validation_failed_count = 0
    leading_fragment_removed_count = 0
    trailing_fragment_removed_count = 0
    protected_prefix_prevented_cleanup_count = 0
    final_text_changed_by_cleanup_count = 0
    post_cleanup_needs_review_count = 0
    pre_llm_needs_review_count = 0
    llm_called_block_count = 0
    llm_selected_count = 0
    llm_resolved_count = 0
    human_review_required_count = 0
    normalized_needs_review_count = 0
    review_reason_char_split_count = 0
    suggested_final_text_count = 0
    boundary_suggestion_count = 0
    domain_candidate_switch_count = 0
    domain_error_avoided_count = 0
    domain_text_corrected_block_count = 0
    domain_text_correction_count = 0
    trailing_boundary_suggestion_count = 0
    large_span_drift_warning_count = 0
    boundary_hint_used_in_alignment_blocking = False
    boundary_hint_used_in_candidate_boundary_eval = False
    boundary_hint_used_in_cleanup = False
    conversation_boundary_hint_stage = "after_apple_sentence_units_before_alignment_blocks"
    boundary_hint_applied_sources = set()
    boundary_hint_applied_count_by_source = Counter()
    short_response_period_count_by_source = Counter()
    boundary_text_generated_count_by_source = Counter()
    display_text_generated_count_by_source = Counter()
    selected_source_by_block_id = {str(block.get("block_id")): str(block.get("selected_source", "unknown")) for block in final_blocks}
    selected_refinement_sec_by_engine = Counter()
    nonselected_refinement_sec_by_engine = Counter()
    selected_refinement_eval_by_engine = Counter()
    nonselected_refinement_eval_by_engine = Counter()
    nonselected_refinement_block_count_by_engine = Counter()
    for row in block_rows:
        risk_counts.update(row.get("risk_flags", []))
        selected_source_for_block = selected_source_by_block_id.get(str(row.get("block_id")), "unknown")
        row_stage_sec = row.get("block_build_sec_by_stage", {}) if isinstance(row.get("block_build_sec_by_stage"), dict) else {}
        for source in ("apple", "qwen", "nemotron", "whisper"):
            candidate = row.get(source)
            if isinstance(candidate, dict):
                if source != "apple":
                    refinement_sec = float(row_stage_sec.get(f"{source}_refinement", 0.0) or 0.0)
                    eval_count = int(candidate.get("refinement_candidate_eval_count", 0) or 0)
                    if selected_source_for_block == source:
                        selected_refinement_sec_by_engine[source] += refinement_sec
                        selected_refinement_eval_by_engine[source] += eval_count
                    else:
                        nonselected_refinement_sec_by_engine[source] += refinement_sec
                        nonselected_refinement_eval_by_engine[source] += eval_count
                        if refinement_sec > 0 or eval_count > 0:
                            nonselected_refinement_block_count_by_engine[source] += 1
                hints = list(candidate.get("boundary_hints", []) or [])
                if hints:
                    boundary_hint_applied_sources.add(source)
                    boundary_hint_applied_count_by_source[source] += len(hints)
                    for hint in hints:
                        if hint.get("type") == "short_response_period":
                            short_response_period_count_by_source[source] += 1
                            short_response_period_count += 1
                if candidate.get("boundary_text"):
                    boundary_text_generated_count_by_source[source] += 1
                if candidate.get("display_text"):
                    display_text_generated_count_by_source[source] += 1
        if row.get("apple_boundary_hints"):
            boundary_hint_used_in_candidate_boundary_eval = True
        for engine in ("qwen", "apple", "nemotron", "whisper"):
            candidate = row.get(engine)
            if not isinstance(candidate, dict):
                continue
            if candidate.get("span_refined"):
                refined_counts[engine] += 1
            if candidate.get("usable_for_agreement"):
                usable_counts[engine] += 1
            local_scores[engine] += float(candidate.get("local_alignment_score", candidate.get("alignment_score", 0.0)))
            drift_abs[engine] += abs(int(candidate.get("span_drift_start", 0))) + abs(int(candidate.get("span_drift_end", 0)))
            if candidate.get("unusable_reason"):
                unusable_reason = candidate.get("unusable_reason")
                if isinstance(unusable_reason, list):
                    unusable_reason = "|".join(str(item) for item in unusable_reason)
                unusable_counts[str(unusable_reason)] += 1
            if candidate.get("boundary_contamination"):
                contamination_counts[engine] += 1
            if abs(int(candidate.get("span_drift_start", 0))) > 80 or abs(int(candidate.get("span_drift_end", 0))) > 80:
                large_span_drift_count_by_engine[engine] += 1
        if row.get("needs_review"):
            review_reason_counts.update(row.get("risk_flags", []))
            if row.get("large_span_drift") and float(row.get("qwen", {}).get("local_alignment_score", 0.0)) < 0.90:
                large_span_drift_review_count += 1
        else:
            auto_accepted_count += 1
        qwen_diff_counts[row.get("qwen_apple_difference_type", "unknown")] += 1
        if row.get("important_term_disagreement"):
            important_term_disagreement_count += 1
        if row.get("qwen_apple_difference_type") == "surface":
            surface_difference_only_count += 1
        if row.get("qwen_apple_difference_type") == "soft_domain":
            soft_domain_difference_count += 1
        if row.get("critical_term_disagreement"):
            critical_term_disagreement_count += 1
        usable_distribution[row.get("candidate_agreement_score", 0.0) >= 0.7] += 1
    for block in final_blocks:
        selected_source_counts[block.get("selected_source", "unknown")] += 1
        selection_method_counts[block.get("selection_method", "unknown")] += 1
        if block.get("llm_used"):
            llm_called_block_count += 1
        if block.get("llm_selected"):
            llm_selected_count += 1
        if block.get("llm_resolved"):
            llm_resolved_count += 1
        if block.get("pre_llm_needs_review"):
            pre_llm_needs_review_count += 1
        if block.get("normalized_needs_review"):
            normalized_needs_review_count += 1
        if block.get("human_review_required"):
            human_review_required_count += 1
        if block.get("boundary_cleanup_attempted"):
            boundary_cleanup_attempted_count += 1
            boundary_hint_used_in_cleanup = True
        if block.get("boundary_cleanup_applied"):
            boundary_cleanup_applied_count += 1
        if block.get("boundary_cleanup_reverted"):
            boundary_cleanup_reverted_count += 1
        if block.get("cleanup_validation_failed"):
            cleanup_validation_failed_count += 1
        if block.get("protected_prefix_prevented_cleanup"):
            protected_prefix_prevented_cleanup_count += 1
        boundary_reasons = block.get("boundary_cleanup_reason", [])
        if isinstance(boundary_reasons, list):
            if "leading_fragment_removed" in boundary_reasons:
                leading_fragment_removed_count += 1
            if "trailing_fragment_removed" in boundary_reasons:
                trailing_fragment_removed_count += 1
        if str(block.get("final_text_before_cleanup", block.get("final_text", ""))) != str(block.get("final_text", "")):
            final_text_changed_by_cleanup_count += 1
        if str(block.get("final_text_raw", block.get("final_text", ""))) == str(block.get("final_text_display", block.get("final_text", ""))):
            final_text_raw_equals_display_count += 1
        if str(block.get("final_text_raw", block.get("final_text", ""))) != str(block.get("final_text_display", block.get("final_text", ""))):
            final_text_display_changed_count += 1
        if block.get("punctuation_hints"):
            punctuation_normalized_count += 1
            punctuation_inserted_period_count += int(block.get("punctuation_inserted_period_count", 0))
            punctuation_inserted_comma_count += int(block.get("punctuation_inserted_comma_count", 0))
            possible_speaker_change_period_count += int(block.get("possible_speaker_change_period_count", 0))
            boundary_hint_used_count += int(block.get("boundary_hint_used_count", 0))
        if block.get("apple_boundary_hints"):
            boundary_hint_used_count += len(block.get("apple_boundary_hints", []))
        if block.get("needs_review") and block.get("boundary_cleanup_applied"):
            post_cleanup_needs_review_count += 1
        if str(block.get("final_text", "")) == str(block.get("candidate_summary", {}).get("apple", "")):
            final_text_equals_apple_text_count += 1
        if block.get("selected_source") != "apple" and str(block.get("final_text", "")) == str(block.get("candidate_summary", {}).get("apple", "")):
            selected_source_not_apple_final_equals_apple_count += 1
        if block.get("suggested_final_text") and str(block.get("suggested_final_text")) != str(block.get("final_text", "")):
            suggested_final_text_count += 1
        if "leading_boundary_fragment_suggested" in (block.get("suggested_final_text_reason") or []):
            boundary_suggestion_count += 1
        if "trailing_boundary_fragment_suggested" in (block.get("suggested_final_text_reason") or []):
            trailing_boundary_suggestion_count += 1
        if block.get("domain_candidate_switched"):
            domain_candidate_switch_count += 1
        if block.get("domain_text_corrected"):
            domain_text_corrected_block_count += 1
            domain_text_correction_count += len(block.get("domain_text_corrections") or [])
        if "domain_error_avoided" in (block.get("risk_flags") or []):
            domain_error_avoided_count += 1
        if "large_span_drift_warning" in (block.get("risk_flags") or []):
            large_span_drift_warning_count += 1
    for row in review_rows:
        reasons = row.get("review_reason", [])
        if isinstance(reasons, list) and reasons and all(len(str(reason)) == 1 for reason in reasons):
            review_reason_char_split_count += 1
    final_risk_flag_counts = Counter()
    final_needs_review_reason_counts = Counter()
    review_level_counts = Counter()
    review_sample_level_counts = Counter()
    unusual_final_text_pattern_count = 0
    unusual_final_text_pattern_human_required_count = 0
    unusual_final_text_pattern_machine_note_count = 0
    unusual_final_text_patterns_by_level = {"human_required": Counter(), "machine_note": Counter()}
    qwen_alignment_low_by_selected_source = Counter()
    qwen_alignment_low_human_required_by_selected_source = Counter()
    for block in final_blocks:
        final_risk_flag_counts.update(block.get("risk_flags", []) or [])
        if block.get("human_review_required") or block.get("needs_review"):
            final_needs_review_reason_counts.update(block.get("human_review_reason") or block.get("review_reason", []) or [])
        review_level_counts[str(block.get("review_level") or "auto_accept")] += 1
        review_sample_level_counts[str(block.get("review_level") or "auto_accept")] += 1
        selected_source = str(block.get("selected_source") or "unknown")
        qwen_alignment = float(block.get("qwen", {}).get("local_alignment_score", 0.0)) if isinstance(block.get("qwen"), dict) else 0.0
        if qwen_alignment < 0.82:
            qwen_alignment_low_by_selected_source[selected_source] += 1
            if block.get("human_review_required"):
                qwen_alignment_low_human_required_by_selected_source[selected_source] += 1
        if "unusual_final_text_pattern" in (block.get("final_risk_flags") or block.get("risk_flags") or []):
            unusual_final_text_pattern_count += 1
            if block.get("human_review_required"):
                unusual_final_text_pattern_human_required_count += 1
                unusual_final_text_patterns_by_level["human_required"].update(
                    [str(block.get("final_text", ""))[:40]]
                )
            elif block.get("machine_review_note"):
                unusual_final_text_pattern_machine_note_count += 1
                unusual_final_text_patterns_by_level["machine_note"].update(
                    [str(block.get("final_text", ""))[:40]]
                )
    sentence_units = apple_timeline["apple_sentence_units"]
    blocks = apple_timeline["alignment_blocks"]
    if any(unit.get("boundary_hints") for unit in sentence_units):
        boundary_hint_used_in_alignment_blocking = True
    avg_sentences_per_block = round(sum(len(b["sentence_ids"]) for b in blocks) / max(len(blocks), 1), 2)
    summary = {
        "episode_id": episode_id,
        "input_dir": str(input_dir),
        "input_files": sorted(str(path) for path in source_files.values()),
        "apple_sentence_unit_count": len(sentence_units),
        "alignment_block_count": len(blocks),
        "avg_sentences_per_block": avg_sentences_per_block,
        "min_sentence_length": validation["sentence"]["min_sentence_length"],
        "max_sentence_length": validation["sentence"]["max_sentence_length"],
        "very_short_sentence_count": validation["sentence"]["very_short_sentence_count"],
        "block_text_range_mismatch_count": validation["block"]["block_text_range_mismatch_count"],
        "sentence_block_consistency_error_count": validation["block"]["sentence_block_consistency_error_count"],
        "apple_stable_full_text_length": len(apple_timeline["apple_stable_full_text"]),
        "asr_text_lengths": {engine: alignments[engine].asr_text_length for engine in alignments},
        "global_alignment_scores": {engine: alignments[engine].global_alignment_score for engine in alignments},
        "coverage_ratios": {engine: alignments[engine].coverage_ratio for engine in alignments},
        "alignment_quality_counts": dict(sorted(quality_counts.items())),
        "needs_review_count": sum(1 for row in block_rows if row["needs_review"]),
        "pre_llm_needs_review_count": pre_llm_needs_review_count,
        "llm_selected_count": llm_selected_count,
        "llm_resolved_count": llm_resolved_count,
        "human_review_required_count": human_review_required_count,
        "normalized_needs_review_count": normalized_needs_review_count,
        "final_block_count": len(final_blocks),
        "final_sentence_timeline_count": len(final_rows),
        "review_level_counts": dict(review_level_counts),
        "review_sample_level_counts": dict(review_sample_level_counts),
        "unusual_final_text_pattern_count": unusual_final_text_pattern_count,
        "unusual_final_text_pattern_human_required_count": unusual_final_text_pattern_human_required_count,
        "unusual_final_text_pattern_machine_note_count": unusual_final_text_pattern_machine_note_count,
        "unusual_final_text_patterns_by_level": {
            "human_required": dict(unusual_final_text_patterns_by_level["human_required"]),
            "machine_note": dict(unusual_final_text_patterns_by_level["machine_note"]),
        },
        "qwen_alignment_low_by_selected_source": dict(qwen_alignment_low_by_selected_source),
        "qwen_alignment_low_human_required_by_selected_source": dict(qwen_alignment_low_human_required_by_selected_source),
        "candidate_build_wall_sec": block_summary.get("candidate_build_wall_sec", block_summary.get("candidate_build_total_sec")),
        "support_global_alignment_skip_threshold": block_summary.get("support_global_alignment_skip_threshold"),
        "candidate_build_cumulative_stage_sec": block_summary.get("candidate_build_cumulative_stage_sec"),
        "candidate_build_sec_by_stage": block_summary.get("candidate_build_sec_by_stage", {}),
        "candidate_build_sec_by_stage_mean": block_summary.get("candidate_build_sec_by_stage_mean", {}),
        "candidate_refinement_executed_count_by_engine": block_summary.get("candidate_refinement_executed_count_by_engine", {}),
        "candidate_refinement_skipped_count_by_engine": block_summary.get("candidate_refinement_skipped_count_by_engine", {}),
        "candidate_refinement_skipped_reason_counts": block_summary.get("candidate_refinement_skipped_reason_counts", {}),
        "refinement_search_profile_counts": block_summary.get("refinement_search_profile_counts", {}),
        "refinement_candidate_eval_sum_by_engine": block_summary.get("refinement_candidate_eval_sum_by_engine", {}),
        "refinement_candidate_eval_max_by_engine": block_summary.get("refinement_candidate_eval_max_by_engine", {}),
        "refinement_candidate_pruned_sum_by_engine": block_summary.get("refinement_candidate_pruned_sum_by_engine", {}),
        "refinement_candidate_pruned_max_by_engine": block_summary.get("refinement_candidate_pruned_max_by_engine", {}),
        "qwen_high_confidence_reject_reason_counts": block_summary.get("qwen_high_confidence_reject_reason_counts", {}),
        "qwen_high_confidence_primary_reject_reason_counts": block_summary.get("qwen_high_confidence_primary_reject_reason_counts", {}),
        "candidate_refinement_early_exit_count_by_engine": block_summary.get("candidate_refinement_early_exit_count_by_engine", {}),
        "candidate_refinement_early_exit_block_ids_by_engine": block_summary.get("candidate_refinement_early_exit_block_ids_by_engine", {}),
        "cheap_span_accept_count_by_engine": block_summary.get("cheap_span_accept_count_by_engine", {}),
        "heavy_refinement_skipped_count_by_engine": block_summary.get("heavy_refinement_skipped_count_by_engine", {}),
        "candidate_build_slowest_blocks": block_summary.get("candidate_build_slowest_blocks", []),
        "candidate_build_slowest_block_ids": block_summary.get("candidate_build_slowest_block_ids", []),
        "candidate_build_slowest_dominant_stage_counts": block_summary.get("candidate_build_slowest_dominant_stage_counts", {}),
        "candidate_build_block_time_sum": block_summary.get("candidate_build_block_time_sum"),
        "candidate_build_slowest_block_time_sum": block_summary.get("candidate_build_slowest_block_time_sum"),
        "candidate_build_slowest_block_time_ratio": block_summary.get("candidate_build_slowest_block_time_ratio"),
        "candidate_build_slow_block_threshold_sec": block_summary.get("candidate_build_slow_block_threshold_sec"),
        "candidate_build_slow_block_count": block_summary.get("candidate_build_slow_block_count"),
        "selected_refinement_sec_by_engine": dict(selected_refinement_sec_by_engine),
        "nonselected_refinement_sec_by_engine": dict(nonselected_refinement_sec_by_engine),
        "selected_refinement_eval_by_engine": dict(selected_refinement_eval_by_engine),
        "nonselected_refinement_eval_by_engine": dict(nonselected_refinement_eval_by_engine),
        "nonselected_refinement_block_count_by_engine": dict(nonselected_refinement_block_count_by_engine),
        "final_text_equals_apple_text_count": final_text_equals_apple_text_count,
        "selected_source_not_apple_final_equals_apple_count": selected_source_not_apple_final_equals_apple_count,
        "selected_source_counts": dict(selected_source_counts),
        "selection_method_counts": dict(selection_method_counts),
        "conversation_boundary_hint_stage": conversation_boundary_hint_stage,
        "boundary_hint_applied_sources": sorted(boundary_hint_applied_sources),
        "boundary_hint_applied_count_by_source": dict(boundary_hint_applied_count_by_source),
        "short_response_period_count_by_source": dict(short_response_period_count_by_source),
        "boundary_text_generated_count_by_source": dict(boundary_text_generated_count_by_source),
        "display_text_generated_count_by_source": dict(display_text_generated_count_by_source),
        "boundary_hint_generated_before_alignment_blocks": bool(any(unit.get("boundary_hints") for unit in sentence_units)),
        "boundary_hint_used_in_alignment_blocking": boundary_hint_used_in_alignment_blocking,
        "boundary_hint_used_in_candidate_boundary_eval": boundary_hint_used_in_candidate_boundary_eval,
        "boundary_hint_used_in_cleanup": boundary_hint_used_in_cleanup,
        "final_text_raw_equals_display_count": final_text_raw_equals_display_count,
        "final_text_raw_display_mismatch_count": sum(
            1
            for row in final_blocks
            if str(row.get("final_text_raw", "")) != str(row.get("final_text_display", ""))
        ),
        "final_transcript_uses_accepted_llm_text": True,
        "final_text_display_changed_count": final_text_display_changed_count,
        "punctuation_normalized_count": punctuation_normalized_count,
        "punctuation_inserted_period_count": punctuation_inserted_period_count,
        "punctuation_inserted_comma_count": punctuation_inserted_comma_count,
        "short_response_period_count": short_response_period_count,
        "possible_speaker_change_period_count": possible_speaker_change_period_count,
        "boundary_hint_used_count": boundary_hint_used_count,
        "cleanup_reverted_by_punctuation_hint_count": cleanup_reverted_by_punctuation_hint_count,
        "boundary_cleanup_attempted_count": boundary_cleanup_attempted_count,
        "boundary_cleanup_applied_count": boundary_cleanup_applied_count,
        "boundary_cleanup_reverted_count": boundary_cleanup_reverted_count,
        "cleanup_validation_failed_count": cleanup_validation_failed_count,
        "leading_fragment_removed_count": leading_fragment_removed_count,
        "trailing_fragment_removed_count": trailing_fragment_removed_count,
        "protected_prefix_prevented_cleanup_count": protected_prefix_prevented_cleanup_count,
        "post_cleanup_needs_review_count": post_cleanup_needs_review_count,
        "final_text_changed_by_cleanup_count": final_text_changed_by_cleanup_count,
        "llm_called_block_count": llm_called_block_count,
        "review_reason_char_split_count": review_reason_char_split_count,
        "suggested_final_text_count": suggested_final_text_count,
        "boundary_suggestion_count": boundary_suggestion_count,
        "trailing_boundary_suggestion_count": trailing_boundary_suggestion_count,
        "domain_candidate_switch_count": domain_candidate_switch_count,
        "domain_error_avoided_count": domain_error_avoided_count,
        "domain_text_corrected_block_count": domain_text_corrected_block_count,
        "domain_text_correction_count": domain_text_correction_count,
        "large_span_drift_warning_count": large_span_drift_warning_count,
        "auto_accepted_count": auto_accepted_count,
        "auto_accepted_ratio": round(auto_accepted_count / max(len(block_rows), 1), 3),
        "risk_flag_counts": dict(risk_counts.most_common()),
        "boundary_contamination_suspected_count": risk_counts.get("boundary_contamination_suspected", 0),
        "needs_review_reason_counts": dict(review_reason_counts.most_common()),
        "final_risk_flag_counts": dict(final_risk_flag_counts.most_common()),
        "final_needs_review_reason_counts": dict(final_needs_review_reason_counts.most_common()),
        "qwen_apple_difference_type_counts": dict(qwen_diff_counts),
        "usable_agreement_candidate_count_distribution": dict(usable_distribution),
        "critical_term_disagreement_count": critical_term_disagreement_count,
        "soft_domain_difference_count": soft_domain_difference_count,
        "soft_domain_auto_accepted_count": sum(1 for row in block_rows if row.get("qwen_apple_difference_type") == "soft_domain" and not row.get("needs_review")),
        "large_span_drift_count_by_engine": dict(large_span_drift_count_by_engine),
        "large_span_drift_review_count": large_span_drift_review_count,
        "important_term_disagreement_count": important_term_disagreement_count,
        "surface_difference_only_count": surface_difference_only_count,
        "usable_candidate_count_by_engine": dict(usable_counts),
        "refined_candidate_count_by_engine": dict(refined_counts),
        "boundary_contamination_count_by_engine": dict(contamination_counts),
        "unusable_reason_counts": dict(unusable_counts),
        "local_alignment_score_sum_by_engine": dict(local_scores),
        "span_drift_abs_sum_by_engine": dict(drift_abs),
        "sample_blocks": block_rows[:10],
        "problem_blocks": [row for row in block_rows if row["needs_review"]][:10],
        "llm_used": bool(llm_stats.get("llm_used")),
        "llm_call_count": llm_stats.get("llm_call_count", 0),
        "llm_success_count": llm_stats.get("llm_success_count", 0),
        "llm_failure_count": llm_stats.get("llm_failure_count", 0),
        "llm_timeout_count": llm_stats.get("llm_timeout_count", 0),
        "llm_timeout_fallback_applied_count": llm_stats.get("llm_timeout_fallback_applied_count", 0),
        "qwen_suspicious_timeout_fallback_count": llm_stats.get("qwen_suspicious_timeout_fallback_count", 0),
        "llm_cache_hit_count": llm_stats.get("llm_cache_hit_count", 0),
        "llm_changed_final_text_count": llm_stats.get("llm_changed_final_text_count", 0),
        "llm_changed_needs_review_true_count": llm_stats.get("llm_changed_needs_review_true_count", 0),
        "llm_changed_needs_review_false_count": llm_stats.get("llm_changed_needs_review_false_count", 0),
        "final_segment_count": len(final_rows),
        "review_queue_count": len(review_rows),
        "output_dir": str(output_dir),
    }
    return summary


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['episode_id']} サマリー", ""]
    lines.append("## 入力")
    for path in summary["input_files"]:
        lines.append(f"- {path}")
    lines.append("")
    lines.append("## 件数")
    lines.append(f"- Apple sentence unit 数: {summary['apple_sentence_unit_count']}")
    lines.append(f"- alignment block 数: {summary['alignment_block_count']}")
    lines.append(f"- block あたり平均 sentence 数: {summary['avg_sentences_per_block']}")
    lines.append(f"- Apple 安定全文の文字数: {summary['apple_stable_full_text_length']}")
    lines.append(f"- 要確認件数: {summary['needs_review_count']}")
    lines.append(f"- 要確認件数(final blocks): {summary['needs_review_count_final_blocks']}")
    lines.append(f"- 要確認件数(normalized blocks): {summary['needs_review_count_normalized_blocks']}")
    lines.append(f"- human_review_required 件数: {summary.get('human_review_required_count', 0)}")
    lines.append(f"- machine_review_note 件数: {summary.get('machine_review_note_count', 0)}")
    lines.append(f"- auto_accept_final 件数: {summary.get('auto_accept_final_count', 0)}")
    lines.append(f"- 自動採用件数(final blocks): {summary.get('auto_accept_final_count', 0)}")
    lines.append(f"- review_queue 行数: {summary['review_queue_row_count']}")
    lines.append(f"- machine_review_notes 行数: {summary.get('machine_review_notes_row_count', 0)}")
    lines.append(f"- review_queue 単位: {summary['review_queue_unit']}")
    lines.append(f"- review_queue と human_review_required の一致: {summary.get('review_queue_matches_human_review_required')}")
    lines.append(f"- review_level_counts: {summary.get('review_level_counts', {})}")
    lines.append(f"- human_required_before_demote_count: {summary.get('human_required_before_demote_count', 0)}")
    lines.append(f"- demoted_from_human_to_machine_count: {summary.get('demoted_from_human_to_machine_count', 0)}")
    lines.append(f"- human_required_after_demote_count: {summary.get('human_required_after_demote_count', 0)}")
    lines.append(f"- demote_reason_counts: {summary.get('demote_reason_counts', {})}")
    lines.append(f"- recommended_next_action_counts: {summary.get('recommended_next_action_counts', {})}")
    lines.append(f"- recommended_next_action_missing_count: {summary.get('recommended_next_action_missing_count', 0)}")
    lines.append(f"- review_sample_level_counts: {summary.get('review_sample_level_counts', {})}")
    lines.append(f"- qwen_alignment_low_by_selected_source: {summary.get('qwen_alignment_low_by_selected_source', {})}")
    lines.append(f"- qwen_alignment_low_human_required_by_selected_source: {summary.get('qwen_alignment_low_human_required_by_selected_source', {})}")
    lines.append(f"- qwen_alignment_low_machine_note_by_selected_source: {summary.get('qwen_alignment_low_machine_note_by_selected_source', {})}")
    lines.append(f"- unusual_final_text_pattern_count: {summary.get('unusual_final_text_pattern_count', 0)}")
    lines.append(f"- unusual_final_text_pattern_human_required_count: {summary.get('unusual_final_text_pattern_human_required_count', 0)}")
    lines.append(f"- unusual_final_text_pattern_machine_note_count: {summary.get('unusual_final_text_pattern_machine_note_count', 0)}")
    lines.append(f"- unusual_final_text_patterns_by_level: {summary.get('unusual_final_text_patterns_by_level', {})}")
    lines.append(f"- llm_target_max_blocks: {summary.get('llm_target_max_blocks', 10)}")
    lines.append(f"- llm_candidate_block_count: {summary.get('llm_candidate_block_count', 0)}")
    lines.append(f"- llm_target_block_count: {summary.get('llm_target_block_count', 0)}")
    lines.append(f"- llm_skipped_human_required_count: {summary.get('llm_skipped_human_required_count', 0)}")
    lines.append(f"- llm_target_selection_reasons: {summary.get('llm_target_selection_reasons', {})}")
    lines.append(f"- llm_target_block_ids: {summary.get('llm_target_block_ids', [])}")
    lines.append(f"- 自動採用件数(normalized): {summary['auto_accepted_count']}")
    lines.append(f"- 自動採用率(normalized): {summary['auto_accepted_ratio']}")
    lines.append(f"- usable candidate 数: {summary['usable_candidate_count_by_engine']}")
    lines.append(f"- support global alignment skip 閾値: {summary.get('support_global_alignment_skip_threshold')}")
    lines.append(f"- refinement skip 理由: {summary.get('candidate_refinement_skipped_reason_counts', {})}")
    lines.append(f"- refinement search profile: {summary.get('refinement_search_profile_counts', {})}")
    lines.append(f"- refinement candidate eval 合計: {summary.get('refinement_candidate_eval_sum_by_engine', {})}")
    lines.append(f"- refinement candidate eval 最大: {summary.get('refinement_candidate_eval_max_by_engine', {})}")
    lines.append(f"- refinement candidate prune 合計: {summary.get('refinement_candidate_pruned_sum_by_engine', {})}")
    lines.append(f"- refinement candidate prune 最大: {summary.get('refinement_candidate_pruned_max_by_engine', {})}")
    lines.append(f"- Qwen高信頼skip不可理由: {summary.get('qwen_high_confidence_reject_reason_counts', {})}")
    lines.append(f"- Qwen高信頼skip不可の主理由: {summary.get('qwen_high_confidence_primary_reject_reason_counts', {})}")
    lines.append(f"- 遅いblock ID: {summary.get('candidate_build_slowest_block_ids', [])}")
    lines.append(f"- 遅いblock支配stage: {summary.get('candidate_build_slowest_dominant_stage_counts', {})}")
    lines.append(f"- 遅いblock時間合計: {summary.get('candidate_build_slowest_block_time_sum')}")
    lines.append(f"- 遅いblock時間比率: {summary.get('candidate_build_slowest_block_time_ratio')}")
    lines.append(
        f"- slow block数(>={summary.get('candidate_build_slow_block_threshold_sec')}s): "
        f"{summary.get('candidate_build_slow_block_count')}"
    )
    lines.append(f"- selected refinement 秒数: {summary.get('selected_refinement_sec_by_engine', {})}")
    lines.append(f"- non-selected refinement 秒数: {summary.get('nonselected_refinement_sec_by_engine', {})}")
    lines.append(f"- selected refinement eval 数: {summary.get('selected_refinement_eval_by_engine', {})}")
    lines.append(f"- non-selected refinement eval 数: {summary.get('nonselected_refinement_eval_by_engine', {})}")
    lines.append(f"- non-selected refinement block 数: {summary.get('nonselected_refinement_block_count_by_engine', {})}")
    lines.append(f"- conversation_boundary_hint_stage: {summary['conversation_boundary_hint_stage']}")
    lines.append(f"- boundary_hint_applied_sources: {summary['boundary_hint_applied_sources']}")
    lines.append(f"- boundary_hint_applied_count_by_source: {summary['boundary_hint_applied_count_by_source']}")
    lines.append(f"- short_response_period_count_by_source: {summary['short_response_period_count_by_source']}")
    lines.append(f"- boundary_text_generated_count_by_source: {summary['boundary_text_generated_count_by_source']}")
    lines.append(f"- display_text_generated_count_by_source: {summary['display_text_generated_count_by_source']}")
    lines.append(f"- boundary_hint_generated_before_alignment_blocks: {summary['boundary_hint_generated_before_alignment_blocks']}")
    lines.append(f"- boundary_hint_used_in_alignment_blocking: {summary['boundary_hint_used_in_alignment_blocking']}")
    lines.append(f"- boundary_hint_used_in_candidate_boundary_eval: {summary['boundary_hint_used_in_candidate_boundary_eval']}")
    lines.append(f"- boundary_hint_used_in_cleanup: {summary['boundary_hint_used_in_cleanup']}")
    lines.append(f"- sentence_timeline 行数: {summary['sentence_timeline_row_count']}")
    lines.append(f"- sentence_timeline block leak 件数: {summary['sentence_timeline_display_text_block_leak_count']}")
    lines.append(f"- sentence_timeline 長すぎる display_text 件数: {summary['sentence_timeline_display_text_too_long_count']}")
    lines.append(f"- 句点補正件数: {summary['punctuation_normalized_count']}")
    lines.append(f"- 句点挿入数: {summary['punctuation_inserted_period_count']}")
    lines.append(f"- 読点挿入数: {summary['punctuation_inserted_comma_count']}")
    lines.append(f"- short response 句点件数: {summary['short_response_period_count']}")
    lines.append(f"- speaker change 句点件数: {summary['possible_speaker_change_period_count']}")
    lines.append(f"- boundary hint 使用件数: {summary['boundary_hint_used_count']}")
    lines.append(f"- cleanup reverted by punctuation hint: {summary['cleanup_reverted_by_punctuation_hint_count']}")
    lines.append(f"- boundary cleanup 試行件数: {summary['boundary_cleanup_attempted_count']}")
    lines.append(f"- boundary cleanup 適用件数: {summary['boundary_cleanup_applied_count']}")
    lines.append(f"- boundary cleanup 巻き戻し件数: {summary['boundary_cleanup_reverted_count']}")
    lines.append(f"- cleanup validation 失敗件数: {summary['cleanup_validation_failed_count']}")
    lines.append(f"- leading fragment 削除件数: {summary['leading_fragment_removed_count']}")
    lines.append(f"- trailing fragment 削除件数: {summary['trailing_fragment_removed_count']}")
    lines.append(f"- protected prefix による抑止件数: {summary['protected_prefix_prevented_cleanup_count']}")
    lines.append(f"- cleanup 後の要確認件数: {summary['post_cleanup_needs_review_count']}")
    lines.append(f"- cleanup による final_text 変更件数: {summary['final_text_changed_by_cleanup_count']}")
    lines.append(f"- final_text_raw == display 件数: {summary['final_text_raw_equals_display_count']}")
    lines.append(f"- final_text_display 変更件数: {summary['final_text_display_changed_count']}")
    lines.append(f"- final_text_raw_display_mismatch_count: {summary.get('final_text_raw_display_mismatch_count', 0)}")
    lines.append(f"- final_transcript_uses_accepted_llm_text: {summary.get('final_transcript_uses_accepted_llm_text')}")
    lines.append(f"- suggested_final_text 件数: {summary.get('suggested_final_text_count', 0)}")
    lines.append(f"- boundary suggestion 件数: {summary.get('boundary_suggestion_count', 0)}")
    lines.append(f"- trailing boundary suggestion 件数: {summary.get('trailing_boundary_suggestion_count', 0)}")
    lines.append(f"- domain candidate switch 件数: {summary.get('domain_candidate_switch_count', 0)}")
    lines.append(f"- domain error avoided 件数: {summary.get('domain_error_avoided_count', 0)}")
    lines.append(f"- domain text corrected block 件数: {summary.get('domain_text_corrected_block_count', 0)}")
    lines.append(f"- domain text correction 件数: {summary.get('domain_text_correction_count', 0)}")
    lines.append(f"- large span drift warning 件数: {summary.get('large_span_drift_warning_count', 0)}")
    lines.append(f"- regression_phrase_counts: {summary.get('regression_phrase_counts', {})}")
    lines.append(f"- domain_error_phrase_counts: {summary.get('domain_error_phrase_counts', {})}")
    lines.append(f"- 出力検証: {summary.get('output_validation', {})}")
    lines.append(f"- missing_output_files: {summary.get('missing_output_files', [])}")
    lines.append("")
    lines.append("## 品質")
    for quality, count in summary["alignment_quality_counts"].items():
        lines.append(f"- {quality}: {count}")
    lines.append("")
    lines.append("## 検証")
    lines.append(f"- sentence 最小長: {summary['min_sentence_length']}")
    lines.append(f"- sentence 最大長: {summary['max_sentence_length']}")
    lines.append(f"- 非常に短い sentence 数: {summary['very_short_sentence_count']}")
    lines.append(f"- block_text_range_mismatch_count: {summary['block_text_range_mismatch_count']}")
    lines.append(f"- sentence_block_consistency_error_count: {summary['sentence_block_consistency_error_count']}")
    lines.append("")
    lines.append("## リスクフラグ")
    for name, count in summary["risk_flag_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## 差分判定")
    lines.append(f"- qwen_apple_difference_type_counts: {summary['qwen_apple_difference_type_counts']}")
    lines.append(f"- important_term_disagreement_count: {summary['important_term_disagreement_count']}")
    lines.append(f"- surface_difference_only_count: {summary['surface_difference_only_count']}")
    lines.append("")
    lines.append("## LLM")
    lines.append(f"- 使用: {summary['llm_used']}")
    lines.append(f"- 呼び出し回数: {summary['llm_call_count']}")
    lines.append(f"- 成功回数: {summary['llm_success_count']}")
    lines.append(f"- 失敗回数: {summary['llm_failure_count']}")
    lines.append(f"- キャッシュヒット数: {summary['llm_cache_hit_count']}")
    lines.append("")
    lines.append("## block サンプル")
    for row in summary["sample_blocks"]:
        lines.append(f"- {row['block_id']} {row['alignment_quality']} review={row['needs_review']} text={row['apple']['text'][:80]}")
    if summary.get("candidate_build_slowest_blocks"):
        lines.append("")
        lines.append("## 遅いblock上位")
        for row in summary["candidate_build_slowest_blocks"][:10]:
            lines.append(
                f"- {row.get('block_id')} {row.get('block_build_total_sec')}s "
                f"quality={row.get('alignment_quality')} review={row.get('needs_review')} "
                f"diff={row.get('qwen_apple_difference_type')} stages={row.get('stage_sec', {})}"
            )
    if summary.get("review_sample_path"):
        lines.append("")
        lines.append("## review サンプル")
        lines.append(f"- 全文サンプル: {summary['review_sample_path']}")
    lines.append("")
    return "\n".join(lines)


def _sanitize_review_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(key): _sanitize_review_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_review_value(item) for item in value]
    return value


def _review_candidate_texts(block: dict[str, Any]) -> dict[str, str]:
    final_text = str(block.get("final_text_display") or block.get("final_text") or block.get("final_text_raw") or "(missing)")
    candidate_texts = block.get("candidate_texts")
    if not isinstance(candidate_texts, dict):
        candidate_texts = {}
    return {
        "apple": str(block.get("apple_text") or candidate_texts.get("apple") or final_text),
        "qwen": str(block.get("qwen_text") or candidate_texts.get("qwen") or final_text),
        "nemotron": str(block.get("nemotron_text") or candidate_texts.get("nemotron") or final_text),
        "whisper": str(block.get("whisper_text") or candidate_texts.get("whisper") or final_text),
    }


def _build_review_sample_rows(final_blocks: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    review_rows = [row for row in final_blocks if row.get("human_review_required")]
    sample_rows: list[dict[str, Any]] = []
    for row in review_rows[:limit]:
        sample_row = {
            "episode_id": row.get("episode_id", ""),
            "block_id": row.get("block_id", ""),
            "sentence_ids": list(row.get("sentence_ids") or []),
            "time": _sanitize_review_value(row.get("time") or {}),
            "alignment_quality": row.get("alignment_quality", ""),
            "needs_review": bool(row.get("needs_review")),
            "pre_llm_needs_review": bool(row.get("pre_llm_needs_review")),
            "normalized_needs_review": bool(row.get("normalized_needs_review")),
            "human_review_required": bool(row.get("human_review_required")),
            "machine_review_note": bool(row.get("machine_review_note")),
            "review_level": row.get("review_level", ""),
            "review_priority": row.get("review_priority", ""),
            "review_gate_reasons": list(row.get("review_gate_reasons") or []),
            "machine_note_reasons": list(row.get("machine_note_reasons") or []),
            "qwen_apple_difference_type": row.get("qwen_apple_difference_type", ""),
            "qwen_apple_similarity": float(row.get("qwen_apple_similarity") if row.get("qwen_apple_similarity") is not None else 0.0),
            "risk_flags": list(row.get("risk_flags") or []),
            "final_risk_flags": list(row.get("final_risk_flags") or row.get("risk_flags") or []),
            "selected_source": row.get("selected_source", ""),
            "selection_method": row.get("selection_method", ""),
            "llm_called": bool(row.get("llm_called")),
            "llm_selected": bool(row.get("llm_selected")),
            "llm_resolved": bool(row.get("llm_resolved")),
            "final_text_raw": row.get("final_text_raw") or row.get("final_text", "") or "",
            "final_text_display": row.get("final_text_display") or row.get("final_text", "") or "",
            "apple_text": row.get("apple_text") or row.get("final_text_display") or row.get("final_text", "") or "(missing)",
            "qwen_text": row.get("qwen_text") or row.get("final_text_display") or row.get("final_text", "") or "(missing)",
            "nemotron_text": row.get("nemotron_text") or row.get("final_text_display") or row.get("final_text", "") or "(missing)",
            "whisper_text": row.get("whisper_text") or row.get("final_text_display") or row.get("final_text", "") or "(missing)",
            "candidate_texts": _review_candidate_texts(row),
            "unusual_final_text_patterns": list(row.get("unusual_final_text_patterns") or []),
        }
        sample_rows.append(_sanitize_review_value(sample_row))
    return sample_rows


def _review_summary_metrics(final_blocks: list[dict[str, Any]], review_rows: list[dict[str, Any]], review_sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    qwen_alignment_low_by_selected_source = Counter()
    qwen_alignment_low_human_required_by_selected_source = Counter()
    qwen_alignment_low_machine_note_by_selected_source = Counter({"apple": 0, "qwen": 0, "nemotron": 0, "whisper": 0})
    unusual_final_text_patterns_by_level = {"human_required": Counter(), "machine_note": Counter()}
    unusual_final_text_pattern_count = 0
    unusual_final_text_pattern_human_required_count = 0
    unusual_final_text_pattern_machine_note_count = 0
    review_sample_level_counts = Counter()
    llm_resolved_human_review_count = 0
    llm_kept_human_review_count = 0
    llm_changed_final_text_count = 0
    llm_no_change_count = 0
    llm_candidate_out_of_set_violation_count = 0
    human_required_before_demote_count = 0
    demoted_from_human_to_machine_count = 0
    human_required_after_demote_count = 0
    demote_reason_counts = Counter()
    recommended_next_action_counts = Counter()
    recommended_next_action_missing_count = 0

    def _target_reasons(block: dict[str, Any]) -> list[str]:
        flags = set(str(flag) for flag in (block.get("final_risk_flags") or block.get("risk_flags") or []))
        review_gate_reasons = set(str(reason) for reason in (block.get("review_gate_reasons") or []))
        unusual_patterns = list(block.get("unusual_final_text_patterns") or [])
        reasons: list[str] = []
        if "numeric_disagreement" in flags:
            reasons.append("numeric_disagreement")
        if "critical_term_disagreement" in flags:
            reasons.append("critical_term_disagreement")
        if "domain_error_phrase" in flags:
            reasons.append("domain_error_phrase")
        if any(isinstance(item, dict) and item.get("severity") == "human_required" for item in unusual_patterns):
            reasons.append("genuine_unusual_final_text_pattern")
        if "boundary_contamination_suspected" in flags:
            reasons.append("boundary_contamination_suspected")
        selected_source = str(block.get("selected_source") or "")
        selected_alignment = float(block.get("selected_source_alignment_score") or 0.0)
        if selected_source and selected_source != "apple" and selected_alignment < 0.82:
            reasons.append("selected_source_alignment_low")
        if "numeric_disagreement" in review_gate_reasons and "numeric_disagreement" not in reasons:
            reasons.append("numeric_disagreement")
        return reasons

    for row in review_sample_rows:
        review_sample_level_counts[str(row.get("review_level") or "unknown")] += 1

    for block in final_blocks:
        selected_source = str(block.get("selected_source") or "unknown")
        recommended_next_action = block.get("recommended_next_action")
        if recommended_next_action:
            recommended_next_action_counts[str(recommended_next_action)] += 1
        else:
            recommended_next_action_missing_count += 1
        if block.get("demoted_from_human_required"):
            demoted_from_human_to_machine_count += 1
            for reason in block.get("demote_reason") or []:
                demote_reason_counts[str(reason)] += 1
        if "qwen_alignment_low" in (block.get("review_gate_reasons") or []) or "qwen_alignment_low" in (block.get("machine_note_reasons") or []):
            qwen_alignment_low_by_selected_source[selected_source] += 1
            if block.get("human_review_required"):
                qwen_alignment_low_human_required_by_selected_source[selected_source] += 1
            if block.get("machine_review_note"):
                qwen_alignment_low_machine_note_by_selected_source[selected_source] += 1
        unusual_patterns = list(block.get("unusual_final_text_patterns") or [])
        if unusual_patterns:
            unusual_final_text_pattern_count += 1
            review_level = str(block.get("review_level") or "unknown")
            if review_level not in unusual_final_text_patterns_by_level:
                unusual_final_text_patterns_by_level[review_level] = Counter()
            for pattern in unusual_patterns:
                pattern_id = str(pattern.get("pattern_id") or pattern.get("matched_text") or "unknown")
                unusual_final_text_patterns_by_level[review_level][pattern_id] += 1
            if block.get("human_review_required"):
                unusual_final_text_pattern_human_required_count += 1
            if block.get("machine_review_note"):
                unusual_final_text_pattern_machine_note_count += 1
        if block.get("llm_selected"):
            if block.get("llm_resolved"):
                llm_resolved_human_review_count += 1
            elif block.get("human_review_required"):
                llm_kept_human_review_count += 1
            if block.get("llm_changed_final_text"):
                llm_changed_final_text_count += 1
            elif str(block.get("final_text_before_cleanup") or "") == str(block.get("final_text") or ""):
                llm_no_change_count += 1
            else:
                llm_changed_final_text_count += 1

    human_review_required_count = sum(1 for row in final_blocks if row.get("human_review_required"))
    human_required_after_demote_count = human_review_required_count
    human_required_before_demote_count = human_review_required_count + demoted_from_human_to_machine_count
    review_queue_row_count = len(review_rows)
    llm_candidate_block_count = sum(1 for row in final_blocks if row.get("llm_candidate"))
    llm_target_block_count = sum(1 for row in final_blocks if row.get("llm_target"))
    llm_skipped_human_required_count = max(0, llm_candidate_block_count - llm_target_block_count)
    llm_target_block_ids = [str(row.get("block_id") or "") for row in final_blocks if row.get("llm_target")]
    llm_target_selection_reasons = Counter()
    for row in final_blocks:
        if row.get("llm_target"):
            for reason in row.get("review_gate_reasons") or []:
                llm_target_selection_reasons[str(reason)] += 1
    return {
        "review_queue_row_count": review_queue_row_count,
        "machine_review_notes_row_count": sum(1 for row in final_blocks if row.get("review_level") == "machine_note"),
        "needs_review_count_for_review_queue": review_queue_row_count,
        "review_queue_unit": "block",
        "needs_review_count": human_review_required_count,
        "human_required_before_demote_count": human_required_before_demote_count,
        "demoted_from_human_to_machine_count": demoted_from_human_to_machine_count,
        "human_required_after_demote_count": human_required_after_demote_count,
        "demote_reason_counts": dict(demote_reason_counts),
        "recommended_next_action_counts": dict(recommended_next_action_counts),
        "recommended_next_action_missing_count": recommended_next_action_missing_count,
        "review_queue_matches_human_review_required": human_review_required_count == review_queue_row_count,
        "review_queue_matches_final_blocks": human_review_required_count == review_queue_row_count,
        "review_sample_count": len(review_sample_rows),
        "review_sample_level_counts": dict(review_sample_level_counts),
        "qwen_alignment_low_by_selected_source": dict(qwen_alignment_low_by_selected_source),
        "qwen_alignment_low_human_required_by_selected_source": dict(qwen_alignment_low_human_required_by_selected_source),
        "qwen_alignment_low_machine_note_by_selected_source": dict(qwen_alignment_low_machine_note_by_selected_source),
        "llm_target_max_blocks": 10,
        "llm_candidate_block_count": llm_candidate_block_count,
        "llm_target_block_count": llm_target_block_count,
        "llm_skipped_human_required_count": llm_skipped_human_required_count,
        "llm_target_selection_reasons": dict(llm_target_selection_reasons),
        "llm_target_block_ids": llm_target_block_ids,
        "llm_resolved_human_review_count": llm_resolved_human_review_count,
        "llm_kept_human_review_count": llm_kept_human_review_count,
        "llm_changed_final_text_count": llm_changed_final_text_count,
        "llm_no_change_count": llm_no_change_count,
        "llm_candidate_out_of_set_violation_count": llm_candidate_out_of_set_violation_count,
        "unusual_final_text_pattern_count": unusual_final_text_pattern_count,
        "unusual_final_text_pattern_human_required_count": unusual_final_text_pattern_human_required_count,
        "unusual_final_text_pattern_machine_note_count": unusual_final_text_pattern_machine_note_count,
        "unusual_final_text_patterns_by_level": {
            "human_required": dict(unusual_final_text_patterns_by_level["human_required"]),
            "machine_note": dict(unusual_final_text_patterns_by_level["machine_note"]),
        },
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    if args.resume_output_dir:
        output_dir = Path(args.resume_output_dir).expanduser().resolve()
        print(f"[pipeline] resume from output_dir={output_dir}", flush=True)
        episode_id, apple_timeline, sentence_units, blocks, alignments, block_rows, validation = _load_resume_bundle(output_dir)
        block_summary_path = output_dir / "normalized" / f"{episode_id}.block_candidates.json"
        block_summary = read_json_from_path(block_summary_path) if block_summary_path.exists() else {}
        source_files = {Path(path).name: Path(path) for path in apple_timeline.get("source_files", {}).values() if path}
        llm_client = None
        if args.use_llm:
            print("[pipeline] llm client init", flush=True)
            llm_client = LLMClient(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                api_key=args.llm_api_key,
                cache_dir=output_dir / "llm_cache",
            )
            print("[pipeline] llm ready check start", flush=True)
            llm_client.wait_for_ready()
            print("[pipeline] llm ready check end", flush=True)
        print(f"[pipeline] final selection start use_llm={args.use_llm}", flush=True)
        final_blocks, final_rows, review_rows, llm_stats, llm_audit_rows = select_final_candidates(
            episode_id=episode_id,
            block_rows=block_rows,
            sentence_units=sentence_units,
            llm_client=llm_client,
            use_llm=args.use_llm,
            llm_only_risky=args.llm_only_risky,
            llm_max_segments=args.llm_max_segments,
            output_dir=output_dir,
        )
        print(
            f"[pipeline] final selection end final_rows={len(final_rows)} review_rows={len(review_rows)} "
            f"llm_used={llm_stats.get('llm_used')}",
            flush=True,
        )
        summary = _build_summary(
            episode_id=episode_id,
            input_dir=Path(apple_timeline.get("source_files", {}).get("final_json", output_dir)),
            source_files=source_files,
            apple_timeline=apple_timeline,
            alignments=alignments,
            block_rows=block_rows,
            block_summary=block_summary,
            final_blocks=final_blocks,
            final_rows=final_rows,
            review_rows=review_rows,
            llm_stats=llm_stats,
            validation=validation,
            output_dir=output_dir,
        )
        summary["normalized_summary"] = read_json_from_path(output_dir / "reports" / f"{episode_id}.normalized_summary.json") if (output_dir / "reports" / f"{episode_id}.normalized_summary.json").exists() else {}
        review_sample_rows = _build_review_sample_rows(final_blocks)
        summary.update(_review_summary_metrics(final_blocks, review_rows, review_sample_rows))
        review_sample_path = output_dir / "reports" / f"{episode_id}.review_sample_blocks.json"
        review_sample_path.write_text(json.dumps(review_sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["review_sample_path"] = str(review_sample_path)
        summary["review_sample_count"] = len(review_sample_rows)
        demote_debug_rows = _demote_debug_rows(final_blocks, episode_id)
        demote_debug_short_path = output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.jsonl"
        demote_debug_short_md_path = output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.md"
        save_jsonl(demote_debug_short_path, demote_debug_rows)
        demote_lines = [f"# {episode_id}", "", "## demote_debug"]
        for row in demote_debug_rows:
            demote_lines.append(
                f"- {row['block_id']}: candidate={row['demote_candidate']} applied={row['demote_applied']} blockers={row['demote_blockers']}"
            )
        demote_debug_short_md_path.write_text("\n".join(demote_lines) + "\n", encoding="utf-8")
        summary["demote_debug_path"] = str(demote_debug_short_path)
        summary["demote_debug_count"] = len(demote_debug_rows)
        summary["needs_review_count_final_blocks"] = sum(1 for row in final_blocks if row.get("needs_review"))
        summary["needs_review_count_normalized_blocks"] = sum(1 for row in block_rows if row.get("needs_review"))
        summary["human_review_required_count"] = sum(1 for row in final_blocks if row.get("human_review_required"))
        summary["machine_review_note_count"] = sum(1 for row in final_blocks if row.get("machine_review_note"))
        summary["auto_accept_final_count"] = sum(1 for row in final_blocks if row.get("review_level") == "auto_accept")
        summary["review_queue_row_count"] = len(review_rows)
        summary["machine_review_notes_row_count"] = sum(1 for row in final_blocks if row.get("review_level") == "machine_note")
        summary["needs_review_count_for_review_queue"] = len(review_rows)
        summary["review_queue_unit"] = "block"
        summary["needs_review_count"] = summary["human_review_required_count"]
        summary["review_queue_matches_human_review_required"] = summary["human_review_required_count"] == summary["review_queue_row_count"]
        summary["review_queue_matches_final_blocks"] = summary["review_queue_matches_human_review_required"]
        summary["sentence_timeline_row_count"] = len(final_rows)
        summary["sentence_timeline_display_text_block_leak_count"] = sum(
            1
            for row in final_rows
            if row.get("sentence_display_text") == row.get("final_text_display")
            and len(str(row.get("sentence_display_text", ""))) >= len(str(row.get("apple_text", ""))) + 8
        )
        summary["sentence_timeline_display_text_too_long_count"] = sum(
            1
            for row in final_rows
            if len(str(row.get("sentence_display_text", ""))) > max(80, len(str(row.get("apple_text", ""))) + 24)
        )
        final_texts = [str(row.get("final_text_display", row.get("final_text", ""))) for row in final_blocks]
        summary["regression_phrase_counts"] = _phrase_counts(
            final_texts,
            ("なるほど一体", "8050問題はい", "聞いたことないですか8050問題", "ね。そうですね", "うことです", "のだから"),
        )
        summary["domain_error_phrase_counts"] = _phrase_counts(final_texts, ("排水の陣", "各家族", "社会行動", "整形立てて"))
        required_outputs = {
            "final_blocks": output_dir / "fusion" / f"{episode_id}.final_blocks.jsonl",
            "final_segments": output_dir / "fusion" / f"{episode_id}.final_segments.jsonl",
            "final_transcript_md": output_dir / "fusion" / f"{episode_id}.final_transcript.md",
            "review_queue": output_dir / "fusion" / f"{episode_id}.review_queue.jsonl",
            "machine_review_notes": output_dir / "fusion" / f"{episode_id}.machine_review_notes.jsonl",
            "llm_audit": output_dir / "fusion" / f"{episode_id}.llm_audit.jsonl",
            "llm_audit_md": output_dir / "fusion" / f"{episode_id}.llm_audit.md",
            "sentence_timeline": output_dir / "fusion" / f"{episode_id}.sentence_timeline.jsonl",
            "summary_json": output_dir / "reports" / f"{episode_id}.summary.json",
            "summary_md": output_dir / "reports" / f"{episode_id}.summary.md",
            "normalized_summary_json": output_dir / "reports" / f"{episode_id}.normalized_summary.json",
            "normalized_summary_md": output_dir / "reports" / f"{episode_id}.normalized_summary.md",
            "normalized_block_candidates": output_dir / "aligned_segments" / f"{episode_id}.block_candidates.jsonl",
            "review_sample_blocks": output_dir / "reports" / f"{episode_id}.review_sample_blocks.json",
            "demoted_review_blocks": output_dir / "fusion" / f"{episode_id}.demoted_review_blocks.jsonl",
            "demoted_review_blocks_md": output_dir / "fusion" / f"{episode_id}.demoted_review_blocks.md",
        }
        output_validation = {}
        missing_output_files: list[str] = []
        for name, path in required_outputs.items():
            exists = path.exists()
            valid = exists
            row_count = None
            if path.suffix == ".jsonl":
                valid, row_count = _jsonl_is_valid(path)
                output_validation[name] = {"exists": exists, "jsonl_parse_ok": valid}
            elif path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                    valid = True
                except Exception:
                    valid = False
                output_validation[name] = {"exists": exists, "json_parse_ok": valid}
            else:
                output_validation[name] = {"exists": exists}
            if row_count is not None:
                output_validation[name]["row_count"] = row_count
            if not valid:
                missing_output_files.append(name)
        summary["output_validation"] = output_validation
        summary["missing_output_files"] = missing_output_files
        if not summary["review_queue_matches_human_review_required"]:
            missing_output_files.append("review_queue_count_mismatch")
        summary["review_sample_path"] = str(output_dir / "reports" / f"{episode_id}.review_sample_blocks.json")
        summary["review_sample_count"] = len(review_sample_rows)
        summary["llm_called_block_count"] = sum(1 for row in final_blocks if row.get("llm_called"))
        summary["llm_decision_applied_count"] = sum(1 for row in final_blocks if row.get("llm_decision_applied"))
        summary["llm_selected_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_selected"))
        summary["llm_merged_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_merged_candidate"))
        summary["llm_changed_final_text_count"] = sum(1 for row in final_blocks if row.get("llm_changed_final_text"))
        summary["llm_no_change_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_changed_final_text"))
        summary["llm_cleared_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_resolved"))
        summary["llm_kept_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_resolved"))
        summary["llm_resolved_count"] = summary["llm_cleared_human_review_count"]
        summary["llm_audit_row_count"] = len(llm_audit_rows)
        (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
        if missing_output_files:
            raise RuntimeError(f"output validation failed: {missing_output_files}")
        print(f"[pipeline] end seconds={time.time() - t0:.2f}", flush=True)
        return summary
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == input_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("output-dir must be separate from input-dir")
    episode_prefix = _discover_episode_prefix(input_dir, args.episode_prefix)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output-dir already exists: {output_dir} (use --force to replace)")
    if args.force:
        _force_clean_output(output_dir)
    ensure_dir(output_dir)
    for name in ("normalized", "apple_timeline", "alignment", "aligned_segments", "fusion", "llm_cache", "reports"):
        ensure_dir(output_dir / name)
    print(f"[pipeline] start episode_prefix={episode_prefix} output_dir={output_dir}", flush=True)

    source_files = load_episode_files(input_dir, episode_prefix)
    print(f"[pipeline] files loaded count={len(source_files)}", flush=True)
    apple_timeline = build_apple_timeline(input_dir, episode_prefix=episode_prefix, output_dir=output_dir)
    episode_id = apple_timeline["episode_id"]
    print(f"[pipeline] apple timeline built episode={episode_id}", flush=True)
    sentence_units = [AppleSentenceUnit(**row) for row in apple_timeline["apple_sentence_units"]]
    blocks = [AlignmentBlock(**row) for row in apple_timeline["alignment_blocks"]]
    validation = {
        "sentence": __import__("tools.asr_alignment._core", fromlist=["validate_sentence_units"]).validate_sentence_units(
            apple_timeline["apple_stable_full_text"], sentence_units
        ),
        "block": __import__("tools.asr_alignment._core", fromlist=["validate_alignment_blocks"]).validate_alignment_blocks(
            apple_timeline["apple_stable_full_text"], sentence_units, blocks
        ),
    }
    print(
        f"[pipeline] validation sentence_units={len(sentence_units)} blocks={len(blocks)} "
        f"very_short={validation['sentence']['very_short_sentence_count']}",
        flush=True,
    )

    apple_artifact = apple_timeline["apple_artifact"]
    alignments = {}
    engine_files = {
        "qwen": _find_engine_file(source_files, ".qwen3-asr-1.7b.txt"),
        "nemotron": _find_engine_file(source_files, ".mlx-nemotron-3.5-asr-0.6b.txt"),
        "whisper": _find_engine_file(source_files, ".whisper-small.txt"),
    }
    for engine, path in engine_files.items():
        print(f"[pipeline] align start engine={engine} file={path.name}", flush=True)
        alignments[engine] = align_engine_to_apple(
            episode_id=episode_id,
            engine=engine,
            source_text_file=path,
            apple_artifact=apple_artifact,
            output_dir=output_dir,
        )
        print(
            f"[pipeline] align end engine={engine} score={alignments[engine].global_alignment_score:.3f} "
            f"coverage={alignments[engine].coverage_ratio:.3f}",
            flush=True,
        )

    print("[pipeline] build block candidates start", flush=True)
    block_rows, block_summary = build_block_candidates(
        episode_id=episode_id,
        alignment_blocks=blocks,
        sentence_units=sentence_units,
        apple_artifact=apple_artifact,
        alignments=alignments,
        output_dir=output_dir,
        candidate_build_mode=args.candidate_build_mode,
        workers=args.workers,
        no_parallel=args.no_parallel,
    )
    print(f"[pipeline] build block candidates end count={len(block_rows)}", flush=True)
    print("[pipeline] normalize flags start", flush=True)
    normalized_rows, normalized_summary = normalize_file(
        output_dir / "aligned_segments" / f"{episode_id}.block_candidates.jsonl",
        output_dir,
    )
    print(
        f"[pipeline] normalize flags end auto_accepted={normalized_summary.get('auto_accepted_count')} "
        f"needs_review={normalized_summary.get('needs_review_count')}",
        flush=True,
    )

    llm_client = None
    if args.use_llm:
        print("[pipeline] llm client init", flush=True)
        llm_client = LLMClient(
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            api_key=args.llm_api_key,
            cache_dir=output_dir / "llm_cache",
        )
        print("[pipeline] llm ready check start", flush=True)
        llm_client.wait_for_ready()
        print("[pipeline] llm ready check end", flush=True)

    print(f"[pipeline] final selection start use_llm={args.use_llm}", flush=True)
    final_blocks, final_rows, review_rows, llm_stats, llm_audit_rows = select_final_candidates(
        episode_id=episode_id,
        block_rows=normalized_rows,
        sentence_units=sentence_units,
        llm_client=llm_client,
        use_llm=args.use_llm,
        llm_only_risky=args.llm_only_risky,
        llm_max_segments=args.llm_max_segments,
        output_dir=output_dir,
        conversation_punctuation=args.conversation_punctuation,
        short_response_period=args.short_response_period,
    )
    print(
        f"[pipeline] final selection end final_rows={len(final_rows)} review_rows={len(review_rows)} "
        f"llm_used={llm_stats.get('llm_used')}",
        flush=True,
    )

    summary = _build_summary(
        episode_id=episode_id,
        input_dir=input_dir,
        source_files=source_files,
        apple_timeline=apple_timeline,
        alignments=alignments,
        block_rows=normalized_rows,
        block_summary=block_summary,
        final_blocks=final_blocks,
        final_rows=final_rows,
        review_rows=review_rows,
        llm_stats=llm_stats,
        validation=validation,
        output_dir=output_dir,
        conversation_punctuation=args.conversation_punctuation,
        short_response_period=args.short_response_period,
    )
    summary["block_summary"] = block_summary
    summary.update(block_summary)
    summary["normalized_summary"] = normalized_summary
    summary["alignment_text_used_for_text_score"] = True
    summary["boundary_text_used_for_windowing"] = True
    summary["boundary_text_used_for_boundary_score"] = True
    summary["boundary_text_used_for_main_text_score"] = False
    summary["boundary_hints_used_for_boundary_eval"] = bool(summary.get("boundary_hint_used_in_candidate_boundary_eval"))
    summary["punctuation_hard_matched_as_normal_chars"] = True
    summary["parallel_enabled"] = not args.no_parallel
    summary["source_level_boundary_hints_available_by_source"] = {"apple": True, "qwen": False, "nemotron": False, "whisper": False}
    summary["candidate_level_boundary_hints_available_by_source"] = {"apple": True, "qwen": True, "nemotron": True, "whisper": True}
    summary["boundary_hints_generated_before_candidate_extraction_by_source"] = {"apple": True, "qwen": False, "nemotron": False, "whisper": False}
    summary["boundary_hints_generated_after_candidate_extraction_by_source"] = {"apple": False, "qwen": True, "nemotron": True, "whisper": True}
    summary["needs_review_count_final_blocks"] = sum(1 for row in final_blocks if row.get("needs_review"))
    summary["needs_review_count_normalized_blocks"] = sum(1 for row in normalized_rows if row.get("needs_review"))
    summary["human_review_required_count"] = sum(1 for row in final_blocks if row.get("human_review_required"))
    summary["machine_review_note_count"] = sum(1 for row in final_blocks if row.get("machine_review_note"))
    summary["auto_accept_final_count"] = sum(1 for row in final_blocks if row.get("review_level") == "auto_accept")
    summary["review_queue_row_count"] = len(review_rows)
    summary["machine_review_notes_row_count"] = sum(1 for row in final_blocks if row.get("review_level") == "machine_note")
    summary["needs_review_count_for_review_queue"] = len(review_rows)
    summary["review_queue_unit"] = "block"
    summary["needs_review_count"] = summary["human_review_required_count"]
    summary["review_queue_matches_human_review_required"] = summary["human_review_required_count"] == summary["review_queue_row_count"]
    summary["review_queue_matches_final_blocks"] = summary["review_queue_matches_human_review_required"]
    summary["sentence_timeline_row_count"] = len(final_rows)
    summary["llm_called_block_count"] = sum(1 for row in final_blocks if row.get("llm_called"))
    summary["llm_decision_applied_count"] = sum(1 for row in final_blocks if row.get("llm_decision_applied"))
    summary["llm_selected_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_selected"))
    summary["llm_merged_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_merged_candidate"))
    summary["llm_changed_final_text_count"] = sum(1 for row in final_blocks if row.get("llm_changed_final_text"))
    summary["llm_no_change_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_changed_final_text"))
    summary["llm_cleared_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_resolved"))
    summary["llm_kept_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_resolved"))
    summary["llm_resolved_count"] = summary["llm_cleared_human_review_count"]
    summary["llm_audit_row_count"] = len(llm_audit_rows)
    summary["sentence_timeline_display_text_block_leak_count"] = sum(
        1
        for row in final_rows
        if row.get("sentence_display_text") == row.get("final_text_display")
        and len(str(row.get("sentence_display_text", ""))) >= len(str(row.get("apple_text", ""))) + 8
    )
    summary["sentence_timeline_display_text_too_long_count"] = sum(
        1
        for row in final_rows
        if len(str(row.get("sentence_display_text", ""))) > max(80, len(str(row.get("apple_text", ""))) + 24)
    )
    final_texts = [str(row.get("final_text_display", row.get("final_text", ""))) for row in final_blocks]
    summary["regression_phrase_counts"] = _phrase_counts(
        final_texts,
        (
            "なるほど一体",
            "8050問題はい",
            "聞いたことないですか8050問題",
            "ね。そうですね",
            "うことです",
            "のだから",
        ),
    )
    summary["domain_error_phrase_counts"] = _phrase_counts(
        final_texts,
        ("排水の陣", "各家族", "社会行動", "整形立てて"),
    )
    review_sample_rows = _build_review_sample_rows(final_blocks)
    summary.update(_review_summary_metrics(final_blocks, review_rows, review_sample_rows))
    review_sample_path = output_dir / "reports" / f"{episode_id}.review_sample_blocks.json"
    review_sample_path.write_text(json.dumps(review_sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["review_sample_path"] = str(review_sample_path)
    summary["review_sample_count"] = len(review_sample_rows)
    demote_debug_rows = _demote_debug_rows(final_blocks, episode_id)
    demote_debug_short_path = output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.jsonl"
    demote_debug_short_md_path = output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.md"
    save_jsonl(demote_debug_short_path, demote_debug_rows)
    demote_lines = [f"# {episode_id}", "", "## demote_debug"]
    for row in demote_debug_rows:
        demote_lines.append(
            f"- {row['block_id']}: candidate={row['demote_candidate']} applied={row['demote_applied']} blockers={row['demote_blockers']}"
        )
    demote_debug_short_md_path.write_text("\n".join(demote_lines) + "\n", encoding="utf-8")
    summary["demote_debug_path"] = str(demote_debug_short_path)
    summary["demote_debug_count"] = len(demote_debug_rows)
    summary["llm_called_block_count"] = sum(1 for row in final_blocks if row.get("llm_called"))
    summary["llm_decision_applied_count"] = sum(1 for row in final_blocks if row.get("llm_decision_applied"))
    summary["llm_selected_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_selected"))
    summary["llm_merged_candidate_count"] = sum(1 for row in final_blocks if row.get("llm_merged_candidate"))
    summary["llm_changed_final_text_count"] = sum(1 for row in final_blocks if row.get("llm_changed_final_text"))
    summary["llm_no_change_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_changed_final_text"))
    summary["llm_cleared_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_resolved"))
    summary["llm_kept_human_review_count"] = sum(1 for row in final_blocks if row.get("llm_selected") and not row.get("llm_resolved"))
    summary["llm_resolved_count"] = summary["llm_cleared_human_review_count"]
    summary["llm_audit_row_count"] = len(llm_audit_rows)
    (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    (output_dir / "normalized" / f"{episode_id}.manifest.json").write_text(
        json.dumps({"episode_id": episode_id, "input_dir": str(input_dir), "output_dir": str(output_dir), "episode_prefix": episode_prefix, "files": sorted(str(path) for path in source_files.values())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    required_outputs = {
        "final_blocks": output_dir / "fusion" / f"{episode_id}.final_blocks.jsonl",
        "final_segments": output_dir / "fusion" / f"{episode_id}.final_segments.jsonl",
        "final_transcript_md": output_dir / "fusion" / f"{episode_id}.final_transcript.md",
        "review_queue": output_dir / "fusion" / f"{episode_id}.review_queue.jsonl",
        "machine_review_notes": output_dir / "fusion" / f"{episode_id}.machine_review_notes.jsonl",
        "llm_audit": output_dir / "fusion" / f"{episode_id}.llm_audit.jsonl",
        "llm_audit_md": output_dir / "fusion" / f"{episode_id}.llm_audit.md",
        "sentence_timeline": output_dir / "fusion" / f"{episode_id}.sentence_timeline.jsonl",
        "summary_json": output_dir / "reports" / f"{episode_id}.summary.json",
        "summary_md": output_dir / "reports" / f"{episode_id}.summary.md",
        "normalized_summary_json": output_dir / "reports" / f"{episode_id}.normalized_summary.json",
        "normalized_summary_md": output_dir / "reports" / f"{episode_id}.normalized_summary.md",
        "normalized_block_candidates": output_dir / "aligned_segments" / f"{episode_id}.block_candidates.jsonl",
        "review_sample_blocks": output_dir / "reports" / f"{episode_id}.review_sample_blocks.json",
        "demoted_review_blocks": output_dir / "fusion" / f"{episode_id}.demoted_review_blocks.jsonl",
        "demoted_review_blocks_md": output_dir / "fusion" / f"{episode_id}.demoted_review_blocks.md",
        "demote_debug": output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.jsonl",
        "demote_debug_md": output_dir / "fusion" / f"{episode_id.split('-', 1)[0]}.demote_debug.md",
    }
    output_validation = {}
    missing_output_files: list[str] = []
    for name, path in required_outputs.items():
        exists = path.exists()
        valid = exists
        row_count = None
        if path.suffix == ".jsonl":
            valid, row_count = _jsonl_is_valid(path)
            output_validation[name] = {"exists": exists, "jsonl_parse_ok": valid}
        elif path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                valid = True
            except Exception:
                valid = False
            output_validation[name] = {"exists": exists, "json_parse_ok": valid}
        else:
            output_validation[name] = {"exists": exists}
        if row_count is not None:
            output_validation[name]["row_count"] = row_count
        if not valid:
            missing_output_files.append(name)
    summary["output_validation"] = output_validation
    summary["missing_output_files"] = missing_output_files
    if not summary["review_queue_matches_final_blocks"]:
        missing_output_files.append("review_queue_count_mismatch")
    (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    if missing_output_files:
        raise RuntimeError(f"output validation failed: {missing_output_files}")
    print(f"[pipeline] end seconds={time.time() - t0:.2f}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="既存 ASR 出力を Apple SpeechAnalyzer の時刻軸へ後付け整列します。")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-output-dir", default=None)
    parser.add_argument("--episode-prefix", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--conversation-punctuation", action="store_true")
    parser.add_argument("--short-response-period", action="store_true")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8010/v1/chat/completions")
    parser.add_argument("--llm-model", default="assistant")
    parser.add_argument("--llm-api-key", default="local-qwen3-assistant")
    parser.add_argument("--llm-max-segments", type=int, default=10)
    parser.add_argument("--llm-only-risky", action="store_true")
    parser.add_argument("--candidate-build-mode", choices=["full", "staged", "qwen-only"], default="staged")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-parallel", action="store_true")
    return parser


def main() -> None:
    _install_timed_print()
    args = build_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps({"episode_id": summary["episode_id"], "output_dir": summary["output_dir"], "apple_sentence_unit_count": summary["apple_sentence_unit_count"], "alignment_block_count": summary["alignment_block_count"], "needs_review_count": summary["needs_review_count"], "llm_used": summary["llm_used"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
