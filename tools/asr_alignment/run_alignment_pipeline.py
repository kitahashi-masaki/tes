from __future__ import annotations

import argparse
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
        build_text_artifact,
    )
    from tools.asr_alignment.apple_timeline_builder import build_apple_timeline  # type: ignore
    from tools.asr_alignment.flat_text_aligner import align_engine_to_apple  # type: ignore
    from tools.asr_alignment.final_candidate_selector import select_final_candidates  # type: ignore
    from tools.asr_alignment.llm_client import LLMClient  # type: ignore
    from tools.asr_alignment.normalize_fusion_flags import normalize_file  # type: ignore
    from tools.asr_alignment.segment_candidate_builder import build_block_candidates  # type: ignore
else:
    from ._core import AppleSentenceUnit, AlignmentBlock, AlignmentResult, NormEntry, TextArtifact, ensure_dir, load_episode_files, load_jsonl, nfc_text, read_json_from_path, build_text_artifact
    from .apple_timeline_builder import build_apple_timeline
    from .flat_text_aligner import align_engine_to_apple
    from .final_candidate_selector import select_final_candidates
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
    llm_called_block_count = 0
    review_reason_char_split_count = 0
    boundary_hint_used_in_alignment_blocking = False
    boundary_hint_used_in_candidate_boundary_eval = False
    boundary_hint_used_in_cleanup = False
    conversation_boundary_hint_stage = "after_apple_sentence_units_before_alignment_blocks"
    boundary_hint_applied_sources = set()
    boundary_hint_applied_count_by_source = Counter()
    short_response_period_count_by_source = Counter()
    boundary_text_generated_count_by_source = Counter()
    display_text_generated_count_by_source = Counter()
    for row in block_rows:
        risk_counts.update(row.get("risk_flags", []))
        for source in ("apple", "qwen", "nemotron", "whisper"):
            candidate = row.get(source)
            if isinstance(candidate, dict):
                if candidate.get("boundary_hints"):
                    boundary_hint_applied_sources.add(source)
                    boundary_hint_applied_count_by_source[source] += len(candidate.get("boundary_hints", []))
                if candidate.get("short_response_period_count"):
                    short_response_period_count_by_source[source] += int(candidate.get("short_response_period_count", 0))
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
            short_response_period_count += int(block.get("short_response_period_count", 0))
            possible_speaker_change_period_count += int(block.get("possible_speaker_change_period_count", 0))
            boundary_hint_used_count += int(block.get("boundary_hint_used_count", 0))
        if block.get("needs_review") and block.get("boundary_cleanup_applied"):
            post_cleanup_needs_review_count += 1
        if str(block.get("final_text", "")) == str(block.get("candidate_summary", {}).get("apple", "")):
            final_text_equals_apple_text_count += 1
        if block.get("selected_source") != "apple" and str(block.get("final_text", "")) == str(block.get("candidate_summary", {}).get("apple", "")):
            selected_source_not_apple_final_equals_apple_count += 1
    for row in review_rows:
        reasons = row.get("review_reason", [])
        if isinstance(reasons, list) and reasons and all(len(str(reason)) == 1 for reason in reasons):
            review_reason_char_split_count += 1
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
        "final_block_count": len(final_blocks),
        "final_sentence_timeline_count": len(final_rows),
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
        "auto_accepted_count": auto_accepted_count,
        "auto_accepted_ratio": round(auto_accepted_count / max(len(block_rows), 1), 3),
        "risk_flag_counts": dict(risk_counts.most_common()),
        "needs_review_reason_counts": dict(review_reason_counts.most_common()),
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
    lines.append(f"- 自動採用件数: {summary['auto_accepted_count']}")
    lines.append(f"- 自動採用率: {summary['auto_accepted_ratio']}")
    lines.append(f"- usable candidate 数: {summary['usable_candidate_count_by_engine']}")
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
    if summary.get("review_sample_path"):
        lines.append("")
        lines.append("## review サンプル")
        lines.append(f"- 全文サンプル: {summary['review_sample_path']}")
    lines.append("")
    return "\n".join(lines)


def _build_review_sample_rows(block_rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    review_rows = [row for row in block_rows if row.get("needs_review")]
    sample_rows: list[dict[str, Any]] = []
    for row in review_rows[:limit]:
        sample_rows.append(
            {
                "episode_id": row.get("episode_id"),
                "block_id": row.get("block_id"),
                "segment_id": row.get("segment_id"),
                "alignment_quality": row.get("alignment_quality"),
                "needs_review": row.get("needs_review"),
                "needs_review_reason": row.get("needs_review_reason", []),
                "qwen_apple_difference_type": row.get("qwen_apple_difference_type"),
                "qwen_apple_similarity": row.get("qwen_apple_similarity"),
                "risk_flags": row.get("risk_flags", []),
                "apple_text": row.get("apple", {}).get("text", ""),
                "candidate_texts": {
                    engine: row.get(engine, {}).get("text", "")
                    for engine in ("qwen", "nemotron", "whisper")
                    if isinstance(row.get(engine), dict)
                },
            }
        )
    return sample_rows


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    if args.resume_output_dir:
        output_dir = Path(args.resume_output_dir).expanduser().resolve()
        print(f"[pipeline] resume from output_dir={output_dir}", flush=True)
        episode_id, apple_timeline, sentence_units, blocks, alignments, block_rows, validation = _load_resume_bundle(output_dir)
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
        print(f"[pipeline] final selection start use_llm={args.use_llm}", flush=True)
        final_blocks, final_rows, review_rows, llm_stats = select_final_candidates(
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
            final_blocks=final_blocks,
            final_rows=final_rows,
            review_rows=review_rows,
            llm_stats=llm_stats,
            validation=validation,
            output_dir=output_dir,
        )
        summary["normalized_summary"] = read_json_from_path(output_dir / "reports" / f"{episode_id}.normalized_summary.json") if (output_dir / "reports" / f"{episode_id}.normalized_summary.json").exists() else {}
        summary["review_sample_path"] = str(output_dir / "reports" / f"{episode_id}.review_sample_blocks.json")
        summary["review_sample_count"] = len(review_rows)
        (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
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

    print(f"[pipeline] final selection start use_llm={args.use_llm}", flush=True)
    final_blocks, final_rows, review_rows, llm_stats = select_final_candidates(
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
    summary["boundary_text_used_for_main_text_score"] = False
    summary["boundary_hints_used_for_boundary_eval"] = bool(summary.get("boundary_hint_used_in_candidate_boundary_eval"))
    summary["punctuation_hard_matched_as_normal_chars"] = False
    summary["parallel_enabled"] = not args.no_parallel
    summary["source_level_boundary_hints_available_by_source"] = {"apple": True, "qwen": False, "nemotron": False, "whisper": False}
    summary["candidate_level_boundary_hints_available_by_source"] = {"apple": True, "qwen": True, "nemotron": True, "whisper": True}
    summary["boundary_hints_generated_before_candidate_extraction_by_source"] = {"apple": True, "qwen": False, "nemotron": False, "whisper": False}
    summary["boundary_hints_generated_after_candidate_extraction_by_source"] = {"apple": False, "qwen": True, "nemotron": True, "whisper": True}
    review_sample_rows = _build_review_sample_rows(normalized_rows)
    review_sample_path = output_dir / "reports" / f"{episode_id}.review_sample_blocks.json"
    review_sample_path.write_text(json.dumps(review_sample_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["review_sample_path"] = str(review_sample_path)
    summary["review_sample_count"] = len(review_sample_rows)
    (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    (output_dir / "normalized" / f"{episode_id}.manifest.json").write_text(
        json.dumps({"episode_id": episode_id, "input_dir": str(input_dir), "output_dir": str(output_dir), "episode_prefix": episode_prefix, "files": sorted(str(path) for path in source_files.values())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
    parser.add_argument("--llm-max-segments", type=int, default=200)
    parser.add_argument("--llm-only-risky", action="store_true")
    parser.add_argument("--candidate-build-mode", choices=["full", "staged", "qwen-only"], default="staged")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-parallel", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps({"episode_id": summary["episode_id"], "output_dir": summary["output_dir"], "apple_sentence_unit_count": summary["apple_sentence_unit_count"], "alignment_block_count": summary["alignment_block_count"], "needs_review_count": summary["needs_review_count"], "llm_used": summary["llm_used"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
