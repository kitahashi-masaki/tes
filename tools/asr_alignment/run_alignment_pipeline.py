from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        AppleSentenceUnit,
        AlignmentBlock,
        ensure_dir,
        load_episode_files,
        nfc_text,
    )
    from tools.asr_alignment.apple_timeline_builder import build_apple_timeline  # type: ignore
    from tools.asr_alignment.flat_text_aligner import align_engine_to_apple  # type: ignore
    from tools.asr_alignment.final_candidate_selector import select_final_candidates  # type: ignore
    from tools.asr_alignment.segment_candidate_builder import build_block_candidates  # type: ignore
else:
    from ._core import AppleSentenceUnit, AlignmentBlock, ensure_dir, load_episode_files, nfc_text
    from .apple_timeline_builder import build_apple_timeline
    from .flat_text_aligner import align_engine_to_apple
    from .final_candidate_selector import select_final_candidates
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


def _build_summary(
    *,
    episode_id: str,
    input_dir: Path,
    source_files: dict[str, Path],
    apple_timeline: dict[str, Any],
    alignments: dict[str, Any],
    block_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    llm_stats: dict[str, Any],
    validation: dict[str, Any],
    output_dir: Path,
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
    for row in block_rows:
        risk_counts.update(row.get("risk_flags", []))
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
                unusable_counts[candidate.get("unusable_reason")] += 1
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
    sentence_units = apple_timeline["apple_sentence_units"]
    blocks = apple_timeline["alignment_blocks"]
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
    lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
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

    source_files = load_episode_files(input_dir, episode_prefix)
    apple_timeline = build_apple_timeline(input_dir, episode_prefix=episode_prefix, output_dir=output_dir)
    episode_id = apple_timeline["episode_id"]
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

    apple_artifact = apple_timeline["apple_artifact"]
    alignments = {}
    engine_files = {
        "qwen": _find_engine_file(source_files, ".qwen3-asr-1.7b.txt"),
        "nemotron": _find_engine_file(source_files, ".mlx-nemotron-3.5-asr-0.6b.txt"),
        "whisper": _find_engine_file(source_files, ".whisper-small.txt"),
    }
    for engine, path in engine_files.items():
        alignments[engine] = align_engine_to_apple(
            episode_id=episode_id,
            engine=engine,
            source_text_file=path,
            apple_artifact=apple_artifact,
            output_dir=output_dir,
        )

    block_rows, block_summary = build_block_candidates(
        episode_id=episode_id,
        alignment_blocks=blocks,
        sentence_units=sentence_units,
        apple_artifact=apple_artifact,
        alignments=alignments,
        output_dir=output_dir,
    )

    final_rows, review_rows, llm_stats = select_final_candidates(
        episode_id=episode_id,
        block_rows=block_rows,
        sentence_units=sentence_units,
        llm_client=None,
        use_llm=False,
        llm_only_risky=args.llm_only_risky,
        llm_max_segments=args.llm_max_segments,
        output_dir=output_dir,
    )

    summary = _build_summary(
        episode_id=episode_id,
        input_dir=input_dir,
        source_files=source_files,
        apple_timeline=apple_timeline,
        alignments=alignments,
        block_rows=block_rows,
        final_rows=final_rows,
        review_rows=review_rows,
        llm_stats=llm_stats,
        validation=validation,
        output_dir=output_dir,
    )
    summary["block_summary"] = block_summary
    (output_dir / "reports" / f"{episode_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "reports" / f"{episode_id}.summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    (output_dir / "normalized" / f"{episode_id}.manifest.json").write_text(
        json.dumps({"episode_id": episode_id, "input_dir": str(input_dir), "output_dir": str(output_dir), "episode_prefix": episode_prefix, "files": sorted(str(path) for path in source_files.values())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="既存 ASR 出力を Apple SpeechAnalyzer の時刻軸へ後付け整列します。")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-prefix", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8010/v1/chat/completions")
    parser.add_argument("--llm-model", default="assistant")
    parser.add_argument("--llm-api-key", default="local-qwen3-assistant")
    parser.add_argument("--llm-max-segments", type=int, default=200)
    parser.add_argument("--llm-only-risky", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps({"episode_id": summary["episode_id"], "output_dir": summary["output_dir"], "apple_sentence_unit_count": summary["apple_sentence_unit_count"], "alignment_block_count": summary["alignment_block_count"], "needs_review_count": summary["needs_review_count"], "llm_used": summary["llm_used"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
