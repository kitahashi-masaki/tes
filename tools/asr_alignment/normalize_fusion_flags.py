from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import ensure_dir, load_jsonl, save_jsonl  # type: ignore
else:
    from ._core import ensure_dir, load_jsonl, save_jsonl


def _difference_type_flags(row: dict[str, Any]) -> tuple[str, bool, bool, bool]:
    diff_type = str(row.get("qwen_apple_difference_type") or "semantic")
    qwen_disagreement = diff_type == "semantic"
    critical = diff_type == "critical"
    soft = diff_type == "soft_domain"
    if diff_type == "none":
        qwen_disagreement = False
        critical = False
        soft = False
    elif diff_type == "surface":
        qwen_disagreement = False
        critical = False
        soft = False
    elif diff_type == "soft_domain":
        qwen_disagreement = False
        critical = False
        soft = True
    elif diff_type == "critical":
        qwen_disagreement = True
        critical = True
        soft = False
    elif diff_type == "semantic":
        qwen_disagreement = True
        critical = False
        soft = False
    else:
        diff_type = "semantic"
        qwen_disagreement = True
        critical = False
        soft = False
    return diff_type, qwen_disagreement, critical, soft


def normalize_block_candidate_flags(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("segment_id", normalized.get("block_id"))
    diff_type, qwen_disagreement, critical, soft = _difference_type_flags(normalized)
    qwen_similarity = float(normalized.get("qwen_apple_similarity", 0.0))
    qwen_alignment = float(normalized.get("qwen", {}).get("local_alignment_score", normalized.get("qwen", {}).get("alignment_score", 0.0)))
    numeric_disagreement = "numeric_disagreement" in (normalized.get("risk_flags") or [])
    span_too_long = "span_too_long" in (normalized.get("risk_flags") or [])
    large_span_drift = "large_span_drift" in (normalized.get("risk_flags") or [])
    boundary_contamination_suspected = "boundary_contamination_suspected" in (normalized.get("risk_flags") or [])
    domain_error_phrase = "domain_error_phrase" in (normalized.get("risk_flags") or [])
    qwen_review_blocker = (
        diff_type in {"critical", "semantic"}
        or qwen_alignment < 0.82
        or numeric_disagreement
        or span_too_long
        or boundary_contamination_suspected
        or domain_error_phrase
        or (large_span_drift and qwen_alignment < 0.90)
    )
    needs_review = bool(qwen_review_blocker)
    if diff_type in {"none", "surface"}:
        qwen_disagreement = False
        critical = False
        soft = False
    elif diff_type == "soft_domain" and qwen_similarity >= 0.88 and qwen_alignment >= 0.82 and not numeric_disagreement and not span_too_long:
        needs_review = False
    auto_accepted = not needs_review
    risk_flags = []
    if diff_type in {"critical", "semantic"}:
        risk_flags.append("qwen_apple_disagreement")
    if critical:
        risk_flags.append("critical_term_disagreement")
    if soft:
        risk_flags.append("soft_domain_difference")
    if numeric_disagreement:
        risk_flags.append("numeric_disagreement")
    if span_too_long:
        risk_flags.append("span_too_long")
    if large_span_drift:
        risk_flags.append("large_span_drift")
    if boundary_contamination_suspected:
        risk_flags.append("boundary_contamination_suspected")
    if domain_error_phrase:
        risk_flags.append("domain_error_phrase")
    if diff_type == "surface":
        risk_flags.append("surface_difference")
    if qwen_disagreement and "qwen_apple_disagreement" not in risk_flags:
        risk_flags.append("qwen_apple_disagreement")
    normalized.update(
        {
            "qwen_apple_difference_type": diff_type,
            "qwen_apple_disagreement": qwen_disagreement,
            "critical_term_disagreement": critical,
            "soft_domain_difference": soft,
            "risk_flags": sorted(dict.fromkeys(risk_flags)),
            "needs_review": needs_review,
            "normalized_needs_review": needs_review,
            "needs_review_reason": sorted(dict.fromkeys(risk_flags if needs_review else [])),
            "auto_accepted": auto_accepted,
        }
    )
    return normalized


def summarize_normalized(rows: list[dict[str, Any]], *, episode_id: str, output_dir: Path) -> dict[str, Any]:
    diff_counts = Counter(row.get("qwen_apple_difference_type", "semantic") for row in rows)
    risk_counts = Counter()
    reason_counts = Counter()
    for row in rows:
        risk_counts.update(row.get("risk_flags", []))
        if row.get("needs_review"):
            reason_counts.update(row.get("needs_review_reason", []))
    summary = {
        "episode_id": episode_id,
        "block_count": len(rows),
        "qwen_apple_difference_type_counts": dict(diff_counts),
        "risk_flag_counts": dict(risk_counts.most_common()),
        "needs_review_reason_counts": dict(reason_counts.most_common()),
        "auto_accepted_count": sum(1 for row in rows if row.get("auto_accepted")),
        "needs_review_count": sum(1 for row in rows if row.get("needs_review")),
        "normalized_needs_review_count": sum(1 for row in rows if row.get("normalized_needs_review")),
    }
    ensure_dir(output_dir / "reports")
    (output_dir / "reports" / f"{episode_id}.normalized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [f"# {episode_id} 正規化サマリー", "", "## 件数"]
    lines.append(f"- block 数: {summary['block_count']}")
    lines.append(f"- auto_accept 件数: {summary['auto_accepted_count']}")
    lines.append(f"- needs_review 件数: {summary['needs_review_count']}")
    lines.append(f"- normalized_needs_review 件数: {summary['normalized_needs_review_count']}")
    lines.append("")
    lines.append("## difference_type")
    for k, v in summary["qwen_apple_difference_type_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## risk_flags")
    for k, v in summary["risk_flag_counts"].items():
        lines.append(f"- {k}: {v}")
    (output_dir / "reports" / f"{episode_id}.normalized_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def normalize_file(input_path: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    print(f"[normalize] start input={input_path.name}", flush=True)
    rows = load_jsonl(input_path)
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if idx == 1 or idx % 5 == 0 or idx == len(rows):
            print(f"[normalize] progress {idx}/{len(rows)} block_id={row.get('block_id')}", flush=True)
        normalized.append(normalize_block_candidate_flags(row))
    episode_id = normalized[0].get("episode_id") if normalized else input_path.stem
    ensure_dir(output_dir / "fusion")
    out_path = output_dir / "fusion" / f"{episode_id}.normalized_block_candidates.jsonl"
    save_jsonl(out_path, normalized)
    summary = summarize_normalized(normalized, episode_id=str(episode_id), output_dir=output_dir)
    print(
        f"[normalize] end episode={episode_id} blocks={len(normalized)} auto_accepted={summary.get('auto_accepted_count')} "
        f"needs_review={summary.get('needs_review_count')} seconds={time.time() - t0:.2f}",
        flush=True,
    )
    return normalized, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize fusion flags before LLM candidate selection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    normalize_file(Path(args.input).expanduser().resolve(), Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
