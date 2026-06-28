from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        AppleSentenceUnit,
        AlignmentBlock,
        build_sentence_units_from_text,
        build_text_artifact,
        canonicalize_name,
        ensure_dir,
        load_episode_files,
        nfc_text,
        read_json_from_path,
        read_json_members_from_zip,
        save_jsonl,
        _build_char_timeline_from_runs,
        split_text_by_punctuation,
    )
else:
    from ._core import (
        AppleSentenceUnit,
        AlignmentBlock,
        build_sentence_units_from_text,
        build_text_artifact,
        canonicalize_name,
        ensure_dir,
        load_episode_files,
        nfc_text,
        read_json_from_path,
        read_json_members_from_zip,
        save_jsonl,
        _build_char_timeline_from_runs,
        split_text_by_punctuation,
    )


def _find_path(files: dict[str, Path], suffix: str) -> Path | None:
    for name, path in files.items():
        if nfc_text(name).endswith(suffix):
            return path
    return None


def _load_final_segments(final_json: dict[str, Any], txt_fallback: str | None = None) -> list[dict[str, Any]]:
    segments = final_json.get("segments")
    if isinstance(segments, list) and segments:
        return segments
    if txt_fallback:
        return [{"index": 1, "startSeconds": 0.0, "endSeconds": max(0.1, float(len(txt_fallback)) / 20.0), "text": txt_fallback}]
    return []


def _load_snapshot_texts(raw_volatile_zip_path: Path | None) -> list[str]:
    if not raw_volatile_zip_path or not raw_volatile_zip_path.exists():
        return []
    snapshots = read_json_members_from_zip(raw_volatile_zip_path)
    if snapshots and isinstance(snapshots[0], dict) and "results" in snapshots[0]:
        texts = []
        for snapshot in snapshots:
            results = snapshot.get("results") or []
            texts.append("".join((r.get("text", {}) or {}).get("text", "") if isinstance(r.get("text"), dict) else str(r.get("text", "") or "") for r in results))
        return texts
    if isinstance(snapshots, list):
        return ["".join((r.get("text", {}) or {}).get("text", "") if isinstance(r.get("text"), dict) else str(r.get("text", "") or "") for r in snapshots)]
    return []


def build_apple_timeline(input_dir: Path, *, episode_prefix: str | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    print(f"[apple] start input_dir={input_dir} episode_prefix={episode_prefix}", flush=True)
    files = load_episode_files(input_dir, episode_prefix)
    final_json_path = _find_path(files, ".MacOS-SpeechAnalyzer.json")
    final_txt_path = _find_path(files, ".MacOS-SpeechAnalyzer.txt")
    raw_apple_zip_path = _find_path(files, ".MacOS-SpeechAnalyzer.raw_apple.json.zip")
    raw_volatile_zip_path = _find_path(files, ".MacOS-SpeechAnalyzer.raw_volatile_apple.json.zip")
    if final_json_path is None:
        raise FileNotFoundError("Apple SpeechAnalyzer JSON not found")
    episode_id = nfc_text(episode_prefix) if episode_prefix else canonicalize_name(final_json_path.name.split(".MacOS-SpeechAnalyzer", 1)[0])

    final_json = read_json_from_path(final_json_path)
    final_txt = final_txt_path.read_text(encoding="utf-8") if final_txt_path else None
    final_segments = _load_final_segments(final_json, final_txt)

    raw_apple_results = read_json_members_from_zip(raw_apple_zip_path) if raw_apple_zip_path and raw_apple_zip_path.exists() else []
    if not raw_apple_results:
        raw_apple_results = []
        for segment in final_segments:
            text = nfc_text(str(segment.get("text", "")))
            start = float(segment.get("startSeconds") or 0.0)
            end = float(segment.get("endSeconds") or start)
            raw_apple_results.append(
                {
                    "text": {"text": text, "runs": []},
                    "startSeconds": start,
                    "endSeconds": end,
                    "confidence": float(segment.get("confidence") or 0.8),
                }
            )

    volatile_texts = _load_snapshot_texts(raw_volatile_zip_path)
    full_text = "".join(
        nfc_text(
            (result.get("text", {}) or {}).get("text", "") if isinstance(result.get("text"), dict) else str(result.get("text", "") or "")
        )
        for result in raw_apple_results
    )
    if not full_text:
        full_text = nfc_text(final_txt or "")

    char_timeline: list[dict[str, Any]] = []
    for result in raw_apple_results:
        text_obj = result.get("text", {})
        if isinstance(text_obj, dict):
            text = nfc_text(text_obj.get("text", "") or "")
            runs = text_obj.get("runs") or []
        else:
            text = nfc_text(str(text_obj or ""))
            runs = []
        start = float(result.get("startSeconds") or 0.0)
        end = float(result.get("endSeconds") or start)
        char_timeline.extend(_build_char_timeline_from_runs(text, runs, start, end))

    if len(char_timeline) < len(full_text):
        tail = len(full_text) - len(char_timeline)
        last_end = char_timeline[-1]["end_sec"] if char_timeline else 0.0
        for i in range(tail):
            char_timeline.append({"char": full_text[len(char_timeline)], "start_sec": last_end + i * 0.01, "end_sec": last_end + (i + 1) * 0.01, "confidence": 0.8})

    sentence_units = build_sentence_units_from_text(
        episode_id=episode_id,
        full_text=full_text,
        char_timeline=char_timeline,
        source_file=str(final_json_path),
        source_index=1,
        confirmation_count=2 if volatile_texts else 0,
        confirmed_snapshot_indexes=[0, 1] if len(volatile_texts) >= 2 else ([0] if volatile_texts else []),
        start_sec_is_estimated=False,
        end_sec_is_estimated=False,
    )
    # Rebuild sentence units with punctuation-aware positions from full_text slices.
    normalized_units: list[AppleSentenceUnit] = []
    for idx, (text, start, end, punct) in enumerate(split_text_by_punctuation(full_text), start=1):
        slice_timeline = char_timeline[start:end] if end <= len(char_timeline) else char_timeline[start:]
        if not slice_timeline:
            continue
        normalized_units.append(
            AppleSentenceUnit(
                episode_id=episode_id,
                sentence_id=f"sent_{idx:06d}",
                block_id=None,
                text=text,
                char_start=start,
                char_end=end,
                start_sec=float(slice_timeline[0]["start_sec"]),
                end_sec=float(slice_timeline[-1]["end_sec"]),
                confirmation_count=2 if volatile_texts else 0,
                confirmed_snapshot_indexes=[0, 1] if len(volatile_texts) >= 2 else ([0] if volatile_texts else []),
                stability_score=0.85 if volatile_texts else 0.7,
                unit_kind="sentence" if len(text) >= 3 or punct else "tail",
                start_sec_is_estimated=False,
                end_sec_is_estimated=False,
                source_file=str(final_json_path),
                source_index=idx,
                confidence=None,
            )
        )

    # Merge tiny leading/trailing fragments into neighbors.
    merged: list[AppleSentenceUnit] = []
    for unit in normalized_units:
        if merged and len(unit.text.strip()) < 3 and not unit.text.endswith(tuple("。？！?!")):
            prev = merged[-1]
            prev.text += unit.text
            prev.char_end = unit.char_end
            prev.end_sec = unit.end_sec
            prev.stability_score = min(prev.stability_score, unit.stability_score)
            continue
        merged.append(unit)
    # Ensure block_id will be set later by block grouping.
    sentence_units = merged
    blocks = []
    if sentence_units:
        from ._core import merge_sentence_units_into_blocks  # local import for standalone fallback

        blocks = merge_sentence_units_into_blocks(episode_id, sentence_units)
        sentence_to_block = {}
        for block in blocks:
            for sid in block.sentence_ids:
                sentence_to_block[sid] = block.block_id
        for unit in sentence_units:
            unit.block_id = sentence_to_block.get(unit.sentence_id)

    apple_artifact = build_text_artifact(full_text)
    timeline = {
        "episode_id": episode_id,
        "source_files": {
            "final_json": str(final_json_path),
            "final_txt": str(final_txt_path) if final_txt_path else None,
            "raw_apple_zip": str(raw_apple_zip_path) if raw_apple_zip_path else None,
            "raw_volatile_zip": str(raw_volatile_zip_path) if raw_volatile_zip_path else None,
        },
        "apple_stable_full_text": full_text,
        "apple_text_length": len(full_text),
        "apple_sentence_units": [dataclasses.asdict(unit) for unit in sentence_units],
        "alignment_blocks": [dataclasses.asdict(block) for block in blocks],
        "apple_artifact": apple_artifact,
    }

    if output_dir is not None:
        ensure_dir(output_dir / "apple_timeline")
        ensure_dir(output_dir / "normalized")
        save_jsonl(output_dir / "apple_timeline" / f"{episode_id}.apple_sentence_units.jsonl", [dataclasses.asdict(unit) for unit in sentence_units])
        save_jsonl(output_dir / "apple_timeline" / f"{episode_id}.alignment_blocks.jsonl", [dataclasses.asdict(block) for block in blocks])
        (output_dir / "apple_timeline" / f"{episode_id}.apple_stable_full_text.txt").write_text(full_text, encoding="utf-8")
        serializable = {
            "episode_id": episode_id,
            "source_files": timeline["source_files"],
            "apple_stable_full_text": full_text,
            "apple_text_length": len(full_text),
            "apple_sentence_units": [dataclasses.asdict(unit) for unit in sentence_units],
            "alignment_blocks": [dataclasses.asdict(block) for block in blocks],
        }
        (output_dir / "normalized" / f"{episode_id}.apple_timeline.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[apple] end episode={episode_id} sentence_units={len(sentence_units)} blocks={len(blocks)} "
        f"text_len={len(full_text)} seconds={time.time() - t0:.2f}",
        flush=True,
    )

    return timeline
