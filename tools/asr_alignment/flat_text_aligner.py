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
        AlignmentResult,
        build_mapping_from_equal_blocks,
        build_text_artifact,
        ensure_dir,
        sequence_similarity,
    )
else:
    from ._core import (
        AlignmentResult,
        build_mapping_from_equal_blocks,
        build_text_artifact,
        ensure_dir,
        sequence_similarity,
    )


def align_engine_to_apple(
    *,
    episode_id: str,
    engine: str,
    source_text_file: Path,
    apple_artifact,
    output_dir: Path | None = None,
) -> AlignmentResult:
    t0 = time.time()
    print(
        f"[align] start episode={episode_id} engine={engine} source={source_text_file.name}",
        flush=True,
    )
    asr_text = source_text_file.read_text(encoding="utf-8")
    asr_artifact = build_text_artifact(asr_text)
    anchors, apple_to_asr_map, asr_to_apple_map, global_score, coverage = build_mapping_from_equal_blocks(
        asr_artifact.match_norm_text,
        apple_artifact.match_norm_text,
    )
    alignment = AlignmentResult(
        episode_id=episode_id,
        engine=engine,
        source_text_file=str(source_text_file),
        apple_text_length=len(apple_artifact.raw_text),
        asr_text_length=len(asr_artifact.raw_text),
        normalized_apple_text_length=len(apple_artifact.match_norm_text),
        normalized_asr_text_length=len(asr_artifact.match_norm_text),
        global_alignment_score=global_score,
        coverage_ratio=coverage,
        anchors=anchors,
        apple_to_asr_map=apple_to_asr_map,
        asr_to_apple_map=asr_to_apple_map,
        apple_artifact=apple_artifact,
        asr_artifact=asr_artifact,
    )
    print(
        f"[align] end episode={episode_id} engine={engine} score={global_score:.3f} "
        f"coverage={coverage:.3f} seconds={time.time() - t0:.2f}",
        flush=True,
    )

    if output_dir is not None:
        ensure_dir(output_dir / "alignment")
        ensure_dir(output_dir / "normalized")
        summary = {
            "episode_id": episode_id,
            "engine": engine,
            "source_text_file": str(source_text_file),
            "apple_text_length": len(apple_artifact.raw_text),
            "asr_text_length": len(asr_artifact.raw_text),
            "normalized_apple_text_length": len(apple_artifact.match_norm_text),
            "normalized_asr_text_length": len(asr_artifact.match_norm_text),
            "global_alignment_score": global_score,
            "coverage_ratio": coverage,
            "anchors": [dataclasses.asdict(anchor) for anchor in anchors],
        }
        (output_dir / "alignment" / f"{episode_id}.{engine}_alignment.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "normalized" / f"{episode_id}.{engine}.normalized.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "engine": engine,
                    "source_text_file": str(source_text_file),
                    "raw_text": asr_artifact.raw_text,
                    "display_norm_text": asr_artifact.display_norm_text,
                    "match_norm_text": asr_artifact.match_norm_text,
                    "char_map": [dataclasses.asdict(entry) for entry in asr_artifact.entries],
                    "apple_to_asr_map": apple_to_asr_map,
                    "asr_to_apple_map": asr_to_apple_map,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return alignment
