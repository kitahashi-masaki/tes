from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SPLIT_PUNCT = set("。！？!?")
MATCH_DROP_CATEGORIES = ("P", "S")
WHITESPACE_RE = re.compile(r"\s+")
NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)?")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")
KATAKANA_RE = re.compile(r"[\u30A0-\u30FFー]{2,}")
KANJI_RE = re.compile(r"[\u4E00-\u9FFF]{2,4}")
PROPER_NOUN_ANCHORS: dict[str, tuple[str, ...]] = {
    "8050": ("8050", "八〇五〇"),
    "80歳": ("80歳",),
    "50歳": ("50歳",),
    "20歳": ("20歳",),
    "DNA": ("DNA", "ＤＮＡ"),
    "資本主義": ("資本主義",),
    "大家族": ("大家族", "大家族制"),
    "核家族": ("核家族",),
    "子供部屋": ("子供部屋", "子ども部屋"),
    "建築ラボ": ("建築ラボ", "建築LABO", "検知クラボ"),
    "遠藤": ("遠藤",),
    "矢野": ("矢野", "八納", "八野"),
}

PROTECTED_CRITICAL_TERMS = {"8050", "80歳", "50歳", "20歳", "DNA", "遠藤", "矢野", "建築ラボ"}
DOMAIN_SOFT_TERMS = {"資本主義", "大家族", "核家族", "子供部屋", "社会問題", "社会課題", "先進国", "親の家"}


@dataclass
class NormEntry:
    norm_index: int
    raw_index: int
    raw_char: str
    norm_char: str


@dataclass
class TextArtifact:
    raw_text: str
    display_norm_text: str
    match_norm_text: str
    entries: list[NormEntry] = field(default_factory=list)

    def raw_span_from_norm_range(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        if not self.entries:
            return (0, 0)
        selected = [e for e in self.entries if norm_start <= e.norm_index < norm_end]
        if not selected:
            return (0, 0)
        raw_start = min(e.raw_index for e in selected)
        raw_end = max(e.raw_index for e in selected) + 1
        return raw_start, raw_end

    def raw_text_from_norm_range(self, norm_start: int, norm_end: int) -> str:
        raw_start, raw_end = self.raw_span_from_norm_range(norm_start, norm_end)
        return self.raw_text[raw_start:raw_end]

    def norm_index_to_raw_index(self, norm_index: int) -> int | None:
        for entry in self.entries:
            if entry.norm_index == norm_index:
                return entry.raw_index
        return None

    def norm_span_from_raw_range(self, raw_start: int, raw_end: int) -> tuple[int, int]:
        if not self.entries:
            return (0, 0)
        selected = [e for e in self.entries if raw_start <= e.raw_index < raw_end]
        if not selected:
            return (0, 0)
        norm_start = min(e.norm_index for e in selected)
        norm_end = max(e.norm_index for e in selected) + 1
        return norm_start, norm_end


@dataclass
class AppleSegment:
    episode_id: str
    segment_id: str
    start_sec: float
    end_sec: float
    apple_text_raw: str
    apple_text_stable: str
    apple_char_start: int
    apple_char_end: int
    apple_update_count: int
    apple_rewrite_count: int
    apple_stability_score: float
    start_sec_is_estimated: bool
    source_file: str
    source_index: int
    confidence: float | None = None


@dataclass
class AppleSentenceUnit:
    episode_id: str
    sentence_id: str
    block_id: str | None
    text: str
    char_start: int
    char_end: int
    start_sec: float
    end_sec: float
    confirmation_count: int
    confirmed_snapshot_indexes: list[int]
    stability_score: float
    unit_kind: str
    start_sec_is_estimated: bool
    end_sec_is_estimated: bool
    source_file: str
    source_index: int
    confidence: float | None = None
    raw_text: str = ""
    alignment_text: str = ""
    boundary_text: str = ""
    display_text: str = ""
    boundary_hints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AlignmentBlock:
    episode_id: str
    block_id: str
    sentence_ids: list[str]
    sentence_count: int
    text: str
    char_start: int
    char_end: int
    start_sec: float
    end_sec: float
    stability_score: float
    source_index: int
    block_split_reason: str | None = None
    parent_sentence_ids: list[str] = field(default_factory=list)
    is_sub_block: bool = False
    source: str = "sentence_units_merge"


@dataclass
class AlignmentOpcode:
    apple_norm_start: int
    apple_norm_end: int
    asr_norm_start: int
    asr_norm_end: int
    score: float


@dataclass
class AlignmentResult:
    episode_id: str
    engine: str
    source_text_file: str
    apple_text_length: int
    asr_text_length: int
    normalized_apple_text_length: int
    normalized_asr_text_length: int
    global_alignment_score: float
    coverage_ratio: float
    anchors: list[AlignmentOpcode]
    apple_to_asr_map: list[float]
    asr_to_apple_map: list[float]
    apple_artifact: TextArtifact
    asr_artifact: TextArtifact


@dataclass
class CandidateData:
    episode_id: str
    segment_id: str
    time: dict[str, float]
    apple: dict[str, Any]
    qwen: dict[str, Any]
    nemotron: dict[str, Any]
    whisper: dict[str, Any]
    candidate_agreement_score: float
    alignment_quality: str
    risk_flags: list[str]
    needs_review: bool


@dataclass
class LLMDecision:
    success: bool
    cached: bool
    selected_source: str | None = None
    final_text: str | None = None
    confidence: float | None = None
    needs_review: bool | None = None
    review_reason: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None


def nfc_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def canonicalize_name(value: str) -> str:
    return nfc_text(value)


def normalize_text_with_map(text: str, *, mode: str = "match") -> tuple[str, list[NormEntry]]:
    raw_text = nfc_text(text)
    out: list[str] = []
    entries: list[NormEntry] = []
    last_was_space = False
    for raw_index, raw_char in enumerate(raw_text):
        normalized = unicodedata.normalize("NFKC", raw_char).casefold()
        if not normalized:
            continue
        for ch in normalized:
            if ch.isspace():
                ch = " "
            category = unicodedata.category(ch)
            if mode == "match":
                if ch == " " or category.startswith(MATCH_DROP_CATEGORIES):
                    continue
            else:
                if ch == " ":
                    if last_was_space:
                        continue
                    last_was_space = True
                else:
                    last_was_space = False
            out.append(ch)
            entries.append(
                NormEntry(
                    norm_index=len(out) - 1,
                    raw_index=raw_index,
                    raw_char=raw_char,
                    norm_char=ch,
                )
            )
    return "".join(out), entries


def build_text_artifact(text: str) -> TextArtifact:
    display_norm_text, entries_display = normalize_text_with_map(text, mode="display")
    match_norm_text, entries_match = normalize_text_with_map(text, mode="match")
    # Use the more compact match entries for mappings; display text remains available.
    return TextArtifact(
        raw_text=nfc_text(text),
        display_norm_text=display_norm_text,
        match_norm_text=match_norm_text,
        entries=entries_match,
    )


def sequence_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def extract_numbers(text: str) -> set[str]:
    return set(NUMERIC_RE.findall(text))


def extract_tokens(text: str) -> dict[str, set[str]]:
    return {
        "numbers": extract_numbers(text),
        "latin": set(LATIN_RE.findall(text)),
        "katakana": set(KATAKANA_RE.findall(text)),
        "kanji": set(KANJI_RE.findall(text)),
    }


def top_counts(values: Iterable[str], limit: int = 10) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if rows:
        text += "\n"
    atomic_write_text(path, text)


def read_json_from_path(path: Path) -> Any:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if not members:
                raise ValueError(f"zip has no files: {path}")
            data = zf.read(members[0])
            return json.loads(data.decode("utf-8"))
    if path.suffix.lower() in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in handle if line.strip()]
            return json.load(handle)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unsupported file type: {path}")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_hash(encoded)


def parse_time_to_seconds(value: str) -> float:
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def format_seconds(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms >= 1000:
        whole += 1
        ms -= 1000
    hh = whole // 3600
    mm = (whole % 3600) // 60
    ss = whole % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def split_sentence_chunks(text: str) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in SPLIT_PUNCT:
            end = idx + 1
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk, start, end))
            start = end
    if start < len(text):
        chunk = text[start:]
        if chunk.strip():
            chunks.append((chunk, start, len(text)))
    return chunks


def split_text_by_punctuation(text: str) -> list[tuple[str, int, int, str]]:
    chunks: list[tuple[str, int, int, str]] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in SPLIT_PUNCT:
            end = idx + 1
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((chunk, start, end, ch))
            start = end
    if start < len(text):
        chunk = text[start:]
        if chunk.strip():
            chunks.append((chunk, start, len(text), ""))
    return chunks


def raw_span_to_norm_span(artifact: TextArtifact, raw_start: int, raw_end: int) -> tuple[int, int]:
    return artifact.norm_span_from_raw_range(raw_start, raw_end)


def _normalize_span_text(text: str) -> str:
    return normalize_text_with_map(text, mode="match")[0]


def _span_similarity(left: str, right: str) -> float:
    return sequence_similarity(_normalize_span_text(left), _normalize_span_text(right))


def _match_span_similarity(left_match_norm: str, right_match_norm: str) -> float:
    return sequence_similarity(left_match_norm, right_match_norm)


def _extract_anchor_terms(texts: list[str]) -> set[str]:
    normalized_texts = [_normalize_span_text(text) for text in texts if text]
    hits: set[str] = set()
    for canonical, aliases in PROPER_NOUN_ANCHORS.items():
        for text in normalized_texts:
            if any(alias and _normalize_span_text(alias) in text for alias in aliases):
                hits.add(canonical)
                break
    return hits


IMPORTANT_TERMS = {
    "8050",
    "80歳",
    "50歳",
    "20歳",
    "DNA",
    "資本主義",
    "大家族",
    "核家族",
    "子供部屋",
    "建築ラボ",
    "遠藤",
    "矢野",
}


def normalize_surface_for_compare(text: str) -> str:
    value = _normalize_span_text(text)
    value = value.replace("子ども", "子供").replace("できる", "出来る").replace("いう", "言う")
    value = re.sub(r"[。、，,．.\s]+", "", value)
    value = value.replace("ー", "")
    value = value.replace("八〇五〇", "8050")
    value = value.replace("ＤＮＡ", "dna")
    return value


def detect_qwen_apple_difference(apple_text: str, qwen_text: str) -> tuple[str, float, bool]:
    apple_norm = normalize_surface_for_compare(apple_text)
    qwen_norm = normalize_surface_for_compare(qwen_text)
    similarity = sequence_similarity(apple_norm, qwen_norm)
    if apple_norm == qwen_norm:
        return "none", similarity, False
    apple_critical = {term for term in PROTECTED_CRITICAL_TERMS if any(alias in apple_text for alias in PROPER_NOUN_ANCHORS.get(term, (term,)))}
    qwen_critical = {term for term in PROTECTED_CRITICAL_TERMS if any(alias in qwen_text for alias in PROPER_NOUN_ANCHORS.get(term, (term,)))}
    apple_soft = {term for term in DOMAIN_SOFT_TERMS if term in apple_text}
    qwen_soft = {term for term in DOMAIN_SOFT_TERMS if term in qwen_text}
    if apple_critical != qwen_critical:
        return "critical", similarity, True
    if apple_soft != qwen_soft and similarity >= 0.88:
        return "soft_domain", similarity, False
    if similarity >= 0.86:
        return "surface", similarity, False
    return "semantic", similarity, True


def is_large_span_drift(candidate: dict[str, Any]) -> bool:
    return abs(int(candidate.get("span_drift_start", 0))) > 80 or abs(int(candidate.get("span_drift_end", 0))) > 80


def classify_qwen_apple_difference(apple_text: str, qwen_text: str) -> tuple[str, float, bool]:
    diff_type, similarity, critical = detect_qwen_apple_difference(apple_text, qwen_text)
    return diff_type, similarity, critical


def refine_local_asr_span(
    *,
    apple_artifact: TextArtifact,
    asr_artifact: TextArtifact,
    apple_raw_start: int,
    apple_raw_end: int,
    projected_norm_start: int,
    projected_norm_end: int,
    search_radius: int = 18,
) -> dict[str, Any]:
    apple_norm_start, apple_norm_end = raw_span_to_norm_span(apple_artifact, apple_raw_start, apple_raw_end)
    apple_target = apple_artifact.match_norm_text[apple_norm_start:apple_norm_end]
    asr_norm_text = asr_artifact.match_norm_text
    asr_norm_len = len(asr_norm_text)
    if asr_norm_len <= 0:
        return {
            "initial_char_start": 0,
            "initial_char_end": 0,
            "refined_char_start": 0,
            "refined_char_end": 0,
            "local_alignment_score": 0.0,
            "span_refined": False,
            "span_drift_start": 0,
            "span_drift_end": 0,
            "boundary_contamination": True,
            "usable_for_agreement": False,
            "unusable_reason": "empty_asr_text",
            "initial_text": "",
            "refined_text": "",
        }

    projected_norm_start = max(0, min(int(projected_norm_start), asr_norm_len - 1))
    projected_norm_end = max(projected_norm_start + 1, min(int(projected_norm_end), asr_norm_len))
    initial_char_start, initial_char_end = asr_artifact.raw_span_from_norm_range(projected_norm_start, projected_norm_end)
    if initial_char_end <= initial_char_start:
        fallback_start = asr_artifact.norm_index_to_raw_index(projected_norm_start)
        initial_char_start = int(fallback_start or 0)
        initial_char_end = min(
            len(asr_artifact.raw_text),
            max(initial_char_start + max(len(apple_artifact.raw_text[apple_raw_start:apple_raw_end]), 1), initial_char_start + 1),
        )
    initial_text = asr_artifact.raw_text[initial_char_start:initial_char_end]
    initial_local_alignment_score = max(0.0, min(1.0, _span_similarity(apple_target, initial_text)))
    initial_norm_len = len(_normalize_span_text(initial_text))
    apple_norm_len = max(len(_normalize_span_text(apple_target)), 1)
    span_length_ratio = initial_norm_len / apple_norm_len
    initial_boundary_contamination = (
        not initial_text.strip()
        or initial_text[:1].isspace()
        or initial_text[-1:].isspace()
    )
    if (
        initial_local_alignment_score >= 0.96
        and 0.75 <= span_length_ratio <= 1.35
        and not initial_boundary_contamination
    ):
        return {
            "initial_char_start": initial_char_start,
            "initial_char_end": initial_char_end,
            "refined_char_start": initial_char_start,
            "refined_char_end": initial_char_end,
            "local_alignment_score": initial_local_alignment_score,
            "span_refined": False,
            "span_drift_start": 0,
            "span_drift_end": 0,
            "boundary_contamination": False,
            "span_too_short": False,
            "span_too_long": False,
            "span_boundary_adjusted": False,
            "span_boundary_adjust_reason": [],
            "usable_for_agreement": len(_normalize_span_text(initial_text)) > 2,
            "unusable_reason": "" if len(_normalize_span_text(initial_text)) > 2 else "too_short",
            "initial_text": initial_text,
            "refined_text": initial_text,
            "early_exit": True,
            "early_exit_reason": "high_local_alignment",
            "cheap_span_accept": True,
            "cheap_span_accept_reason": "high_initial_alignment",
            "heavy_refinement_skipped": True,
        }

    target_len = max(apple_norm_end - apple_norm_start, 1)
    window_left = max(0, projected_norm_start - search_radius)
    window_right = min(asr_norm_len, projected_norm_end + search_radius)
    min_len = max(1, target_len - max(6, target_len // 3))
    max_len = min(asr_norm_len, target_len + max(10, target_len // 2))
    best = {
        "start": projected_norm_start,
        "end": projected_norm_end,
        "score": -1.0,
        "len_delta": 10**9,
        "drift": 10**9,
    }
    early_exit = False
    candidate_eval_count = 0
    target_len_for_profile = max_len - min_len
    window_span = max(0, window_right - window_left)
    use_coarse_to_fine = target_len >= 120 or window_span >= 220 or target_len_for_profile >= 80
    start_step = 4 if use_coarse_to_fine else 2
    end_step = 4 if use_coarse_to_fine else 2
    upper_start = min(window_right, max(0, asr_norm_len - 1))

    def _score_candidate(start: int, end: int) -> None:
        nonlocal best, early_exit, candidate_eval_count
        candidate = asr_norm_text[start:end]
        if not candidate.strip():
            return
        candidate_eval_count += 1
        text_sim = _match_span_similarity(apple_target, candidate)
        len_score = 1.0 - min(1.0, abs(len(candidate) - target_len) / max(target_len, len(candidate), 1))
        boundary_score = 1.0
        if start <= window_left or end >= window_right:
            boundary_score -= 0.18
        if candidate[:1].isspace() or candidate[-1:].isspace():
            boundary_score -= 0.12
        local_score = max(0.0, min(1.0, 0.7 * text_sim + 0.2 * len_score + 0.1 * boundary_score))
        len_delta = abs(len(candidate) - target_len)
        drift = abs(start - projected_norm_start) + abs(end - projected_norm_end)
        if (
            local_score > best["score"]
            or (local_score == best["score"] and len_delta < best["len_delta"])
            or (local_score == best["score"] and len_delta == best["len_delta"] and drift < best["drift"])
            or (local_score == best["score"] and len_delta == best["len_delta"] and drift == best["drift"] and start < best["start"])
        ):
            best = {"start": start, "end": end, "score": local_score, "len_delta": len_delta, "drift": drift}
            span_length_ratio = len(candidate) / max(target_len, 1)
            boundary_contamination_for_candidate = (
                not candidate.strip()
                or candidate[:1].isspace()
                or candidate[-1:].isspace()
            )
            if (
                local_score >= 0.96
                and 0.75 <= span_length_ratio <= 1.35
                and not boundary_contamination_for_candidate
            ):
                early_exit = True

    for start in range(window_left, upper_start + 1, start_step):
        min_end = min(asr_norm_len, max(start + min_len, start + 1))
        max_end = min(window_right, start + max_len)
        if min_end > max_end:
            continue
        for end in range(min_end, max_end + 1, end_step):
            _score_candidate(start, end)
            if early_exit:
                break
        if early_exit:
            break
    if use_coarse_to_fine and not early_exit and best["score"] >= 0:
        fine_anchors = [
            (int(best["start"]), int(best["end"])),
            (projected_norm_start, projected_norm_end),
        ]
        evaluated_fine_pairs: set[tuple[int, int]] = set()
        for anchor_start, anchor_end in fine_anchors:
            fine_window_left = max(window_left, anchor_start - 12)
            fine_window_right = min(window_right, anchor_start + 12)
            fine_min_end = max(0, anchor_end - 18)
            fine_max_end = min(window_right, anchor_end + 18)
            for start in range(fine_window_left, fine_window_right + 1, 2):
                min_end = max(start + 1, fine_min_end)
                max_end = min(fine_max_end, start + max_len, asr_norm_len)
                if min_end > max_end:
                    continue
                for end in range(min_end, max_end + 1, 2):
                    pair = (start, end)
                    if pair in evaluated_fine_pairs:
                        continue
                    evaluated_fine_pairs.add(pair)
                    _score_candidate(start, end)
                    if early_exit:
                        break
                if early_exit:
                    break
            if early_exit:
                break

    refined_norm_start, refined_norm_end = best["start"], best["end"]
    refined_char_start, refined_char_end = asr_artifact.raw_span_from_norm_range(refined_norm_start, refined_norm_end)
    if refined_char_end <= refined_char_start:
        refined_char_start, refined_char_end = initial_char_start, initial_char_end
    initial_text = asr_artifact.raw_text[initial_char_start:initial_char_end]
    refined_text = asr_artifact.raw_text[refined_char_start:refined_char_end]
    span_refined = (refined_char_start, refined_char_end) != (initial_char_start, initial_char_end)
    boundary_contamination = (
        refined_norm_start <= 0
        or refined_norm_end >= asr_norm_len
        or not refined_text.strip()
        or refined_text[:1].isspace()
        or refined_text[-1:].isspace()
    )
    span_too_short = len(_normalize_span_text(refined_text)) < max(3, len(_normalize_span_text(apple_target)) // 4)
    span_too_long = len(_normalize_span_text(refined_text)) > max(12, int(len(_normalize_span_text(apple_target)) * 2.5))
    adjusted_reasons: list[str] = []
    if refined_text and refined_text[:1] in {"ん", "に", "ち"}:
        refined_char_start = max(0, refined_char_start - 2)
        adjusted_reasons.append("leading_character_restored")
    if refined_text and refined_text[-1:] in {"そ", "と", "で"}:
        refined_char_end = min(len(asr_artifact.raw_text), refined_char_end + 3)
        adjusted_reasons.append("trailing_character_restored")
    if adjusted_reasons:
        refined_text = asr_artifact.raw_text[refined_char_start:refined_char_end]
        refined_norm_start, refined_norm_end = asr_artifact.norm_span_from_raw_range(refined_char_start, refined_char_end)
        span_refined = True
    local_alignment_score = max(0.0, min(1.0, best["score"] if best["score"] >= 0 else _span_similarity(apple_target, refined_text)))
    unusable_reason = ""
    if not refined_text.strip():
        unusable_reason = "empty_candidate"
    elif local_alignment_score < 0.70:
        unusable_reason = "local_alignment_low"
    elif boundary_contamination:
        unusable_reason = "boundary_contamination"
    elif span_too_short:
        unusable_reason = "span_too_short"
    elif span_too_long:
        unusable_reason = "span_too_long"
    usable_for_agreement = not unusable_reason and len(_normalize_span_text(refined_text)) > 2
    if not usable_for_agreement and not unusable_reason:
        unusable_reason = "too_short"
    return {
        "initial_char_start": initial_char_start,
        "initial_char_end": initial_char_end,
        "refined_char_start": refined_char_start,
        "refined_char_end": refined_char_end,
        "local_alignment_score": local_alignment_score,
        "span_refined": span_refined,
        "span_drift_start": refined_char_start - initial_char_start,
        "span_drift_end": refined_char_end - initial_char_end,
        "boundary_contamination": boundary_contamination,
        "span_too_short": span_too_short,
        "span_too_long": span_too_long,
        "span_boundary_adjusted": bool(adjusted_reasons),
        "span_boundary_adjust_reason": adjusted_reasons,
        "usable_for_agreement": usable_for_agreement,
        "unusable_reason": unusable_reason,
        "initial_text": initial_text,
        "refined_text": refined_text,
        "early_exit": early_exit,
        "early_exit_reason": "high_local_alignment" if early_exit else "",
        "cheap_span_accept": False,
        "cheap_span_accept_reason": "",
        "heavy_refinement_skipped": False,
        "refinement_search_profile": "coarse_to_fine" if use_coarse_to_fine else "full",
        "refinement_candidate_eval_count": candidate_eval_count,
        "refinement_window_span": window_span,
        "refinement_target_len": target_len,
    }


def _build_char_timeline_from_runs(text: str, runs: list[dict[str, Any]], start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    if runs:
        timeline: list[dict[str, Any]] = []
        for idx, ch in enumerate(text):
            run = runs[min(idx, len(runs) - 1)]
            audio_range = run.get("audioTimeRange", {})
            timeline.append(
                {
                    "char": ch,
                    "start_sec": float(audio_range.get("startSeconds", start_sec)),
                    "end_sec": float(audio_range.get("endSeconds", end_sec)),
                    "confidence": float(run.get("transcriptionConfidence", 0.8)),
                }
            )
        return timeline
    duration = max(end_sec - start_sec, 0.01)
    timeline = []
    for idx, ch in enumerate(text):
        c_start = start_sec + duration * idx / max(len(text), 1)
        c_end = start_sec + duration * (idx + 1) / max(len(text), 1)
        timeline.append({"char": ch, "start_sec": c_start, "end_sec": c_end, "confidence": 0.8})
    return timeline


def build_sentence_units_from_text(
    *,
    episode_id: str,
    full_text: str,
    char_timeline: list[dict[str, Any]],
    source_file: str,
    source_index: int,
    confirmation_count: int = 0,
    confirmed_snapshot_indexes: list[int] | None = None,
    start_sec_is_estimated: bool = False,
    end_sec_is_estimated: bool = False,
) -> list[AppleSentenceUnit]:
    confirmed_snapshot_indexes = confirmed_snapshot_indexes or []
    units: list[AppleSentenceUnit] = []
    for idx, (chunk, start, end, punct) in enumerate(split_text_by_punctuation(full_text), start=1):
        if len(chunk.strip()) < 3 and not punct:
            continue
        if len(chunk.strip()) < 3 and punct:
            continue
        timeline_slice = char_timeline[start:end]
        if timeline_slice:
            unit_start = float(timeline_slice[0]["start_sec"])
            unit_end = float(timeline_slice[-1]["end_sec"])
        else:
            unit_start = float(idx)
            unit_end = unit_start + 0.1
        units.append(
            AppleSentenceUnit(
                episode_id=episode_id,
                sentence_id=f"sent_{idx:06d}",
                block_id=None,
                text=chunk,
                char_start=start,
                char_end=end,
                start_sec=unit_start,
                end_sec=unit_end,
                confirmation_count=confirmation_count,
                confirmed_snapshot_indexes=list(confirmed_snapshot_indexes),
                stability_score=clamp(0.85 + 0.03 * confirmation_count, 0.1, 0.99),
                unit_kind="sentence",
                start_sec_is_estimated=start_sec_is_estimated,
                end_sec_is_estimated=end_sec_is_estimated,
                source_file=source_file,
                source_index=source_index,
                confidence=None,
                raw_text=chunk,
                alignment_text=chunk,
                boundary_text=chunk,
                display_text=chunk,
                boundary_hints=[],
            )
        )
    if units and not units[-1].text.endswith(tuple(SPLIT_PUNCT)):
        # Keep a tail fragment attached to the last sentence when needed.
        tail = units[-1]
        if len(tail.text) < 3:
            units.pop()
    return units


def merge_sentence_units_into_blocks(
    episode_id: str,
    sentence_units: list[AppleSentenceUnit],
    *,
    min_sentences: int = 2,
    max_sentences: int = 5,
    min_duration: float = 8.0,
    max_duration: float = 25.0,
) -> list[AlignmentBlock]:
    def _make_block(chosen: list[AppleSentenceUnit], *, split_reason: str | None = None, parent_sentence_ids: list[str] | None = None, is_sub_block: bool = False, source_index: int = 0) -> AlignmentBlock:
        return AlignmentBlock(
            episode_id=episode_id,
            block_id="",
            sentence_ids=[u.sentence_id for u in chosen],
            sentence_count=len(chosen),
            text="".join(unit.text for unit in chosen),
            char_start=chosen[0].char_start,
            char_end=chosen[-1].char_end,
            start_sec=chosen[0].start_sec,
            end_sec=chosen[-1].end_sec,
            stability_score=min(u.stability_score for u in chosen),
            source_index=source_index,
            block_split_reason=split_reason,
            parent_sentence_ids=list(parent_sentence_ids or [u.sentence_id for u in chosen]),
            is_sub_block=is_sub_block,
        )

    provisional_blocks: list[AlignmentBlock] = []
    i = 0

    def _has_short_response_hint(unit: AppleSentenceUnit) -> bool:
        return any(hint.get("type") == "short_response_period" for hint in getattr(unit, "boundary_hints", []) or [])

    while i < len(sentence_units):
        start = i
        chosen = [sentence_units[i]]
        i += 1
        while i < len(sentence_units):
            candidate = sentence_units[i]
            combined_count = len(chosen) + 1
            combined_duration = candidate.end_sec - chosen[0].start_sec
            boundary_priority = _has_short_response_hint(chosen[-1]) or _has_short_response_hint(candidate)
            if boundary_priority and len(chosen) >= min_sentences:
                break
            if combined_count <= max_sentences and (combined_duration <= max_duration or len(chosen) < min_sentences):
                chosen.append(candidate)
                i += 1
                if len(chosen) >= min_sentences and combined_duration >= min_duration:
                    if _has_short_response_hint(chosen[-1]) or _has_short_response_hint(chosen[-2] if len(chosen) >= 2 else chosen[-1]):
                        break
                    break
            else:
                break
        if len(chosen) == 1 and i < len(sentence_units):
            nxt = sentence_units[i]
            if (nxt.end_sec - chosen[0].start_sec) <= max_duration and not _has_short_response_hint(chosen[-1]):
                chosen.append(nxt)
                i += 1
        provisional_blocks.append(_make_block(chosen, source_index=start + 1))

    def _split_block(block: AlignmentBlock) -> list[AlignmentBlock]:
        if block.sentence_count <= 1:
            return [block]
        duration = block.end_sec - block.start_sec
        if block.sentence_count <= max_sentences and duration <= max_duration:
            return [block]
        sentence_map = {u.sentence_id: u for u in sentence_units}
        selected_units = [sentence_map[sid] for sid in block.sentence_ids if sid in sentence_map]
        if len(selected_units) <= 1:
            return [block]
        parts: list[list[AppleSentenceUnit]] = []
        current: list[AppleSentenceUnit] = [selected_units[0]]
        for unit in selected_units[1:]:
            projected_count = len(current) + 1
            projected_duration = unit.end_sec - current[0].start_sec
            if projected_count > max_sentences or projected_duration > max_duration:
                parts.append(current)
                current = [unit]
            else:
                current.append(unit)
        if current:
            parts.append(current)
        if len(parts) <= 1:
            return [block]
        return [
            _make_block(
                part,
                split_reason="sentence_boundary_split",
                parent_sentence_ids=list(block.sentence_ids),
                is_sub_block=True,
                source_index=block.source_index,
            )
            for part in parts
        ]

    blocks: list[AlignmentBlock] = []
    for block in provisional_blocks:
        blocks.extend(_split_block(block))
    for idx, block in enumerate(blocks, start=1):
        block.block_id = f"block_{idx:06d}"
        if not block.parent_sentence_ids:
            block.parent_sentence_ids = list(block.sentence_ids)
    return blocks


def merge_short_chunks(
    chunks: list[dict[str, Any]],
    *,
    min_chars: int = 20,
    min_duration: float = 6.0,
    max_chars: int = 120,
    max_duration: float = 25.0,
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        current = dict(chunk)
        current_text = str(current.get("text", ""))
        current_duration = float(current.get("end_sec", 0.0)) - float(current.get("start_sec", 0.0))
        should_merge = False
        if merged:
            prev = merged[-1]
            prev_text = str(prev.get("text", ""))
            prev_duration = float(prev.get("end_sec", 0.0)) - float(prev.get("start_sec", 0.0))
            combined_text = prev_text + current_text
            combined_duration = float(current.get("end_sec", 0.0)) - float(prev.get("start_sec", 0.0))
            if (
                len(prev_text) < min_chars
                or prev_duration < min_duration
                or len(current_text) < min_chars
                or current_duration < min_duration
                or len(prev_text) < 3
                or len(current_text) < 3
            ) and len(combined_text) <= max_chars and combined_duration <= max_duration:
                should_merge = True
            elif len(prev_text) < min_chars and len(combined_text) <= max_chars and combined_duration <= max_duration:
                should_merge = True
        if should_merge:
            prev = merged[-1]
            prev["text"] = str(prev.get("text", "")) + current_text
            prev["end_sec"] = current["end_sec"]
            prev["apple_char_end"] = current["apple_char_end"]
            prev["apple_update_count"] = max(int(prev.get("apple_update_count", 0)), int(current.get("apple_update_count", 0)))
            prev["apple_rewrite_count"] = max(int(prev.get("apple_rewrite_count", 0)), int(current.get("apple_rewrite_count", 0)))
            prev["apple_stability_score"] = min(float(prev.get("apple_stability_score", 0.0)), float(current.get("apple_stability_score", 0.0)))
            prev["start_sec_is_estimated"] = bool(prev.get("start_sec_is_estimated")) or bool(current.get("start_sec_is_estimated"))
        else:
            merged.append(current)
    return merged


def extract_raw_span_from_runs(runs: list[dict[str, Any]] | None, local_start: int, local_end: int) -> tuple[float | None, float | None]:
    if not runs:
        return None, None
    local_start = max(0, min(local_start, len(runs) - 1))
    local_end = max(local_start + 1, min(local_end, len(runs)))
    first = runs[local_start]
    last = runs[local_end - 1]
    first_range = first.get("audioTimeRange", {})
    last_range = last.get("audioTimeRange", {})
    start_sec = first_range.get("startSeconds")
    end_sec = last_range.get("endSeconds")
    return start_sec, end_sec


def extract_runs_list(result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_obj = result.get("text")
    if isinstance(text_obj, dict):
        text = text_obj.get("text") or ""
        runs = text_obj.get("runs") or []
    else:
        text = str(text_obj or "")
        runs = []
    return text, runs


def estimate_time_by_ratio(start_sec: float, end_sec: float, start_char: int, end_char: int, total_chars: int) -> tuple[float, float]:
    if total_chars <= 0:
        return start_sec, end_sec
    duration = max(end_sec - start_sec, 0.01)
    seg_start = start_sec + duration * (start_char / total_chars)
    seg_end = start_sec + duration * (end_char / total_chars)
    return seg_start, max(seg_end, seg_start + 0.01)


def linear_project(points: list[tuple[int, int]], source_length: int, target_length: int) -> list[float]:
    if source_length <= 0:
        return []
    if not points:
        ratio = (target_length - 1) / max(source_length - 1, 1)
        return [i * ratio for i in range(source_length)]
    unique: dict[int, int] = {}
    for src, dst in sorted(points):
        unique[src] = dst
    src_pts = sorted(unique)
    dst_pts = [unique[src] for src in src_pts]
    if len(src_pts) == 1:
        anchor_src = src_pts[0]
        anchor_dst = dst_pts[0]
        ratio = (target_length - 1) / max(source_length - 1, 1)
        return [anchor_dst + (i - anchor_src) * ratio for i in range(source_length)]
    projected: list[float] = []
    for idx in range(source_length):
        pos = bisect_right(src_pts, idx) - 1
        if pos < 0:
            left_src, left_dst = src_pts[0], dst_pts[0]
            right_src, right_dst = src_pts[1], dst_pts[1]
        elif pos >= len(src_pts) - 1:
            left_src, left_dst = src_pts[-2], dst_pts[-2]
            right_src, right_dst = src_pts[-1], dst_pts[-1]
        else:
            left_src, left_dst = src_pts[pos], dst_pts[pos]
            right_src, right_dst = src_pts[pos + 1], dst_pts[pos + 1]
        if right_src == left_src:
            projected.append(float(left_dst))
            continue
        ratio = (idx - left_src) / (right_src - left_src)
        projected.append(left_dst + ratio * (right_dst - left_dst))
    return projected


def build_mapping_from_equal_blocks(src_text: str, dst_text: str) -> tuple[list[AlignmentOpcode], list[float], list[float], float, float]:
    matcher = difflib.SequenceMatcher(None, src_text, dst_text, autojunk=False)
    anchors: list[AlignmentOpcode] = []
    anchor_pairs: list[tuple[int, int]] = []
    matched_chars = 0
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal" and (a1 - a0) > 0:
            anchors.append(
                AlignmentOpcode(
                    apple_norm_start=b0,
                    apple_norm_end=b1,
                    asr_norm_start=a0,
                    asr_norm_end=a1,
                    score=1.0,
                )
            )
            anchor_pairs.append((a0, b0))
            anchor_pairs.append((max(a1 - 1, a0), max(b1 - 1, b0)))
            matched_chars += a1 - a0
    asr_to_apple = linear_project(anchor_pairs, len(src_text), len(dst_text))
    apple_to_asr = linear_project([(dst, src) for src, dst in anchor_pairs], len(dst_text), len(src_text))
    global_score = matched_chars / max(len(src_text), len(dst_text), 1)
    coverage = matched_chars / max(len(src_text), 1)
    return anchors, asr_to_apple, apple_to_asr, global_score, coverage


def load_episode_files(input_dir: Path, episode_prefix: str | None = None) -> dict[str, Path]:
    input_dir = input_dir.resolve()
    candidates = [p for p in input_dir.iterdir() if p.is_file()]
    normalized = [(nfc_text(p.name), p) for p in candidates]
    episode_prefix_norm = nfc_text(episode_prefix) if episode_prefix else None
    if episode_prefix_norm:
        selected = [p for name, p in normalized if name.startswith(episode_prefix_norm)]
        if not selected:
            raise FileNotFoundError(f"no files match prefix: {episode_prefix}")
        return {p.name: p for p in selected}

    groups: dict[str, list[Path]] = {}
    for name, p in normalized:
        key = episode_prefix_from_filename(name)
        if key:
            groups.setdefault(key, []).append(p)
    if not groups:
        raise FileNotFoundError(f"no episode files found in {input_dir}")
    if len(groups) > 1:
        raise ValueError(f"multiple episode prefixes found: {sorted(groups)}")
    only_key = next(iter(groups))
    return {p.name: p for p in groups[only_key]}


def episode_prefix_from_filename(name: str) -> str | None:
    patterns = [
        ".MacOS-SpeechAnalyzer",
        ".mlx-nemotron-3.5-asr-0.6b",
        ".qwen3-asr-1.7b",
        ".whisper-small",
        ".scores",
        ".stt",
        ".llm",
    ]
    for marker in patterns:
        if marker in name:
            return name.split(marker, 1)[0]
    if name.endswith(".txt") or name.endswith(".json") or name.endswith(".zip"):
        return name.rsplit(".", 1)[0]
    return None


def read_json_members_from_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not members:
            return []
        data = json.loads(zf.read(members[0]).decode("utf-8"))
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
            if isinstance(results, list):
                return results
        return data if isinstance(data, list) else []


def read_first_json_from_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not members:
            raise ValueError(f"zip has no json file: {path}")
        return json.loads(zf.read(members[0]).decode("utf-8"))


def match_segment_to_result(segment: dict[str, Any], result: dict[str, Any]) -> bool:
    seg_text = nfc_text(segment.get("text", ""))
    res_text, _ = extract_runs_list(result)
    if not seg_text or not res_text:
        return False
    seg_norm = normalize_text_with_map(seg_text, mode="match")[0]
    res_norm = normalize_text_with_map(res_text, mode="match")[0]
    if not seg_norm or not res_norm:
        return False
    if seg_norm == res_norm:
        return True
    if seg_norm[:12] == res_norm[:12]:
        return True
    return sequence_similarity(seg_norm[:48], res_norm[:48]) >= 0.7


def majority_prefix_text(texts: list[str]) -> str:
    if not texts:
        return ""
    current = texts[0]
    for text in texts[1:]:
        if text.startswith(current):
            current = text
            continue
        if current.startswith(text):
            continue
        prefix = []
        for a, b in zip(current, text):
            if a != b:
                break
            prefix.append(a)
        current = "".join(prefix)
        if not current:
            current = text
    return current


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def summarize_quality(alignment_score: float, agreement_score: float, needs_review: bool, missing_any: bool) -> str:
    if missing_any:
        return "E"
    if alignment_score >= 0.88 and agreement_score >= 0.88 and not needs_review:
        return "A"
    if alignment_score >= 0.75 and agreement_score >= 0.75 and not needs_review:
        return "B"
    if alignment_score >= 0.4 and agreement_score >= 0.45:
        return "C"
    if alignment_score >= 0.25:
        return "D"
    return "E"


def segment_similarity_score(left: str, right: str) -> float:
    return sequence_similarity(normalize_text_with_map(left, mode="match")[0], normalize_text_with_map(right, mode="match")[0])


def candidate_agreement_score(candidates: dict[str, str]) -> float:
    texts = [text for text in candidates.values() if text]
    if len(texts) <= 1:
        return 1.0 if texts else 0.0
    total = 0.0
    count = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += segment_similarity_score(texts[i], texts[j])
            count += 1
    return total / max(count, 1)


def compute_risk_flags(segment_payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    apple = segment_payload.get("apple", {})
    qwen = segment_payload.get("qwen", {})
    nemotron = segment_payload.get("nemotron", {})
    whisper = segment_payload.get("whisper", {})
    usable_asr_candidates = segment_payload.get("usable_asr_candidates") or {}
    usable_rows = [row for row in usable_asr_candidates.values() if isinstance(row, dict)]
    agreement = float(segment_payload.get("candidate_agreement_score", 0.0))
    quality = segment_payload.get("alignment_quality", "E")
    if float(apple.get("stability_score", 0.0)) < 0.45:
        flags.append("apple_unstable")
    if float(qwen.get("alignment_score", 0.0)) < 0.55:
        flags.append("qwen_alignment_low")
    if float(nemotron.get("alignment_score", 0.0)) < 0.55:
        flags.append("nemotron_alignment_low")
    if float(whisper.get("alignment_score", 0.0)) < 0.55:
        flags.append("whisper_alignment_low")
    qwen_diff_type = segment_payload.get("qwen_apple_difference_type", "semantic")
    if qwen_diff_type in {"critical", "semantic"}:
        flags.append("qwen_apple_disagreement")
    if segment_payload.get("critical_term_disagreement"):
        flags.append("critical_term_disagreement")
    if segment_payload.get("important_term_disagreement") or qwen_diff_type == "soft_domain":
        flags.append("soft_domain_difference")
    if len(usable_rows) >= 2 and agreement < 0.5:
        flags.append("all_models_disagree")
    token_sets = [extract_tokens(c.get("text", "")) for c in (apple, qwen, nemotron, whisper)]
    numeric_sets = [frozenset(ts["numbers"]) for ts in token_sets if ts["numbers"]]
    if len(set(numeric_sets)) > 1:
        flags.append("numeric_disagreement")
    anchor_terms = _extract_anchor_terms([apple.get("text", "")] + [row.get("text", "") for row in usable_rows])
    candidate_anchor_sets = [_extract_anchor_terms([row.get("text", "")]) for row in usable_rows]
    if anchor_terms and len({tuple(sorted(s)) for s in candidate_anchor_sets if s}) > 1 and agreement < 0.55:
        flags.append("proper_noun_disagreement")
    if any(is_large_span_drift(row) for row in usable_rows):
        flags.append("large_span_drift")
    times = segment_payload.get("time", {})
    duration = float(times.get("end_sec", 0.0)) - float(times.get("start_sec", 0.0))
    if duration < 0.3:
        flags.append("span_too_short")
    if duration > 25.0:
        flags.append("span_too_long")
    return sorted(set(flags))


def hash_candidate_context(payload: dict[str, Any]) -> str:
    return json_hash(payload)


def cache_key_for_llm(episode_id: str, segment_id: str, prompt_version: str, candidate_hash: str) -> str:
    return stable_hash(f"{episode_id}|{segment_id}|{prompt_version}|{candidate_hash}")


def http_post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: float = 20.0,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_llm_response(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in (raw_text.find("{"), raw_text.find("[")) if idx != -1]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end_candidates = [raw_text.rfind("}"), raw_text.rfind("]")]
        end = max(end_candidates)
        if end <= start:
            raise
        snippet = raw_text[start : end + 1]
        return json.loads(snippet)


def extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("LLM response missing message content")
    return content


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json_if_exists(path: Path) -> Any | None:
    if path.exists():
        return read_json_from_path(path)
    return None


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_best_candidate(candidate_rows: dict[str, dict[str, Any]]) -> str:
    best_source = ""
    best_score = -1.0
    priority = {"qwen": 3, "apple": 2, "nemotron": 1, "whisper": 0}
    for source, row in candidate_rows.items():
        score = float(row.get("combined_score", 0.0))
        if score > best_score or (score == best_score and priority.get(source, -1) > priority.get(best_source, -1)):
            best_source = source
            best_score = score
    return best_source


def resolve_candidate_summary(candidate_rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {source: row.get("text", "") for source, row in candidate_rows.items()}


def normalize_source_choice(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    aliases = {
        "candidate_a": "qwen",
        "candidate_b": "apple",
        "candidate_c": "nemotron",
        "candidate_d": "whisper",
    }
    return aliases.get(value, value)


def merge_review_reasons(*reason_lists: Iterable[str]) -> list[str]:
    items: list[str] = []
    for reasons in reason_lists:
        for reason in reasons:
            if reason and reason not in items:
                items.append(reason)
    return items


def segment_to_json(segment: AppleSegment) -> dict[str, Any]:
    return dataclasses.asdict(segment)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def chunk_text_by_run_ranges(text: str, runs: list[dict[str, Any]]) -> list[tuple[str, int, int, float | None, float | None]]:
    if not text:
        return []
    chunks: list[tuple[str, int, int, float | None, float | None]] = []
    for chunk, start, end in split_sentence_chunks(text):
        start_sec, end_sec = extract_raw_span_from_runs(runs, start, end)
        chunks.append((chunk, start, end, start_sec, end_sec))
    if not chunks:
        start_sec, end_sec = extract_raw_span_from_runs(runs, 0, len(runs))
        chunks.append((text, 0, len(text), start_sec, end_sec))
    return chunks


def rewrite_count_from_volatile(results: list[dict[str, Any]], start_sec: float, end_sec: float) -> tuple[int, int]:
    overlapping = []
    for result in results:
        text, _ = extract_runs_list(result)
        if not text:
            continue
        r_start = float(result.get("startSeconds") or 0.0)
        r_end = float(result.get("endSeconds") or 0.0)
        if r_end < start_sec - 0.3 or r_start > end_sec + 0.3:
            continue
        overlapping.append((r_end, normalize_text_with_map(text, mode="match")[0]))
    overlapping.sort(key=lambda item: item[0])
    if not overlapping:
        return 0, 0
    update_count = len(overlapping)
    rewrite_count = 0
    previous = overlapping[0][1]
    for _, norm_text in overlapping[1:]:
        if norm_text.startswith(previous):
            previous = norm_text
            continue
        if previous.startswith(norm_text):
            continue
        rewrite_count += 1
        previous = norm_text
    return update_count, rewrite_count


def compute_stability_score(confidence: float | None, update_count: int, rewrite_count: int) -> float:
    base = confidence if confidence is not None else 0.82
    penalty = min(0.38, math.log1p(update_count) * 0.015 + rewrite_count * 0.03)
    return clamp(base - penalty, 0.05, 0.99)


def build_segment_records(
    *,
    episode_id: str,
    final_segments: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    volatile_results: list[dict[str, Any]],
    source_file: Path,
) -> tuple[list[AppleSegment], str]:
    full_parts: list[str] = []
    raw_chunks: list[dict[str, Any]] = []
    for index, segment in enumerate(final_segments, start=1):
        raw_result = raw_results[index - 1] if index - 1 < len(raw_results) else None
        raw_text = nfc_text(str(segment.get("text", "")))
        runs: list[dict[str, Any]] = []
        confidence = None
        if raw_result:
            raw_text, runs = extract_runs_list(raw_result)
            raw_text = nfc_text(raw_text)
            confidence = raw_result.get("confidence")
        parent_start = float(segment.get("startSeconds") or segment.get("start_sec") or 0.0)
        parent_end = float(segment.get("endSeconds") or segment.get("end_sec") or parent_start)
        update_count, rewrite_count = rewrite_count_from_volatile(volatile_results, parent_start, parent_end)
        full_parts.append(raw_text)
    full_text = "".join(full_parts)

    char_timeline: list[dict[str, Any]] = []
    for index, segment in enumerate(final_segments, start=1):
        raw_result = raw_results[index - 1] if index - 1 < len(raw_results) else None
        raw_text = nfc_text(str(segment.get("text", "")))
        runs: list[dict[str, Any]] = []
        confidence = None
        if raw_result:
            raw_text, runs = extract_runs_list(raw_result)
            raw_text = nfc_text(raw_text)
            confidence = raw_result.get("confidence")
        parent_start = float(segment.get("startSeconds") or segment.get("start_sec") or 0.0)
        parent_end = float(segment.get("endSeconds") or segment.get("end_sec") or parent_start)
        duration = max(parent_end - parent_start, 0.01)
        if runs:
            char_runs = runs
        else:
            char_runs = [
                {
                    "audioTimeRange": {
                        "startSeconds": parent_start + (duration * i / max(len(raw_text), 1)),
                        "endSeconds": parent_start + (duration * (i + 1) / max(len(raw_text), 1)),
                    },
                    "text": ch,
                    "transcriptionConfidence": float(confidence or 0.8),
                }
                for i, ch in enumerate(raw_text)
            ]
        for local_index, ch in enumerate(raw_text):
            run = char_runs[min(local_index, len(char_runs) - 1)] if char_runs else None
            if run:
                audio_range = run.get("audioTimeRange", {})
                start_sec = float(audio_range.get("startSeconds", parent_start))
                end_sec = float(audio_range.get("endSeconds", parent_end))
                char_conf = float(run.get("transcriptionConfidence", confidence or 0.8))
            else:
                start_sec, end_sec = estimate_time_by_ratio(parent_start, parent_end, local_index, local_index + 1, max(len(raw_text), 1))
                char_conf = float(confidence or 0.8)
            char_timeline.append(
                {
                    "char": ch,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "confidence": char_conf,
                }
            )
    chunk_specs = split_sentence_chunks(full_text)
    if not chunk_specs:
        chunk_specs = [(full_text, 0, len(full_text))]

    for chunk_index, (chunk_text, local_start, local_end) in enumerate(chunk_specs, start=1):
        if not chunk_text.strip():
            continue
        timeline_slice = char_timeline[local_start:local_end]
        if timeline_slice:
            start_sec = float(timeline_slice[0]["start_sec"])
            end_sec = float(timeline_slice[-1]["end_sec"])
            estimated = False
        else:
            start_sec = end_sec = 0.0
            estimated = True
        raw_chunks.append(
            {
                "episode_id": episode_id,
                "segment_id": f"seg_{chunk_index:06d}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "apple_text_raw": chunk_text,
                "apple_text_stable": chunk_text,
                "apple_char_start": local_start,
                "apple_char_end": local_end,
                "apple_update_count": 0,
                "apple_rewrite_count": 0,
                "apple_stability_score": 0.8,
                "start_sec_is_estimated": estimated,
                "source_file": str(source_file),
                "source_index": chunk_index,
                "confidence": None,
            }
        )

    merged_chunks = merge_short_chunks(raw_chunks, min_chars=20, min_duration=6.0, max_chars=120, max_duration=25.0)
    records: list[AppleSegment] = []
    for idx, chunk in enumerate(merged_chunks, start=1):
        records.append(
            AppleSegment(
                episode_id=episode_id,
                segment_id=f"seg_{idx:06d}",
                start_sec=float(chunk["start_sec"]),
                end_sec=float(chunk["end_sec"]),
                apple_text_raw=str(chunk["apple_text_raw"]),
                apple_text_stable=str(chunk["apple_text_stable"]),
                apple_char_start=int(chunk["apple_char_start"]),
                apple_char_end=int(chunk["apple_char_end"]),
                apple_update_count=int(chunk["apple_update_count"]),
                apple_rewrite_count=int(chunk["apple_rewrite_count"]),
                apple_stability_score=float(chunk["apple_stability_score"]),
                start_sec_is_estimated=bool(chunk["start_sec_is_estimated"]),
                source_file=str(source_file),
                source_index=int(chunk["source_index"]),
                confidence=float(chunk["confidence"]) if chunk["confidence"] is not None else None,
            )
        )
    return records, "".join(full_parts)


def validate_sentence_units(full_text: str, sentence_units: list[AppleSentenceUnit]) -> dict[str, Any]:
    errors: list[str] = []
    very_short_sentence_count = 0
    min_len = None
    max_len = 0
    prev_end = 0
    for unit in sentence_units:
        text = unit.text
        expected = full_text[unit.char_start:unit.char_end]
        if text != expected:
            errors.append(f"text_mismatch:{unit.sentence_id}")
        if unit.char_start < prev_end:
            errors.append(f"overlap:{unit.sentence_id}")
        prev_end = unit.char_end
        length = len(text)
        min_len = length if min_len is None else min(min_len, length)
        max_len = max(max_len, length)
        if length < 3:
            very_short_sentence_count += 1
    return {
        "errors": errors,
        "very_short_sentence_count": very_short_sentence_count,
        "min_sentence_length": min_len or 0,
        "max_sentence_length": max_len,
    }


def validate_alignment_blocks(full_text: str, sentence_units: list[AppleSentenceUnit], blocks: list[AlignmentBlock]) -> dict[str, Any]:
    errors: list[str] = []
    sentence_map = {u.sentence_id: u for u in sentence_units}
    assigned: set[str] = set()
    mismatch_count = 0
    for block in blocks:
        if not block.sentence_ids:
            errors.append(f"empty_block:{block.block_id}")
            continue
        included = [sentence_map.get(sid) for sid in block.sentence_ids]
        if any(item is None for item in included):
            errors.append(f"missing_sentence:{block.block_id}")
            continue
        included = [item for item in included if item is not None]
        block_text = "".join(unit.text for unit in included)
        if block.text != block_text:
            mismatch_count += 1
            errors.append(f"text_mismatch:{block.block_id}")
        expected = full_text[block.char_start:block.char_end]
        if block.text != expected:
            mismatch_count += 1
            errors.append(f"range_mismatch:{block.block_id}")
        if block.start_sec != included[0].start_sec or block.end_sec != included[-1].end_sec:
            errors.append(f"time_mismatch:{block.block_id}")
        for item in included:
            if item.sentence_id in assigned:
                errors.append(f"dup_sentence:{item.sentence_id}")
            assigned.add(item.sentence_id)
    if len(assigned) != len(sentence_units):
        errors.append("sentence_block_consistency_error")
    return {
        "errors": errors,
        "block_text_range_mismatch_count": mismatch_count,
        "sentence_block_consistency_error_count": 1 if "sentence_block_consistency_error" in errors else 0,
    }


def score_candidate_texts(candidate_rows: dict[str, dict[str, Any]]) -> tuple[float, list[str]]:
    usable_texts = {source: row.get("text", "") for source, row in candidate_rows.items() if row.get("usable_for_agreement")}
    agreement = candidate_agreement_score(usable_texts)
    risk_flags: list[str] = []
    if len(usable_texts) >= 2 and agreement < 0.7:
        risk_flags.append("all_models_disagree")
    return agreement, risk_flags
