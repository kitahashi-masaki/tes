from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SHORT_RESPONSE_PREFIXES = (
    "はい",
    "うん",
    "そうですね",
    "そうなんですよね",
    "そうですよね",
    "なるほど",
    "ありますね",
    "聞いたことないです",
    "聞いたことないですか",
    "いや本当ですね",
    "本当はね",
    "そうなんです",
)

OPENING_HINTS = (
    "というのも",
    "それに対して",
    "例えば",
    "ただ",
    "だから",
    "で、",
    "この時に",
    "そうすると",
    "一体",
    "これ",
)


@dataclass
class BoundaryHintResult:
    raw_text: str
    alignment_text: str
    boundary_text: str
    display_text: str
    boundary_hints: list[dict[str, Any]]


def _hint(token: str, *, char_pos_in_raw: int, insert: str = "。") -> dict[str, Any]:
    return {
        "type": "short_response_period",
        "char_pos_in_raw": char_pos_in_raw,
        "token": token,
        "insert": insert,
        "reason": "possible_speaker_change",
        "confidence": 0.85,
        "applies_to": ["display_text", "block_boundary", "cleanup_guard"],
    }


def _normalize_text(text: str) -> BoundaryHintResult:
    raw_text = text or ""
    alignment_text = raw_text
    boundary_text = raw_text
    display_text = raw_text
    hints: list[dict[str, Any]] = []

    for prefix in sorted(SHORT_RESPONSE_PREFIXES, key=len, reverse=True):
        if not display_text.startswith(prefix) or len(display_text) <= len(prefix):
            continue
        remainder = display_text[len(prefix):]
        if remainder.startswith(("、", "，", " ", "\u3000")):
            remainder = remainder[1:]
        if remainder and remainder[0] not in "。、!?！？":
            display_text = f"{prefix}。{remainder.lstrip()}"
            boundary_text = display_text
            hints.append(_hint(prefix, char_pos_in_raw=len(prefix)))
        break

    for opener in sorted(OPENING_HINTS, key=len, reverse=True):
        if display_text.startswith(opener) and len(display_text) > len(opener):
            break

    return BoundaryHintResult(
        raw_text=raw_text,
        alignment_text=alignment_text,
        boundary_text=boundary_text,
        display_text=display_text,
        boundary_hints=hints,
    )


def build_conversation_boundary_hints(items: list[Any], *, text_attr: str = "text") -> list[Any]:
    updated: list[Any] = []
    for item in items:
        text = getattr(item, text_attr, item.get(text_attr, "") if isinstance(item, dict) else "")
        result = _normalize_text(str(text))
        payload = {
            "raw_text": result.raw_text,
            "alignment_text": result.alignment_text,
            "boundary_text": result.boundary_text,
            "display_text": result.display_text,
            "boundary_hints": list(result.boundary_hints),
        }
        if isinstance(item, dict):
            item.update(payload)
            updated.append(item)
            continue
        if hasattr(item, "raw_text"):
            item.raw_text = result.raw_text
        if hasattr(item, "alignment_text"):
            item.alignment_text = result.alignment_text
        if hasattr(item, "boundary_text"):
            item.boundary_text = result.boundary_text
        if hasattr(item, "display_text"):
            item.display_text = result.display_text
        if hasattr(item, "boundary_hints"):
            item.boundary_hints = list(result.boundary_hints)
        updated.append(item)
    return updated
