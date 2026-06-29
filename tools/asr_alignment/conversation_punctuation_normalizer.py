from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SHORT_RESPONSE_PREFIXES = (
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

_OPENING_HINTS = (
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

_EXPLANATORY_OPENERS = ("というのも", "それに対して", "例えば", "ただ", "だから", "で、", "この時に", "そうすると", "一体", "これ")
_BOUNDARY_REPLACEMENTS = (
    ("なるほど一体", "なるほど。一体"),
    ("8050問題はい", "8050問題。はい"),
    ("聞いたことないですか8050問題", "聞いたことないですか。8050問題"),
)


@dataclass
class PunctuationNormalizationResult:
    display_text: str
    punctuation_hints: list[dict[str, Any]]
    inserted_period_count: int = 0
    inserted_comma_count: int = 0
    short_response_period_count: int = 0
    possible_speaker_change_period_count: int = 0
    boundary_hint_used_count: int = 0


def _add_hint(hints: list[dict[str, Any]], *, hint_type: str, before: str, after: str, reason: str) -> None:
    hints.append({"type": hint_type, "before": before, "after": after, "reason": reason})


def _normalize_prefix(text: str, short_response_period: bool) -> PunctuationNormalizationResult:
    original = text or ""
    display = original
    hints: list[dict[str, Any]] = []
    inserted_period_count = 0
    inserted_comma_count = 0
    short_response_period_count = 0
    possible_speaker_change_period_count = 0
    boundary_hint_used_count = 0

    for prefix in sorted(_SHORT_RESPONSE_PREFIXES, key=len, reverse=True):
        if not display.startswith(prefix) or len(display) <= len(prefix):
            continue
        remainder = display[len(prefix):]
        if remainder.startswith(("、", "，", " ", "\u3000")):
            remainder = remainder[1:]
        if remainder and remainder[0] not in "。、!?！？":
            before = display
            display = f"{prefix}。{remainder.lstrip()}"
            inserted_period_count += 1
            short_response_period_count += 1
            possible_speaker_change_period_count += 1
            boundary_hint_used_count += 1
            _add_hint(hints, hint_type="short_response_period", before=before, after=display, reason="possible_speaker_change")
        break

    for opener in sorted(_EXPLANATORY_OPENERS, key=len, reverse=True):
        marker = f"。{opener}"
        if display.startswith(opener) and len(display) > len(opener):
            break
        if marker in display:
            before = display
            display = display.replace(marker, f"。{opener}", 1)
            continue

    for before_phrase, after_phrase in _BOUNDARY_REPLACEMENTS:
        if before_phrase in display:
            before = display
            display = display.replace(before_phrase, after_phrase)
            inserted_period_count += display.count(after_phrase) - before.count(after_phrase)
            boundary_hint_used_count += 1
            _add_hint(hints, hint_type="phrase_boundary_period", before=before, after=display, reason="known_conversation_boundary")

    return PunctuationNormalizationResult(
        display_text=display,
        punctuation_hints=hints,
        inserted_period_count=inserted_period_count,
        inserted_comma_count=inserted_comma_count,
        short_response_period_count=short_response_period_count,
        possible_speaker_change_period_count=possible_speaker_change_period_count,
        boundary_hint_used_count=boundary_hint_used_count,
    )


def normalize_conversation_punctuation(text: str, *, short_response_period: bool = True) -> PunctuationNormalizationResult:
    return _normalize_prefix(text, short_response_period)
