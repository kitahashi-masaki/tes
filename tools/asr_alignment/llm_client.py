from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.asr_alignment._core import (  # type: ignore
        LLMDecision,
        cache_key_for_llm,
        atomic_write_text,
        extract_chat_completion_text,
        hash_candidate_context,
        http_post_json,
        load_json_if_exists,
        normalize_source_choice,
        parse_llm_response,
        safe_mkdir,
    )
else:
    from ._core import (
        LLMDecision,
        cache_key_for_llm,
        atomic_write_text,
        extract_chat_completion_text,
        hash_candidate_context,
        http_post_json,
        load_json_if_exists,
        normalize_source_choice,
        parse_llm_response,
        safe_mkdir,
    )


class LLMClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        cache_dir: Path,
        timeout: float = 20.0,
        prompt_version: str = "v1",
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.prompt_version = prompt_version
        safe_mkdir(cache_dir)

    def should_call(self, segment_payload: dict[str, Any], *, only_risky: bool) -> bool:
        if only_risky and not segment_payload.get("needs_review"):
            return False
        quality = segment_payload.get("alignment_quality")
        agreement = float(segment_payload.get("candidate_agreement_score", 1.0))
        flags = segment_payload.get("risk_flags") or []
        return quality in {"B", "C", "D", "E"} or agreement < 0.85 or bool(flags)

    def _build_prompt(self, segment_payload: dict[str, Any]) -> dict[str, Any]:
        candidate_summary = {
            "apple": segment_payload.get("apple", {}).get("text", ""),
            "qwen": segment_payload.get("qwen", {}).get("text", ""),
            "nemotron": segment_payload.get("nemotron", {}).get("text", ""),
            "whisper": segment_payload.get("whisper", {}).get("text", ""),
        }
        prompt = (
            "あなたはASR文字起こし候補の選択器です。\n"
            "候補に存在しない内容を新たに追加してはいけません。\n"
            "音声を聞いたふりをしてはいけません。\n"
            "候補A/B/C/Dの中から最も妥当なものを選んでください。\n"
            "必要なら、候補内の一部表記だけを組み合わせてもよいですが、候補にない語は追加禁止です。\n"
            "数字・固有名詞・専門語は特に慎重に扱ってください。\n"
            "不確実な場合は needs_review=true にしてください。\n"
            "出力は必ずJSONのみとしてください。\n\n"
            f"segment_id: {segment_payload['segment_id']}\n"
            f"time: {segment_payload['time']}\n"
            f"risk_flags: {segment_payload.get('risk_flags', [])}\n"
            f"alignment_quality: {segment_payload.get('alignment_quality')}\n"
            f"candidate_agreement_score: {segment_payload.get('candidate_agreement_score')}\n"
            f"candidates: {json.dumps(candidate_summary, ensure_ascii=False)}\n"
        )
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "max_tokens": 256,
            "stream": False,
        }

    def choose(self, segment_payload: dict[str, Any], *, episode_id: str) -> LLMDecision:
        prompt_payload = self._build_prompt(segment_payload)
        candidate_hash = hash_candidate_context(prompt_payload)
        cache_key = cache_key_for_llm(
            episode_id,
            segment_payload["segment_id"],
            self.prompt_version,
            candidate_hash,
        )
        cache_file = self.cache_dir / f"{cache_key}.json"
        cached = load_json_if_exists(cache_file)
        if cached is not None:
            payload = cached.get("decision") or cached
            return LLMDecision(
                success=True,
                cached=True,
                selected_source=normalize_source_choice(payload.get("selected_source")),
                final_text=payload.get("final_text"),
                confidence=float(payload.get("confidence")) if payload.get("confidence") is not None else None,
                needs_review=payload.get("needs_review"),
                review_reason=list(payload.get("review_reason") or []),
                notes=str(payload.get("notes") or ""),
            )

        try:
            response = http_post_json(self.endpoint, prompt_payload, api_key=self.api_key, timeout=self.timeout)
            content = extract_chat_completion_text(response)
            parsed = parse_llm_response(content)
            selected_source = normalize_source_choice(parsed.get("selected_source") or parsed.get("candidate_id") or parsed.get("source"))
            if selected_source not in {"apple", "qwen", "nemotron", "whisper"}:
                raise ValueError(f"invalid selected_source: {selected_source}")
            decision = {
                "selected_source": selected_source,
                "final_text": parsed.get("final_text"),
                "confidence": parsed.get("confidence"),
                "needs_review": parsed.get("needs_review"),
                "review_reason": parsed.get("review_reason") or [],
                "notes": parsed.get("notes") or "",
            }
            atomic_write_text(
                cache_file,
                json.dumps({"request": prompt_payload, "response": response, "decision": decision}, ensure_ascii=False, indent=2),
            )
            return LLMDecision(
                success=True,
                cached=False,
                selected_source=selected_source,
                final_text=str(parsed.get("final_text") or ""),
                confidence=float(parsed.get("confidence")) if parsed.get("confidence") is not None else None,
                needs_review=bool(parsed.get("needs_review")) if parsed.get("needs_review") is not None else None,
                review_reason=list(parsed.get("review_reason") or []),
                notes=str(parsed.get("notes") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            return LLMDecision(success=False, cached=False, error=str(exc))
