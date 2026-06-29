from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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
        ready_endpoint: str | None = None,
        ready_timeout: float = 600.0,
        ready_poll_interval: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.prompt_version = prompt_version
        self.ready_endpoint = ready_endpoint or self._derive_models_endpoint(endpoint)
        self.ready_timeout = ready_timeout
        self.ready_poll_interval = ready_poll_interval
        safe_mkdir(cache_dir)

    @staticmethod
    def _derive_models_endpoint(endpoint: str) -> str:
        if endpoint.endswith("/chat/completions"):
            return endpoint[: -len("/chat/completions")] + "/models"
        return endpoint.rstrip("/") + "/models"

    def _models_ready(self) -> bool:
        req = urllib.request.Request(
            self.ready_endpoint,
            method="GET",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=min(self.timeout, 10.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") or []
        model_ids = {str(item.get("id") or "") for item in data if isinstance(item, dict)}
        if not model_ids:
            return False
        if self.model and self.model not in model_ids:
            return False
        return True

    def wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        last_status: str = "unknown"
        while True:
            try:
                if self._models_ready():
                    print(f"[llm] ready endpoint={self.ready_endpoint} model={self.model}", flush=True)
                    return
                last_status = "model_not_listed"
            except Exception as exc:  # noqa: BLE001
                last_status = f"{type(exc).__name__}: {exc}"
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"LLM model not ready after {self.ready_timeout:.0f}s: endpoint={self.ready_endpoint} model={self.model} last_status={last_status}"
                )
            print(
                f"[llm] waiting endpoint={self.ready_endpoint} model={self.model} last_status={last_status}",
                flush=True,
            )
            time.sleep(self.ready_poll_interval)

    def should_call(self, segment_payload: dict[str, Any], *, only_risky: bool) -> bool:
        if only_risky and not segment_payload.get("needs_review"):
            return False
        quality = segment_payload.get("alignment_quality")
        difference_type = segment_payload.get("qwen_apple_difference_type")
        if quality == "E":
            return True
        if difference_type in {"critical", "semantic"}:
            return True
        if segment_payload.get("numeric_disagreement"):
            return True
        if segment_payload.get("needs_review"):
            return True
        agreement = float(segment_payload.get("candidate_agreement_score", 1.0))
        flags = segment_payload.get("risk_flags") or []
        return quality in {"B", "C", "D"} and (agreement < 0.85 or bool(flags))

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
            "出力は必ずJSONのみとしてください。\n"
            "JSONは次のキーだけを含めてください: selected_source, final_text, confidence, needs_review, review_reason, notes\n"
            "markdown, 補足説明, コードフェンスは禁止です。\n\n"
            f"segment_id: {segment_payload['segment_id']}\n"
            f"time: {segment_payload['time']}\n"
            f"risk_flags: {segment_payload.get('risk_flags', [])}\n"
            f"alignment_quality: {segment_payload.get('alignment_quality')}\n"
            f"candidate_agreement_score: {segment_payload.get('candidate_agreement_score')}\n"
            f"apple_display_text: {segment_payload.get('apple_display_text', segment_payload.get('apple', {}).get('text', ''))}\n"
            f"apple_boundary_hints: {json.dumps(segment_payload.get('apple_boundary_hints', []), ensure_ascii=False)}\n"
            f"candidates: {json.dumps(candidate_summary, ensure_ascii=False)}\n"
        )
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 320,
            "stream": False,
        }

    @staticmethod
    def _normalize_review_reason(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

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
                review_reason=self._normalize_review_reason(payload.get("review_reason")),
                notes=str(payload.get("notes") or ""),
            )

        atomic_write_text(
            cache_file,
            json.dumps({"request": prompt_payload, "status": "pending"}, ensure_ascii=False, indent=2),
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
                review_reason=self._normalize_review_reason(parsed.get("review_reason")),
                notes=str(parsed.get("notes") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            atomic_write_text(
                cache_file,
                json.dumps({"request": prompt_payload, "error": str(exc)}, ensure_ascii=False, indent=2),
            )
            return LLMDecision(success=False, cached=False, error=str(exc))
