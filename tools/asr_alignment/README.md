# ASR 整列 PoC

Apple SpeechAnalyzer の時刻軸を親にして、既存 ASR テキストを後付けで整列する PoC です。
音声の再 ASR はしません。既存の JSON / TXT / ZIP のみを使います。

## 出力

主な出力先は `_asr_alignment_out/` です。

- `apple_timeline/<episode_id>.apple_sentence_units.jsonl`
- `apple_timeline/<episode_id>.alignment_blocks.jsonl`
- `aligned_segments/<episode_id>.block_candidates.jsonl`
- `aligned_segments/<episode_id>.segment_candidates.jsonl` 互換用エイリアス
- `fusion/<episode_id>.final_segments.jsonl`
- `fusion/<episode_id>.final_transcript.md`
- `fusion/<episode_id>.review_queue.jsonl`
- `reports/<episode_id>.summary.json`
- `reports/<episode_id>.summary.md`

## いまの分割方針

- Apple の安定全文をまず sentence unit に分割します。
- その後、sentence unit をまとめて alignment block を作ります。
- 長すぎる block は sentence 境界でのみ分割します。
- 最終出力は sentence 単位で、各行に `sentence_id` と `block_id` を持ちます。

## ASR span の考え方

- Apple block から ASR の初期 span を引きます。
- その近傍を局所再探索し、より自然な span を採用します。
- 候補ごとに `initial_char_start/end`、`refined_char_start/end`、`local_alignment_score`、`span_refined`、`span_drift_start/end`、`boundary_contamination`、`usable_for_agreement`、`unusable_reason` を出します。
- `Qwen` を第一候補、`Apple` を第2候補兼時刻軸、`Nemotron` と `Whisper` を補助候補として扱います。
- `Whisper` や `Nemotron` の低スコアだけで `needs_review` は立てません。
- `reports/<episode_id>.summary.json` には `auto_accepted_count`、`needs_review_reason_counts`、`qwen_apple_difference_type_counts` などの集計が入ります。

## 品質判定

- `A` / `B`: 高信頼
- `C`: 概ね良好だが差分あり
- `D`: 確認推奨
- `E`: 要確認

## LLM

- この版では LLM は使いません。
- `--use-llm` は互換性のために残していますが、処理には影響しません。

## 実行例

```bash
python3 tools/asr_alignment/run_alignment_pipeline.py \
  --input-dir /Users/masa/Downloads/test_transcription \
  --output-dir /Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out \
  --episode-prefix "017-価値観は資本主義-感覚は大家族-その歪みが生む-8050問題-とは-017"
```
