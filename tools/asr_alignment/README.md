# ASR 整列ツール

Apple SpeechAnalyzer の時刻軸を親にして、既存の ASR 文字起こしを後付けで整列するツールです。  
音声の再 ASR は行わず、手元にあるテキストファイルと Apple 側の時刻情報だけで処理します。

この README は、**別のツールやスクリプトからこのツールを呼び出す人**を想定して、入力・出力・実行方法を具体的に説明します。

## リポジトリの場所

このツールのコードは次のフルパスにあります。

`/Users/masa/Documents/Codex/2026-06-28/tes/tools/asr_alignment`

プロジェクトルートは次です。

`/Users/masa/Documents/Codex/2026-06-28/tes`

## 何をするツールか

このツールは、次の順で処理します。

1. Apple SpeechAnalyzer の結果から、安定した全文と sentence unit を作る
2. Apple の時刻軸に対して、Qwen / Nemotron / Whisper の候補を整列する
3. 各 block ごとに最も自然な候補を選ぶ
4. 必要なら境界ノイズや句読点を整える
5. `final_blocks.jsonl`、`final_transcript.md`、`review_queue.jsonl`、`summary.json` などを出力する

ポイントは、**ASR をやり直すのではなく、既存の ASR 結果を比較・整列する**ことです。

## 入力

標準の入力フォルダは次です。

`/Users/masa/Downloads/test_transcription`

このフォルダには、番組ごとの Apple / Qwen / Nemotron / Whisper 系のテキストや JSON が置かれている前提です。  
複数番組をまとめて扱うこともできますし、個別の番組だけを対象にすることもできます。

### 入力ファイルの考え方

このツールは「この拡張子のファイルだけを読む」という単純な作りではなく、入力ディレクトリの中身から番組単位の素材を集めます。  
そのため、別ツールから呼ぶ場合は次のどちらかで使うのが分かりやすいです。

1. 1番組だけ入った入力ディレクトリを渡す
2. 複数番組入りの入力ディレクトリを渡し、`--episode-prefix` で対象を絞る

## 代表的な実行方法

### 1番組だけ処理する

```bash
python3 /Users/masa/Documents/Codex/2026-06-28/tes/tools/asr_alignment/run_alignment_pipeline.py \
  --input-dir /Users/masa/Downloads/test_transcription \
  --output-dir /Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out \
  --episode-prefix "017-価値観は資本主義-感覚は大家族-その歪みが生む-8050問題-とは-017"
```

### LLM ありで処理する

LLM を使う場合は `--use-llm` を付けます。  
ローカル LLM を使う例は次です。

```bash
python3 /Users/masa/Documents/Codex/2026-06-28/tes/tools/asr_alignment/run_alignment_pipeline.py \
  --input-dir /Users/masa/Downloads/test_transcription \
  --output-dir /Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out \
  --episode-prefix "252-第252回-質問-自分のステイトを理想の状態に変えるもの" \
  --use-llm \
  --llm-endpoint http://127.0.0.1:8010/v1/chat/completions \
  --llm-model assistant \
  --force
```

### 既存の出力先を使い直す

`--resume-output-dir` を指定すると、前回の出力を起点に再実行できます。

```bash
python3 /Users/masa/Documents/Codex/2026-06-28/tes/tools/asr_alignment/run_alignment_pipeline.py \
  --input-dir /Users/masa/Downloads/test_transcription \
  --output-dir /Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out \
  --resume-output-dir /Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out \
  --episode-prefix "252-第252回-質問-自分のステイトを理想の状態に変えるもの" \
  --force
```

## 主なオプション

`run_alignment_pipeline.py` の主な引数は次のとおりです。

- `--input-dir`
  - 入力データのルートディレクトリ
  - 例: `/Users/masa/Downloads/test_transcription`
- `--output-dir`
  - 出力先ディレクトリ
  - 例: `/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out`
- `--episode-prefix`
  - 対象番組やエピソードの接頭辞
  - 例: `252-第252回-質問-自分のステイトを理想の状態に変えるもの`
- `--force`
  - 出力先が既にある場合に上書きする
- `--use-llm`
  - LLM を使って最終候補選択を試す
- `--llm-endpoint`
  - OpenAI 互換 API のエンドポイント
  - 既定値: `http://127.0.0.1:8010/v1/chat/completions`
- `--llm-model`
  - LLM のモデル名
  - 既定値: `assistant`
- `--llm-api-key`
  - ローカル用途の API キー文字列
  - 既定値: `local-qwen3-assistant`
- `--llm-max-segments`
  - LLM を呼ぶ block 数の上限
- `--llm-only-risky`
  - 危険度が高い block だけを LLM 対象にする
- `--candidate-build-mode`
  - 候補構築モード
  - `full` / `staged` / `qwen-only`
- `--workers`
  - 候補構築の並列 worker 数
- `--no-parallel`
  - 並列化を無効にする

## 出力

主な出力先は、指定した `--output-dir` の下です。  
標準では、次のようなディレクトリ構成になります。

`/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out`

### アップル時刻軸

- `apple_timeline/<episode_id>.apple_sentence_units.jsonl`
- `apple_timeline/<episode_id>.alignment_blocks.jsonl`

### 候補整列

- `aligned_segments/<episode_id>.block_candidates.jsonl`
- `aligned_segments/<episode_id>.segment_candidates.jsonl`  
  - 互換用エイリアス

### 最終融合

- `fusion/<episode_id>.final_blocks.jsonl`
- `fusion/<episode_id>.final_segments.jsonl`
- `fusion/<episode_id>.final_transcript.md`
- `fusion/<episode_id>.review_queue.jsonl`
- `fusion/<episode_id>.machine_review_notes.jsonl`
- `fusion/<episode_id>.demoted_review_blocks.jsonl`
- `fusion/<episode_id>.llm_audit.jsonl`
- `fusion/<episode_id>.llm_audit.md`
- `fusion/<episode_id>.sentence_timeline.jsonl`

### レポート

- `reports/<episode_id>.summary.json`
- `reports/<episode_id>.summary.md`
- `reports/<episode_id>.normalized_summary.json`
- `reports/<episode_id>.normalized_summary.md`
- `reports/<episode_id>.review_sample_blocks.json`

## どのファイルを見ればよいか

他ツールから結果を読むときは、まず次の順で確認すると分かりやすいです。

1. `reports/<episode_id>.summary.json`
   - 全体の件数、LLM 使用状況、review 件数、閾値系の集計を見る
2. `fusion/<episode_id>.final_blocks.jsonl`
   - block 単位の最終決定を見る
3. `fusion/<episode_id>.final_transcript.md`
   - 最終的な読み物としての全文を見る
4. `fusion/<episode_id>.review_queue.jsonl`
   - 人手確認や LLM 対象の block を見る
5. `fusion/<episode_id>.llm_audit.jsonl`
   - LLM が何を選んだか、何を変えたかを見る
6. `fusion/<episode_id>.demoted_review_blocks.jsonl`
   - human_required から machine_note に落ちた block を見る

## 複数の Podcast を比較したいとき

複数番組の結果を見たい場合は、入力ディレクトリに複数番組を入れて、番組ごとに `--episode-prefix` を変えながら実行します。  
例えば、以下のように番組ごとに出力先を分けると見やすくなります。

### 例

- 番組 A の出力先  
  `/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out_program_a`
- 番組 B の出力先  
  `/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out_program_b`
- 番組 C の出力先  
  `/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out_program_c`

こうすると、後段のツールが

- review_queue 比率
- machine_note 比率
- LLM 呼び出し件数
- final_text 変更件数
- どの ASR がどれくらい採用されたか

を番組ごとに比較できます。

## 他ツールから呼ぶときの注意

- `--output-dir` は毎回別にするのが安全です
- `--force` を付けると既存出力を上書きします
- `--use-llm` を使う場合は、LLM API が先に起動している必要があります
- ローカル LLM を使うときの既定ポートは `8010` です
- 出力を読む側のツールは、`summary.json` を最初に見ると全体像を掴みやすいです

## 追加の使い方ヒント

- まずは `--use-llm` なしで回して、LLM 以外の整列品質を確認する
- 次に `--use-llm` を付けて、`llm_audit.jsonl` と `review_queue.jsonl` の変化を見る
- 最後に `final_transcript.md` を人間が読む

## 参照パスの例

実運用でよく使う代表パスは次です。

- 入力ルート  
  `/Users/masa/Downloads/test_transcription`
- このツールのコード  
  `/Users/masa/Documents/Codex/2026-06-28/tes/tools/asr_alignment`
- プロジェクトルート  
  `/Users/masa/Documents/Codex/2026-06-28/tes`
- 出力ルート  
  `/Users/masa/Documents/Codex/2026-06-28/tes/_asr_alignment_out`

