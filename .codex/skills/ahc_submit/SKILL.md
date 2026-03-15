# ahc_submit

AHC の提出作業を安全に一貫実行するための skill。

## 使うタイミング
- ユーザーが「再現性確認 -> 提出 -> 順位確認 -> commit」までをまとめて依頼したとき。
- 提出時に手元評価と提出結果を commit message に残したいとき。

## MUST
- 実行順は以下に固定する。
  - 1. `ahc-eval-git` で再現性確認（`eval.py -s 0 49`）を実行する。
  - 2. 結果に `WA` / `TLE` が 1 件でもあれば停止し、提出しない。
  - 3. `cargo compete submit --no-test a` で提出する。
  - 4. 提出後はジャッジ完了まで待つ
  - 5. 中断時に、再開用の案内文をユーザーへ必ず提示する。
  - 6. ユーザーから再開意思が明示されたら再開する。
  - 7. `ahc-snoop-top` を使って順位表と自分の提出を確認する。
  - 8. `ahc-eval-git` を使って commit を実行する。
- commit message には必ず次を含める。
  - 手元再現性確認スコア（`eval.py -s 0 49` の結果）
  - 提出スコア（絶対スコア）
  - 順位（提出時点）、相対スコア
- 中断時の再開案内は次の形式を使う。
  - `提出を実行しました。ジャッジ完了後に続行します。再開する場合は「ahc_submit 再開」と送ってください。`

## SHOULD
- 提出前に `git status` を確認し、提出対象以外の変更混入を避ける。
- 提出結果は URL と提出 ID を残す。
- commit message は検索しやすい固定キーで記録する。
  - `Local total score: ...`
  - `Submitted score: ...`
  - `Rank: ...`

## FORBIDDEN
- `WA` / `TLE` を含む状態で提出しない。
- 順位と提出結果を確認せずに commit しない。
- `ahc-eval-git` の確認フローを飛ばして commit/破棄しない。

## OUTPUT
- 実行コマンド一覧
- 再現性確認結果（`AC/WA/TLE`, Total score, max time）
- 提出結果（提出 ID, 提出スコア）
- 中断時の再開案内文
- 順位情報（順位, 取得時刻）
- commit hash と commit message 要約
