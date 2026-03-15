# ahc-eval-git

AHC の評価実行とコミット判断を安全に進めるための skill。

## 使うタイミング
- ユーザーが評価実行（`eval.py`）やコミット可否判断を依頼したとき。
- `-s 0 49` 実行後の扱いを厳密に運用したいとき。

## MUST
- 標準コマンド:
  - 単独ビルド検証: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s <seed> -v`
  - 再現性確認: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49`
  - 速度込み再現性確認: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 --seq`
- 単独 seed は変更影響が出るものに限定する。
- `seed=0` 影響がある場合は `seed=0` を優先する。
- `-s 0 49` 後は次アクション前に必ずユーザー確認する。
  - 全修正を commit
  - 全修正を破棄
- commit の場合、message に `Total score` と `max time` を含める。
- 破棄操作はユーザー明示確認後のみ行う。

## SHOULD
- 実行前後の `git status` を確認し、影響ファイルを報告する。
- 主要結果は seed ごとの差分とあわせて短く要約する。
- `--seq` は性能変化が疑われるときに優先して実施する。

## FORBIDDEN
- `git push` を行わない。
- `-s 0 49` 実行後に、ユーザー確認なしで commit/破棄を行わない。
- 破棄を伴う操作を暗黙実行しない。
- 評価結果未確認のままコミット文面を確定しない。

## OUTPUT
- 実行コマンド一覧
- 主要結果（Total score, max time, 主要seed差分）
- 次アクションの確認事項（commit or discard）
