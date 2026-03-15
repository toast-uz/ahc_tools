# AHC Codex Instructions (repo-local)

## MUST (always)
- 問題文は `problem.html` を最初に読む。
- `Cargo.toml` の edition を確認し、`2024` でなければ `2024` に変更する。
- 変更範囲は AHC 用。原則 `src/bin/a.rs` のみを最小差分で修正する。
- 必要な仕様確認は `tools/src/lib.rs` と `tools/srs/in/` を参照する。
- Rust は 1.89.0 前提。利用クレートは `Cargo.toml` 記載済みのみ（必要なら uncomment）。
- 乱数は `rand_xorshift` を使い、seed は固定して再現性を担保する。
- 連想配列/集合は `rustc_hash`（HMap/HSet）を優先する。
- 既存 `a.rs` をテンプレートとして使い、既存の `struct` / `enum` は維持する。
- 既存メンバ変数・メソッドの役割を尊重し、追加/修正はメンバ内で完結させる。
- 新しい `struct` / `enum` の追加は、事前にユーザー許可を取る。
- 実装報告では全ソースを貼らず、差分と変更点のみを簡潔に示す。

## SHOULD (default policy)
- 解法方針は `Trivial -> Greedy` を優先し、`DP/SA/Beam` は事前相談してから進める。
- 説明文は日本語で簡潔に書く。
- Seed 非依存の固定値は `const` としてファイル先頭に置く。
- Seed 依存だが 1 実行中は不変な値は `Env` のメンバに初期化して扱う。
- `0/1` 以外のマジックナンバーは直接書かず、意味が分かる定数名を付ける。
- 新しいグローバル定数の追加は、事前にユーザー許可を取る。

## FORBIDDEN
- `tools` 配下を変更しない。
- メンバ変数・メソッド以外のグローバル変数・グローバル関数を作らない。
- リモートへの `git push` は行わない。

## TEST / EVAL
- 単独ビルド検証: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s <seed> -v`
- 再現性確認: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49`
- 速度込み再現性確認: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 --seq`
- 単独 seed は変更影響が出るものに限定する。`seed=0` へ影響がある場合は `seed=0` を優先。
- `eval.py -s 0 49` 実行後は、次アクション前に必ずユーザーへ確認する。
  - 全修正を commit する
  - 全修正を破棄する
- commit を選んだ場合、commit message に `Total score` と `max time` を必ず含める。
- 破棄はユーザー明示確認後のみ行う。

## Skill Split Policy
- 本ファイルは「常時適用ルール」のみを保持し、詳細ワークフローは skills に分離する。
- 依頼トリガー: 上位解推定の依頼は `ahc-snoop-top`、Optuna/パラメータ最適化の依頼は `ahc-optuna`、`eval.py` 実行と commit/破棄判断の依頼は `ahc-eval-git` を必ず呼ぶ。
- 複合依頼では該当 skill をすべて呼び、実行順は `ahc-snoop-top -> ahc-optuna -> ahc-eval-git` を基本とする。
- 次の詳細は skill 化推奨:
  - `SNOOP TOP SOLUTION`（短期/長期の上位解推定手順）
  - `HYPER PARAMS tuning with Optuna`（実装規約・試行手順・反映手順）
  - `TEST & GIT WORKFLOW` の詳細運用（`-s 0 49` 後フロー含む）
- このリポジトリの分離先:
  - `.codex/skills/ahc-snoop-top/SKILL.md`
  - `.codex/skills/ahc-optuna/SKILL.md`
  - `.codex/skills/ahc-eval-git/SKILL.md`
- skill 化後は、本ファイルには「いつその skill を使うか」だけ残す。

## Optuna Rule (summary only)
- Optuna 対応時は、`*_DEFAULT` と `Env::init` の `os_env::get("snake_case_key")` を必ずペアで実装する。
- `eval.py` 側は `AHC_PARAMS_XXX`、`a.rs` 側は `xxx` キー（`AHC_PARAMS_` なし）で対応付ける。
- 最適化対象は、実際に意思決定ロジックで参照されるパラメータのみに限定する。
- 反映前後で `-s 0 -v` と `-s 0 49` を再実行して差分を確認する。

## AHC Tips
- 強い貪欲解を目指すか、後段探索で改善しやすい初期解を目指すかを先に決める。
- 自由度を固定しても可解なら、骨格を先に固めて残り自由度へ探索を集中する。
