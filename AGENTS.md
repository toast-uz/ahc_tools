# AHC Codex Instructions (repo-local)

## Scope / Priority
- この `AGENTS.md` は「その時点で開いている AHC リポジトリ配下」に適用する。
- 指示が衝突する場合は `FORBIDDEN > MUST > SHOULD` を優先する。
- さらに上位（system / developer / user）指示がある場合はそちらを優先する。

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

## Skill Split Policy
- 本ファイルは常時適用ルールのみを保持し、詳細手順は skills に分離する。
- 上位解推定は `ahc-snoop-top`、Optuna/パラメータ最適化は `ahc-optuna`、`eval.py` 実行と commit/破棄判断は `ahc-eval-git` を呼ぶ。
- 提出一連（再現性確認/提出/順位確認/commit）は `ahc_submit` を呼ぶ。
- 複合依頼では該当 skill をすべて呼び、順序は `ahc-snoop-top -> ahc-optuna -> ahc-eval-git` を基本とする。
- 詳細手順は以下に委譲する。
  - `.codex/skills/ahc-snoop-top/SKILL.md`
  - `.codex/skills/ahc-optuna/SKILL.md`
  - `.codex/skills/ahc-eval-git/SKILL.md`
  - `.codex/skills/ahc_submit/SKILL.md`

## AHC Tips
- 強い貪欲解を目指すか、後段探索で改善しやすい初期解を目指すかを先に決める。
- 自由度を固定しても可解なら、骨格を先に固めて残り自由度へ探索を集中する。
- 斜めを考慮すると、非直感的だがブレークするーする場合がある。
- 上位解と乖離している場合は、方針や自由度固定軸を見直す。
