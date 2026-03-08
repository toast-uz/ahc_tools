# AHC Codex Instructions (repo-local)

## CORE RULES (read first)
- Cargo.toml
  - `Cargo.toml` の edition を確認して、2024 でなければ 2024 に変更する
- tools
  - `tools`配下は変更しないこと
  - `tools/src/lib.rs`には入力生成やスコア算出のためのコードが含まれているため、参考にすること
  - `tools/srs/in/`には入力ケースが含まれている
- Problem
  - 問題文は `problem.html` に保存されている
  - 必ず最初に問題文を読むこと
- Snooping top solution
  - 短期（1日以内）コンテストにおいては、1時間経過ごとに、以下の手段により最上位のソリューションを推定すること
    - `get_standings.py`により現在の順位表をダウンロードする
    - 最上位のスコア、問題文のスコア評価式をもとに、スコアパラメータを推定する
    - スコアパラメータを実現するための必要条件から、ソリューションを絞り込んで推定する
  - 長期コンテストにおいては、相対スコアであること、ケース入力によってベストソリューションが異なる場合が多いこと、などから最上位のソリューション推定は複雑になる
    - 最小スコア（絶対スコア=1）を得るtrival_solution、固定スコア（>最小スコア）を得るtrival_solution2を作成しておく
    - ソリューションに影響が出る区分で、入力パラメータを分類する
    - 分類1つずつにおいて、分類範囲はtrival_solution2、それ以外はtrival_solutionを提出することで、自身の絶対スコアにより分類範囲となるケース数が確定できる
    - 以降は、知りたいケース範囲以外でtrivial_solutionを提出することで、分類範囲だけの1ケース平均て自身の相対スコアを知ることが可能となる
    - これにより、ケース入力が分類範囲での、ベストソリューションを推定可能になる
- Scope
  - AHC only / `src/bin/a.rs` single file / respect structure / minimize diffs
  - Rust 1.89.0 / Cargo.toml listed crates only (uncomment → use)
  - Use rustc_hash (HMap/HSet)
  - Trivial → Greedy → (ask) SA/Beam
  - Japanese concise explanation
- Code Design
  - 既存の `a.rs` をテンプレートとして使う
  - `a.rs` の `struct` / `enum` はそのまま使い、メンバ変数・メソッドの追加修正のみ行う
  - 既存のメンバ変数・メソッドの役割を尊重・優先して実装する
  - 新しい `struct` / `enum` を作る場合は、必ず事前に許可を取る
  - メンバ変数・メソッド以外のグローバル変数・グローバル関数は作らない
- Constants / Parameters
  - Seed（テストケース内容）によらない固定値はconstとして、コードの冒頭に記載する
  - Seed（テストケース内容）によって変わるが解く間は不変な値は、Envのメンバ変数として初期化し不変量として扱う
  - グローバル定数を増やす場合は、必ず事前に許可を取る
  - 0/1 以外のマジックナンバーはコード内に直接書かず、先頭で意味が分かる簡潔な名前の定数として定義して利用する
- Reporting
  - 実装後に全ソースは表示しない（差分・変更点のみ簡潔に示す）

## TEST & GIT WORKFLOW
- Test Commands
  - 単独ビルド検証は `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s <seed> -v` で実行する
  - 単独ビルド検証の seed は、変更影響が出るものに限定する（seed=0 に影響がある場合は seed=0 を優先して使う）。
  - スコア再現性確認は `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49` で行う
  - 実行速度まで含めた厳密な再現性確認は `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 --seq` で行う
- After `-s 0 49`
  - `-s 0 49` 実行後は、必ず「全修正を commit する」か「全修正を破棄する」かをユーザーに確認してから実行する
  - commit を選んだ場合は、適切な commit message を作成して commit する
  - commit message には `eval.py -s 0 49` の `Total score` と `max time` を必ず埋め込む
  - 破棄を選んだ場合は、ユーザー確認後にのみ破棄操作を行う
- Git Safety
  - 絶対にリモートリポジトリへ `push` しない

## HYPER PARAMS tuning with Optuna ##
### 目的
- 定数を環境変数経由で切り替え、`eval.py -o` で Optuna 最適化する。

### 実装ルール（必須）
- 最適化対象の定数 `XXX_DEFAULT` を `a.rs` 冒頭に定義する。
- `Env` に対応メンバ `xxx` を追加し、`Env::init` で初期化する。
- 初期化は必ず以下形式にする（`os_env::get` のキーは snake_case）:
  - `self.xxx = os_env::get("xxx").and_then(|s: String| s.parse::<f64>().ok()).unwrap_or(XXX_DEFAULT);`
- `eval.py` 側の `PARAMS` には `AHC_PARAMS_XXX` を書く（大文字・アンダースコア）。
- `a.rs` 側のキー名は `AHC_PARAMS_` を書かない。`os_env::get("xxx")` にする。
- 最適化対象は「実際に選択ロジックで参照される値」のみに限定する。

### 事前チェック（最適化前に必ず実施）
- `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 -v` でビルド通過。
- 極端値を1つ与えてスコアが変わることを確認する（未反映検知）:
  - 例: `AHC_PARAMS_ATTACK_SA_K1_DENY_SA=0.2 ... eval.py -s 0`
  - 例: `AHC_PARAMS_ATTACK_SA_K1_DENY_SA=2.0 ... eval.py -s 0`
- 変化しない場合は、キー名ミスかロジック未参照を疑って修正する。

### 探索範囲の決め方
- しきい値（turn_ratioなど）: 現在値を中心に ±0.10〜0.20、`step=0.01`。
- 混合係数・重み: 0.0〜2.0 程度から開始、`step=0.01`。
- 強制ボーナス系: 0.0〜3.0 程度から開始、`step=0.01`。
- `enque` は必ず現行初期値を入れる。

### 実行手順
- まず短い試行数で実行:
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 -o 20`
- 良さそうなら本試行:
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 -o 100`

### 反映ルール
- `study.best_trial.params` を確認し、`a.rs` の `*_DEFAULT` に反映する。
- 反映後に必ず再検証:
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 -v`
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49`
  - 必要なら `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 --seq`
- `eval.py` の `enque` は最良値で更新する。
- 最良値が探索範囲の端に寄ったら、次回は探索範囲を見直す。

### 注意事項
- 試行が全て同値なら、未反映（キー名・参照漏れ）か探索対象が効いていない可能性が高い。
- 複数パラメータ同時最適化は可能だが、まずは10〜20個以内で開始する。
