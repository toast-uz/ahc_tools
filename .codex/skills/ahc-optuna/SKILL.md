# ahc-optuna

AHC のハイパーパラメータ最適化（`eval.py -o`）を安全に回すための skill。

## 使うタイミング
- ユーザーが Optuna で定数調整を依頼したとき。
- 環境変数経由で `a.rs` の評価パラメータを最適化したいとき。

## MUST
- 対象定数は `*_DEFAULT` として `a.rs` 冒頭に定義する。
- `Env` に対応メンバを追加し、`Env::init` で初期化する。
- 初期化は次の形式にそろえる:
  - `self.xxx = os_env::get("xxx").and_then(|s: String| s.parse::<f64>().ok()).unwrap_or(XXX_DEFAULT);`
- `eval.py` 側は `AHC_PARAMS_XXX`、`a.rs` 側は `os_env::get("xxx")` を使う。
- 最適化対象は実際の選択ロジックで使う値のみに限定する。
- 事前チェック:
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 -v`
  - 極端値でスコア差分を確認（キー反映漏れ検知）
- 実行:
  - 短試行: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 -o 20`
  - 本試行: `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 -o 100`
- 反映:
  - `study.best_trial.params` を `*_DEFAULT` へ反映する。
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 -v`
  - `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49`
  - 必要なら `/Users/toastuz/Develop/.venv/bin/python3 eval.py -s 0 49 --seq`

## SHOULD
- 探索範囲は現行値の近傍から開始し、端に寄ったら再設定する。
- 一度に最適化するパラメータ数は抑える。
- 実行結果は「変更パラメータ」「best」「再検証結果」をセットで報告する。

## FORBIDDEN
- `a.rs` 側で `AHC_PARAMS_` 接頭辞付きキーを直接参照しない。
- 未使用パラメータを最適化対象に入れない。
- 全試行同値を放置して次ステップへ進めない。

## OUTPUT
- 変更した `*_DEFAULT` 一覧
- best trial の主要パラメータ
- 再検証結果（`-s 0 -v`, `-s 0 49`, 必要なら `--seq`）
