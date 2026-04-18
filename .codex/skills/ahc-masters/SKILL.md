# ahc-masters

マスターズ専用のスキル

## 使うタイミング
- マスターズの問題を解くとき。

## ATTENTION
- 問題の分割
 - マスターズでは、通常、複数問題に分割されている
 - <PROBLEM_id>は大文字、<problem_id>は小文字で表され、A/B/Cおよびa/b/cの組み合わせで表されるrことが多い
 - 問題文は共通であり、テストケースの入出力の取りうる値の範囲が異なるだけであることが多い
 - テストケースの入出力の取りうる値の範囲が異なることで、解法は本質的に異なることが多い
 - <PROBLEM_id>のソースは`srs/bin/<problem_id>.rs`、B問題のソースは`srs/bin/<problem_id>.rs`、C問題のソースは`srs/bin/c.rs`である
 - `eval.py`実行時は、`--testee <problem_id> --dir tools/in<PROBLEM_id> tools/out<PROBLEM_id>`のオプションを付与すること
 - A問題のソースを汎用的に作成しておくことで、複数の<PROBLEM_id>に対応することも可能である
 - その場合、`eval.py`実行時は、`--testee`オプションを付与せず、`--dir tools/in<PROBLEM_id> tools/out<PROBLEM_id>`のオプションだけを付与すること
- 提出
  - 提出はマニュアルで行ってもらうこと

## VISUALIZER
  - マスターズではビジュアライザは提供されない
  - 問題文をよく読んで、必要に応じてビジュアライザを作成すること
  - 詳細仕様は、`tools/src/lib.rs` を確認すること
  - ビジュアライザは、`tools/in<PROBLEM_id>/<TESTCASE_id>.txt` と `tools/out<PROBLEM_id>/<TESTCASE_id>.txt` を読み込んで描画するものであることが望ましい
  - <PROBLEM_id>と<TESTCASE_id>は、切り替え可能なものとすることが望ましい
  - プロジェクトルート直下で、`git clone https://github.com/yunix-kyopro/visualizer-template-public` を実行して、ビジュアライザのテンプレートを利用することができる
  - 上記テンプレートには、`AGENTS.md` が含まれており、ビジュアライザの実装方針が記載されている
