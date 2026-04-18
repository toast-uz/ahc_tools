# ahc-masters

マスターズ専用のスキル

## 使うタイミング
- マスターズの問題を解くとき。

## ATTENTION
- 問題の分割
 - マスターズでは、A問題、B問題、C問題に分割されている
 - A問題のソースは`srs/bin/a.rs`、B問題のソースは`srs/bin/b.rs`、C問題のソースは`srs/bin/c.rs`である
 - `eval.py`実行時は、`--testee <problem_id> --dir tools/in<testcase_id> tools/out<testcase_id>`のオプションを付与すること
  - <problem_id>は、aまたはbまたはcである
  - <testcase_id>は、AまたはBまたはCである
- 提出
  - 提出はマニュアルで行ってもらうこと

## VISUALIZER
  - マスターズではビジュアライザは提供されない
  - 問題文をよく読んで、必要に応じてビジュアライザを作成すること
  - 詳細仕様は、`tools/src/lib.rs` を確認すること
  - ビジュアライザは、`tools/in<problem_id>/<case_id>.txt` と `tools/out<problem_id>/<case_id>.txt` を読み込んで描画するものであることが望ましい
  - <problem_id>と<case_id>は、切り替え可能なものとすることが望ましい
  - プロジェクトルート直下で、`git clone https://github.com/yunix-kyopro/visualizer-template-public` を実行して、ビジュアライザのテンプレートを利用することができる
  - 上記テンプレートには、`AGENTS.md` が含まれており、ビジュアライザの実装方針が記載されている
