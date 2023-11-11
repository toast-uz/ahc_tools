#!python
# flake8: noqa

# AHC環境セットアップ用スクリプト
# 冪等性あり（複数回実行してもよい）

import os

# toolsディレクトリが無ければ作成を指示して終了する
if not os.path.isdir('tools'):
    print('toolsディレクトリが無いので作成してください')
    exit(1)
print('Found tools directory.')

# .gitignoreが無ければ作成する
print('Creating .gitignore...')
with open('.gitignore', 'w') as f:
    f.write('''testcases/
tools/
Cargo.lock
.vscode/
eval.py
setup.py
rust-toolchain
vis.html
''')

# rust-toolchainを作成する
print('Creating rust-toolchain...')
with open('rust-toolchain', 'w') as f:
    f.write('1.70.0\n')

# tools/outが無ければ作成する
if not os.path.isdir('tools/out'):
    print('Creating tools/out...')
    os.mkdir('tools/out')
else:
    print('Found tools/out.')

# toolsのツール群をコンパイルする
print('Compiling tools...')
os.chdir('tools')
os.system('cargo build --release')
os.chdir('..')
