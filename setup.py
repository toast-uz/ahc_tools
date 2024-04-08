#!python

# Setup script for the tester driver
# 冪等性あり（複数回実行してもよい）

import os

GITIGNORE = '''testcases/
tools/
Cargo.lock
.vscode/
eval.py
setup.py
rust-toolchain
vis.html
'''
RUST_TOOLCHAIN = '1.70.0'
AHC_STANDINGS_URL = 'https://img.atcoder.jp/ahc_standings/index.html'

def create_if_not_exists(path, type_='file', content=''):
    if type_ == 'file' and os.path.isfile(path) or type_ == 'dir' and os.path.isdir(path):
        print(f'{path} already exists.')
        return
    print(f'Creating {type_} at {path} ...')
    if type_ == 'file':
        with open(path, 'w') as f:
            f.write(content)
    elif type_ == 'dir':
        os.mkdir(path)

def main():
    # toolsディレクトリが無ければ作成を指示して終了する
    if not os.path.isdir('tools'):
        print('toolsディレクトリが無いので作成してください')
        exit(1)
    print('Found tools directory.')
    # toolsの古いビルドを削除する
    print('Removing old tools build...')
    os.chdir('tools')
    os.system('cargo clean')
    os.chdir('..')
    # toolsのツール群をコンパイルする
    print('Compiling tools...')
    os.chdir('tools')
    os.system('cargo build --release')
    os.chdir('..')
    # ahc_standingsをダウンロードする
    print('Downloading ahc_standings ...')
    os.system(f'curl {AHC_STANDINGS_URL} -o tools/out/index.html')
    # 必要なファイルやディレクトリを作成する
    create_if_not_exists('.gitignore', 'file', GITIGNORE)
    create_if_not_exists('rust-toolchain', 'file', RUST_TOOLCHAIN)
    create_if_not_exists('tools/out', 'dir')

if __name__ == '__main__':
    main()
