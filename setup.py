#!python

# Setup script for the tester driver
# 冪等性あり（複数回実行してもよい）

import os
import glob

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

def get_contest_name():
    """コンテスト名を取得する"""
    return os.path.basename(os.getcwd())

def create_if_not_exists(path, type_='file', content=''):
    if type_ == 'file' and os.path.isfile(path) or type_ == 'dir' and os.path.isdir(path):
        print(f'{path} already exists.')
        return
    print(f'Creating {type_} as {path} ...')
    if type_ == 'file':
        with open(path, 'w') as f:
            f.write(content)
    elif type_ == 'dir':
        os.mkdir(path)

def main():
    # 問題をダウンロードする
    contest_name = get_contest_name()
    print('Downloading problem ...')
    os.system(f'curl https://atcoder.jp/contests/{contest_name}/tasks/{contest_name}_a -o problem.html')
    # 問題を読み込んで.zipのファイルパスを取得する
    if not os.path.isfile('problem.html'):
        print('Problem file not found. Please check the contest name or the URL.')
        exit(1)
    with open('problem.html', 'r') as f:
        content = f.read()
    zip_path = None
    for line in content.splitlines():
        if 'href' in line and '.zip' in line:
            zip_path = line.split('"')[1]
            break
    if not zip_path:
        print('No zip file found in the problem page. Please check the contest name or the URL.')
        exit(1)
    print(f'Found zip file: {zip_path}')
    # zipファイルをダウンロードする
    os.system(f'curl {zip_path} -o downloaded_tools.zip')
    # zipファイルを解凍する
    if not os.path.isfile('downloaded_tools.zip'):
        print('Testcases zip file not found. Please check the contest name or the URL.')
        exit(1)
    print('Unzipping testcases.zip ...')
    os.system('unzip -o downloaded_tools.zip -d .')
    # ダウンロードしたzipファイルを削除する
    os.remove('downloaded_tools.zip')
    # toolsディレクトリが無ければ作成を指示して終了する
    if not os.path.isdir('tools'):
        print('toolsディレクトリが無いので作成してください')
        exit(1)
    print('Found tools directory.')
    # rust-toolchainを削除する
    if os.path.isfile('rust-toolchain'):
        print('Removing old rust-toolchain...')
        os.remove('rust-toolchain')
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
    # 必要なファイルやディレクトリを作成する
    create_if_not_exists('.gitignore', 'file', GITIGNORE)
    # v1.70.0は古びたため、toolchainは作成しない
    print('Not create rust-toolchain, because v1.70.0 is too old.')
    #create_if_not_exists('rust-toolchain', 'file', RUST_TOOLCHAIN)
    for dir_ in sorted(glob.glob('tools/in*')):
        create_if_not_exists(dir_.replace('in', 'out'), 'dir')
    # ahc_standingsをダウンロードする
    print('Downloading ahc_standings ...')
    os.system(f'curl {AHC_STANDINGS_URL} -o tools/out/index.html')

if __name__ == '__main__':
    main()
