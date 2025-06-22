#!python

# Setup script for the tester driver
# 冪等性あり（複数回実行してもよい）

import requests
import re
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

def find_zip_file_path(html):
    """HTMLからzipファイルのパスを探す"""
    pattern = r'href=["\'](https://[^"\']+\.zip)["\']'
    match = re.search(pattern, html)
    if match:
        return match.group(1)
    return None

def download_probelm_and_tools():
    contest_name = get_contest_name()
    # problem.htmlがあれば、それを読み込む
    zip_path = None
    if os.path.isfile('problem.html'):
        with open('problem.html', 'r') as f:
            html = f.read()
        zip_path = find_zip_file_path(html)
    if zip_path is None:
        # problem.htmlが正しくない（zipファイルのパスが見つからない）場合は、AtCoderから問題をダウンロードする
        print(f'Downloading problem {contest_name} ...')
        html = requests.get(f'https://atcoder.jp/contests/{contest_name}/tasks/{contest_name}_a').text
        zip_path = find_zip_file_path(html)
        # htmlを保存する
        with open('problem.html', 'w') as f:
            f.write(html)
    if zip_path is None:
        # zipファイルのパスが見つからない場合は、手動でダウンロードするように指示して終了する
        print('Please manually download the problem page from AtCoder to problem.html and run this script again.')
        exit(1)
    # toolsディレクトリが無ければzipファイルをダウンロードして解凍する
    if not os.path.isdir('tools'):
        print(f'Found zip file: {zip_path}')
        # zipファイルをダウンロードする
        os.system(f'curl {zip_path} -o downloaded_tools.zip')
        # zipファイルを解凍する
        if not os.path.isfile('downloaded_tools.zip'):
            print('Testcases zip file not found. Please check the contest name or the URL.')
            exit(1)
        print('Unzipping testcases.zip ...')
        os.system('unzip -o downloaded_tools.zip -d . > /dev/null 2>&1')
        # ダウンロードしたzipファイルを削除する
        os.remove('downloaded_tools.zip')
    else:
        print(f'Tools directory already exists. Skipping download {zip_path}.')

def split_sections(text: str, section_name: str):
    """
    text を [section_name] で分割し、
    (before, current_block, after_next_section) を返す。
    空の部分は '' を返す。
    該当するセクションが無ければ (text, '', '') を返す。
    [section_name] は行頭・行全体がそれのみの場合に限る。
    """
    m1 = re.search(rf'^\[{re.escape(section_name)}\]\s*$', text, re.MULTILINE)
    if not m1:
        return text, '', ''
    # m1 の後に次の [xxxx] (同条件) を探す
    after_m1 = text[m1.end():]
    m2 = re.search(r'^\[.*?\]\s*$', after_m1, re.MULTILINE)
    if m2:
        pos2 = m1.end() + m2.start()
    else:
        pos2 = len(text)
    before = text[:m1.start()].strip()
    current = text[m1.start():pos2].strip()
    after = text[pos2:].strip()
    return before, current, after

def main():
    # 問題とツールをダウンロードする
    download_probelm_and_tools()
    # Cargo.tomlの[dependencies]セクションを、テンプレートからコピーする
    cargo_toml_path = 'Cargo.toml'
    cargo_toml_template_path = '../../rust_snippets/Cargo.toml'
    if os.path.isfile(cargo_toml_template_path):
        print(f'Using Cargo.toml template from {cargo_toml_template_path}')
        with open(cargo_toml_path, 'r') as f:
            before, _, after = split_sections(f.read(), 'dependencies')
        with open(cargo_toml_template_path, 'r') as f:
            _, current, _ = split_sections(f.read(), 'dependencies')
        new_content = '\n\n'.join([before, current, after])
        with open(cargo_toml_path, 'w') as f:
            f.write(new_content)
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
    # vscodeを起動する
    print('Opening VSCode...')
    os.system('code .')

if __name__ == '__main__':
    main()
