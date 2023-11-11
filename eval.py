#!python
# flake8: noqa

# Tester driver for AtCoder Heuristic Contest
# Copyright (c) 2023 toast-uz
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

import ray
import subprocess
from multiprocessing import cpu_count
import time
import math
import os
import argparse
import optuna

DEBUG = True
INTERACTIVE = False
LANGUAGE = 'Rust'  # 'Python' or 'Rust'

# テスト対象の実行コマンド
# （コンテストにあわせて変更する）
if INTERACTIVE:
    TESTER = '../target/release/tester'   # インタラクティブの場合
else:
    TESTER = ''                         # 通常の場合

if LANGUAGE == 'Python':
    TESTEE_SOURCE = 'main.py'             # Pythonの場合
    TESTEE = f'pypy {TESTEE_SOURCE}'      # Pythonの場合
    TESTEE_COMPILE = None                 # Pythonの場合
elif LANGUAGE == 'Rust':
    TESTEE_SOURCE = f'src/bin/a.rs'     # Rustの場合
    TESTEE = f'../target/release/{os.getcwd().split("/")[-1]}-a'       # Rustの場合
    TESTEE_COMPILE = 'cargo build -r'   # Rustの場合

SCORER = '../target/release/vis'      # スコア計算ツール

def dbg(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

@ray.remote
def single_test(i):
    start_time = time.time()
    try:
        cp = subprocess.run(f'{TESTER} {TESTEE} < tools/in/{i:04}.txt > tools/out/{i:04}.txt',
            shell=True, timeout=10, stderr=subprocess.PIPE, text=True)
    except subprocess.TimeoutExpired:
        pass
    duration = time.time() - start_time
    test_err = cp.stderr.rstrip()
    return i, (duration, test_err)

def compute_score(i):
    score = 0
    score_err = None
    cp = subprocess.run(f'{SCORER} tools/in/{i:04}.txt tools/out/{i:04}.txt',
        shell=True, timeout=10, stdout=subprocess.PIPE, text=True)
    try:
        assert cp.stdout.split()[0] == 'Score'
        score = int(cp.stdout.split()[-1])
    except:  # スコア数値が取れない場合、実行エラーメッセージの可能性が高いためそのまま出力
        score_err = cp.stdout.rstrip()
    return score, score_err

def concurrent_tests(test_ids, sequential=False):
    num_cpu = cpu_count()
    max_concurrent_workers = 1 if sequential else max(1, num_cpu // 2 - 1)
    dbg(f'{num_cpu=} {max_concurrent_workers=}')
    dbg('Testing...', flush=True)
    workers = []
    results = {}
    for i in test_ids:
        dbg(f'#{i:02} ', end='', flush=True)
        worker = single_test.remote(i)
        workers.append(worker)
        if len(workers) >= max_concurrent_workers:
            finished, remaining = ray.wait(workers, num_returns=1)
            for result in ray.get(finished):
                results[result[0]] = result[1]
            workers = remaining
    for result in ray.get(workers):
        results[result[0]] = result[1]
    dbg('done.')
    return results

def compute_scores(results):
    dbg('Computing scores...')
    total_score = 0
    total_log_score = 0
    max_duration = 0
    for i in sorted(results.keys()):
        duration, _ = results[i]
        score, _ = compute_score(i)
        log_score = math.log10(1 + score)
        dbg(f'#{i:02} score: {score}, r: {log_score:.3f} time: {duration:.3f}s')
        total_score += score
        total_log_score += log_score
        max_duration = max(max_duration, duration)
    dbg(f'Total score: {total_score} log: {total_log_score:.3f} (max duration: {max_duration:.3f}s)')
    return total_score, total_log_score

def parser():
    parser = argparse.ArgumentParser(
        description='Test driver for AtCoder Heuristic Contest')
    parser.add_argument(
        '-s', '--specified', nargs='*', type=int,
        help='Test specified number as [from [to]] .',
        default=[0, 49])
    parser.add_argument('--seq',
        help='Force sequential tests.', action='store_true')
    return parser.parse_args()

def comlile(force_build=False):
    if TESTEE_COMPILE is None:
        return False
    if ((not force_build) and
        os.path.isfile(TESTEE) and
        os.stat(TESTEE_SOURCE).st_mtime < os.stat(TESTEE).st_mtime):
            return False
    cp = subprocess.run(TESTEE_COMPILE, shell=True)
    if cp.returncode != 0:
        print(cp.stderr)
        exit(1)
    return True

def main():
    ray.init(configure_logging=False)
    # テスト対象のコンパイル
    if comlile():
        dbg('A dummy test just after compiled:')
        concurrent_tests([0])  # コンパイル直後は遅いので、1回だけ実行しておく
    # オプション解析
    args = parser()
    test_ids = list(range(args.specified[0], args.specified[-1] + 1))
    sequential = args.seq
    # テスト実行
    results = concurrent_tests(test_ids, sequential=sequential)
    total_score, total_log_score = compute_scores(results)
    print(f'{total_score} {total_log_score:.3f}')

if __name__ == '__main__':
    main()
