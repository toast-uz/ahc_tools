#!python

# Tester driver for AtCoder Heuristic Contest

# Show help: python eval.py --help
# Verified by Python3.11 on MacOS Sonoma
# Dependency: pip install optuna ray

# Related tools usage:
# View optuna-db: optuna-dashboard sqlite:///tools/out/optuna.db
#   url_example: http://127.0.0.1:8080/
# ahc_standings: python -m http.server --directory tools/out 8000
#   url_example: http://127.0.0.1:8000/index.html?contest=0_99

import ray
import subprocess
from multiprocessing import cpu_count
import time
import math
import os
import argparse
import optuna

# 条件にあわせて以下のみ変更する
LANGUAGE = 'Rust'  # 'Python' or 'Rust'
FEATURES = ['N', 'M', 'K']  # 特徴量

# 以下は設定変更不要なはす
TESTER = '../target/release/tester'   # インタラクティブの場合
if not os.path.isfile(TESTER):
    TESTER = ''                       # 通常の場合

if LANGUAGE == 'Python':
    TESTEE_SOURCE = 'main.py'             # Pythonの場合
    TESTEE = f'pypy3 {TESTEE_SOURCE}'      # Pythonの場合
    TESTEE_COMPILE = None                 # Pythonの場合
elif LANGUAGE == 'Rust':
    TESTEE_SOURCE = f'src/bin/a.rs'     # Rustの場合
    TESTEE = f'../target/release/{os.getcwd().split("/")[-1]}-a'       # Rustの場合
    TESTEE_COMPILE = f'cargo build -r --bin {os.getcwd().split("/")[-1]}-a'   # Rustの場合

SCORER = '../target/release/vis'      # スコア計算ツール
if not os.path.isfile(SCORER):
    SCORER = '../target/release/score'    # スコア計算ツール（候補その2）
    if not os.path.isfile(SCORER):
        SCORER = ''                       # 存在しない場合
TIMEOUT = 30
RED = '\033[1m\033[31m'
GREEN = '\033[1m\033[32m'
BLUE = '\033[1m\033[34m'
NORMAL = '\033[0m'

# 環境変数で流し込むパラメータ
# int: suugest_intの係数、float: suggest_floatの係数（3番目はstep, 4番目はlog）
#   log: Trueの場合、setpは無視される
# enque: enque_trialの値（複数あれば複数回実行）
PARAMS = {
    'AHC_PARAMS_SAMPLE1': {'int': [0, 1000], 'enque': [500]},
    'AHC_PARAMS_SAMPLE2': {'float': [0.0, 1.0], 'enque': [0.5]},
}

# ログの最後の行から Score = 数字 を探して、スコアを取得する
def get_score_from_log(log):
    try:
        line = log.rstrip().split('\n')[-1].split(' ')
        if len(line) == 3 and line[0] == 'Score' and line[1] == '=':
            return int(line[2])
    except: pass

@ray.remote
class SingleTest:
    def __init__(self, id, dirs, testee, env=None, visible=False, timeout=TIMEOUT):
        self.id = id
        self.input = f'{dirs[0]}/{self.id:04}.txt'
        self.output = f'{dirs[1]}/{self.id:04}.txt'
        self.env = env
        self.visible = visible
        self.timeout = timeout
        self.testee = testee
    def test(self):
        start_time = time.time()
        cp = subprocess.Popen(f'exec {TESTER} {self.testee} < {self.input} > {self.output}',
                            shell=True, env=self.env, stderr=subprocess.PIPE, text=True)
        stderr = []
        while True: # visible=Trueの場合は、標準エラー出力をリアルタイムに表示する
            duration = time.time() - start_time
            line = cp.stderr.readline().rstrip()
            if not line and cp.poll() is not None: break
            if self.visible:
                print(f'{GREEN}{line}{NORMAL}')
            stderr.append(line)
            if duration > self.timeout:
                cp.kill()
                stderr.append(f'Time limit exceeded ({self.timeout}s).')
                break
        duration = time.time() - start_time
        stderr = '\n'.join(stderr)
        # スコアを取得する
        # インタラクティブ型の場合は、テスターの標準エラー出力からスコアを取得することができるため、スコアラーを使わない
        # 非インタラクティブ型の場合でも、提出プログラムでスコア出力を実装していれば、スコアを取得できる
        # なお、インタラクティブ型で提出プログラムでもスコア出力を実装している場合、稀に出力が混在するためうまく動作しない
        score = get_score_from_log(stderr.rstrip())
        if score is None:
            if TESTER:
                # スコアが無ければ、テスターとは別にスコアラーを使う
                cp = subprocess.run(f'{SCORER} {self.input} {self.output}',
                    shell=True, timeout=10, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                score = get_score_from_log(cp.stderr.rstrip())
                if score is None: score = get_score_from_log(cp.stdout.rstrip())
                if score is None: # 標準出力と標準エラー出力の両方にスコアが無ければエラー
                    print(f'{RED}{cp.stderr.rstrip()}{NORMAL}')
                    exit()
            else:
                score = 0
        return self.id, (score, duration)
    def __repr__(self):
        return f'#{self.id:02}'

class Result:
    def __init__(self, id, dirs, score, duration):
        self.id = id
        self.score = score
        self.logscore = math.log10(1 + score)
        self.duration = duration
        self.input = f'{dirs[0]}/{self.id:04}.txt'
        self.read_features_()
        print(self.input)
    def __repr__(self):
        features = ' '.join([f'{k}{v}' for k, v in zip(FEATURES, self.features)]) if self.features is not None else '[FEATURES error]'
        return f'#{self.id:02} {features} score: {self.score}, log: {self.logscore:.3f} time: {self.duration:.3f}s'
    def read_features_(self):
        try:
            with open(f'{self.input}') as f:
                self.features = list(map(float, f.readline().rstrip().split()))
                self.features = [int(x) if x.is_integer() else x for x in self.features]
        except: self.features = None

class Results:
    def __init__(self, dirs):
        self.dirs = dirs
        self.items = []
        self.score_sum = 0
        self.logscore_sum = 0
        self.duration_sum = 0
        self.duration_max = 0
    def __len__(self):
        return len(self.items)
    def append(self, result):
        self.items.append(result)
        self.score_sum += result.score
        self.logscore_sum += result.logscore
        self.duration_sum += result.duration
        self.duration_max = max(self.duration_max, result.duration)

class Objective:
    # コマンドライン引数をもとに、テスト動作のオプションを設定する
    def __init__(self, args, dummy_test=False):
        self.dirs = args.dir
        self.testee = '-'.join(TESTEE.split('-')[:-1] + [TESTEE.split('-')[-1].replace('a', args.testee)])
        if dummy_test:
            self.debug = not args.silent
            self.dbg_('Testing as a dummy because just compiled.')
            self.debug = False
            self.max_concurrent_workers = 1
            self.test_ids = [args.specified[0]]
            self.visible = False
            self.standings = False
        else:
            self.debug = False if args.optuna > 0 else not args.silent
            num_cpus = cpu_count()
            self.max_concurrent_workers = 1 if args.seq else max(1, num_cpus - 1)
            self.dbg_(f'{num_cpus=} max_concurrent_workers={self.max_concurrent_workers}')
            self.test_ids = list(range(args.specified[0], args.specified[-1] + 1))
            self.visible = args.visible
            self.standings = args.standings

    # Optunaで最適化できるように、Objectiveのインスタンスを、関数型で呼び出せるようにする
    def __call__(self, trial=None):
        # 並列テスト実行
        self.dbg_('Testing...', flush=True)
        env = self.set_env_(trial)
        start_time = time.time()
        workers, raw_results, results = [], {}, Results(self.dirs)
        for id in self.test_ids:
            self.dbg_(f'#{id} ', end='', flush=True)
            single_test = SingleTest.remote(id, self.dirs, self.testee,
                env=env, visible=self.visible, timeout=TIMEOUT)
            worker = single_test.test.remote()
            workers.append(worker)
            if len(workers) >= self.max_concurrent_workers:
                finished, workers = ray.wait(workers, num_returns=1)
                self.eval_result_(trial, finished, raw_results, results)    # 結果集計と枝刈り
        self.eval_result_(trial, workers, raw_results, results) # 残った結果集計と枝刈り
        self.dbg_('done.')
        # 後処理
        duration_total = time.time() - start_time
        if self.debug: self.print_score_(results, duration_total)
        if self.standings: self.add_standings_(results)
        return -results.logscore_sum

    # optunaの学習指示を環境変数に流し込む
    def set_env_(self, trial):
        env = os.environ.copy()
        if not trial: return env
        for name, values in PARAMS.items():
            for kind, value in values.items():
                log = value[3] if len(value) > 3 else False
                step = value[2] if len(value) > 2 and not log else None
                if kind == 'int':
                    if step is None:
                        env[name] = str(trial.suggest_int(name, *value[:2], log=log))
                    else:
                        env[name] = str(trial.suggest_int(name, *value[:2], step=step))
                elif kind == 'float':
                    if step is None:
                        env[name] = str(trial.suggest_float(name, *value[:2], log=log))
                    else:
                        env[name] = str(trial.suggest_float(name, *value[:2], step=step))
        return env

    # 並列テストの結果を集計する（Optunaの枝刈りも行う）
    def eval_result_(self, trial, workers, raw_results, results):
        for result in ray.get(workers):
            raw_results[result[0]] = result[1]
        # 並列処理のためテストケースの終了順序は不定であるが、枝刈りできるようにテストケース順通りに集計する
        while len(results) < len(self.test_ids) and (id := self.test_ids[len(results)]) in raw_results:
            results.append(Result(id, results.dirs, *raw_results[id]))
            if not trial: continue
            trial.report(-results.logscore_sum, len(results)) # Optunaに結果を報告して枝刈りする
            if trial.should_prune():
                raise optuna.TrialPruned()

    def dbg_(self, *args, **kwargs):
        if self.debug:
            print(f'{GREEN}', end='')
            print(*args, **kwargs)
            print(f'{NORMAL}', end='', flush=True)

    # 結果を表示する
    def print_score_(self, results, duration_total):
        [self.dbg_(result) for result in results.items]
        self.dbg_(f'Total score: {results.score_sum} log: {results.logscore_sum:.3f} (max time: {results.duration_max:.3f}s)')
        self.dbg_(f'Total time: {duration_total:.3f}s ({duration_total / len(results):.3f}s/test)'      
            f' -> x{results.duration_sum / duration_total:.1f} faster than sequential.')

    # 結果をahc_standingsに追加する
    def add_standings_(self, results):
        dir_ = f'tools/out/{self.test_ids[0]}_{self.test_ids[-1]}'
        os.makedirs(dir_, exist_ok=True)
        # 特徴量をinput.csvに書き出す（新規）
        input_file = f'{dir_}/input.csv'
        if not os.path.isfile(input_file):
            self.dbg_(f'Creating {input_file} ...')
            with open(input_file, 'w') as f:
                f.write(f'file,seed,{",".join(FEATURES)}\n')
                for result in results.items:
                    f.write(f'{result.id:04},0,{",".join(map(str, result.features[:len(FEATURES)]))}\n')
        # result.csvのヘッダーを書き出す（新規）
        result_file = f'{dir_}/result.csv'
        if not os.path.isfile(result_file):
            self.dbg_(f'Creating {result_file} ...')
            with open(result_file, 'w') as f:
                f.write('raw,100000000,https://img.atcoder.jp/ahc_standings/usage.html\n')
        # 結果をresult.csvに書き出す（追記）
        study_name=f'{time.strftime("%Y%m%d-%H%M%S")}'
        with open(result_file, 'a') as f:
            self.dbg_(f'Appending {result_file} ...')
            f.write(f'{study_name}')
            for result in results.items:
                f.write(f',{round(result.logscore * 100000000)}')
            f.write('\n')

# コマンドライン引数をパースする
def parser():
    parser = argparse.ArgumentParser(description='Tester driver for AtCoder Heuristic Contest')
    parser.add_argument(
        '-s', '--specified', nargs='*', type=int,
        help='test specified number as [from [to]]', default=[0, 49])
    parser.add_argument('--silent', help='silent mode', action='store_true')
    parser.add_argument('--seq', help='forced sequential tests.', action='store_true')
    parser.add_argument('--standings', help='add standings entry.', action='store_true')
    parser.add_argument('-v', '--visible', help='visible test stderr.', action='store_true')
    parser.add_argument('-o', '--optuna', type=int,
        help='optuna n_trials, forced --silent', default=0)
    parser.add_argument(
        '--dir', nargs='*', type=str,
        help="custom testcase diroctories as --dir tools/in tools/out",
        default=['tools/in', 'tools/out'])
    parser.add_argument('--testee', type=str,
        help="the char in testee program name replaced from 'a'", default='a')
    return parser.parse_args()

# 提出プログラムのソースが更新されていたらコンパイルする
def compile(args):
    testee_source = '-'.join(TESTEE_SOURCE.split('-')[:-1] + [TESTEE_SOURCE.split('-')[-1].replace('a', args.testee)])
    testee_compile = '-'.join(TESTEE_COMPILE.split('-')[:-1] + [TESTEE_COMPILE.split('-')[-1].replace('a', args.testee)])
    testee = '-'.join(TESTEE.split('-')[:-1] + [TESTEE.split('-')[-1].replace('a', args.testee)])
    assert os.path.isfile(testee_source), f'{RED}Source file {testee_source} not found.{NORMAL}'
    if testee_compile is None:
        return False
    if (os.path.isfile(testee) and
        os.stat(testee_source).st_mtime < os.stat(testee).st_mtime):
            return False
    cp = subprocess.run(testee_compile, shell=True)
    if cp.returncode != 0:
        print(cp.stderr)
        exit(1)
    assert os.path.isfile(testee)
    Objective(args, dummy_test=True)()  # 初回実行は遅いので、計測前に1回だけダミー実行しておく
    return True

def main():
    args = parser()
    for dir_ in args.dir:
        if not os.path.isdir(dir_):
            print(f'{RED}Directory {dir_} not found.{NORMAL}')
            exit(1)
    ray.init(configure_logging=False)
    compile(args)
    if args.optuna == 0:    # Optunaを使わないなら、通常のテストを実施して終了する
        Objective(args)()
        return
    # Optuna studyの生成
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    study = optuna.create_study(
        pruner=pruner,
        study_name=f'{os.getcwd().split("/")[-1]}-{time.strftime("%Y%m%d-%H%M%S")}',
        storage=f'sqlite:///./tools/out/optuna.db', load_if_exists=True)
    # enque_trialの値を設定
    best_params = {}
    best_params_len = 0
    for name, value in PARAMS.items():
        for key, value in value.items():
            if key == 'enque':
                best_params[name] = value
                best_params_len = max(best_params_len, len(value))
    for i in range(best_params_len):
        params = { k: v[i] for k, v in best_params.items() if len(v) > i }
        study.enqueue_trial(params)
    # Optuna studyの最適化
    study.optimize(Objective(args), n_trials=args.optuna)

if __name__ == '__main__':
    main()
