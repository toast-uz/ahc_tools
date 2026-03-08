# flake8: noqa

import statistics
import heapq
import random
import itertools
import math
import time
import sys
import bisect

random.seed(0)
ATCODER = False if len(sys.argv) > 1 and sys.argv[1] == '--local' else True
SIMULATE = True if not ATCODER else False
print(f'{ATCODER=} {SIMULATE=}', file=sys.stderr)
MEAN = 10 ** 5
INF = 10 ** 18
np_argmax = lambda x: max([(x, i) for i, x in enumerate(x)])[-1]
np_argmin = lambda x: min([(x, i) for i, x in enumerate(x)])[-1]
np_argsort = lambda x: [x[-1] for x in sorted([(x, i) for i, x in enumerate(x)])]
sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0

class Env:
    def __init__(self, N, D, Q):
        if not SIMULATE:
            N, D, Q = map(int, input().split())
        self.N = N
        self.D = D
        self.Q = Q
        self.seed = [random.randint(0, 2 ** 62) for _ in range(N)]
        self.MAX_SIZE = 10 ** 5 * N // D
        self.true_weights = self.init_true_weights()

    def init_true_weights(self, tries=1):
        weights = []
        lmd = 1 / MEAN
        for _ in range(self.N * tries):
            while True:
                w = max(1, round(random.expovariate(lmd)))
                if w <= self.MAX_SIZE:
                    break
            weights.append(w)
        weights.sort()
        res = [sum(weights[i * tries:(i + 1) * tries]) / tries for i in range(self.N)]
        random.shuffle(res)
        return res

    def reset_true_weights(self):
        self.true_weights = self.init_true_weights()

    def calc_hash(self, objects):
        h = 0
        for i in objects:
            h ^= self.seed[i]
        return h

# 福袋
class Bucket:
    def __init__(self, e, objects, compare, sub_bucket=None):
        self.e = e
        self.objects = objects
        self.sub_bucket = sub_bucket
        self.hash = e.calc_hash(objects)
        self.compare = compare

    def __eq__(self, other):
        return self < other and other < self

    def __gt__(self, other):
        return not(self <= other)

    def __ge__(self, other):
        return not(self < other)

    def __le__(self, other):
        return self < other or self == other

    def __lt__(self, other):
        if self.sub_bucket is None and other.sub_bucket is None:
            return self.compare(self, other)
        else:
            objects1 = self.objects
            objects2 = other.objects
            sub_bucket1 = self.sub_bucket
            sub_bucket2 = other.sub_bucket
            i = 0
            while sub_bucket1 is not None:
                if i % 2 == 0:
                    objects2 += sub_bucket1.objects
                else:
                    objects1 += sub_bucket1.objects
                sub_bucket1 = sub_bucket1.sub_bucket
                i += 1
            i = 0
            while sub_bucket2 is not None:
                if i % 2 == 0:
                    objects1 += sub_bucket2.objects
                else:
                    objects2 += sub_bucket2.objects
                sub_bucket2 = sub_bucket2.sub_bucket
                i += 1
            intersect = set(objects1) & set(objects2)
            objects1 = [i for i in objects1 if i not in intersect]
            objects2 = [i for i in objects2 if i not in intersect]
            bucket1 = Bucket(self.e, objects1, self.compare)
            bucket2 = Bucket(self.e, objects2, self.compare)
            return self.compare(bucket1, bucket2)

    def __add__(self, other):
        assert self.sub_bucket is None
        return Bucket(self.e, self.objects + other.objects, self.compare)

    def __sub__(self, other):
        self_objects_set = set(self.objects)
        other_objects_set = set(other.objects)
        if other_objects_set <= self_objects_set:
            return Bucket(self.e, [i for i in self.objects if i not in other_objects_set], self.compare)
        intersect = self_objects_set & other_objects_set
        objects1 = [i for i in self.objects if i not in intersect]
        objects2 = [i for i in other.objects if i not in intersect]
        sub_bucket = Bucket(self.e, objects2, self.compare)
        return Bucket(self.e, objects1, self.compare, sub_bucket=sub_bucket)

    def __repr__(self):
        if self.sub_bucket is not None:
            return f'Bucket({self.objects} - {self.sub_bucket.objects})'
        else:
            return f'Bucket({self.objects})'

class Compare:
    START_TEMP = 10 ** 5
    TRY_SYNC = 0

    def __init__(self, e, weights, max_counter, trial_counter=False, trial_score=False, true=False):
        self.e = e
        self.weights = weights
        self.cache = {}
        self.hash2objects = {}
        self.counter = 0
        self.max_counter = max_counter
        self.simulate = SIMULATE
        self.trial_counter = trial_counter
        self.trial_score = trial_score
        self.true = true

    def __repr__(self):
        return f'Compare({self.weights=}, {self.cache=}, {self.hash2objects=}, {self.counter=}, {self.max_counter=}, {self.simulate=}, {self.trial=} {self.reverse=}'

    def expired_counter(self):
        return (not self.trial_counter) and (not self.true) and self.counter >= self.max_counter

    # lt: bucket1 < bucket2
    def __call__(self, bucket1, bucket2):
        signal = sign(bucket2.hash - bucket1.hash)
        if signal < 0:
            bucket1, bucket2 = bucket2, bucket1
        # 片方が空ならばもう片方が大きい
        if len(bucket1.objects) == 0:
            return signal > 0
        if len(bucket2.objects) == 0:
            return signal < 0
        # キャッシュにあればそれを返す
        cached_res = self.search_cache(bucket1, bucket2)
        if cached_res is not None:
            return cached_res * signal < 0
        # クエリー残量が無ければ推定サイズで返す
        if self.expired_counter():
            weight1 = sum(self.weights[i] for i in bucket1.objects)
            weight2 = sum(self.weights[i] for i in bucket2.objects)
            return signal * (weight2 - weight1) > 0
        # 新たにクエリーを発行する(object1 > object2)
        res = self.query(self.e, bucket1.objects, bucket2.objects)
        # キャッシュを登録する
        if not self.trial_counter and not self.true:
            assert bucket1.hash != 0 and bucket2.hash != 0
            # 推移率に従ってキャッシュを更新する
            self.add_cache(res, bucket1, bucket2)
            # 重さを更新する
            self.apply_res(res, bucket1.objects, bucket2.objects)
        return res * signal < 0

    def add_cache(self, res, bucket1, bucket2):
        # bucketがキャッシュに含まれていなければ登録する
        for bucket in [bucket1, bucket2]:
            if bucket.hash in self.hash2objects:
                continue
            self.hash2objects[bucket.hash] = bucket.objects.copy()
            '''
            # さらに、bucketに任意のオブジェクト1つ加えたbucketをキャッシュに登録する
            objects_set = set(bucket.objects)
            for i in range(self.e.N):
                if i in objects_set:
                    continue
                added_objects = bucket.objects + [i]
                added_bucket = Bucket(self.e, added_objects, self)
                self.hash2objects[added_bucket.hash] = added_bucket.objects.copy()
                self.cache[bucket.hash].add(added_bucket.hash)
            '''

        # 結果の関係をキャッシュに登録する
        if res == 0:
            if bucket1.hash not in self.cache:
                self.cache[bucket1.hash] = set()
            if bucket2.hash not in self.cache:
                self.cache[bucket2.hash] = set()
            self.cache[bucket1.hash].add(bucket2.hash)
            self.cache[bucket2.hash].add(bucket1.hash)
        elif res < 0:
            if bucket1.hash not in self.cache:
                self.cache[bucket1.hash] = set()
            self.cache[bucket1.hash].add(bucket2.hash)
        else:
            if bucket2.hash not in self.cache:
                self.cache[bucket2.hash] = set()
            self.cache[bucket2.hash].add(bucket1.hash)

    def search_cache(self, bucket1, bucket2):
        u, v = bucket1.hash, bucket2.hash
        u2v = self.reachable_(u, v)
        v2u = self.reachable_(v, u)
        match (u2v, v2u):
            case (True, True):
                return 0
            case (True, False):
                return -1
            case (False, True):
                return 1
            case (False, False):
                return None

    def reachable_(self, u, v):
        todo = [u]
        visited = set()
        while todo:
            u = todo.pop()
            if u == v:
                return True
            if u in visited:
                continue
            visited.add(u)
            if u not in self.cache:
                continue
            for next_ in self.cache[u]:
                todo.append(next_)
        return False

    def apply_res(self, res, object1, object2):
        objects = [object1, object2]
        if res == 0:
            t = [sum(self.weights[i] for i in items) for items in objects]
            rate = [(t[0] + t[1]) / 2 / t[i] for i in range(2)]
            for k in range(2):
                for i in objects[k]:
                    self.weights[i] *= rate[k]
            return
        temp = 1 / (self.counter + 1) * self.START_TEMP
        while True:
            for i in object1:
                self.weights[i] += temp * res
                if self.weights[i] > self.e.MAX_SIZE:
                    self.weights[i] = self.e.MAX_SIZE
            for j in object2:
                self.weights[j] -= temp * res
                if self.weights[j] < 1:
                    self.weights[j] = 1
            t = [sum(self.weights[i] for i in items) for items in objects]
            result_pred = sign(t[0] - t[1])
            if res * result_pred >= 0:
                return

    def query(self, e, object1, object2):
        assert len(object1) > 0 and len(object2) > 0
        assert not self.expired_counter()
        if self.simulate or self.trial_counter or self.trial_score:
            t1 = sum(e.true_weights[i] for i in object1)
            t2 = sum(e.true_weights[i] for i in object2)
            res = sign(t1 - t2)
        else:
            print(f'{len(object1)} {len(object2)} ', end='')
            print(' '.join(map(str, object1 + object2)), flush=True)
            res = input()
            res = 1 if res == '>' else -1 if res == '<' else 0
        self.counter += 1
        return res

    # 残りクエリーを強制消化
    def playout(self, e):
        while self.counter < self.max_counter:
            self.query(e, [0], [1])

class Agent:
    TRY_PAIRS = 10

    def __init__(self, e, true=False, trial_counter=False, trial_score=False):
        self.weights = [MEAN] * e.N
        if true:
            self.weights = e.true_weights
            trial_score = True
        max_counter = e.Q
        self.trial_counter = trial_counter
        self.trial_score = trial_score
        self.true = true
        self.compare = Compare(e, self.weights, max_counter, trial_counter=trial_counter, trial_score=trial_score, true=true)
        self.buckets = [Bucket(e, [i], self.compare) for i in range(e.N)]
        self.ans = []

    # 数分割問題の解法
    def solve(self, e, counter=0, naive=True):
        # ソートのパート
        # 回数がe.Q以下なら普通にソートする
        if self.trial_counter or counter * 1.2 <= e.Q:
            self.buckets.sort()
            # 重さを指数分布に近づける
            weights = sorted(e.init_true_weights(tries=10))
            for i in range(e.N):
                weights[self.buckets[i].objects[0]] = self.weights[i]
            if self.trial_counter:
                return self.compare.counter
        else:
            # 回数がe.Qより大きいなら、ソート回数を減らす
            #print('Special Sort', file=sys.stderr)
            buckets = sorted(self.buckets[:8])
            low_border = buckets[2]
            high_border = buckets[5]
            low = []
            mid = []
            high = []
            for bucket in self.buckets:
                if bucket.hash == low_border.hash or bucket.hash == high_border.hash:
                    continue
                if high_border < bucket:
                    high.append(bucket)
                elif bucket < low_border:
                    low.append(bucket)
                else:
                    mid.append(bucket)
            #high.sort()
            #low.sort()
            # 重さを指数分布に近づける
            weights = sorted(e.init_true_weights(tries=10))
            low_weight = sum(weights[i] for i in range(len(low))) // len(low)
            mid_weight = sum(weights[len(low) + i + 1] for i in range(len(mid))) // len(mid)
            high_weight = sum(weights[len(low) + len(mid) + i + 2] for i in range(len(high))) // len(high)
            for i in range(len(low)):
                weights[low[i].objects[0]] = low_weight
            weights[low_border.objects[0]] = weights[len(low)]
            for i in range(len(mid)):
                weights[mid[i].objects[0]] = mid_weight
            weights[high_border.objects[0]] = weights[len(low) + len(mid) + 1]
            for i in range(len(high)):
                weights[high[i].objects[0]] = high_weight
            self.buckets = low + [low_border] + mid + [high_border] + high

        # バケット分割のパート

        if naive:
            h = [Bucket(e, [], self.compare) for _ in range(e.D)]
            while self.buckets:
                bucket1 = self.buckets.pop()
                bucket2 = heapq.heappop(h)
                bucket = bucket1 + bucket2
                heapq.heappush(h, bucket)
            self.ans = [bucket.objects for bucket in h]
            return

        # バケット分割のパート（複雑）
        sub_ans = [[Bucket(e, [], self.compare) for _ in range(e.D - 1)]
                   + [self.buckets[i]] for i in range(e.N)]
        while len(sub_ans) > 1:
            ans1 = sub_ans.pop()
            ans2 = sub_ans.pop()
            new_ans = [ans1[i] + ans2[e.D - 1 - i] for i in range(e.D)]
            new_ans.sort()
            buckets = [ans[-1] - ans[0] for ans in sub_ans]
            new_ans_bucket = new_ans[-1] - new_ans[0]
            i = bisect.bisect_left(buckets, new_ans_bucket)
            sub_ans.insert(i, new_ans)
        self.ans = [ans.objects for ans in sub_ans[0]]

    def init(self, e):
        self.ans = [[] for _ in range(e.D)]
        for i in range(e.N):
            self.ans[i % e.D].append(i)

    # 残りのクエリーで山登りする
    def optimize(self, e, start_time, limit_time):
        assert (not self.trial_counter) and (not self.true)
        self.buckets = [Bucket(e, objects, self.compare) for objects in self.ans]
        self.buckets.sort()
        j = len(self.buckets) - 1
        while j > 0 and not self.compare.expired_counter() and time.time() - start_time < limit_time:
            i = 0
            while i < j and not self.compare.expired_counter() and time.time() - start_time < limit_time:
                matched = False
                buckets1 = [Bucket(e, [self.buckets[i].objects[k]], self.compare) for k in range(len(self.buckets[i].objects))]
                buckets2 = [Bucket(e, [self.buckets[j].objects[k]], self.compare) for k in range(len(self.buckets[j].objects))]
                buckets2.sort()
                new_buckets1 = self.buckets[i] + buckets2[0]
                new_buckets2 = self.buckets[j] - buckets2[0]
                if self.compare.expired_counter():
                    break
                if new_buckets1 < new_buckets2:
                    self.buckets[i] = new_buckets1
                    self.buckets[j] = new_buckets2
                    matched = True

                if not matched:
                    for i0 in range(len(buckets1)):
                        j0 = bisect.bisect_left(buckets2, buckets1[i0])
                        if j0 == len(buckets2):
                            continue
                        new_buckets1 = self.buckets[i] - buckets1[i0]
                        new_buckets2 = self.buckets[j] - buckets2[j0]
                        if self.compare.expired_counter():
                            break
                        if new_buckets1 < new_buckets2:
                            self.buckets[i] = new_buckets1 + buckets2[j0]
                            self.buckets[j] = new_buckets2 + buckets1[i0]
                            matched = True
                            break
                if matched:
                    self.buckets.sort()
                    j = len(self.buckets) - 1
                    i = 0
                else:
                    i += 1
            j -= 1
        self.ans = [bucket.objects for bucket in self.buckets]

    def compute_score(self, e):
        t = [sum(e.true_weights[i] for i in objects) for objects in self.ans]
        return 1 + round(100 * statistics.pstdev(t))

    def result(self, e):
        assert not SIMULATE
        res = [0] * e.N
        for d, objects in enumerate(self.ans):
            for i in objects:
                res[i] = d
        return ' '.join(map(str, res))

def trial(fixed_N, fixed_D, fixed_Q):
    start_time = time.time()
    e = Env(fixed_N, fixed_D, fixed_Q)
    a_trial = Agent(e, trial_counter=True)
    counter = a_trial.solve(e)

    a = Agent(e)
    if e.N * 2 + e.D * 8 > e.Q:
        policy_id = 0
        a.init(e)
    elif e.N * 9 + e.D * 36 > e.Q:
        policy_id = 1
        a.solve(e, counter=counter, naive=True)
    else:
        policy_id = 2
        a.solve(e, counter=counter, naive=False)
    a.optimize(e, start_time=start_time, limit_time=1.8)
    rest_counter = e.Q - a.compare.counter
    a.compare.playout(e)
    assert a.compare.counter == e.Q
    # 結果出力
    if not SIMULATE:
        print(a.result(e), flush=True)
        exit()

    # スコア
    a_true = Agent(e, true=True)
    a_true.solve(e, naive=True)
    true_score = a_true.compute_score(e)
    score = a.compute_score(e)
    return true_score, score, rest_counter, policy_id

def main():
    if not SIMULATE:
        trial(0, 0, 0)
        exit()

    total_log_score_rate = 0
    num_test_set = 0
    for N, _D, _Q in itertools.product([30, 50, 100], [0, 10, 4], [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32]):
        num_test_set += 1
        if _D == 0:
            D = 2
        else:
            D = N // _D
        Q = N * _Q
        start_time = time.time()
        true_score, score, rest_Q, policy_id = trial(N, D, Q)
        duration = round((time.time() - start_time) * 1000)
        score_rate = score / true_score
        log_score_rate = math.log(1 + score_rate)
        total_log_score_rate += log_score_rate
        print(f'{N=} {D=} {Q=} P{policy_id} {score=} {score_rate=:.1f} -> {duration}ms {rest_Q=:.0f}', file=sys.stderr)
        #exit()
    mean_log_score_rate = total_log_score_rate / num_test_set
    print(f'{mean_log_score_rate=:.3f}', file=sys.stderr)

if __name__ == '__main__':
    main()
