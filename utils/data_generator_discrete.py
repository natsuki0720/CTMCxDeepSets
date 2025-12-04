# new_data_generator.py
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import argparse

from .likelihood import Likelihood_diagonal_exp
from formate_matrix_toMLData import formate_dataMatrix, matrix_trimer

from data_generator import (
    DataGenerator,
    DiagonalTransitionRateMatrixGenerator,
)

class DirichletDeltaT:
    """
    時間間隔を一様分布(1〜100)からサンプルし、
    その比率をDirichlet分布で決定するクラス
    n_intervals 自体もランダムに決定できる
    """
    def __init__(self, n_intervals=None, min_intervals=2, max_intervals=10, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

        # n_intervals をランダムに決定
        if n_intervals is None:
            self.n_intervals = int(self.rng.integers(min_intervals, max_intervals + 1))
        else:
            self.n_intervals = n_intervals

        # 候補となる時間間隔
        self.intervals = self.rng.uniform(1.0, 100.0, size=self.n_intervals)
        # Dirichlet分布で比率決定
        self.weights = self.rng.dirichlet(np.ones(self.n_intervals))

    def sample(self):
        """重みに基づいて1つサンプルを返す"""
        idx = self.rng.choice(self.n_intervals, p=self.weights)
        return self.intervals[idx]


def _seed_for_index(base_seed: int, idx: int) -> int:
    ss = np.random.SeedSequence([base_seed, idx])
    return int(ss.generate_state(1)[0])

def _insert_likelihood_results(M: np.ndarray,) -> np.ndarray:
    mt = matrix_trimer(M)
    data = mt.trim_data(start=3)
    ll = Likelihood_diagonal_exp(data, num_state=4)
    Q_ll = ll.optimize(np.array([-0.5,-1,-1.5]))
    new_M = np.insert(M,4,Q_ll,axis=0)
    return new_M


def _one_dataset_job(idx: int, out_dir: str, states: int, lifespan: float,
                     min_n: int, max_n: int, base_seed: int) -> str:
    # 乱数シード固定
    child_seed = _seed_for_index(base_seed, idx)
    rng = np.random.default_rng(child_seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    name = str(idx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 推移率行列を生成（対角成分のみ）
    TRMG = DiagonalTransitionRateMatrixGenerator(states)
    trm = TRMG.generateMatrix(TRMG.setDiagonalElement_byLifespan, lifespan)

    # サンプル数決定
    n_samples = int(rng.integers(min_n, max_n + 1))
    dg = DataGenerator(trm, n_samples)

    # n_intervals をランダム化した DirichletDeltaT を利用
    del_t_gen = DirichletDeltaT(min_intervals=2, max_intervals=10, rng=rng)
    M = dg.generate_matrix(del_t_gen.sample)
    # M = _insert_likelihood_results(M)
    # CSV出力
    dg.generate_dataFile(M, name, str(out))
    return str(out / name)



def _parse_args_parallel():
    p = argparse.ArgumentParser(description="Parallel Dirichlet delta_t generator")
    p.add_argument("--count", type=int, required=True, help="生成するデータセット数")
    p.add_argument("--out-dir", type=str, help="出力ディレクトリ")
    p.add_argument("--states", type=int, default=4, help="状態数")
    p.add_argument("--lifespan", type=float, default=100.0, help="寿命パラメータ")
    p.add_argument("--min-n", type=int, default=5000, help="最小サンプル数")
    p.add_argument("--max-n", type=int, default=5000, help="最大サンプル数")
    p.add_argument("--workers", type=int, default=None, help="並列ワーカー数")
    p.add_argument("--base-seed", type=int, default=20250924, help="再現用ベースシード")
    p.add_argument("--run-parallel", action="store_true", help="並列実行フラグ")
    return p.parse_args()


def _run_parallel_from_args(args):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ctx = get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                _one_dataset_job,
                i, args.out_dir, args.states, args.lifespan,
                args.min_n, args.max_n, args.base_seed
            )
            for i in range(args.count)
        ]
        for f in as_completed(futures):
            _ = f.result()  # エラーはここで表面化


if __name__ == "__main__":
    import copy
    base_args = _parse_args_parallel()
    # l = [
    #     100,200,400,600,800,1000
    # ]
    if not base_args.run_parallel:
        raise SystemExit("run_parallel を有効にしてください")
    
    # for n in l:
    #     args = copy.deepcopy(base_args)
    #     print(args.base_seed)
    #     args.base_seed += n
    #     print(args.base_seed)
    #     args.min_n = n
    #     args.max_n = n
    #     args.out_dir = base_args.out_dir + f"/testdata_n{n}"
    #     _run_parallel_from_args(args)
    _run_parallel_from_args(args=base_args)
    
        
