# utils/gen_with_MLE.py
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Sequence

import numpy as np

# ★ 既存実装をそのまま呼ぶ（パッケージ実行前提: python -m utils.gen_with_MLE）
from .likelihood import Likelihood_diagonal_exp
from .formate_matrix_toMLData import matrix_trimer
from .data_generator import DataGenerator, DiagonalTransitionRateMatrixGenerator


class DirichletDeltaT:
    """
    data_generator_discrete.py と同一：
    時間間隔を一様分布(1〜100)からサンプルし、その比率をDirichlet分布で決定
    n_intervals 自体もランダムに決定
    """
    def __init__(self, n_intervals=None, min_intervals=2, max_intervals=10, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

        if n_intervals is None:
            self.n_intervals = int(self.rng.integers(min_intervals, max_intervals + 1))
        else:
            self.n_intervals = n_intervals

        self.intervals = self.rng.uniform(1.0, 100.0, size=self.n_intervals)
        self.weights = self.rng.dirichlet(np.ones(self.n_intervals))

    def sample(self):
        idx = self.rng.choice(self.n_intervals, p=self.weights)
        return self.intervals[idx]


def _seed_for_index(base_seed: int, idx: int) -> int:
    ss = np.random.SeedSequence([base_seed, idx])
    return int(ss.generate_state(1)[0])


def _insert_likelihood_results(M: np.ndarray, num_state: int, init_r: np.ndarray) -> np.ndarray:
    """
    data_generator_discrete.py と同一処理：
      mt = matrix_trimer(M)
      data = mt.trim_data(start=3)
      ll = Likelihood_diagonal_exp(data, num_state=4)
      Q_ll = ll.optimize(np.array([-0.5,-1,-1.5]))
      new_M = np.insert(M,4,Q_ll,axis=0)
    """
    mt = matrix_trimer(M)
    data = mt.trim_data(start=3)
    ll = Likelihood_diagonal_exp(data, num_state=num_state)
    Q_ll = ll.optimize(init_r)
    new_M = np.insert(M, num_state, Q_ll, axis=0)  # states行の直下に挿入（従来どおり）
    return new_M


def _one_dataset_job(
    idx: int,
    out_dir: str,
    states: int,
    lifespan: float,
    min_n: int,
    max_n: int,
    base_seed: int,
    init_r: Sequence[float],
) -> None:
    # 従来コードと同じ：np.random を使う実装に合わせて同期
    child_seed = _seed_for_index(base_seed, idx)
    rng = np.random.default_rng(child_seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 推移率行列生成（従来どおり）
    TRMG = DiagonalTransitionRateMatrixGenerator(states)
    trm = TRMG.generateMatrix(TRMG.setDiagonalElement_byLifespan, lifespan)

    # サンプル数（従来どおり）
    n_samples = int(rng.integers(min_n, max_n + 1))
    dg = DataGenerator(trm, n_samples)

    # delta_t（従来どおり）
    del_t_gen = DirichletDeltaT(min_intervals=2, max_intervals=10, rng=rng)
    M = dg.generate_matrix(del_t_gen.sample)

    # MLE を挿入（従来どおり）
    init_r = np.asarray(init_r, dtype=float)
    M2 = _insert_likelihood_results(M, num_state=states, init_r=init_r)

    # ★従来のCSV命名・出力方式をそのまま使う（これ以外は何も作らない）
    name = str(idx)
    dg.generate_dataFile(M2, name, str(out))


def _parse_args():
    p = argparse.ArgumentParser(description="Discrete generator (same spec) + MLE insert (parallel). No extra outputs.")
    p.add_argument("--count", type=int, required=True, help="生成するデータセット数")
    p.add_argument("--out-dir", type=str, required=True, help="出力ディレクトリ")
    p.add_argument("--states", type=int, default=4, help="状態数")
    p.add_argument("--lifespan", type=float, default=100.0, help="寿命パラメータ")
    p.add_argument("--min-n", type=int, default=5000, help="最小サンプル数")
    p.add_argument("--max-n", type=int, default=5000, help="最大サンプル数")
    p.add_argument("--workers", type=int, default=None, help="並列ワーカー数")
    p.add_argument("--base-seed", type=int, default=20250924, help="再現用ベースシード")
    p.add_argument("--init-r", type=str, default="-0.5,-1,-1.5", help="MLE初期値（カンマ区切り）")
    p.add_argument("--run-parallel", action="store_true", help="並列実行フラグ")
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.run_parallel:
        raise SystemExit("--run-parallel を付けて実行してください")

    # BLAS多重並列抑制（従来と同趣旨）
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    init_r = tuple(float(x) for x in args.init_r.split(",") if x.strip() != "")
    ctx = get_context("fork")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                _one_dataset_job,
                i,
                args.out_dir,
                args.states,
                args.lifespan,
                args.min_n,
                args.max_n,
                args.base_seed,
                init_r,
            )
            for i in range(args.count)
        ]
        for f in as_completed(futures):
            f.result()

    print("done")


if __name__ == "__main__":
    main()
