# discrete_generate_mle_save_parallel.py
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

# ★ 既存実装をそのまま呼ぶ（生成プロセス一致）
from likelihood import Likelihood_diagonal_exp
from formate_matrix_toMLData import matrix_trimer  # data_generator_discrete.py と同じ
from data_generator import DataGenerator, DiagonalTransitionRateMatrixGenerator


class DirichletDeltaT:
    """
    data_generator_discrete.py と同一の実装（完全一致させるためここも同形に保持）
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


def _extract_rates_diagonal(Q: np.ndarray) -> np.ndarray:
    """
    対角+隣接上成分のみのモデル（i->i+1）を想定：
    rate_i = Q[i, i+1]
    """
    n = Q.shape[0]
    return np.array([float(Q[i, i + 1]) for i in range(n - 1)], dtype=float)


def _insert_likelihood_results_and_get_Qmle(M: np.ndarray, num_state: int, init_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    data_generator_discrete.py と同じ手順：
      mt = matrix_trimer(M)
      data = mt.trim_data(start=3)
      ll = Likelihood_diagonal_exp(data, num_state=4)
      Q_ll = ll.optimize(np.array([-0.5,-1,-1.5]))
      new_M = np.insert(M, 4, Q_ll, axis=0)
    を踏襲しつつ、Q_ll も返す。
    """
    mt = matrix_trimer(M)
    data = mt.trim_data(start=3)
    ll = Likelihood_diagonal_exp(data, num_state=num_state)
    Q_ll = ll.optimize(init_r)
    new_M = np.insert(M, num_state, Q_ll, axis=0)  # states=4 なら row=4 に挿入（元と一致）
    return new_M, Q_ll


def _save_params_csv(
    path: Path,
    true_rates: np.ndarray,
    mle_rates: np.ndarray,
    true_r: np.ndarray,
    mle_r: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    k = len(true_rates)
    header = ["kind"] + [f"rate_{i+1}" for i in range(k)] + [f"r_{i+1}" for i in range(k)]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(map(str, ["true", *true_rates.tolist(), *true_r.tolist()])) + "\n")
        f.write(",".join(map(str, ["mle", *mle_rates.tolist(), *mle_r.tolist()])) + "\n")


@dataclass
class FitRecord:
    idx: int
    n_samples: int
    num_state: int
    lifespan: float
    seed: int
    true_rates: List[float]
    mle_rates: List[float]
    true_r: List[float]
    mle_r: List[float]
    data_csv: str
    params_csv: str


def _one_dataset_job(
    idx: int,
    out_dir: str,
    states: int,
    lifespan: float,
    min_n: int,
    max_n: int,
    base_seed: int,
    init_r: Sequence[float],
) -> FitRecord:
    # data_generator_discrete.py と同じ乱数同期（np.random を使う実装に合わせて）
    child_seed = _seed_for_index(base_seed, idx)
    rng = np.random.default_rng(child_seed)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 生成（完全一致）
    TRMG = DiagonalTransitionRateMatrixGenerator(states)
    trm = TRMG.generateMatrix(TRMG.setDiagonalElement_byLifespan, lifespan)

    n_samples = int(rng.integers(min_n, max_n + 1))
    dg = DataGenerator(trm, n_samples)

    del_t_gen = DirichletDeltaT(min_intervals=2, max_intervals=10, rng=rng)
    M = dg.generate_matrix(del_t_gen.sample)

    # MLE（完全一致） + CSV上への挿入（完全一致）
    init_r = np.asarray(init_r, dtype=float)
    M2, Q_mle = _insert_likelihood_results_and_get_Qmle(M, num_state=states, init_r=init_r)

    # 生成データCSV保存（ファイル名規約も DataGenerator.generate_dataFile に一致）
    name = str(idx)
    dg.generate_dataFile(M2, name, str(out))
    data_csv_path = out / f"{idx}_{n_samples}_{states}.csv"

    # 真のパラメータ & MLEパラメータ保存
    true_rates = _extract_rates_diagonal(trm)
    mle_rates = _extract_rates_diagonal(Q_mle)
    true_r = np.log(true_rates + 1e-30)

    # Likelihood_diagonal_exp は r を内部最適化して Q を返すだけなので、
    # ここでは mle_r を log(rate) として統一的に保存（比較用途）
    mle_r = np.log(mle_rates + 1e-30)

    params_csv_path = out / f"{idx}_{n_samples}_{states}_params.csv"
    _save_params_csv(params_csv_path, true_rates, mle_rates, true_r, mle_r)

    return FitRecord(
        idx=idx,
        n_samples=n_samples,
        num_state=states,
        lifespan=lifespan,
        seed=child_seed,
        true_rates=true_rates.tolist(),
        mle_rates=mle_rates.tolist(),
        true_r=true_r.tolist(),
        mle_r=mle_r.tolist(),
        data_csv=str(data_csv_path),
        params_csv=str(params_csv_path),
    )


def _write_summary(path: Path, records: List[FitRecord]) -> None:
    if not records:
        return
    k = len(records[0].true_rates)
    header = (
        ["idx", "n_samples", "num_state", "lifespan", "seed"]
        + [f"true_rate_{i+1}" for i in range(k)]
        + [f"mle_rate_{i+1}" for i in range(k)]
        + [f"true_r_{i+1}" for i in range(k)]
        + [f"mle_r_{i+1}" for i in range(k)]
        + ["data_csv", "params_csv"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in sorted(records, key=lambda x: x.idx):
            row = (
                [r.idx, r.n_samples, r.num_state, r.lifespan, r.seed]
                + r.true_rates
                + r.mle_rates
                + r.true_r
                + r.mle_r
                + [r.data_csv, r.params_csv]
            )
            f.write(",".join(map(str, row)) + "\n")


def run_parallel(
    count: int,
    out_dir: str,
    states: int = 4,
    lifespan: float = 100.0,
    min_n: int = 5000,
    max_n: int = 5000,
    workers: Optional[int] = None,
    base_seed: int = 20250924,
    init_r: Sequence[float] = (-0.5, -1.0, -1.5),
) -> Path:
    # data_generator_discrete.py と同じ「BLAS多重並列抑制」
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ctx = get_context("fork")
    records: List[FitRecord] = []

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                _one_dataset_job,
                i, out_dir, states, lifespan, min_n, max_n, base_seed, init_r
            )
            for i in range(count)
        ]
        for f in as_completed(futures):
            records.append(f.result())

    summary_path = Path(out_dir) / "summary_params.csv"
    _write_summary(summary_path, records)
    return summary_path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate discrete data with existing generator and save true vs MLE params (parallel)."
    )
    p.add_argument("--count", type=int, required=True, help="生成するデータセット数")
    p.add_argument("--out-dir", type=str, required=True, help="出力ディレクトリ")
    p.add_argument("--states", type=int, default=4, help="状態数")
    p.add_argument("--lifespan", type=float, default=100.0, help="寿命パラメータ")
    p.add_argument("--min-n", type=int, default=5000, help="最小サンプル数")
    p.add_argument("--max-n", type=int, default=5000, help="最大サンプル数")
    p.add_argument("--workers", type=int, default=None, help="並列ワーカー数")
    p.add_argument("--base-seed", type=int, default=20250924, help="再現用ベースシード")
    p.add_argument("--init-r", type=str, default="-0.5,-1,-1.5", help="Likelihood初期値 r をカンマ区切りで指定")
    p.add_argument("--run-parallel", action="store_true", help="このフラグがあると実行する")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.run_parallel:
        raise SystemExit("run_parallel を有効にしてください（--run-parallel を付けて実行）")

    init_r = tuple(float(x) for x in args.init_r.split(",") if x.strip() != "")
    summary = run_parallel(
        count=args.count,
        out_dir=args.out_dir,
        states=args.states,
        lifespan=args.lifespan,
        min_n=args.min_n,
        max_n=args.max_n,
        workers=args.workers,
        base_seed=args.base_seed,
        init_r=init_r,
    )
    print(f"summary written to: {summary}")
