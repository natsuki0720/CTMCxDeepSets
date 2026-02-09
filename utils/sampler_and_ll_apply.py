import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import argparse

import numpy as np

from likelihood import Likelihood_diagonal_exp
from formate_matrix_toMLData import formate_dataMatrix, matrix_trimer


def _seed_for_index(base_seed: int, idx: int) -> int:
    ss = np.random.SeedSequence([base_seed, idx])
    return int(ss.generate_state(1)[0])

def _insert_likelihood_results(M: np.ndarray,start: int) -> np.ndarray:
    mt = matrix_trimer(M)
    data = mt.trim_data(start=3)
    ll = Likelihood_diagonal_exp(data, num_state=4)
    Q_ll = ll.optimize(np.array([-0.5,-1,-1.5]))
    new_M = np.insert(M,start,Q_ll,axis=0)
    return new_M

def _one_dataset_job(idx: int, out_dir: str, data: np.ndarray,num_samples:int, base_seed: int) -> str:
    out_dir = Path(out_dir)
    os.makedirs(out_dir,exist_ok=True)
    path = out_dir/f"{idx}_{num_samples}_4.csv"
    
    
    child_seed = _seed_for_index(base_seed, idx)
    rng = np.random.default_rng(child_seed)
    
    
    samples = rng.choice(data, size = num_samples, replace = False)
    ll = Likelihood_diagonal_exp(samples, num_state=4)
    
    D = _insert_likelihood_results(samples,0)
    np.savetxt(path, D, delimiter=",")
    

def _parse_args_parallel():
    p = argparse.ArgumentParser(description="Parallel runner compatible with the original per-iteration pattern.")
    p.add_argument("--count", type=int, required=True, help="反復回数（= 生成データセット数）")
    p.add_argument("--out-dir", type=str, required=True, help="出力ディレクトリ（例: /mnt/fast/datas/prototype3）")
    p.add_argument("--num_sumples", type=int, default=100, help="サンプル数")
    p.add_argument("--workers", type=int, default=None, help="並列ワーカー数（未指定で論理コア数）")
    p.add_argument("--base-seed", type=int, default=20250912, help="再現用ベースシード")
    p.add_argument("--run-parallel", action="store_true", help="このフラグがあると並列実行ブロックを起動")
    return p.parse_args()

def _run_parallel_from_args(args,data):
    # 過剰並列（BLAS/OpenMPの多重スレッド）を回避
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ctx = get_context("fork")  
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                _one_dataset_job,
                i, args.out_dir, 
                data,
                args.num_sumples, args.base_seed
            )
            for i in range(args.count)
        ]
        for f in as_completed(futures):
            _ = f.result()  # 例外をここで表面化
            
if __name__ == "__main__":
    import copy
    base_args = _parse_args_parallel()
    l = [
        100,200,400,600,800,1000
    ]
    data = np.loadtxt("test_base_10000_4.csv", delimiter=",")
    if not base_args.run_parallel:
        raise SystemExit("run_parallel を有効にしてください")

    for n in l:
        args = copy.deepcopy(base_args)
        print(args.base_seed)
        args.base_seed += n
        print(args.base_seed)
        args.sumples = n
        args.out_dir = base_args.out_dir + f"/test_from_base{n}"
        _run_parallel_from_args(args,data)