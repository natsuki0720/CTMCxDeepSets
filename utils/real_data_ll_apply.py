import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .likelihood import Likelihood_diagonal_exp

def _load_and_preprocess_csv(csv_path: str, data_name:str,target_cols: list[str] = None) -> pd.DataFrame:
    """CSVを読み込み、カラム名を統一し、必要な列のみ抽出する."""
    df = pd.read_csv(csv_path)
    df = _pre_process(df,data_name)
    if target_cols is None:
        target_cols = ["pre", "post", "time"]
    df = df[target_cols]
    return df

def _pre_process(df:pd.DataFrame,data_name:str) -> dict:
    if data_name == "suidou":
        df = df.rename(columns={'建設時健全度（1と仮定）': "pre", '調査時健全度': "post", '経過年数': "time"})
    if data_name == "shoban":
        df = df.rename(columns={'Be(1-4)': "pre", 'Af(1-4)': "post", 'Ins': "time"})
        
    if data_name == "Frank":
        def grading(x):
            if x <= 30:
                return 1
            elif x <= 50:
                return 2
            elif x <= 70:
                return 3
            else: 4
        df["Post_IRI_Class"] = df['Post-State IRI'].apply(grading)
        df["Pre_IRI_Class"] = df['Pre-State IRI'].apply(grading)
        df["time"] = df['Inspection Time of PostState IRI']-df['Inspection Time of Prestate']
        df = df.rename(columns={'Pre_IRI_Class': "pre", 'Post_IRI_Class': "post"})
        
    if data_name == "Tunnel":
        df = df.rename(columns={'事前健全度': "pre", '事後健全度': "post", '検査間隔(年)': "time"})
        df = df[["pre", "post", "time"]]
        df = df.dropna()
        df["pre"] = df["pre"].astype(int)
        df["post"] = df["post"].astype(int)

    if data_name == "RCBridge":
        df = df.rename(columns={0: "pre", 1: "post", 2: "time"})
        df = df[df["pre"] < 4]
        df = df[df["post"] < 5]
    
    return df
        
def _sampling(df: pd.DataFrame,num:int) -> np.ndarray:
    arr = df.to_numpy()
    n_rows = arr.shape[0]
    
    if num > n_rows:
        raise ValueError("Sample size exceeds number of available rows.")
    indices = np.random.choice(n_rows, size=num, replace=False)
    sampled_arr = arr[indices,:]
    
    return sampled_arr

def _execute_likelihood(sampled_arr: np.ndarray) -> np.ndarray:
   
    ll = Likelihood_diagonal_exp(sampled_arr, num_state=4)
    Q_ll = ll.optimize(np.array([-0.5,-1,-1.5]))
    return Q_ll


def _concatenate_result(Q_ll: np.ndarray, sampled_arr: np.ndarray) -> np.ndarray:
    """
    Q_ll (4x4) の下に sampled_arr を連結。
    sampled_arr は右端の列を0で埋めてから連結する。
    """
    # sampled_arrにゼロ列を追加
    zeros_col = np.zeros((sampled_arr.shape[0], 1))
    sampled_arr_padded = np.hstack([sampled_arr, zeros_col])

    # 縦方向に連結
    concatenated = np.vstack([Q_ll, sampled_arr_padded])
    return concatenated

def _save_to_csv(result_matrix: np.ndarray, output_path: str):
    """行列をCSVファイルに保存."""
    pd.DataFrame(result_matrix).to_csv(output_path, index=False, header=False)

def _one_dataset_job(idx: int, out_dir: str, base_seed: int, full_arr: np.ndarray, num_samples: int) -> str:
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{idx}_{num_samples}_4.csv")

    rng = np.random.default_rng(base_seed + idx)

    n_rows = full_arr.shape[0]
    if num_samples > n_rows:
        raise ValueError(f"Sample size {num_samples} exceeds available rows {n_rows}.")

    indices = rng.choice(n_rows, size=num_samples, replace=False)
    sampled_arr = full_arr[indices, :]  # ← df.to_numpy() の代わりにそのまま抽出

    Q_ll = _execute_likelihood(sampled_arr)
    result_matrix = _concatenate_result(Q_ll, sampled_arr)
    _save_to_csv(result_matrix, output_path)

    return output_path


def run_parallel_estimation(
    csv_path: str,
    data_name: str,                 # ← data_name に統一
    output_dir: str,
    num_samples: int,
    n_jobs: int,
    base_seed: int = 42,
    max_workers: int = os.cpu_count(),
):
    """
    実データCSVを読み込み、並列でサンプリング→最尤推定→保存を実行する。
    """
    # 1) 読み込み＆前処理（親プロセスで一度だけ）
    df = _load_and_preprocess_csv(csv_path, data_name=data_name, target_cols=["pre", "post", "time"])
    # ndarray 化（以降プロセス間受け渡し）
    full_arr = df.to_numpy()

    os.makedirs(output_dir, exist_ok=True)
    print(f"▶ 開始: {n_jobs}ジョブを並列実行 (samples={num_samples})")

    # 2) 並列実行（DF 全体を渡すが、各ジョブは indices のみ作成して抽出）
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_one_dataset_job, idx, output_dir, base_seed, full_arr, num_samples)
            for idx in range(n_jobs)
        ]
        for f in as_completed(futures):
            try:
                result = f.result()
                print(f"完了: {os.path.basename(result)}")
            except Exception as e:
                print(f"エラー: {e}")

    print("完了")


if __name__ == "__main__":
    df = pd.read_csv("../real_data/suidou.csv")
    l = len(df)
    list_num_samples = [int(l * 0.25), int(l * 0.5), int(l*0.75)]
    for num_samples in list_num_samples:
        run_parallel_estimation(
            csv_path="/home/user/Documents/python/CTMCxDeepSets/real_data/shoban.csv",
            data_name="shoban",         # ← データ種別を指定
            output_dir=f"/media/user/TRANSCEND/datas/real_data/shoban/samples_{num_samples}",
            num_samples=num_samples,
            n_jobs=1000,
            base_seed=123,
            max_workers=64,
        )