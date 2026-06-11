# NPE化 構造化√K 実装・検証サマリ

**期間:** 2026-06-11〜06-12
**対象:** 点推定サロゲート(DeepSets回帰)→ 償却ベイズ推定(NPE)への拡張。特に
「サンプルサイズ依存の事後収縮」を表現させる部分。
**結論:** **ガウス構造ヘッドで本実行GO(検証済み)**。flowヘッドはPhase 2。

関連: [NPE化_差分実装設計書.md](NPE化_差分実装設計書.md)(§10 が本作業の設計、§10.6 が確定版)

---

## 1. 発端 — smokeで点推定→NPEのパイプラインを検証

生成→NPE訓練(flow)→評価まで完走を確認。学習自体は健全に収束(val NLL 2.2→−3.6)。
**ただし NPE化の核心機能が出ていなかった**(run `big_01`, flow, log K を head に連結する設計書原案):

| 指標 | big_01 | 目標 | 判定 |
|---|---|---|---|
| 複製テスト比(X⊕XのSD比) | 1.018 | 0.707 (=1/√2) | ❌ |
| 収縮傾き(SD vs K, log-log) | ≈0 | −0.5 | ❌ |
| SBC KS(全体) | 0.05前後 | 一様 | ✅ |
| 被覆 50/90 | 0.49/0.90 | 0.5/0.9 | ✅ |

**診断:** モデルは log K 入力をほぼ無視。SBC/被覆が良いのは評価データが訓練と同じK分布で
**周辺的な事後幅が平均的に当たっているだけ**(K層別SBCでは低K帯が崩れる)。点推定に対する
NPEの優位性そのものが実証できていない状態。

---

## 2. 方針 — 1/√K を「学習」ではなく「構造」で入れる

対数尤度は `ln L = K·⟨ln p⟩_p̂` ゆえ、対数事後の Hessian は厳密に `K ×(経験平均項の曲率)`。
**p̂ を固定すれば事後幅が 1/√K でスケールするのは恒等式**(崩れるのは事前項と非ガウス性のみ、
K≥500で無視可)。512次元 pooled に1次元 log K を混ぜて学ばせるのは勾配信号が弱すぎる
(利得は1次元あたり高々0.1〜0.3 nat)。→ **アーキテクチャに焼き込む。**

```
z = μ(pooled) + u/√K,     u ~ flow/Gaussian(context=[pooled, logK])
```

- μ は **K不変な pooled のみ**から → 複製テストは平均も構成的に不変。
- 1/√K の主項は構造で保証、残差(小Kの非ガウス性・事前の影響)だけ学習。

---

## 3. 実装(コード変更)

| ファイル | 変更 |
|---|---|
| `models/deepsets_regressor.py` | `sqrt_k_scaling` フラグ。`_encode`→(h, pooled)。`_inv_sqrt_k`。ガウス/flow の構造ヘッド(`TransformedDistribution(AffineTransform(loc=μ, scale=K^{-1/2}))`)。`mu_head`(MLP)。μ-detach + `mu_structural` 露出。構造時は head へ `pooled.detach()`。 |
| `train/train_loop.py` | `NPELoss` に補助MSE(`mu_weight·‖μ−z‖²`)。`fit()` に epoch毎ログ。`TrainLoopConfig.log_every`。 |
| `scripts/train_entrypoint.py` | `--sqrt-k-scaling`/`--logk-scale`、model_config へ保存(評価時の再構築に必須)。 |
| `scripts/evaluate_npe.py` | `--train-k-min/--train-k-max`。収縮傾きを訓練帯内に限定、SBCの帯内subsetを別掲(OOD汚染除去)。 |

**単体テストで確認(未学習でも構成的に成立):**

- ガウス: h固定で K→2K の SD比 = **1/√2(誤差1e-7)**、平均K不変。
- flow: **AffineTransformのヤコビアンバグを発見・修正** — `scale`が`(B,1)`だとevent次元1個分しか
  log-detを数えず `(D−1)·½logK` を取りこぼす → `(B,D)`にbroadcastで修正(手計算照合で検出)。
- flow複製比(乱数初期化): 構造 **[0.71]** vs 非構造 **[1.0]** → big_01の失敗を対照で再現。

---

## 4. 詰めた問題 — μ精度が全変種の共通ボトルネック

素朴な構造化は**較正が崩れた**。`u = √K·(z−μ)` は μ が不正確だと O(√K) に膨らむ:

- **flow**: NSF有界スプライン域(±5)に密度を置けず NLL爆発(val +560で停滞)。
- **ガウス**: 共分散がlogKで高K側を膨らませ残差を吸収 → K^{-1/2}収縮を相殺、傾き−0.5→−0.33。

症状は違うが根は同じ「**μ(pooled)→z の回帰精度**」。3点で対処:

1. μ を **detach + 補助MSE** で学習(NLLのμ勾配は精度∝Kで病的)。
2. エンコーダを scale/shape NLL から隔離(`pooled.detach()`)。
3. **`mu_head` を MLP化**(512→128→64→3)← **最重要**。

**μ回帰の下限を実測**(純MSE):

| ヘッド | val-RMSE 下限 |
|---|---|
| Linear | **0.34**(頭打ち) |
| MLP | **0.14**(まだ降下中) |

線形読み出しでは pooled から z を抽出しきれない(エンコーダが万能でも実最適化で不足)。
この0.34がすべてを壊していた。

---

## 5. 検証結果(実データ 8000セット, K∈[500,5000], 既存 `discrete_train_with_mle`)

| 指標 | **struct_gauss2**(構造+MLP) | control(学習) | big_01(学習) | 目標 |
|---|---|---|---|---|
| 複製比 | **0.713** ✓ | 0.973 ✗ | 1.018 ✗ | 0.707 |
| 収縮傾き(帯内) | **−0.48/−0.46/−0.50** ✓ | ≈0 ✗ | ≈0 ✗ | −0.5 |
| SBC KS(帯内) | **0.11/0.06/0.07** ✓ | 0.11/0.11/0.19 | — | 小 |
| 被覆 50/90 | **0.54/0.91** ✓ | 0.51/0.89 | 0.49/0.90 | 0.5/0.9 |
| 厳密事後 平均z誤差 | **0.055** ✓ | 0.082 | — | 小 |
| μ-RMSE | **0.11** | — | — | ~0.08 |

**ガウス構造ヘッドは「K収縮の構造保証」と「較正」を同時達成**。controlの較正に並びつつ、
controlに無いK収縮を獲得、平均精度はcontrolを上回る。

### 中間run一覧(`artifacts/runs/`)

| run | 構成 | 結果 |
|---|---|---|
| `big_01` | flow, 学習, logk_scale=5 | 複製1.02・傾き0。発端の失敗 |
| `control` | flow, 学習, logk_scale=1 | 較正良好だがK収縮なし(複製0.97)。ablation対照 |
| `struct`/`struct2`/`struct3` | flow, 構造(段階的修正) | K-scaling✓だが NSF有界域で較正✗(val+550停滞) |
| `struct_gauss` | gaussian, 構造, **Linear**ヘッド | μ-RMSE0.3 → 傾き−0.33に劣化 |
| **`struct_gauss2`** | **gaussian, 構造, MLP**ヘッド | **全指標達成(上表)。本実行構成** |

---

## 6. 結論と本実行可否

- ✅ **ガウス構造ヘッド = 本実行GO**。8000件の実runで実証済み(おもちゃでない)。
- ⏳ ガチ本番(5万件・--n 40000)自体は未実行だが、**検証済み構成のデータ増**のみ。
  μ精度(唯一のボトルネック)はデータ増で改善方向なので悪化理由なし、同等以上の見込み。
- ⚠️ **flowヘッドは未確認(Phase 2)**。NSF有界域のため、MLP化してもμ過渡期に残差が端に乗る。
  μウォームアップ or NSFの`bound`拡大が必要。当面はガウス推奨。

### 正直な留保
μ-RMSE 0.11 は理想(0.05〜0.08)に僅か届かず、dim2傾き−0.464に微小な共分散インフレの名残。
較正は合格圏だが「完璧」でなく「堅実に通った」。データ量増で詰まる余地あり。

---

## 7. 本実行レシピ(tmux, ガウス構造)

```bash
cd /home/user/Documents/python/CTMCxDeepSets

rm -rf ./artifacts/subset_big && mkdir -p ./artifacts/subset_big
find /mnt/ssd/datas/discrete_train_with_mle -maxdepth 1 -name '*.csv' | head -50000 \
  | xargs -I{} ln -sf {} ./artifacts/subset_big/

tmux new-session -d -s gachi
tmux send-keys -t gachi "cd /home/user/Documents/python/CTMCxDeepSets" Enter
tmux send-keys -t gachi "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
tmux send-keys -t gachi "conda run -n ctmc --no-capture-output python scripts/train_entrypoint.py --data-dir ./artifacts/subset_big --n 40000 --out-dir ./artifacts/runs --run-name gachi --head gaussian --sqrt-k-scaling --logk-scale 5.0 --epochs 300 --batch-size 64 --lr 1e-3 --patience 20 --num-workers 16 --seed 42 --device cuda && conda run -n ctmc --no-capture-output python scripts/evaluate_npe.py --run-dir ./artifacts/runs/gachi --num-sbc 1000 --num-draws 2000 --num-exact 30 --k-min 200 --k-max 5000 --train-k-min 500 --train-k-max 5000 --device cuda" Enter
tmux attach -t gachi
```

- `--head gaussian --sqrt-k-scaling` が検証済み構成。`--no-sqrt-k-scaling --logk-scale 1.0` で対照(ablation)。
- epoch毎に train/val/best がライブ表示(`val` が +台→負へ降りるのを確認)。
- 合格ライン: 複製比≈0.71 / 傾き≈−0.5 / 被覆≈0.5・0.9 / μ-RMSE≲0.15。
- データロードに数分(1万〜5万CSVのparse)かかってから epoch 表示開始。

---

## 8. 残タスク(Phase B, 任意)

- **訓練K帯を log-uniform [100,5000] に拡張**(低K較正＋OOD汚染除去。NPEはMLE不要で生成は安価)。
  拡張後は評価も帯内[100,5000]で測れOOD汚染が構造的に消える。検証主役は複製テスト(ほぼ自明)
  からK層別SBC・厳密事後とのSD比較へ移る。
- **flowヘッド(非ガウス事後対応)**: μウォームアップ or NSF `bound` 拡大。
- 実データ評価(設計書§4.3): 事後平均/中央値 vs v2点推定・MLE、事後SD vs MLE漸近SD。

---

## 環境メモ

- conda env **`ctmc`**(torch 2.5.1 + zuko 1.6.0)。flowヘッドは zuko 必須、`--device cuda`(小文字)。
- データは外付けSSD **`/mnt/ssd/datas/`**(リポジトリ内 `datas/` は空)。本訓練set =
  `discrete_train_with_mle`(199,936件, K∈[500,5000], 2N[真Q/MLE Q'/サンプル]スキーマ)。
- GPU共有のため batch≤64 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
