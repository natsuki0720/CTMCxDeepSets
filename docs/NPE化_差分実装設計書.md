# NPE化 差分実装設計書

**対象リポジトリ:** `natsuki0720/CTMCxDeepSets_publish`
**目的:** 点推定サロゲート(DeepSets回帰)を、サンプルサイズ依存の事後収縮を表現できる償却ベイズ推定器(Neural Posterior Estimation, NPE)へ拡張する。

---

## 0. 方針サマリ

| 項目 | 現行 (v2) | NPE版 |
|---|---|---|
| 出力 | 点推定 q̃ (Softplus, 3次元) | 条件付き事後分布 q_φ(z \| X) |
| パラメータ空間 | q (遷移率) | z = log ν = log(1/q)(対数寿命) |
| 教師ラベル | MLE q̂ₛ(15万回のBFGS) | **生成パラメータ zₛ(MLE計算不要)** |
| 損失 | 逆数MAE(=1-Wasserstein) | NLL: −log q_φ(zₛ \| Xₛ) |
| エンコーダ | DeepSets + attention pooling | **そのまま流用** |
| サンプルサイズ情報 | 未使用(maskのみ) | **log K をheadに注入(必須)** |

理論的根拠: 対数尤度は ln L = K·⟨ln p_ij(t)⟩_p̂ と書けるため、事後分布は(経験分布 p̂, K)のペアの厳密な関数。平均型poolingは ⟨φ⟩_p̂ を計算しているので、**pooled表現 + log K で事後分布の十分統計量を完全に保持できる**。NLLは厳密に適切なスコアリングルールであり、最小解は真の事後分布(幅・収縮率を含む)。

---

## 1. データ層の差分

### 1.1 既存生成データの流用(再生成不要)

`dataset_csv_loader.parse_ctmc_csv` のCSVスキーマは
`[真のQ (N行)] → [MLE Q' (N行)] → [サンプル (i, j, Δt)]`
であり、**真のQが既に保存されている**。教師を q_mle ブロックから q ブロックに切り替えるだけで、生成済みの20万データセットがそのまま使える。

```python
# 変更箇所: dataset構築時のターゲット選択
parsed = parse_ctmc_csv(path)
target = extract_lambdas_from_Q(parsed.q)        # 現行: parsed.q_mle
z = torch.log(1.0 / torch.tensor(target))        # z = log ν 空間へ
```

### 1.2 新規生成する場合

`DatasetGenerationConfig(enable_mle=False)` で生成。MLEラベリングが
消えるため、**訓練データ生成コストはv2比で桁違いに軽くなる**
(README で警告している長時間処理の大半はBFGS分)。

### 1.3 スクリーニング (`dataset_screening.py`)

現行のスクリーニングは q_mle に対するNaN/構造/λ範囲チェック。NPEでは
教師が生成パラメータ(構成上常にvalid、ν∈(1,100)保証)なので、
**MLE起因の除外が不要になる**。副次効果として、論文v2で指摘されうる
「非収束ケース除外によるselection bias」問題が構造的に消える。
残すべきチェックはサンプル列の整合性のみ。

---

## 2. モデルの差分 (`models/deepsets_regressor.py`)

### 2.1 log K の注入(必須・実質1行)

attention pooling(softmax正規化=加重平均)はデータセット複製に不変で、
Kの情報を落とす。headの入力に log K を連結する。

```python
# forward() 末尾、pooled 計算後
logk = torch.log(lengths.float().clamp(min=1.0)).unsqueeze(-1) / 5.0  # 緩いスケーリング
h = torch.cat([pooled, logk], dim=-1)
# out_fc1 の入力次元を token_hidden2 + 1 に変更
```

### 2.2 出力ヘッドの差し替え

`out_fc3 + Softplus` を分布パラメータ出力に置換。2段階で実装する。

**Phase 1: ガウスヘッド(最小実装、約50行)**

```python
# 出力9次元: mean(3) + log_diag(3) + off_diag(3) → Cholesky L
self.out_fc3 = nn.Linear(output_hidden2, 9)

def forward(...) -> tuple[Tensor, Tensor, Tensor]:
    out = self.out_fc3(h)
    mu, log_d, off = out[:, :3], out[:, 3:6].clamp(-7, 3), out[:, 6:9]
    return mu, log_d, off

def nll(self, mu, log_d, off, z):
    L = self._build_cholesky(log_d, off)           # 下三角
    y = torch.linalg.solve_triangular(L, (z - mu).unsqueeze(-1), upper=False)
    return 0.5 * ((y**2).sum((1,2)) + 2*log_d.sum(1) + 3*math.log(2*math.pi))
```

**Phase 2: flowヘッド(本実装)**

事前分布の切断境界(ν≈1, 100)付近・小K領域の非ガウス形状に対応。
`zuko` を推奨(条件付きNSFが数行)。

```python
import zuko
self.flow = zuko.flows.NSF(features=3, context=token_hidden2 + 1,
                           transforms=3, hidden_features=[64, 64])
# loss: -self.flow(h).log_prob(z)
# sampling: self.flow(h).sample((n,))
```

境界対策の代替/併用案: z を logit((z − ln1)/(ln100 − ln1)) で
非有界空間に写してから当てる(Phase 1のガウスでも歪みが軽減)。

### 2.3 既存の点推定モデルとの共存

`build_model()` に `head: "point" | "gaussian" | "flow"` を追加し、
v2モデルを残したままA/B比較可能にする(論文の比較表に必要)。

---

## 3. 訓練ループの差分 (`train/train_loop.py`)

`fit()` は `loss_fn` 注入可能な設計のため変更は局所的。

```python
class NPELoss(nn.Module):
    def forward(self, pred, target):           # pred = (mu, log_d, off) or flow dist
        return model_nll(pred, target).mean()

# _run_epoch 内
pred = model(state, delta_t, lengths)
loss = loss_fn(pred, target)                   # target は z = log ν
```

注意点:
- ターゲットのCSV→z変換は Dataset 側で実施(§1.1)
- 早期終了・チェックポイント機構は無変更で流用
- 学習率等は現行設定から開始可(NLLはW1よりスケールが違うだけ)

---

## 4. 評価モジュール(新規 `eval/`)

### 4.1 `exact_posterior.py` — 厳密事後分布(較正の正解)

尤度は (i, j, Δt) のユニークセル(≤ 3×10通り強)に集計すれば
K非依存で安価に厳密計算できる。`probability.transition_row` を
パラメータ方向にベクトル化して流用。

手順: 集計セル化 → MAP(Nelder-Mead/BFGS, 多点初期値) →
数値Hessianで Laplace近似 → 多変量t提案の自己正規化重点サンプリング
(ESS監視) → 事後モーメント・分位点。

### 4.2 `sbc.py` — 較正検証

1. **SBC**: 新規生成 (z*, X) ペア(~1,000本)に対し、z* のNPE事後CDF値
   (ランク統計)の一様性をKS検定 + ヒストグラムで確認。**Kで層別**して
   K帯域ごとの較正崩れを検出。
2. **複製テスト**(最重要の単体テスト): 同一データ X と X⊕X(全サンプル
   2重化)を入力し、事後SDが 1/√2 倍に収縮することを確認。
   pooling不変性に対する log K 注入の効果を直接反証可能な形で検証。
3. **収縮プロット**: K ∈ {200,...,5000} で事後SD vs K を両対数プロット、
   傾き −1/2(BvM則)への整合を確認。論文Fig.17–19のNPE版に相当。
4. **カバレッジ**: 50%/90%信用区間の被覆率。

### 4.3 実データ評価(論文 §4.2 の再現+拡張)

- 事後平均/中央値が v2 点推定・MLEと整合するか(Table 3–8 の再現)
- 事後SD vs MLE漸近SD(√Var(q̂))の比較 → 現行の
  Var比評価の上位互換になる
- 誤特定下の防御として §5 のIS補正を併用した結果も併記

---

## 5. 推論ユーティリティ(新規・任意 `inference/`)

尤度が厳密計算できる利点を活かす2機能。

```python
def is_correct(npe_samples, cells, prior_logpdf, npe_logpdf):
    """NPE事後を提案分布とした自己正規化IS → 漸近的に厳密な事後へ補正"""
    lw = exact_loglik(npe_samples, cells) + prior_logpdf(npe_samples) \
         - npe_logpdf(npe_samples)
    ...
```

- **IS補正**: NPE近似誤差・誤特定への防御線
- **事前分布の差し替え**: 学習時事前 U(1,100) と異なる事前を使いたい
  ユーザー向けに、尤度比による再重み付けで再学習なしに対応

---

## 6. 工数・計算コスト見積もり

| 作業 | 規模 | 備考 |
|---|---|---|
| データ層差分 (§1) | ~30行 | 既存CSV流用なら生成ゼロ |
| ガウスヘッド (§2) | ~60行 | log K注入含む |
| flowヘッド (§2) | ~30行 + zuko依存 | environment.yml に追記 |
| 損失・ループ (§3) | ~30行 | |
| exact_posterior (§4.1) | ~150行 | プロトタイプ済み |
| sbc + プロット (§4.2) | ~150行 | プロトタイプ済み |
| IS補正 (§5) | ~50行 | |

訓練コスト: v2と同等以下(MLEラベル不要、NLL計算はW1と同オーダー)。
評価コスト: 厳密事後は集計尤度のため1データセット数秒〜数十秒。

---

## 7. リスクと対策

| リスク | 対策 |
|---|---|
| 境界(ν≈1,100)・小Kでの非ガウス事後 | flowヘッド or logit変換(§2.2) |
| 実データ=誤特定下での過信 | 合成SBCで較正提示 + IS補正併用(§5) |
| K>5000への外挿 | log K入力なので緩やかだが、訓練K範囲の明示と範囲外警告を実装 |
| Gaussianヘッドの相関表現不足 | full Cholesky採用済み(対角でない)で大半は吸収 |

---

## 8. 付随する改善(NPEと独立だが推奨)

`mle_diagonal_exp.LikelihoodDiagonalExp.log_likelihood` は全Kサンプルで
毎回 `expm` を呼ぶ実装。ユニーク (i, j, Δt) セルへの集計 +
`probability.transition_row`(解析形)への置換で、ベースラインMLE自体が
桁で高速化する。論文の速度比較(282.7s)の頑健性確保のため、
**最適化済みMLEとの比較**を併記できる状態にしておくのが安全。

---

## 9. 実装順序(推奨)

1. §1 + §2(ガウス) + §3 → 小規模(K=100–800, 数千セット)で学習が回ることを確認
2. §4.2 複製テスト → log K注入が効いているかを最初に検証
3. §4.1 厳密事後との突き合わせ → モーメント精度の確認
4. SBC層別 → 較正の全体像
5. flowヘッド差し替え → 境界・小K領域の改善を確認
6. フルスケール訓練(既存20万CSV流用) → 実データ評価(§4.3)

---

## 10. 改訂: 1/√K 収縮の構造化(addendum, 2026-06-11)

### 10.1 動機 — log K 注入の単体検証が失敗

§2.1 の「pooled + log K を head に連結して幅を学習させる」案を実装し、
flowヘッドで小規模学習(既存 `discrete_train_with_mle`, K∈[500,5000], 8千セット)
した結果、**複製テスト比が 1.018(目標 1/√2≈0.707)、収縮傾きが ≈0(目標 −0.5)**
となった。SBC(KS≈0.05)・被覆率(0.49/0.90)は良好だが、これは**評価データが
訓練と同じK分布から引かれ、周辺的な事後幅が平均的に当たっているだけ**で、
K層別SBCでは低K帯(200–500)が崩れる(KS 0.28–0.41)。

→ NPE化の核心である「サンプルサイズ依存の事後収縮」が実証できていない。
モデルは log K 入力を実質無視している。

### 10.2 なぜスケール調整(`logk_scale`)では不足か

`logk_scale` を下げても「512次元 pooled に1次元スカラーが混ざる」構造は不変。
K で幅を √10 レンジ調整する NLL 利得は1次元あたり高々 0.1–0.3 nat と勾配信号が弱く、
**学習させようとしている関係が、実は学習する必要のない厳密な構造**である点が本質。

対数尤度は `ln L = K·⟨ln p⟩_p̂` ゆえ対数事後の Hessian は厳密に `K ×(経験平均項の曲率)`。
p̂ を固定したとき事後幅が 1/√K でスケールすることは漸近近似ですらなく**恒等式**
(崩れるのは事前分布項と非ガウス性のみ。K≥500 では無視できる)。
**学習させるのではなく、構造として焼き込む。**

### 10.3 構造化 √K スケーリング(本改訂の実装)

flowヘッドに「K=1相当の標準化残差 u」を学習させ、決定論的アフィンで幅を縮める:

```
z = μ(pooled) + u / √K,        u ~ flow(context = [pooled, logK])
```

- **μ は pooled のみ(logK を入れない)**。pooled = attention加重平均はデータ複製に
  厳密不変ゆえ、μ も複製不変 → 複製テストは SD だけでなく**平均も構成的に通る**。
  事前プルによる O(1/K) 平均バイアスは、logK を u 側 context に残すことで補正項として吸収。
- **logK は flow の context に残す**。1/√K の主項は構造で保証、残差(小Kの非ガウス性・
  事前の影響)だけを学習。よって複製テストは「構成的にほぼ通る」=学習の成否に非依存。

#### 実装(`models/deepsets_regressor.py`)

`z = μ + u/√K` を **torch `TransformedDistribution` + `AffineTransform(loc=μ, scale=K^{-1/2})`**
で表現:

```python
u_dist = self.flow(h)                                   # h = [pooled, logK/scale] 経由
mu     = self.mu_head(pooled)                           # K不変
scale  = torch.exp(-0.5 * torch.log(lengths.float()))   # K^{-1/2}, 真のK(logk_scale非依存)
affine = torch.distributions.AffineTransform(mu, scale.unsqueeze(-1), event_dim=1)
return torch.distributions.TransformedDistribution(u_dist, affine)
```

- ヤコビアン `+(D/2)·logK` は `log_prob` が自動計上 → `NPELoss = -log_prob(z)` は不変。
- `.sample()` が `μ + u/√K` を返す → `posterior_moments/sample`・SBC・predictor は**無改修**。
- ガウスヘッドも同様: `MultivariateNormal(μ_head(pooled), scale_tril=K^{-1/2}·L)`(閉形式モーメント維持)。
- **`scale` は真の K^{-1/2}**。`logk_scale` は context 入力の正規化専用で、構造スケーリングには使わない。

#### フラグと後方互換

- `sqrt_k_scaling: bool`(model_config / `--sqrt-k-scaling`)で切替。
- `mu_head` は `sqrt_k_scaling=True` のときのみ生成 → 旧checkpoint(キー無し=False)は従来どおりロード可。
- **対照アーム** `--no-sqrt-k-scaling --logk-scale 1.0` は現行アーキそのもの(学習依存)。
  両者の複製テスト比を並べれば「学習で獲得できるか vs 構造で保証すべきか」が論文の ablation になる。

### 10.4 評価の汚染除去(`scripts/evaluate_npe.py`)

§4.2 の低K帯SBC崩れ・収縮傾きの K=200 点は**訓練帯(500–5000)の外挿**を含み、
「収縮を学んでいない」証拠と「分布外で壊れている」証拠が混ざっていた。

- **収縮傾きは訓練帯内(`--train-k-min`/`--train-k-max`, 既定 500–5000)に限定**して測る。
- SBC は全帯域で生成しつつ、**帯内subsetのKSを別掲**し、層別に in-band フラグを付す。
- 構造版では帯内傾き ≈ −0.5・複製比 ≈ 0.707 が**構成的に**出るはず。検証の主役は
  複製テスト(ほぼ自明)から **K層別SBC・厳密事後とのSD比較**へ移る。

### 10.5 併せて(Phase B, 本改訂の対象外)

- **訓練K帯の拡張**: 現状 500–5000 は1桁。**log-uniform K∈[100,5000]** に広げると
  信号が強まり低K帯SBC崩れにも対処(NPEは MLE不要ゆえ新規生成は安価, §1.2)。
- 帯拡張後は評価も帯内[100,5000]で測れ、OOD汚染が構造的に消える。

### 10.6 実装で詰めた点と検証結果(2026-06-12 確定)

構造化 §10.3 を素朴に実装すると **較正が崩れる**。原因は一点 ——
`u = √K·(z − μ(pooled))` は μ が不正確だと O(√K) に膨らみ、

- **flowヘッド**: NSF の有界スプライン域(±5程度)に密度を置けず NLL 爆発(val +560で停滞)。
- **ガウスヘッド**: 共分散が logK で高K側を膨らませ残差を吸収 → K^{-1/2} 収縮を相殺し
  **傾きが −0.5→−0.33、複製比が 0.707→0.80 に劣化**。

症状は違うが根は同じ「**μ(pooled)→z の回帰精度**」。これを詰める3点を実装:

1. **μ は detach + 補助MSE で学習**(`NPELoss` に `mu_weight·‖μ−z‖²`)。
   NLL の μ 勾配は精度 ∝ K で病的。μ は点推定と同じ良条件の回帰で学ぶ。
2. **エンコーダを scale/shape NLL から隔離**(`_encode` で構造時は head へ `pooled.detach()`)。
   爆発的/不安定な NLL 勾配が共有エンコーダを汚さないようにし、エンコーダは MSE 専従に。
3. **`mu_head` は単層 Linear ではなく MLP**(512→128→64→3)。これが最重要だった。
   純MSE回帰の下限を実測すると **Linear: RMSE 0.34 で頭打ち / MLP: 0.14 まで低下**。
   線形読み出しでは pooled から z を抽出しきれない(エンコーダが万能でも実最適化では不足)。

**検証結果(既存 `discrete_train_with_mle` 8000セット, K∈[500,5000], gaussianヘッド):**

| 指標 | 構造+MLP | 対照(学習) | 目標 |
|---|---|---|---|
| 複製比 | **0.713** | 0.973 | 0.707 |
| 収縮傾き(帯内) | **−0.48/−0.46/−0.50** | ≈0 | −0.5 |
| 被覆 50/90 | **0.54/0.91** | 0.51/0.89 | 0.5/0.9 |
| SBC KS(帯内) | **0.11/0.06/0.07** | 0.11/0.11/0.19 | 小 |
| 厳密事後 平均z誤差 | **0.055** | 0.082 | 小 |
| μ-RMSE | **0.11** | — | ~0.08 |

→ **ガウス構造ヘッドはK収縮の構造保証と較正を同時達成**(設計書 Phase 1 完了)。
単体テスト: h固定で K→2K の SD比 = 1/√2(誤差 1e-7)、複製比 ≈0.71・傾き −0.5 が
**未学習でも構成的に**成立。

**flowヘッド(Phase 2)**: MLP化で μ は改善するが、NSF の有界域のため残差が
端に乗る。μウォームアップ or NSF の `bound` 拡大が要る。当面の本実行はガウスヘッド推奨。
