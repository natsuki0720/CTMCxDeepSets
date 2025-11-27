import os
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# ---- Loss utilities for CTMC ----
def build_Q(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B,3)  -> (q12, q23, q34)
    return: Q (B,4,4)
    """
    B = q.shape[0]
    Q = torch.zeros(B, 4, 4, device=q.device)
    # forward transitions only
    Q[:, 0, 1] = q[:, 0]
    Q[:, 1, 2] = q[:, 1]
    Q[:, 2, 3] = q[:, 2]
    # diagonal = -sum outgoing
    Q[:, 0, 0] = -q[:, 0]
    Q[:, 1, 1] = -q[:, 1]
    Q[:, 2, 2] = -q[:, 2]
    # state 4 absorbing (0 on row)
    return Q

def compute_logP_ij(Q, i_idx, j_idx, dt):
    """
    Q: (B,4,4)
    i_idx, j_idx: (B,L)
    dt: (B,L)
    return log P_ij(t) -> (B,L)
    """
    eps = 1e-12
    B, L = dt.shape

    # (B,L,4,4)
    Qt = Q.unsqueeze(1) * dt.view(B, L, 1, 1)
    P = torch.matrix_exp(Qt)  # autodiff OK

    # index gather
    i = i_idx.long()
    j = j_idx.long()
    P_ij = P[torch.arange(B).unsqueeze(1), i, j]  # (B,L)

    return torch.log(P_ij.clamp_min(eps))

# ============================================================
# Device helper (unchanged API)
# ============================================================
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# Dataset class (same name / same API)
# ============================================================
class varSets_Datasets(Dataset):
    """
    states : List[np.ndarray] or Tensor, each of shape (2, L_i)
            1行目: s_be (before)
            2行目: s_af (after)
    del_t  : List[np.ndarray] or Tensor, each of shape (L_i,)
    outputs: List[np.ndarray] or Tensor, each of shape (3,)  (q12, q23, q34)
    """

    def __init__(self, states, del_t, outputs, transform=None):
        self.states = states
        self.del_t = del_t
        self.outputs = outputs
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        delta_t = self.del_t[idx]
        target = self.outputs[idx]

        if self.transform:
            state, delta_t = self.transform(state, delta_t)

        state = torch.as_tensor(state, dtype=torch.long)      # (2, L)
        delta_t = torch.as_tensor(delta_t, dtype=torch.float32)  # (L,)
        target = torch.as_tensor(target, dtype=torch.float32) # (3,)
        length = torch.tensor(state.shape[1], dtype=torch.long)
        return state, delta_t, target, length


# ============================================================
# collate_fn (same name / same API)
# ============================================================
def collate_fn(batch):
    """
    Input : list of (state(2,L_i), delta_t(L_i), target(3,), length)
    Output:
        state_padded : (B, 2, L_max)
        delta_t_padded : (B, L_max)
        target_batch : (B, 3)
        lengths : (B,)
    """
    state_batch = [item[0] for item in batch]
    delta_t_batch = [item[1] for item in batch]
    target_batch = torch.stack([item[2] for item in batch])
    lengths = torch.tensor([s.shape[1] for s in state_batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0

    # ---- pad states ----
    state_padded = []
    for s in state_batch:
        L = s.shape[1] if s.dim() == 2 else 0
        if s.dim() != 2:
            s = torch.zeros((2, 0), dtype=torch.long)
            L = 0
        pad_size = max(0, max_length - L)
        if pad_size > 0:
            s = F.pad(s, (0, pad_size), mode="constant", value=0)
        state_padded.append(s)
    state_padded = torch.stack(state_padded, dim=0)  # (B, 2, L_max)

    # ---- pad delta_t ----
    delta_t_padded = []
    for dt in delta_t_batch:
        L = dt.shape[0] if dt.dim() == 1 else 0
        if dt.dim() != 1:
            dt = torch.zeros((0,), dtype=torch.float32)
            L = 0
        pad_size = max(0, max_length - L)
        if pad_size > 0:
            dt = F.pad(dt, (0, pad_size), mode="constant", value=0.0)
        delta_t_padded.append(dt)
    delta_t_padded = torch.stack(delta_t_padded, dim=0)  # (B, L_max)

    return state_padded, delta_t_padded, target_batch, lengths


# ============================================================
# DeepSets encoder + MDN (Neural Posterior Estimation)
#   - class 名・コンストラクタ引数・forward のシグネチャは維持
#   - forward: 事後平均 E[q|X] を返す (B,3)
#   - 追加メソッドで事後分布にアクセスできる
# ============================================================
class DeepSets_varSets_forDiagnel(nn.Module):
    """
    4状態CTMC (1→2→3→4) の (i,j,t) セット X から
    q = (q12, q23, q34) の事後分布 p(q | X) を
    Mixture Density Network (MDN) で近似するモデル。

    - forward(...) は事後平均 E[q|X] を返す ( shape: (B,3) )
    - posterior_mixture(...) で混合ガウスのパラメータを返す
    - log_prob(q_true, ...) で log p(q_true | X) を返す
    - sample_posterior(...) で事後サンプルを生成できる
    """

    def __init__(
        self,
        num_categories: int = 4,
        embedding_dim: int = 16,
        token_hidden1: int = 256,
        token_hidden2: int = 512,   # aggregation width
        output_hidden1: int = 128,
        output_hidden2: int = 64,
        dropout: float = 0.2,
        input_is_one_based: bool = True,
        device: Optional[torch.device] = None,
        mdn_components: int = 1,    # mixture 個数 K
    ):
        super().__init__()
        self.device = device if device is not None else set_device()
        self.input_is_one_based = input_is_one_based
        self.mdn_components = mdn_components
        self.output_dim = 3  # q12, q23, q34

        # ---------- Embedding ----------
        # index 0 は PAD 用。1..num_categories が実データ。
        self.embedding = nn.Embedding(
            num_embeddings=num_categories + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # 各トークン: [embed(pre), embed(post), delta_t] -> dim = 2*E + 1
        in_feat = embedding_dim * 2 + 1

        # ---------- Per-token MLP ----------
        self.fc1 = nn.Linear(in_feat, token_hidden1)
        self.ln1 = nn.LayerNorm(token_hidden1)
        self.fc2 = nn.Linear(token_hidden1, token_hidden2)
        self.ln2 = nn.LayerNorm(token_hidden2)
        self.drop = nn.Dropout(dropout)

        # ---------- Attention pooling ----------
        self.att_fc = nn.Linear(token_hidden2, token_hidden2)
        self.att_score = nn.Linear(token_hidden2, 1)

        # ---------- Output MLP (encoder → context h) ----------
        self.out_fc1 = nn.Linear(token_hidden2, output_hidden1)
        self.out_ln1 = nn.LayerNorm(output_hidden1)
        self.out_fc2 = nn.Linear(output_hidden1, output_hidden2)
        self.out_ln2 = nn.LayerNorm(output_hidden2)

        # ---------- MDN head on log-q ----------
        # context → mixture weights π_k, means μ_{k,d}, log_std_{k,d}
        K = mdn_components
        D = self.output_dim
        self.mdn_pi = nn.Linear(output_hidden2, K)        # (B,K)
        self.mdn_mu = nn.Linear(output_hidden2, K * D)    # (B,K*D)
        self.mdn_log_sigma = nn.Linear(output_hidden2, K * D)  # (B,K*D)

        self.to(self.device)

    # ---------------- Utility ----------------
    def _prepare_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B,L)  1-based or 0-based を内部表現に合わせる"""
        if self.input_is_one_based:
            # 0以下は PAD (=0)、1以上はそのまま
            return torch.where(idx > 0, idx, torch.zeros_like(idx))
        else:
            # データが 0-based の場合は +1 して 0 を PAD とする
            return torch.where(idx >= 0, idx + 1, torch.zeros_like(idx))

    # ---------------- Encoder 部分（DeepSets） ----------------
    def _encode(self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        state:   (B, 2, L)
        delta_t: (B, L)
        lengths: (B,)
        return: context h (B, output_hidden2)
        """
        device = state.device
        B, _, L = state.shape

        # pre/post split
        pre = state[:, 0, :]  # (B, L)
        post = state[:, 1, :] # (B, L)

        pre_idx = self._prepare_indices(pre)
        post_idx = self._prepare_indices(post)

        pre_emb = self.embedding(pre_idx)   # (B, L, E)
        post_emb = self.embedding(post_idx) # (B, L, E)
        dt = delta_t.unsqueeze(-1)          # (B, L, 1)

        # token features
        x = torch.cat([pre_emb, post_emb, dt], dim=-1)  # (B, L, 2E+1)

        # per-token MLP
        x = self.drop(F.gelu(self.ln1(self.fc1(x))))    # (B,L,H1)
        x = self.drop(F.gelu(self.ln2(self.fc2(x))))    # (B,L,H2)

        # ===== ここから pooling 部分を修正 =====
        arange_L = torch.arange(L, device=device).unsqueeze(0)  # (1,L)
        key_padding_mask = arange_L >= lengths.unsqueeze(1)     # (B,L) True=PAD

        attn_input = torch.tanh(self.att_fc(x))                 # (B,L,H2)
        raw_score = self.att_score(attn_input).squeeze(-1)      # (B,L)

        # PAD 部分はスコアを非常に小さくしてから softplus → ほぼ0にする
        raw_score = raw_score.masked_fill(key_padding_mask, -1e9)

        # 正規化しない正の重み（情報を「足す」方向）
        attn_weight = F.softplus(raw_score)                     # (B,L)
        attn_weight = attn_weight.masked_fill(key_padding_mask, 0.0)

        # sum pooling（平均ではなく和）
        pooled = torch.sum(x * attn_weight.unsqueeze(-1), dim=1)  # (B,H2)

        # context h
        h = self.drop(F.gelu(self.out_ln1(self.out_fc1(pooled)))) # (B,Hout1)
        h = self.drop(F.gelu(self.out_ln2(self.out_fc2(h))))      # (B,Hout2)
        return h  # (B, output_hidden2)

    # ---------------- MDN: p(log q | X) ----------------
    def posterior_mixture(
        self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns mixture parameters of p(log q | X).

        Returns:
            pi        : (B, K)          ... softmax された混合係数
            mu        : (B, K, D)       ... 各成分の平均 (log q の平均)
            log_sigma : (B, K, D)       ... 各成分の log 標準偏差
        """
        h = self._encode(state, delta_t, lengths)       # (B,H)

        K = self.mdn_components
        D = self.output_dim

        # mixture weights
        pi_logits = self.mdn_pi(h)                      # (B,K)
        pi = F.softmax(pi_logits, dim=-1)               # (B,K)

        # means and log-stds
        mu = self.mdn_mu(h).view(-1, K, D)              # (B,K,D)
        log_sigma = self.mdn_log_sigma(h).view(-1, K, D)# (B,K,D)

        return pi, mu, log_sigma

    # ---------------- 事後平均 E[q | X] を返す (既存API用) ----------------
    def forward(self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        既存コードとの互換のための API。
        返り値: 事後平均 E[q | X] (B,3), すべて > 0
        """
        pi, mu, log_sigma = self.posterior_mixture(state, delta_t, lengths)
        sigma2 = torch.exp(2.0 * log_sigma)  # (B,K,D), variance in log-space

        # log-normal の期待値: E[q] = exp(μ + 0.5 σ^2)
        # mixture の期待値: sum_k π_k E_k[q]
        # shape 調整のために dim を揃える
        Ek = torch.exp(mu + 0.5 * sigma2)   # (B,K,D) -> E[q | comp k]
        pi_expanded = pi.unsqueeze(-1)      # (B,K,1)
        mean_q = torch.sum(pi_expanded * Ek, dim=1)  # (B,D)

        return mean_q  # positive


    def log_prob(
        self,
        state: torch.Tensor,    # (B,2,L)
        delta_t: torch.Tensor,  # (B,L)
        lengths: torch.Tensor,  # (B,)
        q_true: torch.Tensor,   # (B,3)
    ) -> torch.Tensor:

        device = self.device
        state = state.to(device)
        delta_t = delta_t.to(device)
        lengths = lengths.to(device)
        q_true = q_true.to(device)

        eps = 1e-12

        # posterior parameters: mixture over log q
        pi, mu, log_sigma = self.posterior_mixture(state, delta_t, lengths)
        B, K, D = mu.shape

        # log-space of q_true
        log_q = torch.log(q_true.clamp_min(eps))  # (B,3)
        log_q_expand = log_q.unsqueeze(1)         # (B,1,3)

        # Gaussian log-pdf in log-space
        sigma2 = torch.exp(2.0 * log_sigma)
        log_norm = -0.5 * (torch.log(2 * torch.pi * sigma2))
        log_exp = -0.5 * ((log_q_expand - mu) ** 2 / sigma2)
        log_pdf_comp = (log_norm + log_exp).sum(dim=-1)  # (B,K)

        # mixture
        log_pi = torch.log(pi.clamp_min(eps))
        log_p_logq = torch.logsumexp(log_pi + log_pdf_comp, dim=-1)  # (B,)

        # 変数変換: p(q) = p(log q)/∏q
        log_p_q_prior = log_p_logq - log_q.sum(dim=-1)               # (B,)

        # ---- ここから CTMC の log-likelihood p(X|q_true) ----
        B, _, L = state.shape
        pre = state[:,0,:]  # (B,L)
        post = state[:,1,:]
        mask = torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B,L)

        Q = build_Q(q_true)  # (B,4,4)
        logP = compute_logP_ij(Q, pre, post, delta_t)  # (B,L)
        logP = logP * mask  # PAD無効化
        log_p_X_given_q = logP.sum(dim=-1)  # (B,)

        # ---- Bayes rule: log p(q|X) = log p(q) + log p(X|q) up to constant ----
        return log_p_X_given_q + log_p_q_prior  # (B,)
