"""DeepSets-based surrogate for state-transition sequence inputs.

Two output families share a single DeepSets + attention-pooling encoder:

* ``head="point"`` — the original v2 surrogate. Emits a positive raw parameter
  vector (Softplus). Backward compatible: the architecture and weight layout are
  unchanged, so existing checkpoints load as-is.
* ``head="gaussian"`` / ``head="flow"`` — Neural Posterior Estimation (NPE)
  heads that emit a *conditional posterior* ``q_phi(z | X)`` over the
  log-lifetime ``z = log nu``. ``forward`` returns a ``torch.distributions``-style
  object exposing ``log_prob`` / ``sample``.

Sample-size injection (NPE design note, section 2.1): attention pooling is a
softmax-weighted average and is therefore invariant to dataset replication,
which discards the sample count ``K``. The posterior width must shrink with
``K``, so the NPE heads concatenate ``log K`` to the pooled representation. The
point head leaves the encoder untouched to preserve checkpoint compatibility.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

VALID_HEADS = ("point", "gaussian", "flow")


def gaussian_param_dim(output_dim: int) -> int:
    """Number of head outputs for a full-Cholesky Gaussian of dimension D.

    ``mean (D) + log_diag (D) + strictly-lower off-diagonal (D(D-1)/2)``.
    """
    d = int(output_dim)
    return d + d + d * (d - 1) // 2


def build_scale_tril(log_diag: Tensor, off_diag: Tensor) -> Tensor:
    """Assemble a batched lower-triangular Cholesky factor ``L``.

    Args:
        log_diag: ``(B, D)`` log of the positive diagonal entries.
        off_diag: ``(B, D(D-1)/2)`` strictly-lower-triangular entries in
            row-major order (matching :func:`torch.tril_indices`).

    Returns:
        ``(B, D, D)`` lower-triangular matrix with positive diagonal.
    """
    bsz, d = log_diag.shape
    scale_tril = torch.zeros(bsz, d, d, dtype=log_diag.dtype, device=log_diag.device)
    diag_idx = torch.arange(d, device=log_diag.device)
    scale_tril[:, diag_idx, diag_idx] = torch.exp(log_diag)
    if off_diag.shape[1] > 0:
        rows, cols = torch.tril_indices(d, d, offset=-1, device=log_diag.device)
        scale_tril[:, rows, cols] = off_diag
    return scale_tril


class DeepSetsVarSetsAttnRegressor(nn.Module):
    """DeepSets surrogate with selectable point / Gaussian / flow output head."""

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        output_dim: int,
        token_hidden1: int = 256,
        token_hidden2: int = 512,
        output_hidden1: int = 128,
        output_hidden2: int = 64,
        dropout: float = 0.2,
        input_is_one_based: bool = True,
        head: str = "point",
        logk_scale: float = 5.0,
        log_diag_min: float = -7.0,
        log_diag_max: float = 3.0,
        flow_transforms: int = 3,
        flow_hidden: int = 64,
        sqrt_k_scaling: bool = False,
    ) -> None:
        super().__init__()
        head = str(head)
        if head not in VALID_HEADS:
            raise ValueError(f"head must be one of {VALID_HEADS}, got {head!r}.")

        self.head = head
        self.output_dim = int(output_dim)
        self.input_is_one_based = bool(input_is_one_based)
        self.logk_scale = float(logk_scale)
        self.log_diag_min = float(log_diag_min)
        self.log_diag_max = float(log_diag_max)
        # log K is injected only for posterior heads, keeping the point-head
        # weight layout identical to the v2 model.
        self._inject_logk = head != "point"
        # Structural 1/sqrt(K) scaling (design note section 10): the posterior is
        # z = mu(pooled) + u / sqrt(K) with u ~ flow/Gaussian. The 1/sqrt(K) law is
        # an identity (Hessian = K * empirical-curvature), so we bake it in instead
        # of asking the network to learn it from the weak log K signal.
        self.sqrt_k_scaling = bool(sqrt_k_scaling) and (head != "point")

        self.embedding = nn.Embedding(
            num_embeddings=int(num_categories) + 1,
            embedding_dim=int(embedding_dim),
            padding_idx=0,
        )

        token_input_dim = 2 * int(embedding_dim) + 1
        self.fc1 = nn.Linear(token_input_dim, int(token_hidden1))
        self.ln1 = nn.LayerNorm(int(token_hidden1))
        self.drop1 = nn.Dropout(float(dropout))

        self.fc2 = nn.Linear(int(token_hidden1), int(token_hidden2))
        self.ln2 = nn.LayerNorm(int(token_hidden2))
        self.drop2 = nn.Dropout(float(dropout))

        self.att_fc = nn.Linear(int(token_hidden2), int(token_hidden2))
        self.att_score = nn.Linear(int(token_hidden2), 1)

        head_input_dim = int(token_hidden2) + (1 if self._inject_logk else 0)
        self.out_fc1 = nn.Linear(head_input_dim, int(output_hidden1))
        self.out_ln1 = nn.LayerNorm(int(output_hidden1))
        self.out_drop1 = nn.Dropout(float(dropout))

        self.out_fc2 = nn.Linear(int(output_hidden1), int(output_hidden2))
        self.out_ln2 = nn.LayerNorm(int(output_hidden2))
        self.out_drop2 = nn.Dropout(float(dropout))

        if head == "point":
            self.out_fc3 = nn.Linear(int(output_hidden2), int(output_dim))
        elif head == "gaussian":
            self.out_fc3 = nn.Linear(int(output_hidden2), gaussian_param_dim(output_dim))
        else:  # flow
            self.flow = _build_zuko_flow(
                features=int(output_dim),
                context=int(output_hidden2),
                transforms=int(flow_transforms),
                hidden=int(flow_hidden),
            )

        # Mean head for the structural form, fed by the K-invariant pooled vector
        # (no log K) so that the posterior mean is exactly replication-invariant.
        # An MLP (not a single Linear): a linear read-out of `pooled` plateaus at
        # RMSE ~0.34 on z, but mu must reach ~0.1 or the residual u = sqrt(K)(z-mu)
        # stays O(sqrt(K)) and breaks the flow / inflates the Gaussian covariance.
        # An MLP of the same depth as the head reaches ~0.14 and keeps improving.
        # Created only when active, keeping legacy (non-structural) checkpoints loadable.
        if self.sqrt_k_scaling:
            self.mu_head = nn.Sequential(
                nn.Linear(int(token_hidden2), int(output_hidden1)),
                nn.GELU(),
                nn.Linear(int(output_hidden1), int(output_hidden2)),
                nn.GELU(),
                nn.Linear(int(output_hidden2), int(output_dim)),
            )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def _normalize_indices(self, state: Tensor) -> Tensor:
        if self.input_is_one_based:
            return torch.where(state > 0, state, torch.zeros_like(state))
        return torch.where(state >= 0, state + 1, torch.zeros_like(state))

    def _encode(self, state: Tensor, delta_t: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Run the DeepSets encoder.

        Returns ``(h, pooled)`` where ``h`` is the head context ``(B, output_hidden2)``
        (with log K mixed in for posterior heads) and ``pooled`` is the raw
        attention-pooled vector ``(B, token_hidden2)`` *before* log K injection.
        ``pooled`` is exactly invariant to dataset replication (softmax-weighted
        average), so the structural mean head ``mu(pooled)`` is replication-invariant.
        """
        if state.ndim != 3 or state.shape[1] != 2:
            raise ValueError("state must have shape (B, 2, L).")
        if delta_t.ndim != 2:
            raise ValueError("delta_t must have shape (B, L).")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape (B,).")

        device = state.device
        lengths = lengths.to(device)
        if delta_t.device != device:
            delta_t = delta_t.to(device)

        bsz, _, seq_len = state.shape
        if delta_t.shape != (bsz, seq_len):
            raise ValueError("delta_t shape must align with state as (B, L).")
        if lengths.shape[0] != bsz:
            raise ValueError("lengths batch dimension must match state.")
        if torch.any(lengths < 1):
            raise ValueError("Each element of lengths must be at least 1.")
        if int(lengths.max().item()) > seq_len:
            raise ValueError("The maximum value in lengths exceeds state sequence length L.")

        norm_idx = self._normalize_indices(state.long())
        pre_emb = self.embedding(norm_idx[:, 0, :])
        post_emb = self.embedding(norm_idx[:, 1, :])

        dt = delta_t.to(dtype=pre_emb.dtype).unsqueeze(-1)
        x = torch.cat([pre_emb, post_emb, dt], dim=-1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.drop2(x)

        attn_input = torch.tanh(self.att_fc(x))
        score = self.att_score(attn_input).squeeze(-1)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        key_padding_mask = positions >= lengths.unsqueeze(1)
        score = score.masked_fill(key_padding_mask, float("-inf"))
        weight = F.softmax(score, dim=1)

        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)

        # Structural heads: feed the *detached* pooled into the head MLP so the
        # ill-conditioned scale/shape NLL (residual u = sqrt(K)(z - mu) blows up
        # until mu tracks z) cannot corrupt the shared encoder. The encoder is then
        # driven solely by the well-conditioned mu-regression MSE via mu_head(pooled),
        # learning the sufficient statistic; the flow models the residual shape on
        # top of those (frozen-gradient) features. Non-structural path is unchanged.
        head_pooled = pooled.detach() if self.sqrt_k_scaling else pooled

        head_in = head_pooled
        if self._inject_logk:
            logk = torch.log(lengths.to(dtype=pooled.dtype).clamp(min=1.0)).unsqueeze(-1)
            logk = logk / self.logk_scale
            head_in = torch.cat([head_pooled, logk], dim=-1)

        h = self.out_fc1(head_in)
        h = self.out_ln1(h)
        h = F.gelu(h)
        h = self.out_drop1(h)

        h = self.out_fc2(h)
        h = self.out_ln2(h)
        h = F.gelu(h)
        h = self.out_drop2(h)
        return h, pooled

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------
    def _inv_sqrt_k(self, lengths: Tensor, ref: Tensor) -> Tensor:
        """Return ``K^{-1/2}`` as a ``(B, 1)`` tensor on ``ref``'s device/dtype.

        Uses the *true* sample count K (not ``logk_scale``-normalized): the
        1/sqrt(K) contraction is the structural identity, while ``logk_scale``
        only normalizes the log K *input* fed to the encoder.
        """
        logk = torch.log(lengths.to(device=ref.device, dtype=ref.dtype).clamp(min=1.0))
        return torch.exp(-0.5 * logk).unsqueeze(-1)

    def _gaussian_posterior(
        self, h: Tensor, pooled: Tensor, lengths: Tensor
    ) -> torch.distributions.MultivariateNormal:
        out = self.out_fc3(h)
        d = self.output_dim
        log_diag = out[:, d : 2 * d].clamp(self.log_diag_min, self.log_diag_max)
        off_diag = out[:, 2 * d :]
        scale_tril = build_scale_tril(log_diag, off_diag)
        if self.sqrt_k_scaling:
            # z = mu(pooled) + L u / sqrt(K):  mean K-invariant, SD ~ K^{-1/2}.
            # mu is detached in the location and trained by the auxiliary MSE in
            # NPELoss (see design note section 10): the NLL gradient on mu is
            # ill-conditioned (residual scales with sqrt(K)), so mu learns the
            # conditional mean directly while the scale/shape learns via NLL.
            mu = self.mu_head(pooled)
            scale_tril = scale_tril * self._inv_sqrt_k(lengths, scale_tril).unsqueeze(-1)
            dist = torch.distributions.MultivariateNormal(
                loc=mu.detach(), scale_tril=scale_tril, validate_args=False
            )
            dist.mu_structural = mu
            return dist
        return torch.distributions.MultivariateNormal(
            loc=out[:, :d], scale_tril=scale_tril, validate_args=False
        )

    def posterior(self, state: Tensor, delta_t: Tensor, lengths: Tensor):
        """Return the conditional posterior ``q_phi(z | X)`` (Gaussian or flow heads)."""
        if self.head == "point":
            raise RuntimeError("posterior() is only available for the gaussian/flow heads.")
        h, pooled = self._encode(state, delta_t, lengths)
        if self.head == "gaussian":
            return self._gaussian_posterior(h, pooled, lengths)

        flow_dist = self.flow(h)
        if not self.sqrt_k_scaling:
            return flow_dist  # flow models z directly (learned K-dependence)
        # Structural form: flow models the K=1 standardized residual u, then
        # z = mu(pooled) + u / sqrt(K). The affine's Jacobian (+D/2 logK) is
        # accounted for automatically by TransformedDistribution.log_prob.
        mu = self.mu_head(pooled)
        # scale must be (B, D), not (B, 1): AffineTransform.log_abs_det_jacobian sums
        # one factor per event dimension, so a (B, 1) scale would undercount the
        # Jacobian by (D-1)*log(sqrt(K)) and corrupt the flow NLL.
        scale = self._inv_sqrt_k(lengths, mu).expand_as(mu)
        # mu detached in the transform: it is trained by the auxiliary MSE in
        # NPELoss (residual u = sqrt(K)(z - mu) is O(sqrt(K)) until mu tracks z,
        # so the flow NLL cannot condition mu). The flow learns the residual shape.
        affine = torch.distributions.AffineTransform(loc=mu.detach(), scale=scale, event_dim=1)
        dist = torch.distributions.TransformedDistribution(flow_dist, affine)
        dist.mu_structural = mu
        return dist

    def forward(self, state: Tensor, delta_t: Tensor, lengths: Tensor):
        """Forward pass.

        * point head -> positive raw parameter tensor ``(B, output_dim)``.
        * gaussian / flow head -> conditional posterior distribution over ``z``.
        """
        if self.head == "point":
            h, _ = self._encode(state, delta_t, lengths)
            output = F.softplus(self.out_fc3(h))
            if __debug__:
                assert torch.all(output >= 0), "softplus output is negative."
            return output
        return self.posterior(state, delta_t, lengths)


def _build_zuko_flow(features: int, context: int, transforms: int, hidden: int):
    """Lazily construct a conditional Neural Spline Flow via ``zuko``.

    ``zuko`` is only required for the flow head; importing it lazily keeps the
    rest of the package usable when it is not installed.
    """
    try:
        import zuko
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "head='flow' requires the optional dependency 'zuko'. Install it with "
            "`pip install zuko` (also listed in environment.yml)."
        ) from exc

    return zuko.flows.NSF(
        features=int(features),
        context=int(context),
        transforms=int(transforms),
        hidden_features=[int(hidden), int(hidden)],
    )


def build_model(model_config: dict) -> DeepSetsVarSetsAttnRegressor:
    """Build the surrogate model from a config dictionary.

    ``head`` selects the output family and defaults to ``"point"`` so configs
    written for the v2 model continue to build the original regressor.
    """
    required = ["num_categories", "embedding_dim", "output_dim"]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"model_config is missing required keys: {missing}")

    return DeepSetsVarSetsAttnRegressor(
        num_categories=int(model_config["num_categories"]),
        embedding_dim=int(model_config["embedding_dim"]),
        token_hidden1=int(model_config.get("token_hidden1", 256)),
        token_hidden2=int(model_config.get("token_hidden2", 512)),
        output_hidden1=int(model_config.get("output_hidden1", 128)),
        output_hidden2=int(model_config.get("output_hidden2", 64)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_is_one_based=bool(model_config.get("input_is_one_based", True)),
        output_dim=int(model_config["output_dim"]),
        head=str(model_config.get("head", "point")),
        logk_scale=float(model_config.get("logk_scale", 5.0)),
        log_diag_min=float(model_config.get("log_diag_min", -7.0)),
        log_diag_max=float(model_config.get("log_diag_max", 3.0)),
        flow_transforms=int(model_config.get("flow_transforms", 3)),
        flow_hidden=int(model_config.get("flow_hidden", 64)),
        sqrt_k_scaling=bool(model_config.get("sqrt_k_scaling", False)),
    )
