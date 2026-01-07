# model_HFSemiClassifier.py
# Deep Sets + (PyG pooling: mean/sum/max/attn/attn_mean) for HF semi-leptonic e classification
#
# 输入:
#   ele_feat: (B, ele_input_dim)
#   had_feat: (B, N, had_input_dim)
#   had_mask: (B, N) bool (True=valid hadron, False=padding)
#
# 输出:
#   logits: (B, n_classes)
#   (optional) alpha: (B, N) attention weights if return_attn=True and pooling uses attn

import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import softmax, scatter


def build_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, last_activation=False):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    if last_activation:
        layers.append(activation())
    return nn.Sequential(*layers)


class DeepSetsHF(nn.Module):
    """
    DeepSets-like model with PyG-style global pooling.

    pooling options:
      - "mean"      : global_mean_pool
      - "sum"       : global_add_pool
      - "max"       : global_max_pool
      - "attn"      : attention pooling (learned weights per point)
      - "attn_mean" : concat([attn_pool, mean_pool])  -> dim=2*set_embed_dim
    """

    def __init__(
        self,
        had_input_dim: int,
        ele_input_dim: int,
        had_hidden_dims=(256, 256, 256, 256),
        set_embed_dim: int = 256,
        clf_hidden_dims=(256, 256, 256, 256),
        n_classes: int = 2,
        use_ele_in_had_encoder: bool = False,
        pooling: str = "sum",
        attn_hidden_dim: int = 64,
        use_ele_feat: bool = True,
    ):
        super().__init__()

        self.had_input_dim = had_input_dim
        self.ele_input_dim = ele_input_dim
        self.set_embed_dim = set_embed_dim
        self.pooling = pooling
        self.use_ele_in_had_encoder = use_ele_in_had_encoder
        self.use_ele_feat = use_ele_feat

        # per-point encoder φ
        per_had_input_dim = had_input_dim + (ele_input_dim if use_ele_in_had_encoder else 0)
        self.had_encoder = build_mlp(
            input_dim=per_had_input_dim,
            hidden_dims=list(had_hidden_dims),
            output_dim=set_embed_dim,
            activation=nn.ReLU,
            last_activation=True,
        )

        # attention gate (only if needed)
        if pooling in ("attn", "attn_mean"):
            self.attn_gate = nn.Sequential(
                nn.Linear(set_embed_dim, attn_hidden_dim),
                nn.Tanh(),
                nn.Linear(attn_hidden_dim, 1),  # score per point
            )
        else:
            self.attn_gate = None

        # classifier input dim
        if pooling == "attn_mean":
            set_dim_for_clf = 2 * set_embed_dim
        else:
            set_dim_for_clf = set_embed_dim

        clf_input_dim = (ele_input_dim if self.use_ele_feat else 0) + set_dim_for_clf

        self.classifier = build_mlp(
            input_dim=clf_input_dim,
            hidden_dims=list(clf_hidden_dims),
            output_dim=n_classes,
            activation=nn.ReLU,
            last_activation=False,
        )

    @staticmethod
    def _to_pyg_points(had_feat: torch.Tensor, had_mask: torch.Tensor):
        """
        Convert padded (B,N,F) + mask (B,N) into:
          x     : (num_points, F)
          batch : (num_points,) graph index in [0..B-1]
          idx_b, idx_n : indices to map points back to padded positions
        """
        if had_mask.dtype != torch.bool:
            had_mask = had_mask > 0

        B, N, F = had_feat.shape
        if N == 0:
            # empty padding dimension
            x = had_feat.new_zeros((0, F))
            batch = had_feat.new_zeros((0,), dtype=torch.long)
            idx_b = had_feat.new_zeros((0,), dtype=torch.long)
            idx_n = had_feat.new_zeros((0,), dtype=torch.long)
            return x, batch, idx_b, idx_n, B, N

        idx_b, idx_n = had_mask.nonzero(as_tuple=True)  # (num_points,)
        if idx_b.numel() == 0:
            x = had_feat.new_zeros((0, F))
            batch = had_feat.new_zeros((0,), dtype=torch.long)
            return x, batch, idx_b, idx_n, B, N

        x = had_feat[idx_b, idx_n, :]  # (num_points, F)
        batch = idx_b.to(torch.long)   # (num_points,)
        return x, batch, idx_b, idx_n, B, N

    def forward(self, ele_feat, had_feat, had_mask, return_attn: bool = False):
        """
        return_attn:
          - only meaningful for pooling in ("attn","attn_mean")
          - returns alpha_padded: (B, N) (padding=0)
        """
        # 1) to PyG points
        x, batch, idx_b, idx_n, B, N = self._to_pyg_points(had_feat, had_mask)

        # 2) optionally concat ele_feat to each point
        if self.use_ele_in_had_encoder:
            if x.numel() == 0:
                x_in = x.new_zeros((0, self.had_input_dim + self.ele_input_dim))
            else:
                ele_per_point = ele_feat[batch]  # (num_points, ele_dim)
                x_in = torch.cat([x, ele_per_point], dim=-1)
        else:
            x_in = x

        # 3) encode points
        if x_in.numel() == 0:
            z = x_in.new_zeros((0, self.set_embed_dim))
        else:
            z = self.had_encoder(x_in)  # (num_points, set_embed_dim)

        # 4) pooling (graph-level)
        # handle empty-point graphs: pooling ops will output zeros if we set dim_size=B via scatter;
        # for global_*_pool we can safely call on empty? Some versions may fail, so guard.
        alpha_padded = None

        if self.pooling == "mean":
            if z.numel() == 0:
                H = z.new_zeros((B, self.set_embed_dim))
            else:
                H = global_mean_pool(z, batch)  # (B, D)

        elif self.pooling == "sum":
            if z.numel() == 0:
                H = z.new_zeros((B, self.set_embed_dim))
            else:
                H = global_add_pool(z, batch)   # (B, D)

        elif self.pooling == "max":
            if z.numel() == 0:
                H = z.new_zeros((B, self.set_embed_dim))
            else:
                H = global_max_pool(z, batch)   # (B, D)

        elif self.pooling in ("attn", "attn_mean"):
            # attention weights per graph via pyg softmax(scores, batch)
            if self.attn_gate is None:
                raise RuntimeError("attn_gate is None but pooling requires attention.")

            if z.numel() == 0:
                H_attn = z.new_zeros((B, self.set_embed_dim))
                H_mean = z.new_zeros((B, self.set_embed_dim))
                alpha = z.new_zeros((0, 1))
            else:
                scores = self.attn_gate(z)          # (num_points, 1)
                alpha = softmax(scores, batch)      # (num_points, 1), normalized per graph
                H_attn = scatter(alpha * z, batch, dim=0, dim_size=B, reduce="sum")  # (B, D)
                H_mean = global_mean_pool(z, batch)  # (B, D)

            if return_attn:
                # map alpha back to padded (B,N)
                alpha_padded = had_feat.new_zeros((B, N))
                if idx_b.numel() > 0 and alpha.numel() > 0:
                    alpha_padded[idx_b, idx_n] = alpha.squeeze(-1)

            if self.pooling == "attn":
                H = H_attn
            else:
                H = torch.cat([H_attn, H_mean], dim=-1)  # (B, 2D)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # 5) classifier input
        if self.use_ele_feat:
            joint = torch.cat([ele_feat, H], dim=-1)
        else:
            joint = H

        logits = self.classifier(joint)

        if return_attn:
            return logits, alpha_padded
        return logits


if __name__ == "__main__":
    # quick self-test
    B, N = 4, 10
    ele = torch.randn(B, 3)
    had = torch.randn(B, N, 5)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :6] = True

    for pooling in ["mean", "sum", "max", "attn", "attn_mean"]:
        model = DeepSetsHF(
            had_input_dim=5,
            ele_input_dim=3,
            had_hidden_dims=(64, 64),
            set_embed_dim=64,
            clf_hidden_dims=(64, 64),
            n_classes=2,
            use_ele_in_had_encoder=True,
            use_ele_feat=True,
            pooling=pooling,
        )
        out = model(ele, had, mask, return_attn=True)
        logits, alpha = out
        print(pooling, logits.shape, None if alpha is None else alpha.shape)
