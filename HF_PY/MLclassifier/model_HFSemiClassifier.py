# model_HFSemiClassifier.py
# Deep Sets + (可选 Attention Pooling) 的 Heavy-Flavor 半轻衰变电子分类模型
# 这版增加 use_ele_feat 开关，可以选择是否在 classifier 中使用 electron 的特征。
# 当 use_ele_feat=False 时，模型完全忽略 ele_feat，只基于 hadron 点云做分类。

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, last_activation=False):
    """
    构建一个简单的 MLP: Linear + Activation 堆叠.
    hidden_dims: list[int]
    """
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
    使用 Deep Sets 架构的 Heavy-Flavor 半轻衰变电子分类模型。

    输入:
      - had_feat: (B, N, had_input_dim)
      - had_mask: (B, N)  bool / 0-1, True=有效 hadron, False=padding
      - ele_feat: (B, ele_input_dim)，如果 use_ele_feat=False，这个输入会被忽略

    主要步骤:
      1) per-hadron encoder: phi(h_k) -> R^{set_embed_dim}
      2) pooling over hadrons (mean/sum/attn/attn_mean) 得到整团 hadron 的表征 H
      3) 如果 use_ele_feat=True: 拼接 [ele_feat, H] 再喂给 classifier
         如果 use_ele_feat=False: 只用 H 喂给 classifier
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
        pooling: str = "attn_mean",
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

        # ========== per-hadron encoder ==========
        # 如果 use_ele_in_had_encoder=True，则每个 hadron 同时看到 ele_feat
        per_had_input_dim = had_input_dim + (ele_input_dim if use_ele_in_had_encoder else 0)
        self.had_encoder = build_mlp(
            input_dim=per_had_input_dim,
            hidden_dims=list(had_hidden_dims),
            output_dim=set_embed_dim,
            activation=nn.ReLU,
            last_activation=True,
        )

        # ========== attention MLP (可选) ==========
        if self.pooling in ("attn", "attn_mean"):
            self.attn_mlp = nn.Sequential(
                nn.Linear(set_embed_dim, attn_hidden_dim),
                nn.Tanh(),
                nn.Linear(attn_hidden_dim, 1),  # -> score s_k
            )
        else:
            self.attn_mlp = None

        # ========== classifier 输入维度 ==========
        if self.pooling == "attn_mean":
            # 拼接 [H_attn, H_mean]，维度翻倍
            set_feature_dim_for_clf = set_embed_dim * 2
        else:
            set_feature_dim_for_clf = set_embed_dim

        # 根据 use_ele_feat 决定是否在 classifier 中使用 ele_feat
        clf_input_dim = (ele_input_dim if self.use_ele_feat else 0) + set_feature_dim_for_clf

        self.classifier = build_mlp(
            input_dim=clf_input_dim,
            hidden_dims=list(clf_hidden_dims),
            output_dim=n_classes,
            activation=nn.ReLU,
            last_activation=False,
        )

    def forward(self, ele_feat, had_feat, had_mask, return_attn: bool = False):
        """
        参数:
          ele_feat: (B, ele_input_dim)，如果 use_ele_feat=False，仅用于与 had_feat 拼接
                    (当 use_ele_in_had_encoder=True 时)
          had_feat: (B, N, had_input_dim)
          had_mask: (B, N) bool / 0/1，True=有效 hadron

        返回:
          logits: (B, n_classes)
          如果 return_attn=True，并且使用了 attention pooling，则返回 (logits, alpha)
          其中 alpha: (B, N) 是每个 hadron 的注意力权重；否则 alpha=None
        """
        B, N, _ = had_feat.shape

        # ========== per-hadron encoding ==========
        if self.use_ele_in_had_encoder:
            # 将 ele_feat broadcast 到每个 hadron: (B, 1, ele_dim) -> (B, N, ele_dim)
            ele_expanded = ele_feat.unsqueeze(1).expand(-1, N, -1)
            had_input = torch.cat([had_feat, ele_expanded], dim=-1)  # (B, N, had_input+ele_input)
        else:
            had_input = had_feat

        # reshape 成 (B*N, input_dim) 送入 MLP
        had_input_flat = had_input.view(B * N, -1)
        had_encoded_flat = self.had_encoder(had_input_flat)  # (B*N, set_embed_dim)
        had_encoded = had_encoded_flat.view(B, N, self.set_embed_dim)

        # mask: (B, N) -> (B, N, 1)
        if had_mask.dtype != torch.bool:
            had_mask_bool = had_mask > 0
        else:
            had_mask_bool = had_mask
        mask = had_mask_bool.unsqueeze(-1)  # (B, N, 1)

        # 将 padding 位置置零
        had_encoded = had_encoded * mask  # (B, N, set_embed_dim)

        # ========== mean / sum pooling ==========
        # 统计每个 batch 内有效 hadron 数
        valid_counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        had_sum = had_encoded.sum(dim=1)                      # (B, set_embed_dim)
        had_mean = had_sum / valid_counts                     # (B, set_embed_dim)

        # ========== attention pooling（如果需要）==========
        alpha = None
        if self.attn_mlp is not None:
            # scores: (B, N, 1)
            scores = self.attn_mlp(had_encoded)               # padding 的位置目前是 0
            scores = scores.masked_fill(~had_mask_bool.unsqueeze(-1), -1e9)
            alpha = F.softmax(scores, dim=1)                  # (B, N, 1)
            had_attn = torch.sum(alpha * had_encoded, dim=1)  # (B, set_embed_dim)
        else:
            had_attn = None

        # ========== 根据 pooling 类型得到最终 had_embed ==========
        if self.pooling == "mean":
            had_embed = had_mean
        elif self.pooling == "sum":
            had_embed = had_sum
        elif self.pooling == "attn":
            had_embed = had_attn
        elif self.pooling == "attn_mean":
            had_embed = torch.cat([had_attn, had_mean], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # ========== 构造 classifier 输入 ==========
        if self.use_ele_feat:
            joint_feat = torch.cat([ele_feat, had_embed], dim=-1)
        else:
            joint_feat = had_embed

        # ========== 分类器 ==========
        logits = self.classifier(joint_feat)

        if return_attn:
            if alpha is not None:
                alpha_out = alpha.squeeze(-1)  # (B, N)
            else:
                alpha_out = None
            return logits, alpha_out
        else:
            return logits


if __name__ == "__main__":
    # 简单自测
    B, N = 4, 10
    ele = torch.randn(B, 3)
    had = torch.randn(B, N, 5)
    mask = torch.ones(B, N, dtype=torch.bool)

    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=(64, 64),
        set_embed_dim=64,
        clf_hidden_dims=(64, 64),
        n_classes=2,
        use_ele_in_had_encoder=False,
        pooling="attn_mean",
        use_ele_feat=False,  # 测试“只用 hadron 信息”的情况
    )

    logits, alpha = model(ele, had, mask, return_attn=True)
    print("logits shape:", logits.shape)
    print("alpha shape:", None if alpha is None else alpha.shape)
