# model.py
# Deep Sets + (可选 Attention Pooling) 的 Heavy-Flavor 半轻衰变电子分类模型

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
    用 Deep Sets 处理 hadron 点云的 HF 电子分类模型.

    pooling 选项:
      - "mean"      : masked mean pooling (原来的做法)
      - "sum"       : masked sum pooling
      - "attn"      : 纯 attention pooling
      - "attn_mean" : attention pooling 和 mean pooling 拼接后一起送入分类器
    """

    def __init__(
        self,
        had_input_dim: int = 5,
        ele_input_dim: int = 3,
        had_hidden_dims=(64, 64),
        set_embed_dim: int = 64,
        clf_hidden_dims=(64, 64),
        n_classes: int = 3,
        use_ele_in_had_encoder: bool = False,
        pooling: str = "mean",      # "mean" / "sum" / "attn" / "attn_mean"
        attn_hidden_dim: int = 64,  # attention 内部隐藏维度
    ):
        super().__init__()

        self.pooling = pooling
        self.use_ele_in_had_encoder = use_ele_in_had_encoder
        self.set_embed_dim = set_embed_dim

        # 如果想在 encoding hadrons 时也把电子信息拼进去，可以打开这个 flag
        if self.use_ele_in_had_encoder:
            per_had_input_dim = had_input_dim + ele_input_dim
        else:
            per_had_input_dim = had_input_dim

        # per-particle encoder φ
        self.had_encoder = build_mlp(
            input_dim=per_had_input_dim,
            hidden_dims=list(had_hidden_dims),
            output_dim=set_embed_dim,
            activation=nn.ReLU,
            last_activation=True,
        )

        # attention 打分网络（只在 pooling 为 attn/attn_mean 时会用到）
        if self.pooling in ("attn", "attn_mean"):
            self.attn_mlp = nn.Sequential(
                nn.Linear(set_embed_dim, attn_hidden_dim),
                nn.Tanh(),
                nn.Linear(attn_hidden_dim, 1),  # -> score s_k
            )
        else:
            self.attn_mlp = None

        # 根据 pooling 类型确定 classifier 输入维度
        if self.pooling == "attn_mean":
            # 拼接 [H_attn, H_mean]，维度翻倍
            set_feature_dim_for_clf = set_embed_dim * 2
        else:
            set_feature_dim_for_clf = set_embed_dim

        clf_input_dim = ele_input_dim + set_feature_dim_for_clf

        # classifier MLP (作用在 [ele_feat, H_away(可能是attn/mean/sum)] 上)
        self.classifier = build_mlp(
            input_dim=clf_input_dim,
            hidden_dims=list(clf_hidden_dims),
            output_dim=n_classes,
            activation=nn.ReLU,
            last_activation=False,
        )

    def forward(self, ele_feat, had_feat, had_mask):
        """
        参数
        ----
        ele_feat: (B, ele_dim)
        had_feat: (B, N, had_dim)
        had_mask: (B, N) bool

        返回
        ----
        logits: (B, n_classes)
        """
        B, N, _ = had_feat.shape

        # ========== 构造 per-particle 输入 ==========
        if self.use_ele_in_had_encoder:
            # 把 ele_feat broadcast 到每个 hadron 上: (B, 1, ele_dim) -> (B, N, ele_dim)
            ele_expanded = ele_feat.unsqueeze(1).expand(-1, N, -1)
            per_had_input = torch.cat([had_feat, ele_expanded], dim=-1)  # (B, N, had_dim + ele_dim)
        else:
            per_had_input = had_feat  # (B, N, had_dim)

        # 展平 batch + N, 方便过 MLP: (B*N, input_dim)
        per_had_input_flat = per_had_input.view(B * N, -1)

        # 编码所有 hadron: φ(x_k)
        had_encoded_flat = self.had_encoder(per_had_input_flat)          # (B*N, set_embed_dim)
        had_encoded = had_encoded_flat.view(B, N, self.set_embed_dim)    # (B, N, set_embed_dim)

        # ========== mask ==========
        mask = had_mask.unsqueeze(-1).float()      # (B, N, 1), True->1.0, False->0.0
        had_encoded = had_encoded * mask           # padding 清零

        # ========== 基础 sum/mean pooling（有些模式要用到） ==========
        had_sum = had_encoded.sum(dim=1)           # (B, set_embed_dim)
        valid_counts = mask.sum(dim=1)             # (B, 1)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        had_mean = had_sum / valid_counts          # (B, set_embed_dim)

        # ========== attention pooling（可选） ==========
        if self.pooling in ("attn", "attn_mean"):
            # scores: (B, N, 1)
            scores = self.attn_mlp(had_encoded)    # 对于 mask==0 的地方，反正 had_encoded=0，score 也会被屏蔽
            # 将 padding 的位置的score设为很小，防止参与softmax
            scores = scores.masked_fill(had_mask.unsqueeze(-1) == 0, -1e9)
            # alpha: (B, N, 1)
            alpha = F.softmax(scores, dim=1)
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
            # 拼接 attention 和 mean 的信息
            had_embed = torch.cat([had_attn, had_mean], dim=-1)  # (B, 2*set_embed_dim)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # ========== 拼接电子特征 + 点云嵌入 ==========
        joint_feat = torch.cat([ele_feat, had_embed], dim=-1)  # (B, ele_dim + set_feature_dim_for_clf)

        # ========== 分类器 ==========
        logits = self.classifier(joint_feat)  # (B, n_classes)

        return logits


if __name__ == "__main__":
    # 简单自测一下 shape 是否匹配
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
        n_classes=3,
        use_ele_in_had_encoder=False,
        pooling="attn_mean",  # 可以改成 "mean" / "sum" / "attn"
    )

    logits = model(ele, had, mask)
    print("logits shape:", logits.shape)  # 预期: (B, 3)
