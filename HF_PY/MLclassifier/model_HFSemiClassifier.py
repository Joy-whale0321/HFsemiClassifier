# model.py
# Deep Sets 风格的 Heavy-Flavor 半轻衰变电子分类模型
#
# 输入:
#   - ele_feat: (B, 3)  float32, 例如 [log(pt_e), eta_e, charge_e]
#   - had_feat: (B, N, 5) float32, 例如 [log(pt_h), dEta, sin(dPhi), cos(dPhi), charge_h]
#   - had_mask: (B, N)  bool, True=有效 hadron, False=padding
#
# 输出:
#   - logits: (B, n_classes), 默认 n_classes=3 (D / B / other)

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

    每条样本 = 一条电子:
      - 电子特征 ele_feat: (B, ele_dim)
      - hadron 点云 had_feat: (B, N, had_dim)
      - had_mask: (B, N) 表示哪些 hadron 有效

    结构:
      1. per-hadron encoder: φ(x_k) -> h_k
      2. 对 {h_k} 做 masked mean pooling -> H_away
      3. [ele_feat, H_away] 拼接 -> classifier MLP -> logits
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
        pooling: str = "mean",  # "mean" or "sum"
    ):
        super().__init__()

        self.pooling = pooling
        self.use_ele_in_had_encoder = use_ele_in_had_encoder

        # 如果想在 encoding hadrons 时也把电子信息拼进去，可以打开这个 flag
        # 例如: [had_feat, ele_feat] 作为 per-particle 输入
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

        # classifier MLP (作用在 [ele_feat, H_away] 上)
        clf_input_dim = ele_input_dim + set_embed_dim
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
        # ele_feat: (B, ele_dim)
        # had_feat: (B, N, had_dim)
        # had_mask: (B, N)

        B, N, Hdim = had_feat.shape

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
        had_encoded_flat = self.had_encoder(per_had_input_flat)  # (B*N, set_embed_dim)
        had_encoded = had_encoded_flat.view(B, N, -1)            # (B, N, set_embed_dim)

        # ========== masked pooling over hadrons ==========
        # 把 mask 转成 float: (B, N, 1)
        mask = had_mask.unsqueeze(-1).float()  # True->1.0, False->0.0

        # 把 padding 部分强制为 0
        had_encoded = had_encoded * mask  # (B, N, set_embed_dim)

        # sum pooling
        had_sum = had_encoded.sum(dim=1)  # (B, set_embed_dim)

        if self.pooling == "mean":
            # 有效 hadron 数: (B, 1)
            valid_counts = mask.sum(dim=1)  # (B, 1)
            # 防止除 0
            valid_counts = torch.clamp(valid_counts, min=1.0)
            had_embed = had_sum / valid_counts  # (B, set_embed_dim)
        elif self.pooling == "sum":
            had_embed = had_sum
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # 对于极端情况: 某个样本没有任何 hadron (全 False)
        # 上面的 clamped 已经保证不会 NaN, 这时 had_embed=0

        # ========== 拼接电子特征 + 点云嵌入 ==========
        # ele_feat: (B, ele_dim)
        # had_embed: (B, set_embed_dim)
        joint_feat = torch.cat([ele_feat, had_embed], dim=-1)  # (B, ele_dim + set_embed_dim)

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
        pooling="mean",
    )

    logits = model(ele, had, mask)
    print("logits shape:", logits.shape)  # 预期: (B, 3)
