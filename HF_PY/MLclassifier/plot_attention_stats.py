#!/usr/bin/env python3
# plot_attention_stats.py
#
# 从 explain_HFSemiClassifier.py 生成的 attn_dump_val.npz 中，
# 画出 D/B 的 attention-weighted logpt / deta / dphi 分布。

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot attention-weighted feature distributions for D/B."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/attn_dump_val.npz",
        help="explain_HFSemiClassifier.py 输出的 npz 文件路径",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/attn_plots",
        help="输出图片前缀（会加上 _logpt.png 等）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否在屏幕上显示图像（如果在终端服务器上可关掉）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading npz from: {args.npz}")
    data = np.load(args.npz)

    labels = data["labels"]      # (M,)
    had_feat = data["had_feat"]  # (M, N, 5) -> [logpt, deta, sin(dphi), cos(dphi), charge]
    had_mask = data["had_mask"]  # (M, N) bool
    attn = data["attn"]          # (M, N)

    print("[INFO] shapes:")
    print("  labels   :", labels.shape)
    print("  had_feat :", had_feat.shape)
    print("  had_mask :", had_mask.shape)
    print("  attn     :", attn.shape)

    # 只取有效 hadron
    mask_valid = had_mask.astype(bool)   # (M, N)

    logpt = had_feat[..., 0][mask_valid]   # 所有有效 hadron 的 logpt
    deta  = had_feat[..., 1][mask_valid]
    sinDP = had_feat[..., 2][mask_valid]
    cosDP = had_feat[..., 3][mask_valid]
    # charge = had_feat[..., 4][mask_valid]
    w_all = attn[mask_valid]              # attention 权重

    # 还原 dphi 到 [-pi, pi]
    dphi = np.arctan2(sinDP, cosDP)

    # 展开 labels 到 hadron 级别
    # labels_had.shape = (总有效 hadron 数,)
    labels_expanded = np.repeat(labels[:, None], had_mask.shape[1], axis=1)[mask_valid]
    is_D = (labels_expanded == 0)
    is_B = (labels_expanded == 1)

    # 简单 sanity check
    print(f"[INFO] valid hadrons total: {mask_valid.sum()}")
    print(f"[INFO] D-hadrons: {is_D.sum()}, B-hadrons: {is_B.sum()}")

    # 一个小工具函数画带权直方图
    def plot_1d_weighted(x_D, w_D, x_B, w_B,
                         xlabel, title, out_path,
                         bins=50, range=None):
        plt.figure(figsize=(6, 4))
        plt.hist(
            x_D,
            bins=bins,
            range=range,
            weights=w_D,
            alpha=0.5,
            density=True,
            label="D (weighted by attn)",
            histtype="stepfilled",
        )
        plt.hist(
            x_B,
            bins=bins,
            range=range,
            weights=w_B,
            alpha=0.5,
            density=True,
            label="B (weighted by attn)",
            histtype="stepfilled",
        )
        plt.xlabel(xlabel)
        plt.ylabel("attn-weighted density (normalized)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        if args.show:
            plt.show()
        else:
            plt.close()

    # ===== 1) logpt 分布 =====
    logpt_D = logpt[is_D]
    w_D = w_all[is_D]
    logpt_B = logpt[is_B]
    w_B = w_all[is_B]

    plot_1d_weighted(
        logpt_D, w_D,
        logpt_B, w_B,
        xlabel="log(pt_h)",
        title="Attention-weighted log(pt_h) distribution (D vs B)",
        out_path=f"{args.out_prefix}_logpt.png",
        bins=60,
    )

    # ===== 2) deta 分布 =====
    deta_D = deta[is_D]
    deta_B = deta[is_B]

    # 你可以根据你的 pseudorapidity 范围修改这个range
    plot_1d_weighted(
        deta_D, w_D,
        deta_B, w_B,
        xlabel="Δη = eta_h - eta_e (or whatever you stored)",
        title="Attention-weighted Δη distribution (D vs B)",
        out_path=f"{args.out_prefix}_deta.png",
        bins=60,
    )

    # ===== 3) dphi 分布 =====
    dphi_D = dphi[is_D]
    dphi_B = dphi[is_B]

    plot_1d_weighted(
        dphi_D, w_D,
        dphi_B, w_B,
        xlabel="Δφ (rad)",
        title="Attention-weighted Δφ distribution (D vs B)",
        out_path=f"{args.out_prefix}_dphi.png",
        bins=60,
        range=(-np.pi, np.pi),
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
