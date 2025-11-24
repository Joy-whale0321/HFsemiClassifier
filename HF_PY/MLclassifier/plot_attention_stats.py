#!/usr/bin/env python3
# plot_attention_all.py
#
# 从 explain_HFSemiClassifier.py 生成的 attn_dump_val.npz 中，
# 画出：
#  1) D/B 的 attention-weighted logpt / deta / dphi 分布（旧的）
#  2) top-K attention hadron 的 logpt 分布（K=1,3,5）
#  3) attention vs logpt 的 2D map（D、B）
#  4) 按 |dphi| region 切片的 attention-weighted logpt 分布（core/mid/outer）

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced plots for attention analysis (D vs B)."
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
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/attn_all/attn_all",
        help="输出图片前缀",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否在屏幕上显示图像",
    )
    return parser.parse_args()


def plot_1d_weighted(x_D, w_D, x_B, w_B,
                     xlabel, title, out_path,
                     bins=50, range=None, show=False):
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
    if show:
        plt.show()
    else:
        plt.close()


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

    mask_valid = had_mask.astype(bool)   # (M, N)

    # 展开到 hadron-level
    logpt_all = had_feat[..., 0][mask_valid]
    deta_all  = had_feat[..., 1][mask_valid]
    sinDP_all = had_feat[..., 2][mask_valid]
    cosDP_all = had_feat[..., 3][mask_valid]
    dphi_all  = np.arctan2(sinDP_all, cosDP_all)
    w_all     = attn[mask_valid]

    labels_expanded = np.repeat(labels[:, None], had_mask.shape[1], axis=1)[mask_valid]
    is_D = (labels_expanded == 0)
    is_B = (labels_expanded == 1)

    print(f"[INFO] valid hadrons total: {mask_valid.sum()}")
    print(f"[INFO] D-hadrons: {is_D.sum()}, B-hadrons: {is_B.sum()}")

    out_prefix = args.out_prefix

    # ------------------------------------------------------------------
    # 1) 旧：attention-weighted 1D 分布（logpt / deta / dphi）
    # ------------------------------------------------------------------
    print("[INFO] Plotting basic 1D attention-weighted distributions...")

    logpt_D, logpt_B = logpt_all[is_D], logpt_all[is_B]
    w_D, w_B = w_all[is_D], w_all[is_B]

    plot_1d_weighted(
        logpt_D, w_D,
        logpt_B, w_B,
        xlabel="log(pt_h)",
        title="Attention-weighted log(pt_h) distribution (D vs B)",
        out_path=f"{out_prefix}_logpt.png",
        bins=60,
        show=args.show,
    )

    deta_D, deta_B = deta_all[is_D], deta_all[is_B]
    plot_1d_weighted(
        deta_D, w_D,
        deta_B, w_B,
        xlabel="Δη",
        title="Attention-weighted Δη distribution (D vs B)",
        out_path=f"{out_prefix}_deta.png",
        bins=60,
        show=args.show,
    )

    dphi_D, dphi_B = dphi_all[is_D], dphi_all[is_B]
    plot_1d_weighted(
        dphi_D, w_D,
        dphi_B, w_B,
        xlabel="Δφ (rad)",
        title="Attention-weighted Δφ distribution (D vs B)",
        out_path=f"{out_prefix}_dphi.png",
        bins=60,
        range=(-np.pi, np.pi),
        show=args.show,
    )

    # ------------------------------------------------------------------
    # 2) top-K attention hadron 的 logpt 分布（K=1,3,5）
    # ------------------------------------------------------------------
    print("[INFO] Computing top-K attention log(pt_h) distributions...")

    M, N, _ = had_feat.shape
    Ks = [1, 3, 5]

    # 为每个 K 收集 D/B 的 top-K hadron logpt
    logpt_topk_D = {K: [] for K in Ks}
    logpt_topk_B = {K: [] for K in Ks}

    for i in range(M):
        mask_i = had_mask[i].astype(bool)     # (N,)
        if not mask_i.any():
            continue

        attn_i = attn[i][mask_i]             # (n_i,)
        logpt_i = had_feat[i, mask_i, 0]     # (n_i,)

        # 从大到小排序
        idx_sorted = np.argsort(-attn_i)
        for K in Ks:
            k_eff = min(K, idx_sorted.size)
            idx_k = idx_sorted[:k_eff]
            if labels[i] == 0:   # D
                logpt_topk_D[K].extend(logpt_i[idx_k])
            elif labels[i] == 1: # B
                logpt_topk_B[K].extend(logpt_i[idx_k])

    for K in Ks:
        logpt_DK = np.array(logpt_topk_D[K])
        logpt_BK = np.array(logpt_topk_B[K])
        print(f"[INFO] K={K}: D size={logpt_DK.size}, B size={logpt_BK.size}")

        # 用“每个 hadron 1 次计数”的方式画直方
        plt.figure(figsize=(6, 4))
        plt.hist(
            logpt_DK, bins=60, alpha=0.5, density=True,
            label=f"D top-{K}", histtype="stepfilled",
        )
        plt.hist(
            logpt_BK, bins=60, alpha=0.5, density=True,
            label=f"B top-{K}", histtype="stepfilled",
        )
        plt.xlabel("log(pt_h) of top-K attention hadrons")
        plt.ylabel("density (normalized)")
        plt.title(f"log(pt_h) distribution of top-{K} attention hadrons")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = f"{out_prefix}_topK{K}_logpt.png"
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        if args.show:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------------
    # 3) attention vs logpt 的 2D map（D/B）
    # ------------------------------------------------------------------
    print("[INFO] Plotting 2D maps: attention vs log(pt_h)...")

    def plot_2d_map(x, y, title, out_path, xbins=60, ybins=40,
                    xrange=None, yrange=None):
        H, xedges, yedges = np.histogram2d(
            x, y, bins=[xbins, ybins],
            range=[xrange, yrange],
        )
        H = H.T  # imshow 需要 [y, x]

        plt.figure(figsize=(6, 4))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(
            H,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        plt.colorbar(label="counts")
        plt.xlabel("log(pt_h)")
        plt.ylabel("attention α")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        if args.show:
            plt.show()
        else:
            plt.close()

    alpha_all = w_all  # 这里 y = attention 本身

    # 只取 0 < alpha <= 1
    eps = 1e-6
    mask_nonzero = alpha_all > eps
    logpt_nz = logpt_all[mask_nonzero]
    alpha_nz = alpha_all[mask_nonzero]
    labels_nz = labels_expanded[mask_nonzero]

    mask_D_nz = labels_nz == 0
    mask_B_nz = labels_nz == 1

    # logpt 范围大概 [-5,2]，你可以按需要调整
    x_range = (logpt_nz.min(), logpt_nz.max())
    y_range = (0.0, alpha_nz.max())

    plot_2d_map(
        logpt_nz[mask_D_nz],
        alpha_nz[mask_D_nz],
        "D: attention vs log(pt_h)",
        f"{out_prefix}_2D_D_logpt_alpha.png",
        xrange=x_range,
        yrange=y_range,
    )

    plot_2d_map(
        logpt_nz[mask_B_nz],
        alpha_nz[mask_B_nz],
        "B: attention vs log(pt_h)",
        f"{out_prefix}_2D_B_logpt_alpha.png",
        xrange=x_range,
        yrange=y_range,
    )

    # ------------------------------------------------------------------
    # 4) 按 |dphi| region 切片的 attention-weighted logpt 分布
    #    例子：core: |dphi|<0.3, mid: 0.3~0.8, outer: 0.8~1.5
    # ------------------------------------------------------------------
    print("[INFO] Plotting log(pt_h) distributions in different |Δφ| regions...")

    abs_dphi = np.abs(dphi_all)

    regions = [
        ("core",  0.0, 0.3),
        ("mid",   0.3, 0.8),
        ("outer", 0.8, 1.5),
    ]

    for name, lo, hi in regions:
        region_mask = (abs_dphi >= lo) & (abs_dphi < hi)

        logpt_reg = logpt_all[region_mask]
        w_reg     = w_all[region_mask]
        labels_reg = labels_expanded[region_mask]

        is_D_reg = labels_reg == 0
        is_B_reg = labels_reg == 1

        logpt_D_reg, w_D_reg = logpt_reg[is_D_reg], w_reg[is_D_reg]
        logpt_B_reg, w_B_reg = logpt_reg[is_B_reg], w_reg[is_B_reg]

        title = f"Attention-weighted log(pt_h) in |Δφ|∈[{lo},{hi}) (D vs B)"
        out_path = f"{out_prefix}_logpt_region_{name}.png"

        plot_1d_weighted(
            logpt_D_reg, w_D_reg,
            logpt_B_reg, w_B_reg,
            xlabel=f"log(pt_h), region {name} (|Δφ|∈[{lo},{hi}))",
            title=title,
            out_path=out_path,
            bins=60,
            show=args.show,
        )

    print("[INFO] All plots done.")


if __name__ == "__main__":
    main()
