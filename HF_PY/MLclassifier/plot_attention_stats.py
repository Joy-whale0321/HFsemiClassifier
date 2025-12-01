#!/usr/bin/env python3
# plot_attention_all_fixed.py
#
# 从 explain_HFSemiClassifier.py 生成的 attn_dump_val.npz 中，
# 画出（使用“相对重要性” I = alpha * N_valid）：
#  1) D/B 的 importance-weighted logpt / deta / dphi 分布
#  2) top-K attention hadron 的 logpt 分布（K=1,3,5）
#  3) attention vs logpt 的 2D map（D、B）——只是看 (logpt, alpha) 的点云分布
#  4) 按 |dphi| region 切片的 importance-weighted logpt 分布（core/mid/outer）
#  5) 新：在 (deta, dphi) 平面上的 mean-importance map：D/B 及 Δ(B−D)

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced plots for attention analysis (D vs B) with relative importance."
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
    """1D 分布，使用给定的权重（这里是 relative importance）"""
    plt.figure(figsize=(6, 4))
    plt.hist(
        x_D,
        bins=bins,
        range=range,
        weights=w_D,
        alpha=0.5,
        density=True,
        label="D (importance-weighted)",
        histtype="stepfilled",
    )
    plt.hist(
        x_B,
        bins=bins,
        range=range,
        weights=w_B,
        alpha=0.5,
        density=True,
        label="B (importance-weighted)",
        histtype="stepfilled",
    )
    plt.xlabel(xlabel)
    plt.ylabel("importance-weighted density (normalized)")
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


def plot_2d_mean(x, y, w, title, out_path,
                 xbins=50, ybins=50,
                 xrange=None, yrange=None,
                 vmin=None, vmax=None,
                 cmap="viridis",
                 show=False,
                 cbar_label="mean importance"):
    """
    在 (x, y) 上画 mean(w)，即 sum(w)/counts 的 2D map。
    如果某个 bin 没有点，mean 设为 0。
    """
    H_count, xedges, yedges = np.histogram2d(
        x, y, bins=[xbins, ybins], range=[xrange, yrange]
    )
    H_sum, _, _ = np.histogram2d(
        x, y, bins=[xbins, ybins], range=[xrange, yrange], weights=w
    )

    mean = np.zeros_like(H_sum)
    np.divide(
        H_sum,
        H_count,
        out=mean,
        where=H_count > 0,
    )

    mean = mean.T  # imshow 需要 [y, x]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        mean,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.xlabel("Δη")
    plt.ylabel("Δφ (rad)")
    plt.title(title)
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

    # ---------- 计算每个 hadron 的相对重要性 I = alpha * N_valid ----------
    # N_valid: 每个 event 中有效 hadron 的个数
    N_per_evt = mask_valid.sum(axis=1)   # (M,)
    # 展开成和 had_feat 同样 (M, N) 的形状
    N_expanded = np.repeat(N_per_evt[:, None], had_mask.shape[1], axis=1)
    # 只取有效 hadron 的对应 N_valid
    N_all = N_expanded[mask_valid]       # (num_valid_hadrons,)

    # flatten 到 hadron-level
    logpt_all = had_feat[..., 0][mask_valid]
    deta_all  = had_feat[..., 1][mask_valid]
    sinDP_all = had_feat[..., 2][mask_valid]
    cosDP_all = had_feat[..., 3][mask_valid]
    dphi_all  = np.arctan2(sinDP_all, cosDP_all)
    alpha_all = attn[mask_valid]

    # 相对重要性：I_i = alpha_i * N_valid(event)
    I_all = alpha_all * N_all

    # 展开 labels 到 hadron-level
    labels_expanded = np.repeat(labels[:, None], had_mask.shape[1], axis=1)[mask_valid]
    is_D = (labels_expanded == 0)
    is_B = (labels_expanded == 1)

    print(f"[INFO] valid hadrons total: {mask_valid.sum()}")
    print(f"[INFO] D-hadrons: {is_D.sum()}, B-hadrons: {is_B.sum()}")

    out_prefix = args.out_prefix

    # ------------------------------------------------------------------
    # 1) importance-weighted 1D 分布（logpt / deta / dphi）
    # ------------------------------------------------------------------
    print("[INFO] Plotting importance-weighted 1D distributions...")

    logpt_D, logpt_B = logpt_all[is_D], logpt_all[is_B]
    I_D, I_B = I_all[is_D], I_all[is_B]

    plot_1d_weighted(
        logpt_D, I_D,
        logpt_B, I_B,
        xlabel="log(pt_h)",
        title="Importance-weighted log(pt_h) distribution (D vs B)",
        out_path=f"{out_prefix}_logpt_importance.png",
        bins=60,
        show=args.show,
    )

    deta_D, deta_B = deta_all[is_D], deta_all[is_B]
    plot_1d_weighted(
        deta_D, I_D,
        deta_B, I_B,
        xlabel="Δη",
        title="Importance-weighted Δη distribution (D vs B)",
        out_path=f"{out_prefix}_deta_importance.png",
        bins=60,
        show=args.show,
    )

    dphi_D, dphi_B = dphi_all[is_D], dphi_all[is_B]
    plot_1d_weighted(
        dphi_D, I_D,
        dphi_B, I_B,
        xlabel="Δφ (rad)",
        title="Importance-weighted Δφ distribution (D vs B)",
        out_path=f"{out_prefix}_dphi_importance.png",
        bins=60,
        range=(-np.pi, np.pi),
        show=args.show,
    )

    # ------------------------------------------------------------------
    # 2) top-K attention hadron 的 logpt 分布（K=1,3,5）
    #    排序用 alpha 或 I 都一样（同一 event 内 N_valid 是常数）
    # ------------------------------------------------------------------
    print("[INFO] Computing top-K attention log(pt_h) distributions...")

    M, N, _ = had_feat.shape
    Ks = [1, 3, 5]

    logpt_topk_D = {K: [] for K in Ks}
    logpt_topk_B = {K: [] for K in Ks}

    for i in range(M):
        mask_i = had_mask[i].astype(bool)     # (N,)
        if not mask_i.any():
            continue

        attn_i = attn[i][mask_i]             # (n_i,)
        logpt_i = had_feat[i, mask_i, 0]     # (n_i,)

        idx_sorted = np.argsort(-attn_i)     # 从大到小
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
    #    这里只看 (logpt, alpha) 的分布，不做权重归一
    # ------------------------------------------------------------------
    print("[INFO] Plotting 2D maps: attention vs log(pt_h)...")

    def plot_2d_scatter_hist(x, y, title, out_path,
                             xbins=60, ybins=40,
                             xrange=None, yrange=None,
                             show=False):
        H, xedges, yedges = np.histogram2d(
            x, y, bins=[xbins, ybins],
            range=[xrange, yrange],
        )
        H = H.T
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(6, 4))
        im = plt.imshow(
            H,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        plt.colorbar(im, label="counts")
        plt.xlabel("log(pt_h)")
        plt.ylabel("attention α")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        if show:
            plt.show()
        else:
            plt.close()

    eps = 1e-6
    mask_nonzero = alpha_all > eps
    logpt_nz = logpt_all[mask_nonzero]
    alpha_nz = alpha_all[mask_nonzero]
    labels_nz = labels_expanded[mask_nonzero]

    mask_D_nz = labels_nz == 0
    mask_B_nz = labels_nz == 1

    x_range = (logpt_nz.min(), logpt_nz.max())
    y_range = (0.0, alpha_nz.max())

    plot_2d_scatter_hist(
        logpt_nz[mask_D_nz],
        alpha_nz[mask_D_nz],
        "D: attention vs log(pt_h)",
        f"{out_prefix}_2D_D_logpt_alpha.png",
        xrange=x_range,
        yrange=y_range,
        show=args.show,
    )

    plot_2d_scatter_hist(
        logpt_nz[mask_B_nz],
        alpha_nz[mask_B_nz],
        "B: attention vs log(pt_h)",
        f"{out_prefix}_2D_B_logpt_alpha.png",
        xrange=x_range,
        yrange=y_range,
        show=args.show,
    )

    # ------------------------------------------------------------------
    # 4) 按 |dphi| region 切片的 importance-weighted logpt 分布
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
        I_reg     = I_all[region_mask]
        labels_reg = labels_expanded[region_mask]

        is_D_reg = labels_reg == 0
        is_B_reg = labels_reg == 1

        logpt_D_reg, I_D_reg = logpt_reg[is_D_reg], I_reg[is_D_reg]
        logpt_B_reg, I_B_reg = logpt_reg[is_B_reg], I_reg[is_B_reg]

        title = f"Importance-weighted log(pt_h) in |Δφ|∈[{lo},{hi}) (D vs B)"
        out_path = f"{out_prefix}_logpt_region_{name}_importance.png"

        plot_1d_weighted(
            logpt_D_reg, I_D_reg,
            logpt_B_reg, I_B_reg,
            xlabel=f"log(pt_h), region {name} (|Δφ|∈[{lo},{hi}))",
            title=title,
            out_path=out_path,
            bins=60,
            show=args.show,
        )

    # ------------------------------------------------------------------
    # 5) 在 (Δη, Δφ) 平面上的 mean-importance map：D / B / Δ(B−D)
    # ------------------------------------------------------------------
    print("[INFO] Plotting mean-importance maps in (Δη, Δφ) space...")

    deta_D_all, dphi_D_all, I_D_all = deta_all[is_D], dphi_all[is_D], I_all[is_D]
    deta_B_all, dphi_B_all, I_B_all = deta_all[is_B], dphi_all[is_B], I_all[is_B]

    eta_range = (deta_all.min(), deta_all.max())
    phi_range = (-np.pi, np.pi)

    # 先画 D/B 的 mean-importance
    plot_2d_mean(
        deta_D_all,
        dphi_D_all,
        I_D_all,
        "D: mean relative importance in (Δη, Δφ)",
        f"{out_prefix}_2D_D_deta_dphi_meanImportance.png",
        xrange=eta_range,
        yrange=phi_range,
        cbar_label="mean I (D)",
        show=args.show,
    )

    plot_2d_mean(
        deta_B_all,
        dphi_B_all,
        I_B_all,
        "B: mean relative importance in (Δη, Δφ)",
        f"{out_prefix}_2D_B_deta_dphi_meanImportance.png",
        xrange=eta_range,
        yrange=phi_range,
        cbar_label="mean I (B)",
        show=args.show,
    )

    # 再画 Δ(B−D) 的 mean-importance map
    # 为了 Delta，我们需要在同一个 binning 下分别算 D/B 的 mean，然后相减。
    print("[INFO] Computing delta (B - D) mean-importance map...")

    xbins = 50
    ybins = 50

    # D
    Hc_D, xedges, yedges = np.histogram2d(
        deta_D_all, dphi_D_all,
        bins=[xbins, ybins],
        range=[eta_range, phi_range],
    )
    Hw_D, _, _ = np.histogram2d(
        deta_D_all, dphi_D_all,
        bins=[xbins, ybins],
        range=[eta_range, phi_range],
        weights=I_D_all,
    )
    mean_D = np.zeros_like(Hw_D)
    np.divide(Hw_D, Hc_D, out=mean_D, where=Hc_D > 0)

    # B
    Hc_B, _, _ = np.histogram2d(
        deta_B_all, dphi_B_all,
        bins=[xbins, ybins],
        range=[eta_range, phi_range],
    )
    Hw_B, _, _ = np.histogram2d(
        deta_B_all, dphi_B_all,
        bins=[xbins, ybins],
        range=[eta_range, phi_range],
        weights=I_B_all,
    )
    mean_B = np.zeros_like(Hw_B)
    np.divide(Hw_B, Hc_B, out=mean_B, where=Hc_B > 0)

    # Δ = mean_B - mean_D
    delta_mean = mean_B - mean_D
    delta_mean = delta_mean.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    vmax = np.max(np.abs(delta_mean))
    vlim = vmax if vmax > 0 else None

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        delta_mean,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-vlim,
        vmax=vlim,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Δ mean I (B - D)")
    plt.xlabel("Δη")
    plt.ylabel("Δφ (rad)")
    plt.title("Delta mean relative importance (B - D) in (Δη, Δφ)")
    plt.tight_layout()
    out_path = f"{out_prefix}_2D_delta_BminusD_deta_dphi_meanImportance.png"
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved: {out_path}")
    if args.show:
        plt.show()
    else:
        plt.close()

    print("[INFO] All plots done.")


if __name__ == "__main__":
    main()
