#!/usr/bin/env python3
# explain_HFSemiClassifier.py
#
# 读取训练好的 DeepSetsHF 模型，导出每个 event 的 attention 权重，
# 并简单打印前几个 event 的“最重要 hadron”，
# 另外根据验证集的输出画 ROC 曲线并计算 AUC。
# 现在增加：按 electron pT 分 4 个区间 (3–4, 4–6, 6–8, ≥8 GeV)
# 分别画 ROC 和 eff–purity。

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt   # 用于画 ROC / eff–purity 曲线

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain HF semi-leptonic classifier using attention weights."
    )

    parser.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_100M.root",
        help="Pythia 生成的 ROOT 文件路径（和训练时一致）",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/Weight_of_Model/DeepSetsHF_best.pt",
        help="训练时保存的最优模型 checkpoint 路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="DataLoader batch size",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="从全数据中拿多少做解释（和 train.py 一样的 random_split 方式）",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5000,
        help="最多解释多少个 event（防止太大）; <=0 表示全用",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/attn_dump_val.npz",
        help="输出 npz 文件路径（ROC 图也会用这个前缀）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="示例打印中，每个 event 展示 top-k hadron",
    )

    return parser.parse_args()


def compute_roc_auc(y_true, y_score):
    """
    简单实现一个二分类 ROC + AUC 计算（不依赖 sklearn）

    参数
    ----
    y_true : array-like, shape (N,)
        0/1 标签，这里我们约定 1 = B, 0 = D
    y_score: array-like, shape (N,)
        预测为正类(=1=B)的概率或得分

    返回
    ----
    fpr, tpr, thresholds, auc_value
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    # 按 score 从大到小排序
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # 总正例/负例数
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    if P == 0 or N == 0:
        print("[WARN] Only one class present, ROC/AUC not well-defined.")
        # 给一个退化的结果
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf]), 0.5

    # 逐个 threshold 扫描
    tpr_list = []
    fpr_list = []
    thresholds = []

    tp = 0
    fp = 0
    last_score = np.inf

    # 为了在最后加上 (0,0) 和 (1,1)
    tpr_list.append(0.0)
    fpr_list.append(0.0)
    thresholds.append(np.inf)

    for i in range(len(y_score_sorted)):
        score = y_score_sorted[i]
        label = y_true_sorted[i]

        # 每遇到一个新的 score，就把当前点记下来
        if score != last_score:
            tpr = tp / P
            fpr = fp / N
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            thresholds.append(score)
            last_score = score

        if label == 1:
            tp += 1
        else:
            fp += 1

    # 最后再追加终点
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    thresholds.append(-np.inf)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    thresholds = np.array(thresholds)

    # AUC = 积分 \int TPR dFPR，用梯形规则计算
    auc_value = np.trapz(tpr_arr, fpr_arr)

    return fpr_arr, tpr_arr, thresholds, auc_value


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ========= 1. 准备数据 =========
    print(f"[INFO] Loading dataset from: {args.root_file}")
    dataset = HFSemiClassifier(args.root_file, tree_name="tree", use_log_pt=True)

    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    _, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    if args.max_events > 0 and args.max_events < len(val_set):
        from torch.utils.data import Subset
        indices = list(range(args.max_events))
        val_set = Subset(val_set, indices)
        print(f"[INFO] Using only first {args.max_events} events from val_set")
    else:
        print(f"[INFO] Using full val_set of size {len(val_set)}")

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=hf_semi_collate,
    )

    # ========= 2. 构建模型 & load checkpoint =========
    print(f"[INFO] Building model and loading checkpoint from: {args.ckpt}")

    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=(256, 256, 256, 256),
        set_embed_dim=256,
        clf_hidden_dims=(256, 256, 256, 256),
        n_classes=2,
        use_ele_in_had_encoder=False,
        pooling="attn_mean",  # 和你训练时保持一致
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ========= 3. 跑一遍 val_loader，收集 attention =========
    all_labels = []
    all_ele_feat = []
    all_had_feat = []   # 每个元素 shape: (B, N_batch, 5)
    all_had_mask = []   # 每个元素 shape: (B, N_batch)
    all_attn = []       # 每个元素 shape: (B, N_batch)
    all_probs = []

    for batch in val_loader:
        ele = batch["ele_feat"].to(device)   # (B, 3)
        had = batch["had_feat"].to(device)   # (B, N_batch, 5)
        mask = batch["had_mask"].to(device)  # (B, N_batch)
        labels = batch["label"].to(device)   # (B,)

        logits, alpha = model(ele, had, mask, return_attn=True)  # alpha: (B, N_batch)
        probs = torch.softmax(logits, dim=-1)  # (B, 2)

        all_labels.append(labels.cpu().numpy())
        all_ele_feat.append(ele.cpu().numpy())
        all_had_feat.append(had.cpu().numpy())
        all_had_mask.append(mask.cpu().numpy())
        all_attn.append(alpha.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # ========= 4. 把 batch 列表合并成统一大小的数组 =========
    labels_arr = np.concatenate(all_labels, axis=0)        # (M,)
    ele_feat_arr = np.concatenate(all_ele_feat, axis=0)    # (M, 3)
    probs_arr = np.concatenate(all_probs, axis=0)          # (M, 2)

    # 不同 batch 的 N_batch 可能不一样，在这里 pad 到全局 max_N
    max_N = max(arr.shape[1] for arr in all_had_feat)
    M = labels_arr.shape[0]
    feat_dim = all_had_feat[0].shape[2]

    had_feat_arr = np.zeros((M, max_N, feat_dim), dtype=np.float32)
    had_mask_arr = np.zeros((M, max_N), dtype=bool)
    attn_arr = np.zeros((M, max_N), dtype=np.float32)

    offset = 0
    for hf, mk, at in zip(all_had_feat, all_had_mask, all_attn):
        B_cur, N_cur, _ = hf.shape
        had_feat_arr[offset:offset + B_cur, :N_cur, :] = hf
        had_mask_arr[offset:offset + B_cur, :N_cur] = mk.astype(bool)
        attn_arr[offset:offset + B_cur, :N_cur] = at
        offset += B_cur

    print(f"[INFO] Collected attention for {labels_arr.shape[0]} events.")
    print(f"[INFO] had_feat_arr shape = {had_feat_arr.shape}, attn_arr shape = {attn_arr.shape}")

    # ========= 4.5 计算 ROC & AUC（全体 e）并画图 =========
    # 约定：label=1 表示 B，是“正类”
    print("[INFO] Computing ROC curve and AUC for ALL electrons (B vs D, B=positive)...")
    y_true = (labels_arr == 1).astype(np.int64)
    # probs_arr[:,1] 是预测为 B 的概率
    y_score = probs_arr[:, 1]

    fpr, tpr, thresholds, auc_value = compute_roc_auc(y_true, y_score)
    print(f"[RESULT] AUC_all (B vs D) = {auc_value:.4f}")

    base_prefix = os.path.splitext(args.out)[0]

    # 全体 ROC
    roc_out = base_prefix + "_roc_all.png"
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ALL (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (B vs D, positive=B, ALL e)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_out, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve (ALL) saved to: {roc_out}")

    # ========= 4.6 计算 D 和 B 各自的 efficiency × purity 曲线（全体 e） =========
    print("[INFO] Computing efficiency–purity curves for D and B (ALL e)...")

    thresholds_ep = np.linspace(0.0, 1.0, 201)

    eff_B_list, pur_B_list = [], []
    eff_D_list, pur_D_list = [], []

    for thr in thresholds_ep:
        # ----- B 作为信号 -----
        sel_B = (y_score >= thr)

        TP_B = np.sum(sel_B & (y_true == 1))
        FP_B = np.sum(sel_B & (y_true == 0))
        FN_B = np.sum((~sel_B) & (y_true == 1))

        if TP_B + FN_B > 0:
            eff_B = TP_B / (TP_B + FN_B)
        else:
            eff_B = 0.0

        if TP_B + FP_B > 0:
            pur_B = TP_B / (TP_B + FP_B)
        else:
            pur_B = 1.0

        eff_B_list.append(eff_B)
        pur_B_list.append(pur_B)

        # ----- D 作为信号 -----
        sel_D = (y_score <= thr)

        TP_D = np.sum(sel_D & (y_true == 0))
        FP_D = np.sum(sel_D & (y_true == 1))
        FN_D = np.sum((~sel_D) & (y_true == 0))

        if TP_D + FN_D > 0:
            eff_D = TP_D / (TP_D + FN_D)
        else:
            eff_D = 0.0

        if TP_D + FP_D > 0:
            pur_D = TP_D / (TP_D + FP_D)
        else:
            pur_D = 1.0

        eff_D_list.append(eff_D)
        pur_D_list.append(pur_D)

    eff_B_arr_all = np.array(eff_B_list)
    pur_B_arr_all = np.array(pur_B_list)
    eff_D_arr_all = np.array(eff_D_list)
    pur_D_arr_all = np.array(pur_D_list)

    effpur_out_all = base_prefix + "_eff_purity_all.png"
    plt.figure(figsize=(5, 5))
    plt.plot(eff_B_arr_all, pur_B_arr_all, label="B as signal (ALL)")
    plt.plot(eff_D_arr_all, pur_D_arr_all, label="D as signal (ALL)")
    plt.xlabel("Efficiency")
    plt.ylabel("Purity")
    plt.title("Efficiency vs Purity (D/B, ALL e)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(effpur_out_all, dpi=150)
    plt.close()
    print(f"[INFO] Efficiency–purity curves (ALL) saved to: {effpur_out_all}")

    # ========= 4.7 按 electron pT 分区间画 ROC & eff–purity =========
    # ele_feat_arr[:,0] = log(pt_e)，这里用 np.exp 还原 GeV
    pt_e = np.exp(ele_feat_arr[:, 0])

    # 4 个区间: [3,4), [4,6), [6,8), [8, +inf)
    pt_bins = [
        (3.0, 4.0),
        (4.0, 6.0),
        (6.0, 8.0),
        (8.0, np.inf),
    ]

    # ----- (1) 各 pt 区间 ROC 放在一张图上 -----
    roc_pt_out = base_prefix + "_roc_ptbins.png"
    plt.figure(figsize=(5, 5))

    for (low, high) in pt_bins:
        if np.isinf(high):
            mask_bin = (pt_e >= low)
            bin_name = f">= {low:.0f} GeV"
        else:
            mask_bin = (pt_e >= low) & (pt_e < high)
            bin_name = f"{low:.0f}–{high:.0f} GeV"

        n_bin = np.sum(mask_bin)
        if n_bin < 10:
            print(f"[WARN] pT bin {bin_name} has only {n_bin} events, skip ROC.")
            continue

        y_true_bin = y_true[mask_bin]
        y_score_bin = y_score[mask_bin]

        fpr_b, tpr_b, thr_b, auc_b = compute_roc_auc(y_true_bin, y_score_bin)
        print(f"[RESULT] AUC (B vs D) in pT bin {bin_name}: {auc_b:.4f}")

        plt.plot(fpr_b, tpr_b, label=f"{bin_name} (AUC={auc_b:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves in different e pT bins (B vs D, B=positive)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_pt_out, dpi=150)
    plt.close()
    print(f"[INFO] ROC curves (pT bins) saved to: {roc_pt_out}")

    # ----- (2) 各 pt 区间的 eff–purity：B / D 各画一张 -----
    effpur_B_pt_out = base_prefix + "_eff_purity_B_ptbins.png"
    effpur_D_pt_out = base_prefix + "_eff_purity_D_ptbins.png"

    # B as signal
    plt.figure(figsize=(5, 5))
    for (low, high) in pt_bins:
        if np.isinf(high):
            mask_bin = (pt_e >= low)
            bin_name = f">= {low:.0f} GeV"
        else:
            mask_bin = (pt_e >= low) & (pt_e < high)
            bin_name = f"{low:.0f}–{high:.0f} GeV"

        n_bin = np.sum(mask_bin)
        if n_bin < 10:
            print(f"[WARN] pT bin {bin_name} has only {n_bin} events, skip eff–pur (B).")
            continue

        y_true_bin = y_true[mask_bin]
        y_score_bin = y_score[mask_bin]

        eff_B_list, pur_B_list = [], []
        for thr in thresholds_ep:
            sel_B = (y_score_bin >= thr)

            TP_B = np.sum(sel_B & (y_true_bin == 1))
            FP_B = np.sum(sel_B & (y_true_bin == 0))
            FN_B = np.sum((~sel_B) & (y_true_bin == 1))

            if TP_B + FN_B > 0:
                eff_B = TP_B / (TP_B + FN_B)
            else:
                eff_B = 0.0

            if TP_B + FP_B > 0:
                pur_B = TP_B / (TP_B + FP_B)
            else:
                pur_B = 1.0

            eff_B_list.append(eff_B)
            pur_B_list.append(pur_B)

        eff_B_arr = np.array(eff_B_list)
        pur_B_arr = np.array(pur_B_list)
        plt.plot(eff_B_arr, pur_B_arr, label=bin_name)

    plt.xlabel("Efficiency (B as signal)")
    plt.ylabel("Purity (B sample)")
    plt.title("B efficiency vs purity in different e pT bins")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(effpur_B_pt_out, dpi=150)
    plt.close()
    print(f"[INFO] B efficiency–purity (pT bins) saved to: {effpur_B_pt_out}")

    # D as signal
    plt.figure(figsize=(5, 5))
    for (low, high) in pt_bins:
        if np.isinf(high):
            mask_bin = (pt_e >= low)
            bin_name = f">= {low:.0f} GeV"
        else:
            mask_bin = (pt_e >= low) & (pt_e < high)
            bin_name = f"{low:.0f}–{high:.0f} GeV"

        n_bin = np.sum(mask_bin)
        if n_bin < 10:
            print(f"[WARN] pT bin {bin_name} has only {n_bin} events, skip eff–pur (D).")
            continue

        y_true_bin = y_true[mask_bin]
        y_score_bin = y_score[mask_bin]

        eff_D_list, pur_D_list = [], []
        for thr in thresholds_ep:
            sel_D = (y_score_bin <= thr)

            TP_D = np.sum(sel_D & (y_true_bin == 0))
            FP_D = np.sum(sel_D & (y_true_bin == 1))
            FN_D = np.sum((~sel_D) & (y_true_bin == 0))

            if TP_D + FN_D > 0:
                eff_D = TP_D / (TP_D + FN_D)
            else:
                eff_D = 0.0

            if TP_D + FP_D > 0:
                pur_D = TP_D / (TP_D + FP_D)
            else:
                pur_D = 1.0

            eff_D_list.append(eff_D)
            pur_D_list.append(pur_D)

        eff_D_arr = np.array(eff_D_list)
        pur_D_arr = np.array(pur_D_list)
        plt.plot(eff_D_arr, pur_D_arr, label=bin_name)

    plt.xlabel("Efficiency (D as signal)")
    plt.ylabel("Purity (D sample)")
    plt.title("D efficiency vs purity in different e pT bins")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(effpur_D_pt_out, dpi=150)
    plt.close()
    print(f"[INFO] D efficiency–purity (pT bins) saved to: {effpur_D_pt_out}")

    # ========= 5. 保存为 npz =========
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        labels=labels_arr,
        ele_feat=ele_feat_arr,
        had_feat=had_feat_arr,
        had_mask=had_mask_arr,
        attn=attn_arr,
        probs=probs_arr,
    )
    print(f"[INFO] Saved attention dump to: {args.out}")

    # ========= 6. 简单打印前几个 event 的 top-k hadron =========
    print("\n========== Example: show top-k hadrons for first few events ==========")
    topk = max(1, args.topk)
    num_show = min(5, labels_arr.shape[0])

    for i in range(num_show):
        label = labels_arr[i]  # 0 = D, 1 = B
        ele_vec = ele_feat_arr[i]  # [logpt_e, eta_e, charge_e]
        attn_i = attn_arr[i]   # (max_N,)
        mask_i = had_mask_arr[i].astype(bool)
        had_i = had_feat_arr[i]  # (max_N, 5)

        attn_valid = attn_i[mask_i]
        had_valid = had_i[mask_i]

        if attn_valid.size == 0:
            print(f"\nEvent {i}: label={label}, no hadrons (masked all).")
            continue

        idx_sorted = np.argsort(-attn_valid)  # 从大到小
        idx_topk = idx_sorted[:topk]

        print(f"\nEvent {i}: label={label} (0=D, 1=B)")
        print(f"  ele_feat = [logpt_e={ele_vec[0]:.3f}, eta_e={ele_vec[1]:.3f}, charge_e={ele_vec[2]:.1f}]")
        print("  Top-{} hadrons by attention:".format(topk))
        print("    idx  attn    logpt_h    deta     sin(dphi)  cos(dphi)  charge")
        for rank, j in enumerate(idx_topk):
            a = attn_valid[j]
            hf = had_valid[j]
            print(
                f"    {rank:2d}  {a:.4f}  {hf[0]:8.3f}  {hf[1]:7.3f}  "
                f"{hf[2]:10.3f}  {hf[3]:9.3f}  {hf[4]:6.1f}"
            )


if __name__ == "__main__":
    main()
