#!/usr/bin/env python3
# explain_HFSemiClassifier.py
#
# 读取训练好的 DeepSetsHF 模型，导出每个 event 的 attention 权重，
# 并简单打印前几个 event 的“最重要 hadron”。

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

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
        help="输出 npz 文件路径",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="示例打印中，每个 event 展示 top-k hadron",
    )

    return parser.parse_args()


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
        had_hidden_dims=(128, 128, 128),
        set_embed_dim=128,
        clf_hidden_dims=(128, 128, 128),
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
