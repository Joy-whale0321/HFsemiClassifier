# train.py
# With HFSemiClassifier + DeepSetsHF
# weight of model will be saved to:
#   /home/jingyu/HepSimTools/DataDir/HF_PY/MLclassifier/Weight_of_Model

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF

from math import pi
from math import pi
import torch.nn.functional as F  # 如果后面不会用到可以不加，这里保险起见保留

from torch.utils.data import WeightedRandomSampler

# ==========================================================
#  根据 electron 的 pt 做 expo 拟合权重
#   count_D(pt) ≈ exp(A_D + B_D * pt)
#   count_B(pt) ≈ exp(A_B + B_B * pt)
A_D, B_D = 15.1744, -1.91749   # ln count_D(pt) ~ A_D + B_D * pt
A_B, B_B = 12.1074, -1.10499   # ln count_B(pt) ~ A_B + B_B * pt
W_MAX = 5.0 
def get_pt_weight_from_logpt(pt_log: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pt = torch.exp(pt_log)  # (B,)

    w_D = torch.ones_like(pt)
    w_B = torch.ones_like(pt)

    # ---------- 区间 3 <= pt < 6：用拟合的 expo 权重 ----------
    mask_low = (pt >= 3.0) & (pt < 6.0)
    if mask_low.any():
        pt_low = pt[mask_low]

        logc_D = A_D + B_D * pt_low
        logc_B = A_B + B_B * pt_low

        logc_D = torch.clamp(logc_D, min=-50.0, max=50.0)
        logc_B = torch.clamp(logc_B, min=-50.0, max=50.0)

        logc_max = torch.maximum(logc_D, logc_B)

        w_D_low = torch.exp(logc_max - logc_D)
        w_B_low = torch.exp(logc_max - logc_B)

        w_D[mask_low] = w_D_low
        w_B[mask_low] = w_B_low

    # ---------- 区间 6 <= pt < 10：手动固定权重 ----------
    mask_high = (pt >= 6.0) & (pt < 10.0)
    if mask_high.any():
        w_D[mask_high] = 3.0
        w_B[mask_high] = 1.0

    # ---------- 根据标签选对应权重 ----------
    weights = torch.where(labels == 0, w_D, w_B)

    # 保险起见再截一下最大值（保证不会爆炸）
    weights = torch.clamp(weights, max=W_MAX)

    return weights

# hyperparameters setup
def parse_args():
    parser = argparse.ArgumentParser(description="Train HF semi-leptonic electron classifier (Deep Sets).")
    parser.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_1.root",
        help="Pythia 生成的 ROOT 文件路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/Weight_of_Model",
        help="模型权重输出目录",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="验证集占比",
    )
    parser.add_argument(
        "--fair-lambda",
        type=float,
        default=0.0,
        help="平衡两类之间loss差异的正则强度",
    )
    parser.add_argument(
        "--pt-min",
        type=float,
        default=3.0,
        help="electron minimum pt",
    )
    parser.add_argument(
        "--pt-max",
        type=float,
        default=5.0,
        help="electron maximum pt",
    )
    # early stopping patience
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="early stopping 的耐心值：验证集 loss 连续多少个 epoch 无提升就停止",
    )
    return parser.parse_args()

# count classes in dataset
def count_classes(dataset, num_classes=2):
    import torch
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        y = int(dataset[i]["label"])
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts

def downsample_by_class(subset, n_keep_dict, num_classes=2, name="set"):
    """
    手动按类别裁剪：
      - subset: 可以是 HFSemiClassifier 或 torch.utils.data.Subset
      - n_keep_dict: dict，比如 {0: 10000, 1: 8000}
         * 如果某个 label 不在这个 dict 里，就一律“全保留”
      - num_classes: 只用于统计，比如 2 就只统计 0/1，其它 label 归类到“other”里
      - name: 打印信息用的名字（train / val）

    返回：裁剪后的 Subset（如果不需要裁剪就返回原 subset）
    """
    import torch
    from torch.utils.data import Subset

    # 先看原始分布（只统计 0..num_classes-1）
    counts_before = count_classes(subset, num_classes=num_classes)
    print(f"[INFO] {name}: before downsample, class counts (0..{num_classes-1}) = {counts_before.tolist()}")

    # 如果 n_keep_dict 为空或者所有值 <= 0，就不裁剪
    if not n_keep_dict or all((v is None or v <= 0) for v in n_keep_dict.values()):
        print(f"[INFO] {name}: n_keep_dict empty or all <=0, skip downsample.")
        return subset

    idx_per_class = {c: [] for c in range(num_classes)}
    idx_rest = []  # 不在 0..num_classes-1 的都放这里，全部保留

    # 收集索引
    for i in range(len(subset)):
        y = int(subset[i]["label"])
        if 0 <= y < num_classes:
            idx_per_class[y].append(i)
        else:
            idx_rest.append(i)

    # 对每个 class 做裁剪
    selected_indices = []

    for c in range(num_classes):
        idx_list = idx_per_class[c]
        n_all = len(idx_list)
        if n_all == 0:
            continue

        n_keep = n_keep_dict.get(c, None)
        if (n_keep is None) or (n_keep <= 0) or (n_keep >= n_all):
            # 不限制 / 不需要裁剪：全保留
            selected_indices.extend(idx_list)
        else:
            # 随机取 n_keep 个
            idx_tensor = torch.tensor(idx_list, dtype=torch.long)
            perm = torch.randperm(n_all)
            chosen = idx_tensor[perm[:n_keep]].tolist()
            selected_indices.extend(chosen)

    # 其它 label（比如 label=2）全部保留
    selected_indices.extend(idx_rest)

    # 整体 shuffle 一下
    selected_indices = torch.tensor(selected_indices, dtype=torch.long)
    perm_all = torch.randperm(len(selected_indices))
    selected_indices = selected_indices[perm_all].tolist()

    # 用 Subset 包起来
    new_subset = Subset(subset, selected_indices)

    # 再数一次 0..num_classes-1 的分布
    counts_after = count_classes(new_subset, num_classes=num_classes)
    print(f"[INFO] {name}: after  downsample, class counts (0..{num_classes-1}) = {counts_after.tolist()}")

    return new_subset

# main training and valid loop
def main():
    # read args and setting hyperparameters
    args = parse_args()

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # make sure output dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # ========== dataset and prepare ==========
    print(f"[INFO] Loading dataset from: {args.root_file}")
    dataset = HFSemiClassifier(
        args.root_file,
        tree_name="tree",
        use_log_pt=True,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_abs_max=1.0,
        use_had_eta=True,
        # dphi_windows=[
        #     (-1., 1.),     # 近端 |Δφ| < 0.5
        #     (-pi, -2.1),     # 远端左边 |Δφ| > 2.6
        #     (2.1, pi),       # 远端右边 |Δφ| > 2.6
        # ],
        had_pt_min=0.5,    # 举例：只用 pt > 0.5 GeV 的 hadron
        had_pt_max=None,
    )

    n_total = len(dataset) # number of total electrons
    n_val = int(n_total * args.val_frac) # number of validation electrons
    n_train = n_total - n_val # number of training electrons
    train_set, val_set = random_split(dataset, [n_train, n_val]) # random split train/val with given ratio

    print(f"[INFO] Total electrons: {n_total}, train: {n_train}, val: {n_val}")

    # ========= 手动给每个 set 设置要保留多少 =========
    # 你可以在这里直接改这些数字：
    #   比如 train 想要 D/B 各 10000, val 想要 D/B 各 3000
    n_keep_train = {
        0: 4800,   # label 0 (D)，<=0 或 None 表示“不裁剪，全部保留”
        1: 4800,   # label 1 (B)
    }
    n_keep_val = {
        0: 1600,
        1: 1600,
    }

    # 对 train_set 裁剪
    train_set = downsample_by_class(
        train_set,
        n_keep_dict=n_keep_train,
        num_classes=2,
        name="train"
    )

    # 对 val_set 裁剪
    val_set = downsample_by_class(
        val_set,
        n_keep_dict=n_keep_val,
        num_classes=2,
        name="val"
    )

    # ========= 裁剪完以后再统计一下 train 分布，并算 class_weights =========
    train_counts = count_classes(train_set, num_classes=2)
    n_D = train_counts[0].item() # number of D with label 0
    n_B = train_counts[1].item() # number of B with label 1
    print(f"[INFO] Train class counts (after downsample): D(0) = {n_D}, B(1) = {n_B}")

    # Calculate the class weight for loss function
    train_counts_f = train_counts.to(torch.float32).clamp(min=1.0)  
    class_weights = 1.0 / train_counts_f
    class_weights = class_weights / class_weights.mean()
    print(f"[INFO] Using class weights for loss (D, B) = {class_weights.tolist()}")

    class_weights = class_weights.to(device)

    # DataLoaders for train and val sets
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ========== model building ==========
    had_hidden_dims = (128, 128, 128)
    clf_hidden_dims = (128, 128, 128)
    set_embed_dim   = 128
    pooling         = "attn_mean"

    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=had_hidden_dims,
        set_embed_dim=set_embed_dim,
        clf_hidden_dims=clf_hidden_dims,
        n_classes=2,
        use_ele_in_had_encoder=False,
        use_ele_feat=True,
        pooling=pooling,
    ).to(device)

    # loss function and optimizer setup
    criterion = nn.CrossEntropyLoss(reduction="none")
    # criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4, # L2 正则强度
    )

    print("[INFO] Model constructed:")
    print(model)

    # ======================================================================
    # ========== training and valid epoch loop ==========
    # ======================================================================
    best_val_loss = float("inf") # for only saving the best model
    start_save_epoch = int(args.epochs * 0.3)  # start doing something after the front epochs
    train_loss_history = []
    val_loss_history   = []

    # early stopping 相关
    epochs_no_improve = 0   # 连续多少 epoch 没有提升

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- Train ----
        model.train()
        train_loss = 0.0 # cumulative loss
        train_correct = 0 # number of correct predictions
        train_total = 0 # number of total samples

        # for per-class efficiency calculation
        train_class_correct = [0, 0, 0]  # for label 0,1,2
        train_class_total   = [0, 0, 0]

        for batch in train_loader:
            ele = batch["ele_feat"].to(device)      # (B, 3) electron features
            had = batch["had_feat"].to(device)      # (B, N_max, 5) hadron features
            mask = batch["had_mask"].to(device)     # (B, N_max) hadron mask
            labels = batch["label"].to(device)      # (B,) ground-truth labels
            mask_D = (labels == 0)                  # D on labels
            mask_B = (labels == 1)                  # B on labels

            # # ===== let hadron features are 0, electron only =====
            # had = torch.zeros_like(had)

            optimizer.zero_grad()

            logits = model(ele, had, mask) # (B, 3) logits for 2 classes without softmax from DeepSetsHF
            
            # loss function
            # loss = criterion(logits, labels)
            
            per_sample_loss = criterion(logits, labels)        # (B,)
            base_loss = per_sample_loss.mean()

            # ---- 根据 electron 的 log(pt) 计算权重 ----
            # ele_feat 的第 0 个分量就是 log(pt_e)
            # pt_log = ele[:, 0].detach()  # 不让梯度通过权重反传回去
            # weights = get_pt_weight_from_logpt(pt_log, labels)  # (B,)
            # weights = weights.to(device)

            # # ---- 加权的 base loss ----
            # base_loss = (weights * per_sample_loss).mean()

            # ------- fairness regularization: encourage D/B loss similar --------------
            if mask_D.any() and mask_B.any():
                loss_D = per_sample_loss[mask_D].mean()
                loss_B = per_sample_loss[mask_B].mean()
                fairness_penalty = (loss_D - loss_B) ** 2
            else:
                # 如果本 batch 只含一种类，就不要加平衡项，避免数值问题
                fairness_penalty = torch.tensor(0.0, device=device)

            loss = base_loss + args.fair_lambda * fairness_penalty

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            # get the max probobility and corresponding predicted class 
            probs = torch.softmax(logits, dim=-1)              # (B, 2)
            max_prob, preds = probs.max(dim=-1)                # (B,)

            # calculate efficiency
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            # per-class efficiency counting on kept samples
            for c in range(3):  # 0: D, 1: B, 2: other
                mask_c = (labels == c)
                if mask_c.any():
                    train_class_total[c]   += mask_c.sum().item()
                    train_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item()

        avg_train_loss = train_loss / max(1, train_total)

        # showing train efficiency
        train_eff = train_correct / max(1, train_total)
        print(f"Epoch {epoch}: train efficiency = {train_eff:.3f}")

        # showing per-class efficiency
        names = ["D", "B", "other"]
        for c in range(3):
            if train_class_total[c] > 0:
                eff_c = train_class_correct[c] / train_class_total[c]
            else:
                eff_c = 0.0
            print(f"    Train {names[c]} eff: {eff_c:.4f} "
                  f"({train_class_correct[c]}/{train_class_total[c]})")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_total = 0  # validation all samples
        val_correct = 0  # correct number on all samples

        # per-class count, 0: D, 1: B, 2: other
        val_class_correct = [0, 0, 0]  # label 0, 1, 2
        val_class_total   = [0, 0, 0]
        names = ["D", "B", "other"]

        # Disable gradient calculation for validation
        with torch.no_grad():
            for batch in val_loader:
                ele = batch["ele_feat"].to(device)
                had = batch["had_feat"].to(device)
                mask = batch["had_mask"].to(device)
                labels = batch["label"].to(device)
                mask_D = (labels == 0)
                mask_B = (labels == 1)

                # # ===== let hadron features are 0, electron only =====
                # had = torch.zeros_like(had)

                logits = model(ele, had, mask)
                per_sample_loss = criterion(logits, labels)      # (B,)

                base_loss = per_sample_loss.mean()

                # ------- fairness regularization: encourage D/B loss similar --------------
                if mask_D.any() and mask_B.any():
                    loss_D = per_sample_loss[mask_D].mean()
                    loss_B = per_sample_loss[mask_B].mean()
                    fairness_penalty = (loss_D - loss_B) ** 2
                else:
                    fairness_penalty = torch.tensor(0.0, device=device)

                loss = base_loss + args.fair_lambda * fairness_penalty

                val_loss += loss.item() * labels.size(0)

                probs = torch.softmax(logits, dim=-1)            # (B, 2)
                max_prob, preds = probs.max(dim=-1)              # (B,)

                val_total  += labels.size(0)
                val_correct += (preds == labels).sum().item()

                # per-class count on kept samples
                for c in range(3):  # 0: D, 1: B, 3: other
                    mask_c = (labels == c)
                    if mask_c.any():
                        val_class_total[c]   += mask_c.sum().item()
                        val_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item()

        # average loss on kept samples
        avg_val_loss = val_loss / max(1, val_total)

        # efficiency: fraction of correct samples over all
        val_eff = val_correct / max(1, val_total)
        print(f"Epoch {epoch}: valid efficiency = {val_eff:.3f}")

        # show the per-class eff_on_kept
        for c in range(3):
            if val_class_total[c] > 0:
                eff_c = val_class_correct[c] / val_class_total[c]
            else:
                eff_c = 0.0
            print(f"    Validate {names[c]} eff_on_kept: {eff_c:.4f} "
                  f"({val_class_correct[c]}/{val_class_total[c]})")

        #  Time Performance logging and model saving  
        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={avg_train_loss:.4f}, train_eff={train_eff:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_eff={val_eff:.4f}, "
            f"time={dt:.1f}s"
        )

        # record the epoch 的 loss
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # early stopping check
        loss_change = best_val_loss - avg_val_loss
        if loss_change < 1e-4:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0
        if epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping triggered after {args.patience} epochs with no improvement.")
            break

        # only save the best one
        if epoch >= start_save_epoch and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # pt 区间字符串
            pt_min_str = f"{args.pt_min:.1f}" if args.pt_min is not None else "None"
            pt_max_str = f"{args.pt_max:.1f}" if args.pt_max is not None else "None"

            # 网络结构字符串（例如 had3x128_clf3x128_attn_mean）
            had_arch_str = f"had{len(had_hidden_dims)}x{had_hidden_dims[0]}"
            clf_arch_str = f"clf{len(clf_hidden_dims)}x{clf_hidden_dims[0]}"
            arch_str = f"{had_arch_str}_{clf_arch_str}_{pooling}"

            # 拼到文件名里 near sides away sides
            best_name = f"DeepSetsHF_best_ALL_{pt_min_str}-{pt_max_str}_{arch_str}_woW.pt"
            best_path = os.path.join(args.out_dir, best_name)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_eff": val_eff,
                    "train_eff": train_eff,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"[INFO] Best model updated, saved to: {best_path}")

    print("[INFO] Training finished.")

    # ====== 训练结束后画 loss 曲线 ======
    epochs = range(1, args.epochs + 1)

    plt.figure()
    plt.plot(epochs, train_loss_history, label="train loss")
    plt.plot(epochs, val_loss_history,   label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch (pt {args.pt_min}-{args.pt_max} GeV)")
    plt.legend()
    plt.grid(True)

    loss_fig_path = os.path.join(args.out_dir, f"loss_curve_ALL_pt{args.pt_min}-{args.pt_max}_woW.png")
    plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve saved to: {loss_fig_path}")

if __name__ == "__main__":
    main()
