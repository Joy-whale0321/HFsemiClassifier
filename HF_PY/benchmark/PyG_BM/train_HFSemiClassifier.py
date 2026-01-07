# train_HFSemiClassifier.py
# With HFSemiClassifier + DeepSetsHF (PyG pooling benchmark)

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


# ==========================================================
#  根据 electron 的 pt 做 expo 拟合权重（你原代码保留）
A_D, B_D = 15.1744, -1.91749
A_B, B_B = 12.1074, -1.10499
W_MAX = 5.0


def get_pt_weight_from_logpt(pt_log: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pt = torch.exp(pt_log)
    w_D = torch.ones_like(pt)
    w_B = torch.ones_like(pt)

    mask_low = (pt >= 3.0) & (pt < 6.0)
    if mask_low.any():
        pt_low = pt[mask_low]
        logc_D = torch.clamp(A_D + B_D * pt_low, min=-50.0, max=50.0)
        logc_B = torch.clamp(A_B + B_B * pt_low, min=-50.0, max=50.0)
        logc_max = torch.maximum(logc_D, logc_B)
        w_D[mask_low] = torch.exp(logc_max - logc_D)
        w_B[mask_low] = torch.exp(logc_max - logc_B)

    mask_high = (pt >= 6.0) & (pt < 10.0)
    if mask_high.any():
        w_D[mask_high] = 3.0
        w_B[mask_high] = 1.0

    weights = torch.where(labels == 0, w_D, w_B)
    weights = torch.clamp(weights, max=W_MAX)
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train HF semi-leptonic electron classifier (Deep Sets / PyG pooling).")
    parser.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_1_0105.root",
        help="Pythia 生成的 ROOT 文件路径",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 的 num_workers")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/MLclassifier/Weight_of_Model",
        help="模型权重输出目录",
    )
    parser.add_argument("--val-frac", type=float, default=0.25, help="验证集占比")
    parser.add_argument("--fair-lambda", type=float, default=1.0, help="平衡两类之间loss差异的正则强度")
    parser.add_argument("--pt-min", type=float, default=5.0, help="electron minimum pt")
    parser.add_argument("--pt-max", type=float, default=8.0, help="electron maximum pt")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")

    # ======= NEW: benchmark switch =======
    parser.add_argument(
        "--pooling",
        type=str,
        default="sum",
        choices=["mean", "sum", "max", "attn", "attn_mean"],
        help="Set pooling type (benchmark knob).",
    )

    return parser.parse_args()


def count_classes(dataset, num_classes=2):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        y = int(dataset[i]["label"])
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts


def downsample_by_class(subset, n_keep_dict, num_classes=2, name="set"):
    from torch.utils.data import Subset

    counts_before = count_classes(subset, num_classes=num_classes)
    print(f"[INFO] {name}: before downsample, class counts (0..{num_classes-1}) = {counts_before.tolist()}")

    if not n_keep_dict or all((v is None or v <= 0) for v in n_keep_dict.values()):
        print(f"[INFO] {name}: n_keep_dict empty or all <=0, skip downsample.")
        return subset

    idx_per_class = {c: [] for c in range(num_classes)}
    idx_rest = []

    for i in range(len(subset)):
        y = int(subset[i]["label"])
        if 0 <= y < num_classes:
            idx_per_class[y].append(i)
        else:
            idx_rest.append(i)

    selected_indices = []
    for c in range(num_classes):
        idx_list = idx_per_class[c]
        n_all = len(idx_list)
        if n_all == 0:
            continue

        n_keep = n_keep_dict.get(c, None)
        if (n_keep is None) or (n_keep <= 0) or (n_keep >= n_all):
            selected_indices.extend(idx_list)
        else:
            idx_tensor = torch.tensor(idx_list, dtype=torch.long)
            perm = torch.randperm(n_all)
            chosen = idx_tensor[perm[:n_keep]].tolist()
            selected_indices.extend(chosen)

    selected_indices.extend(idx_rest)

    selected_indices = torch.tensor(selected_indices, dtype=torch.long)
    perm_all = torch.randperm(len(selected_indices))
    selected_indices = selected_indices[perm_all].tolist()

    new_subset = Subset(subset, selected_indices)

    counts_after = count_classes(new_subset, num_classes=num_classes)
    print(f"[INFO] {name}: after  downsample, class counts (0..{num_classes-1}) = {counts_after.tolist()}")

    return new_subset


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from: {args.root_file}")
    dataset = HFSemiClassifier(
        args.root_file,
        tree_name="tree",
        use_log_pt=True,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_abs_max=1.0,
        use_had_eta=False,
        had_pt_min=0.2,
        had_pt_max=None,
        min_had=0,
    )

    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"[INFO] Total electrons: {n_total}, train: {n_train}, val: {n_val}")

    # ===== 你原来的手动裁剪（保留）=====
    n_keep_train = {0: 15000, 1: 15000}
    n_keep_val = {0: 5000, 1: 5000}

    train_set = downsample_by_class(train_set, n_keep_dict=n_keep_train, num_classes=2, name="train")
    val_set = downsample_by_class(val_set, n_keep_dict=n_keep_val, num_classes=2, name="val")

    train_counts = count_classes(train_set, num_classes=2)
    n_D = train_counts[0].item()
    n_B = train_counts[1].item()
    print(f"[INFO] Train class counts (after downsample): D(0) = {n_D}, B(1) = {n_B}")

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

    # ======= benchmark knob here =======
    pooling = args.pooling

    had_hidden_dims = (128, 128, 128)
    clf_hidden_dims = (128, 128, 128)
    set_embed_dim = 128

    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=had_hidden_dims,
        set_embed_dim=set_embed_dim,
        clf_hidden_dims=clf_hidden_dims,
        n_classes=2,
        use_ele_in_had_encoder=True,
        use_ele_feat=True,
        pooling=pooling,
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("[INFO] Model constructed:")
    print(model)

    best_val_loss = float("inf")
    start_save_epoch = int(args.epochs * 0.3)
    train_loss_history = []
    val_loss_history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_class_correct = [0, 0, 0]
        train_class_total = [0, 0, 0]

        for batch in train_loader:
            ele = batch["ele_feat"].to(device)
            had = batch["had_feat"].to(device)
            mask = batch["had_mask"].to(device)
            labels = batch["label"].to(device)

            mask_D = (labels == 0)
            mask_B = (labels == 1)

            optimizer.zero_grad()
            logits = model(ele, had, mask)

            per_sample_loss = criterion(logits, labels)
            base_loss = per_sample_loss.mean()

            # fairness penalty（保留你原逻辑）
            if mask_D.any() and mask_B.any():
                loss_D = per_sample_loss[mask_D].mean()
                loss_B = per_sample_loss[mask_B].mean()
                fairness_penalty = (loss_D - loss_B) ** 2
            else:
                fairness_penalty = torch.tensor(0.0, device=device)

            loss = base_loss + args.fair_lambda * fairness_penalty
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=-1)
            _, preds = probs.max(dim=-1)

            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            for c in range(3):
                mask_c = (labels == c)
                if mask_c.any():
                    train_class_total[c] += mask_c.sum().item()
                    train_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item()

        avg_train_loss = train_loss / max(1, train_total)
        train_eff = train_correct / max(1, train_total)
        print(f"Epoch {epoch}: train efficiency = {train_eff:.3f}")

        names = ["D", "B", "other"]
        for c in range(3):
            eff_c = (train_class_correct[c] / train_class_total[c]) if train_class_total[c] > 0 else 0.0
            print(f"    Train {names[c]} eff: {eff_c:.4f} ({train_class_correct[c]}/{train_class_total[c]})")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        val_class_correct = [0, 0, 0]
        val_class_total = [0, 0, 0]

        with torch.no_grad():
            for batch in val_loader:
                ele = batch["ele_feat"].to(device)
                had = batch["had_feat"].to(device)
                mask = batch["had_mask"].to(device)
                labels = batch["label"].to(device)

                mask_D = (labels == 0)
                mask_B = (labels == 1)

                logits = model(ele, had, mask)
                per_sample_loss = criterion(logits, labels)
                base_loss = per_sample_loss.mean()

                if mask_D.any() and mask_B.any():
                    loss_D = per_sample_loss[mask_D].mean()
                    loss_B = per_sample_loss[mask_B].mean()
                    fairness_penalty = (loss_D - loss_B) ** 2
                else:
                    fairness_penalty = torch.tensor(0.0, device=device)

                loss = base_loss + args.fair_lambda * fairness_penalty

                val_loss += loss.item() * labels.size(0)

                probs = torch.softmax(logits, dim=-1)
                _, preds = probs.max(dim=-1)

                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                for c in range(3):
                    mask_c = (labels == c)
                    if mask_c.any():
                        val_class_total[c] += mask_c.sum().item()
                        val_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item()

        avg_val_loss = val_loss / max(1, val_total)
        val_eff = val_correct / max(1, val_total)
        print(f"Epoch {epoch}: valid efficiency = {val_eff:.3f}")

        for c in range(3):
            eff_c = (val_class_correct[c] / val_class_total[c]) if val_class_total[c] > 0 else 0.0
            print(f"    Validate {names[c]} eff_on_kept: {eff_c:.4f} ({val_class_correct[c]}/{val_class_total[c]})")

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={avg_train_loss:.4f}, train_eff={train_eff:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_eff={val_eff:.4f}, "
            f"time={dt:.1f}s"
        )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # early stopping
        loss_change = best_val_loss - avg_val_loss
        if loss_change < 1e-4:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping triggered after {args.patience} epochs with no improvement.")
            break

        # save best model
        if epoch >= start_save_epoch and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            pt_min_str = f"{args.pt_min:.1f}" if args.pt_min is not None else "None"
            pt_max_str = f"{args.pt_max:.1f}" if args.pt_max is not None else "None"

            had_arch_str = f"had{len(had_hidden_dims)}x{had_hidden_dims[0]}"
            clf_arch_str = f"clf{len(clf_hidden_dims)}x{clf_hidden_dims[0]}"
            arch_str = f"{had_arch_str}_{clf_arch_str}_{pooling}"

            best_name = f"DeepSetsHF_best_5FALL_{pt_min_str}-{pt_max_str}_{arch_str}_M10.pt"
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

    # plot loss curve
    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_loss_history, label="train loss")
    plt.plot(epochs, val_loss_history, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch (pt {args.pt_min}-{args.pt_max} GeV) | pooling={args.pooling}")
    plt.legend()
    plt.grid(True)

    loss_fig_path = os.path.join(
        args.out_dir, f"loss_curve_5FALL_pt{args.pt_min}-{args.pt_max}_pool{args.pooling}.png"
    )
    plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve saved to: {loss_fig_path}")


if __name__ == "__main__":
    main()
