# train_HFSemiClassifier.py
# With HFSemiClassifier + DeepSetsHF (PyG pooling benchmark)

import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF, PointNetHF

def parse_args():
    parser = argparse.ArgumentParser(description="Train HF semi-leptonic electron classifier (Deep Sets / PyG pooling).")
    parser.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_1_allAccept.root",
        help="Pythia 生成的 ROOT 文件路径",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 的 num_workers")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM_optimize/Weight_of_Model/deepset/",
        help="模型权重输出目录",
    )
    parser.add_argument("--val-frac", type=float, default=0.25, help="验证集占比")
    parser.add_argument("--fair-lambda", type=float, default=0.0, help="平衡两类之间loss差异的正则强度")
    parser.add_argument("--pt-min", type=float, default=3.0, help="electron minimum pt")
    parser.add_argument("--pt-max", type=float, default=10.0, help="electron maximum pt")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument("--ds-pt-bin-width", type=float, default=0.25,
                    help="downsample用的e pT分bin宽度(GeV)，例如1.0表示3-4-5-...")
    parser.add_argument("--ds-pt-edges", type=str, default="",
                        help="可选：手动指定downsample的pt边界，如 '3,4,5,6,8'；若非空则覆盖bin-width方案")

    # ======= NEW: benchmark switch =======
    parser.add_argument(
        "--pooling",
        type=str,
        default="sum",
        choices=["mean", "sum", "max", "attn", "attn_mean"],
        help="Set pooling type (benchmark knob).",
    )
    # optimize operating-point penalty args
    parser.add_argument("--op-lambda", type=float, default=0.5,
                        help="operating-point penalty strength (route2)")
    parser.add_argument("--op-eff", type=float, default=0.40,
                        help="target efficiency for both B and D operating points")
    parser.add_argument("--op-tau", type=float, default=0.05,
                        help="soft margin temperature for operating-point penalty")


    return parser.parse_args()

def count_classes(dataset, num_classes=2):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        y = int(dataset[i]["label"])
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts

# ================= 下采样 =================
# def pt pin edges 解析
def parse_pt_edges(args):
    # 先看pt edges def，去掉字符串首尾的空格
    if args.ds_pt_edges.strip(): 
        edges = [float(x) for x in args.ds_pt_edges.split(",")]
        edges = sorted(edges)
        if len(edges) < 2:
            raise ValueError("ds-pt-edges must have >=2 numbers")
        return np.array(edges, dtype=np.float32)

    # without pt edges def，用 bin width 自动生成：覆盖 [pt_min, pt_max]
    if args.pt_min is None or args.pt_max is None:
        raise ValueError("Need --pt-min and --pt-max to auto-build ds pt bins")

    w = float(args.ds_pt_bin_width)
    if w <= 0:
        raise ValueError("--ds-pt-bin-width must be > 0")

    # edges: pt_min, pt_min+w, ..., >=pt_max
    edges = [float(args.pt_min)]
    x = float(args.pt_min)
    while x + w < float(args.pt_max) - 1e-6:
        x += w
        edges.append(x)
    edges.append(float(args.pt_max))
    return np.array(edges, dtype=np.float32)

# 用 subset-local index, 映射回 dataset-global index, for取真实 pt
def subset_local_to_global_dataset_idx(subset, i_local):
    # random_split 返回的是 torch.utils.data.Subset
    # subset.indices 是“dataset全局idx”的列表
    return int(subset.indices[i_local])

# get electron pt from dataset, given global idx 
def get_electron_pt_from_dataset(dataset, global_idx):
    # dataset.electron_index[global_idx] = (evt_idx, ele_idx)
    evt_idx, ele_idx = dataset.electron_index[global_idx]
    return float(dataset.ele_pt[evt_idx][ele_idx])

# 把 subset 里的每条样本，按照 (pt bin 编号 b, 类别 y) 分bin，最后返回一个“bin → 样本列表”的字典。
def build_ptbin_class_index(subset, dataset, pt_edges, num_classes=2):
    """
    返回：
      idx_map[(b, c)] = [subset-local idx...]
    其中 b 是 pt bin 编号：0..(n_bins-1)
    """
    n_bins = len(pt_edges) - 1
    idx_map = {(b, c): [] for b in range(n_bins) for c in range(num_classes)}

    for i_local in range(len(subset)):
        y = int(subset[i_local]["label"])
        if not (0 <= y < num_classes):
            continue

        gidx = subset_local_to_global_dataset_idx(subset, i_local)
        pt = get_electron_pt_from_dataset(dataset, gidx)

        # 找 pt 属于哪个 bin：[edge[b], edge[b+1])
        b = int(np.searchsorted(pt_edges, pt, side="right") - 1)
        if 0 <= b < n_bins:
            idx_map[(b, y)].append(i_local)

    return idx_map

# 在每个 pt bin 内，按 min(nD, nB) 随机抽样，并把所有 bin 拼起来
def resample_balanced_by_ptbin(subset, idx_map, pt_edges, generator=None, num_classes=2):
    """
    每个 epoch 调一次：
      对每个 pt bin：
        取 n_keep = min(nD, nB)
        从D、B各自随机抽 n_keep（不放回）
      然后把所有 bin 拼起来并打乱
    """
    from torch.utils.data import Subset

    n_bins = len(pt_edges) - 1
    selected = []

    for b in range(n_bins):
        pools = [idx_map[(b, c)] for c in range(num_classes)]
        if any(len(p) == 0 for p in pools):
            # 某个class在这个bin为0，无法平衡：直接跳过这个bin（更干净）
            continue

        n_keep = min(len(pools[0]), len(pools[1]))
        # base_keep = min(len(pools[0]), len(pools[1]))
        # n_keep = math.floor(0.1 * base_keep) # 下采样比例10%

        for c in range(num_classes):
            pool = pools[c]
            n_all = len(pool)
            perm = torch.randperm(n_all, generator=generator)
            chosen = [pool[j] for j in perm[:n_keep].tolist()]
            selected.extend(chosen)

    if len(selected) == 0:
        # 极端情况：没抽到任何（比如某些bin全空），就退化为原subset
        return subset

    perm_all = torch.randperm(len(selected), generator=generator).tolist()
    selected = [selected[k] for k in perm_all]
    return Subset(subset, selected)


# optimze
def op_point_penalty(pB, labels, eff=0.40, tau=0.05):
    """
    pB: (B,) probability of class B
    labels: (B,) 0=D, 1=B
    Returns:
      penalty_B: D samples with pB > tB  (hurts purity_B at eff_B=eff)
      penalty_D: B samples with pB < tD  (hurts purity_D at eff_D=eff)
    """
    maskD = (labels == 0)
    maskB = (labels == 1)

    # guard: if batch has only one class, no op penalty
    if (not maskD.any()) or (not maskB.any()):
        return pB.new_tensor(0.0), pB.new_tensor(0.0)

    pB_D = pB[maskD]
    pB_B = pB[maskB]

    # ---- threshold for B-eff=eff: keep top-eff fraction among B ----
    # want: fraction(pB_B > tB) = eff  -> tB = quantile at (1-eff)
    tB = torch.quantile(pB_B, 1.0 - eff).detach()

    # penalize D that would pass B cut (false positives for B selection)
    # softplus((pB - tB)/tau) ~ max(0, pB - tB) when tau small
    penalty_B = F.softplus((pB_D - tB) / tau).mean()

    # ---- threshold for D-eff=eff: keep bottom-eff fraction among D ----
    # D selection means small pB; want fraction(pB_D < tD) = eff -> tD = quantile at eff
    tD = torch.quantile(pB_D, eff).detach()

    # penalize B that would fall into D region (false positives for D selection)
    penalty_D = F.softplus((tD - pB_B) / tau).mean()

    return penalty_B, penalty_D



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from: {args.root_file}")  
    dataset = HFSemiClassifier(
        args.root_file,
        tree_name="tree",
        use_log_pt=False,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_abs_max=5.0,
        use_had_eta=True,
        had_pt_min=0.2,
        had_pt_max=None,
        min_had=4,
    )

    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"[INFO] Total electrons: {n_total}, train: {n_train}, val: {n_val}")

    # ===== 下采样用的pt bins=====
    pt_edges = parse_pt_edges(args)
    print(f"[INFO] Downsample pt edges: {pt_edges.tolist()}")

    # 预先构建 (ptbin, class)->indices 映射（一次性）
    train_idx_map = build_ptbin_class_index(train_set, dataset, pt_edges, num_classes=2)
    val_idx_map   = build_ptbin_class_index(val_set, dataset, pt_edges, num_classes=2)

    # 打印每个bin的计数（可选但强烈建议）
    n_bins = len(pt_edges) - 1
    for b in range(n_bins):
        nD = len(train_idx_map[(b,0)])
        nB = len(train_idx_map[(b,1)])
        print(f"[INFO] Train bin {pt_edges[b]:.2f}-{pt_edges[b+1]:.2f}: D={nD}, B={nB}, keep(each)={min(nD,nB)}")


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

    # model = PointNetHF(
    #     had_input_dim=5,
    #     ele_input_dim=3,
    #     point_hidden_dims=(128, 128, 256),
    #     point_embed_dim=256,
    #     clf_hidden_dims=(256, 256),
    #     n_classes=2,
    #     use_ele_in_point_encoder=True,  # 等价于你 DeepSets 里的 use_ele_in_had_encoder
    #     use_ele_feat=True,
    #     pooling="max",                  # PointNet 最常用 max；你也可试 mean/sum
    # ).to(device)


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

        # ===== 每个epoch重新按(ptbin内)平衡下采样 =====
        g = torch.Generator()
        g.manual_seed(12345 + epoch)  # 或者你加个args.seed
        # g.manual_seed(12345)  # 或者你加个args.seed

        train_epoch_set = resample_balanced_by_ptbin(
            train_set, train_idx_map, pt_edges, generator=g, num_classes=2
        )
        val_epoch_set = resample_balanced_by_ptbin(
            val_set, val_idx_map, pt_edges, generator=g, num_classes=2
        )

        train_loader = DataLoader(
            train_epoch_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=hf_semi_collate,
            pin_memory=True if device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_epoch_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=hf_semi_collate,
            pin_memory=True if device.type == "cuda" else False,
        )

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

            # ===== NEW: operating-point penalties (route2) =====
            pB = torch.softmax(logits, dim=-1)[:, 1]  # probability of class B
            pen_B, pen_D = op_point_penalty(pB, labels, eff=args.op_eff, tau=args.op_tau)
            op_penalty = 0.5 * (pen_B + pen_D)

            loss = base_loss + args.fair_lambda * fairness_penalty + args.op_lambda * op_penalty
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

        all_pB = []
        all_y  = []

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

                # collect pB and y for operating-point evaluation later
                pB = torch.softmax(logits, dim=-1)[:, 1]
                mask_db = (labels == 0) | (labels == 1)
                all_pB.append(pB[mask_db].detach().cpu())
                all_y.append(labels[mask_db].detach().cpu())


                if mask_D.any() and mask_B.any():
                    loss_D = per_sample_loss[mask_D].mean()
                    loss_B = per_sample_loss[mask_B].mean()
                    fairness_penalty = (loss_D - loss_B) ** 2
                else:
                    fairness_penalty = torch.tensor(0.0, device=device)

                pB = torch.softmax(logits, dim=-1)[:, 1]
                pen_B, pen_D = op_point_penalty(pB, labels, eff=args.op_eff, tau=args.op_tau)
                op_penalty = 0.5 * (pen_B + pen_D)

                loss = base_loss + args.fair_lambda * fairness_penalty + args.op_lambda * op_penalty

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

        # if epochs_no_improve >= args.patience:
        #     print(f"[INFO] Early stopping triggered after {args.patience} epochs with no improvement.")
        #     break

        if len(all_pB) == 0:
            print("[WARN] No D/B samples collected for purity calc this epoch.")
        else:
            pB_all = torch.cat(all_pB)
            y_all  = torch.cat(all_y)

        maskB = (y_all == 1)
        maskD = (y_all == 0)

        # B operating point: keep top eff among true B
        tB = torch.quantile(pB_all[maskB], 1.0 - args.op_eff)
        selB = (pB_all >= tB)
        purB = (y_all[selB] == 1).float().mean().item()

        # D operating point: keep bottom eff among true D (small pB)
        tD = torch.quantile(pB_all[maskD], args.op_eff)
        selD = (pB_all <= tD)
        purD = (y_all[selD] == 0).float().mean().item()

        print(f"    Purity@eff={args.op_eff:.2f}:  B_pur={purB:.4f}   D_pur={purD:.4f}")

        # save best model
        if epoch >= start_save_epoch and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            pt_min_str = f"{args.pt_min:.1f}" if args.pt_min is not None else "None"
            pt_max_str = f"{args.pt_max:.1f}" if args.pt_max is not None else "None"

            had_arch_str = f"had{len(had_hidden_dims)}x{had_hidden_dims[0]}"
            clf_arch_str = f"clf{len(clf_hidden_dims)}x{clf_hidden_dims[0]}"
            arch_str = f"{had_arch_str}_{clf_arch_str}_{pooling}"

            best_name = f"DeepSetsHF_best_ALL_{pt_min_str}-{pt_max_str}_{arch_str}_M4.pt"
            # best_name = f"PointnetsHF_best_ALL_{pt_min_str}-{pt_max_str}_{arch_str}_M4.pt"
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
    # epochs = range(1, args.epochs + 1)
    epochs = range(1, len(train_loss_history) + 1) # for early stopping case
    plt.figure()
    plt.plot(epochs, train_loss_history, label="train loss")
    plt.plot(epochs, val_loss_history, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch (pt {args.pt_min}-{args.pt_max} GeV) | pooling={args.pooling}")
    plt.legend()
    plt.grid(True)

    loss_fig_path = os.path.join(
        args.out_dir, f"loss_curve_ALL_pt{args.pt_min}-{args.pt_max}_pool{args.pooling}.png"
    )
    plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Loss curve saved to: {loss_fig_path}")


if __name__ == "__main__":
    main()
