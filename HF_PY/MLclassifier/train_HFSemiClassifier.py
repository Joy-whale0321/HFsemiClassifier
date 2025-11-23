# train.py
# With HFSemiClassifier + DeepSetsHF
# weight of model will be saved to:
#   /home/jingyu/HepSimTools/DataDir/HF_PY/MLclassifier/Weight_of_Model

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF

# hyperparameters setup
def parse_args():
    parser = argparse.ArgumentParser(description="Train HF semi-leptonic electron classifier (Deep Sets).")

    parser.add_argument(
        "--root-file",
        type=str,
        default="ppHF_eXDecay.root",
        help="Pythia 生成的 ROOT 文件路径 (默认: ./ppHF_eXDecay.root)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size (默认: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数 (默认: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率 (默认: 1e-3)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 的 num_workers (默认: 0, 先保证不出错)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/jingyu/HepSimTools/DataDir/HF_PY/MLclassifier/Weight_of_Model",
        help="模型权重输出目录",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="验证集占比 (默认: 0.2)",
    )
    parser.add_argument(
        "--fair-lambda",
        type=float,
        default=0.2,
        help="平衡两类之间loss差异的正则强度 λ (默认: 0.1)",
    )

    return parser.parse_args()

# count classes in dataset
def count_classes(dataset, num_classes=2):
    """
    简单统计某个 dataset 里每个 label 的数量。
    这里默认只关心 0(D) 和 1(B)，所以 num_classes=2。
    """
    import torch
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        y = int(dataset[i]["label"])
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts

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
    dataset = HFSemiClassifier(args.root_file, tree_name="tree", use_log_pt=True)

    n_total = len(dataset) # number of total electrons
    n_val = int(n_total * args.val_frac) # number of validation electrons
    n_train = n_total - n_val # number of training electrons
    train_set, val_set = random_split(dataset, [n_train, n_val]) # random split train/val with given ratio

    print(f"[INFO] Total electrons: {n_total}, train: {n_train}, val: {n_val}")

    # count class D/B distribution in training set
    train_counts = count_classes(train_set, num_classes=2)
    n_D = train_counts[0].item() # number of D with label 0
    n_B = train_counts[1].item() # number of B with label 1
    print(f"[INFO] Train class counts: D(0) = {n_D}, B(1) = {n_B}")

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
    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=(128, 128, 128),
        set_embed_dim=128,
        clf_hidden_dims=(128, 128, 128),
        n_classes=2,
        use_ele_in_had_encoder=False,
        pooling="mean",
    ).to(device)

    # loss function and optimizer setup
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[INFO] Model constructed:")
    print(model)

    # ========== training and valid epoch loop ==========
    best_val_eff = 0.0 # for only saving the best model
    start_save_epoch = int(args.epochs * 0.2)  # start doing something after the front epochs

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- Train ----
        model.train()
        train_loss = 0.0 # cumulative loss
        train_correct = 0 # number of correct predictions
        train_total = 0 # number of total samples
        train_kept_total = 0 # number of samples kept after thresholding, cut will reduce some samples 

        # for per-class efficiency calculation
        train_class_correct = [0, 0, 0]  # for label 0,1,2
        train_class_total   = [0, 0, 0]

        for batch in train_loader:
            ele = batch["ele_feat"].to(device)      # (B, 3) electron features
            had = batch["had_feat"].to(device)      # (B, N_max, 5) hadron features
            mask = batch["had_mask"].to(device)     # (B, N_max) hadron mask
            labels = batch["label"].to(device)      # (B,) ground-truth labels

            optimizer.zero_grad()

            logits = model(ele, had, mask) # (B, 3) logits for 3 classes without softmax from DeepSetsHF
            
            # loss computation 1 for all samples, 2 for kept samples only
            # loss = criterion(logits, labels)
            
            per_sample_loss = criterion(logits, labels)        # (B,)
            # get the max probobility and corresponding predicted class 
            probs = torch.softmax(logits, dim=-1)              # (B, 2)
            max_prob, preds = probs.max(dim=-1)                # (B,)

            # setting a epoch dependent threshold to select the clear samples, avoid removing too many samples at early epochs
            t_start, t_end = 0.5, 0.5
            if args.epochs > start_save_epoch:
                alpha = (epoch - 1) / (args.epochs - 1)
            else:
                alpha = 0.01
            thr = t_start + (t_end - t_start) * alpha # threshold of max probability for removing uncertain samples

            keep_mask = (max_prob > thr) # (B,), bool mask of samples, satisfying samples as true

            # if no sample is kept, use all samples to avoid empty tensor error
            if keep_mask.sum() == 0:
                base_loss = per_sample_loss.mean()
                num_kept = labels.size(0)
                used_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                # loss on kept samples only, and count number of kept samples
                base_loss = per_sample_loss[keep_mask].mean()
                num_kept = keep_mask.sum().item() # keep_mask.sum() treats bool to int, true=1, false=0
                used_mask = keep_mask

            # ------- fairness regularization: encourage D/B loss similar -------
            # 只在本 batch 中 actually 用到的样本上统计（used_mask）
            mask_D = (labels == 0) & used_mask
            mask_B = (labels == 1) & used_mask

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

            # calculate efficiency of kept samples
            train_total += labels.size(0)
            train_kept_total += num_kept

            # train_loss += loss.item() * labels.size(0)
            train_loss += loss.item() * num_kept

            # # correct predictions number counting
            # preds = torch.argmax(logits, dim=-1)
            # train_correct += (preds == labels).sum().item()
            
            # # per-class efficiency counting
            # for c in range(3):  # 0: D, 1: B, 2: other
            #     mask_c = (labels == c)
            #     if mask_c.any():
            #         train_class_total[c]   += mask_c.sum().item() # per-class samples counting
            #         train_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item() # per-class correct counting

            if keep_mask.any():
                preds_kept  = preds[keep_mask]
                labels_kept = labels[keep_mask]

                train_correct += (preds_kept == labels_kept).sum().item()

                # per-class efficiency counting on kept samples
                for c in range(3):  # 0: D, 1: B, 2: other
                    mask_c = (labels_kept == c)
                    if mask_c.any():
                        train_class_total[c]   += mask_c.sum().item()
                        train_class_correct[c] += (preds_kept[mask_c] == labels_kept[mask_c]).sum().item()

        # avg_train_loss = train_loss / max(1, train_total)
        avg_train_loss = train_loss / max(1, train_kept_total)

        # showing per-class efficiency
        names = ["D", "B", "other"]
        for c in range(3):
            if train_class_total[c] > 0:
                eff_c = train_class_correct[c] / train_class_total[c]
            else:
                eff_c = 0.0
            print(f"    Train {names[c]} eff: {eff_c:.4f} "
                  f"({train_class_correct[c]}/{train_class_total[c]})")

        train_eff = train_correct / max(1, train_total)
        train_eff_kept = train_correct / max(1, train_kept_total)
        print(f"Epoch {epoch}: train keep efficiency = {train_eff_kept:.3f}")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0

        # 验证集统计量
        val_total   = 0  # validation all samples
        val_kept_total   = 0  # validation samples number after cut
        val_correct_all  = 0  # correct number on all samples
        val_correct_kept = 0  # correct number on kept samples

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

                logits = model(ele, had, mask)
                per_sample_loss = criterion(logits, labels)      # (B,)

                probs = torch.softmax(logits, dim=-1)            # (B, 2)
                max_prob, preds = probs.max(dim=-1)              # (B,)

                # threshold on validate dataset, same as train
                thr_val = 0.8
                keep_mask = (max_prob > thr_val)                 # (B,)

                # statistics loss / eff_on_kept / per-class on kept samples
                if keep_mask.any():
                    loss_kept = per_sample_loss[keep_mask].mean()
                    n_kept = keep_mask.sum().item()

                    # accumulate kept loss (weighted by number of kept samples)
                    val_loss += loss_kept.item() * n_kept
                    val_kept_total += n_kept

                    # correct samples number on kept subset
                    val_correct_kept += (preds[keep_mask] == labels[keep_mask]).sum().item()

                    # per-class count on kept samples
                    for c in range(3):  # 0: D, 1: B, 3: other
                        mask_c = (labels == c) & keep_mask
                        if mask_c.any():
                            val_class_total[c]   += mask_c.sum().item()
                            val_class_correct[c] += (preds[mask_c] == labels[mask_c]).sum().item()

                # statistics on all samples (regardless of whether they passed the cut)
                val_correct_all += (preds == labels).sum().item()
                val_total  += labels.size(0)

        # average loss on kept samples
        avg_val_loss = val_loss / max(1, val_kept_total)

        # overall efficiency (including samples that did not pass the cut)
        val_eff_all = val_correct_all / max(1, val_total)

        # keep efficiency: fraction of validation samples passing the cut
        val_eff = val_kept_total / max(1, val_total)

        # efficiency on kept subset only
        val_eff_kept = val_correct_kept / max(1, val_kept_total)

        # show the per-class eff_on_kept
        for c in range(3):
            if val_class_total[c] > 0:
                eff_c = val_class_correct[c] / val_class_total[c]
            else:
                eff_c = 0.0
            print(f"    Validate {names[c]} eff_on_kept: {eff_c:.4f} "
                  f"({val_class_correct[c]}/{val_class_total[c]})")

        print(f"[VAL] thr={thr_val:.2f}, eff={val_eff:.3f}, "
              f"eff_on_kept={val_eff_kept:.3f}, eff_all={val_eff_all:.3f}")

        #  Time Performance logging and model saving  
        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={avg_train_loss:.4f}, train_eff={train_eff:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_eff={val_eff:.4f}, "
            f"time={dt:.1f}s"
        )

        # only save the best one
        if epoch >= start_save_epoch and val_eff > best_val_eff:
            best_val_eff = val_eff
            best_path = os.path.join(args.out_dir, "DeepSetsHF_best.pt")
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

if __name__ == "__main__":
    main()
