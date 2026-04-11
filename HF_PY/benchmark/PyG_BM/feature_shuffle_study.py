#!/usr/bin/env python3
# feature_shuffle_study.py
#
# Study the impact of feature shuffling on classifier performance.
#
# Main idea:
#   - load the same dataset / checkpoint setting as your scan/explain scripts
#   - evaluate baseline performance
#   - for each requested feature group, shuffle that feature across events
#     while keeping its marginal distribution approximately unchanged
#   - compare AUC / accuracy / cross-entropy loss against baseline
#
# Default intended shuffles for this project:
#   had_pt, had_deta, had_dphi, had_charge
# Optional electron-feature shuffles are also supported.

'''
python /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/feature_shuffle_study.py \
  --ckpt /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt \
  --shuffle-features none,had_pt,had_deta,had_dphi,had_charge,ele_pt \
  --repeat-per-shuffle 3
'''

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


# ----------------------------- small utils -----------------------------
def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_edges_csv(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    vals = sorted(vals)
    if len(vals) < 2:
        raise ValueError("Need >=2 edges.")
    return np.array(vals, dtype=np.float64)


def parse_shuffle_list(s: str) -> List[str]:
    if s is None:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return parts


# ----------------------------- same subset logic as scan -----------------------------
def pick_indices_labels01(dataset: HFSemiClassifier, max_events: Optional[int], seed: int) -> List[int]:
    idx_all = []
    for i in range(len(dataset)):
        evt_idx, ele_idx = dataset.electron_index[i]
        raw_tag = int(dataset.ele_hf_TAG[evt_idx][ele_idx])
        if raw_tag in (1, 3):
            idx_all.append(i)

    if max_events is not None and max_events > 0 and len(idx_all) > max_events:
        rng = np.random.default_rng(seed)
        idx_all = rng.choice(idx_all, size=max_events, replace=False).tolist()
    return idx_all


def get_electron_pt_from_dataset(dataset: HFSemiClassifier, global_idx: int) -> float:
    evt_idx, ele_idx = dataset.electron_index[global_idx]
    return float(dataset.ele_pt[evt_idx][ele_idx])


def parse_ds_pt_edges(ds_pt_edges: str, ds_pt_bin_width: float, pt_min: Optional[float], pt_max: Optional[float]) -> np.ndarray:
    if ds_pt_edges is not None and ds_pt_edges.strip():
        return parse_edges_csv(ds_pt_edges)

    if pt_min is None or pt_max is None:
        raise ValueError("Need pt_min/pt_max to auto-build ds pt bins (or set --ds-pt-edges).")

    w = float(ds_pt_bin_width)
    if w <= 0:
        raise ValueError("--ds-pt-bin-width must be > 0")

    edges = [float(pt_min)]
    x = float(pt_min)
    while x + w < float(pt_max) - 1e-6:
        x += w
        edges.append(x)
    edges.append(float(pt_max))
    return np.array(edges, dtype=np.float64)


def build_ptbin_class_index_from_indices(dataset: HFSemiClassifier, indices: List[int], pt_edges: np.ndarray, num_classes: int = 2):
    n_bins = len(pt_edges) - 1
    idx_map = {(b, c): [] for b in range(n_bins) for c in range(num_classes)}

    for gidx in indices:
        evt_idx, ele_idx = dataset.electron_index[gidx]
        raw_tag = int(dataset.ele_hf_TAG[evt_idx][ele_idx])
        if raw_tag == 1:
            y = 0
        elif raw_tag == 3:
            y = 1
        else:
            continue

        pt = get_electron_pt_from_dataset(dataset, gidx)
        b = int(np.searchsorted(pt_edges, pt, side="right") - 1)
        if 0 <= b < n_bins:
            idx_map[(b, y)].append(gidx)
    return idx_map


def resample_balanced_by_ptbin_indices(idx_map, pt_edges: np.ndarray, seed: int = 12345, frac: float = 1.0, num_classes: int = 2) -> List[int]:
    rng = np.random.default_rng(seed)
    n_bins = len(pt_edges) - 1
    selected: List[int] = []

    for b in range(n_bins):
        pools = [idx_map[(b, c)] for c in range(num_classes)]
        if any(len(p) == 0 for p in pools):
            continue

        base_keep = min(len(pools[0]), len(pools[1]))
        n_keep = int(np.floor(frac * base_keep))
        if n_keep <= 0:
            continue

        for c in range(num_classes):
            chosen = rng.choice(pools[c], size=n_keep, replace=False).tolist()
            selected.extend(chosen)

    if len(selected) == 0:
        return []
    rng.shuffle(selected)
    return selected


# ----------------------------- feature shuffle -----------------------------
ALLOWED_SHUFFLES = {
    "none",
    "had_pt",
    "had_deta",
    "had_dphi",
    "had_charge",
    "ele_pt",
    "ele_eta",
    "ele_charge",
}


def apply_feature_shuffle(
    ele_feat: torch.Tensor,
    had_feat: torch.Tensor,
    had_mask: torch.Tensor,
    shuffle_feature: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffle selected feature across events (batch dimension), while keeping
    within-event particle ordering and mask unchanged.

    This keeps the feature marginal distribution approximately unchanged but
    destroys its event-label association.
    """
    if shuffle_feature == "none":
        return ele_feat, had_feat

    if shuffle_feature not in ALLOWED_SHUFFLES:
        raise ValueError(f"Unsupported shuffle_feature: {shuffle_feature}")

    ele_new = ele_feat.clone()
    had_new = had_feat.clone()

    B = ele_feat.shape[0]
    if B <= 1:
        return ele_new, had_new

    perm = torch.randperm(B, generator=generator, device=ele_feat.device)

    if shuffle_feature == "had_pt":
        had_new[:, :, 0] = had_feat[perm, :, 0]
    elif shuffle_feature == "had_deta":
        had_new[:, :, 1] = had_feat[perm, :, 1]
    elif shuffle_feature == "had_dphi":
        had_new[:, :, 2] = had_feat[perm, :, 2]
        had_new[:, :, 3] = had_feat[perm, :, 3]
    elif shuffle_feature == "had_charge":
        had_new[:, :, 4] = had_feat[perm, :, 4]
    elif shuffle_feature == "ele_pt":
        ele_new[:, 0] = ele_feat[perm, 0]
    elif shuffle_feature == "ele_eta":
        ele_new[:, 1] = ele_feat[perm, 1]
    elif shuffle_feature == "ele_charge":
        ele_new[:, 2] = ele_feat[perm, 2]

    # keep padded positions zero-ish for hadron features
    if had_new.numel() > 0:
        had_new = torch.where(had_mask.unsqueeze(-1), had_new, torch.zeros_like(had_new))

    return ele_new, had_new


# ----------------------------- evaluation -----------------------------
@torch.no_grad()
def evaluate_one_setting(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    shuffle_feature: str,
    seed: int,
) -> Dict[str, float]:
    logits_all = []
    probs_all = []
    labels_all = []

    gen = torch.Generator(device=device.type if device.type in ("cpu", "cuda") else "cpu")
    gen.manual_seed(seed)

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        if shuffle_feature != "none":
            ele, had = apply_feature_shuffle(ele, had, msk, shuffle_feature, generator=gen)

        logits = model(ele, had, msk, return_attn=False)
        probs = F.softmax(logits, dim=1)

        logits_all.append(logits.detach().cpu())
        probs_all.append(probs.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0).numpy()
    probs_all = torch.cat(probs_all, dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy().astype(np.int64)
    preds_all = np.argmax(probs_all, axis=1)

    loss = float(log_loss(labels_all, probs_all, labels=[0, 1]))
    acc = float(accuracy_score(labels_all, preds_all))

    try:
        auc = float(roc_auc_score(labels_all, probs_all[:, 1]))
    except ValueError:
        auc = np.nan

    return {
        "shuffle_feature": shuffle_feature,
        "n": int(labels_all.size),
        "n_D": int(np.sum(labels_all == 0)),
        "n_B": int(np.sum(labels_all == 1)),
        "auc": auc,
        "accuracy": acc,
        "loss": loss,
    }


def make_metric_plots(rows: List[Dict[str, float]], out_prefix: str) -> None:
    names = [r["shuffle_feature"] for r in rows]
    aucs = [r["auc"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    losses = [r["loss"] for r in rows]

    # AUC
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(np.arange(len(names)), aucs)
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right")
    plt.ylabel("AUC")
    plt.title("Feature shuffle impact on AUC")
    plt.tight_layout()
    plt.savefig(out_prefix + "_auc.png", dpi=180)
    plt.close()

    # Accuracy
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(np.arange(len(names)), accs)
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Feature shuffle impact on accuracy")
    plt.tight_layout()
    plt.savefig(out_prefix + "_accuracy.png", dpi=180)
    plt.close()

    # Loss
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(np.arange(len(names)), losses)
    plt.xticks(np.arange(len(names)), names, rotation=30, ha="right")
    plt.ylabel("Cross-entropy loss")
    plt.title("Feature shuffle impact on loss")
    plt.tight_layout()
    plt.savefig(out_prefix + "_loss.png", dpi=180)
    plt.close()

    # --- ΔAUC horizontal bar plot (recommended for paper) ---
    baseline = rows[0]
    names = [r["shuffle_feature"] for r in rows[1:]]
    delta_auc = [r["auc"] - baseline["auc"] for r in rows[1:]]
    delta_auc_std = [r["auc_std"] for r in rows[1:]]
    # 排序：从最负（影响最大）到最不负
    order = np.argsort(delta_auc)
    names = [names[i] for i in order]
    delta_auc = [delta_auc[i] for i in order]
    delta_auc_std = [delta_auc_std[i] for i in order]
    y = np.arange(len(names))

    name_map = {
        "had_pt": r"$p_T^{\mathrm{had}}$",
        "had_deta": r"$\Delta\eta$",
        "had_dphi": r"$\Delta\phi$",
        "had_charge": "charge"
    }

    names = [name_map.get(n, n) for n in names]

    plt.figure(figsize=(6.5, 4.2))  # 更紧凑一点，适合 paper
    plt.barh(y, delta_auc, xerr=delta_auc_std, capsize=3)
    # 画 baseline 参考线
    plt.axvline(0.0, linewidth=1)
    plt.yticks(y, names)
    plt.xlabel(r"$\Delta$AUC relative to baseline")
    plt.title("Impact of feature shuffling on classification performance")
    # 去掉多余边框（更像 PRC 风格）
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_prefix + "_delta_auc.png", dpi=300)
    plt.close()


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Study feature shuffle impact on AUC / accuracy / loss.")
    p.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt",
        help="Path to model checkpoint (.pt) saved by train script.",
    )
    p.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_2_allAccept.root",
        help="Override dataset ROOT file (default: dataset 2)."
    )
    p.add_argument("--tree-name", type=str, default="tree")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda/cpu/empty=auto")
    p.add_argument("--max-events", type=int, default=-1, help="<=0 means all (after D/B filtering, before balancing)")
    p.add_argument("--seed", type=int, default=12345)

    # dataset cuts override
    p.add_argument("--pt-min", type=float, default=None)
    p.add_argument("--pt-max", type=float, default=None)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true", help="Force use_log_pt=True (else follow ckpt if present).")

    # downsample
    p.add_argument("--balance-ds", action="store_true", default=True,
                   help="Build subset by train-style per-pt-bin D/B 1:1 balancing")
    p.add_argument("--no-balance-ds", dest="balance_ds", action="store_false",
                   help="Disable per-pt-bin D/B balancing")
    p.add_argument("--balance-frac", type=float, default=1.0,
                   help="n_keep = floor(frac * min(nD,nB)) in each ds bin.")
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")

    # shuffle setup
    p.add_argument(
        "--shuffle-features",
        type=str,
        default="none,had_pt,had_deta,had_dphi,had_charge",
        help=(
            "Comma-separated feature groups to evaluate. Supported: "
            "none,had_pt,had_deta,had_dphi,had_charge,ele_pt,ele_eta,ele_charge"
        ),
    )
    p.add_argument(
        "--repeat-per-shuffle",
        type=int,
        default=1,
        help="Repeat each shuffled setting several times with different seeds and average metrics.",
    )

    # outputs
    p.add_argument("--out-prefix", type=str, default="",
                   help="Prefix for outputs. If empty, derive from ckpt name.")

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # device
    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}")

    shuffle_features = parse_shuffle_list(args.shuffle_features)
    if len(shuffle_features) == 0:
        raise ValueError("No shuffle features requested.")
    for sf in shuffle_features:
        if sf not in ALLOWED_SHUFFLES:
            raise ValueError(f"Unsupported shuffle feature: {sf}")
    if shuffle_features[0] != "none":
        print("[WARN] First requested setting is not 'none'. Baseline will still be evaluated first.")
        shuffle_features = ["none"] + [x for x in shuffle_features if x != "none"]

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    pooling = ckpt_args.get("pooling", "sum")

    root_file = args.root_file.strip() or ckpt_args.get("root_file", "")
    if root_file == "":
        raise ValueError("root-file not provided and not found in ckpt['args']['root_file'].")

    pt_min = args.pt_min if args.pt_min is not None else ckpt_args.get("pt_min", None)
    pt_max = args.pt_max if args.pt_max is not None else ckpt_args.get("pt_max", None)
    use_log_pt = bool(args.use_log_pt) or bool(ckpt_args.get("use_log_pt", False))

    print("[INFO] dataset:")
    print("   root_file =", root_file)
    print("   pt_min/pt_max =", pt_min, pt_max)
    print("   use_log_pt =", use_log_pt)

    dataset = HFSemiClassifier(
        root_file,
        tree_name=args.tree_name,
        use_log_pt=use_log_pt,
        pt_min=pt_min,
        pt_max=pt_max,
        eta_abs_max=args.eta_abs_max,
        use_had_eta=True,
        had_pt_min=args.had_pt_min,
        had_pt_max=args.had_pt_max,
        min_had=args.min_had,
    )

    # choose indices (D/B only)
    max_events = None if args.max_events <= 0 else int(args.max_events)
    idx_all = pick_indices_labels01(dataset, max_events=max_events, seed=args.seed)
    print(f"[INFO] D/B-only candidates: {len(idx_all)}")

    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(args.ds_pt_edges, args.ds_pt_bin_width, pt_min, pt_max)
        idx_map = build_ptbin_class_index_from_indices(dataset, idx_all, ds_edges, num_classes=2)
        idx_bal = resample_balanced_by_ptbin_indices(
            idx_map, ds_edges, seed=args.seed, frac=float(args.balance_frac), num_classes=2
        )
        print(f"[INFO] balance-ds: {len(idx_all)} -> {len(idx_bal)}")
        if len(idx_bal) > 0:
            idx = idx_bal
        else:
            print("[WARN] balance-ds produced empty set; fallback to unbalanced idx_all.")
            idx = idx_all
    else:
        idx = idx_all

    work_set = Subset(dataset, idx)
    print(f"[INFO] work_set size: {len(work_set)}")

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    # model
    model = DeepSetsHF(
        had_input_dim=5,
        ele_input_dim=3,
        had_hidden_dims=(128, 128, 128),
        set_embed_dim=128,
        clf_hidden_dims=(128, 128, 128),
        n_classes=2,
        use_ele_in_had_encoder=True,
        use_ele_feat=True,
        pooling=pooling,
    ).to(device)

    if "model_state_dict" not in ckpt:
        raise KeyError("ckpt missing 'model_state_dict'.")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # output prefix
    if args.out_prefix.strip():
        prefix = args.out_prefix.strip()
    else:
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        prefix = os.path.join("/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/feature_shuffle/", base)
    ensure_dir_for_file(prefix + "_dummy.txt")
    print("[INFO] out prefix =", prefix)

    # evaluate baseline + shuffles
    rows: List[Dict[str, float]] = []
    for sf in shuffle_features:
        rep_results = []
        n_repeat = 1 if sf == "none" else max(1, int(args.repeat_per_shuffle))
        for ir in range(n_repeat):
            rep_seed = int(args.seed + 1000 * ir + 17)
            res = evaluate_one_setting(model, loader, device, shuffle_feature=sf, seed=rep_seed)
            rep_results.append(res)

        row = {
            "shuffle_feature": sf,
            "n": rep_results[0]["n"],
            "n_D": rep_results[0]["n_D"],
            "n_B": rep_results[0]["n_B"],
            "auc": float(np.mean([r["auc"] for r in rep_results])),
            "accuracy": float(np.mean([r["accuracy"] for r in rep_results])),
            "loss": float(np.mean([r["loss"] for r in rep_results])),
            "auc_std": float(np.std([r["auc"] for r in rep_results])) if len(rep_results) > 1 else 0.0,
            "accuracy_std": float(np.std([r["accuracy"] for r in rep_results])) if len(rep_results) > 1 else 0.0,
            "loss_std": float(np.std([r["loss"] for r in rep_results])) if len(rep_results) > 1 else 0.0,
            "n_repeat": int(n_repeat),
        }
        rows.append(row)
        print(
            f"[INFO] {sf:>10s}: "
            f"AUC={row['auc']:.6f}, ACC={row['accuracy']:.6f}, LOSS={row['loss']:.6f}"
            + (f"  (repeat={n_repeat})" if n_repeat > 1 else "")
        )

    baseline = rows[0]
    for row in rows:
        row["delta_auc"] = float(row["auc"] - baseline["auc"])
        row["delta_accuracy"] = float(row["accuracy"] - baseline["accuracy"])
        row["delta_loss"] = float(row["loss"] - baseline["loss"])

    # save csv
    csv_path = prefix + "_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shuffle_feature",
                "n",
                "n_D",
                "n_B",
                "auc",
                "auc_std",
                "delta_auc",
                "accuracy",
                "accuracy_std",
                "delta_accuracy",
                "loss",
                "loss_std",
                "delta_loss",
                "n_repeat",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[INFO] saved:", csv_path)

    # save npz for later plotting if needed
    npz_path = prefix + "_metrics.npz"
    np.savez_compressed(
        npz_path,
        shuffle_feature=np.array([r["shuffle_feature"] for r in rows], dtype=object),
        auc=np.array([r["auc"] for r in rows], dtype=np.float64),
        auc_std=np.array([r["auc_std"] for r in rows], dtype=np.float64),
        delta_auc=np.array([r["delta_auc"] for r in rows], dtype=np.float64),
        accuracy=np.array([r["accuracy"] for r in rows], dtype=np.float64),
        accuracy_std=np.array([r["accuracy_std"] for r in rows], dtype=np.float64),
        delta_accuracy=np.array([r["delta_accuracy"] for r in rows], dtype=np.float64),
        loss=np.array([r["loss"] for r in rows], dtype=np.float64),
        loss_std=np.array([r["loss_std"] for r in rows], dtype=np.float64),
        delta_loss=np.array([r["delta_loss"] for r in rows], dtype=np.float64),
        n=np.array([r["n"] for r in rows], dtype=np.int64),
        n_D=np.array([r["n_D"] for r in rows], dtype=np.int64),
        n_B=np.array([r["n_B"] for r in rows], dtype=np.int64),
        n_repeat=np.array([r["n_repeat"] for r in rows], dtype=np.int64),
        ckpt=args.ckpt,
        root_file=root_file,
        pooling=pooling,
        use_log_pt=use_log_pt,
        pt_min=pt_min if pt_min is not None else -1.0,
        pt_max=pt_max if pt_max is not None else -1.0,
        balance_ds=bool(args.balance_ds),
        balance_frac=float(args.balance_frac),
        ds_pt_bin_width=float(args.ds_pt_bin_width),
        ds_pt_edges=str(args.ds_pt_edges),
    )
    print("[INFO] saved:", npz_path)

    # plots
    make_metric_plots(rows, prefix)
    print("[INFO] saved:", prefix + "_auc.png")
    print("[INFO] saved:", prefix + "_accuracy.png")
    print("[INFO] saved:", prefix + "_loss.png")
    print("[INFO] saved:", prefix + "_delta_auc.png")
    print("[INFO] saved:", prefix + "_delta_accuracy.png")
    print("[INFO] saved:", prefix + "_delta_loss.png")

    print("\n[SUMMARY]")
    for row in rows:
        print(
            f"  {row['shuffle_feature']:>10s} | "
            f"AUC={row['auc']:.6f} ({row['delta_auc']:+.6f}) | "
            f"ACC={row['accuracy']:.6f} ({row['delta_accuracy']:+.6f}) | "
            f"LOSS={row['loss']:.6f} ({row['delta_loss']:+.6f})"
        )


if __name__ == "__main__":
    main()
