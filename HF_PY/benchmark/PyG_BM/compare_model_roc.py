#!/usr/bin/env python3
# compare_model_roc.py
#
# Overlay ROC curves from multiple trained models on the same evaluation set.
#
# Example:
# python compare_model_roc.py \
#   --model "DeepSets|deepset|/path/to/deepset.pt" \
#   --model "Transformer|transformer|/path/to/transformer.pt" \
#   --model "GNN(k=8)|gnn|/path/to/gnn_k8.pt" \
#   --root-file /path/to/data.root \
#   --pt-min 3 --pt-max 10 \
#   --balance-ds \
#   --out-prefix ./roc_compare/model_cmp
#
# model spec format:
#   --model "LABEL|TYPE|CKPT"
# where TYPE in:
#   deepset, pointnet, transformer, gnn

"""
python compare_model_roc.py \
  --model "DeepSets-sum|deepset|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt" \
  --model "Transformer-L4|transformer|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt" \
  --model "GNN-k8|gnn|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/gnn/gnnHF_best_ALL_3.0-10.0_k8.pt" \
  --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
  --pt-min 3.0 \
  --pt-max 10.0 \
  --balance-ds \
  --out-prefix /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Replot/model_compare
"""


import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF, PointNetHF, SetTransformerHF, GNNHF_EdgeConv

# ----------------------------- utils -----------------------------
def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_model_specs(items: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parse:
      LABEL|TYPE|CKPT
    """
    out = []
    for s in items:
        parts = [x.strip() for x in s.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"Bad --model spec: {s}\n"
                f"Expected format: LABEL|TYPE|CKPT"
            )
        label, model_type, ckpt = parts
        model_type = model_type.lower()
        if model_type not in ("deepset", "pointnet", "transformer", "gnn"):
            raise ValueError(f"Unsupported model type: {model_type}")
        out.append((label, model_type, ckpt))
    return out


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    """
    y_true in {0,1}, y_score higher => more likely class 1.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0]) if P > 0 else np.array([0.0, 0.0])
        thr = np.array([np.inf, -np.inf])
        return fpr, tpr, thr, float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / P
    fpr = fps / N
    thr = y_score

    auc = np.trapz(tpr, fpr)
    return fpr, tpr, thr, float(auc)


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
        edges = [float(x) for x in ds_pt_edges.split(",") if x.strip() != ""]
        edges = sorted(edges)
        if len(edges) < 2:
            raise ValueError("ds-pt-edges must have >=2 numbers")
        return np.array(edges, dtype=np.float64)

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

        # base_keep = min(len(pools[0]), len(pools[1]))
        base_keep = min(len(pools[0]), len(pools[1]), 500)
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


# ----------------------------- model build -----------------------------
def build_model(model_type: str, ckpt_args: dict):
    if model_type == "deepset":
        pooling = ckpt_args.get("pooling", "sum")
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
        )
        return model

    if model_type == "pointnet":
        model = PointNetHF(
            had_input_dim=5,
            ele_input_dim=3,
            point_hidden_dims=(128, 128, 256),
            point_embed_dim=256,
            clf_hidden_dims=(256, 256),
            n_classes=2,
            use_ele_in_point_encoder=True,
            use_ele_feat=True,
            pooling="max",
        )
        return model

    if model_type == "transformer":
        model = SetTransformerHF(
            had_input_dim=5,
            ele_input_dim=3,
            d_model=256,
            nhead=4,
            num_layers=int(ckpt_args.get("num_layers", 2)),
            dim_feedforward=512,
            dropout=0.1,
            n_classes=2,
        )
        return model

    if model_type == "gnn":
        model = GNNHF_EdgeConv(
            had_input_dim=5,
            ele_input_dim=3,
            hidden_dim=128,
            num_layers=3,
            k=int(ckpt_args.get("num_gnn_k", ckpt_args.get("num-gnn-k", 4))),
            pooling="sum",
            n_classes=2,
            use_ele_in_node=True,
            pos_mode="deta_sincos",
            dropout=0.1,
        )
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def evaluate_one_model(
    label: str,
    model_type: str,
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    model = build_model(model_type, ckpt_args).to(device)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path} missing 'model_state_dict'")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_all = []
    pB_all = []

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ele, had, msk)
        probs = torch.softmax(logits, dim=-1)

        y_all.append(y.detach().cpu().numpy().astype(np.int64))
        pB_all.append(probs[:, 1].detach().cpu().numpy())

    y_all = np.concatenate(y_all, axis=0)
    pB_all = np.concatenate(pB_all, axis=0)

    fpr, tpr, _, auc = compute_roc_auc(y_all, pB_all)
    return {
        "label": label,
        "model_type": model_type,
        "ckpt": ckpt_path,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Compare ROC curves from multiple models.")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help='Repeatable. Format: "LABEL|TYPE|CKPT"',
    )

    p.add_argument("--root-file", type=str, required=True)
    p.add_argument("--tree-name", type=str, default="tree")

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda/cpu/empty=auto")
    p.add_argument("--max-events", type=int, default=-1, help="<=0 means all")
    p.add_argument("--seed", type=int, default=12345)

    # dataset cuts
    p.add_argument("--pt-min", type=float, default=3.0)
    p.add_argument("--pt-max", type=float, default=10.0)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true")

    # balancing
    p.add_argument("--balance-ds", action="store_true")
    p.add_argument("--balance-frac", type=float, default=1.0)
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")

    p.add_argument("--out-prefix", type=str, default="./roc_compare/model_cmp")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}")

    model_specs = parse_model_specs(args.model)

    dataset = HFSemiClassifier(
        args.root_file,
        tree_name=args.tree_name,
        use_log_pt=args.use_log_pt,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_abs_max=args.eta_abs_max,
        use_had_eta=True,
        had_pt_min=args.had_pt_min,
        had_pt_max=args.had_pt_max,
        min_had=args.min_had,
    )

    max_events = None if args.max_events <= 0 else int(args.max_events)
    idx_all = pick_indices_labels01(dataset, max_events=max_events, seed=args.seed)
    print(f"[INFO] D/B-only candidates: {len(idx_all)}")

    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(args.ds_pt_edges, args.ds_pt_bin_width, args.pt_min, args.pt_max)
        idx_map = build_ptbin_class_index_from_indices(dataset, idx_all, ds_edges, num_classes=2)
        idx_bal = resample_balanced_by_ptbin_indices(
            idx_map,
            ds_edges,
            seed=args.seed,
            frac=float(args.balance_frac),
            num_classes=2,
        )
        print(f"[INFO] balance-ds: {len(idx_all)} -> {len(idx_bal)}")
        if len(idx_bal) > 0:
            idx_eval = idx_bal
        else:
            print("[WARN] balance-ds produced empty set; fallback to unbalanced sample.")
            idx_eval = idx_all
    else:
        idx_eval = idx_all

    work_set = Subset(dataset, idx_eval)
    print(f"[INFO] eval subset size: {len(work_set)}")

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    results = []
    for label, model_type, ckpt_path in model_specs:
        print(f"[INFO] evaluating: {label} | {model_type} | {ckpt_path}")
        res = evaluate_one_model(label, model_type, ckpt_path, loader, device)
        results.append(res)
        print(f"[INFO]   AUC = {res['auc']:.6f}")

    # sort by AUC descending for nicer legend / summary
    results = sorted(results, key=lambda x: x["auc"], reverse=True)

    ensure_dir_for_file(args.out_prefix + "_roc.png")

    plt.figure(figsize=(6.2, 5.2))
    for res in results:
        plt.plot(res["fpr"], res["tpr"], label=f"{res['label']} (AUC={res['auc']:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC comparison")
    plt.grid(True)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_png = args.out_prefix + "_roc.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[INFO] saved: {out_png}")

    out_txt = args.out_prefix + "_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ROC comparison summary\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"pt range = [{args.pt_min}, {args.pt_max})\n")
        f.write(f"eta_abs_max = {args.eta_abs_max}\n")
        f.write(f"had_pt_min = {args.had_pt_min}\n")
        f.write(f"had_pt_max = {args.had_pt_max}\n")
        f.write(f"min_had = {args.min_had}\n")
        f.write(f"balance_ds = {args.balance_ds}\n")
        f.write(f"eval_N = {len(work_set)}\n\n")
        for i, res in enumerate(results, 1):
            f.write(f"{i}. {res['label']}\n")
            f.write(f"   type = {res['model_type']}\n")
            f.write(f"   ckpt = {res['ckpt']}\n")
            f.write(f"   auc = {res['auc']:.6f}\n\n")
    print(f"[INFO] saved: {out_txt}")

    print("\n[SUMMARY]")
    for i, res in enumerate(results, 1):
        print(f"{i:>2d}. {res['label']:<20s} AUC = {res['auc']:.6f}")


if __name__ == "__main__":
    main()