#!/usr/bin/env python3
# compare_model_auc_vs_ptbin.py
#
# Compare AUC values across electron pT bins for multiple models.
#
# x-axis  : electron pT bins (categorical labels)
# y-axis  : AUC in each pT bin
#
# Main features:
# - evaluate multiple checkpoints on the same evaluation subset
# - optional train-style per-pt-bin D/B balancing
# - optional cap inside each balancing bin: base_keep=min(nD, nB, 500)
# - plot AUC vs pT bin with categorical x-axis
# - save summary txt/csv
#
# model spec format:
#   --model "LABEL|TYPE|CKPT"
# where TYPE in:
#   deepset, pointnet, transformer, gnn

"""
Example:

python compare_model_auc_vs_ptbin.py \
  --model "DeepSets-sum|deepset|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt" \
  --model "Transformer-L4|transformer|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt" \
  --model "GNN-k8|gnn|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/gnn/gnnHF_best_ALL_3.0-10.0_k8.pt" \
  --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
  --pt-min 3.0 \
  --pt-max 10.0 \
  --pt-edges "3,4,5,6,7,8" \
  --balance-ds \
  --max-keep-per-bin-per-class 99999 \
  --out-prefix /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Replot/model_auc_vs_ptbin
"""

import os
import csv
import argparse
from typing import List, Tuple, Optional, Dict

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
    out = []
    for s in items:
        parts = [x.strip() for x in s.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"Bad --model spec: {s}\n"
                f'Expected format: "LABEL|TYPE|CKPT"'
            )
        label, model_type, ckpt = parts
        model_type = model_type.lower()
        if model_type not in ("deepset", "pointnet", "transformer", "gnn"):
            raise ValueError(f"Unsupported model type: {model_type}")
        out.append((label, model_type, ckpt))
    return out


def parse_edges(s: str) -> np.ndarray:
    if s is None or str(s).strip() == "":
        return np.array([3.0, 4.0, 6.0, 8.0, 1e9], dtype=np.float64)
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    vals = sorted(vals)
    if len(vals) < 2:
        raise ValueError("pt-edges must have at least 2 numbers, e.g. '3,4,5,6,7,8'")
    return np.array(vals, dtype=np.float64)


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


def resample_balanced_by_ptbin_indices(
    idx_map,
    pt_edges: np.ndarray,
    seed: int = 12345,
    frac: float = 1.0,
    num_classes: int = 2,
    max_keep_per_bin_per_class: int = 500,
) -> List[int]:
    rng = np.random.default_rng(seed)
    n_bins = len(pt_edges) - 1
    selected: List[int] = []

    for b in range(n_bins):
        pools = [idx_map[(b, c)] for c in range(num_classes)]
        if any(len(p) == 0 for p in pools):
            continue

        base_keep = min(len(pools[0]), len(pools[1]), max_keep_per_bin_per_class)
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
    use_log_pt: bool,
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
    pt_all = []

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ele, had, msk)
        probs = torch.softmax(logits, dim=-1)

        y_all.append(y.detach().cpu().numpy().astype(np.int64))
        pB_all.append(probs[:, 1].detach().cpu().numpy())

        pt_feat = ele[:, 0].detach().cpu().numpy()
        pt = np.exp(pt_feat) if use_log_pt else pt_feat
        pt_all.append(pt)

    y_all = np.concatenate(y_all, axis=0)
    pB_all = np.concatenate(pB_all, axis=0)
    pt_all = np.concatenate(pt_all, axis=0)

    return {
        "label": label,
        "model_type": model_type,
        "ckpt": ckpt_path,
        "y": y_all,
        "pB": pB_all,
        "pt": pt_all,
    }


def compute_auc_by_ptbin(y: np.ndarray, pB: np.ndarray, pt: np.ndarray, pt_edges: np.ndarray, min_entries: int = 50):
    n_bins = len(pt_edges) - 1
    auc_bins = []
    counts = []
    labels = []

    for b in range(n_bins):
        lo, hi = pt_edges[b], pt_edges[b + 1]
        m = (pt >= lo) & (pt < hi)
        counts.append(int(m.sum()))
        labels.append(f"{lo:g}-{hi:g}")

        if int(m.sum()) < min_entries:
            auc_bins.append(np.nan)
            continue

        _, _, _, auc_b = compute_roc_auc(y[m], pB[m])
        auc_bins.append(float(auc_b))

    return np.array(auc_bins, dtype=np.float64), np.array(counts, dtype=np.int64), labels


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Compare AUC vs electron pT bin for multiple models.")
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
    p.add_argument("--max-keep-per-bin-per-class", type=int, default=500)

    # auc-vs-ptbin
    p.add_argument("--pt-edges", type=str, default="3,4,5,6,7,8")
    p.add_argument("--min-bin-count", type=int, default=50)

    p.add_argument("--out-prefix", type=str, default="./roc_compare/model_auc_vs_ptbin")
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
        idx_eval = resample_balanced_by_ptbin_indices(
            idx_map,
            ds_edges,
            seed=args.seed,
            frac=float(args.balance_frac),
            num_classes=2,
            max_keep_per_bin_per_class=int(args.max_keep_per_bin_per_class),
        )
        print(f"[INFO] balance-ds: {len(idx_all)} -> {len(idx_eval)}")
        if len(idx_eval) == 0:
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

    pt_edges = parse_edges(args.pt_edges)

    results = []
    for label, model_type, ckpt_path in model_specs:
        print(f"[INFO] evaluating: {label} | {model_type} | {ckpt_path}")
        res = evaluate_one_model(
            label=label,
            model_type=model_type,
            ckpt_path=ckpt_path,
            loader=loader,
            device=device,
            use_log_pt=bool(args.use_log_pt),
        )

        auc_bins, counts, bin_labels = compute_auc_by_ptbin(
            y=res["y"],
            pB=res["pB"],
            pt=res["pt"],
            pt_edges=pt_edges,
            min_entries=int(args.min_bin_count),
        )

        res["auc_bins"] = auc_bins
        res["counts"] = counts
        res["bin_labels"] = bin_labels
        results.append(res)

        print("[INFO]   AUC by pt bin:")
        for lab, n, auc in zip(bin_labels, counts, auc_bins):
            auc_str = f"{auc:.4f}" if np.isfinite(auc) else "nan"
            print(f"[INFO]     {lab:>10s} | n={int(n):>5d} | AUC={auc_str}")

    ensure_dir_for_file(args.out_prefix + "_auc_vs_ptbin.png")

    # ---------- plot ----------
    x = np.arange(len(pt_edges) - 1)

    plt.figure(figsize=(6.6, 4.8))
    for res in results:
        plt.plot(x, res["auc_bins"], marker="o", label=res["label"])

    # annotate first model counts on x tick labels
    # tick_labels = []
    # counts_ref = results[0]["counts"] if len(results) > 0 else np.zeros_like(x)
    # bin_labels_ref = results[0]["bin_labels"] if len(results) > 0 else [str(i) for i in x]
    # for lab, n in zip(bin_labels_ref, counts_ref):
    #     tick_labels.append(f"{lab}\n(n={int(n)})")
    tick_labels = results[0]["bin_labels"]

    plt.xticks(x, tick_labels)
    plt.xlabel(r"electron $p_T$ bin [GeV]")
    plt.ylabel("AUC")
    plt.title("AUC vs electron pT bin")
    plt.grid(True)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_png = args.out_prefix + "_auc_vs_ptbin.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[INFO] saved: {out_png}")

    # ---------- csv ----------
    out_csv = args.out_prefix + "_auc_vs_ptbin.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_label", "model_type", "pt_bin", "n", "auc", "ckpt"])
        for res in results:
            for lab, n, auc in zip(res["bin_labels"], res["counts"], res["auc_bins"]):
                writer.writerow([
                    res["label"],
                    res["model_type"],
                    lab,
                    int(n),
                    "" if not np.isfinite(auc) else f"{auc:.6f}",
                    res["ckpt"],
                ])
    print(f"[INFO] saved: {out_csv}")

    # ---------- txt summary ----------
    out_txt = args.out_prefix + "_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("AUC vs electron pT bin summary\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"pt range = [{args.pt_min}, {args.pt_max})\n")
        f.write(f"pt_edges = {pt_edges.tolist()}\n")
        f.write(f"eta_abs_max = {args.eta_abs_max}\n")
        f.write(f"had_pt_min = {args.had_pt_min}\n")
        f.write(f"had_pt_max = {args.had_pt_max}\n")
        f.write(f"min_had = {args.min_had}\n")
        f.write(f"balance_ds = {args.balance_ds}\n")
        f.write(f"balance_frac = {args.balance_frac}\n")
        f.write(f"max_keep_per_bin_per_class = {args.max_keep_per_bin_per_class}\n")
        f.write(f"eval_N = {len(work_set)}\n\n")

        for i, res in enumerate(results, 1):
            f.write(f"{i}. {res['label']}\n")
            f.write(f"   type = {res['model_type']}\n")
            f.write(f"   ckpt = {res['ckpt']}\n")
            for lab, n, auc in zip(res["bin_labels"], res["counts"], res["auc_bins"]):
                auc_str = f"{auc:.6f}" if np.isfinite(auc) else "nan"
                f.write(f"   {lab:>10s} | n={int(n):>5d} | auc={auc_str}\n")
            f.write("\n")
    print(f"[INFO] saved: {out_txt}")


if __name__ == "__main__":
    main()