#!/usr/bin/env python3
# compare_model_purxeff.py
#
# For each input model, draw one Purity-vs-Efficiency figure containing:
#   - B as signal
#   - D as signal
#
# Dataset / balancing logic follows the current compare_model_roc.py style.
# In balance-ds, per-pt-bin balanced sampling keeps:
#   base_keep = min(nD, nB, max_keep_per_bin_per_class)
#
# Extra plotting choice:
#   - only draw points with efficiency >= eff_min
#   - also set x-axis limit to [eff_min, 1.0]
#
# model spec format:
#   --model "LABEL|TYPE|CKPT"
# where TYPE in:
#   deepset, pointnet, transformer, gnn

"""
Example:

python compare_model_purxeff.py \
  --model "DeepSets-sum|deepset|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt" \
  --model "Transformer-L4|transformer|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt" \
  --model "GNN-k8|gnn|/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/gnn/gnnHF_best_ALL_3.0-10.0_k8.pt" \
  --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
  --pt-min 3.0 \
  --pt-max 10.0 \
  --balance-ds \
  --max-keep-per-bin-per-class 1000 \
  --eff-min 0.02 \
  --out-prefix /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Replot/purxeff_models/model
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


def sanitize_filename(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def purity_eff_curve(y_true: np.ndarray, score: np.ndarray):
    """
    y_true in {0,1}, score higher => more likely signal(1).
    Return:
      eff (TPR), pur (precision), thr
    """
    y_true = np.asarray(y_true).astype(np.int64)
    score = np.asarray(score).astype(np.float64)

    order = np.argsort(-score)
    y = y_true[order]
    s = score[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P = np.sum(y == 1)

    eff = tp / max(P, 1)
    pur = tp / (tp + fp + 1e-12)
    thr = s
    return eff, pur, thr


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
    s_all = []

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ele, had, msk)

        y_all.append(y.detach().cpu().numpy().astype(np.int64))
        s_all.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())

    y_all = np.concatenate(y_all, axis=0)
    s_all = np.concatenate(s_all, axis=0)

    # B as signal
    effB, purB, thrB = purity_eff_curve(y_all, s_all)

    # D as signal
    yD = 1 - y_all
    sD = -s_all
    effD, purD, thrD = purity_eff_curve(yD, sD)

    return {
        "label": label,
        "model_type": model_type,
        "ckpt": ckpt_path,
        "n": int(y_all.size),
        "n_D": int(np.sum(y_all == 0)),
        "n_B": int(np.sum(y_all == 1)),
        "effB": effB,
        "purB": purB,
        "thrB": thrB,
        "effD": effD,
        "purD": purD,
        "thrD": thrD,
    }


# ----------------------------- plotting -----------------------------
def make_one_plot(res: Dict, out_png: str, eff_min: float = 0.02) -> None:
    ensure_dir_for_file(out_png)

    eff_max = 0.95

    # ---- B as signal ----
    effB_all = np.asarray(res["effB"], dtype=np.float64)
    purB_all = np.asarray(res["purB"], dtype=np.float64)

    maskB = (
        np.isfinite(effB_all)
        & np.isfinite(purB_all)
        & (effB_all >= eff_min)
        & (effB_all <= eff_max)
    )

    effB = effB_all[maskB]
    purB = purB_all[maskB]

    # ---- D as signal ----
    effD_all = np.asarray(res["effD"], dtype=np.float64)
    purD_all = np.asarray(res["purD"], dtype=np.float64)

    maskD = (
        np.isfinite(effD_all)
        & np.isfinite(purD_all)
        & (effD_all >= eff_min)
        & (effD_all <= eff_max)
    )

    effD = effD_all[maskD]
    purD = purD_all[maskD]

    plt.figure(figsize=(5.6, 4.6))

    if effB.size > 0:
        plt.plot(
            effB,
            purB,
            label="B as signal",
            linewidth=2.0,
        )

    if effD.size > 0:
        plt.plot(
            effD,
            purD,
            label="D as signal",
            linewidth=2.0,
            linestyle="--",
        )

    plt.xlabel("Efficiency")
    plt.ylabel("Purity")
    plt.title(res["label"])
    plt.grid(True)

    # ✅ legend 放左下（更像 paper）
    plt.legend(loc="lower left")

    # 坐标轴保持完整
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Draw per-model purity-efficiency curves (B and D) for multiple models.")
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

    # plotting
    p.add_argument("--eff-min", type=float, default=0.02)

    p.add_argument("--out-prefix", type=str, default="./purxeff_models/model")
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

    results = []
    for label, model_type, ckpt_path in model_specs:
        print(f"[INFO] evaluating: {label} | {model_type} | {ckpt_path}")
        res = evaluate_one_model(
            label=label,
            model_type=model_type,
            ckpt_path=ckpt_path,
            loader=loader,
            device=device,
        )
        results.append(res)
        print(
            f"[INFO]   N={res['n']} | D={res['n_D']} | B={res['n_B']}"
        )

    # save one figure per model
    for res in results:
        safe_label = sanitize_filename(res["label"])
        out_png = f"{args.out_prefix}_{safe_label}_purxeff.png"
        make_one_plot(res, out_png, eff_min=float(args.eff_min))
        print(f"[INFO] saved: {out_png}")

    # csv summary
    out_csv = args.out_prefix + "_summary.csv"
    ensure_dir_for_file(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_label",
            "model_type",
            "n_total",
            "n_D",
            "n_B",
            "ckpt",
        ])
        for res in results:
            writer.writerow([
                res["label"],
                res["model_type"],
                res["n"],
                res["n_D"],
                res["n_B"],
                res["ckpt"],
            ])
    print(f"[INFO] saved: {out_csv}")

    # txt summary
    out_txt = args.out_prefix + "_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Purity-Efficiency per-model summary\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"pt range = [{args.pt_min}, {args.pt_max})\n")
        f.write(f"eta_abs_max = {args.eta_abs_max}\n")
        f.write(f"had_pt_min = {args.had_pt_min}\n")
        f.write(f"had_pt_max = {args.had_pt_max}\n")
        f.write(f"min_had = {args.min_had}\n")
        f.write(f"balance_ds = {args.balance_ds}\n")
        f.write(f"balance_frac = {args.balance_frac}\n")
        f.write(f"max_keep_per_bin_per_class = {args.max_keep_per_bin_per_class}\n")
        f.write(f"eff_min = {args.eff_min}\n")
        f.write(f"eval_N = {len(work_set)}\n\n")

        for i, res in enumerate(results, 1):
            f.write(f"{i}. {res['label']}\n")
            f.write(f"   type = {res['model_type']}\n")
            f.write(f"   ckpt = {res['ckpt']}\n")
            f.write(f"   N = {res['n']}\n")
            f.write(f"   D = {res['n_D']}\n")
            f.write(f"   B = {res['n_B']}\n\n")
    print(f"[INFO] saved: {out_txt}")


if __name__ == "__main__":
    main()