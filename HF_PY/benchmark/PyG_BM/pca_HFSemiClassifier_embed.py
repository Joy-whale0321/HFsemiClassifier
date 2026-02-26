#!/usr/bin/env python3
# pca_HFSemiClassifier_embed.py
#
# Standalone PCA script for HF SemiClassifier embeddings (NOT in explain script).
# - Loads ckpt + dataset
# - (optional) Builds a train/explain-style balanced subset (per-pt-bin D/B 1:1)  <-- NEW
# - Runs forward with a hook to capture classifier input embedding
# - Runs PCA (numpy SVD)
# - Plots PCA scatter (D vs B), optionally per e-pt bin (ROC-style edges)
# - Saves npz dump for downstream analysis

import os
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


# ----------------------------- small utils -----------------------------
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_edges(s: str) -> np.ndarray:
    """Parse comma-separated edges string into float array."""
    if s is None or str(s).strip() == "":
        return np.array([3.0, 4.0, 6.0, 8.0, 1e9], dtype=np.float64)
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) < 2:
        raise ValueError("pt-edges must have at least 2 numbers, e.g. '3,4,6,8,999'")
    vals = sorted(vals)
    return np.array(vals, dtype=np.float64)

def get_electron_pt_from_dataset(dataset: HFSemiClassifier, global_idx: int) -> float:
    evt_idx, ele_idx = dataset.electron_index[global_idx]
    return float(dataset.ele_pt[evt_idx][ele_idx])

def pick_indices_labels01(dataset: HFSemiClassifier, max_events: Optional[int], seed: int) -> List[int]:
    """
    Keep only label 0/1 (D/B) using raw_tag {1->D, 3->B} just like your dataset.
    Returns a list of dataset-global indices.
    """
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

def pca_fit_transform(X: np.ndarray, n_components: int = 2, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD.
    Returns:
      Z: (N, n_components) projected
      components: (n_components, D)
      explained_var_ratio: (n_components,)
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
    else:
        Xc = X

    # SVD: Xc = U S V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components, :]  # (k, D)
    Z = Xc @ comps.T              # (N, k)

    # explained variance ratio
    N = X.shape[0]
    eigvals = (S**2) / max(N - 1, 1)
    total = eigvals.sum()
    evr = (eigvals[:n_components] / total) if total > 0 else np.full((n_components,), np.nan)
    return Z, comps, evr


# ----------------------------- NEW: train/explain-style downsample (balanced) -----------------------------
def parse_ds_pt_edges(ds_pt_edges: str, ds_pt_bin_width: float, pt_min: Optional[float], pt_max: Optional[float]) -> np.ndarray:
    """
    Same logic as explain_HFSemiClassifier_bm.py:
    - if ds_pt_edges given: use it
    - else: auto-build by width covering [pt_min, pt_max]
    """
    if ds_pt_edges is not None and ds_pt_edges.strip():
        edges = [float(x) for x in ds_pt_edges.split(",") if str(x).strip() != ""]
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
    """
    Same as explain script: build map (bin, class) -> list of dataset-global indices.
    """
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
    """
    For each pt bin: keep n_keep = floor(frac * min(nD, nB)) from each class (no replacement).
    Returns shuffled dataset-global indices.
    """
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
            pool = pools[c]
            chosen = rng.choice(pool, size=n_keep, replace=False).tolist()
            selected.extend(chosen)

    if len(selected) == 0:
        return []

    rng.shuffle(selected)
    return selected


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("PCA on DeepSetsHF embeddings (standalone).")

    p.add_argument("--ckpt",
                   type=str,
                   default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt",
                   help="Path to model checkpoint (.pt).")
    p.add_argument("--root-file", type=str, default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_2_allAccept.root",
                   help="Override dataset ROOT file (else try ckpt['args']['root_file']).")
    p.add_argument("--tree-name", type=str, default="tree")
    p.add_argument("--device", type=str, default="", help="cuda / cpu / empty=auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-events", type=int, default=-1, help="<=0 means all (after filtering to D/B).")
    p.add_argument("--seed", type=int, default=12345)

    # dataset cuts (override-friendly)
    p.add_argument("--pt-min", type=float, default=None)
    p.add_argument("--pt-max", type=float, default=None)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true", help="Force use_log_pt=True (else follow ckpt if present).")

    # ===== NEW: downsample/balance like train/explain =====
    p.add_argument("--balance-ds", type=bool, default=True,
                   help="Build subset by train-style per-pt-bin D/B 1:1 balancing using ds bins (NOT roc pt-edges).")
    p.add_argument("--balance-frac", type=float, default=1.0,
                   help="Optional fraction applied to per-bin keep: n_keep=floor(frac*min(nD,nB)).")
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25,
                   help="Downsample bins width (GeV). Used if --ds-pt-edges empty.")
    p.add_argument("--ds-pt-edges", type=str, default="",
                   help="Optional manual ds bin edges, e.g. '3,4,5,6,8'. Overrides bin-width.")

    # PCA / plotting
    p.add_argument("--pt-edges", type=str, default="3,4,6,8,999", help="Edges for per-pt-bin plots (ROC-style bins).")
    p.add_argument("--out-prefix", type=str, default="", help="Output prefix (dir/file prefix). If empty, derive from ckpt.")
    p.add_argument("--save-npz", action="store_true", help="Save npz dump (emb, pca2, labels, pt).")

    # ===== NEW: scatter style for dense plots =====
    p.add_argument("--pt-size", type=float, default=5.0, help="Scatter point size (smaller reduces occlusion).")
    p.add_argument("--pt-alpha", type=float, default=0.20, help="Scatter alpha (smaller reduces occlusion).")
    p.add_argument("--pt-marker", type=str, default=".", help="Scatter marker, e.g. '.', 'o'. '.' is best for dense.")
    p.add_argument("--pt-rasterized", action="store_true", help="Use rasterized scatter (helps huge dense pdf/png rendering).")

    return p.parse_args()

def main():
    args = parse_args()

    # ---- device ----
    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # ---- load ckpt ----
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    pooling = ckpt_args.get("pooling", "sum")
    print(f"[INFO] Loaded ckpt: {args.ckpt}")
    print(f"[INFO] pooling(from ckpt) = {pooling}")

    # ---- dataset config ----
    root_file = args.root_file.strip() or ckpt_args.get("root_file", "")
    if root_file == "":
        raise ValueError("root-file not provided and not found in ckpt['args']['root_file'].")

    pt_min = args.pt_min if args.pt_min is not None else ckpt_args.get("pt_min", None)
    pt_max = args.pt_max if args.pt_max is not None else ckpt_args.get("pt_max", None)

    use_log_pt = bool(args.use_log_pt) or bool(ckpt_args.get("use_log_pt", False))

    print("[INFO] Building dataset with:")
    print(f"       root_file = {root_file}")
    print(f"       use_log_pt = {use_log_pt}")
    print(f"       pt_min/pt_max = {pt_min}/{pt_max}")
    print(f"       eta_abs_max = {args.eta_abs_max}")
    print(f"       had_pt_min/had_pt_max = {args.had_pt_min}/{args.had_pt_max}")
    print(f"       min_had = {args.min_had}")

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

    # ---- pick indices (D/B only) ----
    max_events = None if args.max_events <= 0 else int(args.max_events)
    idx_all = pick_indices_labels01(dataset, max_events=max_events, seed=args.seed)
    print(f"[INFO] D/B-only candidates: {len(idx_all)}")

    # ---- NEW: balance/downsample like train/explain ----
    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(args.ds_pt_edges, args.ds_pt_bin_width, pt_min, pt_max)
        idx_map = build_ptbin_class_index_from_indices(dataset, idx_all, ds_edges, num_classes=2)
        idx_bal = resample_balanced_by_ptbin_indices(
            idx_map,
            ds_edges,
            seed=args.seed,
            frac=float(args.balance_frac),
            num_classes=2
        )
        if len(idx_bal) == 0:
            print("[WARN] balance-ds produced empty set; fallback to unbalanced D/B-only indices.")
            idx = idx_all
        else:
            idx = idx_bal
            # print a small summary per bin (optional but useful)
            n_bins = len(ds_edges) - 1
            print("[INFO] balance-ds enabled. ds pt edges =", ds_edges.tolist())
            kept_total = 0
            for b in range(n_bins):
                nD = len(idx_map[(b, 0)])
                nB = len(idx_map[(b, 1)])
                base_keep = min(nD, nB)
                n_keep = int(np.floor(float(args.balance_frac) * base_keep))
                if n_keep > 0:
                    kept_total += 2 * n_keep
                print(f"[INFO] DS bin {ds_edges[b]:.2f}-{ds_edges[b+1]:.2f}: D={nD}, B={nB}, keep(each)={max(n_keep,0)}")
            print(f"[INFO] Balanced subset size: {len(idx)} (expected ~{kept_total})")
    else:
        idx = idx_all

    work_set = Subset(dataset, idx)
    print(f"[INFO] Work subset size: {len(work_set)}")

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ---- build model (match your benchmark defaults) ----
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

    # ---- hook: capture classifier input embedding (joint = [ele_feat, H_set]) ----
    captured = {"emb": []}

    def hook_fn(module, inp, out):
        x = inp[0]  # (B, D_joint)
        captured["emb"].append(x.detach().cpu())

    h = model.classifier[0].register_forward_hook(hook_fn)

    # ---- forward loop ----
    y_all = []
    pt_all = []

    with torch.no_grad():
        for batch in loader:
            ele = batch["ele_feat"].to(device)
            had = batch["had_feat"].to(device)
            mask = batch["had_mask"].to(device)
            y = batch["label"].to(device)

            _ = model(ele, had, mask, return_attn=False)

            y_cpu = y.detach().cpu().numpy().astype(np.int64)

            # ele_feat[:,0] is pt feature (maybe log-pt)
            pt_feat = ele[:, 0].detach().cpu().numpy()
            pt = np.exp(pt_feat) if use_log_pt else pt_feat

            y_all.append(y_cpu)
            pt_all.append(pt)

    h.remove()

    y_all = np.concatenate(y_all, axis=0)
    pt_all = np.concatenate(pt_all, axis=0)

    emb = torch.cat(captured["emb"], dim=0).numpy()
    print(f"[INFO] Embedding shape: {emb.shape}")

    # ---- PCA ----
    Z2, comps2, evr2 = pca_fit_transform(emb, n_components=2, center=True)
    print(f"[INFO] PCA explained variance ratio (PC1, PC2) = {evr2}")

    # ---- output prefix ----
    if args.out_prefix.strip():
        prefix = args.out_prefix.strip()
    else:
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        prefix = os.path.join("/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/lowD_output/", base + "_PCA")
    ensure_dir(prefix)

    # ---- plot config ----
    pt_size = float(args.pt_size)
    pt_alpha = float(args.pt_alpha)
    marker = str(args.pt_marker)
    rasterized = bool(args.pt_rasterized)

    isD = (y_all == 0)
    isB = (y_all == 1)

    # ---- plot: overall PCA scatter ----
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[isD, 0], Z2[isD, 1], s=pt_size, alpha=pt_alpha, marker=marker,
                label="D (label=0)", linewidths=0, rasterized=rasterized)
    plt.scatter(Z2[isB, 0], Z2[isB, 1], s=pt_size, alpha=pt_alpha, marker=marker,
                label="B (label=1)", linewidths=0, rasterized=rasterized)
    plt.xlabel(f"PC1 ({evr2[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({evr2[1]*100:.2f}%)")
    plt.title("PCA of classifier-input embedding (joint)")
    plt.grid(True)
    plt.legend(loc="best", markerscale=3)
    out_all = prefix + "_pca2_all.png"
    plt.tight_layout()
    plt.savefig(out_all, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {out_all}")

    # ---- plot: per-pt-bin PCA scatter (ROC-style edges) ----
    edges = parse_edges(args.pt_edges)
    n_bins = len(edges) - 1
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (pt_all >= lo) & (pt_all < hi)
        if m.sum() < 200:
            continue

        isD_b = m & isD
        isB_b = m & isB

        plt.figure(figsize=(6, 5))
        plt.scatter(Z2[isD_b, 0], Z2[isD_b, 1], s=pt_size, alpha=pt_alpha, marker=marker,
                    label=f"D (n={isD_b.sum()})", linewidths=0, rasterized=rasterized)
        plt.scatter(Z2[isB_b, 0], Z2[isB_b, 1], s=pt_size, alpha=pt_alpha, marker=marker,
                    label=f"B (n={isB_b.sum()})", linewidths=0, rasterized=rasterized)
        plt.xlabel(f"PC1 ({evr2[0]*100:.2f}%)")
        plt.ylabel(f"PC2 ({evr2[1]*100:.2f}%)")
        plt.title(f"PCA (e pT in [{lo:g}, {hi:g}) GeV)")
        plt.grid(True)
        plt.legend(loc="best", markerscale=3)
        out_bin = prefix + f"_pca2_pt{lo:g}-{hi:g}.png"
        plt.tight_layout()
        plt.savefig(out_bin, dpi=200)
        plt.close()
        print(f"[INFO] Saved: {out_bin}")

    # ---- save npz ----
    if args.save_npz:
        npz_path = prefix + "_dump.npz"
        np.savez_compressed(
            npz_path,
            y=y_all,
            pt=pt_all,
            emb=emb,
            pca2=Z2,
            pca_components=comps2,
            pca_evr=evr2,
            ckpt=args.ckpt,
            pooling=pooling,
            use_log_pt=use_log_pt,
            pt_min=pt_min if pt_min is not None else -1.0,
            pt_max=pt_max if pt_max is not None else -1.0,
            pt_edges=edges,
            balance_ds=bool(args.balance_ds),
            balance_frac=float(args.balance_frac),
            ds_pt_bin_width=float(args.ds_pt_bin_width),
            ds_pt_edges=str(args.ds_pt_edges),
            scatter_size=pt_size,
            scatter_alpha=pt_alpha,
            scatter_marker=marker,
            scatter_rasterized=rasterized,
        )
        print(f"[INFO] Saved: {npz_path}")

    # ---- tiny summary ----
    summary_path = prefix + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"root_file: {root_file}\n")
        f.write(f"pooling: {pooling}\n")
        f.write(f"use_log_pt: {use_log_pt}\n")
        f.write(f"electron pt cut: [{pt_min}, {pt_max})\n")
        f.write(f"balance_ds: {bool(args.balance_ds)}\n")
        if args.balance_ds:
            f.write(f"balance_frac: {float(args.balance_frac)}\n")
            f.write(f"ds_pt_bin_width: {float(args.ds_pt_bin_width)}\n")
            f.write(f"ds_pt_edges: {str(args.ds_pt_edges)}\n")
        f.write(f"N (D/B only): {len(y_all)}\n")
        f.write(f"emb dim: {emb.shape[1]}\n")
        f.write(f"PCA EVR: {evr2.tolist()}\n")
        f.write(f"scatter: size={pt_size}, alpha={pt_alpha}, marker={marker}, rasterized={rasterized}\n")
    print(f"[INFO] Saved: {summary_path}")

if __name__ == "__main__":
    main()
