#!/usr/bin/env python3
# explain_HFSemiClassifier_bm.py
#
# Simple, readable evaluation/visualization for the DeepSetsHF benchmark.
# - Auto-loads training args from checkpoint (pt range, pooling, root file, etc.)
# - Computes scores and plots:
#     * p(B) and (logit_B - logit_D) distributions
#     * ROC curve + AUC (overall and per electron-pT bin)
# - If pooling is "attn" / "attn_mean", can print top-k hadrons by attention.
#
# Usage examples:
#   python explain_HFSemiClassifier_bm.py --ckpt path/to/model.pt
#   python explain_HFSemiClassifier_bm.py --ckpt model.pt --pt-edges "3,4,6,8,999"
#   python explain_HFSemiClassifier_bm.py --ckpt model.pt --max-events 50000 --print-attn
#
# Notes:
# - This script avoids saving huge tensors by default (no hadron point clouds dumped).
# - It supports evaluation on label {0=D, 1=B}; other labels are ignored by default.

import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


# ----------------------------- utils -----------------------------

def parse_edges(s: str) -> np.ndarray:
    """Parse comma-separated edges string into a float array."""
    if s is None or str(s).strip() == "":
        return np.array([3.0, 4.0, 6.0, 8.0, 1e9], dtype=np.float64)
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) < 2:
        raise ValueError("pt-edges must have at least 2 numbers, e.g. '3,4,6,8,999'")
    # ensure sorted
    vals = sorted(vals)
    return np.array(vals, dtype=np.float64)

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simple ROC + AUC (no sklearn). y_true in {0,1}, y_score higher => more likely 1.
    Returns (fpr, tpr, thresholds, auc).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    # sort by score desc
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        # degenerate
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0]) if P > 0 else np.array([0.0, 0.0])
        thr = np.array([np.inf, -np.inf])
        return fpr, tpr, thr, float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / P
    fpr = fps / N

    # thresholds: unique score points (use sorted scores)
    thr = y_score

    # AUC via trapezoid (fpr increasing? it's increasing with the cumulative)
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, thr, float(auc)


def purity_eff_curve(y_true: np.ndarray, score: np.ndarray):
    """
    y_true in {0,1}, score higher => more likely class 1.
    Return: eff (TPR), pur (precision), thr (threshold at each point).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    score  = np.asarray(score).astype(np.float64)

    order = np.argsort(-score)
    y = y_true[order]
    s = score[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P  = np.sum(y == 1)

    eff = tp / max(P, 1)  # TPR
    pur = tp / (tp + fp + 1e-12)
    thr = s
    return eff, pur, thr


def pick_eval_indices(dataset: HFSemiClassifier, max_events: int, seed: int, keep_labels=(0,1)) -> List[int]:
    idx_all = []
    for i in range(len(dataset)):
        evt_idx, ele_idx = dataset.electron_index[i]
        raw_tag = int(dataset.ele_hf_TAG[evt_idx][ele_idx])
        if raw_tag == 1:
            y = 0
        elif raw_tag == 3:
            y = 1
        else:
            y = 2

        if y in keep_labels:
            idx_all.append(i)

    if max_events is not None and max_events > 0 and len(idx_all) > max_events:
        rng = np.random.default_rng(seed)
        idx_all = rng.choice(idx_all, size=max_events, replace=False).tolist()

    return idx_all

def get_electron_pt_from_dataset(dataset, global_idx: int) -> float:
    evt_idx, ele_idx = dataset.electron_index[global_idx]
    return float(dataset.ele_pt[evt_idx][ele_idx])

def parse_ds_pt_edges(args, pt_min, pt_max):
    if args.ds_pt_edges.strip():
        edges = [float(x) for x in args.ds_pt_edges.split(",")]
        edges = sorted(edges)
        if len(edges) < 2:
            raise ValueError("ds-pt-edges must have >=2 numbers")
        return np.array(edges, dtype=np.float64)

    if pt_min is None or pt_max is None:
        raise ValueError("Need pt_min/pt_max to auto-build ds pt bins (or set --ds-pt-edges).")

    w = float(args.ds_pt_bin_width)
    if w <= 0:
        raise ValueError("--ds-pt-bin-width must be > 0")

    edges = [float(pt_min)]
    x = float(pt_min)
    while x + w < float(pt_max) - 1e-6:
        x += w
        edges.append(x)
    edges.append(float(pt_max))
    return np.array(edges, dtype=np.float64)

def build_ptbin_class_index_from_indices(dataset, indices, pt_edges, num_classes=2):
    n_bins = len(pt_edges) - 1
    idx_map = {(b, c): [] for b in range(n_bins) for c in range(num_classes)}

    for gidx in indices:
        # label：你如果愿意继续沿用 pick_eval_indices 的过滤，这里也可直接 dataset[gidx]["label"]
        # 但为了快，推荐用 ele_hf_TAG 直接算 label（和 __getitem__ 一致）
        evt_idx, ele_idx = dataset.electron_index[gidx]
        raw_tag = int(dataset.ele_hf_TAG[evt_idx][ele_idx])
        if raw_tag == 1:
            y = 0
        elif raw_tag == 3:
            y = 1
        else:
            y = 2

        if not (0 <= y < num_classes):
            continue

        pt = get_electron_pt_from_dataset(dataset, gidx)

        b = int(np.searchsorted(pt_edges, pt, side="right") - 1)
        if 0 <= b < n_bins:
            idx_map[(b, y)].append(gidx)

    return idx_map

def resample_balanced_by_ptbin_indices(idx_map, pt_edges, seed=12345, frac=1.0, num_classes=2):
    rng = np.random.default_rng(seed)
    n_bins = len(pt_edges) - 1
    selected = []

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
    p = argparse.ArgumentParser(description="Explain/evaluate DeepSetsHF benchmark model.")
    p.add_argument("--ckpt", 
                   type=str, 
                   default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM_optimize/Weight_of_Model/deepset/DeepSetsHF_best_ALL_3.0-10.0_had3x128_clf3x128_sum_M4.pt", 
                   help="Path to model checkpoint (.pt) saved by train script.")
    p.add_argument("--root-file", type=str, default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_2_allAccept.root", help="Override dataset ROOT file (default: from ckpt args).")
    p.add_argument("--tree-name", type=str, default="tree", help="TTree name (default: tree).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda / cpu / empty=auto")
    p.add_argument("--max-events", type=int, default=-1, help="Max samples to evaluate (after filtering labels). <=0 means all.")
    p.add_argument("--seed", type=int, default=12345)

    # dataset cuts (override-friendly)
    p.add_argument("--pt-min", type=float, default=None, help="Override electron pt_min cut (default: from ckpt args).")
    p.add_argument("--pt-max", type=float, default=None, help="Override electron pt_max cut (default: from ckpt args).")
    p.add_argument("--eta-abs-max", type=float, default=5.0, help="|eta| cut (default: 5.0 like train).")
    p.add_argument("--had-pt-min", type=float, default=0.2, help="hadron pt min cut (default: 0.2 like train).")
    p.add_argument("--had-pt-max", type=float, default=None, help="hadron pt max cut (default: None).")
    p.add_argument("--min-had", type=int, default=4, help="Min hadrons required per sample (default: 4 like train).")
    p.add_argument("--use-log-pt", action="store_true", help="Use log(pt) features (default: follow ckpt if present, else False).")

    # evaluation bins
    p.add_argument("--pt-edges", type=str, default="", help="Electron pT edges for per-bin ROC, e.g. '3,4,6,8,999'.")
    p.add_argument("--out-prefix", type=str, default="", help="Output prefix for plots (default: derived from ckpt name).")
    
    # ======= NEW: downsample (train-style) bins for balancing =======
    p.add_argument("--balance-ds", action="store_true",
                   help="Build evaluation subset by train-style per-pt-bin D/B 1:1 balancing using ds bins (NOT roc pt-edges).")
    p.add_argument("--balance-frac", type=float, default=1.0,
                   help="Optional fraction applied to per-bin keep: n_keep=floor(frac*min(nD,nB)).")
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25,
                   help="Downsample bins width (GeV). Used if --ds-pt-edges empty.")
    p.add_argument("--ds-pt-edges", type=str, default="",
                   help="Optional manual ds bin edges, e.g. '3,4,5,6,8'. Overrides bin-width.")

    # attention printing (only meaningful for attn / attn_mean)
    p.add_argument("--print-attn", action="store_true", help="If attn pooling, print top-k hadrons for a few samples.")
    p.add_argument("--n-print", type=int, default=5, help="How many samples to print attention summary for.")
    p.add_argument("--topk", type=int, default=5, help="Top-k hadrons to show for each printed sample.")

    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    # ---- device ----
    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # ---- load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    print(f"[INFO] Loaded ckpt: {args.ckpt}")
    if ckpt_args:
        # keep it short
        keys_show = ["root_file", "pt_min", "pt_max", "pooling", "batch_size", "lr"]
        show = {k: ckpt_args.get(k, None) for k in keys_show if k in ckpt_args}
        print(f"[INFO] ckpt args (partial): {show}")
    else:
        print("[WARN] ckpt has no 'args'. Will rely on CLI defaults/overrides.")

    # ---- dataset config (match train unless overridden) ----
    root_file = args.root_file.strip() or ckpt_args.get("root_file", "")
    if root_file == "":
        raise ValueError("root-file not provided and not found in ckpt['args']['root_file'].")

    pt_min = args.pt_min if args.pt_min is not None else ckpt_args.get("pt_min", None)
    pt_max = args.pt_max if args.pt_max is not None else ckpt_args.get("pt_max", None)

    # train script uses use_log_pt=False. We follow:
    #   - if user passes --use-log-pt -> True
    #   - else if ckpt args recorded it -> use that
    #   - else -> False
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
        # min_had=0.0,
    )

    # ---- choose evaluation subset ----
    max_events = None if args.max_events <= 0 else int(args.max_events)
    idx_eval = pick_eval_indices(dataset, max_events=max_events, seed=args.seed, keep_labels=(0,1))
    # ===== [optional] train-style ds-bin balancing for evaluation subset =====
    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(args, pt_min, pt_max)
        print(f"[INFO] DS (balancing) pt edges: {ds_edges.tolist()}")

        idx_map = build_ptbin_class_index_from_indices(dataset, idx_eval, ds_edges, num_classes=2)

        # optional: print counts per ds bin
        n_bins = len(ds_edges) - 1
        for b in range(n_bins):
            nD = len(idx_map[(b,0)])
            nB = len(idx_map[(b,1)])
            keep_each = int(np.floor(args.balance_frac * min(nD, nB)))
            if min(nD, nB) > 0:
                print(f"[INFO] DS bin {ds_edges[b]:.3f}-{ds_edges[b+1]:.3f}: D={nD}, B={nB}, keep(each)=    {keep_each}")

        idx_bal = resample_balanced_by_ptbin_indices(
            idx_map, ds_edges, seed=args.seed, frac=float(args.balance_frac), num_classes=2
        )
        print(f"[INFO] balance-ds: {len(idx_eval)} -> {len(idx_bal)}")
        if len(idx_bal) > 0:
            idx_eval = idx_bal

    work_set = Subset(dataset, idx_eval)
    
    print(f"[INFO] Eval subset size (labels 0/1 only): {len(work_set)}")

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ---- build model (match benchmark defaults; pooling from ckpt unless overridden later) ----
    pooling = ckpt_args.get("pooling", "sum")
    # allow CLI override via ckpt args not needed; keep simple: use ckpt pooling
    print(f"[INFO] Building model: pooling={pooling}")

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

    # ---- forward loop ----
    y_all = []
    pt_all = []
    pB_all = []
    s_all = []  # logit_B - logit_D

    # attention printing buffers (small)
    printed = 0

    for batch in loader:
        ele = batch["ele_feat"].to(device)    # (B,3)
        had = batch["had_feat"].to(device)    # (B,N,5)
        mask = batch["had_mask"].to(device)   # (B,N) bool
        y = batch["label"].to(device)         # (B,)

        out = model(ele, had, mask, return_attn=True)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            logits, alpha = out
        else:
            logits, alpha = out, None

        probs = torch.softmax(logits, dim=-1)
        pB = probs[:, 1]
        s = logits[:, 1] - logits[:, 0]

        y_cpu = y.detach().cpu().numpy().astype(np.int64)
        # electron pt feature is ele[:,0] (pt or log pt)
        pt_feat = ele[:, 0].detach().cpu().numpy()
        pt = np.exp(pt_feat) if use_log_pt else pt_feat

        y_all.append(y_cpu)
        pt_all.append(pt)
        pB_all.append(pB.detach().cpu().numpy())
        s_all.append(s.detach().cpu().numpy())

        # optional attention print
        if args.print_attn and (pooling in ("attn", "attn_mean")) and (alpha is not None) and printed < args.n_print:
            alpha_cpu = alpha.detach().cpu().numpy()  # (B,N)
            had_cpu = had.detach().cpu().numpy()
            mask_cpu = mask.detach().cpu().numpy()

            B, N = alpha_cpu.shape
            for i in range(B):
                if printed >= args.n_print:
                    break
                # valid hadrons
                valid = mask_cpu[i].astype(bool)
                if valid.sum() == 0:
                    printed += 1
                    continue
                a = alpha_cpu[i].copy()
                a[~valid] = -np.inf
                topk = min(args.topk, int(valid.sum()))
                idx = np.argsort(-a)[:topk]

                # had features: [pt, deta, sin(dphi), cos(dphi), charge]
                had_pt_feat = had_cpu[i, idx, 0]
                had_pt = np.exp(had_pt_feat) if use_log_pt else had_pt_feat
                had_deta = had_cpu[i, idx, 1]
                had_sin = had_cpu[i, idx, 2]
                had_cos = had_cpu[i, idx, 3]
                had_q = had_cpu[i, idx, 4]
                attn_vals = alpha_cpu[i, idx]

                print(f"\n[ATTN] sample #{printed} | y={int(y_cpu[i])} | e_pt={float(pt[i]):.3f} | pB={float(pB.detach().cpu().numpy()[i]):.3f}")
                for k in range(topk):
                    dphi = float(np.arctan2(had_sin[k], had_cos[k]))
                    print(f"   k={k:02d}  attn={attn_vals[k]:.4f}  had_pt={had_pt[k]:.3f}  dEta={had_deta[k]:+.3f}  dPhi={dphi:+.3f}  q={had_q[k]:+.0f}")
                printed += 1

    y_all = np.concatenate(y_all, axis=0)
    pt_all = np.concatenate(pt_all, axis=0)
    pB_all = np.concatenate(pB_all, axis=0)
    s_all = np.concatenate(s_all, axis=0)

    # ---- output prefix ----
    if args.out_prefix.strip() != "":
        prefix = args.out_prefix.strip()
    else:
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        prefix = os.path.join("/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM_optimize/Replot/", base + "_eval")
    ensure_dir(prefix)

    # ---- plots: score distributions ----
    isB = (y_all == 1)
    isD = (y_all == 0)

    plt.figure(figsize=(5, 4))
    plt.hist(pB_all[isB], bins=60, density=True, histtype="step", label="True B")
    plt.hist(pB_all[isD], bins=60, density=True, histtype="step", label="True D")
    plt.xlabel("p(B)")
    plt.ylabel("Density")
    plt.title("Score distribution: p(B)")
    plt.grid(True)
    plt.legend(loc="best")
    out1 = prefix + "_score_pB.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out1}")

    plt.figure(figsize=(5, 4))
    plt.hist(s_all[isB], bins=60, density=True, histtype="step", label="True B")
    plt.hist(s_all[isD], bins=60, density=True, histtype="step", label="True D")
    plt.xlabel("s = logit_B - logit_D")
    plt.ylabel("Density")
    plt.title("Score distribution: logit diff")
    plt.grid(True)
    plt.legend(loc="best")
    out2 = prefix + "_score_logitdiff.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out2}")

    # ---- ROC (overall) using p(B) ----
    fpr, tpr, thr, auc = compute_roc_auc(y_all, pB_all)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (overall) using p(B)")
    plt.grid(True)
    plt.legend(loc="best")
    out3 = prefix + "_roc_all.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out3}")

    # ---- per-pt-bin ROC ----
    edges = parse_edges(args.pt_edges)
    n_bins = len(edges) - 1

    plt.figure(figsize=(6, 5))
    auc_bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (pt_all >= lo) & (pt_all < hi)
        if m.sum() < 50:
            auc_bins.append(np.nan)
            continue
        fpr_b, tpr_b, _, auc_b = compute_roc_auc(y_all[m], pB_all[m])
        auc_bins.append(auc_b)
        label = f"{lo:g}-{hi:g} (n={m.sum()}, AUC={auc_b:.3f})"
        plt.plot(fpr_b, tpr_b, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC by electron pT bin (p(B))")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    out4 = prefix + "_roc_by_pt.png"
    plt.tight_layout()
    plt.savefig(out4, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out4}")

    # ======= Purity vs Efficiency (B and D on the SAME figure) ==========
    # ----- B as signal -----
    effB, purB, thrB = purity_eff_curve(y_all, s_all)

    # ----- D as signal -----
    # map: D -> 1, B -> 0
    yD = 1 - y_all
    sD = -s_all
    effD, purD, thrD = purity_eff_curve(yD, sD)

    plt.figure(figsize=(5.5, 4.5))

    plt.plot(
        effB, purB,
        label="B as signal",
        linewidth=2
    )

    plt.plot(
        effD, purD,
        label="D as signal",
        linewidth=2,
        linestyle="--"
    )

    plt.xlabel("Efficiency")
    plt.ylabel("Purity")
    plt.title("Purity vs Efficiency (D vs B)")
    plt.grid(True)
    plt.legend(loc="best")

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    out_pe = prefix + "_purxeff_DB.png"
    plt.tight_layout()
    plt.savefig(out_pe, dpi=150)
    plt.close()

    print(f"[INFO] Saved: {out_pe}")

    # ---- quick text summary ----
    summary_path = prefix + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"root_file: {root_file}\n")
        f.write(f"pooling: {pooling}\n")
        f.write(f"use_log_pt: {use_log_pt}\n")
        f.write(f"electron pt cut: [{pt_min}, {pt_max})\n")
        f.write(f"eval N (labels 0/1): {len(y_all)}\n")
        f.write(f"overall AUC (pB): {auc:.6f}\n")
        f.write("per-bin AUC:\n")
        for b in range(n_bins):
            lo, hi = edges[b], edges[b+1]
            f.write(f"  {lo:g}-{hi:g}: {auc_bins[b]}\n")
    print(f"[INFO] Saved: {summary_path}")

    # ---- optional npz dump (small) ----
    npz_path = prefix + "_scores.npz"
    np.savez_compressed(
        npz_path,
        y=y_all,
        pt=pt_all,
        pB=pB_all,
        s=s_all,
        pt_edges=edges,
        ckpt=args.ckpt,
        pooling=pooling,
        use_log_pt=use_log_pt,
        pt_min=pt_min if pt_min is not None else -1.0,
        pt_max=pt_max if pt_max is not None else -1.0,
    )
    print(f"[INFO] Saved: {npz_path}")

if __name__ == "__main__":
    main()
