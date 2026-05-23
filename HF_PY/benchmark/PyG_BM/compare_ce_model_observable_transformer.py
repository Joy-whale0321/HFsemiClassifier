#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# compare_ce_model_observable_transformer.py
#
# Purpose:
#   For the four observables used in Fig.7, compare D/B classification
#   cross entropy using:
#
#       1. observable only: O
#       2. model score only: s = logit_B - logit_D
#       3. model score + observable: (s, O)
#
#   Output exactly one plot for each observable.
#
# Example:
#
# cd /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM
#
# python compare_ce_model_observable_transformer.py \
#   --ckpt /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt \
#   --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
#   --pt-min 3.0 \
#   --pt-max 10.0 \
#   --balance-ds \
#   --max-keep-per-bin-per-class 500 \
#   --out-dir /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan/TF4plotwithData/ce_compare \
#   --tag transformer_ce_compare

import os
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import SetTransformerHF


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def sanitize_filename(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def parse_edges_csv(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
    vals = sorted(vals)
    if len(vals) < 2:
        raise ValueError("Need >=2 edges.")
    return np.array(vals, dtype=np.float64)


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 1 else 0.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out


def log_loss_binary(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    good = np.isfinite(score) & ((y == 0) | (y == 1))
    y = y[good]
    score = score[good]

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)

    sorted_score = score[order]
    i = 0

    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1

        if j - i > 1:
            avg_rank = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg_rank

        i = j

    rank_sum_pos = np.sum(ranks[y == 1])
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


# ============================================================
# Dataset balancing
# ============================================================
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


def parse_ds_pt_edges(
    ds_pt_edges: str,
    ds_pt_bin_width: float,
    pt_min: Optional[float],
    pt_max: Optional[float],
) -> np.ndarray:
    if ds_pt_edges is not None and str(ds_pt_edges).strip():
        return parse_edges_csv(ds_pt_edges)

    if pt_min is None or pt_max is None:
        raise ValueError("Need pt_min/pt_max to auto-build ds pt bins.")

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


def build_ptbin_class_index_from_indices(
    dataset: HFSemiClassifier,
    indices: List[int],
    pt_edges: np.ndarray,
    num_classes: int = 2,
):
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
    max_keep_per_bin_per_class: int = 1000,
) -> List[int]:
    rng = np.random.default_rng(seed)
    n_bins = len(pt_edges) - 1
    selected: List[int] = []

    for b in range(n_bins):
        pools = [idx_map[(b, c)] for c in range(num_classes)]

        if any(len(p) == 0 for p in pools):
            continue

        base_keep = min(len(pools[0]), len(pools[1]), int(max_keep_per_bin_per_class))
        n_keep = int(np.floor(float(frac) * base_keep))

        if n_keep <= 0:
            continue

        for c in range(num_classes):
            chosen = rng.choice(pools[c], size=n_keep, replace=False).tolist()
            selected.extend(chosen)

    if len(selected) == 0:
        return []

    rng.shuffle(selected)
    return selected


# ============================================================
# Observables
# ============================================================
def compute_observables(batch: Dict[str, torch.Tensor], use_log_pt: bool) -> Dict[str, np.ndarray]:
    ele = batch["ele_feat"].cpu().numpy()
    had = batch["had_feat"].cpu().numpy()
    msk = batch["had_mask"].cpu().numpy().astype(bool)

    B, _, _ = had.shape

    e_pt_feat = ele[:, 0]
    e_pt = np.exp(e_pt_feat) if use_log_pt else e_pt_feat

    n_had = msk.sum(axis=1).astype(np.float64)

    had_pt_feat = had[:, :, 0]
    had_pt = np.exp(had_pt_feat) if use_log_pt else had_pt_feat
    had_pt = np.where(msk, had_pt, 0.0)

    had_sin = np.where(msk, had[:, :, 2], 0.0)
    had_cos = np.where(msk, had[:, :, 3], 1.0)

    had_dphi = np.arctan2(had_sin, had_cos)
    had_abs_dphi = np.abs(had_dphi)

    sum_had_pt = had_pt.sum(axis=1)
    mean_had_pt = np.divide(sum_had_pt, np.maximum(n_had, 1.0))

    mean_abs_dphi = np.array(
        [safe_mean(had_abs_dphi[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )

    std_abs_dphi = np.array(
        [safe_std(had_abs_dphi[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )

    return {
        "e_pt": e_pt.astype(np.float64),
        "mean_had_pt": mean_had_pt.astype(np.float64),
        "std_abs_dphi": std_abs_dphi.astype(np.float64),
        "mean_abs_dphi": mean_abs_dphi.astype(np.float64),
    }


def selected_observables() -> List[Tuple[str, str]]:
    return [
        ("e_pt", r"electron $p_T$"),
        ("mean_had_pt", r"mean hadron $p_T$"),
        ("std_abs_dphi", r"std($|\Delta\phi|$)"),
        ("mean_abs_dphi", r"mean($|\Delta\phi|$)"),
    ]


# ============================================================
# Transformer loading and inference
# ============================================================
def build_transformer_model(ckpt_args: dict) -> SetTransformerHF:
    return SetTransformerHF(
        had_input_dim=5,
        ele_input_dim=3,
        d_model=int(ckpt_args.get("d_model", 256)),
        nhead=int(ckpt_args.get("nhead", 4)),
        num_layers=int(ckpt_args.get("num_layers", 4)),
        dim_feedforward=int(ckpt_args.get("dim_feedforward", 512)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        n_classes=2,
    )


@torch.no_grad()
def evaluate_scores_and_obs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_log_pt: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    y_all = []
    s_all = []
    obs_acc: Dict[str, List[np.ndarray]] = {}

    model.eval()

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ele, had, msk)
        s = logits[:, 1] - logits[:, 0]

        y_all.append(y.detach().cpu().numpy().astype(np.int64))
        s_all.append(s.detach().cpu().numpy().astype(np.float64))

        obs = compute_observables(batch, use_log_pt=use_log_pt)

        for k, v in obs.items():
            obs_acc.setdefault(k, []).append(v.astype(np.float64))

    y_all = np.concatenate(y_all, axis=0)
    s_all = np.concatenate(s_all, axis=0)
    obs_all = {k: np.concatenate(v, axis=0) for k, v in obs_acc.items()}

    return y_all, s_all, obs_all


# ============================================================
# Logistic regression for CE comparison
# ============================================================
def make_stratified_split(y: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        n_test = int(round(test_frac * len(idx)))
        n_test = max(1, min(n_test, len(idx) - 1))

        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def standardize_train_apply(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0, keepdims=True)
    sig = np.std(x_train, axis=0, keepdims=True)

    sig = np.where(np.isfinite(sig) & (sig > 1e-12), sig, 1.0)

    return (x_train - mu) / sig, (x_test - mu) / sig, mu.reshape(-1), sig.reshape(-1)


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Fit binary logistic regression:
        p = sigmoid(beta_0 + X @ beta)
    with L2 regularization on beta but not intercept.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape

    Xb = np.ones((n, d + 1), dtype=np.float64)
    Xb[:, 1:] = X

    beta = np.zeros(d + 1, dtype=np.float64)

    reg = np.zeros(d + 1, dtype=np.float64)
    reg[1:] = float(l2)

    for _ in range(max_iter):
        z = Xb @ beta
        p = sigmoid(z)
        r = p * (1.0 - p)

        grad = Xb.T @ (p - y) + reg * beta

        H = Xb.T @ (Xb * r[:, None])
        H += np.diag(reg)

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + 1e-8 * np.eye(d + 1), grad, rcond=None)[0]

        beta_new = beta - step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    Xb = np.ones((n, d + 1), dtype=np.float64)
    Xb[:, 1:] = X

    return sigmoid(Xb @ beta)


def evaluate_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    l2: float,
) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train, X_test, mu, sig = standardize_train_apply(X_train_raw, X_test_raw)

    beta = fit_logistic_newton(
        X=X_train,
        y=y_train,
        l2=l2,
        max_iter=100,
        tol=1e-10,
    )

    p_train = predict_logistic(X_train, beta)
    p_test = predict_logistic(X_test, beta)

    ce_train = log_loss_binary(y_train, p_train)
    ce_test = log_loss_binary(y_test, p_test)

    auc_train = auc_score(y_train, p_train)
    auc_test = auc_score(y_test, p_test)

    return {
        "ce_train": float(ce_train),
        "ce_test": float(ce_test),
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "beta0": float(beta[0]),
        "beta": beta[1:].copy(),
        "mu": mu.copy(),
        "sig": sig.copy(),
    }


def compare_for_one_observable(
    y: np.ndarray,
    s: np.ndarray,
    o: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    l2: float,
) -> Dict[str, Dict[str, float]]:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)

    X_o = o.reshape(-1, 1)
    X_s = s.reshape(-1, 1)
    X_so = np.stack([s, o], axis=1)

    res_o = evaluate_feature_set(X_o, y, train_idx, test_idx, l2=l2)
    res_s = evaluate_feature_set(X_s, y, train_idx, test_idx, l2=l2)
    res_so = evaluate_feature_set(X_so, y, train_idx, test_idx, l2=l2)

    return {
        "observable_only": res_o,
        "model_s_only": res_s,
        "model_s_plus_observable": res_so,
    }


# ============================================================
# Plot
# ============================================================
def plot_ce_comparison(
    result: Dict[str, Dict[str, float]],
    observable_key: str,
    observable_label: str,
    out_pdf: str,
) -> None:
    ensure_dir_for_file(out_pdf)

    names = [
        "Observable only",
        "Model score only",
        "Model score + observable",
    ]

    keys = [
        "observable_only",
        "model_s_only",
        "model_s_plus_observable",
    ]

    ce_vals = [result[k]["ce_test"] for k in keys]
    auc_vals = [result[k]["auc_test"] for k in keys]

    x = np.arange(len(names))

    plt.figure(figsize=(6.3, 4.8))
    bars = plt.bar(x, ce_vals)

    plt.axhline(np.log(2.0), linestyle="--", linewidth=1.0, label=r"random: $\ln 2$")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.008,
            f"CE={ce_vals[i]:.4f}\nAUC={auc_vals[i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ymax = max(max(ce_vals) + 0.08, np.log(2.0) + 0.06)
    ymin = max(0.0, min(ce_vals) - 0.08)

    plt.ylim(ymin, ymax)
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Cross entropy / log loss")
    plt.title(f"D/B classification CE comparison\n{observable_label}")
    plt.grid(True, axis="y", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


# ============================================================
# Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Compare CE: observable only vs model score only vs model score + observable.")

    p.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt",
    )
    p.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root",
    )
    p.add_argument("--tree-name", type=str, default="tree")

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-events", type=int, default=-1)

    p.add_argument("--pt-min", type=float, default=3.0)
    p.add_argument("--pt-max", type=float, default=10.0)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true")

    p.add_argument("--balance-ds", action="store_true")
    p.add_argument("--balance-frac", type=float, default=1.0)
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")
    p.add_argument("--max-keep-per-bin-per-class", type=int, default=1000)

    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--l2", type=float, default=1e-4)

    p.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan/TF4plotwithData/ce_compare",
    )
    p.add_argument("--tag", type=str, default="transformer_ce_compare")

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] device = {device}")
    print(f"[INFO] ckpt = {args.ckpt}")
    print(f"[INFO] root_file = {args.root_file}")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    use_log_pt = bool(args.use_log_pt) or bool(ckpt_args.get("use_log_pt", False))

    dataset = HFSemiClassifier(
        args.root_file,
        tree_name=args.tree_name,
        use_log_pt=use_log_pt,
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
    print(f"[INFO] D/B-only candidates = {len(idx_all)}")

    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(
            args.ds_pt_edges,
            args.ds_pt_bin_width,
            args.pt_min,
            args.pt_max,
        )

        print(f"[INFO] balancing pt edges = {ds_edges.tolist()}")

        idx_map = build_ptbin_class_index_from_indices(
            dataset,
            idx_all,
            ds_edges,
            num_classes=2,
        )

        idx_eval = resample_balanced_by_ptbin_indices(
            idx_map,
            ds_edges,
            seed=args.seed,
            frac=float(args.balance_frac),
            num_classes=2,
            max_keep_per_bin_per_class=int(args.max_keep_per_bin_per_class),
        )

        print(
            f"[INFO] balance-ds with cap={args.max_keep_per_bin_per_class}: "
            f"{len(idx_all)} -> {len(idx_eval)}"
        )

        if len(idx_eval) == 0:
            print("[WARN] balance-ds produced empty set. Fallback to unbalanced D/B set.")
            idx_eval = idx_all
    else:
        idx_eval = idx_all

    work_set = Subset(dataset, idx_eval)

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = build_transformer_model(ckpt_args).to(device)

    if "model_state_dict" not in ckpt:
        raise KeyError("checkpoint missing model_state_dict")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_all, s_all, obs_all = evaluate_scores_and_obs(
        model=model,
        loader=loader,
        device=device,
        use_log_pt=use_log_pt,
    )

    print(
        f"[INFO] final N = {len(y_all)} | "
        f"D = {int(np.sum(y_all == 0))} | "
        f"B = {int(np.sum(y_all == 1))}"
    )

    tag = sanitize_filename(args.tag)

    npz_path = os.path.join(args.out_dir, f"{tag}_scores_obs.npz")
    np.savez_compressed(
        npz_path,
        y=y_all,
        s=s_all,
        **obs_all,
        ckpt=args.ckpt,
        root_file=args.root_file,
        balance_ds=bool(args.balance_ds),
        balance_frac=float(args.balance_frac),
        max_keep_per_bin_per_class=int(args.max_keep_per_bin_per_class),
        pt_min=float(args.pt_min),
        pt_max=float(args.pt_max),
    )
    print(f"[INFO] saved arrays: {npz_path}")

    train_idx, test_idx = make_stratified_split(
        y=y_all,
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    print(f"[INFO] train N = {len(train_idx)} | test N = {len(test_idx)}")

    rows = []

    for key, label in selected_observables():
        if key not in obs_all:
            print(f"[WARN] missing observable: {key}")
            continue

        result = compare_for_one_observable(
            y=y_all,
            s=s_all,
            o=obs_all[key],
            train_idx=train_idx,
            test_idx=test_idx,
            l2=float(args.l2),
        )

        out_pdf = os.path.join(
            args.out_dir,
            f"{tag}_ce_compare_{sanitize_filename(key)}.pdf",
        )

        plot_ce_comparison(
            result=result,
            observable_key=key,
            observable_label=label,
            out_pdf=out_pdf,
        )

        print(f"[INFO] saved plot: {out_pdf}")

        row = {
            "observable": key,
            "label": label,
            "plot": out_pdf,
            "ce_obs": result["observable_only"]["ce_test"],
            "ce_s": result["model_s_only"]["ce_test"],
            "ce_s_obs": result["model_s_plus_observable"]["ce_test"],
            "auc_obs": result["observable_only"]["auc_test"],
            "auc_s": result["model_s_only"]["auc_test"],
            "auc_s_obs": result["model_s_plus_observable"]["auc_test"],
            "delta_ce_s_minus_sobs": result["model_s_only"]["ce_test"] - result["model_s_plus_observable"]["ce_test"],
            "delta_ce_random_minus_obs": np.log(2.0) - result["observable_only"]["ce_test"],
        }

        rows.append(row)

        print(
            f"[CE] {key:16s} "
            f"O={row['ce_obs']:.6f} "
            f"s={row['ce_s']:.6f} "
            f"s+O={row['ce_s_obs']:.6f} "
            f"improve(s->s+O)={row['delta_ce_s_minus_sobs']:.6f}"
        )

    csv_path = os.path.join(args.out_dir, f"{tag}_ce_compare_summary.csv")
    ensure_dir_for_file(csv_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "observable",
            "label",
            "CE_observable_only",
            "CE_model_s_only",
            "CE_model_s_plus_observable",
            "AUC_observable_only",
            "AUC_model_s_only",
            "AUC_model_s_plus_observable",
            "delta_CE_model_s_minus_model_s_plus_observable",
            "delta_CE_random_ln2_minus_observable_only",
            "plot_file",
        ])

        for r in rows:
            writer.writerow([
                r["observable"],
                r["label"],
                f"{r['ce_obs']:.8g}",
                f"{r['ce_s']:.8g}",
                f"{r['ce_s_obs']:.8g}",
                f"{r['auc_obs']:.8g}",
                f"{r['auc_s']:.8g}",
                f"{r['auc_s_obs']:.8g}",
                f"{r['delta_ce_s_minus_sobs']:.8g}",
                f"{r['delta_ce_random_minus_obs']:.8g}",
                r["plot"],
            ])

    print(f"[INFO] saved CSV: {csv_path}")

    txt_path = os.path.join(args.out_dir, f"{tag}_ce_compare_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("CE comparison: observable only vs model score only vs model score + observable\n")
        f.write(f"ckpt = {args.ckpt}\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"N = {len(y_all)}\n")
        f.write(f"D = {int(np.sum(y_all == 0))}\n")
        f.write(f"B = {int(np.sum(y_all == 1))}\n")
        f.write(f"balance_ds = {bool(args.balance_ds)}\n")
        f.write(f"max_keep_per_bin_per_class = {int(args.max_keep_per_bin_per_class)}\n")
        f.write(f"random CE ln2 = {np.log(2.0):.6f}\n\n")

        for r in rows:
            f.write(
                f"{r['observable']:16s} "
                f"CE[O]={r['ce_obs']:.6f} "
                f"CE[s]={r['ce_s']:.6f} "
                f"CE[s+O]={r['ce_s_obs']:.6f} "
                f"AUC[O]={r['auc_obs']:.6f} "
                f"AUC[s]={r['auc_s']:.6f} "
                f"AUC[s+O]={r['auc_s_obs']:.6f} "
                f"delta_CE_s_to_sO={r['delta_ce_s_minus_sobs']:.6f}\n"
            )

    print(f"[INFO] saved TXT: {txt_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()