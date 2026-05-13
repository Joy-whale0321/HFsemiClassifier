#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a diagnostic BDT using hand-crafted physics observables,
optionally combined with the trained Transformer score s_TF.

Recommended location
--------------------
/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/train_bdt_tf_score_physics.py

Default outputs
---------------
BDT weights:
    /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/weight_of_BDT/

Plots / CSV / NPZ:
    /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/results_bdt/

Typical usage
-------------
# physics-only BDT
python train_bdt_tf_score_physics.py \
    --data /path/to/data.pt \
    --checkpoint /path/to/tf_checkpoint.pt \
    --no_use_score \
    --tag phys_only

# TF score + physics BDT
python train_bdt_tf_score_physics.py \
    --data /path/to/data.pt \
    --checkpoint /path/to/tf_checkpoint.pt \
    --use_score \
    --tag score_phys
"""

import os
import sys
import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib


# ============================================================
# 0. Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = THIS_DIR.parent
PYG_BM_DIR = BENCHMARK_DIR / "PyG_BM"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PYG_BM_DIR))


# ============================================================
# 1. Candidate model imports
# ============================================================
# 如果你的 TF 模型类名不同，优先改这里。
# 格式：
#     ("python_file_without_py", "ClassName")
#
# 例如：
#     PyG_BM/models.py 里面 class TransformerClassifier
#     那这里就是 ("models", "TransformerClassifier")
MODEL_CANDIDATES = [
    ("model", "TransformerClassifier"),
    ("models", "TransformerClassifier"),
    ("models", "TFClassifier"),
    ("models", "HFSemiTransformer"),
    ("model_PyG", "TransformerClassifier"),
    ("PyG_model", "TransformerClassifier"),
    ("HFSemiClassifier_model", "TransformerClassifier"),
]


# ============================================================
# 2. Basic utility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_get(data, names, default=None):
    for name in names:
        if hasattr(data, name):
            val = getattr(data, name)
            if val is not None:
                return val
        if isinstance(data, dict) and name in data:
            return data[name]
    return default


def infer_label(data):
    """
    Assumed label convention:
        D/charm   = 0
        B/bottom  = 1
    """
    y = safe_get(data, ["y", "label", "target", "truth"])
    if y is None:
        raise AttributeError(
            "Cannot find label. Expected one of: y, label, target, truth."
        )

    if torch.is_tensor(y):
        y = y.detach().cpu()
        if y.numel() == 1:
            return int(y.item())
        return int(torch.argmax(y).item())

    y = np.asarray(y)
    if y.size == 1:
        return int(y.item())
    return int(np.argmax(y))


def load_dataset(path):
    """
    Load PyG-style dataset.

    Supports:
        - list of Data
        - tuple/list of splits
        - dict containing train/val/test/all/data/dataset
        - InMemoryDataset-like object
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["data", "all", "dataset", "test", "val", "train"]:
            if key in obj:
                obj = obj[key]
                break

    if isinstance(obj, tuple):
        obj = list(obj)

    if isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
            flat = []
            for part in obj:
                flat.extend(list(part))
            return flat
        return obj

    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return [obj[i] for i in range(len(obj))]

    raise TypeError(f"Unsupported dataset object type: {type(obj)}")


def db_balance_and_limit(dataset, max_per_class=500, seed=42):
    """
    D/B balance + upper limit.

    每类最多 max_per_class 个。
    如果某一类本身少于 max_per_class，则取两类中较小的数量。
    """
    rng = np.random.default_rng(seed)

    by_class = {0: [], 1: []}
    for data in dataset:
        y = infer_label(data)
        if y in by_class:
            by_class[y].append(data)

    n0 = len(by_class[0])
    n1 = len(by_class[1])

    if n0 == 0 or n1 == 0:
        raise RuntimeError(f"Need both classes. Got D/0={n0}, B/1={n1}")

    n_keep = min(n0, n1, max_per_class)

    selected = []
    for cls in [0, 1]:
        idx = rng.choice(len(by_class[cls]), size=n_keep, replace=False)
        selected.extend([by_class[cls][i] for i in idx])

    rng.shuffle(selected)

    print(f"[balance] original: D/0={n0}, B/1={n1}")
    print(f"[balance] selected: D/0={n_keep}, B/1={n_keep}, total={2 * n_keep}")

    return selected


# ============================================================
# 3. Load trained TF model and compute score
# ============================================================

def find_model_class():
    for module_name, class_name in MODEL_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"[model] using {module_name}.{class_name}")
                return getattr(module, class_name)
        except Exception:
            continue

    raise ImportError(
        "Cannot automatically import TF model class.\n"
        "Please edit MODEL_CANDIDATES in this script.\n"
        f"Current sys.path includes:\n"
        f"  THIS_DIR={THIS_DIR}\n"
        f"  BENCHMARK_DIR={BENCHMARK_DIR}\n"
        f"  PYG_BM_DIR={PYG_BM_DIR}\n"
    )


def build_model(args):
    """
    Build TF model.

    这里尽量兼容常见 __init__。
    如果你的原 train 代码里 model 初始化参数不同，
    只需要改这个函数。
    """
    ModelClass = find_model_class()

    init_trials = [
        dict(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
        ),
        dict(
            in_channels=args.input_dim,
            hidden_channels=args.hidden_dim,
            out_channels=2,
        ),
        dict(
            node_input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
        ),
        dict(
            in_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            n_classes=2,
        ),
        dict(),
    ]

    last_err = None
    for kwargs in init_trials:
        try:
            model = ModelClass(**kwargs)
            print(f"[model] built with kwargs={kwargs}")
            return model
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Failed to build TF model. Please modify build_model(args).\n"
        f"Last error: {last_err}"
    )


def load_checkpoint_to_model(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    new_state = {}
    for k, v in ckpt.items():
        nk = k
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"[checkpoint] loaded: {checkpoint_path}")
    print(f"[checkpoint] missing keys: {len(missing)}")
    print(f"[checkpoint] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[checkpoint] first missing keys:", missing[:5])
    if len(unexpected) > 0:
        print("[checkpoint] first unexpected keys:", unexpected[:5])

    model.to(device)
    model.eval()
    return model


def model_forward_score(model, data, device):
    """
    Compute s_TF = P(B).

    已兼容：
        model(data)
        model(x, edge_index, batch)
        model(x)

    如果你原来的 TF forward 特别定制，就改这里。
    """
    data = data.to(device)

    with torch.no_grad():
        out = None

        try:
            out = model(data)
        except Exception:
            out = None

        if out is None:
            x = safe_get(data, ["x", "hadron_x", "h_x", "node_features"])
            edge_index = safe_get(data, ["edge_index"])
            batch = safe_get(data, ["batch"])

            if x is not None and batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            try:
                out = model(x, edge_index, batch)
            except Exception:
                out = None

        if out is None:
            x = safe_get(data, ["x", "hadron_x", "h_x", "node_features"])
            try:
                out = model(x)
            except Exception:
                out = None

        if out is None:
            raise RuntimeError(
                "Cannot forward TF model. Please modify model_forward_score()."
            )

        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.detach()

        if out.dim() == 1:
            if out.numel() == 1:
                s = torch.sigmoid(out.reshape(-1))[0].item()
            else:
                prob = torch.softmax(out.reshape(1, -1), dim=1)
                s = prob[0, 1].item()
        elif out.dim() == 2:
            if out.shape[1] == 1:
                s = torch.sigmoid(out[:, 0])[0].item()
            else:
                prob = torch.softmax(out, dim=1)
                s = prob[0, 1].item()
        else:
            out = out.reshape(1, -1)
            prob = torch.softmax(out, dim=1)
            s = prob[0, 1].item()

    return float(s)


# ============================================================
# 4. Physics observables
# ============================================================

def get_hadron_array(data):
    x = safe_get(data, ["x", "hadron_x", "h_x", "node_features"])
    if x is None:
        raise AttributeError(
            "Cannot find hadron array. Expected one of: x, hadron_x, h_x, node_features."
        )
    return to_numpy(x).astype(np.float64)


def get_electron_array(data):
    e = safe_get(data, ["electron", "ele", "e", "electron_x", "e_x"])
    if e is None:
        return None
    return to_numpy(e).astype(np.float64).reshape(-1)


def delta_phi(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def weighted_std(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() <= 1:
        return 0.0

    v = values[mask]
    w = weights[mask]
    mean = np.average(v, weights=w)
    var = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(max(var, 0.0)))


def build_physics_features(
    data,
    pt_idx=0,
    eta_idx=1,
    phi_idx=2,
    e_pt_idx=0,
    e_eta_idx=1,
    e_phi_idx=2,
):
    """
    默认 hadron x 列顺序：
        x[:, 0] = hadron pT
        x[:, 1] = delta_eta 或 eta
        x[:, 2] = delta_phi 或 phi

    如果你的输入是：
        [delta_eta, delta_phi, pt, ...]
    运行时加：
        --pt_idx 2 --eta_idx 0 --phi_idx 1
    """
    x = get_hadron_array(data)
    n = x.shape[0]

    if n == 0:
        pt = np.array([])
        eta = np.array([])
        phi = np.array([])
    else:
        pt = x[:, pt_idx] if 0 <= pt_idx < x.shape[1] else np.ones(n)
        eta = x[:, eta_idx] if 0 <= eta_idx < x.shape[1] else np.zeros(n)
        phi = x[:, phi_idx] if 0 <= phi_idx < x.shape[1] else np.zeros(n)

    pt = np.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    pt_pos = np.clip(pt, 0.0, None)
    phi_wrapped = delta_phi(phi)

    if n > 0:
        sum_hadron_pt = float(np.sum(pt_pos))
        mean_hadron_pt = float(np.mean(pt_pos))
        lead_hadron_pt = float(np.max(pt_pos))
    else:
        sum_hadron_pt = 0.0
        mean_hadron_pt = 0.0
        lead_hadron_pt = 0.0

    width_eta = float(np.std(eta)) if n > 1 else 0.0
    width_phi = float(np.std(phi_wrapped)) if n > 1 else 0.0

    weighted_width_eta = weighted_std(eta, pt_pos + 1e-12)
    weighted_width_phi = weighted_std(phi_wrapped, pt_pos + 1e-12)

    if n > 0:
        delta_r = np.sqrt(eta ** 2 + phi_wrapped ** 2)
        rms_deltaR = float(np.sqrt(np.mean(delta_r ** 2)))
    else:
        rms_deltaR = 0.0

    e = get_electron_array(data)
    electron_pt = 0.0
    electron_eta = 0.0
    electron_phi = 0.0
    electron_charge = 0.0

    if e is not None and e.size > 0:
        if 0 <= e_pt_idx < e.size:
            electron_pt = float(e[e_pt_idx])
        if 0 <= e_eta_idx < e.size:
            electron_eta = float(e[e_eta_idx])
        if 0 <= e_phi_idx < e.size:
            electron_phi = float(e[e_phi_idx])
        electron_charge = float(e[-1])

    feats = {
        "n_hadron": float(n),
        "sum_hadron_pt": sum_hadron_pt,
        "mean_hadron_pt": mean_hadron_pt,
        "lead_hadron_pt": lead_hadron_pt,
        "width_eta": width_eta,
        "width_phi": width_phi,
        "weighted_width_eta": weighted_width_eta,
        "weighted_width_phi": weighted_width_phi,
        "rms_deltaR": rms_deltaR,
        "electron_pt": electron_pt,
        "electron_eta": electron_eta,
        "electron_phi": electron_phi,
        "electron_charge": electron_charge,
    }

    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


# ============================================================
# 5. Build BDT table
# ============================================================

def build_feature_table(args, dataset, model=None, device="cpu"):
    rows = []

    for i, data in enumerate(dataset):
        y = infer_label(data)

        feats = build_physics_features(
            data,
            pt_idx=args.pt_idx,
            eta_idx=args.eta_idx,
            phi_idx=args.phi_idx,
            e_pt_idx=args.e_pt_idx,
            e_eta_idx=args.e_eta_idx,
            e_phi_idx=args.e_phi_idx,
        )

        if args.use_score:
            feats["s_TF"] = model_forward_score(model, data, device)

        feats["label"] = y
        feats["event_index"] = i
        rows.append(feats)

        if (i + 1) % 100 == 0:
            print(f"[features] processed {i + 1}/{len(dataset)}")

    df = pd.DataFrame(rows)

    hadron_feature_cols = [
        "n_hadron",
        "sum_hadron_pt",
        "mean_hadron_pt",
        "lead_hadron_pt",
        "width_eta",
        "width_phi",
        "weighted_width_eta",
        "weighted_width_phi",
        "rms_deltaR",
    ]

    electron_feature_cols = [
        "electron_pt",
        "electron_eta",
        "electron_phi",
        "electron_charge",
    ]

    feature_cols = []

    if args.use_score:
        feature_cols.append("s_TF")

    if args.feature_set in ["hadron", "all"]:
        feature_cols.extend(hadron_feature_cols)

    if args.feature_set in ["electron", "all"]:
        feature_cols.extend(electron_feature_cols)

    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols


# ============================================================
# 6. BDT training and plots
# ============================================================

def train_bdt(args, df, feature_cols):
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.seed,
    )

    clf.fit(X_train, y_train)

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, prob_train)
    auc_test = roc_auc_score(y_test, prob_test)

    pred_test = (prob_test > 0.5).astype(int)
    acc_test = accuracy_score(y_test, pred_test)

    print("\n========== BDT result ==========")
    print(f"feature_set = {args.feature_set}")
    print(f"use_score   = {args.use_score}")
    print(f"features    = {feature_cols}")
    print(f"AUC train   = {auc_train:.6f}")
    print(f"AUC test    = {auc_test:.6f}")
    print(f"ACC test    = {acc_test:.6f}")
    print("================================\n")

    return {
        "clf": clf,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prob_train": prob_train,
        "prob_test": prob_test,
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "acc_test": float(acc_test),
    }


def plot_roc(result, outdir, tag):
    y_test = result["y_test"]
    prob_test = result["prob_test"]
    auc_test = result["auc_test"]

    fpr, tpr, _ = roc_curve(y_test, prob_test)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, label=f"BDT, AUC={auc_test:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(frameon=False)
    plt.tight_layout()

    path_pdf = os.path.join(outdir, f"roc_{tag}.pdf")
    path_png = os.path.join(outdir, f"roc_{tag}.png")
    plt.savefig(path_pdf)
    plt.savefig(path_png, dpi=200)
    plt.close()

    print(f"[plot] saved {path_pdf}")


def plot_feature_importance(result, outdir, tag):
    clf = result["clf"]
    feature_cols = result["feature_cols"]

    if not hasattr(clf, "feature_importances_"):
        return

    imp = np.asarray(clf.feature_importances_)
    order = np.argsort(imp)

    plt.figure(figsize=(6.2, 4.8))
    plt.barh(np.array(feature_cols)[order], imp[order])
    plt.xlabel("Feature importance")
    plt.tight_layout()

    path_pdf = os.path.join(outdir, f"feature_importance_{tag}.pdf")
    path_png = os.path.join(outdir, f"feature_importance_{tag}.png")
    plt.savefig(path_pdf)
    plt.savefig(path_png, dpi=200)
    plt.close()

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    csv_path = os.path.join(outdir, f"feature_importance_{tag}.csv")
    df_imp.to_csv(csv_path, index=False)

    print(f"[plot] saved {path_pdf}")
    print(f"[csv] saved {csv_path}")


def plot_permutation_importance(result, outdir, tag, seed=42):
    clf = result["clf"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    feature_cols = result["feature_cols"]

    perm = permutation_importance(
        clf,
        X_test,
        y_test,
        n_repeats=20,
        random_state=seed,
        scoring="roc_auc",
    )

    mean = perm.importances_mean
    std = perm.importances_std
    order = np.argsort(mean)

    plt.figure(figsize=(6.2, 4.8))
    plt.barh(np.array(feature_cols)[order], mean[order], xerr=std[order])
    plt.xlabel("Permutation importance in AUC")
    plt.tight_layout()

    path_pdf = os.path.join(outdir, f"permutation_importance_{tag}.pdf")
    path_png = os.path.join(outdir, f"permutation_importance_{tag}.png")
    plt.savefig(path_pdf)
    plt.savefig(path_png, dpi=200)
    plt.close()

    df_perm = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": mean,
        "perm_importance_std": std,
    }).sort_values("perm_importance_mean", ascending=False)

    csv_path = os.path.join(outdir, f"permutation_importance_{tag}.csv")
    df_perm.to_csv(csv_path, index=False)

    print(f"[plot] saved {path_pdf}")
    print(f"[csv] saved {csv_path}")


# ============================================================
# 7. Args and main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True,
                        help="Path to PyG data .pt file.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained TF checkpoint. Required if --use_score.")

    parser.add_argument(
        "--weight_dir",
        type=str,
        default=str(THIS_DIR / "weight_of_BDT"),
        help="Directory to save trained BDT weights.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(THIS_DIR / "results_bdt"),
        help="Directory to save plots, csv, npz.",
    )

    parser.add_argument("--use_score", dest="use_score", action="store_true")
    parser.add_argument("--no_use_score", dest="use_score", action="store_false")
    parser.set_defaults(use_score=True)

    parser.add_argument(
        "--feature_set",
        type=str,
        default="hadron",
        choices=["hadron", "electron", "all"],
        help="Which hand-crafted observables to use.",
    )

    parser.add_argument("--max_per_class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Model construction parameters.
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Hadron feature column indices.
    parser.add_argument("--pt_idx", type=int, default=0)
    parser.add_argument("--eta_idx", type=int, default=1)
    parser.add_argument("--phi_idx", type=int, default=2)

    # Electron feature column indices.
    parser.add_argument("--e_pt_idx", type=int, default=0)
    parser.add_argument("--e_eta_idx", type=int, default=1)
    parser.add_argument("--e_phi_idx", type=int, default=2)

    # BDT hyperparameters.
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_estimators", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=0.8)

    parser.add_argument("--tag", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.weight_dir)
    ensure_dir(args.outdir)

    if args.use_score and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --use_score is enabled.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[path] THIS_DIR      = {THIS_DIR}")
    print(f"[path] BENCHMARK_DIR = {BENCHMARK_DIR}")
    print(f"[path] PYG_BM_DIR    = {PYG_BM_DIR}")
    print(f"[device] {device}")

    print(f"[data] loading {args.data}")
    dataset = load_dataset(args.data)
    print(f"[data] loaded {len(dataset)} events")

    dataset = db_balance_and_limit(
        dataset,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    model = None
    if args.use_score:
        model = build_model(args)
        model = load_checkpoint_to_model(model, args.checkpoint, device)

    df, feature_cols = build_feature_table(
        args,
        dataset,
        model=model,
        device=device,
    )

    tag = args.tag
    if tag is None:
        score_tag = "score" if args.use_score else "noscore"
        tag = f"{score_tag}_{args.feature_set}_max{args.max_per_class}"

    feature_table_path = os.path.join(args.outdir, f"feature_table_{tag}.csv")
    df.to_csv(feature_table_path, index=False)
    print(f"[csv] saved {feature_table_path}")

    result = train_bdt(args, df, feature_cols)

    # Save BDT weight to weight_of_BDT.
    bdt_weight_path = os.path.join(args.weight_dir, f"bdt_{tag}.joblib")
    joblib.dump(
        {
            "clf": result["clf"],
            "feature_cols": result["feature_cols"],
            "args": vars(args),
        },
        bdt_weight_path,
    )
    print(f"[weight] saved {bdt_weight_path}")

    # Save numerical result.
    npz_path = os.path.join(args.outdir, f"bdt_result_{tag}.npz")
    np.savez(
        npz_path,
        X_train=result["X_train"],
        X_test=result["X_test"],
        y_train=result["y_train"],
        y_test=result["y_test"],
        prob_train=result["prob_train"],
        prob_test=result["prob_test"],
        auc_train=result["auc_train"],
        auc_test=result["auc_test"],
        acc_test=result["acc_test"],
        feature_cols=np.array(result["feature_cols"], dtype=object),
    )
    print(f"[npz] saved {npz_path}")

    summary = {
        "tag": tag,
        "use_score": args.use_score,
        "feature_set": args.feature_set,
        "features": result["feature_cols"],
        "auc_train": result["auc_train"],
        "auc_test": result["auc_test"],
        "acc_test": result["acc_test"],
        "n_total_after_balance_limit": len(df),
        "max_per_class": args.max_per_class,
        "bdt_weight_path": bdt_weight_path,
    }

    summary_path = os.path.join(args.outdir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] saved {summary_path}")

    plot_roc(result, args.outdir, tag)
    plot_feature_importance(result, args.outdir, tag)
    plot_permutation_importance(result, args.outdir, tag, seed=args.seed)


if __name__ == "__main__":
    main()