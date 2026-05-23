#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train diagnostic BDT from ROOT using the same ROOT branches and selection logic
as PyG_BM/data_HFSemiClassifier.py.

python train_bdt_from_root_with_tf.py \
    --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_1_allAccept.root \
    --use-score \
    --feature-set hadron \
    --max-per-class 500 \
    --tag score_hadron

python train_bdt_from_root_with_tf.py \
    --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_1_allAccept.root \
    --feature-set hadron \
    --max-per-class 500 \
    --tag handcrafted_hadron

python train_bdt_from_root_with_tf.py \
    --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_1_allAccept.root \
    --use-score \
    --feature-set hadron \
    --max-per-class 500 \
    --n-estimators 150 \
    --max-depth 3 \
    --tag score_hadron_deeper

Workflow
--------
ROOT file
  -> read tree="tree"
  -> lightweight preselection over electrons
  -> pT-bin D/B balance first:
         n_keep = min(nD, nB, max_per_class)
  -> build hand-crafted features only for selected electrons
  -> optionally load TransformerHF and compute s_TF only for selected electrons
  -> train BDT
  -> save BDT weight to benchmark/BDT/weight_of_BDT/

Recommended location
--------------------
/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/train_bdt_from_root_with_tf.py
"""

import os
import sys
import json
import random
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib


# ============================================================
# paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = THIS_DIR.parent
PYG_BM_DIR = BENCHMARK_DIR / "PyG_BM"

DEFAULT_ROOT_FILE = (
    "/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/"
    "ppHF_eXDecay_5B_1_allAccept.root"
)

DEFAULT_TF_CKPT = (
    PYG_BM_DIR
    / "Weight_of_Model"
    / "transformer"
    / "TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt"
)

DEFAULT_WEIGHT_DIR = THIS_DIR / "weight_of_BDT"
DEFAULT_OUTDIR = THIS_DIR / "results_bdt"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PYG_BM_DIR))


# ============================================================
# model candidates
# ============================================================
# Transformer model in PyG_BM/train_HFSemiClassifier.py is SetTransformerHF
# from model_HFSemiClassifier.
MODEL_CANDIDATES = [
    ("model_HFSemiClassifier", "SetTransformerHF"),
    ("model_HFSemiClassifier", "TransformerHF"),
    ("model_HFSemiClassifier", "TransformerClassifier"),
]


# ============================================================
# utils
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ak_to_np(x, dtype=np.float32):
    return np.asarray(ak.to_numpy(x), dtype=dtype)


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


# ============================================================
# load Transformer
# ============================================================

def find_model_class():
    for module_name, class_name in MODEL_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"[TF] using {module_name}.{class_name}")
                return getattr(module, class_name)
        except Exception:
            continue

    raise ImportError(
        "Cannot import SetTransformerHF from PyG_BM/model_HFSemiClassifier.py. "
        "Please check file/class name."
    )


def build_tf_model(args):
    """
    Match the commented Transformer baseline in PyG_BM/train_HFSemiClassifier.py:

    SetTransformerHF(
        had_input_dim=5,
        ele_input_dim=3,
        d_model=256,
        nhead=4,
        num_layers=args.num_layers,
        dim_feedforward=512,
        dropout=0.1,
        n_classes=2,
    )
    """
    ModelClass = find_model_class()

    model = ModelClass(
        had_input_dim=5,
        ele_input_dim=3,
        d_model=args.tf_d_model,
        nhead=args.tf_nhead,
        num_layers=args.tf_num_layers,
        dim_feedforward=args.tf_dim_feedforward,
        dropout=args.tf_dropout,
        n_classes=2,
    )

    return model


def load_tf_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    clean = {}
    for k, v in ckpt.items():
        nk = k
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        clean[nk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)

    print(f"[TF] checkpoint loaded: {ckpt_path}")
    print(f"[TF] missing keys: {len(missing)}")
    print(f"[TF] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[TF] first missing:", missing[:8])
    if len(unexpected) > 0:
        print("[TF] first unexpected:", unexpected[:8])

    model.to(device)
    model.eval()
    return model


def compute_tf_score(model, ele_feat_np, had_feat_np, device):
    """
    Same forward signature as training:
        logits = model(ele, had, mask)

    ele:  (B, 3)
    had:  (B, N, 5)
    mask: (B, N)
    """
    ele = torch.tensor(ele_feat_np[None, :], dtype=torch.float32, device=device)

    if had_feat_np.shape[0] == 0:
        had = torch.zeros((1, 0, 5), dtype=torch.float32, device=device)
        mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
    else:
        had = torch.tensor(had_feat_np[None, :, :], dtype=torch.float32, device=device)
        mask = torch.ones((1, had_feat_np.shape[0]), dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(ele, had, mask)
        prob = torch.softmax(logits, dim=-1)
        s = prob[0, 1].item()

    return float(s)


# ============================================================
# ROOT reading and sample construction
# ============================================================

def load_root_arrays(root_file, tree_name="tree"):
    branches = [
        "nEle",
        "ele_charge",
        "ele_pt",
        "ele_eta",
        "ele_phi",
        "ele_hf_TAG",
        "nHad_away",
        "had_fromEle",
        "had_charge",
        "had_pt",
        "had_eta",   # actually Δη
        "had_phi",   # actually Δφ
    ]

    print(f"[ROOT] file = {root_file}")
    print(f"[ROOT] tree = {tree_name}")
    print("[ROOT] branches:")
    for b in branches:
        print(f"  - {b}")

    with uproot.open(root_file) as f:
        tree = f[tree_name]
        arrays = tree.arrays(branches, library="ak")

    print(f"[ROOT] events = {len(arrays['nEle'])}")
    return arrays


def pass_dphi_windows(had_dphi_all, dphi_windows):
    if dphi_windows is None:
        return np.ones_like(had_dphi_all, dtype=bool)

    mask = np.zeros_like(had_dphi_all, dtype=bool)
    for low_phi, high_phi in dphi_windows:
        mask |= (had_dphi_all >= low_phi) & (had_dphi_all < high_phi)

    return mask


def build_one_sample(arrays, evt, i_ele, args):
    """
    Reproduce HFSemiClassifier.__getitem__ logic, but also return raw arrays
    for physics-feature construction.

    Selection logic follows:
      - ele_hf_TAG in (1, 3)
      - electron pt_min <= pt < pt_max
      - |ele_eta| <= eta_abs_max if eta_abs_max is not None
      - hadrons with had_fromEle == i_ele
      - hadron global eta cut: |ele_eta + had_deta| <= eta_abs_max
      - hadron dphi windows if provided
      - hadron pt cut
      - min_had
    """
    eta_e = float(arrays["ele_eta"][evt][i_ele])
    pt_e = float(arrays["ele_pt"][evt][i_ele])
    phi_e = float(arrays["ele_phi"][evt][i_ele])
    charge_e = float(arrays["ele_charge"][evt][i_ele])

    if args.eta_abs_max is not None and abs(eta_e) > args.eta_abs_max:
        return None

    raw_tag = int(arrays["ele_hf_TAG"][evt][i_ele])
    if raw_tag not in (1, 3):
        return None

    if args.pt_min is not None and pt_e < args.pt_min:
        return None
    if args.pt_max is not None and pt_e >= args.pt_max:
        return None

    label = 0 if raw_tag == 1 else 1

    had_fromEle_evt = ak_to_np(arrays["had_fromEle"][evt], dtype=np.int32)
    base_mask = had_fromEle_evt == i_ele

    had_pt_all = ak_to_np(arrays["had_pt"][evt][base_mask], dtype=np.float32)
    had_deta_all = ak_to_np(arrays["had_eta"][evt][base_mask], dtype=np.float32)
    had_dphi_all = ak_to_np(arrays["had_phi"][evt][base_mask], dtype=np.float32)
    had_charge_all = ak_to_np(arrays["had_charge"][evt][base_mask], dtype=np.float32)

    if args.eta_abs_max is not None and had_deta_all.size > 0:
        had_eta_global = had_deta_all + eta_e
        mask_eta = np.abs(had_eta_global) <= args.eta_abs_max
    else:
        mask_eta = np.ones_like(had_pt_all, dtype=bool)

    mask_dphi = pass_dphi_windows(had_dphi_all, args.dphi_windows)

    mask = mask_eta & mask_dphi

    if args.had_pt_min is not None:
        mask &= had_pt_all >= args.had_pt_min
    if args.had_pt_max is not None:
        mask &= had_pt_all < args.had_pt_max

    had_pt = had_pt_all[mask]
    had_deta = had_deta_all[mask]
    had_dphi = had_dphi_all[mask]
    had_charge = had_charge_all[mask]

    if int(had_pt.size) < args.min_had:
        return None

    # same as training: use_log_pt=False by default
    if args.use_log_pt:
        ele_pt_feat = np.log(pt_e + 1e-6)
        had_pt_feat = np.log(had_pt + 1e-6)
    else:
        ele_pt_feat = pt_e
        had_pt_feat = had_pt

    ele_feat = np.array(
        [ele_pt_feat, eta_e, charge_e],
        dtype=np.float32,
    )

    if had_pt.size > 0:
        if args.use_had_eta:
            had_deta_feat = had_deta
        else:
            had_deta_feat = np.zeros_like(had_deta, dtype=np.float32)

        had_feat = np.stack(
            [
                had_pt_feat,
                had_deta_feat,
                np.sin(had_dphi),
                np.cos(had_dphi),
                had_charge,
            ],
            axis=-1,
        ).astype(np.float32)
    else:
        had_feat = np.zeros((0, 5), dtype=np.float32)

    return {
        "label": label,
        "pt_e": pt_e,
        "eta_e": eta_e,
        "phi_e": phi_e,
        "charge_e": charge_e,
        "had_pt": np.asarray(had_pt, dtype=np.float32),
        "had_deta": np.asarray(had_deta, dtype=np.float32),
        "had_dphi": np.asarray(had_dphi, dtype=np.float32),
        "had_charge": np.asarray(had_charge, dtype=np.float32),
        "ele_feat": ele_feat,
        "had_feat": had_feat,
    }


# ============================================================
# pT-bin preselection and balance first
# ============================================================

def parse_pt_edges_for_bdt(args):
    """
    Same pT-bin logic as PyG_BM train_HFSemiClassifier.py.

    If --ds-pt-edges is provided:
        use explicit edges, e.g. "3,4,5,6,8,10"

    Otherwise:
        build uniform bins from pt_min to pt_max with ds_pt_bin_width.
    """
    if args.ds_pt_edges.strip():
        edges = [float(x) for x in args.ds_pt_edges.split(",")]
        edges = sorted(edges)
        if len(edges) < 2:
            raise ValueError("--ds-pt-edges must have >=2 numbers")
        return np.array(edges, dtype=np.float32)

    if args.pt_min is None or args.pt_max is None:
        raise ValueError("Need --pt-min and --pt-max to auto-build pt bins")

    w = float(args.ds_pt_bin_width)
    if w <= 0:
        raise ValueError("--ds-pt-bin-width must be > 0")

    edges = [float(args.pt_min)]
    x = float(args.pt_min)

    while x + w < float(args.pt_max) - 1e-6:
        x += w
        edges.append(x)

    edges.append(float(args.pt_max))
    return np.array(edges, dtype=np.float32)


def lightweight_pass_selection(arrays, evt, i_ele, args):
    """
    Lightweight version of selection.

    It does NOT build hand-crafted features and does NOT compute Transformer score.
    It only returns a record needed for pT-bin balance.

    It still applies min_had using the same hadron cuts, so selected records should
    also pass build_one_sample later.
    """
    eta_e = float(arrays["ele_eta"][evt][i_ele])
    pt_e = float(arrays["ele_pt"][evt][i_ele])

    if args.eta_abs_max is not None and abs(eta_e) > args.eta_abs_max:
        return None

    raw_tag = int(arrays["ele_hf_TAG"][evt][i_ele])
    if raw_tag not in (1, 3):
        return None

    if args.pt_min is not None and pt_e < args.pt_min:
        return None
    if args.pt_max is not None and pt_e >= args.pt_max:
        return None

    label = 0 if raw_tag == 1 else 1

    # Check min_had with the same hadron-level cuts.
    if args.min_had > 0:
        had_fromEle_evt = ak_to_np(arrays["had_fromEle"][evt], dtype=np.int32)
        base_mask = had_fromEle_evt == i_ele

        had_pt_all = ak_to_np(arrays["had_pt"][evt][base_mask], dtype=np.float32)
        had_deta_all = ak_to_np(arrays["had_eta"][evt][base_mask], dtype=np.float32)
        had_dphi_all = ak_to_np(arrays["had_phi"][evt][base_mask], dtype=np.float32)

        if args.eta_abs_max is not None and had_deta_all.size > 0:
            had_eta_global = had_deta_all + eta_e
            mask_eta = np.abs(had_eta_global) <= args.eta_abs_max
        else:
            mask_eta = np.ones_like(had_pt_all, dtype=bool)

        mask_dphi = pass_dphi_windows(had_dphi_all, args.dphi_windows)

        mask = mask_eta & mask_dphi

        if args.had_pt_min is not None:
            mask &= had_pt_all >= args.had_pt_min
        if args.had_pt_max is not None:
            mask &= had_pt_all < args.had_pt_max

        if int(mask.sum()) < args.min_had:
            return None

    return {
        "event": int(evt),
        "ele_index": int(i_ele),
        "electron_pt": float(pt_e),
        "label": int(label),
    }


def select_records_by_ptbin_balance(arrays, args):
    """
    First scan ROOT with lightweight selection, then do pT-bin D/B balance.

    For each pT bin:
        n_keep = min(nD, nB, args.max_per_class)

    Return selected records:
        [{"event": evt, "ele_index": i_ele, "electron_pt": pt, "label": label}, ...]
    """
    pt_edges = parse_pt_edges_for_bdt(args)

    pools = {}
    for b in range(len(pt_edges) - 1):
        pools[(b, 0)] = []
        pools[(b, 1)] = []

    n_events = len(arrays["nEle"])

    print(f"[preselect] pt edges = {pt_edges.tolist()}")
    print("[preselect] scanning ROOT for candidate electrons...")

    for evt in range(n_events):
        n_ele_evt = int(arrays["nEle"][evt])

        for i_ele in range(n_ele_evt):
            rec = lightweight_pass_selection(arrays, evt, i_ele, args)
            if rec is None:
                continue

            pt = rec["electron_pt"]
            label = rec["label"]

            b = int(np.searchsorted(pt_edges, pt, side="right") - 1)
            if 0 <= b < len(pt_edges) - 1:
                pools[(b, label)].append(rec)

        if (evt + 1) % args.print_every == 0:
            total_now = sum(len(v) for v in pools.values())
            print(f"[preselect] event {evt + 1}/{n_events}, candidates = {total_now}")

    selected = []
    rng = np.random.default_rng(args.seed)

    total_d = sum(len(pools[(b, 0)]) for b in range(len(pt_edges) - 1))
    total_b = sum(len(pools[(b, 1)]) for b in range(len(pt_edges) - 1))

    print(f"[ptbin-balance] before total candidates: D={total_d}, B={total_b}")

    for b in range(len(pt_edges) - 1):
        low = float(pt_edges[b])
        high = float(pt_edges[b + 1])

        pool_d = pools[(b, 0)]
        pool_b = pools[(b, 1)]

        nD = len(pool_d)
        nB = len(pool_b)

        if nD == 0 or nB == 0:
            print(
                f"[ptbin-balance] bin {low:.2f}-{high:.2f}: "
                f"D={nD}, B={nB}, keep(each)=0, skipped"
            )
            continue

        n_keep = min(nD, nB, args.max_per_class)

        idx_d = rng.choice(nD, size=n_keep, replace=False)
        idx_b = rng.choice(nB, size=n_keep, replace=False)

        selected.extend([pool_d[i] for i in idx_d])
        selected.extend([pool_b[i] for i in idx_b])

        print(
            f"[ptbin-balance] bin {low:.2f}-{high:.2f}: "
            f"D={nD}, B={nB}, keep(each)={n_keep}"
        )

    if len(selected) == 0:
        raise RuntimeError("No samples selected after pT-bin D/B balancing.")

    rng.shuffle(selected)

    after_d = sum(1 for r in selected if r["label"] == 0)
    after_b = sum(1 for r in selected if r["label"] == 1)

    print(f"[ptbin-balance] after selected: D={after_d}, B={after_b}, total={len(selected)}")

    return selected


# ============================================================
# physics features
# ============================================================

def build_physics_features(sample):
    had_pt = sample["had_pt"]
    had_deta = sample["had_deta"]
    had_dphi = delta_phi(sample["had_dphi"])
    had_charge = sample["had_charge"]

    n = len(had_pt)
    had_pt_pos = np.clip(had_pt, 0.0, None)

    if n > 0:
        sum_hadron_pt = float(np.sum(had_pt_pos))
        mean_hadron_pt = float(np.mean(had_pt_pos))
        lead_hadron_pt = float(np.max(had_pt_pos))
        delta_r = np.sqrt(had_deta ** 2 + had_dphi ** 2)
        rms_deltaR = float(np.sqrt(np.mean(delta_r ** 2)))
        min_deltaR = float(np.min(delta_r))
        sum_abs_charge = float(np.sum(np.abs(had_charge)))
        sum_charge = float(np.sum(had_charge))
    else:
        sum_hadron_pt = 0.0
        mean_hadron_pt = 0.0
        lead_hadron_pt = 0.0
        rms_deltaR = 0.0
        min_deltaR = 0.0
        sum_abs_charge = 0.0
        sum_charge = 0.0

    width_eta = float(np.std(had_deta)) if n > 1 else 0.0
    width_phi = float(np.std(had_dphi)) if n > 1 else 0.0
    weighted_width_eta = weighted_std(had_deta, had_pt_pos + 1e-12)
    weighted_width_phi = weighted_std(had_dphi, had_pt_pos + 1e-12)

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
        "min_deltaR": min_deltaR,
        "sum_hadron_charge": sum_charge,
        "sum_abs_hadron_charge": sum_abs_charge,

        "electron_pt": float(sample["pt_e"]),
        "electron_eta": float(sample["eta_e"]),
        "electron_phi": float(sample["phi_e"]),
        "electron_charge": float(sample["charge_e"]),
    }

    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


# ============================================================
# build dataframe only after balance
# ============================================================

def build_dataframe_from_selected_records(args, arrays, selected_records, tf_model=None, device="cpu"):
    """
    Build hand-crafted features and optional s_TF only for selected records.
    """
    rows = []
    n_total = len(selected_records)

    print(f"[build] building features only for selected electrons: {n_total}")

    for i, rec in enumerate(selected_records):
        evt = rec["event"]
        i_ele = rec["ele_index"]

        sample = build_one_sample(arrays, evt, i_ele, args)
        if sample is None:
            # This should not happen because lightweight preselection applies same cuts.
            continue

        row = build_physics_features(sample)
        row["label"] = int(sample["label"])
        row["event"] = int(evt)
        row["ele_index"] = int(i_ele)

        if args.use_score:
            row["s_TF"] = compute_tf_score(
                tf_model,
                sample["ele_feat"],
                sample["had_feat"],
                device,
            )

        rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"[build] selected electron {i + 1}/{n_total}")

    df = pd.DataFrame(rows)

    print("[build] final selected electrons:")
    print(df["label"].value_counts().sort_index())

    return df


def get_feature_columns(args):
    hadron_cols = [
        "n_hadron",
        "sum_hadron_pt",
        "mean_hadron_pt",
        "lead_hadron_pt",
        "width_eta",
        "width_phi",
        "weighted_width_eta",
        "weighted_width_phi",
        "rms_deltaR",
        "min_deltaR",
        "sum_hadron_charge",
        "sum_abs_hadron_charge",
    ]

    electron_cols = [
        "electron_pt",
        "electron_eta",
        "electron_phi",
        "electron_charge",
    ]

    cols = []
    if args.use_score:
        cols.append("s_TF")

    if args.feature_set in ["hadron", "all"]:
        cols.extend(hadron_cols)

    if args.feature_set in ["electron", "all"]:
        cols.extend(electron_cols)

    return cols


# ============================================================
# BDT
# ============================================================

def train_bdt(args, df, feature_cols):
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
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
    acc_test = accuracy_score(y_test, (prob_test > 0.5).astype(int))

    print("\n========== BDT result ==========")
    print(f"use_score   = {args.use_score}")
    print(f"feature_set = {args.feature_set}")
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
    fpr, tpr, _ = roc_curve(result["y_test"], result["prob_test"])

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, label=f"BDT, AUC={result['auc_test']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(frameon=False)
    plt.tight_layout()

    pdf = os.path.join(outdir, f"roc_{tag}.pdf")
    png = os.path.join(outdir, f"roc_{tag}.png")
    plt.savefig(pdf)
    plt.savefig(png, dpi=200)
    plt.close()

    print(f"[plot] saved {pdf}")


def plot_feature_importance(result, outdir, tag):
    clf = result["clf"]
    feature_cols = result["feature_cols"]
    imp = np.asarray(clf.feature_importances_)

    order = np.argsort(imp)

    plt.figure(figsize=(6.8, 5.0))
    plt.barh(np.array(feature_cols)[order], imp[order])
    plt.xlabel("Feature importance")
    plt.tight_layout()

    pdf = os.path.join(outdir, f"feature_importance_{tag}.pdf")
    png = os.path.join(outdir, f"feature_importance_{tag}.png")
    plt.savefig(pdf)
    plt.savefig(png, dpi=200)
    plt.close()

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    csv = os.path.join(outdir, f"feature_importance_{tag}.csv")
    df_imp.to_csv(csv, index=False)

    print(f"[plot] saved {pdf}")
    print(f"[csv] saved {csv}")


def plot_permutation_importance(result, outdir, tag, seed):
    perm = permutation_importance(
        result["clf"],
        result["X_test"],
        result["y_test"],
        scoring="roc_auc",
        n_repeats=20,
        random_state=seed,
    )

    feature_cols = result["feature_cols"]
    mean = perm.importances_mean
    std = perm.importances_std

    order = np.argsort(mean)

    plt.figure(figsize=(6.8, 5.0))
    plt.barh(np.array(feature_cols)[order], mean[order], xerr=std[order])
    plt.xlabel("Permutation importance in AUC")
    plt.tight_layout()

    pdf = os.path.join(outdir, f"permutation_importance_{tag}.pdf")
    png = os.path.join(outdir, f"permutation_importance_{tag}.png")
    plt.savefig(pdf)
    plt.savefig(png, dpi=200)
    plt.close()

    df_perm = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": mean,
        "perm_importance_std": std,
    }).sort_values("perm_importance_mean", ascending=False)

    csv = os.path.join(outdir, f"permutation_importance_{tag}.csv")
    df_perm.to_csv(csv, index=False)

    print(f"[plot] saved {pdf}")
    print(f"[csv] saved {csv}")


def save_outputs(args, df, result):
    ensure_dir(args.weight_dir)
    ensure_dir(args.outdir)

    tag = args.tag

    feature_table_path = os.path.join(args.outdir, f"feature_table_{tag}.csv")
    df.to_csv(feature_table_path, index=False)
    print(f"[csv] saved {feature_table_path}")

    weight_path = os.path.join(args.weight_dir, f"bdt_{tag}.joblib")
    joblib.dump(
        {
            "clf": result["clf"],
            "feature_cols": result["feature_cols"],
            "args": vars(args),
        },
        weight_path,
    )
    print(f"[weight] saved {weight_path}")

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
        "root_file": args.root_file,
        "tree": "tree",
        "use_score": args.use_score,
        "tf_checkpoint": str(args.tf_checkpoint),
        "feature_set": args.feature_set,
        "features": result["feature_cols"],
        "auc_train": result["auc_train"],
        "auc_test": result["auc_test"],
        "acc_test": result["acc_test"],
        "max_per_class_per_ptbin": args.max_per_class,
        "ds_pt_bin_width": args.ds_pt_bin_width,
        "ds_pt_edges": args.ds_pt_edges,
        "n_after_ptbin_balance_limit": len(df),
        "weight_path": weight_path,
        "feature_table_path": feature_table_path,
        "selection": {
            "pt_min": args.pt_min,
            "pt_max": args.pt_max,
            "eta_abs_max": args.eta_abs_max,
            "use_log_pt": args.use_log_pt,
            "use_had_eta": args.use_had_eta,
            "had_pt_min": args.had_pt_min,
            "had_pt_max": args.had_pt_max,
            "min_had": args.min_had,
            "dphi_windows": args.dphi_windows,
        },
    }

    summary_path = os.path.join(args.outdir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] saved {summary_path}")

    plot_roc(result, args.outdir, tag)
    plot_feature_importance(result, args.outdir, tag)
    plot_permutation_importance(result, args.outdir, tag, args.seed)


# ============================================================
# args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root-file",
        type=str,
        default=DEFAULT_ROOT_FILE,
        help="ROOT file generated by ppHF_eXDecay.",
    )

    parser.add_argument(
        "--tf-checkpoint",
        type=str,
        default=str(DEFAULT_TF_CKPT),
        help="trained Transformer checkpoint.",
    )

    parser.add_argument("--use-score", dest="use_score", action="store_true")
    parser.add_argument("--no-use-score", dest="use_score", action="store_false")
    parser.set_defaults(use_score=False)

    parser.add_argument(
        "--feature-set",
        type=str,
        default="hadron",
        choices=["hadron", "electron", "all"],
    )

    # Same selection defaults as train_HFSemiClassifier.py
    parser.add_argument("--pt-min", type=float, default=3.0)
    parser.add_argument("--pt-max", type=float, default=10.0)
    parser.add_argument("--eta-abs-max", type=float, default=5.0)
    parser.add_argument("--use-log-pt", action="store_true", default=False)
    parser.add_argument("--no-use-had-eta", dest="use_had_eta", action="store_false")
    parser.set_defaults(use_had_eta=True)

    parser.add_argument("--had-pt-min", type=float, default=0.2)
    parser.add_argument("--had-pt-max", type=float, default=None)
    parser.add_argument("--min-had", type=int, default=4)

    # Optional dphi cuts. Default is None, same as training.
    parser.add_argument(
        "--dphi-windows",
        type=str,
        default="",
        help=(
            "Optional dphi windows, e.g. '-3.14,-1.57;1.57,3.14'. "
            "Empty means no dphi cut."
        ),
    )

    # pT-bin balance settings
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Maximum number of D and B samples kept in each electron-pT bin.",
    )
    parser.add_argument(
        "--ds-pt-bin-width",
        type=float,
        default=0.25,
        help="Electron-pT bin width for pT-bin D/B balancing.",
    )
    parser.add_argument(
        "--ds-pt-edges",
        type=str,
        default="",
        help="Optional pT bin edges, e.g. '3,4,5,6,8,10'. If non-empty, overrides --ds-pt-bin-width.",
    )

    parser.add_argument("--seed", type=int, default=42)

    # Transformer architecture for the given checkpoint
    parser.add_argument("--tf-d-model", type=int, default=256)
    parser.add_argument("--tf-nhead", type=int, default=4)
    parser.add_argument("--tf-num-layers", type=int, default=4)
    parser.add_argument("--tf-dim-feedforward", type=int, default=512)
    parser.add_argument("--tf-dropout", type=float, default=0.1)

    # BDT hyperparameters
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=0.8)

    parser.add_argument("--weight-dir", type=str, default=str(DEFAULT_WEIGHT_DIR))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--tag", type=str, default=None)

    parser.add_argument("--print-every", type=int, default=10000)

    args = parser.parse_args()

    if args.dphi_windows.strip():
        windows = []
        for item in args.dphi_windows.split(";"):
            low, high = item.split(",")
            windows.append((float(low), float(high)))
        args.dphi_windows = windows
    else:
        args.dphi_windows = None

    if args.tag is None:
        score_tag = "score" if args.use_score else "noscore"
        args.tag = f"{score_tag}_{args.feature_set}_ptbinmax{args.max_per_class}"

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.weight_dir)
    ensure_dir(args.outdir)

    print("========== Configuration ==========")
    print(f"THIS_DIR       = {THIS_DIR}")
    print(f"PYG_BM_DIR     = {PYG_BM_DIR}")
    print(f"root_file      = {args.root_file}")
    print(f"tree           = tree")
    print(f"use_score      = {args.use_score}")
    print(f"tf_checkpoint  = {args.tf_checkpoint}")
    print(f"feature_set    = {args.feature_set}")
    print(f"ptbin max/class= {args.max_per_class}")
    print(f"ds_pt_bin_width= {args.ds_pt_bin_width}")
    print(f"ds_pt_edges    = {args.ds_pt_edges}")
    print(f"weight_dir     = {args.weight_dir}")
    print(f"outdir         = {args.outdir}")
    print(f"tag            = {args.tag}")
    print("===================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    arrays = load_root_arrays(args.root_file, tree_name="tree")

    # Important: balance happens before feature building and before s_TF computation.
    selected_records = select_records_by_ptbin_balance(arrays, args)

    tf_model = None
    if args.use_score:
        if not os.path.exists(args.tf_checkpoint):
            raise FileNotFoundError(f"TF checkpoint not found: {args.tf_checkpoint}")
        tf_model = build_tf_model(args)
        tf_model = load_tf_checkpoint(tf_model, args.tf_checkpoint, device)

    df = build_dataframe_from_selected_records(
        args,
        arrays,
        selected_records,
        tf_model=tf_model,
        device=device,
    )

    feature_cols = get_feature_columns(args)
    result = train_bdt(args, df, feature_cols)
    save_outputs(args, df, result)


if __name__ == "__main__":
    main()