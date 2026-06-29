#!/usr/bin/env python3
# plot_s_physics_profiles_transformer.py
#
# Purpose:
#   Plot inclusive s-physics profiles for selected handcrafted observables.
#
#   s = logit_B - logit_D
#
#   The observables are NOT model inputs as handcrafted summaries.
#   They are computed after inference from the same hadron point cloud
#   to interpret how the learned score varies with physical quantities.
#
# Features:
#   - Transformer checkpoint
#   - D/B pT-bin balancing
#   - optional per-pT-bin cap / limitation
#   - inclusive <s>(O) profiles, not separated by truth D/B
#   - representative observables:
#       width/topology:
#           mean_abs_dphi, std_abs_dphi, mean_abs_deta, std_abs_deta
#       hardness/activity:
#           mean_had_pt, sum_had_pt, lead_had_pt, std_had_pt
#       momentum concentration:
#           pt_conc
#       electron-level control:
#           e_pt, e_eta, e_q
#       charge correlation:
#           same_sign_frac
#
# Example:
# cd /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM
#
"""




 \
  --ckpt /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt \
  --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
  --pt-min 3.0 \
  --pt-max 10.0 \
  --balance-ds \
  --max-keep-per-bin-per-class 500 \
  --out-dir /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan/TF4plotwithData \
  --tag transformer_s_physics
"""


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


# -----------------------------
# utils
# -----------------------------
def plot_truth_fraction(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    title: str,
    out_pdf: str,
    n_bins: int = 30,
    x_quantile_clip: float = 0.005,
    min_count: int = 30,
):
    ensure_dir_for_file(out_pdf)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    good = np.isfinite(x) & ((y == 0) | (y == 1))
    x = x[good]
    y = y[good]

    lo = np.quantile(x, x_quantile_clip)
    hi = np.quantile(x, 1.0 - x_quantile_clip)

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bfrac = np.full(n_bins, np.nan)
    berr = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    bid = np.searchsorted(edges, x, side="right") - 1

    for b in range(n_bins):
        sel = bid == b
        n = int(np.sum(sel))
        if n < min_count:
            continue

        nb = int(np.sum(y[sel] == 1))
        f = nb / n

        bfrac[b] = f
        berr[b] = np.sqrt(f * (1.0 - f) / n)
        count[b] = n

    good = np.isfinite(bfrac)

    plt.figure(figsize=(6.2, 4.8))
    plt.errorbar(
        centers[good],
        bfrac[good],
        yerr=berr[good],
        fmt="o",
        markersize=4,
        linewidth=1.4,
        capsize=2.5,
    )

    plt.axhline(0.5, linestyle="--", linewidth=1.0)
    plt.ylim(0.0, 1.0)

    plt.xlabel(x_label)
    plt.ylabel(r"$N_B/(N_D+N_B)$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


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


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)

    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j

    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan
    return pearsonr(rankdata(x), rankdata(y))


# -----------------------------
# dataset balancing
# -----------------------------
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


# -----------------------------
# observables
# -----------------------------
def compute_observables(batch: Dict[str, torch.Tensor], use_log_pt: bool) -> Dict[str, np.ndarray]:
    ele = batch["ele_feat"].cpu().numpy()
    had = batch["had_feat"].cpu().numpy()
    msk = batch["had_mask"].cpu().numpy().astype(bool)

    B, _, _ = had.shape

    e_pt_feat = ele[:, 0]
    e_pt = np.exp(e_pt_feat) if use_log_pt else e_pt_feat
    e_eta = ele[:, 1]
    e_q = ele[:, 2]

    n_had = msk.sum(axis=1).astype(np.float64)

    had_pt_feat = had[:, :, 0]
    had_pt = np.exp(had_pt_feat) if use_log_pt else had_pt_feat
    had_pt = np.where(msk, had_pt, 0.0)

    had_deta = np.where(msk, had[:, :, 1], 0.0)
    had_abs_deta = np.abs(had_deta)

    had_sin = np.where(msk, had[:, :, 2], 0.0)
    had_cos = np.where(msk, had[:, :, 3], 1.0)
    had_q = np.where(msk, had[:, :, 4], 0.0)

    had_dphi = np.arctan2(had_sin, had_cos)
    had_abs_dphi = np.abs(had_dphi)

    sum_had_pt = had_pt.sum(axis=1)
    mean_had_pt = np.divide(sum_had_pt, np.maximum(n_had, 1.0))
    std_had_pt = np.array(
        [safe_std(had_pt[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )

    mean_abs_dphi = np.array(
        [safe_mean(had_abs_dphi[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )
    std_abs_dphi = np.array(
        [safe_std(had_abs_dphi[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )
    mean_abs_deta = np.array(
        [safe_mean(had_abs_deta[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )
    std_abs_deta = np.array(
        [safe_std(had_abs_deta[i, msk[i]]) for i in range(B)],
        dtype=np.float64,
    )

    lead_pt = np.zeros(B, dtype=np.float64)
    lead_abs_dphi = np.zeros(B, dtype=np.float64)
    lead_abs_deta = np.zeros(B, dtype=np.float64)
    same_sign_frac = np.zeros(B, dtype=np.float64)

    for i in range(B):
        vi = msk[i]
        if vi.sum() == 0:
            continue

        pt_i = had_pt[i, vi]
        idx = int(np.argmax(pt_i))
        pos = np.where(vi)[0][idx]

        lead_pt[i] = float(had_pt[i, pos])
        lead_abs_dphi[i] = float(had_abs_dphi[i, pos])
        lead_abs_deta[i] = float(had_abs_deta[i, pos])

        q_i = had_q[i, vi]
        same_sign_frac[i] = float(np.mean((q_i * e_q[i]) > 0.0))

    pt_conc = np.divide(lead_pt, np.maximum(sum_had_pt, 1e-12))

    return {
        # electron-level checks
        "e_pt": e_pt.astype(np.float64),
        "e_eta": e_eta.astype(np.float64),
        "e_q": e_q.astype(np.float64),

        # multiplicity / activity
        "n_had": n_had.astype(np.float64),
        "sum_had_pt": sum_had_pt.astype(np.float64),
        "mean_had_pt": mean_had_pt.astype(np.float64),
        "std_had_pt": std_had_pt.astype(np.float64),
        "lead_had_pt": lead_pt.astype(np.float64),
        "pt_conc": pt_conc.astype(np.float64),

        # geometry / topology
        "mean_abs_dphi": mean_abs_dphi.astype(np.float64),
        "std_abs_dphi": std_abs_dphi.astype(np.float64),
        "lead_abs_dphi": lead_abs_dphi.astype(np.float64),
        "mean_abs_deta": mean_abs_deta.astype(np.float64),
        "std_abs_deta": std_abs_deta.astype(np.float64),
        "lead_abs_deta": lead_abs_deta.astype(np.float64),

        # charge correlation
        "same_sign_frac": same_sign_frac.astype(np.float64),
    }


# -----------------------------
# model
# -----------------------------
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


# -----------------------------
# profile
# -----------------------------
def make_inclusive_profile(
    x: np.ndarray,
    s: np.ndarray,
    n_bins: int = 30,
    x_quantile_clip: float = 0.005,
    min_count: int = 30,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(s)
    x = x[finite]
    s = s[finite]

    if x.size < min_count:
        return {}

    lo = np.quantile(x, x_quantile_clip)
    hi = np.quantile(x, 1.0 - x_quantile_clip)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))

    if hi <= lo:
        return {}

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_id = np.searchsorted(edges, x, side="right") - 1

    mean_s = np.full(n_bins, np.nan)
    median_s = np.full(n_bins, np.nan)
    q16_s = np.full(n_bins, np.nan)
    q84_s = np.full(n_bins, np.nan)
    std_s = np.full(n_bins, np.nan)
    err_mean_s = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        sel = bin_id == b
        n = int(np.sum(sel))
        if n >= min_count:
            vals = s[sel]
            mean_s[b] = float(np.mean(vals))
            median_s[b] = float(np.median(vals))
            q16_s[b] = float(np.quantile(vals, 0.16))
            q84_s[b] = float(np.quantile(vals, 0.84))
            std_s[b] = float(np.std(vals))
            err_mean_s[b] = float(std_s[b] / np.sqrt(max(n, 1)))
            count[b] = n

    profile_range = np.nan
    good = np.isfinite(mean_s)
    if np.sum(good) >= 2:
        profile_range = float(np.nanmax(mean_s[good]) - np.nanmin(mean_s[good]))

    return {
        "centers": centers,
        "edges": edges,
        "mean_s": mean_s,
        "median_s": median_s,
        "q16_s": q16_s,
        "q84_s": q84_s,
        "std_s": std_s,
        "err_mean_s": err_mean_s,
        "count": count,
        "profile_range": np.array([profile_range], dtype=np.float64),
    }


def smooth_guide_line(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    frac: float = 0.10,
    n_grid: int = 300,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Local Gaussian weighted smoothing for guide-to-the-eye curve.

    This is not a model fit and not an uncertainty estimate.
    The smooth curve is only used to guide the eye.
    """
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x = np.asarray(x[good], dtype=np.float64)
    y = np.asarray(y[good], dtype=np.float64)
    yerr = np.asarray(yerr[good], dtype=np.float64)

    if x.size < 4:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return None

    x_grid = np.linspace(xmin, xmax, n_grid)

    bw = float(frac) * (xmax - xmin)
    if bw <= 0:
        return None

    stat_w = 1.0 / np.maximum(yerr, 1e-8) ** 2
    y_smooth = np.full_like(x_grid, np.nan, dtype=np.float64)

    for i, x0 in enumerate(x_grid):
        local_w = np.exp(-0.5 * ((x - x0) / bw) ** 2)
        w = local_w * stat_w
        sw = np.sum(w)

        if sw > 0:
            y_smooth[i] = np.sum(w * y) / sw

    return {
        "x_grid": x_grid,
        "y_smooth": y_smooth,
    }


def plot_profile(
    prof: Dict[str, np.ndarray],
    x_label: str,
    title: str,
    out_pdf: str,
    band: str = "stderr",
) -> None:
    ensure_dir_for_file(out_pdf)

    x = prof["centers"]
    y = prof["mean_s"]
    yerr = prof["err_mean_s"]

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)

    plt.figure(figsize=(6.2, 4.8))

    smooth = smooth_guide_line(
        x=x[good],
        y=y[good],
        yerr=yerr[good],
        frac=0.05,
        n_grid=300,
    )

    if smooth is not None:
        plt.plot(
            smooth["x_grid"],
            smooth["y_smooth"],
            linewidth=2.0,
            alpha=0.80,
            label="model score trend",
            zorder=2,
        )

    plt.errorbar(
        x[good],
        y[good],
        yerr=yerr[good],
        fmt="o",
        markersize=4,
        linewidth=1.4,
        capsize=2.5,
        label=r"profile model score $\langle s\rangle$",
        zorder=3,
    )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.ylim(-2.2, 1.0)

    plt.xlabel(x_label)
    plt.ylabel(r"$score s,\ \log(N_B/N_D)$")
    plt.title(title)
    plt.grid(True)
    # plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


# -----------------------------
# selected observables
# -----------------------------
def selected_observables() -> List[Tuple[str, str, str]]:
    """
    Return:
      key, display label, category
    """
    return [
        # Width / topology
        ("mean_abs_dphi", r"mean($|\Delta\phi|$)", "width_topology"),
        ("std_abs_dphi", r"std($|\Delta\phi|$)", "width_topology"),
        ("mean_abs_deta", r"mean($|\Delta\eta|$)", "width_topology"),
        ("std_abs_deta", r"std($|\Delta\eta|$)", "width_topology"),

        # Hardness / activity
        ("mean_had_pt", r"mean hadron $p_T$", "hardness_activity"),
        ("sum_had_pt", r"sum hadron $p_T$", "hardness_activity"),
        ("lead_had_pt", r"leading hadron $p_T$", "hardness_activity"),
        ("std_had_pt", r"std(hadron $p_T$)", "hardness_activity"),

        # Momentum concentration
        ("pt_conc", r"$p_T$ concentration", "momentum_concentration"),

        # Electron-level controls
        ("e_pt", r"electron $p_T$", "electron_control"),
        ("e_eta", r"electron $\eta$", "electron_control"),
        ("e_q", r"electron charge", "electron_control"),

        # Charge correlation
        ("same_sign_frac", "same-sign fraction", "charge_correlation"),
    ]


# -----------------------------
# main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Inclusive s-physics profiles for Transformer.")

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

    # dataset cuts
    p.add_argument("--pt-min", type=float, default=3.0)
    p.add_argument("--pt-max", type=float, default=10.0)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true")

    # balance + limitation
    p.add_argument("--balance-ds", action="store_true")
    p.add_argument("--balance-frac", type=float, default=1.0)
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")
    p.add_argument("--max-keep-per-bin-per-class", type=int, default=1000)

    # profile
    p.add_argument("--profile-bins", type=int, default=30)
    p.add_argument("--x-quantile-clip", type=float, default=0.005)
    p.add_argument("--min-count-per-bin", type=int, default=30)
    p.add_argument(
        "--band",
        type=str,
        default="stderr",
        choices=["stderr", "quantile", "std"],
        help=(
            "Kept only for backward compatibility. "
            "The current plot always uses point with errorbar and a smooth guide line."
        ),
    )

    # output
    p.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan",
    )
    p.add_argument("--tag", type=str, default="transformer_s_physics")

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

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    tag = sanitize_filename(args.tag)

    truth_ratio_dir = os.path.join(args.out_dir, f"{tag}_truth_fraction")
    ensure_dir(truth_ratio_dir)

    truth_ratio_obs = [
        ("e_pt", r"electron $p_T$"),
        ("mean_had_pt", r"mean hadron $p_T$"),
        ("std_abs_dphi", r"std($|\Delta\phi|$)"),
        ("mean_abs_dphi", r"mean($|\Delta\phi|$)"),
    ]

    for key, label in truth_ratio_obs:
        out_pdf = os.path.join(
            truth_ratio_dir,
            f"{tag}_B_fraction_vs_{sanitize_filename(key)}.pdf"
        )

        plot_truth_fraction(
            x=obs_all[key],
            y=y_all,
            x_label=label,
            title=f"Truth B fraction vs {label}",
            out_pdf=out_pdf,
            n_bins=int(args.profile_bins),
            x_quantile_clip=float(args.x_quantile_clip),
            min_count=int(args.min_count_per_bin),
        )
        print(f"[INFO] saved truth fraction plot: {out_pdf}")

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    print(f"[INFO] final N = {len(y_all)} | D = {np.sum(y_all == 0)} | B = {np.sum(y_all == 1)}")

    # Save arrays
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
    print(f"[INFO] saved: {npz_path}")

    # Plot selected observables
    plot_dir = os.path.join(args.out_dir, f"{tag}_profiles")
    ensure_dir(plot_dir)

    rows = []

    for key, label, category in selected_observables():
        if key not in obs_all:
            print(f"[WARN] skip missing observable: {key}")
            continue

        prof = make_inclusive_profile(
            x=obs_all[key],
            s=s_all,
            n_bins=int(args.profile_bins),
            x_quantile_clip=float(args.x_quantile_clip),
            min_count=int(args.min_count_per_bin),
        )

        if not prof:
            print(f"[WARN] skip invalid profile: {key}")
            continue

        safe_key = sanitize_filename(key)
        out_pdf = os.path.join(plot_dir, f"{tag}_s_vs_{safe_key}.pdf")

        plot_profile(
            prof=prof,
            x_label=label,
            title=f"Score profile vs {label}",
            out_pdf=out_pdf,
            band=args.band,
        )

        x = obs_all[key]
        pear = pearsonr(x, s_all)
        spear = spearmanr(x, s_all)
        prof_range = float(prof["profile_range"][0]) if np.isfinite(prof["profile_range"][0]) else np.nan

        rows.append({
            "observable": key,
            "label": label,
            "category": category,
            "pearson": pear,
            "spearman": spear,
            "profile_range": prof_range,
            "plot": out_pdf,
        })

        print(
            f"[INFO] saved {out_pdf} | "
            f"{key}: pearson={pear:+.4f}, spearman={spear:+.4f}, profile_range={prof_range:.4f}"
        )

    # Save summary CSV
    csv_path = os.path.join(args.out_dir, f"{tag}_selected_s_physics_summary.csv")
    ensure_dir_for_file(csv_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "observable",
            "label",
            "category",
            "pearson_s_obs",
            "spearman_s_obs",
            "profile_range_mean_s",
            "plot_file",
        ])

        for r in rows:
            writer.writerow([
                r["observable"],
                r["label"],
                r["category"],
                f"{r['pearson']:.8g}" if np.isfinite(r["pearson"]) else "nan",
                f"{r['spearman']:.8g}" if np.isfinite(r["spearman"]) else "nan",
                f"{r['profile_range']:.8g}" if np.isfinite(r["profile_range"]) else "nan",
                r["plot"],
            ])

    print(f"[INFO] saved: {csv_path}")

    # Save human-readable text summary grouped by category
    txt_path = os.path.join(args.out_dir, f"{tag}_selected_s_physics_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Inclusive s-physics profiles for Transformer\n")
        f.write(f"ckpt = {args.ckpt}\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"N = {len(y_all)}\n")
        f.write(f"D = {int(np.sum(y_all == 0))}\n")
        f.write(f"B = {int(np.sum(y_all == 1))}\n")
        f.write(f"balance_ds = {bool(args.balance_ds)}\n")
        f.write(f"balance_frac = {float(args.balance_frac)}\n")
        f.write(f"max_keep_per_bin_per_class = {int(args.max_keep_per_bin_per_class)}\n")
        f.write(f"profile_bins = {int(args.profile_bins)}\n")
        f.write("plot_style = point_with_errorbar_plus_local_weighted_smooth_guide_frac_0p05\n\n")

        categories = [
            "width_topology",
            "hardness_activity",
            "momentum_concentration",
            "electron_control",
            "charge_correlation",
        ]

        for cat in categories:
            f.write(f"[{cat}]\n")
            for r in rows:
                if r["category"] != cat:
                    continue
                f.write(
                    f"{r['observable']:18s} "
                    f"pearson={r['pearson']:+.6f} "
                    f"spearman={r['spearman']:+.6f} "
                    f"profile_range={r['profile_range']:.6f} "
                    f"plot={r['plot']}\n"
                )
            f.write("\n")

    print(f"[INFO] saved: {txt_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()