#!/usr/bin/env python3
# scan_s_delta_s_vs_physics_transformer.py
#
# Transformer version:
#   1. load Transformer checkpoint
#   2. compute s = logit_B - logit_D
#   3. compute handcrafted physics observables
#   4. make two series of plots:
#        A) <s>_D(O), <s>_B(O) vs O
#        B) Delta s(O) = <s>_B(O) - <s>_D(O) vs O
#   5. save two correlation/ranking csv files:
#        corr_s_vs_physics.csv
#        corr_delta_s_vs_physics.csv
#
# Example:
# python scan_s_delta_s_vs_physics_transformer.py \
#   --ckpt /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt \
#   --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
'''
--pt-min 3.0 \
  --pt-max 10.0 \
  --balance-ds \
  --max-keep-per-bin-per-class 500 \
  --out-dir /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan

python scan_s_delta_s_vs_physics_transformer.py \
  --ckpt /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/transformer/TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt \
  --root-file /mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_5B_2_allAccept.root \
  --pt-min 3.0 \
  --pt-max 10.0 \
  --balance-ds \
  --max-keep-per-bin-per-class 500 \
  --out-dir /mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan_morebeautiful \
  --tag transformer
'''

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


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 1 else 0.0


# -----------------------------
# balancing
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
    max_keep_per_bin_per_class: int = 99999,
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
        [safe_mean(np.abs(had_deta[i, msk[i]])) for i in range(B)],
        dtype=np.float64,
    )

    std_deta = np.array(
        [safe_std(had_deta[i, msk[i]]) for i in range(B)],
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
        lead_abs_deta[i] = float(abs(had_deta[i, pos]))

        q_i = had_q[i, vi]
        same_sign_frac[i] = float(np.mean((q_i * e_q[i]) > 0.0))

    pt_conc = np.divide(lead_pt, np.maximum(sum_had_pt, 1e-12))

    return {
        "e_pt": e_pt.astype(np.float64),
        "e_eta": e_eta.astype(np.float64),
        "e_q": e_q.astype(np.float64),
        "n_had": n_had.astype(np.float64),
        "sum_had_pt": sum_had_pt.astype(np.float64),
        "mean_had_pt": mean_had_pt.astype(np.float64),
        "std_had_pt": std_had_pt.astype(np.float64),
        "lead_had_pt": lead_pt.astype(np.float64),
        "pt_conc": pt_conc.astype(np.float64),
        "mean_abs_dphi": mean_abs_dphi.astype(np.float64),
        "std_abs_dphi": std_abs_dphi.astype(np.float64),
        "lead_abs_dphi": lead_abs_dphi.astype(np.float64),
        "mean_abs_deta": mean_abs_deta.astype(np.float64),
        "std_deta": std_deta.astype(np.float64),
        "lead_abs_deta": lead_abs_deta.astype(np.float64),
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
def evaluate_transformer_scores_and_obs(
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
# profile and delta profile
# -----------------------------
def make_profile_arrays(
    x: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    n_bins: int = 30,
    x_quantile_clip: float = 0.005,
    min_count: int = 20,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    finite = np.isfinite(x) & np.isfinite(s)
    x = x[finite]
    s = s[finite]
    y = y[finite]

    lo = np.quantile(x, x_quantile_clip)
    hi = np.quantile(x, 1.0 - x_quantile_clip)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))

    if hi <= lo:
        return {}

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def prof(mask):
        xm = x[mask]
        sm = s[mask]
        bin_id = np.searchsorted(edges, xm, side="right") - 1

        mu = np.full(n_bins, np.nan)
        sd = np.full(n_bins, np.nan)
        q16 = np.full(n_bins, np.nan)
        q84 = np.full(n_bins, np.nan)
        cnt = np.zeros(n_bins, dtype=np.int64)

        for b in range(n_bins):
            sel = bin_id == b
            if np.sum(sel) >= min_count:
                vals = sm[sel]
                mu[b] = float(np.mean(vals))
                sd[b] = float(np.std(vals))
                q16[b] = float(np.quantile(vals, 0.16))
                q84[b] = float(np.quantile(vals, 0.84))
                cnt[b] = int(np.sum(sel))

        return mu, sd, q16, q84, cnt

    isD = y == 0
    isB = y == 1

    muD, sdD, q16D, q84D, cD = prof(isD)
    muB, sdB, q16B, q84B, cB = prof(isB)

    delta = muB - muD

    # Error on mean difference, only for visual reference:
    # sigma_delta = sqrt(sdB^2 / nB + sdD^2 / nD)
    se_delta = np.full(n_bins, np.nan)
    good = (cB > 0) & (cD > 0) & np.isfinite(sdB) & np.isfinite(sdD)
    se_delta[good] = np.sqrt((sdB[good] ** 2) / cB[good] + (sdD[good] ** 2) / cD[good])

    return {
        "centers": centers,
        "edges": edges,
        "muD": muD,
        "sdD": sdD,
        "q16D": q16D,
        "q84D": q84D,
        "cD": cD,
        "muB": muB,
        "sdB": sdB,
        "q16B": q16B,
        "q84B": q84B,
        "cB": cB,
        "delta": delta,
        "se_delta": se_delta,
    }


def plot_s_vs_physics(
    prof: Dict[str, np.ndarray],
    var_label: str,
    out_pdf: str,
    band: str = "quantile",
) -> None:
    ensure_dir_for_file(out_pdf)

    x = prof["centers"]

    plt.figure(figsize=(6.2, 4.8))

    if band == "std":
        loD = prof["muD"] - prof["sdD"]
        hiD = prof["muD"] + prof["sdD"]
        loB = prof["muB"] - prof["sdB"]
        hiB = prof["muB"] + prof["sdB"]
        band_label = r"$\pm 1\sigma_s$"
    else:
        loD = prof["q16D"]
        hiD = prof["q84D"]
        loB = prof["q16B"]
        hiB = prof["q84B"]
        band_label = "16--84%"

    if np.any(np.isfinite(prof["muD"])):
        plt.plot(x, prof["muD"], label="True D: mean(s)")
        plt.fill_between(x, loD, hiD, alpha=0.20, label=f"True D {band_label}")

    if np.any(np.isfinite(prof["muB"])):
        plt.plot(x, prof["muB"], label="True B: mean(s)")
        plt.fill_between(x, loB, hiB, alpha=0.20, label=f"True B {band_label}")

    plt.xlabel(var_label)
    plt.ylabel("s = logit_B - logit_D")
    plt.title(rf"Score profile vs {var_label}")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


def plot_delta_s_vs_physics(
    prof: Dict[str, np.ndarray],
    var_label: str,
    out_pdf: str,
) -> None:
    ensure_dir_for_file(out_pdf)

    x = prof["centers"]
    delta = prof["delta"]
    se = prof["se_delta"]

    plt.figure(figsize=(6.2, 4.4))

    plt.plot(x, delta, marker="o", linewidth=1.8, label=r"$\Delta s = \langle s\rangle_B - \langle s\rangle_D$")

    good = np.isfinite(delta) & np.isfinite(se)
    if np.any(good):
        plt.fill_between(x, delta - se, delta + se, alpha=0.20, label="error on mean difference")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel(var_label)
    plt.ylabel(r"$\Delta s$")
    plt.title(rf"$\Delta s$ profile vs {var_label}")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


def make_quantile_delta_arrays(
    x: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    n_bins: int = 30,
    min_count: int = 20,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    finite = np.isfinite(x) & np.isfinite(s)
    x = x[finite]
    s = s[finite]
    y = y[finite]

    if x.size < n_bins:
        return {}

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    s = s[order]
    y = y[order]

    percentiles = 100.0 * (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    delta = np.full(n_bins, np.nan)
    se_delta = np.full(n_bins, np.nan)
    cD = np.zeros(n_bins, dtype=np.int64)
    cB = np.zeros(n_bins, dtype=np.int64)

    split_edges = np.linspace(0, x.size, n_bins + 1).astype(int)

    for b in range(n_bins):
        i0 = split_edges[b]
        i1 = split_edges[b + 1]

        xb = x[i0:i1]
        sb = s[i0:i1]
        yb = y[i0:i1]

        if xb.size == 0:
            continue

        sD = sb[yb == 0]
        sB = sb[yb == 1]
        cD[b] = int(sD.size)
        cB[b] = int(sB.size)

        if sD.size >= min_count and sB.size >= min_count:
            delta[b] = float(np.mean(sB) - np.mean(sD))
            se_delta[b] = float(np.sqrt(np.var(sB, ddof=1) / sB.size + np.var(sD, ddof=1) / sD.size))

    return {
        "percentiles": percentiles,
        "delta": delta,
        "se_delta": se_delta,
        "cD": cD,
        "cB": cB,
        "value_min": np.array([float(np.min(x))], dtype=np.float64),
        "value_max": np.array([float(np.max(x))], dtype=np.float64),
    }


def plot_combined_delta_s_quantile(
    obs_all: Dict[str, np.ndarray],
    s: np.ndarray,
    y: np.ndarray,
    out_pdf: str,
    observables: List[str],
    n_bins: int = 30,
    min_count: int = 20,
) -> None:
    ensure_dir_for_file(out_pdf)

    plt.figure(figsize=(7.2, 4.8))

    style_map = {
        "mean_had_pt": dict(marker="s", linestyle="-"),
        "mean_abs_dphi": dict(marker="o", linestyle="--"),
        "std_abs_dphi": dict(marker="o", linestyle=":"),
    }
    fillstyle_map = {
        "mean_had_pt": "full",
        "mean_abs_dphi": "full",
        "std_abs_dphi": "none",
    }

    for vn in observables:
        if vn not in obs_all:
            continue

        prof = make_quantile_delta_arrays(
            x=obs_all[vn],
            s=s,
            y=y,
            n_bins=n_bins,
            min_count=min_count,
        )

        if not prof:
            continue

        xq = prof["percentiles"]
        delta = prof["delta"]
        se = prof["se_delta"]
        good = np.isfinite(delta) & np.isfinite(se)

        if not np.any(good):
            continue

        vmin = float(prof["value_min"][0])
        vmax = float(prof["value_max"][0])
        label_map = {
            "mean_had_pt": r"$\langle p_T^{\rm hadron}\rangle$",
            "mean_abs_dphi": r"$\langle |\Delta\phi| \rangle$",
            "std_abs_dphi": r"$\sigma(|\Delta\phi|)$",
        }
        label = f"{label_map.get(vn, vn)} [{vmin:.3g}, {vmax:.3g}]"

        # plt.errorbar(
        #     xq[good],
        #     delta[good],
        #     yerr=se[good],
        #     marker="o",
        #     linewidth=1.2,
        #     markersize=4,
        #     capsize=2,
        #     label=label,
        # )

        style = style_map.get(vn, dict(marker="o", linestyle="-"))
        fillstyle = fillstyle_map.get(vn, "full")

        plt.errorbar(
            xq[good],
            delta[good],
            yerr=se[good],
            marker=style["marker"],
            linestyle=style["linestyle"],
            fillstyle=fillstyle,
            linewidth=1.2,
            markersize=4.5,
            capsize=2,
            label=label,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Observable percentile")
    plt.ylabel(r"$\Delta s = \langle s\rangle_B - \langle s\rangle_D$")
    plt.title(r"$\Delta s$ vs selected hadronic-environment observables")
    plt.grid(True)
    plt.legend(title="Observable [0--100% value range]", loc="best", fontsize=8, title_fontsize=8)

    plt.xlim(-2, 102)
    plt.xticks([0, 20, 40, 60, 80, 100])

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=180)
    plt.close()


# -----------------------------
# ranking
# -----------------------------
def compute_s_correlations(
    s: np.ndarray,
    y: np.ndarray,
    obs_all: Dict[str, np.ndarray],
) -> List[Tuple]:
    rows = []

    mD = y == 0
    mB = y == 1

    for vn in sorted(obs_all.keys()):
        x = obs_all[vn]

        pear_all = pearsonr(x, s)
        spear_all = spearmanr(x, s)

        pear_D = pearsonr(x[mD], s[mD]) if np.sum(mD) >= 3 else np.nan
        spear_D = spearmanr(x[mD], s[mD]) if np.sum(mD) >= 3 else np.nan

        pear_B = pearsonr(x[mB], s[mB]) if np.sum(mB) >= 3 else np.nan
        spear_B = spearmanr(x[mB], s[mB]) if np.sum(mB) >= 3 else np.nan

        rows.append((vn, pear_all, spear_all, pear_D, spear_D, pear_B, spear_B))

    rows = sorted(rows, key=lambda r: -(abs(r[1]) if np.isfinite(r[1]) else -1.0))
    return rows


def compute_delta_s_correlations(
    profile_map: Dict[str, Dict[str, np.ndarray]],
) -> List[Tuple]:
    rows = []

    for vn, prof in profile_map.items():
        x = prof["centers"]
        delta = prof["delta"]
        cD = prof["cD"]
        cB = prof["cB"]

        valid = np.isfinite(x) & np.isfinite(delta) & (cD > 0) & (cB > 0)

        if np.sum(valid) < 3:
            pear = np.nan
            spear = np.nan
            slope = np.nan
            mean_delta = np.nan
            weighted_mean_delta = np.nan
            delta_range = np.nan
        else:
            xv = x[valid]
            dv = delta[valid]
            w = (cD[valid] + cB[valid]).astype(np.float64)

            pear = pearsonr(xv, dv)
            spear = spearmanr(xv, dv)

            # linear slope of delta vs observable center
            try:
                slope = float(np.polyfit(xv, dv, 1)[0])
            except Exception:
                slope = np.nan

            mean_delta = float(np.mean(dv))
            weighted_mean_delta = float(np.average(dv, weights=w))
            delta_range = float(np.nanmax(dv) - np.nanmin(dv))

        rows.append((vn, pear, spear, slope, mean_delta, weighted_mean_delta, delta_range))

    rows = sorted(rows, key=lambda r: -(abs(r[1]) if np.isfinite(r[1]) else -1.0))
    return rows


def save_s_corr_csv(rows: List[Tuple], out_csv: str) -> None:
    ensure_dir_for_file(out_csv)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "observable",
            "pearson_all",
            "spearman_all",
            "pearson_D",
            "spearman_D",
            "pearson_B",
            "spearman_B",
            "rank_by_abs_pearson_all",
        ])

        for i, r in enumerate(rows, 1):
            writer.writerow([
                r[0],
                f"{r[1]:.8g}" if np.isfinite(r[1]) else "nan",
                f"{r[2]:.8g}" if np.isfinite(r[2]) else "nan",
                f"{r[3]:.8g}" if np.isfinite(r[3]) else "nan",
                f"{r[4]:.8g}" if np.isfinite(r[4]) else "nan",
                f"{r[5]:.8g}" if np.isfinite(r[5]) else "nan",
                f"{r[6]:.8g}" if np.isfinite(r[6]) else "nan",
                i,
            ])


def save_delta_corr_csv(rows: List[Tuple], out_csv: str) -> None:
    ensure_dir_for_file(out_csv)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "observable",
            "pearson_delta_vs_obs",
            "spearman_delta_vs_obs",
            "slope_delta_vs_obs",
            "mean_delta_s",
            "weighted_mean_delta_s",
            "delta_s_range",
            "rank_by_abs_pearson_delta",
        ])

        for i, r in enumerate(rows, 1):
            writer.writerow([
                r[0],
                f"{r[1]:.8g}" if np.isfinite(r[1]) else "nan",
                f"{r[2]:.8g}" if np.isfinite(r[2]) else "nan",
                f"{r[3]:.8g}" if np.isfinite(r[3]) else "nan",
                f"{r[4]:.8g}" if np.isfinite(r[4]) else "nan",
                f"{r[5]:.8g}" if np.isfinite(r[5]) else "nan",
                f"{r[6]:.8g}" if np.isfinite(r[6]) else "nan",
                i,
            ])


# -----------------------------
# main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Transformer scan: s-physics and delta_s-physics.")

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

    # balancing
    p.add_argument("--balance-ds", action="store_true")
    p.add_argument("--balance-frac", type=float, default=1.0)
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")
    p.add_argument("--max-keep-per-bin-per-class", type=int, default=99999)

    # profile
    p.add_argument("--profile-bins", type=int, default=10)
    p.add_argument("--x-quantile-clip", type=float, default=0.005)
    p.add_argument("--min-count-per-bin", type=int, default=20)
    p.add_argument(
        "--band",
        type=str,
        default="quantile",
        choices=["quantile", "std"],
        help="Band for s-vs-physics plots. quantile = 16--84%, std = mean +/- std.",
    )

    # output
    p.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="transformer",
        help="Prefix tag for output files.",
    )

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

        print(f"[INFO] balance-ds: {len(idx_all)} -> {len(idx_eval)}")

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

    y_all, s_all, obs_all = evaluate_transformer_scores_and_obs(
        model=model,
        loader=loader,
        device=device,
        use_log_pt=use_log_pt,
    )

    print(f"[INFO] final N = {len(y_all)} | D = {np.sum(y_all == 0)} | B = {np.sum(y_all == 1)}")

    tag = sanitize_filename(args.tag)

    # Save event-level arrays for later combined-cut studies.
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

    # Correlation ranking: s vs physics
    s_rows = compute_s_correlations(s_all, y_all, obs_all)
    s_corr_csv = os.path.join(args.out_dir, f"{tag}_corr_s_vs_physics.csv")
    save_s_corr_csv(s_rows, s_corr_csv)
    print(f"[INFO] saved: {s_corr_csv}")

    print("\n[TOP s-physics correlations by |pearson_all|]")
    for r in s_rows[:10]:
        print(
            f"  {r[0]:16s}  pearson_all={r[1]:+.4f}  spearman_all={r[2]:+.4f}  "
            f"D={r[3]:+.4f}  B={r[5]:+.4f}"
        )

    # Profiles and delta-s profiles
    profile_map: Dict[str, Dict[str, np.ndarray]] = {}

    s_profile_dir = os.path.join(args.out_dir, f"{tag}_s_vs_physics_plots")
    delta_profile_dir = os.path.join(args.out_dir, f"{tag}_delta_s_vs_physics_plots")
    ensure_dir(s_profile_dir)
    ensure_dir(delta_profile_dir)

    for vn in sorted(obs_all.keys()):
        prof = make_profile_arrays(
            x=obs_all[vn],
            s=s_all,
            y=y_all,
            n_bins=int(args.profile_bins),
            x_quantile_clip=float(args.x_quantile_clip),
            min_count=int(args.min_count_per_bin),
        )

        if not prof:
            print(f"[WARN] skip {vn}: invalid profile")
            continue

        profile_map[vn] = prof

        safe_vn = sanitize_filename(vn)

        out_s_pdf = os.path.join(s_profile_dir, f"{tag}_s_vs_{safe_vn}.pdf")
        plot_s_vs_physics(
            prof=prof,
            var_label=vn,
            out_pdf=out_s_pdf,
            band=args.band,
        )

        out_delta_pdf = os.path.join(delta_profile_dir, f"{tag}_delta_s_vs_{safe_vn}.pdf")
        plot_delta_s_vs_physics(
            prof=prof,
            var_label=vn,
            out_pdf=out_delta_pdf,
        )

        print(f"[INFO] saved plots for {vn}")

    combined_delta_pdf = os.path.join(args.out_dir, f"{tag}_delta_s_selected_quantile.pdf")
    plot_combined_delta_s_quantile(
        obs_all=obs_all,
        s=s_all,
        y=y_all,
        out_pdf=combined_delta_pdf,
        observables=["mean_had_pt", "mean_abs_dphi", "std_abs_dphi"],
        n_bins=int(args.profile_bins),
        min_count=int(args.min_count_per_bin),
    )

    # Correlation ranking: delta s vs physics
    delta_rows = compute_delta_s_correlations(profile_map)
    delta_corr_csv = os.path.join(args.out_dir, f"{tag}_corr_delta_s_vs_physics.csv")
    save_delta_corr_csv(delta_rows, delta_corr_csv)
    print(f"[INFO] saved: {delta_corr_csv}")

    print("\n[TOP delta-s physics correlations by |pearson_delta_vs_obs|]")
    for r in delta_rows[:10]:
        print(
            f"  {r[0]:16s}  pearson_delta={r[1]:+.4f}  "
            f"spearman_delta={r[2]:+.4f}  slope={r[3]:+.4f}  "
            f"mean_delta={r[4]:+.4f}"
        )

    # Also save a small text summary.
    summary_txt = os.path.join(args.out_dir, f"{tag}_scan_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Transformer s-physics and delta-s-physics scan\n")
        f.write(f"ckpt = {args.ckpt}\n")
        f.write(f"root_file = {args.root_file}\n")
        f.write(f"N = {len(y_all)}\n")
        f.write(f"D = {int(np.sum(y_all == 0))}\n")
        f.write(f"B = {int(np.sum(y_all == 1))}\n")
        f.write(f"balance_ds = {bool(args.balance_ds)}\n")
        f.write(f"balance_frac = {float(args.balance_frac)}\n")
        f.write(f"max_keep_per_bin_per_class = {int(args.max_keep_per_bin_per_class)}\n")
        f.write(f"profile_bins = {int(args.profile_bins)}\n")
        f.write(f"band = {args.band}\n\n")

        f.write("[Top s-physics correlations]\n")
        for r in s_rows[:20]:
            f.write(
                f"{r[0]:20s} pearson_all={r[1]:+.6f} spearman_all={r[2]:+.6f} "
                f"pearson_D={r[3]:+.6f} pearson_B={r[5]:+.6f}\n"
            )

        f.write("\n[Top delta-s-physics correlations]\n")
        for r in delta_rows[:20]:
            f.write(
                f"{r[0]:20s} pearson_delta={r[1]:+.6f} spearman_delta={r[2]:+.6f} "
                f"slope={r[3]:+.6f} mean_delta={r[4]:+.6f} "
                f"weighted_mean_delta={r[5]:+.6f} delta_range={r[6]:+.6f}\n"
            )

    print(f"[INFO] saved: {summary_txt}")
    print("[DONE]")


if __name__ == "__main__":
    main()