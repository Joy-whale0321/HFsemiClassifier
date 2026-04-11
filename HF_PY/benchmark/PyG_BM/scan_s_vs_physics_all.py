#!/usr/bin/env python3
# scan_s_vs_physics.py
#
# Study correlations between
#   s = logit_B - logit_D
# and various physics observables.
#
# Features:
# - Loads ckpt + dataset (same as explain/pca)
# - Optional train-style downsample: per-pt-bin D/B 1:1 balancing
# - Computes observables per sample
# - Computes Pearson/Spearman correlations (overall + per true label)
# - Makes profile plots: <s> vs x (with std band)
#   [UPDATED] plot includes: All + True D + True B
#
# Example:
#   python scan_s_vs_physics.py --ckpt /path/to/model.pt --balance-ds \
#       --ds-pt-bin-width 0.25 --max-events 200000
#
# Outputs:
#   out_prefix_corr.csv
#   out_prefix_obs.npz
#   out_prefix_profile__<var>.png  (one per variable)

import os
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import DeepSetsHF


# ----------------------------- small utils -----------------------------
def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_edges_csv(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    vals = sorted(vals)
    if len(vals) < 2:
        raise ValueError("Need >=2 edges.")
    return np.array(vals, dtype=np.float64)

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x*x).sum() * (y*y).sum())
    if denom <= 0:
        return np.nan
    return float((x*y).sum() / denom)

def rankdata(a: np.ndarray) -> np.ndarray:
    """Average rank for ties, numpy-only."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # handle ties
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
    if x.size < 3:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


# ----------------------------- downsample (same spirit as your explain/pca/train) -----------------------------
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
        return parse_edges_csv(ds_pt_edges)

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

        base_keep = min(len(pools[0]), len(pools[1]))
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


# ----------------------------- observables -----------------------------
def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 1 else 0.0

def compute_observables(batch: Dict[str, torch.Tensor], use_log_pt: bool) -> Dict[str, np.ndarray]:
    """
    batch keys from hf_semi_collate:
      ele_feat: (B,3) [pt(or logpt), eta, charge]
      had_feat: (B,N,5) [pt(or logpt), dEta, sin(dPhi), cos(dPhi), charge]
      had_mask: (B,N) bool
      label:    (B,)
    """
    ele = batch["ele_feat"].cpu().numpy()  # (B,3)
    had = batch["had_feat"].cpu().numpy()  # (B,N,5)
    msk = batch["had_mask"].cpu().numpy().astype(bool)  # (B,N)
    y   = batch["label"].cpu().numpy().astype(np.int64)

    B, N, _ = had.shape

    e_pt_feat = ele[:, 0]
    e_pt = np.exp(e_pt_feat) if use_log_pt else e_pt_feat
    e_eta = ele[:, 1]
    e_q   = ele[:, 2]

    n_had = msk.sum(axis=1).astype(np.float64)

    had_pt_feat = had[:, :, 0]
    had_pt = np.exp(had_pt_feat) if use_log_pt else had_pt_feat
    had_pt = np.where(msk, had_pt, 0.0)

    had_deta = np.where(msk, had[:, :, 1], 0.0)
    had_sin  = np.where(msk, had[:, :, 2], 0.0)
    had_cos  = np.where(msk, had[:, :, 3], 1.0)  # avoid weird atan2 when masked
    had_q    = np.where(msk, had[:, :, 4], 0.0)

    had_dphi = np.arctan2(had_sin, had_cos)
    had_abs_dphi = np.abs(had_dphi)

    sum_had_pt = had_pt.sum(axis=1)
    mean_had_pt = np.divide(sum_had_pt, np.maximum(n_had, 1.0))
    std_had_pt = np.array([safe_std(had_pt[i, msk[i]]) for i in range(B)], dtype=np.float64)

    mean_abs_dphi = np.array([safe_mean(had_abs_dphi[i, msk[i]]) for i in range(B)], dtype=np.float64)
    std_abs_dphi  = np.array([safe_std(had_abs_dphi[i, msk[i]])  for i in range(B)], dtype=np.float64)
    mean_abs_deta = np.array([safe_mean(np.abs(had_deta[i, msk[i]])) for i in range(B)], dtype=np.float64)
    std_deta      = np.array([safe_std(had_deta[i, msk[i]]) for i in range(B)], dtype=np.float64)

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
        lead_abs_deta[i] = float(np.abs(had_deta[i, pos]))

        q_i = had_q[i, vi]
        same_sign_frac[i] = float(np.mean((q_i * e_q[i]) > 0.0))

    pt_conc = np.divide(lead_pt, np.maximum(sum_had_pt, 1e-12))

    obs = {
        "label": y.astype(np.float64),
        "e_pt": e_pt.astype(np.float64),
        "e_eta": e_eta.astype(np.float64),
        "e_q": e_q.astype(np.float64),
        "n_had": n_had,
        "sum_had_pt": sum_had_pt.astype(np.float64),
        "mean_had_pt": mean_had_pt.astype(np.float64),
        "std_had_pt": std_had_pt,
        "lead_had_pt": lead_pt,
        "pt_conc": pt_conc.astype(np.float64),
        "mean_abs_dphi": mean_abs_dphi,
        "std_abs_dphi": std_abs_dphi,
        "lead_abs_dphi": lead_abs_dphi,
        "mean_abs_deta": mean_abs_deta,
        "std_deta": std_deta,
        "lead_abs_deta": lead_abs_deta,
        "same_sign_frac": same_sign_frac,
    }
    return obs


# ----------------------------- plots -----------------------------
def profile_plot(x: np.ndarray, s: np.ndarray, y: np.ndarray, var_name: str, out_png: str,
                 n_bins: int = 30, x_quantile_clip: float = 0.005):
    """
    Make profile: <s> vs x, with std band.
    UPDATED: plot All + True D + True B.
    """
    ensure_dir_for_file(out_png)

    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # robust x-range
    lo = np.quantile(x, x_quantile_clip)
    hi = np.quantile(x, 1.0 - x_quantile_clip)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])


    def prof(mask: np.ndarray):
        xm = x[mask]
        sm = s[mask]
        bin_id = np.searchsorted(edges, xm, side="right") - 1
        mu = np.full(n_bins, np.nan)
        sd = np.full(n_bins, np.nan)
        cnt = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            sel = (bin_id == b)
            if np.sum(sel) >= 20:
                mu[b] = float(np.mean(sm[sel]))
                sd[b] = float(np.std(sm[sel]))
                cnt[b] = int(np.sum(sel))
        return mu, sd, cnt

    isAll = np.ones_like(y, dtype=bool)
    isD = (y == 0)
    isB = (y == 1)

    muAll, sdAll, cAll = prof(isAll)
    muD,   sdD,   cD   = prof(isD)
    muB,   sdB,   cB   = prof(isB)

    plt.figure(figsize=(6.2, 4.8))

    # All (overall) — black line + light band
    if np.any(np.isfinite(muAll)):
        plt.plot(centers, muAll, label="All: mean(s)", color="black", linewidth=2.0)
        plt.fill_between(centers, muAll - sdAll, muAll + sdAll, alpha=0.12, color="black")

    # True D / True B — keep your original style
    if np.any(np.isfinite(muD)):
        plt.plot(centers, muD, label="True D: mean(s)")
        plt.fill_between(centers, muD - sdD, muD + sdD, alpha=0.2)
    if np.any(np.isfinite(muB)):
        plt.plot(centers, muB, label="True B: mean(s)")
        plt.fill_between(centers, muB - sdB, muB + sdB, alpha=0.2)

    plt.xlabel(var_name)
    plt.ylabel("s = logit_B - logit_D")
    plt.title(f"Profile: s vs {var_name}")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ----------------------------- main -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Scan correlations: s vs physics observables.")
    p.add_argument("--ckpt",
                   type=str,
                   default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/Weight_of_Model/deepset/DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10.pt",
                   help="Path to model checkpoint (.pt) saved by train script.")
    p.add_argument(
        "--root-file",
        type=str,
        default="/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_2_allAccept.root",
        help="Override dataset ROOT file (default: dataset 2)."
    )
    p.add_argument("--tree-name", type=str, default="tree")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="cuda/cpu/empty=auto")
    p.add_argument("--max-events", type=int, default=-1, help="<=0 means all (after D/B filtering, before balancing)")
    p.add_argument("--seed", type=int, default=12345)

    # dataset cuts override
    p.add_argument("--pt-min", type=float, default=None)
    p.add_argument("--pt-max", type=float, default=None)
    p.add_argument("--eta-abs-max", type=float, default=5.0)
    p.add_argument("--had-pt-min", type=float, default=0.2)
    p.add_argument("--had-pt-max", type=float, default=None)
    p.add_argument("--min-had", type=int, default=4)
    p.add_argument("--use-log-pt", action="store_true", help="Force use_log_pt=True (else follow ckpt if present).")

    # downsample
    # keep your default True behavior, but make it less error-prone to pass from CLI:
    p.add_argument("--balance-ds", action="store_true", default=True,
                   help="Enable train-style per-pt-bin D/B 1:1 balancing (default: ON).")
    p.add_argument("--no-balance-ds", action="store_false", dest="balance_ds",
                   help="Disable balancing (use all D/B candidates).")
    p.add_argument("--balance-frac", type=float, default=1.0,
                   help="n_keep = floor(frac * min(nD,nB)) in each ds bin.")
    p.add_argument("--ds-pt-bin-width", type=float, default=0.25)
    p.add_argument("--ds-pt-edges", type=str, default="")

    # outputs
    p.add_argument("--out-prefix", type=str, default="",
                   help="Prefix for outputs. If empty, derive from ckpt name.")

    # plotting
    p.add_argument("--profile-bins", type=int, default=30)

    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    # device
    if args.device.strip() == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}")

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(ckpt_args)

    pooling = ckpt_args.get("pooling", "sum")

    root_file = args.root_file.strip() or ckpt_args.get("root_file", "")
    if root_file == "":
        raise ValueError("root-file not provided and not found in ckpt['args']['root_file'].")

    pt_min = args.pt_min if args.pt_min is not None else ckpt_args.get("pt_min", None)
    pt_max = args.pt_max if args.pt_max is not None else ckpt_args.get("pt_max", None)
    use_log_pt = bool(args.use_log_pt) or bool(ckpt_args.get("use_log_pt", False))

    print("[INFO] dataset:")
    print("   root_file =", root_file)
    print("   pt_min/pt_max =", pt_min, pt_max)
    print("   use_log_pt =", use_log_pt)

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

    # choose indices (D/B only)
    max_events = None if args.max_events <= 0 else int(args.max_events)
    idx_all = pick_indices_labels01(dataset, max_events=max_events, seed=args.seed)
    print(f"[INFO] D/B-only candidates: {len(idx_all)}")

    if args.balance_ds:
        ds_edges = parse_ds_pt_edges(args.ds_pt_edges, args.ds_pt_bin_width, pt_min, pt_max)
        idx_map = build_ptbin_class_index_from_indices(dataset, idx_all, ds_edges, num_classes=2)
        idx_bal = resample_balanced_by_ptbin_indices(
            idx_map, ds_edges, seed=args.seed, frac=float(args.balance_frac), num_classes=2
        )
        print(f"[INFO] balance-ds: {len(idx_all)} -> {len(idx_bal)}")
        if len(idx_bal) > 0:
            idx = idx_bal
        else:
            print("[WARN] balance-ds produced empty set; fallback to unbalanced idx_all.")
            idx = idx_all
    else:
        idx = idx_all

    work_set = Subset(dataset, idx)
    print(f"[INFO] work_set size: {len(work_set)}")

    loader = DataLoader(
        work_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_semi_collate,
        pin_memory=True if device.type == "cuda" else False,
    )

    # model (DeepSetsHF; keep consistent with your explain/pca)
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

    # output prefix
    if args.out_prefix.strip():
        prefix = args.out_prefix.strip()
    else:
        base = os.path.splitext(os.path.basename(args.ckpt))[0]
        prefix = os.path.join("/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan/", "deepset_scan" + base)
        # prefix = os.path.join("/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/s_scan/", base)
    ensure_dir_for_file(prefix + "_dummy.txt")
    print("[INFO] out prefix =", prefix)

    # loop
    all_s = []
    all_label = []
    obs_acc: Dict[str, List[np.ndarray]] = {}

    for batch in loader:
        ele = batch["ele_feat"].to(device)
        had = batch["had_feat"].to(device)
        msk = batch["had_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(ele, had, msk, return_attn=False)  # (B,2)
        s = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float64)

        obs = compute_observables(batch, use_log_pt=use_log_pt)

        all_s.append(s)
        all_label.append(obs["label"].astype(np.float64))

        for k, v in obs.items():
            if k == "label":
                continue
            obs_acc.setdefault(k, []).append(v.astype(np.float64))

    s_all = np.concatenate(all_s, axis=0)
    y_all = np.concatenate(all_label, axis=0).astype(np.int64)

    obs_all: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in obs_acc.items()}

    print(f"[INFO] total N = {s_all.size}")
    print(f"[INFO] label counts: D={(y_all==0).sum()}, B={(y_all==1).sum()}")

    # correlations
    var_names = sorted(list(obs_all.keys()))
    rows = []
    for vn in var_names:
        x = obs_all[vn]
        r_p = pearsonr(x, s_all)
        r_s = spearmanr(x, s_all)

        mD = (y_all == 0)
        mB = (y_all == 1)
        r_p_D = pearsonr(x[mD], s_all[mD]) if mD.sum() >= 3 else np.nan
        r_s_D = spearmanr(x[mD], s_all[mD]) if mD.sum() >= 3 else np.nan
        r_p_B = pearsonr(x[mB], s_all[mB]) if mB.sum() >= 3 else np.nan
        r_s_B = spearmanr(x[mB], s_all[mB]) if mB.sum() >= 3 else np.nan

        rows.append((vn, r_p, r_s, r_p_D, r_s_D, r_p_B, r_s_B))

    csv_path = prefix + "_corr.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("var,pearson_all,spearman_all,pearson_D,spearman_D,pearson_B,spearman_B\n")
        for r in rows:
            f.write(",".join([r[0]] + [f"{x:.6g}" if np.isfinite(x) else "nan" for x in r[1:]]) + "\n")
    print("[INFO] saved:", csv_path)

    npz_path = prefix + "_obs.npz"
    np.savez_compressed(
        npz_path,
        s=s_all,
        label=y_all,
        **obs_all,
        ckpt=args.ckpt,
        root_file=root_file,
        pooling=pooling,
        use_log_pt=use_log_pt,
        pt_min=pt_min if pt_min is not None else -1.0,
        pt_max=pt_max if pt_max is not None else -1.0,
        balance_ds=bool(args.balance_ds),
        balance_frac=float(args.balance_frac),
        ds_pt_bin_width=float(args.ds_pt_bin_width),
        ds_pt_edges=str(args.ds_pt_edges),
    )
    print("[INFO] saved:", npz_path)

    for vn in var_names:
        out_png = prefix + f"_profile__{vn}.png"
        profile_plot(obs_all[vn], s_all, y_all, vn, out_png, n_bins=int(args.profile_bins))
        print("[INFO] saved:", out_png)

    rows_sorted = sorted(rows, key=lambda t: (-(abs(t[1]) if np.isfinite(t[1]) else -1.0)))
    print("\n[TOP correlations by |pearson_all|]")
    for vn, rpa, rsa, rpD, rsD, rpB, rsB in rows_sorted[:10]:
        print(f"  {vn:16s}  pearson_all={rpa:+.4f}  spearman_all={rsa:+.4f}  (D {rpD:+.4f}, B {rpB:+.4f})")

if __name__ == "__main__":
    main()
