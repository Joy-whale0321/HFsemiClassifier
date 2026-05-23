#!/usr/bin/env python3
# plot_s_deltaR_profiles.py
#
# Plot:
#   <s> vs mean(ΔR)
#   <s> vs std(ΔR)
#
# using the same Transformer model / dataset setup
# as the existing s-physics scan workflow.


import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from data_HFSemiClassifier import HFSemiClassifier, hf_semi_collate
from model_HFSemiClassifier import SetTransformerHF


# =========================================================
# config
# =========================================================

CKPT = (
    "/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/PyG_BM/"
    "Weight_of_Model/transformer/"
    "TransformerHF_best_ALL_3.0-10.0_layer4_M4.pt"
)

ROOT_FILE = (
    "/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/"
    "ppHF_eXDecay_5B_2_allAccept.root"
)

OUT_DIR = (
    "/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/"
    "PyG_BM/s_scan/TF4plot/transformer_deltaR_profiles"
)

PT_MIN = 3.0
PT_MAX = 10.0

MAX_KEEP_PER_BIN_PER_CLASS = 500

BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# utils
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


# =========================================================
# balancing
# =========================================================

def get_electron_pt(dataset, global_idx):
    evt_idx, ele_idx = dataset.electron_index[global_idx]
    return float(dataset.ele_pt[evt_idx][ele_idx])


def build_balanced_indices(dataset, pt_edges, seed=12345):

    idx_map = {}

    for b in range(len(pt_edges) - 1):
        for c in [0, 1]:
            idx_map[(b, c)] = []

    for i in range(len(dataset)):

        evt_idx, ele_idx = dataset.electron_index[i]
        tag = int(dataset.ele_hf_TAG[evt_idx][ele_idx])

        if tag == 1:
            y = 0
        elif tag == 3:
            y = 1
        else:
            continue

        pt = get_electron_pt(dataset, i)

        b = np.searchsorted(pt_edges, pt, side="right") - 1

        if 0 <= b < len(pt_edges) - 1:
            idx_map[(b, y)].append(i)

    rng = np.random.default_rng(seed)

    selected = []

    for b in range(len(pt_edges) - 1):

        d_pool = idx_map[(b, 0)]
        b_pool = idx_map[(b, 1)]

        if len(d_pool) == 0 or len(b_pool) == 0:
            continue

        n_keep = min(
            len(d_pool),
            len(b_pool),
            MAX_KEEP_PER_BIN_PER_CLASS,
        )

        selected.extend(
            rng.choice(d_pool, size=n_keep, replace=False).tolist()
        )

        selected.extend(
            rng.choice(b_pool, size=n_keep, replace=False).tolist()
        )

    rng.shuffle(selected)

    return selected


# =========================================================
# model
# =========================================================

def build_model(ckpt_args):

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


# =========================================================
# observables
# =========================================================

@torch.no_grad()
def evaluate(model, loader, use_log_pt):

    all_s = []

    all_mean_dr = []
    all_std_dr = []

    model.eval()

    for batch in loader:

        ele = batch["ele_feat"].to(DEVICE)
        had = batch["had_feat"].to(DEVICE)
        msk = batch["had_mask"].to(DEVICE)

        logits = model(ele, had, msk)

        s = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()

        all_s.append(s)

        had_np = batch["had_feat"].cpu().numpy()
        mask_np = batch["had_mask"].cpu().numpy().astype(bool)

        had_deta = had_np[:, :, 1]

        had_sin = had_np[:, :, 2]
        had_cos = had_np[:, :, 3]

        had_dphi = np.arctan2(had_sin, had_cos)

        had_dr = np.sqrt(had_deta ** 2 + had_dphi ** 2)

        B = had_dr.shape[0]

        mean_dr = np.zeros(B)
        std_dr = np.zeros(B)

        for i in range(B):

            vals = had_dr[i][mask_np[i]]

            mean_dr[i] = safe_mean(vals)
            std_dr[i] = safe_std(vals)

        all_mean_dr.append(mean_dr)
        all_std_dr.append(std_dr)

    return (
        np.concatenate(all_s),
        np.concatenate(all_mean_dr),
        np.concatenate(all_std_dr),
    )


# =========================================================
# profile
# =========================================================

def make_profile(x, y, n_bins=30):

    lo = np.quantile(x, 0.005)
    hi = np.quantile(x, 0.995)

    edges = np.linspace(lo, hi, n_bins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_y = np.full(n_bins, np.nan)
    err_y = np.full(n_bins, np.nan)

    bin_id = np.searchsorted(edges, x, side="right") - 1

    for b in range(n_bins):

        sel = bin_id == b

        if np.sum(sel) < 30:
            continue

        vals = y[sel]

        mean_y[b] = np.mean(vals)
        err_y[b] = np.std(vals) / np.sqrt(np.sum(sel))

    return centers, mean_y, err_y


def smooth_curve(x, y, yerr):

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)

    x = x[good]
    y = y[good]
    yerr = yerr[good]

    order = np.argsort(x)

    x = x[order]
    y = y[order]
    yerr = yerr[order]

    x_grid = np.linspace(np.min(x), np.max(x), 300)

    bw = 0.05 * (np.max(x) - np.min(x))

    y_smooth = np.zeros_like(x_grid)

    for i, x0 in enumerate(x_grid):

        w = np.exp(-0.5 * ((x - x0) / bw) ** 2)
        w *= 1.0 / np.maximum(yerr, 1e-8) ** 2

        y_smooth[i] = np.sum(w * y) / np.sum(w)

    return x_grid, y_smooth


def plot_profile(x, s, xlabel, out_pdf):

    centers, mean_s, err_s = make_profile(x, s)

    good = (
        np.isfinite(centers)
        & np.isfinite(mean_s)
        & np.isfinite(err_s)
    )

    plt.figure(figsize=(6.2, 4.8))

    xg, yg = smooth_curve(
        centers[good],
        mean_s[good],
        err_s[good],
    )

    plt.plot(
        xg,
        yg,
        linewidth=2.0,
        alpha=0.8,
    )

    plt.errorbar(
        centers[good],
        mean_s[good],
        yerr=err_s[good],
        fmt="o",
        markersize=4,
        linewidth=1.4,
        capsize=2.5,
    )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    # plt.xlim(0.0, 2.8)
    plt.ylim(-2.2, 1.0)

    plt.xlabel(xlabel)
    plt.ylabel(r"$s = \mathrm{logit}_B - \mathrm{logit}_D$")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(out_pdf, dpi=180)

    plt.close()


# =========================================================
# main
# =========================================================

@torch.no_grad()
def main():

    ensure_dir(OUT_DIR)

    ckpt = torch.load(CKPT, map_location=DEVICE)

    ckpt_args = ckpt.get("args", {})

    use_log_pt = bool(ckpt_args.get("use_log_pt", False))

    dataset = HFSemiClassifier(
        ROOT_FILE,
        tree_name="tree",
        use_log_pt=use_log_pt,
        pt_min=PT_MIN,
        pt_max=PT_MAX,
        eta_abs_max=5.0,
        use_had_eta=True,
        had_pt_min=0.2,
        had_pt_max=None,
        min_had=4,
    )

    pt_edges = np.arange(PT_MIN, PT_MAX + 0.25, 0.25)

    selected = build_balanced_indices(dataset, pt_edges)

    subset = Subset(dataset, selected)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=hf_semi_collate,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    model = build_model(ckpt_args).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    s, mean_dr, std_dr = evaluate(
        model,
        loader,
        use_log_pt,
    )

    plot_profile(
        mean_dr,
        s,
        xlabel=r"mean($\Delta R$)",
        out_pdf=os.path.join(
            OUT_DIR,
            "transformer_s_vs_mean_deltaR.pdf",
        ),
    )

    plot_profile(
        std_dr,
        s,
        xlabel=r"std($\Delta R$)",
        out_pdf=os.path.join(
            OUT_DIR,
            "transformer_s_vs_std_deltaR.pdf",
        ),
    )

    print("[DONE]")


if __name__ == "__main__":
    main()