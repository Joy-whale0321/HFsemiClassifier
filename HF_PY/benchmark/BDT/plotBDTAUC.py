#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 手动填这里
# =========================
AUC_HANDCRAFTED_ONLY = 0.6854
AUC_TRANSFORMER_SCORE = 0.7810
AUC_TRANSFORMER_SCORE_PLUS_HANDCRAFTED = 0.7815

OUT_PDF = "/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/AUCplot/bdt_auc_input_set_compare.pdf"
OUT_PNG = "/mnt/e/sphenix/HFsemiClassifier/HF_PY/benchmark/BDT/AUCplot/bdt_auc_input_set_compare.png"


def main():
    # 从下到上：
    # BDT handcrafted -> BDT combined -> Transformer performance
    labels = [
        "BDT: handcrafted",
        "BDT: model s +\nhandcrafted",
        "Transformer",
    ]

    aucs = np.array([
        AUC_HANDCRAFTED_ONLY,
        AUC_TRANSFORMER_SCORE_PLUS_HANDCRAFTED,
        AUC_TRANSFORMER_SCORE,
    ], dtype=float)

    y = np.arange(len(labels))

    # 原来 7.8 x 2.8，这里纵向拉长约 20%
    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    # random baseline
    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=1,
    )

    # 从 0.5 baseline 拉线到 AUC 点
    for yi, auc in zip(y, aucs):
        ax.hlines(
            yi,
            0.5,
            auc,
            linewidth=1.6,
            alpha=0.65,
            zorder=2,
        )

    ax.scatter(
        aucs,
        y,
        s=70,
        zorder=3,
    )

    for yi, auc in zip(y, aucs):
        ax.text(
            auc + 0.010,
            yi,
            f"{auc:.4f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("Classification AUC", fontsize=12)

    # AUC = 0.5 is random classification
    ax.set_xlim(0.5, 0.9)
    ax.set_xticks(np.arange(0.5, 0.91, 0.1))

    ax.set_ylim(-0.45, len(labels) - 0.55)

    ax.grid(axis="x", alpha=0.30)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_dir = os.path.dirname(OUT_PDF)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print(f"[plot] saved {OUT_PDF}")
    print(f"[plot] saved {OUT_PNG}")


if __name__ == "__main__":
    main()