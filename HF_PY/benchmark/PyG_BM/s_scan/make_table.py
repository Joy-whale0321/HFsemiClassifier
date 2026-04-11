#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


def main():
    # ===== 输入文件 =====
    input_path = Path("DeepSetsHF_best_5FALL_3.0-10.0_had3x128_clf3x128_sum_M10_corr.csv")   # 改成你的文件名
    output_dir = input_path.parent

    # ===== 读取数据 =====
    df = pd.read_csv(input_path)

    # ===== 添加重要性排序（|pearson_all|）=====
    df["importance"] = df["pearson_all"].abs()
    df_sorted = df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ===== 名字映射（论文风格，可自行改）=====
    name_map = {
        "e_eta": r"$\eta^e$",
        "e_pt": r"$p_T^e$",
        "e_q": r"$q^e$",
        "lead_abs_deta": r"lead $|\Delta\eta|$",
        "lead_abs_dphi": r"lead $|\Delta\phi|$",
        "lead_had_pt": r"lead hadron $p_T$",
        "mean_abs_deta": r"mean $|\Delta\eta|$",
        "mean_abs_dphi": r"mean $|\Delta\phi|$",
        "mean_had_pt": r"mean hadron $p_T$",
        "n_had": r"$N_{\mathrm{had}}$",
        "pt_conc": r"$p_T$ concentration",
        "same_sign_frac": r"same-sign fraction",
        "std_abs_dphi": r"std($|\Delta\phi|$)",
        "std_deta": r"std($\Delta\eta$)",
        "std_had_pt": r"std(hadron $p_T$)",
        "sum_had_pt": r"sum hadron $p_T$",
    }

    df_sorted["Observable"] = df_sorted["var"].map(lambda x: name_map.get(x, x))

    # ===== 保存排序后的 CSV =====
    df_sorted.to_csv(output_dir / "corr_sorted.csv", index=False)

    # =========================================================
    # ===== 主文 Table（只取前 N 个）=====
    # =========================================================
    TOP_N = 8

    df_top = df_sorted.loc[:TOP_N - 1, ["Observable", "pearson_all", "spearman_all"]].copy()

    # 保留三位小数
    df_top["pearson_all"] = df_top["pearson_all"].map(lambda x: f"{x:.3f}")
    df_top["spearman_all"] = df_top["spearman_all"].map(lambda x: f"{x:.3f}")

    latex_top = df_top.to_latex(
        index=False,
        escape=False,
        column_format="lcc",
        caption="Top observables ranked by the absolute Pearson correlation with the classifier score $s$.",
        label="tab:corr_top"
    )

    # =========================================================
    # ===== Appendix Table（全量）=====
    # =========================================================
    df_full = df_sorted[[
        "Observable",
        "pearson_all", "spearman_all",
        "pearson_D", "spearman_D",
        "pearson_B", "spearman_B"
    ]].copy()

    for col in ["pearson_all", "spearman_all", "pearson_D", "spearman_D", "pearson_B", "spearman_B"]:
        df_full[col] = df_full[col].map(lambda x: f"{x:.3f}")

    latex_full = df_full.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc",
        caption="Correlations between physics observables and the classifier score $s$.",
        label="tab:corr_full"
    )

    # =========================================================
    # ===== Markdown 预览 =====
    # =========================================================
    md_preview = df_top.to_markdown(index=False)

    # ===== 保存文件 =====
    (output_dir / "corr_table_top.tex").write_text(latex_top, encoding="utf-8")
    (output_dir / "corr_table_full.tex").write_text(latex_full, encoding="utf-8")
    (output_dir / "corr_table_top.md").write_text(md_preview, encoding="utf-8")

    # ===== 打印预览 =====
    print("\n=== Top table preview ===\n")
    print(md_preview)

    print("\nSaved files:")
    print(output_dir / "corr_sorted.csv")
    print(output_dir / "corr_table_top.tex")
    print(output_dir / "corr_table_full.tex")
    print(output_dir / "corr_table_top.md")


if __name__ == "__main__":
    main()