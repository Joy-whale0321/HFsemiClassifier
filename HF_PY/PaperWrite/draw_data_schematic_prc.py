import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_round_box(ax, x, y, w, h, text, fontsize=12, lw=1.4):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.2,
    )
    return {"x": x, "y": y, "w": w, "h": h}


def pt_right(box):
    return box["x"] + box["w"], box["y"] + box["h"] / 2


def pt_left(box):
    return box["x"], box["y"] + box["h"] / 2


def add_arrow(ax, p1, p2, lw=1.3, ms=11, style="-"):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color="black",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 4.8))

    # Main horizontal pipeline
    gen = add_round_box(
        ax,
        0.35,
        1.65,
        2.65,
        1.55,
        "Generator event\n"
        "pp @ $\\sqrt{s}$\n"
        "with charm/beauty",
        fontsize=13,
    )

    decay = add_round_box(
        ax,
        3.45,
        1.65,
        2.95,
        1.55,
        "HF semi-leptonic\n"
        "decay tagging\n"
        "$D/B \\rightarrow e + \\nu_e + X$",
        fontsize=13,
    )

    selection = add_round_box(
        ax,
        6.9,
        1.65,
        3.05,
        1.55,
        "Object selection\n"
        "trigger $e$: $p_T^e > 3\\,\\mathrm{GeV}/c$\n"
        "assoc: charged non-leptons",
        fontsize=12.2,
    )

    features = add_round_box(
        ax,
        10.4,
        1.65,
        2.95,
        1.55,
        "Per-hadron features\n"
        "$h_i=(q_i,p_{T,i},\\Delta\\eta_i,\\Delta\\phi_i)$\n"
        "w.r.t. trigger $e$",
        fontsize=12.2,
    )

    add_arrow(ax, pt_right(gen), pt_left(decay))
    add_arrow(ax, pt_right(decay), pt_left(selection))
    add_arrow(ax, pt_right(selection), pt_left(features))

    # Final tensor block and output note
    add_round_box(
        ax,
        5.15,
        0.2,
        3.5,
        1.0,
        "Model input per event: $\\{h_i\\}_{i=1}^{N_h}$\n"
        "(variable-size hadron set)",
        fontsize=12.5,
    )
    add_arrow(ax, (11.85, 1.65), (8.6, 1.2), lw=1.2)

    # Small annotation panel for publication clarity
    add_round_box(
        ax,
        0.35,
        0.2,
        4.3,
        1.0,
        "Schematic for generator-level data acquisition\n"
        "used in HF $e$-hadron correlation ML dataset",
        fontsize=11.5,
    )
    add_arrow(ax, (3.0, 1.65), (2.7, 1.2), lw=1.1, style=(0, (3, 3)))

    # Axis and save
    ax.set_xlim(0.0, 13.7)
    ax.set_ylim(0.0, 3.55)
    ax.axis("off")

    plt.tight_layout(pad=0.3)
    plt.savefig("hf_data_schematic_prc.pdf", bbox_inches="tight")
    plt.savefig("hf_data_schematic_prc.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
