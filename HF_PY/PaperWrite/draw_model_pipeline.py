import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, text, fontsize=10, lw=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        fill=False
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )
    return {"x": x, "y": y, "w": w, "h": h}


def pt_right(box):
    return (box["x"] + box["w"], box["y"] + box["h"] / 2)


def pt_left(box):
    return (box["x"], box["y"] + box["h"] / 2)


def pt_top(box):
    return (box["x"] + box["w"] / 2, box["y"] + box["h"])


def pt_bottom(box):
    return (box["x"] + box["w"] / 2, box["y"])


def add_arrow(ax, p1, p2, lw=1.2, ms=10):
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=lw,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def main():
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    fig, ax = plt.subplots(figsize=(13.0, 3.2))

    # =========================================================
    # Main row: ALL top edges aligned
    # =========================================================
    top_y = 1.25
    h_small = 0.78
    h_big = 1.52

    # x positions / widths
    had_x, had_w = 0.25, 2.10
    set_x, set_w = 2.90, 3.95
    H_x, H_w = 7.30, 1.75
    clf_x, clf_w = 9.45, 2.20
    out_x, out_w = 12.10, 1.35

    # all top borders aligned
    had_y = top_y
    set_y = top_y - (h_big - h_small)
    H_y = top_y
    clf_y = top_y
    out_y = top_y

    # electron box below classifier
    ele_w, ele_h = 1.75, 0.72
    ele_x = clf_x + (clf_w - ele_w) / 2
    ele_y = 0.18

    # =========================================================
    # Boxes
    # =========================================================
    had = add_box(
        ax, had_x, had_y, had_w, h_small,
        "Hadron set\n"
        "$(N\\times5)$\n"
        "$p_T,\\ \\Delta\\eta,\\ \\sin\\Delta\\phi,\\ \\cos\\Delta\\phi,\\ q$",
        fontsize=10
    )

    set_block = add_box(
        ax, set_x, set_y, set_w, h_big,
        "Set modeling\n\n"
        "DeepSets:\n"
        "shared encoder + pooling\n\n"
        "Transformer:\n"
        "tokens + CLS\n\n"
        "GNN:\n"
        "kNN + EdgeConv + pooling",
        fontsize=9.5
    )

    H = add_box(
        ax, H_x, H_y, H_w, h_small,
        "$H$\n"
        "(set-level embedding)",
        fontsize=10.3
    )

    clf = add_box(
        ax, clf_x, clf_y, clf_w, h_small,
        "MLP classifier\n"
        "$(H \\oplus e)$",
        fontsize=10.5
    )

    out = add_box(
        ax, out_x, out_y, out_w, h_small,
        "Output\n"
        "$D/B$",
        fontsize=10.5
    )

    ele = add_box(
        ax, ele_x, ele_y, ele_w, ele_h,
        "Electron\n"
        "$(3)$\n"
        "$p_T,\\ \\eta,\\ q$",
        fontsize=10
    )

    # =========================================================
    # Arrows: keep all top-row arrows strictly horizontal
    # =========================================================
    y_main = had_y + h_small / 2.0

    # Hadron -> Set modeling
    add_arrow(
        ax,
        (had_x + had_w, y_main),
        (set_x, y_main)
    )

    # Set modeling -> H
    add_arrow(
        ax,
        (set_x + set_w, y_main),
        (H_x, y_main)
    )

    # H -> Classifier
    add_arrow(
        ax,
        (H_x + H_w, y_main),
        (clf_x, y_main)
    )

    # Classifier -> Output
    add_arrow(
        ax,
        (clf_x + clf_w, y_main),
        (out_x, y_main)
    )

    # Electron -> Classifier
    add_arrow(
        ax,
        pt_top(ele),
        pt_bottom(clf)
    )

    # =========================================================
    # Canvas
    # =========================================================
    ax.set_xlim(0.0, 13.7)
    ax.set_ylim(0.0, 2.35)
    ax.axis("off")

    plt.tight_layout(pad=0.2)
    plt.savefig("model_multimodel_aligned.pdf", bbox_inches="tight")
    plt.savefig("model_multimodel_aligned.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()