import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, text, fontsize=11.5, lw=1.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.05",
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


def pt_top(box):
    return (box["x"] + box["w"] / 2, box["y"] + box["h"])


def pt_bottom(box):
    return (box["x"] + box["w"] / 2, box["y"])


def add_arrow(ax, p1, p2, lw=1.8, ms=12):
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
        "font.size": 11.5,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })

    fig, ax = plt.subplots(figsize=(13.5, 3.8))

    # =========================================================
    # Main row: ALL top edges aligned
    # =========================================================
    top_y = 1.45
    h_small = 0.82
    h_big = 1.62

    had_x, had_w = 0.25, 2.35
    set_x, set_w = 3.10, 4.25
    H_x, H_w = 7.80, 1.95
    clf_x, clf_w = 10.15, 2.35
    out_x, out_w = 12.95, 1.45

    had_y = top_y
    set_y = top_y - (h_big - h_small)
    H_y = top_y
    clf_y = top_y
    out_y = top_y

    ele_w, ele_h = 1.95, 0.76
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
        fontsize=11.3
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
        fontsize=10.8
    )

    H = add_box(
        ax, H_x, H_y, H_w, h_small,
        "$H$\n"
        "(set-level embedding)",
        fontsize=11.5
    )

    clf = add_box(
        ax, clf_x, clf_y, clf_w, h_small,
        "MLP classifier\n"
        "$(H \\oplus e)$",
        fontsize=11.7
    )

    out = add_box(
        ax, out_x, out_y, out_w, h_small,
        "Output\n"
        "$D/B$",
        fontsize=11.7
    )

    ele = add_box(
        ax, ele_x, ele_y, ele_w, ele_h,
        "Electron\n"
        "$(3)$\n"
        "$p_T,\\ \\eta,\\ q$",
        fontsize=11.3
    )

    # =========================================================
    # Arrows: keep all top-row arrows strictly horizontal
    # =========================================================
    y_main = had_y + h_small / 2.0

    add_arrow(ax, (had_x + had_w, y_main), (set_x, y_main))
    add_arrow(ax, (set_x + set_w, y_main), (H_x, y_main))
    add_arrow(ax, (H_x + H_w, y_main), (clf_x, y_main))
    add_arrow(ax, (clf_x + clf_w, y_main), (out_x, y_main))

    add_arrow(ax, pt_top(ele), pt_bottom(clf))

    # =========================================================
    # Canvas
    # =========================================================
    ax.set_xlim(0.0, 14.65)
    ax.set_ylim(0.0, 2.65)
    ax.axis("off")

    plt.tight_layout(pad=0.25)

    plt.savefig(
        "model_multimodel_large.pdf",
        bbox_inches="tight",
        pad_inches=0.05
    )
    plt.savefig(
        "model_multimodel_large.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.05
    )
    plt.close(fig)


if __name__ == "__main__":
    main()