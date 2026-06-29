import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D


def add_box(ax, x, y, w, h, text="", fc="#ffffff", ec="black",
            fontsize=12, lw=1.4, rounding=0.08):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)
    return {"x": x, "y": y, "w": w, "h": h}


def add_arrow(ax, p1, p2, lw=2.0, ms=18):
    ax.add_patch(FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color="black",
        shrinkA=3,
        shrinkB=3
    ))


def add_poly_arrow(ax, points, lw=2.0, ms=18):
    for p1, p2 in zip(points[:-2], points[1:-1]):
        ax.add_line(Line2D(
            [p1[0], p2[0]], [p1[1], p2[1]],
            linewidth=lw, color="black"
        ))

    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1],
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color="black",
        shrinkA=0,
        shrinkB=3
    ))


def add_feature_stack(ax, x, y, w=0.68):
    cell_h = 0.29
    gap = 0.025
    dots_h = 0.38

    top_y = y + 5 * cell_h + 5 * gap + dots_h

    add_box(ax, x, top_y - cell_h, w, cell_h,
            fc="#dff1ff", ec="black", lw=1.0, rounding=0.025)

    add_box(ax, x, top_y - 2 * cell_h - gap, w, cell_h,
            fc="#dff1ff", ec="black", lw=1.0, rounding=0.025)

    dots_y = top_y - 2 * cell_h - 2 * gap - dots_h
    add_box(ax, x, dots_y, w, dots_h,
            fc="#dff1ff", ec="black", lw=1.0, rounding=0.025)

    ax.text(x + w / 2, dots_y + dots_h / 2, r"$\vdots$",
            ha="center", va="center", fontsize=15)

    blue3_y = dots_y - gap - cell_h
    add_box(ax, x, blue3_y, w, cell_h,
            fc="#dff1ff", ec="black", lw=1.0, rounding=0.025)

    red_start_y = blue3_y - gap
    for i in range(3):
        yy = red_start_y - (i + 1) * cell_h - i * gap
        add_box(ax, x, yy, w, cell_h,
                fc="#ffc6bf", ec="black", lw=1.0, rounding=0.025)


def add_mlp(ax, x, y, w, h):
    s = 1.2
    dx = 0.10
    dy = 0.12

    cx = x + 0.95 + dx
    cy = y + 0.85 + dy

    xs0 = [x + 0.30 + dx, x + 0.95 + dx, x + 1.55 + dx]

    ys1_0 = [y + 1.45 + dy, y + 1.05 + dy, y + 0.65 + dy, y + 0.25 + dy]
    ys2_0 = [y + 1.45 + dy, y + 1.05 + dy, y + 0.25 + dy]
    ys3_0 = [y + 1.10 + dy, y + 0.60 + dy]

    def scale_point(px, py):
        return cx + s * (px - cx), cy + s * (py - cy)

    xs = [scale_point(xx, cy)[0] for xx in xs0]
    ys1 = [scale_point(cx, yy)[1] for yy in ys1_0]
    ys2 = [scale_point(cx, yy)[1] for yy in ys2_0]
    ys3 = [scale_point(cx, yy)[1] for yy in ys3_0]

    layers = [ys1, ys2, ys3]

    for a in range(len(layers) - 1):
        for yy1 in layers[a]:
            for yy2 in layers[a + 1]:
                ax.add_line(Line2D(
                    [xs[a], xs[a + 1]], [yy1, yy2],
                    lw=0.8, color="black", alpha=0.65
                ))

    for yy in ys1:
        ax.add_patch(Circle((xs[0], yy), 0.12,
                            facecolor="#edf7df", edgecolor="black", lw=1.0))

    for yy in ys2:
        ax.add_patch(Circle((xs[1], yy), 0.12,
                            facecolor="#edf7df", edgecolor="black", lw=1.0))

    ax.text(xs[1], scale_point(cx, y + 0.65 + dy)[1], r"$\vdots$",
            ha="center", va="center", fontsize=16)

    for yy in ys3:
        ax.add_patch(Circle((xs[2], yy), 0.12,
                            facecolor="#edf7df", edgecolor="black", lw=1.0))

    ax.text(x + w / 2, y - 0.10, "Fully connected network",
            ha="center", va="top", fontsize=12)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavuserif",
    })

    fig, ax = plt.subplots(figsize=(16.35, 7.0))

    bg = FancyBboxPatch(
        (0.05, 0.45), 16.25, 6.5,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2,
        edgecolor="#8a6a2f",
        facecolor="#f7eadb"
    )
    ax.add_patch(bg)

    had = add_box(
        ax, 0.45, 4.05, 2.35, 1.85,
        "Hadron set\n"
        r"$(N\times 5)$" "\n"
        r"$p_T,\ q,\ \Delta\eta,$" "\n"
        r"$\sin\Delta\phi,\ \cos\Delta\phi$",
        fc="#dff1ff", ec="black", fontsize=18
    )

    enc = add_box(
        ax, 3.40, 3.0, 4.15, 3.55,
        "Hadron-set encoder\n\n"
        "DeepSets:\n"
        "shared MLP + pooling\n\n"
        "Transformer:\n"
        "self-attention + CLS\n\n"
        "GNN:\n"
        "kNN graph + EdgeConv + pooling",
        fc="#dcd5ff", ec="#6f50c8", fontsize=17
    )

    ele = add_box(
        ax, 4.45, 0.90, 2.15, 1.45,
        "Electron\n"
        r"$(3)$" "\n"
        r"$p_T,\ q,\ \eta$",
        fc="#ffc4bf", ec="black", fontsize=18
    )

    plus_center = (8.62, 3.90)
    ax.add_patch(Circle(plus_center, 0.28,
                        facecolor="white", edgecolor="black", lw=2.0))
    ax.text(*plus_center, "+", ha="center", va="center", fontsize=32)

    ax.text(10.18, 5.55, r"$H \oplus e$",
            ha="center", va="center", fontsize=20)
    ax.text(10.18, 5.30, "(combined features)",
            ha="center", va="center", fontsize=13)

    add_feature_stack(ax, 9.92, 3.0)

    clf = add_box(ax, 11.40, 2.65, 2.35, 2.55,
                  fc="#eaf6df", ec="#66a64a", lw=1.4)
    ax.text(12.58, 5.35, "MLP classifier",
            ha="center", va="bottom", fontsize=15, weight="bold")
    add_mlp(ax, 11.60, 3.10, 2.0, 1.65)

    out = add_box(
        ax, 14.55, 3.25, 1.60, 1.45,
        "",
        fc="#fff1c8",
        ec="black"
    )

    ax.text(15.35, 4.30, "Output",
            ha="center", va="center", fontsize=18)

    ax.text(15.35, 3.95, r"$D/B$ origin",
            ha="center", va="center", fontsize=16)

    ax.text(15.35, 3.55,
            r"$(s=\mathrm{logit}\ B-\mathrm{logit}\ D)$",
            ha="center", va="center", fontsize=10)

    # arrows
    add_arrow(ax, (2.83, 4.98), (3.40, 4.98))

    add_poly_arrow(
        ax,
        [
            (7.62, 5.00),
            (8.62, 5.00),
            (8.62, 4.20),
        ]
    )

    add_poly_arrow(
        ax,
        [
            (6.67, 1.62),
            (8.62, 1.62),
            (8.62, 3.60),
        ]
    )

    add_arrow(ax, (8.90, 3.90), (9.92, 3.90))
    add_arrow(ax, (10.62, 3.90), (11.40, 3.90))
    add_arrow(ax, (13.77, 3.90), (14.55, 3.90))

    ax.text(8.70, 4.4, "Concatenate",
            ha="left", va="center", fontsize=12)

    ax.set_xlim(0, 16.35)
    ax.set_ylim(0.40, 7.0)
    ax.axis("off")

    plt.tight_layout(pad=0.1)
    plt.savefig("model_architecture.pdf", bbox_inches="tight")
    plt.savefig("model_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()