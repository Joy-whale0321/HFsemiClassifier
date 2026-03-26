
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, text, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2,
        fill=False
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return {"x": x, "y": y, "w": w, "h": h}


def pt_right(box):
    return (box["x"] + box["w"], box["y"] + box["h"] / 2)


def pt_left(box):
    return (box["x"], box["y"] + box["h"] / 2)


def pt_top(box):
    return (box["x"] + box["w"] / 2, box["y"] + box["h"])


def pt_bottom(box):
    return (box["x"] + box["w"] / 2, box["y"])


def pt_northeast(box):
    return (box["x"] + box["w"], box["y"] + box["h"])


def pt_southwest(box):
    return (box["x"], box["y"])


def add_arrow(ax, p1, p2):
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.2
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 3.3))

    had = add_box(
        ax, 0.3, 1.95, 2.15, 0.92,
        "Hadron set\npoint cloud\n$p_T,\\,\\Delta\\eta,\\,\\sin\\Delta\\phi,\\,\\cos\\Delta\\phi,\\,q$",
        fontsize=9.5
    )

    enc = add_box(
        ax, 2.95, 1.95, 1.75, 0.92,
        "Hadron\nencoder",
        fontsize=10
    )

    ele = add_box(
        ax, 2.95, 0.35, 1.75, 0.82,
        "Electron\n$p_T,\\,\\eta,\\,q$",
        fontsize=10
    )

    setenc = add_box(
        ax, 5.15, 1.15, 2.05, 0.9,
        "Set encoder\nDS / TF / GNN",
        fontsize=10
    )

    pool = add_box(
        ax, 5.15, 0.0, 2.05, 0.82,
        "Pooling\nsum / mean / attention",
        fontsize=10
    )

    clf = add_box(
        ax, 7.9, 0.0, 1.6, 0.82,
        "Classifier",
        fontsize=10
    )

    out = add_box(
        ax, 10.0, 0.0, 1.45, 0.82,
        "Output\n$D/B$",
        fontsize=10
    )

    add_arrow(ax, pt_right(had), pt_left(enc))
    add_arrow(ax, pt_right(enc), pt_left(setenc))
    add_arrow(ax, pt_northeast(ele), pt_southwest(setenc))
    add_arrow(ax, pt_bottom(setenc), pt_top(pool))
    add_arrow(ax, pt_right(pool), pt_left(clf))
    add_arrow(ax, pt_right(clf), pt_left(out))

    ax.set_xlim(0, 11.8)
    ax.set_ylim(-0.15, 3.1)
    ax.axis("off")

    plt.tight_layout(pad=0.2)
    plt.savefig("model_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
