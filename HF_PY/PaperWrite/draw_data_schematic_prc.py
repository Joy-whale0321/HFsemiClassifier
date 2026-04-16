import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Rectangle
import numpy as np


def add_arrow(ax, p1, p2, lw=1.8, ms=14, ls='-', z=2):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2,
            arrowstyle='->',
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color='black',
            zorder=z
        )
    )


def add_text(ax, x, y, s, fs=14, ha='center', va='center',
             weight='normal', style='normal'):
    ax.text(
        x, y, s,
        fontsize=fs,
        ha=ha,
        va=va,
        weight=weight,
        style=style
    )


def add_box(ax, xy, w, h, lw=1.5):
    x, y = xy
    rect = Rectangle((x, y), w, h, fill=False, linewidth=lw, color='black')
    ax.add_patch(rect)
    return rect


def unit_vec(deg):
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad), np.sin(rad)])


def main():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # =========================================================
    # Left: physics sketch
    # =========================================================
    collision = np.array([2.8, 3.45])

    # hadrons: 2 away-side + 2 near-side
    hadron_specs = [
        (150, 2.5),   # away upper
        (-125, 2.55), # away lower
        (35, 2.2),    # near upper -> h_i
        (-30, 2.0),   # near lower
    ]

    hadron_ends = []
    for ang, length in hadron_specs:
        end = collision + length * unit_vec(ang)
        hadron_ends.append(end)
        add_arrow(ax, tuple(collision), tuple(end), lw=1.7, ms=13, z=1)

    # collision point on top
    ax.add_patch(Circle(tuple(collision), 0.16, color="#4C9ED9", zorder=5))

    add_text(ax, 4.1, 6.25, "charged hadrons in the same event", fs=19)
    add_text(ax, collision[0], 0.8, "pp collision", fs=18)

    hi_end = hadron_ends[2]
    add_text(ax, hi_end[0] + 0.16, hi_end[1] + 0.10, r"$h_i$", fs=18, ha='left')
    add_text(
        ax,
        hi_end[0] + 0.85,
        hi_end[1] + 0.52,
        r"$(q_i,\ p_{T,i},\ \Delta\eta_i,\ \Delta\phi_i)$",
        fs=16,
        ha='left'
    )

    # D/B
    decay = collision + np.array([2.55, -0.03])
    add_arrow(ax, tuple(collision), tuple(decay), lw=2.4, ms=14, z=2)
    add_text(ax, (collision[0] + decay[0]) / 2, collision[1] - 0.58, "D/B", fs=22)

    ax.add_patch(Circle(tuple(decay), 0.055, color='black', zorder=4))

    # decay products
    e_dir = 20
    e_end = decay + 2.10 * unit_vec(e_dir)
    nu_end = decay + 1.95 * unit_vec(-10)
    x_end = decay + 1.55 * unit_vec(-58)

    add_arrow(ax, tuple(decay), tuple(e_end), lw=2.0, ms=13, z=2)
    add_arrow(ax, tuple(decay), tuple(x_end), lw=1.7, ms=13, z=2)

    ax.plot(
        [decay[0], nu_end[0]],
        [decay[1], nu_end[1]],
        linestyle=(0, (8, 5)),
        linewidth=1.8,
        color='black',
        zorder=1
    )

    add_text(ax, e_end[0] + 0.10, e_end[1] + 0.02, r"$e$", fs=18, ha='left')
    add_text(ax, nu_end[0] + 0.12, nu_end[1] - 0.02, r"$\nu_e$", fs=17, ha='left')
    add_text(ax, x_end[0] + 0.10, x_end[1] - 0.02, r"$X$", fs=18, ha='left')

    # Delta phi
    arc = Arc(
        tuple(decay),
        width=1.8,
        height=1.8,
        theta1=20,
        theta2=35,
        linewidth=1.8
    )
    ax.add_patch(arc)
    add_text(ax, decay[0] + 0.95, decay[1] + 0.80, r"$\Delta\phi$", fs=18)

    # connector to right box
    add_arrow(ax, (7.85, 3.45), (8.65, 3.45), lw=1.2, ms=10, z=1)

    # =========================================================
    # Right: dataset construction box
    # =========================================================
    box_x, box_y, box_w, box_h = 8.85, 1.45, 6.0, 4.95
    add_box(ax, (box_x, box_y), box_w, box_h, lw=1.5)

    x0 = box_x + 0.35
    y = box_y + box_h - 0.40

    add_text(ax, x0, y, "Dataset construction", fs=21, ha='left', weight='bold')

    y -= 0.58
    add_text(ax, x0, y, "Trigger electron:", fs=16, ha='left', weight='bold')
    y -= 0.50
    add_text(ax, x0 + 0.18, y, r"heavy-flavor semi-leptonic $e$", fs=15, ha='left')
    y -= 0.48
    add_text(ax, x0 + 0.18, y, r"$p_T^e > 3\ \mathrm{GeV}/c$", fs=15, ha='left')

    y -= 0.72
    add_text(ax, x0, y, "Associated hadrons:", fs=16, ha='left', weight='bold')
    y -= 0.50
    add_text(ax, x0 + 0.18, y, "final-state, charged, non-lepton", fs=15, ha='left')

    y -= 0.72
    add_text(ax, x0, y, "Per-hadron features relative to the electron:", fs=16, ha='left', weight='bold')
    y -= 0.50
    add_text(ax, x0 + 0.18, y, r"$(q_i,\ p_{T,i},\ \Delta\eta_i,\ \Delta\phi_i)$", fs=15, ha='left')

    y -= 0.72
    add_text(ax, x0, y, "Set input:", fs=16, ha='left', weight='bold')
    y -= 0.50
    add_text(ax, x0 + 0.18, y, r"$\{h_i\}_{i=1}^{N_h}$", fs=16, ha='left')

    plt.tight_layout()
    plt.savefig("hf_data_schematic_clean.pdf", bbox_inches="tight")
    plt.savefig("hf_data_schematic_clean.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()