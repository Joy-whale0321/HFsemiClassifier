import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
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

    fig, ax = plt.subplots(figsize=(14.0, 5.5))
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # =========================================================
    # Left: physics sketch
    # =========================================================
    collision = np.array([3.15, 3.45])

    # hadrons: two away-side + one representative near-side + one near-side lower
    hadron_specs = [
        (138, 2.45),   # away upper
        (-137, 2.55),  # away lower
        (35, 2.05),    # representative h_i
        (-40, 1.90),   # near lower
    ]

    hadron_ends = []
    for ang, length in hadron_specs:
        end = collision + length * unit_vec(ang)
        hadron_ends.append(end)
        add_arrow(ax, tuple(collision), tuple(end), lw=1.7, ms=13, z=1)

    # collision point on top
    ax.add_patch(Circle(tuple(collision), 0.16, color="#4C9ED9", zorder=5))

    add_text(ax, 4.45, 6.15, "charged hadrons in the same event", fs=18)
    add_text(ax, collision[0], 0.95, "pp collision", fs=17)

    # representative hadron label + feature text
    hi_end = hadron_ends[2]
    add_text(ax, hi_end[0] + 0.15, hi_end[1] + 0.10, r"$h_i$", fs=18, ha='left')
    add_text(
        ax,
        hi_end[0] + 0.78,
        hi_end[1] + 0.46,
        r"$(q_i,\ p_{T,i},\ \Delta\eta_i,\ \Delta\phi_i)$",
        fs=15,
        ha='left'
    )

    # D/B short line
    decay = collision + np.array([2.45, -0.03])
    add_arrow(ax, tuple(collision), tuple(decay), lw=2.3, ms=14, z=2)
    add_text(ax, (collision[0] + decay[0]) / 2, collision[1] - 0.56, "D/B", fs=21)

    # decay vertex
    ax.add_patch(Circle(tuple(decay), 0.055, color='black', zorder=4))

    # decay products
    e_dir = 20
    e_end = decay + 2.00 * unit_vec(e_dir)
    nu_end = decay + 1.90 * unit_vec(-10)
    x_end = decay + 1.50 * unit_vec(-58)

    add_arrow(ax, tuple(decay), tuple(e_end), lw=1.95, ms=13, z=2)
    add_arrow(ax, tuple(decay), tuple(x_end), lw=1.75, ms=13, z=2)

    ax.plot(
        [decay[0], nu_end[0]],
        [decay[1], nu_end[1]],
        linestyle=(0, (8, 5)),
        linewidth=1.8,
        color='black',
        zorder=1
    )

    add_text(ax, e_end[0] + 0.10, e_end[1] + 0.02, r"$e$", fs=17, ha='left')
    add_text(ax, nu_end[0] + 0.10, nu_end[1] - 0.02, r"$\nu_e$", fs=16, ha='left')
    add_text(ax, x_end[0] + 0.08, x_end[1] - 0.02, r"$X$", fs=17, ha='left')

    # subtle connector to the box
    add_arrow(ax, (7.95, 3.45), (8.55, 3.45), lw=1.05, ms=10, z=1)

    # =========================================================
    # Right: dataset construction box
    # =========================================================
    box_x, box_y, box_w, box_h = 8.75, 0.95, 6.95, 5.95
    add_box(ax, (box_x, box_y), box_w, box_h, lw=1.5)

    x0 = box_x + 0.42
    y = box_y + box_h - 0.36

    add_text(ax, x0, y, "Dataset construction", fs=19, ha='left', weight='bold')

    section_gap = 0.60
    line_gap = 0.38

    y -= 0.60
    add_text(ax, x0, y, "Trigger electron:", fs=14.5, ha='left', weight='bold')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, r"heavy-flavor semi-leptonic $e$", fs=13.2, ha='left')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, r"$p_T^e > 3\ \mathrm{GeV}/c$", fs=13.2, ha='left')

    y -= section_gap
    add_text(ax, x0, y, "Associated hadrons:", fs=14.5, ha='left', weight='bold')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, "final-state, charged, non-lepton", fs=13.2, ha='left')

    y -= section_gap
    add_text(ax, x0, y, "Per-hadron features relative to the electron:", fs=14.5, ha='left', weight='bold')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, r"$(q_i,\ p_{T,i},\ \Delta\eta_i,\ \Delta\phi_i)$", fs=13.2, ha='left')

    y -= section_gap
    add_text(ax, x0, y, "Reference object:", fs=14.5, ha='left', weight='bold')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, r"trigger electron $e$", fs=13.2, ha='left')

    y -= section_gap
    add_text(ax, x0, y, "Set input to the model:", fs=14.5, ha='left', weight='bold')
    y -= line_gap
    add_text(ax, x0 + 0.18, y, r"$\{h_i\}_{i=1}^{N_h}$", fs=13.8, ha='left')

    plt.tight_layout()
    plt.savefig("hf_data_schematic_final.pdf", bbox_inches="tight")
    plt.savefig("hf_data_schematic_final.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()