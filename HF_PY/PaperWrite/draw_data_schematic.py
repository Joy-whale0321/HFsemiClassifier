import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc
import numpy as np


def add_arrow(ax, p1, p2, lw=1.8, ms=14, ls='-', z=1):
    ax.add_patch(FancyArrowPatch(
        p1, p2,
        arrowstyle='->',
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color='black',
        zorder=z
    ))


def add_text(ax, x, y, s, fs=13, ha='center', va='center'):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va)


def unit_vec(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def main():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    # =========================================================
    # collision point
    # =========================================================
    collision = np.array([3.25, 3.25])

    # =========================================================
    # hadrons: 2 away-side + 2 near-side
    # relative to D/B direction (~0 deg, to the right)
    # near-side: around 0 deg
    # away-side: around 180 deg
    # =========================================================
    hadron_angles = [32, -32, 145, -120]
    hadron_lengths = [2.55, 2.70, 2.90, 3.00]

    hadron_ends = []
    for ang, L in zip(hadron_angles, hadron_lengths):
        end = collision + L * unit_vec(ang)
        hadron_ends.append(end)
        add_arrow(ax, tuple(collision), tuple(end), lw=1.7, z=1)

    # collision on top
    ax.add_patch(Circle(tuple(collision), 0.22, color='#4AA3DF', zorder=5))
    add_text(ax, collision[0], 0.55, "pp collision", fs=16)

    # only one hadron label
    hi_end = hadron_ends[0]
    add_text(ax, hi_end[0] + 0.12, hi_end[1] + 0.12, r"$h_i$", fs=18, ha='left')

    # =========================================================
    # title: move clearly to the left
    # =========================================================
    add_text(ax, 8.1, 6.25, "charged hadrons in the same event", fs=18)

    # =========================================================
    # D/B
    # =========================================================
    decay = collision + np.array([2.25, -0.03])
    add_arrow(ax, tuple(collision), tuple(decay), lw=2.8, z=2)
    add_text(ax, (collision[0] + decay[0]) / 2, collision[1] - 0.78, "D/B", fs=24)

    ax.add_patch(Circle(tuple(decay), 0.055, color='black', zorder=3))

    # =========================================================
    # decay products
    # =========================================================
    e_dir = 20
    e_end = decay + 2.45 * unit_vec(e_dir)
    nue_end = decay + 2.55 * unit_vec(-10)
    x_end = decay + 1.95 * unit_vec(-58)

    add_arrow(ax, tuple(decay), tuple(e_end), lw=2.1, z=2)
    add_arrow(ax, tuple(decay), tuple(x_end), lw=1.8, z=2)

    ax.plot(
        [decay[0], nue_end[0]],
        [decay[1], nue_end[1]],
        linestyle=(0, (8, 5)),
        linewidth=2.0,
        color='black',
        zorder=1
    )

    add_text(ax, e_end[0] + 0.10, e_end[1] + 0.02, r"$e$", fs=19, ha='left')
    add_text(ax, nue_end[0] + 0.12, nue_end[1] - 0.02, r"$\nu_e$", fs=17, ha='left')
    add_text(ax, x_end[0] + 0.10, x_end[1] - 0.05, r"$X$", fs=19, ha='left')

    # trigger electron text: move left a bit more
    add_text(ax, e_end[0] + 0.62, e_end[1] + 0.00, r"trigger $e$", fs=18, ha='left')
    add_text(ax, e_end[0] + 0.82, e_end[1] - 0.42, r"$p_T > 3\ \mathrm{GeV}/c$", fs=18, ha='left')

    # =========================================================
    # Delta phi
    # =========================================================
    arc = Arc(
        tuple(decay),
        width=2.5,
        height=2.5,
        theta1=20,
        theta2=32,
        linewidth=2.0
    )
    ax.add_patch(arc)
    add_text(ax, decay[0] + 1.32, decay[1] + 1.12, r"$\Delta\phi$", fs=19)

    # =========================================================
    # feature annotation: move left/down more
    # =========================================================
    add_text(
        ax,
        hi_end[0] + 0.55,
        hi_end[1] + 0.42,
        r"$(p_T,\ q,\ \Delta\eta,\ \Delta\phi)$",
        fs=17,
        ha='left'
    )

    plt.tight_layout()
    plt.savefig("hf_event_sketch_v6.pdf", bbox_inches='tight')
    plt.savefig("hf_event_sketch_v6.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()