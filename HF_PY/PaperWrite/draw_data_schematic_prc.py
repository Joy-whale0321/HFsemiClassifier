import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Rectangle
import numpy as np


def add_box(ax, xy, w, h, text, fontsize=13, lw=1.5):
    x, y = xy
    rect = Rectangle((x, y), w, h, fill=False, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)


def add_arrow(ax, p1, p2, lw=1.5, ms=12):
    ax.add_patch(FancyArrowPatch(p1, p2,
                                arrowstyle='-|>',
                                mutation_scale=ms,
                                linewidth=lw))


def unit_vec(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


def main():
    fig, ax = plt.subplots(figsize=(13,5))
    ax.set_xlim(0,14)
    ax.set_ylim(0,6)
    ax.axis('off')

    # =====================
    # pp collision（缩短）
    # =====================
    c = np.array([1.5,3])

    add_arrow(ax, (0.6,3.8), (1.3,3.2))   # shorter
    add_arrow(ax, (0.6,2.2), (1.3,2.8))

    ax.text(0.5,4.0,"p")
    ax.text(0.5,2.0,"p")

    ax.add_patch(Circle(c,0.06))
    ax.text(1.5,0.9,"pp collision", ha='center')

    # =====================
    # HF hadron（缩短 + 去掉中间箭头）
    # =====================
    hf = np.array([3.4,3.7])
    decay = np.array([4.6,3.7])

    add_arrow(ax, (1.7,3.1), hf)   # shorter
    # ❌ 不再画 hf→decay 的箭头（删掉红线）

    ax.add_patch(Circle(decay,0.05))
    ax.text(3.2,4.2,"HF hadron (D/B)", ha='center')
    ax.text(4.6,4.6,"semi-leptonic decay", ha='center')

    # =====================
    # decay products（缩短）
    # =====================
    e_end = np.array([6.5,4.6])
    nu_end = np.array([6.2,3.2])

    add_arrow(ax, decay, e_end)
    add_arrow(ax, decay, nu_end)

    ax.text(6.6,4.8,r"$e^\pm$")
    ax.text(6.3,3.1,r"$\nu_e + X$")

    # =====================
    # trigger box
    # =====================
    add_box(ax,(6.9,4.2),2.3,1,
            "trigger electron\n$p_T > 3$ GeV/c",
            fontsize=12)

    add_arrow(ax, e_end, (6.9,4.7))  # shorter

    # =====================
    # hadron region
    # =====================
    ref = np.array([8.2,2.8])
    ax.add_patch(Circle(ref,0.05))

    add_arrow(ax,(8.0,4.2),(8.2,3.4))  # shorter

    # electron axis（保留，但不连右边）
    axis_end = ref + np.array([1.8,0])
    add_arrow(ax, ref, axis_end)
    ax.text(axis_end[0]+0.1,axis_end[1],"trigger electron axis", fontsize=11)

    # hadrons（短）
    angles = [60,25,-30,-60]
    lengths = [1.2,1.0,1.1,1.3]

    ends = []
    for i,(a,L) in enumerate(zip(angles,lengths)):
        end = ref + L*unit_vec(a)
        ends.append(end)
        add_arrow(ax, ref, end, lw=1.3)
        ax.text(end[0]+0.05,end[1],f"$h_{i+1}$",fontsize=11)

    # Δφ（只留一个）
    arc = Arc(ref,1.4,1.4,theta1=0,theta2=60)
    ax.add_patch(arc)
    ax.text(ref[0]+0.5,ref[1]+0.3,r"$\Delta\phi$")

    # Δη（只留虚线，无箭头）
    h1 = ends[0]
    ax.plot([h1[0],h1[0]],[ref[1],h1[1]],'--',lw=1)
    ax.text(h1[0]+0.1,ref[1]+0.5,r"$\Delta\eta$")

    ax.text(8.5,5.6,"charged hadrons in the same event", ha='center')

    # =====================
    # point cloud（无连接箭头）
    # =====================
    box_x, box_y = 10.2,1.6
    box_w, box_h = 2.7,2.6

    add_box(ax,(box_x,box_y),box_w,box_h,"",fontsize=12)

    # ❌ 删除 hadron → box 的多余箭头

    # inset axes
    o = np.array([10.6,2.0])
    add_arrow(ax,o,o+np.array([1.0,0]),lw=1,ms=9)
    add_arrow(ax,o,o+np.array([0,1.0]),lw=1,ms=9)

    ax.text(o[0]+1.1,o[1],r"$\Delta\phi$",fontsize=10)
    ax.text(o[0],o[1]+1.1,r"$\Delta\eta$",fontsize=10)

    pts = np.array([[0.2,0.2],[0.6,0.4],[0.9,0.9],[0.4,0.8]])
    for dx,dy in pts:
        ax.add_patch(Circle(o+np.array([dx,dy]),0.03))

    ax.text(box_x+box_w/2,box_y+1.3,"hadron set / point cloud", ha='center')
    ax.text(box_x+box_w/2,box_y+0.3,r"$\{(p_T,\Delta\eta,\Delta\phi)_i\}$", ha='center')

    # =====================
    # caption
    # =====================
    ax.text(7,0.3,
            "Event schematic for heavy-flavor semi-leptonic electron selection and charged-hadron set construction",
            ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig("clean_v2.pdf",bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()