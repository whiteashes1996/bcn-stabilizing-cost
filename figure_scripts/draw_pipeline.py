"""
Global pipeline of Algorithm (alg:master) with the cost of every stage, drawn
with matplotlib.

Companion to figures/draw_condensation.py (the global SCC view, Steps 1--2) and
figures/draw_augment.py (the per-component downstream folding, Step 3): this figure shows
the CONTROL FLOW of the master algorithm and annotates each stage with its time
complexity, so that the polynomial skeleton and the exponential kernel can be
read off at a glance.

    input -> (Step 1) -> (Step 2) -> reverse topological order
          -> [ per component: (Step 3) -> beneficial-cycle test -> easy | hard ]
          -> any SCC left? --yes--> Step 3
                           --no---> all S solved -> J*; then RecoverFeedback

The beneficial-cycle test is the label-correcting sweeps of T from e:
D(S)+1 sweeps if S has no beneficial cycle, otherwise a cap of |S|.

Polynomial stages are annotated in green; exponential (kernel) costs in red.

All connectors are orthogonal polylines (no arcs).  Label text is kept short on
purpose: the figure is typeset at roughly half its natural width, so a line much
longer than ~40 characters becomes unreadable in print.  To re-label, edit only
the LABELS dictionary below.

Output: figures/pipeline_complexity.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

# ---------------------------------------------------------------------------
# All text labels live here. Edit these (and nothing else) to re-label.
# Strings may use matplotlib mathtext, e.g. r"$O(|E|)$".
# ---------------------------------------------------------------------------
LABELS = {
    "in_1":      r"input",
    "in_2":      r"$\mathcal{G}=(\mathcal{V},\mathcal{E})$, $x_e$, $\lambda\in(0,1)$",

    "t1_1":      r"(Step 1) Restriction",
    "t1_2":      r"restrict to $\mathcal{V}_s$, $J_\lambda^*(x_e)=0$",
    "t1_c":      r"$O(|\mathcal{E}|)$",

    "t2_1":      r"(Step 2) Condensation",
    "t2_2":      r"build the DAG $\mathcal{G}_{\mathrm{scc}}$",
    "t2_c":      r"$O(|\mathcal{V}_s|+|\mathcal{E}_{\mathcal{V}_s}|)$",

    "loop":      r"for each component $S$, in reverse topological order",

    "t3_1":      r"(Step 3) Folding",
    "t3_2":      r"downstream $\to$ absorbing $\omega$",
    "t3_3":      r"$e(x)=\min\{g(x,u)+\lambda J_\lambda^*(b)\}$",
    "t3_c":      r"$O(M|S|)$ per component",

    "dia_1":     r"beneficial cycle in $S$?",
    "dia_2":     r"sweeps of $T$ from $e$",
    "dia_c":     r"w/o BC: $O(M|S|D(S))$",
    "dia_c2":    r"with BC: $O(M|S|^2)$",

    "yes":       r"yes",
    "no":        r"no",

    "more_1":    r"any SCC left?",

    "easy_1":     r"easy: the test has already solved $S$",
    "easy_2":     r"$J_\lambda^*=V_\lambda^*$; store the value",
    "easy_c":     r"$O(1)$",

    "hard_1":    r"beneficial-cycle SCC (NP-complete):",
    "hard_2":    r"min discounted-cost simple path to $\omega$",
    "hard_3":    r"$\mathbf{SolveCore}$ (subset DP): $O(2^{|S|}|S|M)$",

    "back":      (r"next component: solved $S$ gives"
                  r" boundary data $J_\lambda^*(b)$"),
    "exit":      r"all $S$ solved",

    "out_1":     r"output",
    "out_2":     r"recover $\mu(x)$ from $J_\lambda^*$",
    "out_3":     r"$O(M|\mathcal{V}_s|)$",

}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16,
    "text.latex.preamble": r"\usepackage{amsmath,xcolor}",
})

BLACK = "black"
GRAY = "0.45"
GREEN = "#1b7837"
RED = "#d73027"

# The overall-complexity expression is not drawn here: it duplicates
# \eqref{eq:overall-complexity} of Proposition prop:complexity term for term,
# and at print width it exceeds the ~40-character legibility budget noted above.

# font sizes (the figure is typeset at about half its natural width)
FS_TITLE = 16
FS_SUB = 14
FS_COST = 14.5
FS_NOTE = 14

# stage centres
POS = {
    "in":   (3.15, 11.6),
    "t1":   (11.90, 11.6),
    "t2":   (20.95, 11.6),
    "t3":   (4.0, 5.7),
    "dia":  (12.20, 5.7),
    "easy":  (19.5, 8.0),
    "hard": (19.5, 3.3),
    "more": (29.80, 5.7),
    "out":  (29.80, 11.6),
}
# stage box sizes (width, height); the diamond uses half-diagonals
SIZE = {
    "in":   (5.1, 1.9),
    "t1":   (6.1, 1.9),
    "t2":   (5.7, 1.9),
    "t3":   (6.5, 2.7),
    "easy":  (8.6, 2.0),
    "hard": (8.6, 2.3),
    "out":  (6.4, 1.9),
}
DIA = (3.55, 1.72)
MORE_DIA = (1.70, 1.20)  # loop-exit test; smaller than the detect diamond

PANEL = (0.3, 0.35, 31.65, 9.95)  # x0, y0, x1, y1 of the dashed loop panel

PATCH = {}


def box(ax, key, lines, edge=BLACK, lw=1.6, gap=0.66):
    """Rounded box at POS[key]; lines is a list of (text, fontsize, colour)."""
    cx, cy = POS[key]
    w, h = SIZE[key]
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.22",
                       facecolor="white", edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    top = cy + gap * (len(lines) - 1) / 2
    for i, (text, size, color) in enumerate(lines):
        ax.text(cx, top - i * gap, text, ha="center", va="center",
                color=color, fontsize=size)
    PATCH[key] = p
    return p


def diamond(ax, key, lines, edge=BLACK, lw=1.6, gap=0.40, size=None, dy=0.0):
    """Diamond at POS[key]; lines is a list of (text, fontsize, colour)."""
    cx, cy = POS[key]
    hw, hh = DIA if size is None else size
    p = Polygon([(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
                closed=True, facecolor="white", edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    top = cy + dy + gap * (len(lines) - 1) / 2
    for i, (text, size, color) in enumerate(lines):
        ax.text(cx, top - i * gap, text, ha="center", va="center",
                color=color, fontsize=size)
    PATCH[key] = p
    return p


def arrow(ax, pa, pb, color=BLACK, lw=1.6):
    a = FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=16, lw=lw,
                        color=color, shrinkA=3, shrinkB=3,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)
    return a


def ortho(ax, pts, color=BLACK, lw=1.6, dashed=False, zorder=1):
    """Polyline through waypoints; arrowhead on the final segment."""
    style = (0, (5, 3)) if dashed else "solid"
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, lw=lw, solid_capstyle="butt",
                solid_joinstyle="miter", linestyle=style, zorder=zorder)
    # final leg: keep the line style up to the arrow root, then a solid head
    (px, py), (qx, qy) = pts[-2], pts[-1]
    seg = ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5
    head = min(0.35, seg)
    rx, ry = qx, qy
    if seg > 0:
        rx = qx - (qx - px) / seg * head
        ry = qy - (qy - py) / seg * head
    ax.plot([px, rx], [py, ry], color=color, lw=lw, solid_capstyle="butt",
            solid_joinstyle="miter", linestyle=style, zorder=zorder)
    ax.add_patch(FancyArrowPatch((rx, ry), (qx, qy), arrowstyle="-|>",
                                 mutation_scale=16, lw=lw, color=color,
                                 shrinkA=0, shrinkB=3,
                                 connectionstyle="arc3,rad=0", zorder=zorder + 1))


def block_arrow(ax, x0, x1, y, color=BLACK, lw=1.5,
                shaft=0.22, head=0.55, head_len=0.72):
    """Wide right-pointing arrow drawn as two strokes, open at the tail.

    The white fill masks whatever it crosses (here the dashed panel border), so
    the arrow reads as leaving the loop rather than as one more connector.
    """
    outline = [(x0, y - shaft), (x1 - head_len, y - shaft),
               (x1 - head_len, y - head), (x1, y),
               (x1 - head_len, y + head), (x1 - head_len, y + shaft),
               (x0, y + shaft)]
    ax.add_patch(Polygon(outline, closed=True, facecolor="white",
                         edgecolor="none", zorder=3))
    upper = [(x0, y + shaft), (x1 - head_len, y + shaft),
             (x1 - head_len, y + head), (x1, y)]
    lower = [(x0, y - shaft), (x1 - head_len, y - shaft),
             (x1 - head_len, y - head), (x1, y)]
    for path in (upper, lower):
        ax.plot([p[0] for p in path], [p[1] for p in path], color=color,
                lw=lw, solid_capstyle="butt", solid_joinstyle="miter",
                zorder=4)


def right_of(key):
    cx, cy = POS[key]
    return (cx + SIZE[key][0] / 2, cy)


def left_of(key):
    cx, cy = POS[key]
    return (cx - SIZE[key][0] / 2, cy)


def top_of(key):
    cx, cy = POS[key]
    return (cx, cy + SIZE[key][1] / 2)


def bottom_of(key):
    cx, cy = POS[key]
    return (cx, cy - SIZE[key][1] / 2)


def main():
    # figsize matches the data aspect below (33.9 / 14.65), so that
    # aspect="equal" does not letterbox the axes inside the canvas.
    fig, ax = plt.subplots(figsize=(16.18, 6.086))

    # ---- dashed panel for the per-component loop ---------------------------
    x0, y0, x1, y1 = PANEL
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.0,rounding_size=0.3",
                                facecolor="none", edgecolor=GRAY,
                                linewidth=1.3, linestyle=(0, (6, 4))))
    ax.text(4.55, y1 - 0.42, LABELS["loop"], ha="left", va="center",
            color=GRAY, fontsize=FS_NOTE + 1)

    # ---- preprocessing lane -------------------------------------------------
    box(ax, "in", [(LABELS["in_1"], FS_TITLE, BLACK),
                   (LABELS["in_2"], FS_SUB, GRAY)], gap=0.58)
    box(ax, "t1", [(LABELS["t1_1"], FS_TITLE, BLACK),
                   (LABELS["t1_2"], FS_SUB, GRAY),
                   (LABELS["t1_c"], FS_COST, GREEN)], gap=0.58)
    box(ax, "t2", [(LABELS["t2_1"], FS_TITLE, BLACK),
                   (LABELS["t2_2"], FS_SUB, GRAY),
                   (LABELS["t2_c"], FS_COST, GREEN)], gap=0.58)

    arrow(ax, right_of("in"), left_of("t1"))
    arrow(ax, right_of("t1"), left_of("t2"))

    # ---- entry into the loop (t2 -> t3, orthogonal) ------------------------
    # t2 bottom -> down to y=11.0 -> left to x=4.0 -> down into t3 top
    ortho(ax, [
        bottom_of("t2"),
        (POS["t2"][0], 10.30),
        (POS["t3"][0], 10.30),
        top_of("t3"),
    ])

    # ---- per-component stages ----------------------------------------------
    box(ax, "t3", [(LABELS["t3_1"], FS_TITLE, BLACK),
                   (LABELS["t3_2"], FS_SUB, GRAY),
                   (LABELS["t3_3"], FS_SUB, GRAY),
                   (LABELS["t3_c"], FS_COST, GREEN)], gap=0.62)
    diamond(ax, "dia", [(LABELS["dia_1"], FS_SUB, BLACK),
                        (LABELS["dia_2"], FS_SUB - 1, GRAY),
                        (LABELS["dia_c"], 12, GREEN),
                        (LABELS["dia_c2"], 12, GREEN)], gap=0.42, dy=-0.16)

    box(ax, "easy", [(LABELS["easy_1"], FS_TITLE, BLACK),
                    (LABELS["easy_2"], FS_SUB, GRAY),
                    (LABELS["easy_c"], FS_COST, GREEN)], gap=0.62)
    box(ax, "hard", [(LABELS["hard_1"], FS_TITLE - 1, BLACK),
                     (LABELS["hard_2"], FS_SUB, GRAY),
                     (LABELS["hard_3"], FS_SUB, RED)], edge=BLACK, gap=0.70)
    diamond(ax, "more", [(LABELS["more_1"], FS_SUB - 1.5, BLACK)],
            size=MORE_DIA, gap=0.0)

    arrow(ax, right_of("t3"), (POS["dia"][0] - DIA[0], POS["dia"][1]))

    # dia top -> up to easy.y -> right into easy left ("no": no beneficial cycle)
    ortho(ax, [
        (POS["dia"][0], POS["dia"][1] + DIA[1]),
        (POS["dia"][0], POS["easy"][1]),
        left_of("easy"),
    ])
    # dia bottom -> down to hard.y -> right into hard left ("yes": has beneficial cycle)
    ortho(ax, [
        (POS["dia"][0], POS["dia"][1] - DIA[1]),
        (POS["dia"][0], POS["hard"][1]),
        left_of("hard"),
    ], color=BLACK)

    ax.text(POS["dia"][0] + 0.55, POS["easy"][1] + 0.45, LABELS["no"],
            ha="left", va="center", fontsize=FS_NOTE, color=GRAY)
    ax.text(POS["dia"][0] + 0.55, POS["hard"][1] + 0.45, LABELS["yes"],
            ha="left", va="center", fontsize=FS_NOTE, color=BLACK)

    # ---- output (top row) ---------------------------------------------------
    box(ax, "out", [(LABELS["out_1"], FS_TITLE, BLACK),
                    (LABELS["out_2"], FS_SUB, GRAY),
                    (LABELS["out_3"], FS_COST, GREEN)], gap=0.58)

    # easy and hard both enter the loop-exit test from the left;
    # the vertical fold sits at the midpoint of the gap
    more_x, more_y = POS["more"]
    mhw, mhh = MORE_DIA
    join_x = (right_of("easy")[0] + (more_x - mhw)) / 2
    for key in ("easy", "hard"):
        ax.plot([right_of(key)[0], join_x], [POS[key][1], POS[key][1]],
                color=BLACK, lw=1.6, solid_capstyle="butt", zorder=1)
    ax.plot([join_x, join_x],
            [POS["easy"][1], POS["hard"][1]],
            color=BLACK, lw=1.6, solid_capstyle="butt", zorder=1)
    arrow(ax, (join_x, more_y), (more_x - mhw, more_y))

    # yes: down, left, then up into Step 3 from below (two corners)
    ortho(ax, [
        (more_x, more_y - mhh),
        (more_x, 1.0),
        (POS["t3"][0], 1.0),
        bottom_of("t3"),
    ], color=BLACK)
    ax.text(more_x + 0.35, more_y - mhh - 0.55, LABELS["yes"],
            ha="left", va="center", fontsize=FS_NOTE, color=GRAY)
    ax.text(more_x - 0.25, 1.55, LABELS["back"], ha="right", va="center",
            color=GRAY, fontsize=FS_NOTE)

    # no: every SCC is done -> solid line straight up into output
    ortho(ax, [
        (more_x, more_y + mhh),
        bottom_of("out"),
    ])
    ax.text(more_x + 0.40, 8.60, LABELS["no"],
            ha="left", va="center", fontsize=FS_NOTE, color=BLACK)
    ax.text(more_x + 0.40, 10.30, LABELS["exit"],
            ha="left", va="center", color=GRAY, fontsize=FS_NOTE)

    ax.set_xlim(-0.3, 33.6)
    ax.set_ylim(0.0, 12.75)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("figures/pipeline_complexity.png", dpi=200)
    fig.savefig("figures/pipeline_complexity.pdf")
    print("saved figures/pipeline_complexity.png and .pdf")


if __name__ == "__main__":
    main()
