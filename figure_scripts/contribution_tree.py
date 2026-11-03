r"""Contribution tree for PPT / oral presentation (NOT included in main.tex).

Layout:
  ROOT -> {A undiscounted (baseline), B discounted (our focus)}
  B -> beneficial-cycle criterion -> SHARED PIPELINE (big central box:
       expanded Step 1/2/3 text + the paper's condensation-DAG figure embedded)
  pipeline -> per-component solving (DAG layer) with three cases.

Gold star (*) marks a genuine contribution; gray marks standard tools / baseline.

Outputs:
  figures/contribution_tree.pdf  (vector, for slides)
  figures/contribution_tree.png  (raster preview, 200 dpi)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(__file__)
OUT_PDF = os.path.join(HERE, "contribution_tree.pdf")
OUT_PNG = os.path.join(HERE, "contribution_tree.png")
EMBED = os.path.join(HERE, "condensation_dag.png")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
})

# ---- palette ---------------------------------------------------------------
INK = "#1f2329"
GRAY = "#6b7178"
BLUE = "#2166ac"

STD_EC = "#9aa0a6"
STD_FC = "#f4f5f6"
STD_TC = "#4b5158"

CON_EC = "#c8860a"
CON_FC = "#fff6e0"
CON_TC = "#9a6500"
STAR = "#e8a317"

BAND_EC = "#b9c0c7"
BAND_FC = "#f0f6fb"

PIPE_FC = "#fcfcfd"
PIPE_EC = "#8a9097"

TODO_EC = "#c0392b"
TODO_FC = "#fdecea"
TODO_TC = "#a5281b"

TBL_EC = "#b9c0c7"
TBL_FC = "#fafbfc"


def star_at(ax, x, y, size=15):
    ax.plot([x], [y], marker="*", markersize=size, color=STAR,
            markeredgecolor=CON_EC, markeredgewidth=0.6, zorder=6)


def node(ax, cx, cy, w, h, title, body, contribution=False,
         title_fs=12.5, body_fs=10.0):
    if contribution:
        fc, ec, tc, lw = CON_FC, CON_EC, CON_TC, 2.1
    else:
        fc, ec, tc, lw = STD_FC, STD_EC, STD_TC, 1.5
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    top = cy + h / 2
    if title:
        ax.text(cx, top - 0.32, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=tc, zorder=4)
        body_top = top - 0.32 - 0.46
    else:
        body_top = top - 0.38
    for i, line in enumerate(body):
        ax.text(cx, body_top - i * 0.40, line, ha="center", va="center",
                fontsize=body_fs, color=INK, zorder=4)
    if contribution:
        star_at(ax, cx - w / 2 + 0.30, top - 0.30)


def connect(ax, parent, child, color=GRAY):
    (pcx, pcy, _pw, ph) = parent
    (ccx, ccy, _cw, ch) = child
    ax.add_patch(FancyArrowPatch(
        (pcx, pcy - ph / 2), (ccx, ccy + ch / 2),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
        color=color, shrinkA=1, shrinkB=1, zorder=2))


def tblock(ax, x, y, header, lines, gold=False):
    hc = CON_TC if gold else STD_TC
    ax.text(x, y, header, ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=hc, zorder=5)
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y - 0.44 - i * 0.36, line, ha="left", va="top",
                fontsize=9.5, color=INK, zorder=5)
    if gold:
        star_at(ax, x - 0.24, y - 0.16, size=13)


fig, ax = plt.subplots(figsize=(13.2, 15.6))

# ---- node geometry: (cx, cy, w, h) -----------------------------------------
ROOT = (9.6, 15.30, 9.4, 1.05)
A = (3.4, 13.30, 4.4, 1.35)
B = (14.7, 13.30, 6.6, 1.70)
A_LEAF = (3.4, 10.35, 5.1, 2.70)
CRIT = (14.7, 10.55, 6.6, 1.80)
PIPE = (9.95, 6.30, 14.8, 4.90)
CONT = (9.95, 0.55, 17.4, 3.45)
C_TRIV = (3.7, 0.05, 4.6, 2.45)
C_EASY = (9.95, 0.05, 5.3, 2.45)
C_HARD = (16.0, 0.05, 5.4, 2.45)

# ---- container (drawn first, behind) ---------------------------------------
ax.add_patch(FancyBboxPatch(
    (CONT[0] - CONT[2] / 2, CONT[1] - CONT[3] / 2), CONT[2], CONT[3],
    boxstyle="round,pad=0.05,rounding_size=0.14",
    facecolor=BAND_FC, edgecolor=BAND_EC, linewidth=1.4,
    linestyle=(0, (6, 3)), zorder=1))
ax.text(CONT[0], CONT[1] + CONT[3] / 2 - 0.34,
        "Across components: solve in reverse topological order over the "
        "condensed DAG  (boundary $J_\\lambda^*(b)$ propagates)",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=GRAY, zorder=2)
ax.text(CONT[0], CONT[1] + CONT[3] / 2 - 0.72,
        "within each component, exactly one of:",
        ha="center", va="center", fontsize=9.5, style="italic",
        color=GRAY, zorder=2)

# ---- arrows ----------------------------------------------------------------
connect(ax, ROOT, A, color=STD_EC)
connect(ax, ROOT, B, color=CON_EC)
connect(ax, A, A_LEAF, color=STD_EC)
connect(ax, B, CRIT, color=CON_EC)
connect(ax, CRIT, PIPE, color=CON_EC)
ax.add_patch(FancyArrowPatch(
    (PIPE[0], PIPE[1] - PIPE[3] / 2), (PIPE[0], CONT[1] + CONT[3] / 2),
    arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
    color=GRAY, shrinkA=1, shrinkB=1, zorder=2))

# ---- standard / baseline nodes ---------------------------------------------
node(ax, *ROOT,
     title="Minimum control cost under stabilization",
     body=["finite-state systems   (BCN as the motivating special case)"],
     title_fs=14, body_fs=11)
node(ax, *A, title=r"A.  Undiscounted  ($\lambda=1$)",
     body=["baseline"], title_fs=13, body_fs=10.5)
node(ax, *A_LEAF, title="Single-sink shortest path",
     body=["one reverse Dijkstra sweep",
           r"$J^*$ + proper in-tree $\mu^*$",
           r"exact,  $O(|E|+|R|\log|R|)$",
           "(standard; cycles harmless)"])

# ---- contribution nodes ----------------------------------------------------
node(ax, *B, title=r"B.  Discounted  ($\lambda\in(0,1)$)",
     body=[r"proper-feedback $J_\lambda^* \geq V_\lambda^*$",
           r"std VI / Floyd return $V_\lambda^*$ (wrong)"],
     contribution=True, title_fs=13, body_fs=10.0)
node(ax, *CRIT, title="Beneficial-cycle criterion",
     body=[r"Lemma:  $g_{\mathrm{loop}} < (1-\lambda^{\ell})\,J_{\lambda}^{\mathrm{tail}}$",
           r"$\Rightarrow$ exactly when $V_\lambda^* < J_\lambda^*$",
           r"$\Rightarrow$ why reverse Dijkstra fails"],
     contribution=True, body_fs=10.0)

# ---- SHARED PIPELINE: big box with expanded Step 1/2/3 + embedded paper figure
px0, px1 = PIPE[0] - PIPE[2] / 2, PIPE[0] + PIPE[2] / 2
py0, py1 = PIPE[1] - PIPE[3] / 2, PIPE[1] + PIPE[3] / 2
ax.add_patch(FancyBboxPatch(
    (px0, py0), PIPE[2], PIPE[3],
    boxstyle="round,pad=0.05,rounding_size=0.12",
    facecolor=PIPE_FC, edgecolor=PIPE_EC, linewidth=1.7, zorder=3))
ax.text(PIPE[0], py1 - 0.34, "Shared pipeline   (three graph transforms)",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=STD_TC, zorder=5)

# left column: expanded Step 1 / Step 2 / Step 3
tx = px0 + 0.55
tblock(ax, tx, py1 - 0.95,
       "Step 1.  Restriction",
       [r"keep $R=\{x:\ x\to^* x_e\}$, drop the rest;",
        r"states outside $R$ have $J_\lambda^*=\infty$.  $O(|E|)$"])
tblock(ax, tx, py1 - 2.20,
       "Step 2.  Condensation",
       ["collapse each strongly-connected team to a point",
        r"$\Rightarrow$ a DAG, giving a reverse topological order"])
tblock(ax, tx, py1 - 3.45,
       "Step 3.  Folding $\\to \\omega$",
       [r"fold solved downstream into one absorbing $\omega$,",
        r"exit $e(x)=g+\lambda J_\lambda^*(b)$ (single $\lambda$);",
        r"each component $\to$ discounted simple path"],
       gold=True)

# separator between text and figure
ax.plot([PIPE[0] + 0.10, PIPE[0] + 0.10], [py0 + 0.35, py1 - 0.65],
        color="#d4d7dc", linewidth=1.0, zorder=4)

# embedded paper figure (condensation DAG)
img = mpimg.imread(EMBED)
ratio = img.shape[0] / img.shape[1]
img_w = 6.7
img_h = img_w * ratio
img_cx, img_cy = PIPE[0] + 3.85, PIPE[1] - 0.05
ex0, ex1 = img_cx - img_w / 2, img_cx + img_w / 2
ey0, ey1 = img_cy - img_h / 2, img_cy + img_h / 2
ax.imshow(img, extent=[ex0, ex1, ey0, ey1], aspect="auto", zorder=4)
ax.add_patch(Rectangle((ex0, ey0), img_w, img_h, fill=False,
             edgecolor="#cfd3d8", linewidth=0.8, zorder=5))
ax.text(img_cx, ey0 - 0.16, "paper Fig.: condensation DAG (Step 2) + reverse-topo order",
        ha="center", va="top", fontsize=8.5, style="italic",
        color=GRAY, zorder=5)

# ---- three per-component cases (inside container) --------------------------
node(ax, *C_TRIV, title="Trivial node (DAG)",
     body=["no internal arc", r"$J_\lambda^* = \min$ exit",
           "(one Bellman step)"])
node(ax, *C_EASY, title="SCC, no beneficial cycle",
     body=["label-correcting (not Dijkstra)",
           r"walk = simple path $\Rightarrow$ in-tree",
           r"exact cost = edges $\times$ iterations,",
           "iterations = feedback depth (small)"])
node(ax, *C_HARD, title="SCC with beneficial cycle",
     body=["core is NP-hard (Ham. path)",
           r"SolveCore: subset DP",
           r"$O(2^{|S|}\,|S|\,M)$, exp. only in $|S|$",
           "+ depth-parametrized scheme"],
     contribution=True)

# ---- TODO note: beneficial-cycle detection step ----------------------------
todo_cx, todo_cy, todo_w, todo_h = 14.75, 3.05, 7.7, 1.30
ax.add_patch(FancyBboxPatch(
    (todo_cx - todo_w / 2, todo_cy - todo_h / 2), todo_w, todo_h,
    boxstyle="round,pad=0.05,rounding_size=0.10",
    facecolor=TODO_FC, edgecolor=TODO_EC, linewidth=1.6,
    linestyle=(0, (5, 2.5)), zorder=4))
ax.text(todo_cx, todo_cy + 0.36, "TODO  (detection step)",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=TODO_TC, zorder=5)
ax.text(todo_cx, todo_cy - 0.06,
        r"which case? test whether the $V_\lambda^*$-greedy policy is improper",
        ha="center", va="center", fontsize=9.3, color=INK, zorder=5)
ax.text(todo_cx, todo_cy - 0.40,
        r"improper $\Rightarrow$ beneficial cycle $\Rightarrow$ SolveCore;  "
        r"else $V_\lambda^*=J_\lambda^*$",
        ha="center", va="center", fontsize=9.3, color=INK, zorder=5)
ax.add_patch(FancyArrowPatch(
    (todo_cx - 1.6, todo_cy - todo_h / 2), (C_HARD[0], C_HARD[1] + C_HARD[3] / 2),
    arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
    color=TODO_EC, linestyle=(0, (4, 2)), shrinkA=2, shrinkB=2, zorder=4))

# ---- variable table at the very bottom -------------------------------------
TBL_X0, TBL_X1 = 0.6, 18.8
TBL_TOP, TBL_BOT = -1.55, -5.55
ax.add_patch(FancyBboxPatch(
    (TBL_X0, TBL_BOT), TBL_X1 - TBL_X0, TBL_TOP - TBL_BOT,
    boxstyle="round,pad=0.05,rounding_size=0.10",
    facecolor=TBL_FC, edgecolor=TBL_EC, linewidth=1.4, zorder=1))
ax.text((TBL_X0 + TBL_X1) / 2, TBL_TOP - 0.36, "Symbols",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=STD_TC, zorder=2)

table_cols = [
    [
        (r"$x_e$", "target equilibrium (cost-free sink)"),
        (r"$\lambda$", r"discount factor, $\lambda\in(0,1)$"),
        (r"$g$", r"stage cost $g(x,u)\geq 0$"),
        (r"$J^*$", "min undiscounted stab. cost"),
        (r"$J_\lambda^*$", "min discounted stab. cost (proper)"),
        (r"$V_\lambda^*$", "unconstrained optimum (VI / Bellman)"),
    ],
    [
        (r"$\mu^*$", "optimal proper feedback (in-tree)"),
        (r"$R$", r"stabilizable set (can reach $x_e$)"),
        (r"$M$", "number of control inputs"),
        (r"$|E|$", "number of arcs (transitions)"),
        (r"$D$", r"optimal-feedback depth ($\leq|R|$)"),
        (r"$S,\,|S|$", "a component (SCC) and its size"),
    ],
    [
        (r"$\omega$", "virtual absorbing sink node"),
        (r"$e(x)$", r"exit weight $g+\lambda J_\lambda^*(b)$"),
        (r"$b$", r"downstream boundary state ($\notin S$)"),
        (r"$C,\,\ell$", "non-target cycle and its length"),
        (r"$g_{\mathrm{loop}}$", r"one-pass discounted cost of $C$"),
        (r"$J_{\lambda}^{\mathrm{tail}}$", "proper discounted cost-to-go"),
    ],
]
col_x = [1.15, 7.35, 13.35]
row_y0 = TBL_TOP - 0.95
row_step = 0.475
for ci, col in enumerate(table_cols):
    sx = col_x[ci]
    for ri, (sym, meaning) in enumerate(col):
        yy = row_y0 - ri * row_step
        ax.text(sx, yy, sym, ha="left", va="center", fontsize=10.5,
                color=BLUE, zorder=2)
        ax.text(sx + 1.05, yy, meaning, ha="left", va="center",
                fontsize=9.5, color=INK, zorder=2)

# ---- legend ----------------------------------------------------------------
lx, ly = 1.35, 15.55
star_at(ax, lx, ly)
ax.text(lx + 0.35, ly, "= our contribution", ha="left", va="center",
        fontsize=10.5, color=CON_TC, fontweight="bold")
ax.add_patch(FancyBboxPatch((lx - 0.15, ly - 0.82), 0.42, 0.30,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             facecolor=STD_FC, edgecolor=STD_EC, linewidth=1.3, zorder=5))
ax.text(lx + 0.35, ly - 0.67, "= standard tool / baseline", ha="left",
        va="center", fontsize=10.5, color=STD_TC)

ax.set_xlim(0.3, 19.4)
ax.set_ylim(-5.8, 16.1)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print("wrote", OUT_PDF)
print("wrote", OUT_PNG)
