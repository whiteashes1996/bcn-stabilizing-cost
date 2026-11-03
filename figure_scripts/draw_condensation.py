"""
Global SCC-condensation view (Steps 1--2), drawn with NetworkX + matplotlib.

Companion to figures/augment_networkx.py: that figure zooms into ONE component
(the downstream folding, Step 3); this figure shows the GLOBAL picture, i.e. how
G|_R is triangularized into a DAG of components solved in reverse topological
order, with the hard kernel confined to beneficial-cycle SCCs.

The condensed DAG drawn here:
    S_a (beneficial-cycle SCC)  --> t1 --> t2 --\
                                                 +--> S_e (contains x_e, sink)
    S_b (easy SCC)            ---------------- /
i.e. a hard SCC reaches the sink through a chain of trivial single-state
components, while an easy SCC depends on the sink directly.

To re-label, edit only the LABELS dictionary below.

Output: figures/condensation_dag.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch
import networkx as nx

# ---------------------------------------------------------------------------
# All text labels live here. Edit these (and nothing else) to re-label.
# ---------------------------------------------------------------------------
LABELS = {
    "Sa":        r"$S_a$",
    "Sa_sub":    r"beneficial-cycle SCC",
    "Sa_foot":   r"hard core: SolveCore",
    "Sb":        r"$S_b$",
    "Sb_sub":    r"easy SCC",
    "Sb_foot":   r"$O(|S_b|)$ sweeps",
    "t1":        r"$t_1$",
    "t2":        r"$t_2$",
    "triv_foot": r"trivial components",
    "Se":        r"$S_e \ni x_e$",
    "Se_sub":    r"sink, $J_\lambda^*(x_e)=0$",
    "bnd_edge":  r"boundary $J_\lambda^*(b)$",
    "order":     r"reverse topological order (solve $S_e$ first)",
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
})

BLACK = "black"
GRAY = "0.45"
GREEN = "#1b7837"
RED = "#b2182b"

# component layout (centres) and sizes
POS = {
    "Sa": (0.0, 1.05),
    "Sb": (0.0, -1.25),
    "t1": (2.3, 1.05),
    "t2": (3.7, 1.05),
    "Se": (5.7, -0.10),
}
ELL = {"Sa": (2.0, 1.30), "Sb": (2.0, 1.30), "Se": (2.1, 1.35)}
RTRIV = 0.36


def add_scc(ax, key, color, dashed, main, sub):
    w, h = ELL[key]
    cx, cy = POS[key]
    e = Ellipse((cx, cy), w, h, facecolor="none", edgecolor=color,
                linewidth=1.6, linestyle=((0, (5, 3)) if dashed else "solid"))
    ax.add_patch(e)
    ax.text(cx, cy + 0.20, main, ha="center", va="center", color=color)
    ax.text(cx, cy - 0.26, sub, ha="center", va="center", color=color,
            fontsize=9.5)
    return e


def add_trivial(ax, key, label):
    cx, cy = POS[key]
    c = Circle((cx, cy), RTRIV, facecolor="white", edgecolor=BLACK, linewidth=1.6)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center")
    return c


def arrow(ax, pa, pb, color=BLACK, rad=0.0):
    a = FancyArrowPatch(
        POS_of(pa) if isinstance(pa, str) else pa,
        POS_of(pb) if isinstance(pb, str) else pb,
        patchA=PATCH[pa] if isinstance(pa, str) else None,
        patchB=PATCH[pb] if isinstance(pb, str) else None,
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color,
        shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    return a


def POS_of(k):
    return POS[k]


def main():
    # NetworkX DAG of components (defines structure + reverse topological order)
    G = nx.DiGraph()
    G.add_edges_from([("Sa", "t1"), ("t1", "t2"), ("t2", "Se"),
                      ("Sb", "Se")])

    fig, ax = plt.subplots(figsize=(10, 4.6))

    global PATCH
    PATCH = {}
    PATCH["Sa"] = add_scc(ax, "Sa", RED, True, LABELS["Sa"], LABELS["Sa_sub"])
    PATCH["Sb"] = add_scc(ax, "Sb", BLACK, False, LABELS["Sb"], LABELS["Sb_sub"])
    PATCH["t1"] = add_trivial(ax, "t1", LABELS["t1"])
    PATCH["t2"] = add_trivial(ax, "t2", LABELS["t2"])
    PATCH["Se"] = add_scc(ax, "Se", GREEN, False, LABELS["Se"], LABELS["Se_sub"])

    for (u, v) in G.edges():
        arrow(ax, u, v, color=GRAY if v == "Se" else BLACK,
              rad=-0.18 if (u, v) == ("Sb", "Se") else 0.0)

    # footnotes under components
    ax.text(POS["Sa"][0], POS["Sa"][1] - 0.95, LABELS["Sa_foot"],
            ha="center", va="top", color=RED, fontsize=9.5)
    ax.text(POS["Sb"][0], POS["Sb"][1] - 0.95, LABELS["Sb_foot"],
            ha="center", va="top", color=GRAY, fontsize=9.5)
    ax.text((POS["t1"][0] + POS["t2"][0]) / 2, POS["t1"][1] - 0.62,
            LABELS["triv_foot"], ha="center", va="top", color=GRAY, fontsize=9.5)

    # boundary-value label on t2 -> Se
    ax.text(5.15, 0.55, LABELS["bnd_edge"], ha="center", va="bottom",
            color=GRAY, fontsize=9.5)

    # reverse-topological-order banner (arrow pointing upstream)
    yb = 2.45
    ax.add_patch(FancyArrowPatch((6.4, yb), (-0.9, yb), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color="0.25"))
    ax.text(2.75, yb + 0.12, LABELS["order"], ha="center", va="bottom",
            color="0.25", fontsize=10.5)

    ax.set_xlim(-1.4, 7.1)
    ax.set_ylim(-2.7, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("figures/condensation_dag.png", dpi=200)
    fig.savefig("figures/condensation_dag.pdf")
    print("saved figures/condensation_dag.png and .pdf")


if __name__ == "__main__":
    main()
