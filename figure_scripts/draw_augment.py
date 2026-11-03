"""
Per-component downstream folding into a virtual sink (Step 3), drawn with NetworkX + matplotlib.

Mirrors Fig. (fig:augment) in main.tex. The SCC S is drawn once and the two
states of the transformation are overlaid on it, since folding leaves S itself
untouched and only redirects the exit arcs:
  before (grey, dashed) : exit arcs of weight g reach the downstream boundary
                          states b1, b2, whose optimal costs J*(b) are known,
                          and continue toward x_e.
  after  (green)        : every exit arc is redirected to one virtual sink
                          omega with weight g + lambda * J*(b); J*(omega) = 0.

To re-label the figure, edit only the LABELS dictionary below.

Output: figures/augment_networkx.png (and .pdf)
"""

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# All text labels live here. Edit these (and nothing else) to re-label the
# figure. Strings may use matplotlib mathtext, e.g. r"$J_\lambda^*(b_1)$".
# ---------------------------------------------------------------------------
LABELS = {
    # container / nodes
    "scc":        r"SCC $S$",
    "x":          r"$x$",
    "v1":         r"$v_1$",
    "v2":         r"$v_2$",
    "b1":         r"$b_1$",
    "b2":         r"$b_2$",
    "omega":      r"$\omega$",
    # internal arc weight (all three internal arcs share this label)
    "g_internal": r"$g$",
    # before: exit arc weights, downstream boundary values, continuation
    "g_exit":     r"$g$",
    "J_b1":       r"$J_\lambda^*(b_1)$",
    "J_b2":       r"$J_\lambda^*(b_2)$",
    "to_xe":      r"$\cdots\, x_e$",
    # after: rerouted exit weights and the sink value
    "exit_om1":   r"$g+\lambda J_\lambda^*(b_1)$",
    "exit_om2":   r"$g+\lambda J_\lambda^*(b_2)$",
    "J_omega":    r"$J_\lambda^*(\omega)=0$",
    # the two overlaid states of the transformation
    "before":     r"before folding",
    "after":      r"after folding",
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
})

NODE = dict(node_size=1300, node_color="white", edgecolors="black", linewidths=1.6)
GRAY = "0.45"
GREEN = "#1b7837"


def draw_node_set(ax, G, pos, nodes, **kw):
    opts = dict(NODE)
    opts.update(kw)
    return nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, nodelist=nodes, **opts)


def draw_edges(ax, G, pos, edges, rad=0.0, style="solid", color="black", width=1.6):
    return nx.draw_networkx_edges(
        ax=ax, G=G, pos=pos, edgelist=edges,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", arrowsize=16, width=width,
        edge_color=color, style=style, node_size=1300,
        min_source_margin=0, min_target_margin=0,
    )


def scc_box(ax, x0, y0, x1, y1):
    box = plt.matplotlib.patches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor=GRAY, facecolor="none", linestyle=(0, (4, 3)),
    )
    ax.add_patch(box)
    ax.text((x0 + x1) / 2, y1 - 0.18, LABELS["scc"], color=GRAY,
            ha="center", va="top", fontsize=12)


def panel_merged(ax):
    """SCC drawn once, with the pre- and post-folding exit arcs overlaid."""
    G = nx.DiGraph()
    pos = {
        "x":  (0.0, 0.0),
        "v1": (1.05, 0.95),
        "v2": (1.05, -0.95),
        "b1": (2.55, 1.05),
        "b2": (2.55, -1.05),
        "om": (4.45, 0.0),
    }
    G.add_nodes_from(pos)
    internal = [("x", "v1"), ("v1", "v2"), ("v2", "x")]
    exits = [("v1", "b1"), ("v2", "b2")]
    toomega = [("v1", "om"), ("v2", "om")]
    G.add_edges_from(internal + exits + toomega)

    scc_box(ax, -0.55, -1.75, 1.65, 1.75)
    draw_node_set(ax, G, pos, ["x", "v1", "v2"])
    # boundary nodes: dashed outline, since folding removes them from the graph
    bnd = draw_node_set(ax, G, pos, ["b1", "b2"], edgecolors=GRAY, node_color="white")
    bnd.set_linestyle((0, (4, 3)))
    # omega: double circle = solid node + outer ring
    draw_node_set(ax, G, pos, ["om"], node_color="white", edgecolors=GREEN)
    draw_node_set(ax, G, pos, ["om"], node_size=2050, node_color="none",
                  edgecolors=GREEN, linewidths=1.4)

    draw_edges(ax, G, pos, internal, rad=0.16)
    draw_edges(ax, G, pos, exits, rad=0.0, color=GRAY, style=(0, (5, 3)))
    # bow each redirected arc toward the axis so that it clears b1 / b2
    draw_edges(ax, G, pos, [("v1", "om")], rad=0.16, color=GREEN)
    draw_edges(ax, G, pos, [("v2", "om")], rad=-0.16, color=GREEN)

    nx.draw_networkx_labels(ax=ax, G=G, pos=pos,
                            labels={"x": LABELS["x"], "v1": LABELS["v1"],
                                    "v2": LABELS["v2"], "b1": LABELS["b1"],
                                    "b2": LABELS["b2"], "om": LABELS["omega"]})

    # internal arc weights: untouched by the transformation
    ax.text(0.45, 0.62, LABELS["g_internal"], ha="center", va="center")
    ax.text(1.20, 0.0, LABELS["g_internal"], ha="center", va="center")
    ax.text(0.45, -0.62, LABELS["g_internal"], ha="center", va="center")

    # before: exit weights, boundary values, continuation toward x_e
    ax.text(1.95, 1.28, LABELS["g_exit"], ha="center", va="center", color=GRAY)
    ax.text(1.95, -1.28, LABELS["g_exit"], ha="center", va="center", color=GRAY)
    for yy in (1.05, -1.05):
        ax.annotate("", xy=(3.42, yy), xytext=(2.82, yy),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                    linestyle="dashed", lw=1.4))
    ax.text(2.55, 1.62, LABELS["J_b1"], ha="center", va="center", color=GRAY)
    ax.text(2.55, -1.62, LABELS["J_b2"], ha="center", va="center", color=GRAY)
    ax.text(3.52, 1.05, LABELS["to_xe"], ha="left", va="center", color=GRAY)
    ax.text(3.52, -1.05, LABELS["to_xe"], ha="left", va="center", color=GRAY)

    # after: rerouted exit weights and the sink value
    ax.text(3.30, 0.55, LABELS["exit_om1"], ha="center", va="center",
            color=GREEN, fontsize=11)
    ax.text(3.30, -0.55, LABELS["exit_om2"], ha="center", va="center",
            color=GREEN, fontsize=11)
    ax.text(4.45, -0.62, LABELS["J_omega"], ha="center", va="top", color=GREEN)

    # which overlay is which: a two-column strip under the drawing, so that the
    # two arc styles are named without hunting for free space inside it
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], color=GRAY, lw=1.6, linestyle=(0, (5, 3)),
                   label=LABELS["before"]),
            Line2D([0], [0], color=GREEN, lw=1.6, label=LABELS["after"]),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
        fontsize=12, handlelength=2.4, columnspacing=2.2,
    )

    ax.set_xlim(-0.70, 5.05)
    ax.set_ylim(-1.88, 1.88)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, ax = plt.subplots(figsize=(6.15, 4.02))
    panel_merged(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("figures/augment_networkx.png", dpi=200,
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig("figures/augment_networkx.pdf",
                bbox_inches="tight", pad_inches=0.02)
    print("saved figures/augment_networkx.png and .pdf")


if __name__ == "__main__":
    main()
