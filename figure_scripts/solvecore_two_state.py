r"""SolveCore on the two-state beneficial-cycle component.

lambda = 1/2, internal arcs cost 1/2, exits cost 2.
J*(x) = J*(y) = 3/2.

The stored arc depends on the set of states still available.
The path from x uses y -> exit after x has been removed.
The path from y uses y -> x while x is still available.
The two actions at y are not one feedback.

Output: figures/solvecore_two_state.pdf
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "solvecore_two_state.pdf")

OPT = "#1a7f37"
IDLE = "#b0b6be"
NODE = "white"

# Shared geometry: x left, y right, omega above.
pos = {"x": (0.0, 0.0), "y": (2.4, 0.0), "w": (1.2, 1.55)}
labels = {"x": r"$x$", "y": r"$y$", "w": r"$\omega$"}


def draw_panel(ax, active, title, note):
    G = nx.DiGraph()
    G.add_nodes_from(pos)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=900,
        node_color=NODE, edgecolors="black", linewidths=1.2,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=13)

    # (u, v, text, rad, key)
    arcs = [
        ("x", "y", r"$1/2$", 0.18, "xy"),
        ("y", "x", r"$1/2$", 0.18, "yx"),
        ("x", "w", r"$2$", -0.05, "xw"),
        ("y", "w", r"$2$", 0.05, "yw"),
    ]
    label_at = {
        "xy": (1.20, 0.28),
        "yx": (1.20, -0.38),
        "xw": (0.42, 0.95),
        "yw": (1.98, 0.95),
    }
    for u, v, text, rad, key in arcs:
        on = key in active
        color = OPT if on else IDLE
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=color, width=2.2 if on else 1.2,
            arrows=True, arrowsize=14, arrowstyle="-|>",
            node_size=900, min_source_margin=12, min_target_margin=12,
            connectionstyle=f"arc3,rad={rad}",
        )
        lx, ly = label_at[key]
        ax.text(
            lx, ly, text, fontsize=9, ha="center", va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"),
        )

    ax.set_xlim(-0.85, 3.25)
    ax.set_ylim(-0.95, 2.15)
    ax.set_title(title, fontsize=11, pad=2)
    ax.text(1.2, -0.82, note, fontsize=8.5, ha="center", va="top", color="black")
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
draw_panel(
    axes[0],
    active=set(),
    title="(a) Component",
    note=r"$\lambda=1/2$,  $J^*=3/2$",
)
draw_panel(
    axes[1],
    active={"xy", "yw"},
    title="(b) Path from $x$",
    note=r"at $y$: set $\{y\}$, exit",
)
draw_panel(
    axes[2],
    active={"yx", "xw"},
    title="(c) Path from $y$",
    note=r"at $y$: set $\{x,y\}$, go to $x$",
)
fig.tight_layout(pad=0.35)
fig.savefig(OUT, bbox_inches="tight", transparent=True)
print("wrote", OUT)
