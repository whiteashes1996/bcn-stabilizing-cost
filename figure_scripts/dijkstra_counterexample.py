r"""Counterexample graph for Remark (failure of reverse Dijkstra under discounting).

lambda = 1/2. Arcs:  B->x_e (60),  B->A (1),  A->x_e (100).
Optimal: J*(B) = min{60, 1 + 0.5*100} = 51, attained through the *more
expensive* node A (J*(A)=100). A reverse Dijkstra sweep would fix B at 60 first.

Output: figures/dijkstra_counterexample.pdf  (vector, for LaTeX \includegraphics)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "dijkstra_counterexample.pdf")

pos = {"B": (0.0, 0.0), "A": (2.0, 1.15), "xe": (4.0, 0.0)}
labels = {"B": "$B$", "A": "$A$", "xe": "$x_e$"}

# (u, v, weight, on_optimal_path)
edges = [
    ("B", "A", 1, True),
    ("A", "xe", 100, True),
    ("B", "xe", 60, False),
]

G = nx.DiGraph()
G.add_nodes_from(pos)
for u, v, w, _opt in edges:
    G.add_edge(u, v, weight=w)

fig, ax = plt.subplots(figsize=(4.6, 2.3))

OPT = "#1a7f37"   # green: optimal route B->A->x_e
DIR = "#9aa0a6"   # gray: dominated direct arc B->x_e

nx.draw_networkx_nodes(
    G, pos, ax=ax, node_size=1500,
    node_color="white", edgecolors="black", linewidths=1.4,
)
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=15)

for u, v, w, opt in edges:
    color = OPT if opt else DIR
    rad = 0.0 if (u, v) != ("B", "xe") else 0.42  # bend the direct arc below
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], ax=ax,
        edge_color=color, width=2.0 if opt else 1.6,
        arrows=True, arrowsize=18, arrowstyle="-|>",
        node_size=1500, min_source_margin=16, min_target_margin=16,
        connectionstyle=f"arc3,rad={rad}",
    )

# edge weight labels
edge_label_pos = {
    ("B", "A"): (0.92, 0.72),
    ("A", "xe"): (3.18, 0.72),
    ("B", "xe"): (2.0, -0.82),
}
edge_label_color = {("B", "A"): OPT, ("A", "xe"): OPT, ("B", "xe"): "black"}
for (u, v, w, opt) in edges:
    x, y = edge_label_pos[(u, v)]
    ax.text(
        x, y, f"{w}", fontsize=12, ha="center", va="center",
        color=edge_label_color[(u, v)],
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
    )

ax.set_xlim(-0.8, 4.8)
ax.set_ylim(-1.35, 1.75)
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(OUT, bbox_inches="tight", transparent=True)
print("wrote", OUT)
