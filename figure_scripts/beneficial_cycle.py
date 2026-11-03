r"""Beneficial-cycle characterization (Lemma): why a discounted greedy policy
gets trapped in a non-target cycle.

At a state y != x_e the stationary policy faces two options:
  (i)  a *proper* continuation reaching x_e, with discounted cost-to-go
       J_λ^tail(y);
  (ii) a cycle C of length l that returns to y without visiting x_e, whose
       infinite repetition has discounted value g_loop(C)/(1 - lambda^l).

Looping is strictly cheaper (the optimal stationary choice is improper) iff
       g_loop(C) < (1 - lambda^l) * J_λ^tail(y).

Output: figures/beneficial_cycle.pdf  (vector, for LaTeX \includegraphics)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(__file__), "beneficial_cycle.pdf")

LOOP = "#d1495b"   # red: the non-target cycle (the trap)
PROP = "#1a7f37"   # green: proper continuation to x_e
DOT = "#9aa0a6"    # gray: "many steps" continuation

pos = {
    "y":  (0.0, 0.0),
    "y1": (-1.05, 1.45),
    "y2": (-2.55, 0.55),
    "p1": (1.75, 0.55),
    "xe": (4.35, 0.0),
}
labels = {
    "y": "$y$", "y1": "$y_1$", "y2": "$y_2$",
    "p1": "$\\,$", "xe": "$x_e$",
}

# cycle edges (red) + proper-path edges (green)
loop_edges = [("y", "y1"), ("y1", "y2"), ("y2", "y")]
prop_edges = [("y", "p1")]

G = nx.DiGraph()
G.add_nodes_from(pos)
G.add_edges_from(loop_edges + prop_edges + [("p1", "xe")])

fig, ax = plt.subplots(figsize=(5.6, 3.0))

# nodes: x_e highlighted as the target
node_colors = {n: "white" for n in pos}
node_colors["xe"] = "#eaf5ec"
node_edge = {n: "black" for n in pos}
node_edge["xe"] = PROP
nx.draw_networkx_nodes(
    G, pos, ax=ax, node_size=1450,
    node_color=[node_colors[n] for n in G.nodes()],
    edgecolors=[node_edge[n] for n in G.nodes()], linewidths=1.6,
)
# p1 is a small via-node on the proper path
nx.draw_networkx_nodes(
    G, pos, ax=ax, nodelist=["p1"], node_size=320,
    node_color="white", edgecolors=PROP, linewidths=1.4,
)
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=16)

# --- cycle C (red) ---
nx.draw_networkx_edges(
    G, pos, edgelist=loop_edges, ax=ax, edge_color=LOOP, width=2.2,
    arrows=True, arrowsize=18, arrowstyle="-|>",
    node_size=1450, min_source_margin=16, min_target_margin=16,
    connectionstyle="arc3,rad=0.12",
)
# --- proper continuation (green): y -> p1 solid, p1 -> x_e dashed ("...") ---
nx.draw_networkx_edges(
    G, pos, edgelist=[("y", "p1")], ax=ax, edge_color=PROP, width=2.2,
    arrows=True, arrowsize=18, arrowstyle="-|>",
    node_size=1450, min_source_margin=16, min_target_margin=10,
)
nx.draw_networkx_edges(
    G, pos, edgelist=[("p1", "xe")], ax=ax, edge_color=PROP, width=2.2,
    style="dashed", arrows=True, arrowsize=18, arrowstyle="-|>",
    node_size=1450, min_source_margin=10, min_target_margin=16,
)
ax.text(3.05, 0.46, r"$\cdots$", fontsize=15, ha="center", va="center",
        color=PROP)

# --- per-step cycle costs ---
cycle_cost = {
    ("y", "y1"): (0.25, 0.80, r"$g(y_0,u_0)$"),
    ("y1", "y2"): (-1.55, 0.78, r"$g(y_1,u_1)$"),
    ("y2", "y"): (-1.30, -0.18, r"$g(y_2,u_2)$"),
}
for _e, (x, yy, txt) in cycle_cost.items():
    ax.text(x, yy, txt, fontsize=10.5, ha="center", va="center", color=LOOP,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

# --- branch annotations ---
ax.text(-2.55, 2.15,
        r"cycle $C$ ($\ell=3$), repeat forever:" "\n"
        r"value $=\dfrac{g_{\mathrm{loop}}(C)}{1-\lambda^{\ell}}$",
        fontsize=11, ha="left", va="bottom", color=LOOP)
ax.text(2.6, 1.02,
        r"proper continuation:  $J_{\lambda}^{\mathrm{tail}}(y)$",
        fontsize=11, ha="center", va="bottom", color=PROP)

# --- the trapping condition ---
ax.text(1.25, -1.32,
        r"improper optimum at $y$  $\Longleftrightarrow$  "
        r"$g_{\mathrm{loop}}(C) < (1-\lambda^{\ell})\,J_{\lambda}^{\mathrm{tail}}(y)$",
        fontsize=12, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#bbbbbb"))

ax.set_xlim(-3.3, 5.0)
ax.set_ylim(-1.85, 3.05)
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(OUT, bbox_inches="tight", transparent=True)
print("wrote", OUT)
