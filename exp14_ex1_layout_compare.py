"""Generate Example~1 STG figures under multiple NetworkX / Graphviz layouts.

Run:
    python exp14_ex1_layout_compare.py

Outputs numbered previews under figures/ex1_stg_layouts/ for side-by-side review.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx

from core_solver import condense, reverse_reachable
from exp14_ex1_n4 import (
    EXPECTED_IMPROPER,
    FIGURES,
    LAM,
    UE,
    XE,
    _condensation_layers,
    _draw_stg,
    _normalize_lr,
    _scc_internal_offsets,
    baseline_vi,
    build,
    build_topo_pos,
    greedy_mu,
    layout_11_multipartite_states,
    restricted_adjacency,
)
from core_solver import master

OUT_DIR = os.path.join(FIGURES, "ex1_stg_layouts")
INTRA = 1.05


def _condensation_digraph(scc) -> nx.DiGraph:
    H = nx.DiGraph()
    for cid in range(len(scc.components)):
        H.add_node(cid, label=",".join(str(x) for x in sorted(scc.components[cid])))
    for a, b in scc.dag_edges:
        H.add_edge(a, b)
    return H


def _expand_from_super(
    G, scc, R, super_pos: Dict[int, Tuple[float, float]], dead_shift: bool = True,
) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    for cid, (sx, sy) in super_pos.items():
        comp = scc.components[cid]
        for nd, dy in _scc_internal_offsets(comp, G, INTRA).items():
            pos[nd] = (sx, sy + dy)
    dead = sorted(set(G.states) - R)
    if dead and dead_shift and pos:
        min_x = min(pos[n][0] for n in R)
        dead_y = min(pos[n][1] for n in R) - 1.6
        for i, nd in enumerate(dead):
            pos[nd] = (min_x - 1.0 + i * 0.45, dead_y - i * 0.35)
    return pos


def _layer_dict(scc) -> Dict[int, int]:
    return _condensation_layers(scc)


# ---------------------------------------------------------------------------
# Layout 01: current custom (condensation longest-path + lane bands)
# ---------------------------------------------------------------------------


def layout_01_custom(G, scc, R, trapped):
    pos, layer, _ = build_topo_pos(G, scc, R, trapped)
    return pos, layer, "01 custom (condensation layers + lane bands)"


# ---------------------------------------------------------------------------
# Layout 02: multipartite_layout on condensation DAG
# ---------------------------------------------------------------------------


def layout_02_multipartite(G, scc, R, trapped):
    layer = _layer_dict(scc)
    H = _condensation_digraph(scc)
    for cid, lev in layer.items():
        H.nodes[cid]["subset"] = lev
    super_pos = nx.multipartite_layout(
        H, subset_key="subset", align="vertical", scale=2.0,
    )
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), layer, "02 multipartite_layout (condensation DAG)"


# ---------------------------------------------------------------------------
# Layout 03: topological_generations — dot-like layered DAG (pure NetworkX)
# ---------------------------------------------------------------------------


def layout_03_topological_generations(G, scc, R, trapped):
    H = _condensation_digraph(scc)
    super_pos: Dict[int, Tuple[float, float]] = {}
    layer: Dict[int, int] = {}
    for gen_i, gen in enumerate(nx.topological_generations(H)):
        cids = sorted(gen)
        for j, cid in enumerate(cids):
            layer[cid] = gen_i
            super_pos[cid] = (gen_i * 2.2, (j - (len(cids) - 1) / 2.0) * 2.0)
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), layer, "03 topological_generations (dot-like, NetworkX)"


# ---------------------------------------------------------------------------
# Layout 04: fruchterman_reingold on condensation (neato-like force layout)
# ---------------------------------------------------------------------------


def layout_04_spring_fr(G, scc, R, trapped):
    H = _condensation_digraph(scc)
    super_pos = nx.fruchterman_reingold_layout(H, seed=14, k=2.5 / max(len(H), 1))
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), _layer_dict(scc), "04 fruchterman_reingold (neato-like, condensation)"


# ---------------------------------------------------------------------------
# Layout 05: spring_layout on condensation super-graph
# ---------------------------------------------------------------------------


def layout_05_spring_condensation(G, scc, R, trapped):
    H = _condensation_digraph(scc)
    super_pos = nx.spring_layout(H, seed=14, k=2.5 / max(len(H), 1))
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), _layer_dict(scc), "05 spring_layout (condensation super-graph)"


# ---------------------------------------------------------------------------
# Layout 06: kamada_kawai on condensation super-graph
# ---------------------------------------------------------------------------


def layout_06_kamada_condensation(G, scc, R, trapped):
    H = _condensation_digraph(scc)
    super_pos = nx.kamada_kawai_layout(H)
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), _layer_dict(scc), "06 kamada_kawai_layout (condensation)"


# ---------------------------------------------------------------------------
# Layout 07: spectral_layout on condensation super-graph
# ---------------------------------------------------------------------------


def layout_07_spectral_condensation(G, scc, R, trapped):
    H = _condensation_digraph(scc)
    UG = H.to_undirected()
    super_pos = nx.spectral_layout(UG)
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), _layer_dict(scc), "07 spectral_layout (condensation, undirected)"


# ---------------------------------------------------------------------------
# Layout 08: shell_layout — one shell per condensation layer
# ---------------------------------------------------------------------------


def layout_08_shell_condensation(G, scc, R, trapped):
    layer = _layer_dict(scc)
    n_layers = max(layer.values()) + 1
    shells = [[] for _ in range(n_layers)]
    for cid, lev in layer.items():
        shells[lev].append(cid)
    H = _condensation_digraph(scc)
    super_pos = nx.shell_layout(H, nlist=shells)
    pos = _expand_from_super(G, scc, R, super_pos)
    return _normalize_lr(pos), layer, "08 shell_layout (condensation layer = shell)"


# ---------------------------------------------------------------------------
# Layout 09: spring_layout on full 16-state graph
# ---------------------------------------------------------------------------


def _full_state_digraph(G) -> nx.DiGraph:
    D = nx.DiGraph()
    for x, _u, y, _w in G.arcs:
        if x != y:
            D.add_edge(x, y)
    return D


def layout_09_spring_full(G, scc, R, trapped):
    D = _full_state_digraph(G)
    pos = nx.spring_layout(D, seed=14, k=1.8 / max(len(D), 1))
    return _normalize_lr(pos), _layer_dict(scc), "09 spring_layout (full state graph)"


# ---------------------------------------------------------------------------
# Layout 10: kamada_kawai on full state graph
# ---------------------------------------------------------------------------


def layout_10_kamada_full(G, scc, R, trapped):
    D = _full_state_digraph(G)
    pos = nx.kamada_kawai_layout(D)
    return _normalize_lr(pos), _layer_dict(scc), "10 kamada_kawai_layout (full state graph)"


# Layout 11 is defined in exp14_ex1_n4.py (official figure_stg uses it).


LAYOUTS: List[Callable] = [
    layout_01_custom,
    layout_02_multipartite,
    layout_03_topological_generations,
    layout_04_spring_fr,
    layout_05_spring_condensation,
    layout_06_kamada_condensation,
    layout_07_spectral_condensation,
    layout_08_shell_condensation,
    layout_09_spring_full,
    layout_10_kamada_full,
    layout_11_multipartite_states,
]


def render_all() -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    G = build()
    R = reverse_reachable(G, XE)
    scc = condense(G, R, XE)
    adjR = restricted_adjacency(G, R)
    mu_vi = greedy_mu(baseline_vi(adjR, XE, LAM, 64)[-1], adjR, XE, LAM)
    _, mu_master = master(G, XE, LAM, ue=UE)
    trapped = EXPECTED_IMPROPER

    written: List[str] = []
    index_lines = ["# Example 1 STG layout comparison\n"]

    for fn in LAYOUTS:
        try:
            pos, layer, title = fn(G, scc, R, trapped)
        except Exception as exc:
            slug = fn.__name__.replace("layout_", "")
            index_lines.append(f"- **{slug}**: FAILED ({exc})\n")
            print(f"SKIP {fn.__name__}: {exc}")
            continue

        slug = fn.__name__.replace("layout_", "")
        fig, ax = plt.subplots(figsize=(13.0, 6.4))
        _draw_stg(
            ax, G, pos, R, scc, layer, {}, mu_vi, mu_master, trapped,
            layout_title=title,
        )
        fig.tight_layout(pad=0.25)
        pdf = os.path.join(OUT_DIR, f"ex1_stg_{slug}.pdf")
        png = pdf.replace(".pdf", ".png")
        fig.savefig(pdf, bbox_inches="tight", transparent=True)
        fig.savefig(png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        written.append(pdf)
        index_lines.append(f"- [{title}](ex1_stg_{slug}.pdf)\n")
        print(f"WROTE {pdf}")

    index_path = os.path.join(OUT_DIR, "README.md")
    with open(index_path, "w") as fh:
        fh.writelines(index_lines)
    print(f"\nIndex: {index_path}")
    print(f"Generated {len(written)} / {len(LAYOUTS)} layouts.")
    return written


if __name__ == "__main__":
    render_all()
