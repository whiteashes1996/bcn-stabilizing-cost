"""E9: a beneficial cycle on a real network (ara operon, N=512, M=16).

Under the natural Hamming cost the ara operon is easy (E3): no component
carries a beneficial cycle and the unconstrained value V_lambda* already equals
the stabilization cost J_lambda*.  Here the cost is made side-effect aware: one
non-target control fixed point x_d is a quiescent gene-activity profile that
sustains itself and costs almost nothing to maintain, so its self-loop is
charged eps instead of its Hamming distance to the target.

That single change makes the self-loop at x_d a beneficial cycle, and the two
regimes below show that what governs tractability is the size of the
beneficial-cycle component, not the size N of the state space:

  (i)  with all sixteen input patterns available the component that carries the
       cycle has 108 states, and the exact subset solver is out of reach - this
       is the NP-complete kernel of Theorem nphard;
  (ii) restricting the actuation to the two realizable input patterns (an
       instance of the constraint C_u of Problem 2) leaves the state space
       untouched at N = 512 but shrinks that component to 20 states, and
       SolveCore settles it exactly in a fraction of a second.

In both regimes value iteration returns an improper policy that parks six
states on the decoy for ever, while Algorithm master stabilizes all 512.
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import build_ara_dynamics, hamming_cost_factory  # noqa: E402
from core_solver import (  # noqa: E402
    _internal_arcs,
    augment_component,
    build_graph,
    condense,
    detect_beneficial_cycle_v2,
    local_lc_paper,
    master,
    reverse_reachable,
    simulate_closed_loop,
    sweep_T,
    terminal_to_augmented,
    to_exact_component,
    vi_policy,
    vi_unconstrained,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")

XE, UE, LAM = 512, 11, 0.5
XD, UD = 9, 2          # decoy control fixed point and the input that holds it
EPS = 0.05             # maintenance cost of the quiescent profile
CU_REALIZABLE = [2, 11]

TRAP = "#d1495b"
GOOD = "#1a7f37"
FAINT = "#c8ccd0"


def load():
    n, m, N, M, L, f = build_ara_dynamics()
    base = hamming_cost_factory(XE, UE, n)

    def g_decoy(x: int, u: int) -> float:
        return EPS if (x == XD and u == UD) else base(x, u)

    return n, m, N, M, L, f, base, g_decoy


def graph_for(f, g, N, M, Cu: Optional[List[int]]):
    cu = None if Cu is None else (lambda x, C=tuple(Cu): list(C))
    return build_graph(f, g, list(range(1, N + 1)), list(range(1, M + 1)), Cu=cu, xe=XE)


def component_report(G, R, lam: float) -> Tuple[List[Tuple[int, Set[int]]], List[int]]:
    """Per-component beneficial-cycle test in reverse topological order."""
    scc = condense(G, R, XE)
    carriers: List[Tuple[int, Set[int]]] = []
    J_known: Dict[int, float] = {XE: 0.0}
    for cid in scc.reverse_topo:
        S = scc.components[cid]
        if XE in S:
            if len(S) == 1:
                continue
            S2, internal, e = terminal_to_augmented(S, _internal_arcs(G, S), XE, lam)
        else:
            S2, (internal, e) = S, augment_component(G, S, J_known, lam)
        if not S2:
            continue
        flag, trapped, _V = detect_beneficial_cycle_v2(S2, internal, e, lam)
        if flag:
            carriers.append((len(S2), trapped))
        for x in S2:
            J_known.setdefault(x, 0.0)
    sizes = sorted((len(c) for c in scc.components), reverse=True)
    return carriers, sizes


def carrier_component(G, R, lam: float):
    """The augmented component that carries the beneficial cycle."""
    scc = condense(G, R, XE)
    J_known: Dict[int, float] = {XE: 0.0}
    for cid in scc.reverse_topo:
        S = scc.components[cid]
        if XE in S:
            if len(S) == 1:
                continue
            S2, internal, e = terminal_to_augmented(
                S, _internal_arcs(G, S), XE, lam)
        else:
            S2, (internal, e) = S, augment_component(G, S, J_known, lam)
        if not S2:
            continue
        flag, _trapped, _V = detect_beneficial_cycle_v2(S2, internal, e, lam)
        if flag:
            return S2, internal, e
        for x in S2:
            J_known.setdefault(x, 0.0)
    raise AssertionError("no beneficial-cycle component")


def detection_precision(f, g, N, M, lam: float) -> dict:
    """Why the carrier is decided by tight arcs and not by a sweep count.

    A fixed-point test would have to see a strict decrease at sweep |S|-1.  On
    this component that decrease is of order lambda^{|S|} times the gap between
    the exit weight and the cycle value, so double precision loses it long
    before sweep 106 and reports a fixed point that has not been reached, while
    the tight-arc criterion (P) of Lemma detect settles it exactly.
    """
    G = graph_for(f, g, N, M, None)
    R = reverse_reachable(G, XE)
    S, internal, e = carrier_component(G, R, lam)

    t0 = time.perf_counter()
    _J, _pol, flag_f, sweeps_f = local_lc_paper(S, internal, e, lam, tol=1e-12)
    t_float = time.perf_counter() - t0

    internal_q, e_q, lam_q = to_exact_component(internal, e, lam)
    t0 = time.perf_counter()
    J_q, _pol, flag_q, sweeps_q = local_lc_paper(
        S, internal_q, e_q, lam_q, tol=0)
    t_exact = time.perf_counter() - t0

    # Criterion (P): no V*-greedy path to omega.  It reads off the converged
    # value instead of watching a vanishing decrement, so double precision is
    # enough even at this size.
    t0 = time.perf_counter()
    flag_p, trapped, _V = detect_beneficial_cycle_v2(S, internal, e, lam)
    t_p = time.perf_counter() - t0

    # The decrease the verification sweep has to see, in exact arithmetic.
    J_next, _ = sweep_T(S, internal_q, e_q, lam_q, J_q)
    drops = [J_q[x] - J_next[x] for x in S if J_next[x] < J_q[x]]
    decrement = max(drops) if drops else Fraction(0)

    assert not flag_f, "float sweeps unexpectedly caught the cycle"
    assert flag_q, "exact sweeps missed the cycle"
    assert flag_p, "criterion (P) missed the cycle"
    assert sweeps_q == len(S), (sweeps_q, len(S))

    return {
        "size": len(S),
        "float_verdict": flag_f,
        "float_sweeps": sweeps_f,
        "float_time": t_float,
        "exact_verdict": flag_q,
        "exact_sweeps": sweeps_q,
        "exact_time": t_exact,
        "p_verdict": flag_p,
        "p_trapped": len(trapped),
        "p_time": t_p,
        "decrement": float(decrement),
    }


def improper(f, mu, R) -> List[int]:
    return [
        x for x in sorted(R)
        if x != XE and not simulate_closed_loop(f, mu, x, XE, max_steps=4 * len(R))[0]
    ]


def analyse(f, g, N, M, Cu, lam: float, want_master: bool = True) -> dict:
    G = graph_for(f, g, N, M, Cu)
    R = reverse_reachable(G, XE)
    carriers, sizes = component_report(G, R, lam)
    V = vi_unconstrained(G, lam, R, XE)
    mu_vi = vi_policy(G, V, lam, R)
    out = {
        "G": G, "R": R, "V": V, "mu_vi": mu_vi,
        "carriers": carriers, "sizes": sizes,
        "vi_improper": improper(f, mu_vi, R),
    }
    if want_master:
        t0 = time.time()
        try:
            J, mu = master(G, XE, lam, ue=UE)
            out["J"] = J
            out["mu"] = mu
            out["t_master"] = time.time() - t0
            out["master_improper"] = improper(f, mu, R)
            out["gap"] = [x for x in sorted(R) if J[x] > V[x] + 1e-7]
        except RuntimeError as exc:
            out["refused"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_local_stg(res: dict, f) -> str:
    """Closed loops around the decoy: value iteration versus Algorithm master."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    trapped = res["vi_improper"]

    def trajectory(mu, x0, limit=12):
        seq, cur, seen = [x0], x0, {x0}
        for _ in range(limit):
            if cur == XE or cur not in mu:
                break
            cur = f(cur, mu[cur][0])
            seq.append(cur)
            if cur in seen:
                break
            seen.add(cur)
        return seq

    vi_edges: Set[Tuple[int, int]] = set()
    mu_edges: Set[Tuple[int, int]] = set()
    nodes: Set[int] = {XE}
    for x in trapped:
        for seq, bucket in ((trajectory(res["mu_vi"], x), vi_edges),
                            (trajectory(res["mu"], x), mu_edges)):
            nodes |= set(seq)
            bucket |= set(zip(seq, seq[1:]))

    # The picture has a rigid shape - a fan of states feeding the decoy on the
    # left and the corridor to x_e on the right - so it is laid out by hand.
    feeders = [x for x in trapped if x != XD]
    chain = trajectory(res["mu"], XD)[1:]
    pos = {XD: (1.35, 1.25)}
    span = 0.75 * (len(feeders) - 1)
    for k, x in enumerate(feeders):
        pos[x] = (0.0, span / 2 - 0.75 * k)
    for k, x in enumerate(chain):
        pos[x] = (2.7 + 1.05 * k, 0.0)

    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(vi_edges | mu_edges)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.5))
    for ax, edges_, colour, title in (
        (axes[0], vi_edges, TRAP,
         r"(a) greedy policy from $V_\lambda^*$: all six states end on the decoy"),
        (axes[1], mu_edges, GOOD,
         r"(b) $\mu^*$ from Algorithm 1: all six states reach $x_e$"),
    ):
        fill, ecol = [], []
        for nd in H.nodes():
            if nd == XE:
                fill.append("#eaf5ec")
                ecol.append(GOOD)
            elif nd == XD:
                fill.append("#f8d7da")
                ecol.append(TRAP)
            elif nd in trapped:
                fill.append("#fdf3f4")
                ecol.append(TRAP)
            else:
                fill.append("white")
                ecol.append("#555555")
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=720, node_color=fill,
                               edgecolors=ecol, linewidths=1.5)
        nx.draw_networkx_labels(
            H, pos, ax=ax, font_size=7.5,
            labels={i: rf"$\delta_{{512}}^{{{i}}}$" for i in H.nodes()},
        )
        rest = sorted((vi_edges | mu_edges) - edges_ - {(XD, XD)})
        nx.draw_networkx_edges(
            H, pos, ax=ax, edgelist=rest, edge_color=FAINT, width=0.9,
            arrows=True, arrowsize=9, arrowstyle="-|>", node_size=720,
            min_source_margin=10, min_target_margin=10,
        )
        nx.draw_networkx_edges(
            H, pos, ax=ax, edgelist=sorted(edges_ - {(XD, XD)}), edge_color=colour,
            width=1.9, arrows=True, arrowsize=13, arrowstyle="-|>",
            node_size=720, min_source_margin=10, min_target_margin=10,
        )
        loop_colour = colour if (XD, XD) in edges_ else FAINT
        ax.annotate(
            "", xy=(pos[XD][0] - 0.08, pos[XD][1] + 0.28),
            xytext=(pos[XD][0] + 0.12, pos[XD][1] + 0.26),
            arrowprops=dict(arrowstyle="-|>", color=loop_colour, lw=1.9,
                            connectionstyle="arc3,rad=3.1",
                            shrinkA=0.0, shrinkB=0.0),
        )
        if (XD, XD) in edges_:
            ax.text(pos[XD][0] + 0.30, pos[XD][1] + 0.52,
                    rf"cost $\varepsilon={EPS}$", fontsize=9, color=TRAP)
        ax.text(pos[XD][0] + 0.42, pos[XD][1] - 0.05, r"$x_d$", fontsize=10,
                color=TRAP, va="center")
        ax.text(pos[XE][0], pos[XE][1] - 0.50, r"$x_e$", fontsize=10,
                color=GOOD, ha="center")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-0.55, pos[XE][0] + 0.6)
        ax.set_ylim(-span / 2 - 0.6, 2.35)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(FIGURES, "ara_decoy_stg.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def figure_lambda(rows: List[Tuple[float, int, int, float, float]],
                  lam_pred: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = [r[0] for r in rows]
    gaps = [r[1] for r in rows]
    Vd = [r[3] for r in rows]
    Jd = [r[4] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.1))
    ax1.plot(lams, Jd, color=GOOD, lw=2.0, label=r"$J_\lambda^*(x_d)$")
    ax1.plot(lams, Vd, color=TRAP, lw=2.0, ls="--", label=r"$V_\lambda^*(x_d)$")
    ax1.axvline(lam_pred, color="#555555", lw=1.0, ls=":")
    ax1.set_xlabel(r"discount factor $\lambda$")
    ax1.set_ylabel("discounted cost at $x_d$")
    ax1.legend(fontsize=9, frameon=False, loc="upper left")
    ax1.set_xscale("logit")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.step(lams, gaps, where="post", color=TRAP, lw=2.0)
    ax2.axvline(lam_pred, color="#555555", lw=1.0, ls=":")
    ax2.annotate(rf"$\varepsilon/(1-\lambda)=J_\lambda^*(x_d)$"
                 "\n"
                 rf"at $\lambda\approx{lam_pred:.4f}$",
                 xy=(lam_pred, max(gaps) * 0.55), xytext=(0.75, max(gaps) * 0.62),
                 fontsize=9, ha="center", color="#333333")
    ax2.set_xlabel(r"discount factor $\lambda$")
    ax2.set_ylabel(r"states with $V_\lambda^*<J_\lambda^*$")
    ax2.set_xscale("logit")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(FIGURES, "ara_decoy_lambda.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    n, m, N, M, L, f, base_g, g_decoy = load()

    easy = analyse(f, base_g, N, M, None, LAM)
    full = analyse(f, g_decoy, N, M, None, LAM)
    restricted = analyse(f, g_decoy, N, M, CU_REALIZABLE, LAM)

    # --- checks -----------------------------------------------------------
    assert easy["carriers"] == [], easy["carriers"]
    assert easy["gap"] == [] and easy["master_improper"] == []
    assert easy["sizes"][0] == 108 and len(easy["sizes"]) == 405

    assert len(full["carriers"]) == 1 and full["carriers"][0][0] == 107
    assert "refused" in full and "107" in full["refused"]

    prec = detection_precision(f, g_decoy, N, M, LAM)

    assert len(restricted["carriers"]) == 1 and restricted["carriers"][0][0] == 19
    assert restricted["sizes"][0] == 20 and len(restricted["R"]) == 512
    assert restricted["master_improper"] == []
    assert restricted["vi_improper"] == full["vi_improper"]
    assert restricted["gap"] == restricted["vi_improper"]
    assert abs(restricted["V"][XD] - EPS / (1 - LAM)) < 1e-6, restricted["V"][XD]

    Jd = restricted["J"][XD]

    # The self-loop is beneficial iff eps/(1-lambda) < J_lambda*(x_d), and the
    # right-hand side itself grows with lambda, so the crossing is located by
    # bisection rather than from the value at a single lambda.
    def decoy_beneficial(lam: float) -> bool:
        r = analyse(f, g_decoy, N, M, CU_REALIZABLE, lam, want_master=True)
        return r["J"][XD] > r["V"][XD] + 1e-9

    lo, hi = 0.5, 0.9999
    assert decoy_beneficial(lo) and not decoy_beneficial(hi)
    for _ in range(24):
        mid = (lo + hi) / 2
        if decoy_beneficial(mid):
            lo = mid
        else:
            hi = mid
    lam_pred = (lo + hi) / 2

    # --- lambda sweep on the tractable regime -----------------------------
    rows: List[Tuple[float, int, int, float, float]] = []
    for lam in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.9956,
                0.997, 0.9975, 0.998, 0.999, 0.9995]:
        r = analyse(f, g_decoy, N, M, CU_REALIZABLE, lam)
        rows.append((lam, len(r["gap"]), len(r["vi_improper"]),
                     r["V"][XD], r["J"][XD]))
        assert r["master_improper"] == [], (lam, r["master_improper"])
    assert rows[0][1] > 0 and rows[-1][1] == 0, rows

    fig1 = figure_local_stg(restricted, f)
    fig2 = figure_lambda(rows, lam_pred)

    lines = [
        f"E9: ara operon with a side-effect-aware cost (N={N}, M={M},"
        f" xe=delta_512^{XE}, ue=delta_16^{UE}, lambda={LAM})",
        f"  decoy: x_d = delta_512^{XD}, held by u = delta_16^{UD},"
        f" maintenance cost eps = {EPS}",
        "",
        "  (0) natural Hamming cost, all inputs available",
        f"      |R| = {len(easy['R'])}/{N}, components = {len(easy['sizes'])},"
        f" max = {easy['sizes'][0]}",
        f"      beneficial-cycle components = {len(easy['carriers'])};"
        f" V != J states = {len(easy['gap'])};"
        f" VI improper = {len(easy['vi_improper'])}",
        "",
        "  (i) side-effect-aware cost, all inputs available",
        f"      components = {len(full['sizes'])}, max = {full['sizes'][0]}",
        f"      beneficial-cycle components = "
        f"{[(sz, len(tr)) for sz, tr in full['carriers']]} (size, trapped)",
        f"      VI improper states = {len(full['vi_improper'])}:"
        f" {full['vi_improper']}",
        f"      exact SolveCore: {full['refused']}",
        f"      detection on that component (|S|={prec['size']}):",
        f"        Lemma detect, tight arcs in double precision:"
        f" verdict={prec['p_verdict']},"
        f" {prec['p_trapped']} trapped states, {prec['p_time']:.3f}s",
        f"        sweep count, exact rational: {prec['exact_sweeps']} sweeps,"
        f" verdict={prec['exact_verdict']}, {prec['exact_time']:.3f}s;"
        f" the decrease at sweep |S|-1 is {prec['decrement']:.3e}",
        f"        sweep count, double precision: reports a fixed point at sweep"
        f" {prec['float_sweeps']}, verdict={prec['float_verdict']},"
        f" {prec['float_time']:.3f}s",
        "",
        "  (ii) side-effect-aware cost, actuation restricted to"
        f" C_u = {{delta_16^{CU_REALIZABLE[0]}, delta_16^{CU_REALIZABLE[1]}}}",
        f"      |R| = {len(restricted['R'])}/{N},"
        f" components = {len(restricted['sizes'])},"
        f" max = {restricted['sizes'][0]}",
        f"      beneficial-cycle components = "
        f"{[(sz, sorted(tr)) for sz, tr in restricted['carriers']]}",
        f"      VI improper states = {len(restricted['vi_improper'])}:"
        f" {restricted['vi_improper']}",
        f"      master improper states = {len(restricted['master_improper'])},"
        f" runtime = {restricted['t_master']:.3f}s",
        f"      V*(x_d) = {restricted['V'][XD]:.4f} = eps/(1-lambda);"
        f"  J*(x_d) = {Jd:.4f}",
        f"      V != J states = {len(restricted['gap'])}: {restricted['gap']}",
        "",
        "  state      V_lambda*   J_lambda*   status",
    ]
    for x in restricted["gap"]:
        lines.append(
            f"    d_512^{x:<4d} {restricted['V'][x]:9.4f} {restricted['J'][x]:9.4f}"
            "   VI parks it on the decoy"
        )
    lines += [
        "",
        f"  vanishing discount: eps/(1-lambda) = J_lambda*(x_d) at"
        f" lambda = {lam_pred:.6f}; above it the self-loop is no longer"
        " beneficial and V* = J* again",
        "  lambda   V!=J   VI improper   V*(x_d)    J*(x_d)",
    ]
    for lam, ngap, nbad, v, j in rows:
        lines.append(f"   {lam:<7.4f} {ngap:<6d} {nbad:<13d} {v:9.4f} {j:9.4f}")
    lines += [
        "",
        f"  figures: {os.path.basename(fig1)}, {os.path.basename(fig2)}",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp9_ara_decoy.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {
        "bc_full": full["carriers"][0][0],
        "bc_restricted": restricted["carriers"][0][0],
        "vi_improper": len(restricted["vi_improper"]),
        "master_improper": len(restricted["master_improper"]),
        "t_master": restricted["t_master"],
        "lambda_pred": lam_pred,
    }


if __name__ == "__main__":
    run()
