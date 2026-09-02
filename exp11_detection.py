"""E11: how the beneficial-cycle predicate of Algorithm 1 is decided.

Proposition gap-W states three equivalent characterizations of a
beneficial-cycle-free component, and Lemma detect turns one of them -- the
tight-arc reachability criterion (P) -- into the test on line 7 of
Algorithm 1.  This experiment checks that the implementations of all four
really do coincide:

  (D) the definition itself: enumerate every simple non-target cycle and test
      g_loop(C) < (1 - lambda^l) J_tail(y) at every vertex of the cycle;
  (W) V_lambda* < J_lambda* somewhere on S;
  (P) no V_lambda*-greedy policy reaches the sink from some state;
  (T) a verification sweep of the operator T strictly lowers J_{|S|-1}.

(D), (W) and (P) are evaluated in exact rational arithmetic, with J_lambda*
from the subset solver and V_lambda* from exact policy iteration, so the
comparison is free of floating-point ambiguity.

Part B is why Algorithm 1 decides the predicate through (P) rather than through
the sweep count (T).  The decrease that (T) looks for after k sweeps is of
order lambda^k times the initial spread of the labels, so on a component
carrying a cheap cycle it drops below machine precision once |S| exceeds
roughly log(u)/log(lambda), u the unit roundoff, and the float version of (T)
then reports a fixed point that has not been reached.  The exact-arithmetic
version of (T), and the criteria (W) and (P), are unaffected: they read off the
converged V_lambda* rather than watching a vanishing decrement.
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (  # noqa: E402
    INF,
    brute_force_simple_paths,
    detect_beneficial_cycle_v2,
    enumerate_beneficial_cycles,
    greedy_action_set,
    local_lc_paper,
    policy_iteration_component,
    solve_core_heldkarp,
    to_exact_component,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")

Component = Tuple[Set[int], Dict[int, List[Tuple[int, int, float]]], Dict[int, float]]


# ---------------------------------------------------------------------------
# The four criteria
# ---------------------------------------------------------------------------


def criterion_D(comp: Component, lam: float) -> bool:
    """Definition beneficial-cycle, by exhaustive enumeration, over Fractions."""
    S, internal, e = comp
    iq, eq, lq = to_exact_component(internal, e, lam)
    J = brute_force_simple_paths(S, iq, eq, lq)
    return len(enumerate_beneficial_cycles(S, iq, eq, lq, J)) > 0


def criterion_W(comp: Component, lam: float) -> bool:
    """V_lambda* < J_lambda* somewhere on S, in exact arithmetic.

    Both sides are rational: V from exact policy iteration, J from the subset
    recursion run over Fractions.  Comparing an exact V against a float J would
    turn rounding noise into spurious strict gaps.
    """
    S, internal, e = comp
    V = policy_iteration_component(S, internal, e, lam, _exact_out=True)[0]
    iq, eq, lq = to_exact_component(internal, e, lam)
    J = solve_core_heldkarp(S, iq, eq, lq)[0]
    return any(V[x] < J[x] for x in S)


def criterion_P(comp: Component, lam: float) -> bool:
    """Some state has no V_lambda*-greedy path to the sink."""
    S, internal, e = comp
    return detect_beneficial_cycle_v2(S, internal, e, lam, exact=True)[0]


def criterion_T(comp: Component, lam: float, exact: bool) -> Tuple[bool, int]:
    """Sweep-count test of Lemma detect, in float or in exact arithmetic."""
    S, internal, e = comp
    if exact:
        iq, eq, lq = to_exact_component(internal, e, lam)
        _J, _p, bc, sw = local_lc_paper(S, iq, eq, lq, tol=0.0)
    else:
        _J, _p, bc, sw = local_lc_paper(S, internal, e, lam)
    return bc, sw


# ---------------------------------------------------------------------------
# Part A: the four criteria agree
# ---------------------------------------------------------------------------


def random_component(n: int, seed: int, exit_scale: float) -> Component:
    """Random strongly connected augmented component.

    Arc costs are drawn from [1, 3] and exit weights from [0.2, 1] * exit_scale,
    so sweeping exit_scale across the calls produces both components whose exits
    are cheap enough to be worth taking and components where looping pays.
    """
    import random

    rng = random.Random(seed)
    S = set(range(1, n + 1))
    internal: Dict[int, List[Tuple[int, int, float]]] = {i: [] for i in S}
    for i in range(1, n + 1):
        internal[i].append((0, i, 1.0 + 2.0 * rng.random()))
        nxt = i + 1 if i < n else 1
        internal[i].append((1, nxt, 1.0 + 2.0 * rng.random()))
        for j in rng.sample([k for k in S if k != i], min(2, n - 1)):
            internal[i].append((2 + j, j, 1.0 + 2.0 * rng.random()))
    e = {i: exit_scale * (0.2 + 0.8 * rng.random()) for i in S}
    if rng.random() < 0.3:  # some components can only be left from one state
        keep = rng.choice(sorted(S))
        e = {i: (e[i] if i == keep else INF) for i in S}
    return S, internal, e


def part_a() -> dict:
    rows = []
    disagree = []
    for lam in (0.3, 0.5, 0.8, 0.95):
        for n in range(3, 10):
            for seed in range(20):
                scale = (0.5, 2.0, 8.0, 40.0)[seed % 4]
                comp = random_component(n, seed * 31 + n, scale)
                d = criterion_D(comp, lam)
                w = criterion_W(comp, lam)
                p = criterion_P(comp, lam)
                t, _sw = criterion_T(comp, lam, exact=True)
                rows.append((lam, n, seed, d, w, p, t))
                if not (d == w == p == t):
                    disagree.append((lam, n, seed, d, w, p, t))
    n_bc = sum(1 for r in rows if r[3])
    return {
        "instances": len(rows),
        "with_cycle": n_bc,
        "without_cycle": len(rows) - n_bc,
        "disagreements": disagree,
    }


# ---------------------------------------------------------------------------
# Part B: where the float sweep test breaks
# ---------------------------------------------------------------------------


def decoy_ring(n: int, eps: float, exit_cost: float) -> Component:
    """A ring of n states, every one able to exit, with a cheap self-loop at 1.

    This is the shape the decoy takes on the ara operon: the state carrying the
    beneficial cycle can also leave immediately, so its label converges
    geometrically to eps/(1-lambda) instead of waiting for information to travel
    around the component.  After k sweeps the remaining decrease is of order
    lambda^k (exit_cost - eps/(1-lambda)), which is what eventually underflows.
    """
    S = set(range(1, n + 1))
    internal: Dict[int, List[Tuple[int, int, float]]] = {}
    internal[1] = [(0, 1, eps), (1, 2 if n > 1 else 1, 1.0)]
    for x in range(2, n + 1):
        internal[x] = [(1, x + 1 if x < n else 1, 1.0)]
    e = {x: exit_cost for x in S}
    return S, internal, e


def predicted_breakdown(
    eps: float, lam: float, exit_cost: float, tol: float = 1e-12
) -> float:
    """Estimated |S| at which the float sweep test can no longer see a decrease.

    The label at the decoy approaches J_inf = eps/(1-lambda) from above with a
    residual of order lambda^k (exit_cost - J_inf); the test goes blind once
    that residual drops below its resolution tol * J_inf.  Other states of the
    component also feed the decoy, so this is an estimate rather than an
    identity, and the observed breakdown comes slightly earlier.
    """
    J_inf = eps / (1 - lam)
    return math.log(tol * J_inf / (exit_cost - J_inf)) / math.log(lam) + 1


def figure_precision(rows: List[tuple]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [r for r in rows if r[0] != "predicted"]
    preds = {r[1]: (r[2], r[3]) for r in rows if r[0] == "predicted"}
    colours = {0.5: "#1f4e79", 0.8: "#1a7f37", 0.9: "#d1495b"}

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for lam in sorted(colours):
        sub = [r for r in data if r[0] == lam]
        ns = [r[1] for r in sub]
        ok = [1 if r[2] else 0 for r in sub]
        ax.step(ns, ok, where="post", color=colours[lam], lw=1.8,
                label=rf"$\lambda={lam}$")
        pred, _fail = preds[lam]
        ax.axvline(pred, color=colours[lam], ls=":", lw=1.2)
    ax.set_xlabel(r"$|S|$")
    ax.set_ylabel("float test detects the cycle")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_ylim(-0.2, 1.9)
    ax.set_xlim(0, 330)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.04))
    ax.text(0.03, 0.40,
            r"dotted: estimate $\lambda^{|S|}(e-J_\infty)=\mathrm{tol}\cdot J_\infty$",
            transform=ax.transAxes, fontsize=8.5, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(FIGURES, "detect_precision.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    a = part_a()
    assert not a["disagreements"], a["disagreements"]
    assert a["with_cycle"] > 0 and a["without_cycle"] > 0

    EPS, EXIT = 0.05, 50.0
    b_rows: List[tuple] = []
    for lam in (0.5, 0.8, 0.9):
        predicted = predicted_breakdown(EPS, lam, EXIT)
        first_fail = None
        for n in range(4, 241, 2):
            comp = decoy_ring(n, eps=EPS, exit_cost=EXIT)
            t_float, sw_f = criterion_T(comp, lam, exact=False)
            t_exact, _sw_e = criterion_T(comp, lam, exact=True)
            p = criterion_P(comp, lam)
            assert t_exact == p, (lam, n, t_exact, p)
            assert t_exact, (lam, n)  # the ring always carries the cycle
            b_rows.append((lam, n, t_float, t_exact, p, sw_f))
            if not t_float and first_fail is None:
                first_fail = n
        b_rows.append(("predicted", lam, predicted, first_fail, None, None))

    fig = figure_precision(b_rows)

    lines = [
        "E11: deciding the beneficial-cycle predicate of Algorithm 1",
        "",
        "  Part A: criteria (D) definition, (W) V*<J*, (P) tight-arc",
        "          reachability, (T) sweep count -- all in exact arithmetic",
        f"    random augmented components tested: {a['instances']}",
        f"      carrying a beneficial cycle: {a['with_cycle']}",
        f"      beneficial-cycle free:       {a['without_cycle']}",
        f"    instances where the four verdicts differ: {len(a['disagreements'])}",
        "",
        "  Part B: float vs exact arithmetic for the sweep test (T)",
        f"    family: ring of |S| states, exit weight {EXIT} everywhere, cheap",
        f"    self-loop (eps={EPS}) at state 1; the cycle is always beneficial",
        "    lambda   estimated breakdown   first |S| where float (T) misses it",
    ]
    for r in b_rows:
        if r[0] == "predicted":
            _tag, lam, pred, fail, _a, _b = r
            lines.append(
                f"     {lam:<8.2f} {pred:<21.0f}"
                f" {fail if fail is not None else 'none up to |S|=240'}"
            )
    lines += [
        "",
        "    lambda  |S|   float (T)   exact (T)   (P)   sweeps (float)",
    ]
    for r in b_rows:
        if r[0] == "predicted":
            continue
        lam, n, tf, te, p, sw = r
        if n % 20 != 0:
            continue
        lines.append(
            f"     {lam:<7.2f} {n:<5d} {str(tf):<11s} {str(te):<11s}"
            f" {str(p):<5s} {sw}"
        )
    lines += [
        "",
        "  Conclusion: the four criteria are interchangeable in exact",
        "  arithmetic; in floating point the sweep test needs the residual",
        "  lambda^{|S|} to stay above the unit roundoff, which is why",
        "  Algorithm 1 decides the predicate through criterion (P) instead.",
        f"  figure: {os.path.basename(fig)}",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp11_detection.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {
        "instances": a["instances"],
        "disagreements": len(a["disagreements"]),
        "float_breakdown": {
            r[1]: r[3] for r in b_rows if r[0] == "predicted"
        },
    }


if __name__ == "__main__":
    run()
