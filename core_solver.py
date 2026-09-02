"""
Discounted minimum stabilization cost — solver implementing main.tex algorithms.

Algorithms: reverse reachability, SCC condensation, downstream folding,
label-correcting (Lemma easy), beneficial-cycle detection, SolveCore (Held-Karp),
unconstrained VI, master assembly.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

INF = float("inf")
MAX_EXACT_CORE = 20

# Components at most this large are analysed with exact rational arithmetic, so
# that beneficial-cycle detection never depends on a convergence tolerance.
EXACT_PI_LIMIT = 64

# Sentinel "infinity" for exact arithmetic; large enough to dominate any cost
# arising in the experiments, small enough to keep denominators bounded.
FR_INF = Fraction(10**15)


def _to_fraction(v: Union[float, int, Fraction, None]) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if v is None:
        return FR_INF
    if isinstance(v, float):
        if math.isinf(v) or v >= float(FR_INF):
            return FR_INF
        return Fraction(v).limit_denominator(10**9)
    return Fraction(v)

Arc = Tuple[int, int, int, float]  # (x, u, y, g)


@dataclass
class WeightedGraph:
    """Weighted state transition graph G = (X, E)."""

    states: Set[int]
    arcs: List[Arc] = field(default_factory=list)
    by_state: Dict[int, List[Tuple[int, int, float]]] = field(default_factory=dict)
    rev: Dict[int, List[int]] = field(default_factory=dict)
    xe: Optional[int] = None

    def add_arc(self, x: int, u: int, y: int, g: float) -> None:
        self.states.add(x)
        self.states.add(y)
        self.arcs.append((x, u, y, g))
        self.by_state.setdefault(x, []).append((u, y, g))
        self.rev.setdefault(y, []).append(x)

    def restrict(self, R: Set[int]) -> "WeightedGraph":
        g2 = WeightedGraph(states=set(R), xe=self.xe)
        for x, u, y, w in self.arcs:
            if x in R and y in R:
                g2.add_arc(x, u, y, w)
        return g2

    def num_arcs(self) -> int:
        return len(self.arcs)


def build_graph(
    f: Callable[[int, int], int],
    g: Callable[[int, int], float],
    states: List[int],
    inputs: List[int],
    Cx: Optional[Set[int]] = None,
    Cu: Optional[Callable[[int], List[int]]] = None,
    xe: Optional[int] = None,
) -> WeightedGraph:
    Cx = set(states) if Cx is None else set(Cx)
    graph = WeightedGraph(states=set(), xe=xe)
    for x in Cx:
        us = inputs if Cu is None else Cu(x)
        for u in us:
            y = f(x, u)
            if y in Cx:
                graph.add_arc(x, u, y, float(g(x, u)))
    return graph


def reverse_reachable(G: WeightedGraph, xe: int) -> Set[int]:
    R: Set[int] = set()
    dq = deque([xe])
    R.add(xe)
    while dq:
        v = dq.popleft()
        for p in G.rev.get(v, []):
            if p not in R:
                R.add(p)
                dq.append(p)
    return R


@dataclass
class SCCResult:
    components: List[Set[int]]
    comp_of: Dict[int, int]
    dag_edges: List[Tuple[int, int]]
    reverse_topo: List[int]
    xe_comp: int


def condense(G: WeightedGraph, R: Set[int], xe: int) -> SCCResult:
    """Tarjan SCC on G|_R, build condensation DAG, reverse topological order.

    Tarjan pops a component only after everything reachable from it has been
    popped, so the pop order is already the reverse topological order the
    master algorithm needs (x_e's component, the unique sink on X_s, first).
    """
    sub = G.restrict(R)
    adj: Dict[int, List[int]] = {
        v: sorted({y for _u, y, _w in sub.by_state.get(v, [])}) for v in R
    }

    index: Dict[int, int] = {}
    low: Dict[int, int] = {}
    on_stack: Set[int] = set()
    tstack: List[int] = []
    comp_of: Dict[int, int] = {}
    components: List[Set[int]] = []
    counter = 0

    for root in sorted(R):
        if root in index:
            continue
        work: List[Tuple[int, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = counter
                low[v] = counter
                counter += 1
                tstack.append(v)
                on_stack.add(v)
            descended = False
            nbrs = adj.get(v, [])
            for i in range(pi, len(nbrs)):
                w = nbrs[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            if low[v] == index[v]:
                cid = len(components)
                comp: Set[int] = set()
                while True:
                    w = tstack.pop()
                    on_stack.discard(w)
                    comp.add(w)
                    comp_of[w] = cid
                    if w == v:
                        break
                components.append(comp)
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])

    xe_comp = comp_of[xe]
    dag_edges_set: Set[Tuple[int, int]] = set()
    for x, _u, y, _w in sub.arcs:
        cx, cy = comp_of[x], comp_of[y]
        if cx != cy:
            dag_edges_set.add((cx, cy))
    dag_edges = list(dag_edges_set)

    rev_topo = list(range(len(components)))
    position = {c: i for i, c in enumerate(rev_topo)}
    for a, b in dag_edges:
        assert position[b] < position[a], "condensation order violated"

    return SCCResult(
        components=components,
        comp_of=comp_of,
        dag_edges=dag_edges,
        reverse_topo=rev_topo,
        xe_comp=xe_comp,
    )


def augment_component(
    G: WeightedGraph,
    S: Set[int],
    J_known: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, List[Tuple[int, int, float]]], Dict[int, float]]:
    """Internal arcs within S and exit weights e(x) (Definition augment)."""
    internal: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    e: Dict[int, float] = {x: INF for x in S}
    for x in S:
        for u, y, w in G.by_state.get(x, []):
            if y in S:
                internal[x].append((u, y, w))
            else:
                exit_cost = w + lam * J_known.get(y, INF)
                if exit_cost < e[x]:
                    e[x] = exit_cost
        if e[x] == INF and not internal[x]:
            e[x] = INF
    return dict(internal), e


def local_lc(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    J_init: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]], int]:
    """Label-correcting sweeps (eq. local-lc), Lemma easy. Returns (J, policy, sweeps)."""
    if J_init is None:
        J = {x: e.get(x, INF) for x in S}
    else:
        J = {x: J_init.get(x, e.get(x, INF)) for x in S}
    policy: Dict[int, Tuple[int, int]] = {}
    n = len(S)
    sweeps = 0
    for _ in range(max(1, n - 1)):
        sweeps += 1
        changed = False
        for x in S:
            best = J[x]
            best_act: Optional[Tuple[int, int]] = None
            if e.get(x, INF) < best:
                best = e[x]
                best_act = ("exit", x)
            for u, y, w in internal.get(x, []):
                val = w + lam * J.get(y, INF)
                if val < best - 1e-12:
                    best = val
                    best_act = ("internal", u, y)
            if best < J[x] - 1e-12:
                J[x] = best
                changed = True
            if best_act is not None and best <= J[x] + 1e-12:
                policy[x] = best_act  # type: ignore
        if not changed:
            break
    # finalize policy at fixed point
    for x in S:
        best = e.get(x, INF)
        best_act: Optional[Tuple] = ("exit", x) if best < INF else None
        for u, y, w in internal.get(x, []):
            val = w + lam * J.get(y, INF)
            if val < best - 1e-12:
                best = val
                best_act = ("internal", u, y)
        if best_act is not None:
            policy[x] = best_act  # type: ignore
        J[x] = best
    return J, policy, sweeps


def local_lc_warm(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    J_init: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]], int]:
    """Label-correcting with warm start.

    J_init must upper-bound J_λ* (e.g. discounted cost of undiscounted-SP tree).
    """
    return local_lc(S, internal, e, lam, J_init=J_init)


def undiscounted_sp_to_omega(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
) -> Dict[int, float]:
    """Undiscounted shortest path to virtual sink omega on augmented component."""
    import heapq

    J = {x: e.get(x, INF) for x in S}
    heap = [(J[x], x) for x in S]
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if d > J[v]:
            continue
        for u, y, w in internal.get(v, []):
            if y not in S:
                continue
            nd = w + d
            if nd < J[y]:
                J[y] = nd
                heapq.heappush(heap, (nd, y))
    return J


def discounted_cost_sp_tree(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    J_undisc: Dict[int, float],
) -> Dict[int, float]:
    """Discounted cost of the undiscounted-SP greedy policy (proper upper bound)."""
    # Greedy successor from undiscounted values
    succ: Dict[int, Tuple[str, int, int, float]] = {}
    for x in S:
        best = e.get(x, INF)
        act = ("exit", 0, x, best)
        for u, y, w in internal.get(x, []):
            if y not in S:
                continue
            val = w + J_undisc.get(y, INF)
            if val < best - 1e-12:
                best = val
                act = ("internal", u, y, w)
        succ[x] = act

    def disc_cost(x: int, depth: int, seen: Set[int]) -> float:
        if x in seen:
            return INF
        kind, u, y, w = succ[x]
        if kind == "exit":
            return (lam**depth) * e.get(x, INF)
        return (lam**depth) * w + disc_cost(y, depth + 1, seen | {x})

    return {x: disc_cost(x, 0, set()) for x in S}


def _vi_on_component(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    tol: float = 1e-10,
    maxit: int = 100_000,
) -> Tuple[Dict[int, float], Dict[int, Union[Tuple[str, int], Tuple[str, int, int]]]]:
    """Unconstrained Bellman on augmented component S ∪ {ω}."""
    V = {x: e.get(x, 0.0) for x in S}
    for _ in range(maxit):
        psi = 0.0
        for x in S:
            best = e.get(x, INF)
            for u, y, w in internal.get(x, []):
                val = w + lam * V.get(y, INF)
                if val < best:
                    best = val
            psi = max(psi, abs(best - V[x]))
            V[x] = best
        if psi < tol:
            break
    policy: Dict[int, Union[Tuple[str, int], Tuple[str, int, int]]] = {}
    for x in S:
        best = e.get(x, INF)
        act: Union[Tuple[str, int], Tuple[str, int, int]] = ("exit", x)
        for u, y, w in internal.get(x, []):
            val = w + lam * V.get(y, INF)
            if val < best - 1e-12:
                best = val
                act = ("internal", u, y)
        policy[x] = act
        V[x] = best
    return V, policy


def local_lc_to_target(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    xe: int,
    lam: float,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]]]:
    """Label-correcting to absorbing target xe inside S (terminal SCC)."""
    J = {x: INF for x in S}
    J[xe] = 0.0
    policy: Dict[int, Tuple[int, int]] = {}
    n = len(S)
    for _ in range(max(1, n - 1)):
        changed = False
        for x in S:
            if x == xe:
                continue
            best = J[x]
            best_act: Optional[Tuple[int, int]] = None
            for u, y, w in internal.get(x, []):
                val = w + lam * J.get(y, INF)
                if val < best - 1e-12:
                    best = val
                    best_act = (u, y)
            if best < J[x] - 1e-12:
                J[x] = best
                changed = True
            if best_act is not None and abs(best - J[x]) < 1e-9 or best < J[x] - 1e-12:
                policy[x] = best_act  # type: ignore
                J[x] = best
        if not changed:
            break
    for x in S:
        if x == xe:
            continue
        best = J[x]
        best_act = policy.get(x)
        for u, y, w in internal.get(x, []):
            val = w + lam * J.get(y, INF)
            if val < best - 1e-12:
                best = val
                best_act = (u, y)
        J[x] = best
        if best_act:
            policy[x] = best_act
    return J, policy


def solve_core_to_target(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    xe: int,
    lam: float,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]]]:
    """Held-Karp simple paths to target xe inside S."""
    S_list = sorted(S)
    idx = {v: i for i, v in enumerate(S_list)}
    xe_i = idx[xe]
    H: Dict[Tuple[int, FrozenSet[int]], float] = {}
    choice: Dict[Tuple[int, FrozenSet[int]], Tuple[int, int]] = {}

    H[(xe_i, frozenset([xe_i]))] = 0.0

    for size in range(2, len(S_list) + 1):
        for comb in combinations(range(len(S_list)), size):
            W = frozenset(comb)
            if xe_i not in W:
                continue
            for i in W:
                if i == xe_i:
                    if (i, W) not in H:
                        H[(i, W)] = 0.0
                    continue
                v = S_list[i]
                best = INF
                best_ch: Optional[Tuple[int, int]] = None
                Wminus = frozenset(w for w in W if w != i)
                for u, y, w in internal.get(v, []):
                    j = idx.get(y)
                    if j is None or j not in Wminus:
                        continue
                    val = w + lam * H.get((j, Wminus), INF)
                    if val < best - 1e-12:
                        best = val
                        best_ch = (u, y)
                H[(i, W)] = best
                if best_ch:
                    choice[(i, W)] = best_ch

    full = frozenset(range(len(S_list)))
    J: Dict[int, float] = {}
    policy: Dict[int, Tuple[int, int]] = {}
    for i, v in enumerate(S_list):
        J[v] = H.get((i, full), INF)
        if v != xe and (i, full) in choice:
            policy[v] = choice[(i, full)]
    J[xe] = 0.0
    return J, policy


def _internal_arcs(G: WeightedGraph, S: Set[int]) -> Dict[int, List[Tuple[int, int, float]]]:
    internal: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for x in S:
        for u, y, w in G.by_state.get(x, []):
            if y in S:
                internal[x].append((u, y, w))
    return dict(internal)


def detect_beneficial_cycle_to_target(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    xe: int,
    lam: float,
) -> bool:
    """Improper VI minimizer in S\\{xe} (loops without reaching xe)."""
    sub = set(S)
    e = {x: INF for x in sub}
    e[xe] = 0.0
    V, pol = _vi_on_component(sub, internal, e, lam)
    for x0 in sub:
        if x0 == xe:
            continue
        cur = x0
        seen: Set[int] = set()
        for _ in range(len(sub) + 2):
            if cur == xe:
                break
            act = pol.get(cur)
            if act is None or act[0] == "exit":
                break
            if cur in seen:
                return True
            seen.add(cur)
            _t, _u, y = act  # type: ignore
            cur = y
        else:
            return True
    return False


def detect_beneficial_cycle(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> bool:
    """True iff VI-greedy policy is improper (loops inside S)."""
    _V, pol = _vi_on_component(S, internal, e, lam)
    for x0 in S:
        cur = x0
        seen: Set[int] = set()
        for _ in range(len(S) + 2):
            act = pol.get(cur)
            if act is None or act[0] == "exit":
                break
            if cur in seen:
                return True
            seen.add(cur)
            _t, u, y = act  # type: ignore
            cur = y
        else:
            return True
    return False


# ---------------------------------------------------------------------------
# Beneficial-cycle detection via the greedy arc set (Proposition gap-W(b))
# ---------------------------------------------------------------------------
#
# Proposition gap-W(b): S carries no beneficial cycle iff SOME V_λ*-induced
# policy is proper.  The existential quantifier matters: extracting one greedy
# policy and testing it for cycles (detect_beneficial_cycle above) can report a
# beneficial cycle when ties among equally optimal internal arcs happen to be
# resolved towards a loop.  The test below keeps every minimizing action and
# asks whether an acyclic selection exists, which is exactly the proposition.

Action = Union[Tuple[str, int], Tuple[str, int, int]]

Num = Union[float, Fraction]


def _is_inf(v: Num) -> bool:
    """Infinity test valid for both float and exact-rational values."""
    return v >= FR_INF if isinstance(v, Fraction) else v >= float(FR_INF)


def to_exact_component(
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, List[Tuple[int, int, Fraction]]], Dict[int, Fraction], Fraction]:
    """Rational copy of an augmented component."""
    internal_q = {
        x: [(u, y, _to_fraction(w)) for u, y, w in arcs] for x, arcs in internal.items()
    }
    e_q = {x: _to_fraction(v) for x, v in e.items()}
    return internal_q, e_q, _to_fraction(lam)


def greedy_action_set(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, Num]]],
    e: Dict[int, Num],
    lam: Num,
    V: Dict[int, Num],
    tol: float = 1e-9,
) -> Dict[int, List[Action]]:
    """All actions attaining the Bellman minimum at each state of S.

    Keeping every minimizer (not just one) is what makes the subsequent
    reachability test match the existential quantifier of Proposition gap-W(b).
    """
    acts: Dict[int, List[Action]] = {}
    for x in S:
        vals: List[Tuple[Num, Action]] = []
        ex = e.get(x, INF)
        if not _is_inf(ex):
            vals.append((ex, ("exit", x)))
        for u, y, w in internal.get(x, []):
            vy = V.get(y, INF)
            if _is_inf(vy):
                continue
            vals.append((w + lam * vy, ("internal", u, y)))
        if not vals:
            acts[x] = []
            continue
        best = min(v for v, _a in vals)
        slack = 0.0 if tol == 0 else tol * max(1.0, abs(float(best)))
        acts[x] = [a for v, a in vals if float(v) <= float(best) + slack]
    return acts


def tight_arc_in_tree(
    S: Set[int], acts: Dict[int, List[Action]]
) -> Tuple[Set[int], Dict[int, Action]]:
    """Backward BFS on the tight arcs: reachable set plus an in-tree to omega.

    Layer 0 is the virtual sink, and a state enters layer k+1 through a tight
    action whose successor already sits in layer k.  The recorded action
    therefore strictly decreases the layer index, so the selection is acyclic
    and proper -- the in-tree of Lemma detect(b).
    """
    pred: Dict[int, List[Tuple[int, Action]]] = defaultdict(list)
    good: Set[int] = set()
    tree: Dict[int, Action] = {}
    dq: deque[int] = deque()
    for x in S:
        for a in acts.get(x, []):
            if a[0] == "exit":
                if x not in good:
                    good.add(x)
                    tree[x] = a
                    dq.append(x)
            else:
                pred[a[2]].append((x, a))
    while dq:
        y = dq.popleft()
        for x, a in pred.get(y, []):
            if x not in good:
                good.add(x)
                tree[x] = a
                dq.append(x)
    return good, tree


def states_reaching_sink(S: Set[int], acts: Dict[int, List[Action]]) -> Set[int]:
    """States admitting a greedy path to the virtual sink (backward BFS)."""
    return tight_arc_in_tree(S, acts)[0]


def component_closed_loop_value(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    policy: Dict[int, Action],
    x0: int,
) -> float:
    """Value of a stationary feedback on a folded component, INF if trapped.

    A repeated state means the closed loop never reaches omega, which is an
    improper closed loop in the sense of Definition proper.
    """
    cur, seen, total, disc = x0, set(), 0.0, 1.0
    while True:
        if cur in seen:
            return INF
        seen.add(cur)
        act = policy.get(cur)
        if act is None or act[0] == "exit":
            return total + disc * e.get(cur, INF)
        _t, u, nxt = act
        wgt = next(
            (w for u2, y, w in internal.get(cur, []) if y == nxt and u2 == u), None
        )
        if wgt is None:
            return INF
        total += disc * wgt
        disc *= lam
        cur = nxt


def component_trapped_states(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    policy: Dict[int, Action],
) -> Set[int]:
    """States whose closed loop under this feedback never leaves the component."""
    return {
        x
        for x in S
        if component_closed_loop_value(S, internal, e, lam, policy, x) == INF
    }


def assemble_proper_in_tree(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, float], Dict[int, Action]]:
    """One stationary feedback on the component that is guaranteed proper.

    SolveCore returns, for each state, the first arc of that state's own
    optimal simple path; on a beneficial-cycle component those arcs may point
    at each other and trap the closed loop, and no stationary feedback attains
    every pointwise optimum there anyway.  This settles the states one at a
    time, always choosing an arc into the already settled part, so every state
    is strictly closer to omega than its successor and the selection is
    acyclic, hence proper.  Its value is at or above J_lambda*.
    """
    V: Dict[int, float] = {}
    tree: Dict[int, Action] = {}
    pending = set(S)
    while pending:
        pick_x, pick_val, pick_act = None, INF, None
        for x in pending:
            cand, act = e.get(x, INF), ("exit", x)
            for u, y, w in internal.get(x, []):
                if y in V:
                    val = w + lam * V[y]
                    if val < cand - 1e-12:
                        cand, act = val, ("internal", u, y)
            if cand < pick_val:
                pick_x, pick_val, pick_act = x, cand, act
        if pick_x is None or pick_val == INF:
            break  # remaining states cannot reach omega at all
        V[pick_x], tree[pick_x] = pick_val, pick_act
        pending.discard(pick_x)
    return V, tree


def detect_with_in_tree(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    exact: Optional[bool] = None,
    tol: float = 1e-9,
) -> Tuple[bool, Set[int], Dict[int, float], Dict[int, Action]]:
    """Lines 5-8 of Algorithm master: exact V_λ*, tight arcs, backward BFS.

    Returns (has_beneficial_cycle, trapped, V, in_tree).  When nothing is
    trapped, ``V`` is already J_λ* on S and ``in_tree`` an optimal proper
    feedback (Lemma detect(b)); otherwise ``V`` is the lower bound of
    Lemma detect(c) and the component must go to SolveCore.
    """
    if exact is None:
        exact = len(S) <= EXACT_PI_LIMIT
    if exact:
        internal_q, e_q, lam_q = to_exact_component(internal, e, lam)
        V_q = policy_iteration_component(S, internal, e, lam, _exact_out=True)[0]
        acts = greedy_action_set(S, internal_q, e_q, lam_q, V_q, tol=0.0)
        V = {x: float(v) for x, v in V_q.items()}
    else:
        V, _pol = _vi_on_component(S, internal, e, lam)
        acts = greedy_action_set(S, internal, e, lam, V, tol=tol)
    good, tree = tight_arc_in_tree(S, acts)
    trapped = set(S) - good
    return bool(trapped), trapped, V, tree


def detect_beneficial_cycle_v2(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    exact: Optional[bool] = None,
    tol: float = 1e-9,
) -> Tuple[bool, Set[int], Dict[int, float]]:
    """DetectBeneficialCycle.

    Returns (has_beneficial_cycle, trapped_states, V_λ*).  ``trapped_states``
    are those from which every Bellman-greedy action set loops inside S; the
    component carries a beneficial cycle iff this set is nonempty.
    """
    has_bc, trapped, V, _tree = detect_with_in_tree(
        S, internal, e, lam, exact=exact, tol=tol
    )
    return has_bc, trapped, V


def sweep_T(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, Num]]],
    e: Dict[int, Num],
    lam: Num,
    J: Dict[int, Num],
) -> Tuple[Dict[int, Num], Dict[int, Action]]:
    """One synchronous application of the operator T of eq. local-lc.

    (T J)(x) = min{ e(x), min over internal arcs (x,u,y) of g(x,u) + lam J(y) }.
    Updating out of place is what makes J_k the minimum cost over walks using at
    most k internal arcs, which is the reading Lemma detect(a) relies on.
    """
    out: Dict[int, Num] = {}
    pol: Dict[int, Action] = {}
    for x in S:
        best = e.get(x, INF)
        act: Optional[Action] = ("exit", x) if not _is_inf(best) else None
        for u, y, w in internal.get(x, []):
            jy = J.get(y, INF)
            if _is_inf(jy):
                continue
            val = w + lam * jy
            if val < best:
                best, act = val, ("internal", u, y)
        out[x] = best
        if act is not None:
            pol[x] = act
    return out, pol


def _strictly_below(A: Dict[int, Num], B: Dict[int, Num], tol: float) -> bool:
    """Whether A(x) < B(x) somewhere, up to a relative tolerance.

    With tol = 0 and rational inputs the comparison is exact, which matters:
    after k sweeps the residual decrease along a cheap cycle is of order
    lambda^k, so a float test silently declares a fixed point long before one
    is reached (see the discussion of Lemma detect in the experiments).
    """
    for x, a in A.items():
        b = B.get(x, INF)
        if _is_inf(b):
            if not _is_inf(a):
                return True
            continue
        if tol == 0:
            if a < b:
                return True
            continue
        if float(a) < float(b) - tol * max(1.0, abs(float(b))):
            return True
    return False


def local_lc_paper(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, Num]]],
    e: Dict[int, Num],
    lam: Num,
    tol: float = 1e-12,
) -> Tuple[Dict[int, Num], Dict[int, Action], bool, int]:
    """Lines 6-8 of Algorithm master: |S|-1 sweeps plus one verification sweep.

    Returns (J_{|S|-1}, induced policy, has_beneficial_cycle, sweeps performed).
    By Lemma detect(b) the component has no beneficial cycle exactly when the
    verification sweep leaves the labels unchanged, in which case J_{|S|-1} is
    already J_lambda* on S; otherwise J_{|S|-1} is the lower bound of
    Lemma detect(c).  T J <= J holds throughout, so the fixed-point test only
    has to look for a strict decrease.
    """
    J: Dict[int, Num] = {x: e.get(x, INF) for x in S}
    pol: Dict[int, Action] = {x: ("exit", x) for x in S if not _is_inf(J[x])}
    sweeps = 0
    for _ in range(max(0, len(S) - 1)):
        J_next, pol_next = sweep_T(S, internal, e, lam, J)
        sweeps += 1
        settled = not _strictly_below(J_next, J, tol)
        J, pol = J_next, pol_next
        if settled:
            return J, pol, False, sweeps
    J_test, pol_test = sweep_T(S, internal, e, lam, J)
    sweeps += 1
    if _strictly_below(J_test, J, tol):
        return J, pol, True, sweeps
    return J_test, pol_test, False, sweeps


def terminal_to_augmented(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    xe: int,
    lam: float,
) -> Tuple[Set[int], Dict[int, List[Tuple[int, int, float]]], Dict[int, float]]:
    """Recast the terminal SCC as an augmented component with omega = x_e."""
    S2 = {x for x in S if x != xe}
    internal2: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    e2: Dict[int, float] = {x: INF for x in S2}
    for x in S2:
        for u, y, w in internal.get(x, []):
            if y == xe:
                if w < e2[x]:
                    e2[x] = w
            elif y in S2:
                internal2[x].append((u, y, w))
    return S2, dict(internal2), e2


def detect_beneficial_cycle_to_target_v2(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    xe: int,
    lam: float,
    exact: Optional[bool] = None,
) -> Tuple[bool, Set[int], Dict[int, float]]:
    """Same test on the terminal component, where x_e plays the role of omega."""
    S2, internal2, e2 = terminal_to_augmented(S, internal, xe, lam)
    if not S2:
        return False, set(), {}
    return detect_beneficial_cycle_v2(S2, internal2, e2, lam, exact=exact)


# ---------------------------------------------------------------------------
# Exact policy iteration (Fraction arithmetic, no convergence tolerance)
# ---------------------------------------------------------------------------


def evaluate_policy_exact(
    S: Set[int],
    policy: Dict[int, Action],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: Fraction,
) -> Dict[int, Fraction]:
    """Exact value of a stationary policy on an augmented component.

    The closed loop is a functional graph: trees hanging off cycles.  Cycle
    values come from the lasso formula c/(1-lam^l); tree values follow by back
    substitution.  No linear solve and no tolerance is involved.
    """
    arc_cost: Dict[int, Fraction] = {}
    succ: Dict[int, Optional[int]] = {}
    for x in S:
        a = policy.get(x)
        if a is None or a[0] == "exit":
            succ[x] = None
            ex = e.get(x, INF)
            arc_cost[x] = _to_fraction(ex)
        else:
            _t, u, y = a
            succ[x] = y
            w = next(
                (ww for uu, yy, ww in internal.get(x, []) if uu == u and yy == y),
                INF,
            )
            arc_cost[x] = _to_fraction(w)

    V: Dict[int, Fraction] = {}
    color: Dict[int, int] = {}  # 0 unvisited, 1 on the current path, 2 done

    for x0 in S:
        if color.get(x0, 0) == 2:
            continue
        path: List[int] = []
        pos: Dict[int, int] = {}
        cur: Optional[int] = x0
        while cur is not None and color.get(cur, 0) == 0:
            color[cur] = 1
            pos[cur] = len(path)
            path.append(cur)
            cur = succ[cur]

        if cur is not None and color.get(cur, 0) == 1:
            start = pos[cur]
            cyc = path[start:]
            ell = len(cyc)
            c = Fraction(0)
            disc = Fraction(1)
            for v in cyc:
                c += disc * arc_cost[v]
                disc *= lam
            head = min(c / (1 - lam**ell), FR_INF)
            V[cyc[0]] = head
            for k in range(1, ell):
                prev = cyc[k - 1]
                V[cyc[k]] = min((V[prev] - arc_cost[prev]) / lam, FR_INF)
            for v in cyc:
                color[v] = 2
            path = path[:start]

        for v in reversed(path):
            s = succ[v]
            val = arc_cost[v] if s is None else arc_cost[v] + lam * V.get(s, FR_INF)
            V[v] = min(val, FR_INF)
            color[v] = 2
    return V


def policy_iteration_component(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    max_iter: int = 10_000,
    _exact_out: bool = False,
) -> Tuple[Dict[int, Num], Dict[int, Action]]:
    """Exact policy iteration for the unconstrained value on S u {omega}.

    Terminates in finitely many steps with the exact V_lambda*, so a reported
    improper minimizer cannot be a convergence-tolerance artifact.
    """
    V, policy, _steps = policy_iteration_exact(S, internal, e, lam, max_iter)
    if _exact_out:
        return V, policy
    return {x: float(v) for x, v in V.items()}, policy


def policy_iteration_exact(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    max_iter: int = 10_000,
) -> Tuple[Dict[int, Fraction], Dict[int, Action], int]:
    """Same computation, also reporting the number of steps performed.

    The step count is the D of Proposition complexity: one step evaluates the
    current policy in closed form and rescans every arc once, so it costs
    O(M|S|).
    """
    internal_q, e_q, lam_q = to_exact_component(internal, e, lam)
    policy: Dict[int, Action] = {}
    for x in S:
        if not _is_inf(e_q.get(x, FR_INF)):
            policy[x] = ("exit", x)
        elif internal_q.get(x):
            u, y, _w = internal_q[x][0]
            policy[x] = ("internal", u, y)

    V: Dict[int, Fraction] = {}
    steps = 0
    for _ in range(max_iter):
        V = evaluate_policy_exact(S, policy, internal_q, e_q, lam_q)
        steps += 1
        stable = True
        for x in S:
            cur = V.get(x, FR_INF)
            best = cur
            best_act = policy.get(x)
            ex = e_q.get(x, FR_INF)
            if not _is_inf(ex) and ex < best:
                best = ex
                best_act = ("exit", x)
            for u, y, w in internal_q.get(x, []):
                vy = V.get(y, FR_INF)
                if _is_inf(vy):
                    continue
                val = w + lam_q * vy
                if val < best:
                    best = val
                    best_act = ("internal", u, y)
            if best_act is not None and best < cur:
                policy[x] = best_act
                stable = False
        if stable:
            break
    return dict(V), policy, steps


def policy_iteration_graph(
    G: WeightedGraph,
    xe: int,
    lam: float,
    R: Optional[Set[int]] = None,
    max_iter: int = 10_000,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]]]:
    """Exact policy iteration for V_lambda* on the whole graph (x_e absorbing).

    Same fixed point as vi_unconstrained, reached in finitely many exact steps,
    so the improper minimizer it may return is not a tolerance artifact.
    """
    states = set(R if R is not None else G.states)
    S = {x for x in states if x != xe}
    internal: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    e: Dict[int, float] = {x: INF for x in S}
    for x in S:
        for u, y, w in G.by_state.get(x, []):
            if y not in states:
                continue
            if y == xe:
                if w < e[x]:
                    e[x] = w
            else:
                internal[x].append((u, y, w))
    V, pol = policy_iteration_component(S, dict(internal), e, lam)
    V_out: Dict[int, float] = {xe: 0.0}
    V_out.update({x: float(v) for x, v in V.items()})
    mu: Dict[int, Tuple[int, int]] = {}
    for x in S:
        a = pol.get(x)
        if a is None:
            continue
        if a[0] == "exit":
            best, bu = INF, None
            for u, y, w in G.by_state.get(x, []):
                if y == xe and w < best:
                    best, bu = w, u
            if bu is not None:
                mu[x] = (bu, xe)
        else:
            _t, u, y = a
            mu[x] = (u, y)
    return V_out, mu


def enumerate_beneficial_cycles(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    J: Dict[int, float],
    max_states: int = 14,
) -> List[Tuple[List[int], float, float]]:
    """Oracle: every beneficial cycle, straight from Definition beneficial-cycle.

    Enumerates each simple non-target cycle once (canonically, from its smallest
    vertex) and then tests condition eq:beneficial-cycle at *every* vertex of
    the cycle: g_loop depends on where the traversal starts, so a cycle can be
    beneficial at one of its states and not at another.  Returns the
    (rotated cycle, g_loop, threshold) triples that satisfy it.  Exhaustive, so
    only for small components; used to cross-validate the detectors.
    """
    if len(S) > max_states:
        raise ValueError(f"|S|={len(S)} too large for exhaustive cycle enumeration")
    found: List[Tuple[List[int], float, float]] = []
    cycles: List[Tuple[List[int], List[float]]] = []

    def walk(start: int, cur: int, path: List[int], weights: List[float]) -> None:
        for u, y, w in internal.get(cur, []):
            if y == start:
                cycles.append((list(path), weights + [w]))
            elif y in S and y not in path and y > start:
                walk(start, y, path + [y], weights + [w])

    for s in sorted(S):
        walk(s, s, [s], [])

    for verts, ws in cycles:
        ell = len(verts)
        for r in range(ell):
            g_loop = sum(lam**i * ws[(r + i) % ell] for i in range(ell))
            y = verts[r]
            thr = (1 - lam**ell) * J.get(y, INF)
            if g_loop < thr:
                rot = [verts[(r + i) % ell] for i in range(ell)] + [y]
                found.append((rot, g_loop, thr))
    return found


def solve_core_bitmask(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, float], Dict[int, Action], int]:
    """SolveCore (Algorithm core) with bitmask subsets and vectorized sweeps.

    Same recursion and same return convention as solve_core_heldkarp, but
    subsets are indexed by integers and each subset-size layer is evaluated with
    array operations, which keeps |S| = 20 inside a few hundred megabytes
    instead of several gigabytes.
    """
    import numpy as np

    S_list = sorted(S)
    n = len(S_list)
    if n > MAX_EXACT_CORE:
        raise RuntimeError(f"|S|={n} exceeds exact SolveCore limit {MAX_EXACT_CORE}")
    idx = {v: i for i, v in enumerate(S_list)}
    arcs: List[List[Tuple[int, float, int]]] = [
        [(idx[y], float(w), u) for u, y, w in internal.get(v, []) if y in idx]
        for v in S_list
    ]
    e_arr = np.array([float(e.get(v, INF)) for v in S_list], dtype=np.float64)

    size = 1 << n
    H = np.full((size, n), np.inf, dtype=np.float64)
    CH = np.full((size, n), -1, dtype=np.int16)  # -1 encodes the exit to omega
    for i in range(n):
        H[1 << i, i] = e_arr[i]

    # popcnt[2^i + k] = popcnt[k] + 1 for k < 2^i, so the filled prefix doubles
    # at every step and the whole table is built in n vectorized assignments.
    popcnt = np.zeros(size, dtype=np.int8)
    for i in range(n):
        half = 1 << i
        popcnt[half : 2 * half] = popcnt[:half] + 1
    masks_all = np.arange(size, dtype=np.int64)
    order = [masks_all[popcnt == s] for s in range(n + 1)]

    evals = n
    for s in range(2, n + 1):
        masks = order[s]
        if masks.size == 0:
            continue
        for i in range(n):
            bit = np.int64(1) << np.int64(i)
            sel = masks[(masks & bit) != 0]
            if sel.size == 0:
                continue
            prev = sel ^ bit
            best = np.full(sel.size, e_arr[i], dtype=np.float64)
            bch = np.full(sel.size, -1, dtype=np.int16)
            for ai, (j, w, _u) in enumerate(arcs[i]):
                ok = (prev & (np.int64(1) << np.int64(j))) != 0
                if not ok.any():
                    continue
                cand = np.full(sel.size, np.inf, dtype=np.float64)
                cand[ok] = w + lam * H[prev[ok], j]
                upd = cand < best
                if upd.any():
                    best = np.where(upd, cand, best)
                    bch = np.where(upd, np.int16(ai), bch)
            H[sel, i] = best
            CH[sel, i] = bch
            evals += int(sel.size)

    full = size - 1
    J: Dict[int, float] = {}
    policy: Dict[int, Action] = {}
    for i, v in enumerate(S_list):
        J[v] = float(H[full, i])
        a = int(CH[full, i])
        if a < 0:
            policy[v] = ("exit", v)
        else:
            j, _w, u = arcs[i][a]
            policy[v] = ("internal", u, S_list[j])
    return J, policy, evals


BITMASK_FROM = 10


def solve_core(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, float], Dict[int, Action], int]:
    """Algorithm core: dictionary recursion when tiny, bitmask sweeps when not.

    See solve_core_heldkarp for what the returned policy does and does not
    guarantee.
    """
    if len(S) >= BITMASK_FROM:
        return solve_core_bitmask(S, internal, e, lam)
    return solve_core_heldkarp(S, internal, e, lam)


def solve_core_heldkarp(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    ub: Optional[Dict[int, float]] = None,
) -> Tuple[
    Dict[int, float],
    Dict[int, Tuple[str, int] | Tuple[str, int, int]],
    int,
]:
    """Held-Karp subset DP (Algorithm core). Returns (J, policy, eval_count).

    ``J[v]`` is the minimum discounted cost of a simple path from v to omega,
    and ``policy[v]`` is the first arc of one such path.  The arcs are read off
    the table entries (v, S), so each is optimal for its own state starting
    fresh; they are not a joint feedback.  On a beneficial-cycle component they
    may point at each other, and no stationary feedback attains every J[v] at
    once, so callers that need one feedback should check the reading with
    component_trapped_states and fall back on assemble_proper_in_tree.
    """
    S_list = sorted(S)
    idx = {v: i for i, v in enumerate(S_list)}
    n = len(S_list)
    ub = ub or {}
    eval_count = 0

    H: Dict[Tuple[int, FrozenSet[int]], float] = {}
    choice: Dict[Tuple[int, FrozenSet[int]], Tuple] = {}

    for i, v in enumerate(S_list):
        W = frozenset([i])
        H[(i, W)] = e.get(v, INF)
        choice[(i, W)] = ("exit", v)
        eval_count += 1

    for size in range(2, n + 1):
        for comb in combinations(range(n), size):
            W = frozenset(comb)
            for i in W:
                v = S_list[i]
                if ub.get(v, INF) < H.get((i, frozenset([i])), INF) and size == len(W):
                    pass
                best = e.get(v, INF)
                best_ch: Tuple = ("exit", v)
                Wminus = frozenset(w for w in W if w != i)
                for u, y, wgt in internal.get(v, []):
                    j = idx.get(y)
                    if j is None or j not in Wminus:
                        continue
                    val = wgt + lam * H.get((j, Wminus), INF)
                    if val < best - 1e-12:
                        best = val
                        best_ch = ("internal", u, y)
                if ub.get(v, INF) < best:
                    best = min(best, ub[v])
                H[(i, W)] = best
                choice[(i, W)] = best_ch
                eval_count += 1
                if ub.get(v, INF) <= best + 1e-12 and size < n:
                    continue

    full = frozenset(range(n))
    J: Dict[int, float] = {}
    policy: Dict[int, Tuple] = {}
    for i, v in enumerate(S_list):
        J[v] = H.get((i, full), INF)
        policy[v] = choice.get((i, full), ("exit", v))
    return J, policy, eval_count


def solve_core_heldkarp_bounded(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    ub: Dict[int, float],
) -> Tuple[
    Dict[int, float],
    Dict[int, Tuple[str, int] | Tuple[str, int, int]],
    int,
]:
    """Held-Karp with per-vertex upper bounds from Dijkstra warm start."""
    S_list = sorted(S)
    idx = {v: i for i, v in enumerate(S_list)}
    n = len(S_list)
    eval_count = 0
    pruned = 0

    H: Dict[Tuple[int, FrozenSet[int]], float] = {}
    choice: Dict[Tuple[int, FrozenSet[int]], Tuple] = {}

    for i, v in enumerate(S_list):
        W = frozenset([i])
        H[(i, W)] = min(e.get(v, INF), ub.get(v, INF))
        choice[(i, W)] = ("exit", v)
        eval_count += 1

    for size in range(2, n + 1):
        for comb in combinations(range(n), size):
            W = frozenset(comb)
            for i in W:
                v = S_list[i]
                cap = ub.get(v, INF)
                best = min(e.get(v, INF), cap)
                best_ch: Tuple = ("exit", v)
                Wminus = frozenset(w for w in W if w != i)
                partial_pruned = False
                for u, y, wgt in internal.get(v, []):
                    j = idx.get(y)
                    if j is None or j not in Wminus:
                        continue
                    sub = H.get((j, Wminus), INF)
                    if sub >= INF:
                        continue
                    val = wgt + lam * sub
                    if val >= cap - 1e-12:
                        partial_pruned += 1
                        continue
                    if val < best - 1e-12:
                        best = val
                        best_ch = ("internal", u, y)
                if partial_pruned:
                    pruned += 1
                if best > cap:
                    best = cap
                H[(i, W)] = best
                choice[(i, W)] = best_ch
                eval_count += 1

    full = frozenset(range(n))
    J: Dict[int, float] = {}
    policy: Dict[int, Tuple] = {}
    for i, v in enumerate(S_list):
        J[v] = H.get((i, full), INF)
        policy[v] = choice.get((i, full), ("exit", v))
    return J, policy, eval_count + pruned


def solve_core_depth(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
    D: int,
) -> Tuple[
    Dict[int, float],
    Dict[int, Tuple[str, int] | Tuple[str, int, int]],
    int,
]:
    """Depth-parametrized core solver (the XP route of Section main).

    Returns the optimum over stabilizing simple paths that use at most ``D``
    internal arcs of ``S``.  This is an upper bound on the exact core value,
    with equality once ``D`` reaches the depth of the optimal simple path.
    Enumerating those paths costs O(|S| M^D), i.e. polynomial in |S| and M for
    every fixed D, whereas the subset solver of Algorithm core costs
    O(2^{|S|} |S| M).

    Returns (J, policy, expansions), where ``expansions`` counts visited path
    prefixes and plays the role of the (v, W) evaluation count of SolveCore.
    """
    J: Dict[int, float] = {}
    policy: Dict[int, Tuple] = {}
    visited: Set[int] = set()
    expansions = 0

    def dfs(v: int, depth: int) -> Tuple[float, Tuple]:
        nonlocal expansions
        expansions += 1
        best = e.get(v, INF)
        best_ch: Tuple = ("exit", v)
        if depth < D:
            for u, y, w in internal.get(v, []):
                if y in visited:
                    continue
                visited.add(y)
                sub, _ = dfs(y, depth + 1)
                visited.discard(y)
                if sub >= INF:
                    continue
                val = w + lam * sub
                if val < best - 1e-12:
                    best = val
                    best_ch = ("internal", u, y)
        return best, best_ch

    for v in sorted(S):
        visited.clear()
        visited.add(v)
        val, ch = dfs(v, 0)
        J[v] = val
        policy[v] = ch
    return J, policy, expansions


def brute_force_simple_paths(
    S: Set[int],
    internal: Dict[int, List[Tuple[int, int, float]]],
    e: Dict[int, float],
    lam: float,
) -> Dict[int, float]:
    """Oracle: enumerate all simple paths to exit."""
    result: Dict[int, float] = {}

    def dfs(x: int, visited: Set[int], cost: float, depth: int) -> float:
        best = cost + (lam**depth) * e.get(x, INF)
        for u, y, w in internal.get(x, []):
            if y in visited:
                continue
            cand = dfs(y, visited | {y}, cost + (lam**depth) * w, depth + 1)
            if cand < best:
                best = cand
        return best

    for x0 in S:
        result[x0] = dfs(x0, {x0}, 0.0, 0)
    return result


def vi_unconstrained(
    G: WeightedGraph,
    lam: float,
    R: Optional[Set[int]] = None,
    xe: Optional[int] = None,
    tol: float = 1e-9,
    maxit: int = 200_000,
) -> Dict[int, float]:
    """Unconstrained discounted optimum V_λ* (Bellman)."""
    xe = xe if xe is not None else G.xe
    states = sorted(R if R is not None else G.states)
    V = {x: 0.0 for x in states}
    if xe is not None:
        V[xe] = 0.0
    for _ in range(maxit):
        psi = 0.0
        for x in states:
            if x == xe:
                V[x] = 0.0
                continue
            best = INF
            for u, y, w in G.by_state.get(x, []):
                if R is not None and y not in R:
                    continue
                val = w + lam * V.get(y, INF)
                if val < best:
                    best = val
            psi = max(psi, abs(best - V.get(x, INF)))
            V[x] = best
        if psi < tol:
            break
    return V


def vi_policy(
    G: WeightedGraph,
    V: Dict[int, float],
    lam: float,
    R: Optional[Set[int]] = None,
) -> Dict[int, Tuple[int, int]]:
    """Greedy policy for V."""
    pol: Dict[int, Tuple[int, int]] = {}
    states = R if R is not None else G.states
    xe = G.xe
    for x in states:
        if x == xe:
            continue
        best = INF
        bu, by = 0, x
        for u, y, w in G.by_state.get(x, []):
            if R is not None and y not in R:
                continue
            val = w + lam * V.get(y, INF)
            if val < best - 1e-12:
                best = val
                bu, by = u, y
        pol[x] = (bu, by)
    return pol


def simulate_closed_loop(
    f: Callable[[int, int], int],
    mu: Dict[int, Tuple[int, int]],
    x0: int,
    xe: int,
    max_steps: Optional[int] = None,
) -> Tuple[bool, int, List[int]]:
    """Return (proper, steps_to_xe, trajectory)."""
    N = max_steps or 10_000
    traj = [x0]
    cur = x0
    seen: Set[int] = set()
    for t in range(N):
        if cur == xe:
            return True, t, traj
        if cur in seen:
            return False, t, traj
        seen.add(cur)
        if cur not in mu:
            return False, t, traj
        u, _y = mu[cur]
        cur = f(cur, u)
        traj.append(cur)
    return False, N, traj


def reverse_dijkstra_discounted_wrong(
    G: WeightedGraph,
    xe: int,
    lam: float,
    R: Optional[Set[int]] = None,
) -> Dict[int, float]:
    """
    Naive label-setting assuming J(x) >= J(successor) — fails under discounting.
    Uses undiscounted-style locking on reversed graph with discounted relaxation once.
    """
    states = R if R is not None else G.states
    J = {x: INF for x in states}
    J[xe] = 0.0
    finalized = {xe}
    # priority by current J
    import heapq

    heap: List[Tuple[float, int]] = [(0.0, xe)]
    while heap:
        jv, v = heapq.heappop(heap)
        if v in finalized and jv > J[v] + 1e-9:
            continue
        finalized.add(v)
        for p in G.rev.get(v, []):
            if p not in states:
                continue
            # find best arc p -> v
            best = INF
            for u, y, w in G.by_state.get(p, []):
                if y == v:
                    cand = w + lam * J[v]
                    if cand < best:
                        best = cand
            if p not in finalized:
                if best < J[p]:
                    J[p] = best
                    heapq.heappush(heap, (best, p))
    return J


def _comp_policy_to_global(
    S: Set[int],
    comp_policy: Dict[int, Tuple],
    G: WeightedGraph,
    J_known: Dict[int, float],
    lam: float,
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, float]]:
    """Convert component policy (internal/exit) to global (u, y) pairs."""
    mu: Dict[int, Tuple[int, int]] = {}
    J: Dict[int, float] = {}
    for x in S:
        act = comp_policy.get(x)
        if act is None:
            continue
        if act[0] == "exit":
            # pick minimizing exit arc
            best = INF
            bu, by = 0, x
            for u, y, w in G.by_state.get(x, []):
                if y not in S:
                    val = w + lam * J_known.get(y, INF)
                    if val < best - 1e-12:
                        best = val
                        bu, by = u, y
            mu[x] = (bu, by)
            J[x] = comp_policy.get(x, best) if isinstance(comp_policy.get(x), float) else best
        else:
            _t, u, y = act
            mu[x] = (u, y)
    return mu, J


def master(
    G: WeightedGraph,
    xe: int,
    lam: float,
    f: Optional[Callable[[int, int], int]] = None,
    ue: Optional[int] = None,
    log: Optional[List[Tuple[int, int, bool]]] = None,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]]]:
    """Algorithm master: J_λ* on R, together with a proper feedback.

    ``J`` is the pointwise optimum of Problem discounted-feedback, that is, for
    each state separately the best a proper feedback can do there.  ``mu`` is a
    single stationary feedback, proper at every state of R; it attains ``J``
    everywhere except possibly inside a beneficial-cycle component, where no
    stationary feedback attains every pointwise optimum at once.

    ``log``, if given, collects one record per component, holding |S|, the
    number of states the tight-arc test found trapped, and the resulting
    beneficial-cycle verdict.
    """
    G.xe = xe
    R = reverse_reachable(G, xe)
    J_star: Dict[int, float] = {xe: 0.0}
    mu_star: Dict[int, Tuple[int, int]] = {}
    detect_log: List[Tuple[int, int, bool]] = [] if log is None else log
    if ue is not None:
        mu_star[xe] = (ue, xe)

    if len(R) == 1:
        return J_star, mu_star

    scc = condense(G, R, xe)

    for cid in scc.reverse_topo:
        S = scc.components[cid]
        if not S:
            continue

        is_terminal = xe in S

        if is_terminal:
            J_star[xe] = 0.0
            if ue is not None:
                mu_star[xe] = (ue, xe)
            if len(S) == 1:
                continue
            # x_e is cost free and absorbing, so it plays the role of the
            # virtual sink and the terminal component reduces to the same
            # augmented form as every other component.
            S, internal, e = terminal_to_augmented(
                S, _internal_arcs(G, S), xe, lam
            )
        else:
            internal, e = augment_component(G, S, J_star, lam)

        # Lemma detect: the exact V_lambda* and one backward BFS over the arcs
        # attaining the Bellman minimum both decide the predicate and, when it
        # is negative, solve the component.  Reading the minimizers instead of
        # the increments of an iteration is what keeps the test exact: the
        # decrease a fixed-point test would have to see is of order
        # lambda^{|S|} and underflows on large components.
        has_bc, trapped, V_S, tree_S = detect_with_in_tree(S, internal, e, lam)
        detect_log.append((len(S), len(trapped), has_bc))

        if has_bc:
            if len(S) > MAX_EXACT_CORE:
                raise RuntimeError(
                    f"Beneficial-cycle SCC |S|={len(S)} exceeds exact SolveCore "
                    f"limit {MAX_EXACT_CORE}; use the depth-parametrized route."
                )
            J_S, pol_S, _ = solve_core(S, internal, e, lam)
            assert all(
                float(J_S[x]) >= float(V_S[x]) - 1e-9 for x in S
            ), "SolveCore fell below the V_lambda* lower bound of Lemma detect(c)"
            # The stored minimizers are each state's own optimal first arc, and
            # on a beneficial-cycle component they may point at each other.  No
            # stationary feedback attains every pointwise optimum here, so when
            # the reading is trapped we fall back on an in-tree, which is proper
            # at every state at the price of leaving some above J_lambda*.
            if component_trapped_states(S, internal, e, lam, pol_S):
                _V_tree, pol_S = assemble_proper_in_tree(S, internal, e, lam)
        else:
            # Lemma detect(b): V_lambda* is already J_lambda* here, and the
            # in-tree is an optimal proper feedback.
            J_S, pol_S = V_S, tree_S

        for x, val in J_S.items():
            J_star[x] = val

        for x in S:
            act = pol_S.get(x)
            if act is None:
                continue
            if act[0] == "exit":
                best = INF
                bu, by = 0, x
                for u, y, w in G.by_state.get(x, []):
                    if y not in S:
                        val = w + lam * J_star.get(y, INF)
                        if val < best - 1e-12:
                            best = val
                            bu, by = u, y
                if best < INF:
                    mu_star[x] = (bu, by)
            else:
                _t, u, y = act
                mu_star[x] = (u, y)

    return J_star, mu_star


def undiscounted_shortest_path(
    G: WeightedGraph, xe: int, R: Optional[Set[int]] = None
) -> Dict[int, float]:
    """Reverse Dijkstra — correct for undiscounted case."""
    import heapq

    states = R if R is not None else reverse_reachable(G, xe)
    J = {x: INF for x in states}
    J[xe] = 0.0
    heap = [(0.0, xe)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > J[v]:
            continue
        for p in G.rev.get(v, []):
            if p not in states:
                continue
            for u, y, w in G.by_state.get(p, []):
                if y == v:
                    nd = w + d
                    if nd < J[p]:
                        J[p] = nd
                        heapq.heappush(heap, (nd, p))
    return J


def make_toy_dijkstra_graph() -> Tuple[WeightedGraph, int, float]:
    """Fig. dijkstra-ce: B, A, xe with lambda=0.5."""
    xe = 3  # x_e
    G = WeightedGraph(states={1, 2, 3}, xe=xe)
    # B=1, A=2, xe=3
    G.add_arc(1, 1, 3, 60.0)  # B -> xe
    G.add_arc(1, 2, 2, 1.0)   # B -> A
    G.add_arc(2, 1, 3, 100.0)  # A -> xe
    return G, xe, 0.5


def make_beneficial_cycle_graph() -> Tuple[WeightedGraph, int, float]:
    """Three-node cycle + exits, lambda=0.5, g=1 on cycle, e=100."""
    # y=1, y1=2, y2=3, omega represented as exit only (not a state in G)
    G = WeightedGraph(states={1, 2, 3})
    G.add_arc(1, 1, 2, 1.0)
    G.add_arc(2, 1, 3, 1.0)
    G.add_arc(3, 1, 1, 1.0)
    # exits: each state can "exit" via u=2 to external (we add dummy target 0 not in S)
    # For master we need xe; use xe=0 as absorbing target outside S
    # Actually for component-only test, use brute_force / solve_core directly.
    # For full graph test: add target 4 as xe
    xe = 4
    G.states.add(4)
    for s in (1, 2, 3):
        G.add_arc(s, 2, 4, 0.0)  # exit cost in g; e will add lambda*0
    # exit weight e=100 means g on exit arc should be 100
    G = WeightedGraph(states={1, 2, 3, 4}, xe=xe)
    G.add_arc(1, 1, 2, 1.0)
    G.add_arc(2, 1, 3, 1.0)
    G.add_arc(3, 1, 1, 1.0)
    for s in (1, 2, 3):
        G.add_arc(s, 2, 4, 100.0)
    G.add_arc(4, 1, 4, 0.0)
    return G, xe, 0.5


def make_discount_effect_graph() -> Tuple[WeightedGraph, int, float]:
    """
    Two paths from start s=1 to xe=4:
      P1: 1 -[50]-> 2 -[1]-> 4  (expensive early)
      P2: 1 -[1]-> 3 -[50]-> 4  (expensive late)
    """
    xe = 4
    G = WeightedGraph(states={1, 2, 3, 4}, xe=xe)
    G.add_arc(1, 1, 2, 50.0)
    G.add_arc(2, 1, 4, 1.0)
    G.add_arc(1, 2, 3, 1.0)
    G.add_arc(3, 1, 4, 50.0)
    G.add_arc(4, 1, 4, 0.0)
    return G, xe, 0.5


def random_beneficial_scc(
    n: int,
    lam: float,
    m_out: int = 3,
    seed: int = 0,
) -> Tuple[Set[int], Dict[int, List[Tuple[int, int, float]]], Dict[int, float]]:
    """Random strongly connected component with beneficial self-loops."""
    import random

    rng = random.Random(seed)
    S = set(range(1, n + 1))
    internal: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    delta = 1.0
    E = delta / (1 - lam) + 10.0  # ensure beneficial self-loop: delta < (1-lam)*E
    # random internal arcs
    for i in range(1, n + 1):
        internal[i].append((0, i, delta))  # self-loop u=0
        targets = rng.sample([j for j in S if j != i], min(m_out, n - 1))
        for j in targets:
            internal[i].append((j, j, delta + rng.random()))
    # ensure strong connectivity: chain + extra
    for i in range(1, n):
        internal[i].append((100 + i, i + 1, delta))
    e = {i: E for i in S}
    return S, dict(internal), e
