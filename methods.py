"""Authoritative method library for the discounted-stabilization experiments.

One implementation of Algorithm master (paper: Algorithms master/solveeasy/
core/recover) and of the literature baselines (VI, PI, Floyd-type discounted
matrix powers, simple-path DFS oracle), with every numeric decision routed
through a NumericContext:

    exact()        -- Fraction arithmetic, eps = 0, exact comparisons;
    floating(eps)  -- float64, tolerance-guarded comparisons.

Switching the context switches lambda, stage costs, the infinity sentinel,
settled / strict-decrease tests and the beneficial-cycle comparison at once;
the algorithm code is shared.

Design plan: notes/rewrite_experiments/task0_methods_plan.md.  This module
does NOT import core_solver (independent implementation, used for two-way
cross-checks); docstrings reference core_solver counterparts by name only.

FLOAT DETECTION SEMANTICS (design plan section 3; paper Remark detect-exact)
----------------------------------------------------------------------------
* Under floating(eps) the sweep-based verdict of solve_easy is a heuristic
  only: the residual a beneficial cycle leaves after k sweeps is of order
  lambda^k and can be swallowed by the tolerance, flipping the verdict.
  run_master therefore uses detect_via_pi (policy iteration + tight-arc
  backward BFS, Proposition gap-W(b)) as the primary detector under
  floating(); the sweep verdict is recorded per component and disagreements
  are counted in MasterResult.detect_disagreements.
* If lambda and all stage costs are dyadic rationals (e.g. Hamming costs with
  lambda = 1/2), float64 arithmetic is bitwise exact and the two detectors
  cannot disagree; the caveat bites only for non-dyadic data.
* Publication-grade verdicts should be recomputed under exact(); a per-
  component re-run is cheap.

C++ PORTING NOTES
-----------------
Core routines avoid Python-only constructs on the hot path: SolveCore is an
explicit bitmask layered DP, Tarjan is iterative, policy evaluation is in
closed form on the functional graph (no linear solver, no iteration
tolerance).  Each core function's docstring states the recursion/invariant,
the complexity, and which comparisons must be exact.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    Callable, Dict, FrozenSet, Iterable, List, Literal, Optional, Sequence,
    Set, Tuple, Union,
)

Num = Union[Fraction, float]
State = int    # index i of delta_N^i (1-based)
Input = int    # index k of delta_M^k (1-based)
Arc = Tuple[State, Input, State, Num]                    # (x, u, y, g)
Action = Union[Tuple[Literal["exit"], State],
               Tuple[Literal["internal"], Input, State]]
Internal = Dict[State, List[Tuple[Input, State, Num]]]
ExitW = Dict[State, Num]

INF: float = math.inf

_warned_float_into_exact = False


# ===========================================================================
# 1. Numeric layer
# ===========================================================================


@dataclass(frozen=True)
class NumericContext:
    """Scalar semantics shared by every algorithm in this module.

    name       "exact" or "float64".
    eps        0.0 under exact(); tolerance under floating(eps).
    INF        math.inf in both contexts (Fraction op inf propagates to inf;
               lambda > 0 and costs >= 0 keep 0*inf out of the pipeline).
    lt(a, b)   directional strict-improvement test used by every argmin and
               settled check.  exact: a < b.  float: a < b - eps*scale.
    close(a,b) symmetric equality test used for tightness and cross-checks.
    Comparisons branch on is_inf BEFORE any arithmetic, so inf never meets
    eps-arithmetic (inf - eps*inf would be NaN).
    """

    name: str
    eps: float
    INF: float = INF

    # -- scalar injection ---------------------------------------------------
    def num(self, v) -> Num:
        """Inject a scalar. exact: Fraction (a float argument is converted to
        its exact binary expansion, with a one-time warning); float: float."""
        global _warned_float_into_exact
        if self.name == "exact":
            if isinstance(v, Fraction):
                return v
            if isinstance(v, float):
                if not _warned_float_into_exact:
                    warnings.warn(
                        "float injected into exact(): using its exact binary "
                        "expansion; pass Fraction/str for decimal intent",
                        stacklevel=2,
                    )
                    _warned_float_into_exact = True
                return Fraction(v)
            return Fraction(v)
        return float(v)

    def lam(self, v) -> Num:
        """Inject the discount factor (audited entry point); requires 0<v<1."""
        w = self.num(v)
        assert 0 < w < 1, f"discount factor must lie in (0,1), got {w}"
        return w

    # -- predicates ----------------------------------------------------------
    def is_inf(self, v) -> bool:
        return v == INF or (self.name != "exact" and isinstance(v, float) and v >= 1e300)

    def lt(self, a, b) -> bool:
        """True iff a is a strict improvement over b (minimization)."""
        if self.is_inf(b):
            return not self.is_inf(a)
        if self.is_inf(a):
            return False
        if self.eps == 0:
            return a < b
        fa, fb = float(a), float(b)
        return fa < fb - self.eps * max(1.0, abs(fa), abs(fb))

    def close(self, a, b) -> bool:
        """Symmetric equality up to the context tolerance."""
        ia, ib = self.is_inf(a), self.is_inf(b)
        if ia or ib:
            return ia and ib
        if self.eps == 0:
            return a == b
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= self.eps * max(1.0, abs(fa), abs(fb))

    def improved_somewhere(self, A: Dict[State, Num], B: Dict[State, Num]) -> bool:
        """Whether A(x) strictly improves on B(x) at some state (settled test)."""
        return any(self.lt(v, B.get(x, INF)) for x, v in A.items())

    def as_float(self, v) -> float:
        return INF if self.is_inf(v) else float(v)


def exact() -> NumericContext:
    """Fraction arithmetic, eps = 0, exact comparisons (exp14 semantics)."""
    return NumericContext(name="exact", eps=0.0)


def floating(eps: float = 1e-9) -> NumericContext:
    """float64 arithmetic with tolerance eps for lt/close (Example 2 semantics)."""
    assert eps > 0
    return NumericContext(name="float64", eps=float(eps))


# ===========================================================================
# 2. Graph layer (decoupled from any concrete network)
# ===========================================================================


@dataclass
class STG:
    """Weighted state transition graph; the only input the pipeline sees.

    Arcs are (x, u, y, g) with g already injected through the owning context;
    ctx_name records that context and every pipeline entry point asserts it,
    so a Fraction graph can never silently meet a float lambda.
    """

    ctx_name: str
    states: Set[State] = field(default_factory=set)
    arcs: List[Arc] = field(default_factory=list)
    out: Dict[State, List[Tuple[Input, State, Num]]] = field(default_factory=dict)
    rev: Dict[State, List[State]] = field(default_factory=dict)

    def add_arc(self, x: State, u: Input, y: State, g: Num) -> None:
        self.states.add(x)
        self.states.add(y)
        self.arcs.append((x, u, y, g))
        self.out.setdefault(x, []).append((u, y, g))
        self.rev.setdefault(y, []).append(x)

    def restrict(self, R: Set[State]) -> "STG":
        """Subgraph on R (Step 1: arcs leaving R are dropped)."""
        H = STG(ctx_name=self.ctx_name, states=set(R))
        for x, u, y, g in self.arcs:
            if x in R and y in R:
                H.add_arc(x, u, y, g)
        return H


def build_stg(
    next_state: Callable[[State, Input], State],
    stage_cost: Callable[[State, Input], object],
    states: Iterable[State],
    inputs: Iterable[Input],
    ctx: NumericContext,
    Cu: Optional[Callable[[State], Iterable[Input]]] = None,
) -> STG:
    """Build the weighted STG of a BCN from callbacks (loader-agnostic)."""
    G = STG(ctx_name=ctx.name)
    inputs = list(inputs)
    for x in states:
        for u in (Cu(x) if Cu is not None else inputs):
            G.add_arc(x, u, next_state(x, u), ctx.num(stage_cost(x, u)))
    return G


def stg_from_L(
    L: Sequence[int],
    N: int,
    M: int,
    stage_cost: Callable[[State, Input], object],
    ctx: NumericContext,
) -> STG:
    """STG from a column-major transition matrix L = delta_N[...] of length N*M
    (column (u-1)*N + x holds f(x,u)); the interface Example 2's loader feeds."""
    assert len(L) == N * M, (len(L), N, M)

    def f(x: State, u: Input) -> State:
        return L[(u - 1) * N + (x - 1)]

    return build_stg(f, stage_cost, range(1, N + 1), range(1, M + 1), ctx)


def _check_ctx(G: STG, ctx: NumericContext) -> None:
    assert G.ctx_name == ctx.name, (
        f"graph built under {G.ctx_name!r} used with context {ctx.name!r}"
    )


# ===========================================================================
# 3. Pipeline layer (Algorithm master, stage by stage)
# ===========================================================================


def reverse_reachable(G: STG, xe: State) -> Set[State]:
    """Step 1: states that can reach xe (reverse BFS, O(|E|)); this is V_s."""
    R = {xe}
    dq = deque([xe])
    while dq:
        v = dq.popleft()
        for p in G.rev.get(v, []):
            if p not in R:
                R.add(p)
                dq.append(p)
    return R


@dataclass
class Condensation:
    components: List[Set[State]]
    comp_of: Dict[State, int]
    dag_edges: List[Tuple[int, int]]
    reverse_topo: List[int]      # component ids; xe's component first
    xe_comp: int


def condense(G: STG, R: Set[State], xe: State) -> Condensation:
    """Step 2: iterative Tarjan SCC on G|_R plus the condensation DAG.

    Tarjan pops a component only after everything reachable from it has been
    popped, so the pop order IS the reverse topological order Algorithm
    master needs (asserted on the DAG edges).  Complexity O(|R| + |E|).
    No numerics involved; identical under both contexts.
    """
    sub = G.restrict(R)
    adj: Dict[State, List[State]] = {
        v: sorted({y for _u, y, _w in sub.out.get(v, [])}) for v in R
    }
    index: Dict[State, int] = {}
    low: Dict[State, int] = {}
    on_stack: Set[State] = set()
    tstack: List[State] = []
    comp_of: Dict[State, int] = {}
    components: List[Set[State]] = []
    counter = 0

    for root in sorted(R):
        if root in index:
            continue
        work: List[Tuple[State, int]] = [(root, 0)]
        while work:
            v, pi_ = work[-1]
            if pi_ == 0:
                index[v] = low[v] = counter
                counter += 1
                tstack.append(v)
                on_stack.add(v)
            descended = False
            nbrs = adj.get(v, [])
            for i in range(pi_, len(nbrs)):
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
                comp: Set[State] = set()
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

    dag_edges = sorted({
        (comp_of[x], comp_of[y])
        for x, _u, y, _w in sub.arcs
        if comp_of[x] != comp_of[y]
    })
    reverse_topo = list(range(len(components)))
    for a, b in dag_edges:
        assert b < a, "Tarjan pop order violated the reverse topological order"
    return Condensation(components, comp_of, dag_edges, reverse_topo, comp_of[xe])


def fold_component(
    G: STG, S: Set[State], J_known: Dict[State, Num], lam: Num, ctx: NumericContext
) -> Tuple[Internal, ExitW]:
    """Step 3 / eq. exit-value: internal arcs of S plus exit weights
    e(x) = min over arcs leaving S of g(x,u) + lam * J(f(x,u)); +inf if none.
    Downstream values must already be in J_known (reverse topological order).
    Comparison via ctx.lt; O(M|S|)."""
    internal: Internal = defaultdict(list)
    e: ExitW = {x: INF for x in S}
    for x in S:
        for u, y, w in G.out.get(x, []):
            if y in S:
                internal[x].append((u, y, w))
            else:
                jy = J_known.get(y, INF)
                if ctx.is_inf(jy):
                    continue
                cand = w + lam * jy
                if ctx.lt(cand, e[x]):
                    e[x] = cand
    return dict(internal), e


def fold_terminal(
    G: STG, S: Set[State], xe: State, lam: Num, ctx: NumericContext
) -> Tuple[Set[State], Internal, ExitW]:
    """Terminal component: xe itself plays the role of omega (no new vertex);
    arcs into xe become exits of weight g (J(xe) = 0)."""
    S2 = {x for x in S if x != xe}
    internal: Internal = defaultdict(list)
    e: ExitW = {x: INF for x in S2}
    for x in S2:
        for u, y, w in G.out.get(x, []):
            if y == xe:
                if ctx.lt(w, e[x]):
                    e[x] = w
            elif y in S2:
                internal[x].append((u, y, w))
    return S2, dict(internal), e


def sweep_T(
    S: Set[State], internal: Internal, e: ExitW, lam: Num,
    J: Dict[State, Num], ctx: NumericContext,
) -> Tuple[Dict[State, Num], Dict[State, Action]]:
    """One SYNCHRONOUS (Jacobi) application of the operator T of eq. local-lc:

        (T J)(x) = min{ e(x),  min over internal (x,u,y):  g(x,u) + lam J(y) }.

    Updating out of place keeps the invariant that after k sweeps from J_0 = e
    the labels are the minimum discounted cost over x->omega walks with at
    most k internal arcs -- the reading Lemma lc-cap relies on.  Never replace
    this with a Gauss-Seidel (in-place) sweep: it would break the sweep-count
    semantics of solve_easy.  O(M|S|) per sweep.
    (core_solver counterpart: sweep_T.)"""
    out: Dict[State, Num] = {}
    pol: Dict[State, Action] = {}
    for x in S:
        best = e.get(x, INF)
        act: Optional[Action] = None if ctx.is_inf(best) else ("exit", x)
        for u, y, w in internal.get(x, []):
            jy = J.get(y, INF)
            if ctx.is_inf(jy):
                continue
            val = w + lam * jy
            if ctx.lt(val, best):
                best, act = val, ("internal", u, y)
        out[x] = best
        if act is not None:
            pol[x] = act
    return out, pol


@dataclass
class EasyResult:
    labels: Dict[State, Num]        # J_lambda* if not has_cycle_sweep (exact)
    policy: Dict[State, Action]
    has_cycle_sweep: bool
    sweeps: int                     # sweeps performed incl. verification sweep
    strict_sweeps: int              # sweeps that strictly improved some label


def solve_easy(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext
) -> EasyResult:
    """Algorithm solveeasy: |S|-1 synchronous sweeps plus one verification
    sweep, started from J_0 = e (Lemma lc-cap).

    exact(): the returned verdict IS the beneficial-cycle predicate, and when
    it is False the labels are J_lambda* on S.  The settled test is the exact
    vector comparison T J = J; it MUST be exact -- after k sweeps the residual
    a beneficial cycle leaves is of order lambda^k (paper Remark detect-exact).

    floating(eps): the verdict is a HEURISTIC ONLY (tolerance can swallow the
    residual and flip it); run_master uses detect_via_pi as the primary
    detector and keeps this verdict as an audit signal.

    Complexity O(M|S| D(S)) without a beneficial cycle, O(M|S|^2) with one.
    (core_solver counterpart: local_lc_paper.)"""
    J: Dict[State, Num] = {x: e.get(x, INF) for x in S}
    pol: Dict[State, Action] = {
        x: ("exit", x) for x in S if not ctx.is_inf(J[x])
    }
    sweeps = strict = 0
    for _ in range(max(0, len(S) - 1)):
        J_next, pol_next = sweep_T(S, internal, e, lam, J, ctx)
        sweeps += 1
        improved = ctx.improved_somewhere(J_next, J)
        if improved:
            strict += 1
        J, pol = J_next, pol_next
        if not improved:
            return EasyResult(J, pol, False, sweeps, strict)
    J_test, pol_test = sweep_T(S, internal, e, lam, J, ctx)
    sweeps += 1
    if ctx.improved_somewhere(J_test, J):
        return EasyResult(J, pol, True, sweeps, strict + 1)
    return EasyResult(J_test, pol_test, False, sweeps, strict)


def evaluate_policy_closed_form(
    S: Set[State], policy: Dict[State, Action], internal: Internal, e: ExitW,
    lam: Num, ctx: NumericContext,
) -> Dict[State, Num]:
    """Exact value of a stationary policy on the folded component, in closed
    form: the closed loop is a functional graph (trees hanging off cycles).

    Cycle states get the rotated lasso sum   V(v_r) = (sum_i lam^i c_{r+i}) /
    (1 - lam^ell)  (Lemma beneficial-cycle(1)), evaluated per vertex rather
    than by back-substitution -- identical under exact(), numerically stabler
    under floating().  Tree states follow by  V = c + lam V(succ).  No linear
    solve and no iteration tolerance; O(|S|) after the walk decomposition.
    (core_solver counterpart: evaluate_policy_exact.)"""
    succ: Dict[State, Optional[State]] = {}
    cost: Dict[State, Num] = {}
    for x in S:
        a = policy.get(x)
        if a is None:
            succ[x], cost[x] = None, INF
        elif a[0] == "exit":
            succ[x], cost[x] = None, e.get(x, INF)
        else:
            _t, u, y = a
            succ[x] = y
            # An action is encoded as (u, y); if the raw arc list carries
            # parallel arcs with the SAME (u, y) but different costs (a
            # well-formed BCN never does: (x, u) determines one arc), the
            # encoding collapses them to the cheapest representative, which
            # is the one every argmin scan selects.  Pricing with min keeps
            # evaluation consistent with the improvement step.
            matches = [
                w for uu, yy, w in internal.get(x, []) if uu == u and yy == y
            ]
            cost[x] = min(matches) if matches else INF

    V: Dict[State, Num] = {}
    color: Dict[State, int] = {}  # 0 unvisited, 1 on current path, 2 done
    for x0 in S:
        if color.get(x0, 0) == 2:
            continue
        path: List[State] = []
        pos: Dict[State, int] = {}
        cur: Optional[State] = x0
        while cur is not None and color.get(cur, 0) == 0:
            color[cur] = 1
            pos[cur] = len(path)
            path.append(cur)
            cur = succ[cur]
        if cur is not None and color.get(cur, 0) == 1:
            cyc = path[pos[cur]:]
            ell = len(cyc)
            if any(ctx.is_inf(cost[v]) for v in cyc):
                for v in cyc:
                    V[v], color[v] = INF, 2
            else:
                denom = 1 - lam ** ell
                for r in range(ell):
                    s = ctx.num(0)
                    disc = ctx.num(1)
                    for i in range(ell):
                        s += disc * cost[cyc[(r + i) % ell]]
                        disc *= lam
                    V[cyc[r]] = s / denom
                for v in cyc:
                    color[v] = 2
            path = path[:pos[cur]]
        for v in reversed(path):
            s = succ[v]
            if ctx.is_inf(cost[v]) or (s is not None and ctx.is_inf(V.get(s, INF))):
                V[v] = INF
            else:
                V[v] = cost[v] if s is None else cost[v] + lam * V[s]
            color[v] = 2
    return V


def _policy_iteration(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext,
    max_iter: int = 10_000,
) -> Tuple[Dict[State, Num], Dict[State, Action], int]:
    """Policy iteration for the unconstrained value V_lambda* on S u {omega}.

    Each step evaluates the current policy in closed form and rescans every
    arc once (O(M|S|)); acceptance of an improvement goes through ctx.lt, so
    it terminates in finitely many steps under both contexts (each accepted
    switch strictly lowers some value by more than the tolerance).  Unlike
    value iteration it compares POLICIES, not lambda^k-sized residuals, which
    is what makes it the float-robust detector backbone (plan section 3)."""
    policy: Dict[State, Action] = {}
    for x in S:
        if not ctx.is_inf(e.get(x, INF)):
            policy[x] = ("exit", x)
        elif internal.get(x):
            u, y, _w = internal[x][0]
            policy[x] = ("internal", u, y)
    V: Dict[State, Num] = {}
    steps = 0
    for _ in range(max_iter):
        V = evaluate_policy_closed_form(S, policy, internal, e, lam, ctx)
        steps += 1
        stable = True
        for x in S:
            cur = V.get(x, INF)
            best, best_act = cur, policy.get(x)
            ex = e.get(x, INF)
            if ctx.lt(ex, best):
                best, best_act = ex, ("exit", x)
            for u, y, w in internal.get(x, []):
                vy = V.get(y, INF)
                if ctx.is_inf(vy):
                    continue
                val = w + lam * vy
                if ctx.lt(val, best):
                    best, best_act = val, ("internal", u, y)
            if best_act is not None and ctx.lt(best, cur):
                policy[x] = best_act
                stable = False
        if stable:
            break
    return V, policy, steps


def _tight_backward_bfs(
    S: Set[State], acts: Dict[State, List[Action]]
) -> Tuple[Set[State], Dict[State, Action]]:
    """Backward BFS over the tight arcs: states admitting a tight path to
    omega, plus an in-tree.  A state enters layer k+1 through a tight action
    whose successor sits in layer k, so the recorded selection strictly
    decreases the layer index: acyclic, hence proper (Lemma recover-tree)."""
    pred: Dict[State, List[Tuple[State, Action]]] = defaultdict(list)
    good: Set[State] = set()
    tree: Dict[State, Action] = {}
    dq: deque = deque()
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


@dataclass
class DetectResult:
    has_cycle: bool
    trapped: Set[State]
    V: Dict[State, Num]             # V_lambda* on the folded component
    in_tree: Dict[State, Action]    # proper in-tree on S \\ trapped
    tight_margin: Optional[float]   # min gap non-tight vs best; None if no gap


def detect_via_pi(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext
) -> DetectResult:
    """Beneficial-cycle detector via Proposition gap-W(b): compute the exact
    fixed point V_lambda* by policy iteration, keep EVERY ctx-tight Bellman
    minimizer (the existential quantifier of the proposition), and test by
    backward BFS whether some state reaches omega through tight arcs only.
    trapped != {} iff S carries a beneficial cycle; otherwise V is already
    J_lambda* on S and in_tree an optimal proper feedback (Lemma detect).

    Primary detector under floating(): it compares argmin STRUCTURE, whose
    gaps are data-sized, instead of lambda^|S|-sized sweep residuals.
    tight_margin reports the smallest gap separating a rejected action from
    the minimum; margins near eps flag an ill-conditioned instance that
    should be re-run under exact() (plan section 3.4).
    (core_solver counterpart: detect_with_in_tree.)"""
    V, _pol, _steps = _policy_iteration(S, internal, e, lam, ctx)
    acts: Dict[State, List[Action]] = {}
    margin: Optional[float] = None
    for x in S:
        vals: List[Tuple[Num, Action]] = []
        ex = e.get(x, INF)
        if not ctx.is_inf(ex):
            vals.append((ex, ("exit", x)))
        for u, y, w in internal.get(x, []):
            vy = V.get(y, INF)
            if ctx.is_inf(vy):
                continue
            vals.append((w + lam * vy, ("internal", u, y)))
        if not vals:
            acts[x] = []
            continue
        best = min(v for v, _a in vals)
        tight = [a for v, a in vals if ctx.close(v, best)]
        for v, _a in vals:
            if not ctx.close(v, best):
                gap = ctx.as_float(v) - ctx.as_float(best)
                if margin is None or gap < margin:
                    margin = gap
        acts[x] = tight
    good, tree = _tight_backward_bfs(S, acts)
    trapped = set(S) - good
    return DetectResult(bool(trapped), trapped, V, tree, margin)


@dataclass
class CoreResult:
    J: Dict[State, Num]
    first_arc: Dict[State, Action]
    evals: int


def solve_core(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext,
    max_core: Optional[int] = None,
) -> CoreResult:
    """Algorithm core (SolveCore): Held-Karp bitmask layered DP, pure Python.

    H(v, W) = min discounted cost of a simple v->omega path confined to
    W (v in W):   H(v,{v}) = e(v);
    H(v, W) = min{ e(v),  min over internal (v,u,y), y in W\\{v}:
                            g(v,u) + lam * H(y, W\\{v}) }        (eq. held-karp)
    and J_lambda*(v) = H(v, S).  Subsets are ints (bit i = i-th state of the
    sorted list), processed in increasing popcount; first_arc[v] stores the
    arc attaining H(v, S) -- each is optimal for its OWN state; on a
    beneficial-cycle component they may close a cycle and are then not a
    joint feedback (Proposition no-joint; recover falls back on an in-tree).
    O(2^|S| |S| M) time, O(2^|S| |S|) space.  Size cap: 16 under exact()
    (Fraction constants), 20 under floating(); beyond, raise and use the
    depth-parametrized route.  (core_solver counterpart: solve_core_*.)"""
    S_list = sorted(S)
    n = len(S_list)
    cap = max_core if max_core is not None else (16 if ctx.name == "exact" else 20)
    if n > cap:
        raise RuntimeError(f"|S|={n} exceeds SolveCore cap {cap} under {ctx.name}")
    idx = {v: i for i, v in enumerate(S_list)}
    arcs_of: List[List[Tuple[int, Input, State, Num]]] = [
        [(idx[y], u, y, w) for u, y, w in internal.get(v, []) if y in idx]
        for v in S_list
    ]
    size = 1 << n
    H: List[List[Num]] = [[INF] * n for _ in range(size)]
    CH: List[List[Optional[Action]]] = [[None] * n for _ in range(size)]
    for i, v in enumerate(S_list):
        H[1 << i][i] = e.get(v, INF)
        CH[1 << i][i] = ("exit", v)
    by_pop: List[List[int]] = [[] for _ in range(n + 1)]
    for mask in range(1, size):
        by_pop[bin(mask).count("1")].append(mask)
    evals = n
    for k in range(2, n + 1):
        for mask in by_pop[k]:
            row, chrow = H[mask], CH[mask]
            for i in range(n):
                if not (mask >> i) & 1:
                    continue
                v = S_list[i]
                prev = mask ^ (1 << i)
                best: Num = e.get(v, INF)
                ch: Action = ("exit", v)
                for j, u, y, w in arcs_of[i]:
                    if not (prev >> j) & 1:
                        continue
                    sub = H[prev][j]
                    if ctx.is_inf(sub):
                        continue
                    val = w + lam * sub
                    if ctx.lt(val, best):
                        best, ch = val, ("internal", u, y)
                row[i], chrow[i] = best, ch
                evals += 1
    full = size - 1
    J = {v: H[full][i] for i, v in enumerate(S_list)}
    first_arc = {v: CH[full][i] or ("exit", v) for i, v in enumerate(S_list)}
    return CoreResult(J, first_arc, evals)


def _selection_trapped(
    S: Set[State], comp_pol: Dict[State, Action]
) -> bool:
    """Whether following the per-state first arcs loops inside S somewhere."""
    for x0 in S:
        cur, seen = x0, set()
        while True:
            a = comp_pol.get(cur)
            if a is None or a[0] == "exit":
                break
            if cur in seen:
                return True
            seen.add(cur)
            cur = a[2]
    return False


def _proper_in_tree(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext
) -> Dict[State, Action]:
    """One guaranteed-proper feedback: settle states one at a time, always
    picking the cheapest arc into the settled part (or an exit), so every
    state is strictly closer to omega than its successor -- acyclic, hence
    proper; its value may exceed J_lambda* inside a beneficial-cycle
    component, and no stationary feedback avoids that (Proposition no-joint).
    (core_solver counterpart: assemble_proper_in_tree.)"""
    V: Dict[State, Num] = {}
    tree: Dict[State, Action] = {}
    pending = set(S)
    while pending:
        pick, pick_val, pick_act = None, INF, None
        for x in pending:
            cand, act = e.get(x, INF), ("exit", x)
            for u, y, w in internal.get(x, []):
                if y in V:
                    val = w + lam * V[y]
                    if ctx.lt(val, cand):
                        cand, act = val, ("internal", u, y)
            if ctx.lt(cand, pick_val):
                pick, pick_val, pick_act = x, cand, act
        if pick is None or ctx.is_inf(pick_val):
            break
        V[pick], tree[pick] = pick_val, pick_act
        pending.discard(pick)
    return tree


def recover(
    G: STG,
    cond: Condensation,
    J: Dict[State, Num],
    has_cycle: Dict[int, bool],
    first_arcs: Dict[int, Dict[State, Action]],
    lam: Num,
    ctx: NumericContext,
    xe: State,
    ue: Optional[Input] = None,
) -> Dict[State, Tuple[Input, State]]:
    """Algorithm recover: one feedback proper on V_s from the value table.

    Components without a beneficial cycle: keep every J-tight action
    (Definition tight, decided by ctx.close) and take a backward-BFS in-tree
    to omega (Lemma recover-tree) -- optimal and proper.  Components with
    one: read SolveCore's first arcs; if they close a cycle, fall back on a
    proper in-tree (Proposition no-joint).  Component actions are then mapped
    to concrete (u, y) arcs, exits resolved to the ctx-cheapest arc leaving
    the component.  J must be complete on V_s (folding here reproduces the
    master-time exit weights because exits only point downstream).
    O(M |V_s|) plus the per-component folds."""
    _check_ctx(G, ctx)
    mu: Dict[State, Tuple[Input, State]] = {}
    if ue is not None:
        mu[xe] = (ue, xe)
    for cid, S0 in enumerate(cond.components):
        S = set(S0)
        if xe in S:
            if len(S) == 1:
                continue
            S, internal, e = fold_terminal(G, S, xe, lam, ctx)
        else:
            internal, e = fold_component(G, S, J, lam, ctx)
        if not has_cycle.get(cid, False):
            acts: Dict[State, List[Action]] = {}
            for x in S:
                tight: List[Action] = []
                if ctx.close(e.get(x, INF), J.get(x, INF)):
                    tight.append(("exit", x))
                for u, y, w in internal.get(x, []):
                    jy = J.get(y, INF)
                    if ctx.is_inf(jy):
                        continue
                    if ctx.close(w + lam * jy, J.get(x, INF)):
                        tight.append(("internal", u, y))
                acts[x] = tight
            _good, comp_pol = _tight_backward_bfs(S, acts)
        else:
            comp_pol = dict(first_arcs.get(cid, {}))
            if _selection_trapped(S, comp_pol):
                comp_pol = _proper_in_tree(S, internal, e, lam, ctx)
        for x, a in comp_pol.items():
            if a[0] == "exit":
                # Resolve the folded exit to the ctx-cheapest concrete arc
                # leaving the component (for the terminal component, arcs
                # into xe: y == xe is not in S and J(xe) = 0).
                bx, bu, by = INF, None, None
                for u, y, w in G.out.get(x, []):
                    if y in S:
                        continue
                    jy = J.get(y, INF)
                    if ctx.is_inf(jy):
                        continue
                    val = w + lam * jy
                    if ctx.lt(val, bx):
                        bx, bu, by = val, u, y
                if bu is not None:
                    mu[x] = (bu, by)
            else:
                mu[x] = (a[1], a[2])
    return mu


@dataclass
class ComponentRecord:
    S: Set[State]
    e: ExitW
    has_cycle: bool                 # the verdict master acted on
    detector: str                   # "sweep" (exact) | "pi-tight" (float)
    sweeps: int
    strict_sweeps: int
    sweep_verdict: bool             # solve_easy's verdict (audit under float)
    disagreement: bool
    tight_margin: Optional[float]   # float context only; None under exact


@dataclass
class MasterResult:
    J: Dict[State, Num]
    mu: Dict[State, Tuple[Input, State]]
    Vs: Set[State]
    records: List[ComponentRecord]
    detect_disagreements: int


def run_master(
    G: STG, xe: State, lam_in, ctx: NumericContext, ue: Optional[Input] = None
) -> MasterResult:
    """Algorithm master end to end: restriction, condensation, per-component
    folding + detection + solve, then Algorithm recover.

    Detection: exact() acts on the sweep verdict of solve_easy (Lemma lc-cap,
    theorem-backed); floating(eps) acts on detect_via_pi (plan section 3) and
    records the sweep verdict as an audit signal, counting disagreements in
    detect_disagreements.  On benign components the value written into J is
    the solve_easy fixed point under exact() and the PI fixed point under
    floating() (equal on paper; the PI value avoids settled-by-tolerance
    drift).  J(x) = +inf outside V_s is represented by absence from J."""
    _check_ctx(G, ctx)
    lam = ctx.lam(lam_in)
    R = reverse_reachable(G, xe)
    Gr = G.restrict(R)
    J: Dict[State, Num] = {xe: ctx.num(0)}
    records: List[ComponentRecord] = []
    disagreements = 0
    if len(R) == 1:
        mu = {xe: (ue, xe)} if ue is not None else {}
        return MasterResult(J, mu, R, records, 0)
    cond = condense(Gr, R, xe)
    has_cycle_by_cid: Dict[int, bool] = {}
    first_arcs_by_cid: Dict[int, Dict[State, Action]] = {}
    for cid in cond.reverse_topo:
        S = set(cond.components[cid])
        if xe in S:
            if len(S) == 1:
                continue
            S, internal, e = fold_terminal(Gr, S, xe, lam, ctx)
        else:
            internal, e = fold_component(Gr, S, J, lam, ctx)
        easy = solve_easy(S, internal, e, lam, ctx)
        if ctx.name == "exact":
            verdict, detector, margin = easy.has_cycle_sweep, "sweep", None
            benign_labels = easy.labels
        else:
            det = detect_via_pi(S, internal, e, lam, ctx)
            verdict, detector, margin = det.has_cycle, "pi-tight", det.tight_margin
            benign_labels = det.V
        disagree = verdict != easy.has_cycle_sweep
        if disagree:
            disagreements += 1
        has_cycle_by_cid[cid] = verdict
        if verdict:
            core = solve_core(S, internal, e, lam, ctx)
            Jc = core.J
            first_arcs_by_cid[cid] = core.first_arc
        else:
            Jc = benign_labels
        for x, v in Jc.items():
            if not ctx.is_inf(v):
                J[x] = v
        records.append(ComponentRecord(
            S=S, e=e, has_cycle=verdict, detector=detector,
            sweeps=easy.sweeps, strict_sweeps=easy.strict_sweeps,
            sweep_verdict=easy.has_cycle_sweep, disagreement=disagree,
            tight_margin=margin,
        ))
    mu = recover(Gr, cond, J, has_cycle_by_cid, first_arcs_by_cid, lam, ctx, xe, ue)
    return MasterResult(J, mu, R, records, disagreements)


# ===========================================================================
# 4. Baselines
# ===========================================================================


def restricted_adjacency(
    G: STG, R: Set[State]
) -> Dict[State, List[Tuple[Input, State, Num]]]:
    """Adjacency of G|_R (helper shared by the baselines and greedy readers)."""
    return {
        x: [(u, y, w) for u, y, w in G.out.get(x, []) if y in R] for x in R
    }


def vi(
    G: STG, R: Set[State], xe: State, lam_in, ctx: NumericContext, sweeps: int
) -> List[Dict[State, Num]]:
    """Synchronous Bellman iteration on G|_R with xe absorbing, started from
    V_0 = +inf, V_0(xe) = 0, so V_k is the minimum discounted cost over walks
    of at most k arcs ending at xe: the sequence is pointwise non-increasing
    and lower-bounded by V_lambda*.  Returns [V_0, ..., V_sweeps].  Converges
    to V_lambda* only asymptotically -- compare policies, not values."""
    _check_ctx(G, ctx)
    lam = ctx.lam(lam_in)
    adj = restricted_adjacency(G, R)
    V: Dict[State, Num] = {x: INF for x in R}
    V[xe] = ctx.num(0)
    seq = [dict(V)]
    for _ in range(sweeps):
        nxt: Dict[State, Num] = {}
        for x in R:
            if x == xe:
                nxt[x] = ctx.num(0)
                continue
            best: Num = INF
            for u, y, w in adj[x]:
                vy = V[y]
                if ctx.is_inf(vy):
                    continue
                val = w + lam * vy
                if ctx.lt(val, best):
                    best = val
            nxt[x] = best
        V = nxt
        seq.append(dict(V))
    return seq


def pi(
    G: STG, R: Set[State], xe: State, lam_in, ctx: NumericContext
) -> Tuple[Dict[State, Num], Dict[State, Tuple[Input, State]]]:
    """Exact policy iteration for V_lambda* on G|_R (xe absorbing): folds
    R\\{xe} with exits into xe and runs the closed-form policy iteration.
    Finitely many exact steps -- an improper minimizer it returns is not a
    tolerance artifact.  (core_solver counterpart: policy_iteration_graph.)"""
    _check_ctx(G, ctx)
    lam = ctx.lam(lam_in)
    S = {x for x in R if x != xe}
    internal: Internal = defaultdict(list)
    e: ExitW = {x: INF for x in S}
    for x in S:
        for u, y, w in G.out.get(x, []):
            if y not in R:
                continue
            if y == xe:
                if ctx.lt(w, e[x]):
                    e[x] = w
            else:
                internal[x].append((u, y, w))
    V, pol, _steps = _policy_iteration(S, dict(internal), e, lam, ctx)
    V_out: Dict[State, Num] = {xe: ctx.num(0)}
    V_out.update(V)
    mu: Dict[State, Tuple[Input, State]] = {}
    for x in S:
        a = pol.get(x)
        if a is None:
            continue
        if a[0] == "exit":
            bw, bu = INF, None
            for u, y, w in G.out.get(x, []):
                if y == xe and ctx.lt(w, bw):
                    bw, bu = w, u
            if bu is not None:
                mu[x] = (bu, xe)
        else:
            mu[x] = (a[1], a[2])
    return V_out, mu


def floyd(
    G: STG, R: Set[State], xe: State, lam_in, ctx: NumericContext, K: int
) -> List[Dict[State, Num]]:
    """Floyd-type discounted (min,+) matrix powers (Cheng/Zhu route):

        W_1[x][y] = min one-step cost x->y;
        W_{k+1}   = W_1  (+)  ( W_1  (x)_lam  W_k ),
        (A (x)_lam B)[x][y] = min_z A[x][z] + lam * B[z][y],

    so W_k[x][y] is the minimum discounted cost over walks of 1..k arcs.
    Returns the xe-column sequence [V_1, ..., V_K], V_k(xe) = 0: pointwise
    non-increasing, lower-bounded by V_lambda*, same fixed point as VI."""
    _check_ctx(G, ctx)
    lam = ctx.lam(lam_in)
    states = sorted(R)
    adj = restricted_adjacency(G, R)
    W1: Dict[State, Dict[State, Num]] = {x: {} for x in states}
    for x in states:
        for _u, y, w in adj[x]:
            if y not in W1[x] or ctx.lt(w, W1[x][y]):
                W1[x][y] = w

    def column(W: Dict[State, Dict[State, Num]]) -> Dict[State, Num]:
        col = {x: W[x].get(xe, INF) for x in states}
        col[xe] = ctx.num(0)
        return col

    Wk = {x: dict(W1[x]) for x in states}
    cols = [column(Wk)]
    for _ in range(K - 1):
        nxt: Dict[State, Dict[State, Num]] = {}
        for x in states:
            row = dict(W1[x])
            for z, wxz in W1[x].items():
                for y, wzy in Wk[z].items():
                    cand = wxz + lam * wzy
                    if y not in row or ctx.lt(cand, row[y]):
                        row[y] = cand
            nxt[x] = row
        Wk = nxt
        cols.append(column(Wk))
    return cols


def dfs_simple_paths(
    G: STG, xe: State, lam_in, ctx: NumericContext
) -> Dict[State, Num]:
    """Oracle: exact minimum discounted cost over ALL simple paths x -> xe on
    the full graph (Proposition simple-path); +inf where no path exists.
    Exponential -- small instances only."""
    _check_ctx(G, ctx)
    lam = ctx.lam(lam_in)

    def dfs(x: State, visited: Set[State]) -> Num:
        if x == xe:
            return ctx.num(0)
        best: Num = INF
        for _u, y, w in G.out.get(x, []):
            if y in visited:
                continue
            sub = dfs(y, visited | {y})
            if ctx.is_inf(sub):
                continue
            cand = w + lam * sub
            if ctx.lt(cand, best):
                best = cand
        return best

    return {x: dfs(x, {x}) for x in sorted(G.states)}


MILP_TIME_LIMIT_S = 60.0  # per-source CBC budget (Example 2 plan section 4.4)


def _milp_single_source(
    v: State,
    S_list: List[State],
    internal: Internal,
    e: ExitW,
    lam_f: float,
    ctx: NumericContext,
    time_limit: float,
) -> Tuple[str, float]:
    """One layered time-expanded MILP: min discounted cost over SIMPLE paths
    v -> omega inside the folded component (Example 2 plan section 4.4).

    Layers k = 0..K with K = |S|-1: binary z[x,a,k] picks internal arc a of x
    at step k (k < K), binary w[x,k] takes the exit of x at step k.  Flow
    conservation between layers plus "at most one departure per state over
    all layers" (the simplicity constraint that makes this an IP; its LP
    relaxation is not integral exactly on beneficial-cycle instances).
    Objective sums lambda^k-discounted arc/exit costs.  Optimal value equals
    H(v, S) of eq. held-karp; infeasibility means no simple v->omega path,
    i.e. J = +inf.  Returns (status, value)."""
    import pulp

    K = len(S_list) - 1
    prob = pulp.LpProblem(f"simple_path_{v}", pulp.LpMinimize)
    arcs: Dict[State, List[Tuple[int, Input, State, float]]] = {
        x: [(i, u, y, ctx.as_float(wg))
            for i, (u, y, wg) in enumerate(internal.get(x, []))]
        for x in S_list
    }
    exit_ok = {x for x in S_list if not ctx.is_inf(e.get(x, INF))}
    z: Dict[Tuple[State, int, int], object] = {}
    w: Dict[Tuple[State, int], object] = {}
    for k in range(K):
        for x in S_list:
            for i, _u, _y, _wf in arcs[x]:
                z[(x, i, k)] = pulp.LpVariable(f"z_{x}_{i}_{k}", cat="Binary")
    for k in range(K + 1):
        for x in exit_ok:
            w[(x, k)] = pulp.LpVariable(f"w_{x}_{k}", cat="Binary")

    def out_expr(x: State, k: int):
        terms = [z[(x, i, k)] for i, _u, _y, _wf in arcs[x] if k < K]
        if x in exit_ok:
            terms.append(w[(x, k)])
        return pulp.lpSum(terms)

    def in_expr(x: State, k: int):
        return pulp.lpSum(
            z[(xx, i, k - 1)]
            for xx in S_list for i, _u, y, _wf in arcs[xx] if y == x
        )

    prob += out_expr(v, 0) == 1
    for k in range(1, K + 1):
        for x in S_list:
            prob += in_expr(x, k) == out_expr(x, k)
    for x in S_list:
        prob += pulp.lpSum(out_expr(x, k) for k in range(K + 1)) <= 1

    disc = [lam_f ** k for k in range(K + 1)]
    obj = pulp.lpSum(
        disc[k] * wf * z[(x, i, k)]
        for k in range(K) for x in S_list for i, _u, _y, wf in arcs[x]
    ) + pulp.lpSum(
        disc[k] * ctx.as_float(e[x]) * w[(x, k)]
        for k in range(K + 1) for x in exit_ok
    )
    prob += obj
    cbc = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(cbc)
    status = pulp.LpStatus[prob.status]
    if status == "Optimal":
        return status, float(pulp.value(prob.objective))
    if status == "Infeasible":
        return status, INF
    return status, INF


def milp_oracle(
    S: Set[State], internal: Internal, e: ExitW, lam: Num, ctx: NumericContext,
    *, solver: str = "pulp",
) -> Dict[State, Num]:
    """Layered time-expanded MILP oracle on a folded component (PuLP + CBC).

    Contract: returns the values dfs_simple_paths / SolveCore return on the
    folded component, pointwise up to ctx.close.  One MILP is solved per
    source state (formulation: _milp_single_source).  Refinement adopted at
    implementation time (Example 2 plan section 10): a source whose solve
    ends neither Optimal nor Infeasible within MILP_TIME_LIMIT_S seconds is
    OMITTED from the returned dict (mirroring "absent = uncertified"), so
    callers can report certification coverage; Infeasible is a definite
    answer and is returned as +inf."""
    if solver != "pulp":
        raise ValueError(f"unsupported solver {solver!r}; only 'pulp' (CBC)")
    lam_f = ctx.as_float(lam)
    S_list = sorted(S)
    if len(S_list) == 1:
        v = S_list[0]
        return {v: e.get(v, INF)}
    out: Dict[State, Num] = {}
    for v in S_list:
        status, val = _milp_single_source(
            v, S_list, internal, e, lam_f, ctx, MILP_TIME_LIMIT_S
        )
        if status in ("Optimal", "Infeasible"):
            if ctx.is_inf(val):
                out[v] = INF
            elif ctx.name == "exact":
                # deliberate float -> Fraction: CBC works in float64
                out[v] = Fraction(val)
            else:
                out[v] = ctx.num(val)
    return out


# ===========================================================================
# 5. Utilities
# ===========================================================================


def greedy_actions(
    V: Dict[State, Num],
    adjR: Dict[State, List[Tuple[Input, State, Num]]],
    xe: State,
    lam: Num,
    ctx: NumericContext,
) -> Dict[State, FrozenSet[Input]]:
    """All ctx-tight Bellman argmin inputs at each state of adjR \\ {xe}."""
    acts: Dict[State, FrozenSet[Input]] = {}
    for x in adjR:
        if x == xe:
            continue
        vals: List[Tuple[Num, Input]] = []
        for u, y, w in adjR[x]:
            vy = V.get(y, INF)
            if ctx.is_inf(vy):
                continue
            vals.append((w + lam * vy, u))
        if not vals:
            acts[x] = frozenset()
            continue
        best = min(v for v, _u in vals)
        acts[x] = frozenset(u for v, u in vals if ctx.close(v, best))
    return acts


def greedy_mu(
    V: Dict[State, Num],
    adjR: Dict[State, List[Tuple[Input, State, Num]]],
    xe: State,
    lam: Num,
    ctx: NumericContext,
    next_state: Callable[[State, Input], State],
    ue: Optional[Input] = None,
) -> Dict[State, Tuple[Input, State]]:
    """One greedy policy from V (smallest minimizing input as tie-break)."""
    mu: Dict[State, Tuple[Input, State]] = {}
    if ue is not None:
        mu[xe] = (ue, xe)
    for x, us in greedy_actions(V, adjR, xe, lam, ctx).items():
        if us:
            u = min(us)
            mu[x] = (u, next_state(x, u))
    return mu


def simulate_closed_loop(
    next_state: Callable[[State, Input], State],
    mu: Dict[State, Tuple[Input, State]],
    x0: State,
    xe: State,
    max_steps: int = 10_000,
) -> Tuple[bool, int, List[State]]:
    """Follow the closed loop; (proper, steps to xe, trajectory)."""
    traj = [x0]
    cur = x0
    seen: Set[State] = set()
    for t in range(max_steps):
        if cur == xe:
            return True, t, traj
        if cur in seen or cur not in mu:
            return False, t, traj
        seen.add(cur)
        u, _y = mu[cur]
        cur = next_state(cur, u)
        traj.append(cur)
    return False, max_steps, traj


def improper_states(
    next_state: Callable[[State, Input], State],
    mu: Dict[State, Tuple[Input, State]],
    states: Set[State],
    xe: State,
) -> Set[State]:
    """States whose closed loop under mu never reaches xe."""
    bad: Set[State] = set()
    for x in states:
        if x == xe:
            continue
        proper, _t, _traj = simulate_closed_loop(
            next_state, mu, x, xe, max_steps=4 * max(len(states), 4)
        )
        if not proper:
            bad.add(x)
    return bad


def closed_loop_cost(
    next_state: Callable[[State, Input], State],
    stage_cost: Callable[[State, Input], object],
    mu: Dict[State, Tuple[Input, State]],
    x: State,
    xe: State,
    lam_in,
    ctx: NumericContext,
) -> Num:
    """Discounted cost of the closed loop from x; +inf if it never reaches xe."""
    lam = ctx.lam(lam_in)
    total, disc, cur, seen = ctx.num(0), ctx.num(1), x, set()
    while cur != xe:
        if cur in seen or cur not in mu:
            return INF
        seen.add(cur)
        u, _y = mu[cur]
        total += disc * ctx.num(stage_cost(cur, u))
        disc *= lam
        cur = next_state(cur, u)
    return total


def bisect_rational_threshold(
    predicate: Callable[[Fraction], bool],
    lo: Fraction,
    hi: Fraction,
    ctx: NumericContext,
    iters: int = 60,
) -> Tuple[Fraction, Fraction]:
    """Rational bisection for threshold analyses (e.g. the critical discount
    factor lambda*).  EXACT CONTEXT ONLY: near the root the predicate flips on
    differences far below any float tolerance, so a floating() call is
    refused outright (plan section 7.5) rather than silently unreliable.
    predicate must be True at lo and False at hi; returns the final bracket."""
    if ctx.name != "exact":
        raise ValueError(
            "threshold analysis requires the exact context; "
            "floating() verdicts are unreliable near the root"
        )
    lo, hi = Fraction(lo), Fraction(hi)
    assert predicate(lo) and not predicate(hi), "bracket does not straddle"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi
