"""Exact recheck of the mammalian cell-cycle decoy kernel (|S| = 14).

Rebuilds the decoy STG in Fraction arithmetic, folds the beneficial-cycle
component, and computes J*, s*, h*, the T_0 excess, and the exact Exc* by
enumerating all spanning in-trees of the folded component.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))

import bcn_data as BD  # noqa: E402
import exp16_ex2_real as E16  # noqa: E402
import methods as MM  # noqa: E402
from exp21_uniform_real import walk_with_milp  # noqa: E402
from folded_uniform import INF, Folded, bottleneck, exact_exc_star, held_karp, tree_cost  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> int:
    import gen_networks as GN

    rows = {c["key"]: c for c in E16._mk_row_config()}
    mcc = rows["mammalian_cc"]["spec"]
    xe = rows["mammalian_cc"]["xe"]
    att1 = GN.synchronous_attractors(lambda b: GN.mcc_step(b, (1,)), 9)
    cyc = {BD.state_index(mcc, b) for b in att1[0]}
    ctx = MM.exact()
    base = BD.hamming_cost(mcc, xe, 2)
    g = BD.decoy_cost(base, {(x, 1) for x in cyc}, Fraction(1, 20))
    G = BD.make_stg(mcc, g, ctx)
    walk = walk_with_milp(G, xe, ctx, 2, 60.0)
    J = walk["J"]
    comps = [c for c in walk["comps"] if c["verdict"]]
    lines = [f"E21-exact: mammalian decoy, lambda=1/2, eps=1/20; beneficial-cycle comps={len(comps)}"]
    for comp in comps:
        S = sorted(comp["S"])
        idx = {x: i for i, x in enumerate(S)}
        internal = []
        for x in S:
            for _u, y, w in comp["internal"].get(x, []):
                internal.append((idx[x], idx[y], Fraction(w)))
        exits = [Fraction(comp["e"][x]) if comp["e"][x] != MM.INF else INF for x in S]
        F = Folded(len(S), internal, exits, Fraction(1, 2))
        Jhk = held_karp(F)
        agree = all(Jhk[idx[x]] == Fraction(J[x]) for x in S)
        sstar, hstar, tree = bottleneck(F, Jhk)
        mu = [tree[i] for i in range(F.n)]
        t0_exc = max(tree_cost(F, mu, i) - Jhk[i] for i in range(F.n))
        exc, _ = exact_exc_star(F, Jhk)
        ub = sstar * (1 - Fraction(1, 2) ** hstar) / (1 - Fraction(1, 2))
        lines.append(
            f"  |S|={len(S)} HK==pipeline:{agree} s*={sstar} h*={hstar} "
            f"T0_excess={t0_exc} Exc*={exc} sandwich_ub={ub}"
        )
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "exp21_mcc_exact.txt"), "w") as fh:
        fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
