"""Exact-rational checks for the uniform-optimality theorems.

Run: python simulations/verify_uniform.py
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from folded_uniform import (
    INF,
    Folded,
    bottleneck,
    enumerate_intrees,
    exact_exc_star,
    excess_of_tree,
    greedy_vstar_tree,
    held_karp,
    identity_holds,
    spanning_from_acts,
    tight_arcs,
    tree_cost,
    value_star,
)

F = Fraction
n_fail = 0
n_pass = 0


def check(name: str, ok: bool, detail: str = ""):
    global n_fail, n_pass
    if ok:
        n_pass += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        n_fail += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def two_state(lam=F(1, 2), c=F(1, 2), E=F(2)) -> Folded:
    return Folded(2, [(0, 1, c), (1, 0, c)], [E, E], lam)


def n_ring(n: int, lam=F(1, 2), c=F(1), E=F(4)) -> Folded:
    arcs = [(i, (i + 1) % n, c) for i in range(n)]
    return Folded(n, arcs, [E] * n, lam)


def five_state_ratio() -> Folded:
    # Sep 22 report: lam=1/2, Exc*=9/2, s*=83/32, ratio ≈ 1.7349
    internal = [
        (0, 3, 4),
        (0, 1, 3),
        (1, 0, F(3, 2)),
        (2, 0, 2),
        (2, 3, F(1, 8)),
        (3, 4, 2),
        (4, 1, F(3, 4)),
        (4, 2, F(1, 2)),
        (4, 3, F(1, 8)),
    ]
    e = [24, 16, INF, INF, INF]
    # state 1 also has second exit 24 in the report; keep cheaper 16
    return Folded(5, internal, e, F(1, 2))


def main():
    print("=== identity and Theorem 1 on two-state / n-ring / 5-state ===")
    for name, comp in [
        ("two-state", two_state()),
        ("3-ring", n_ring(3)),
        ("5-state", five_state_ratio()),
    ]:
        J = held_karp(comp)
        n_id = 0
        n_chk = 0
        for mu in enumerate_intrees(comp):
            for x in range(comp.n):
                n_chk += 1
                if identity_holds(comp, J, mu, x):
                    n_id += 1
        check(f"identity {name}", n_id == n_chk and n_chk > 0, f"{n_id}/{n_chk}")
        acts = tight_arcs(comp, J)
        joint = spanning_from_acts(comp, acts)
        Exc, mu_star = exact_exc_star(comp, J)
        check(f"Thm1 iff {name}", joint == (Exc == 0), f"joint={joint} Exc*={Exc}")

    print("=== two-state: Exc* = (1-λ)E-c, s*=Exc* ===")
    ts = two_state()
    J = held_karp(ts)
    Exc, _ = exact_exc_star(ts, J)
    sstar, hstar, _ = bottleneck(ts, J)
    check("two-state J*", J == [F(3, 2), F(3, 2)], str(J))
    check("two-state Exc*", Exc == F(1, 2), str(Exc))
    check("two-state s*", sstar == Exc, f"s*={sstar} h*={hstar}")

    print("=== n-ring: lower bound tight, h*=1 ===")
    for n in (2, 3, 4, 5):
        lam, c, E = F(1, 2), F(1), F(4)
        comp = n_ring(n, lam, c, E)
        J = held_karp(comp)
        A = (1 - lam ** (n - 1)) / (1 - lam)
        D = (1 - lam) * E - c
        Jexp = c * A + lam ** (n - 1) * E
        check(f"{n}-ring J*", all(j == Jexp for j in J), str(J[0]))
        Exc, _ = exact_exc_star(comp, J)
        sstar, hstar, _ = bottleneck(comp, J)
        check(f"{n}-ring Exc*=s*", Exc == sstar == A * D, f"Exc*={Exc} s*={sstar} h*={hstar}")
        check(f"{n}-ring h*=1", hstar == 1, str(hstar))

    print("=== sandwich on 5-state example ===")
    fs = five_state_ratio()
    J = held_karp(fs)
    Exc, _ = exact_exc_star(fs, J)
    sstar, hstar, tree = bottleneck(fs, J)
    ub = sstar * (1 - fs.lam ** hstar) / (1 - fs.lam) if sstar > 0 else F(0)
    check("5-state sandwich", sstar <= Exc <= ub, f"s*={sstar} Exc*={Exc} ub={ub} h*={hstar}")
    print(f"    ratio Exc*/s* = {float(Exc / sstar):.4f}  factor=(1-λ^h*)/(1-λ)={float(ub / sstar):.4f}")

    print("=== greedy V* certificate ===")
    for name, comp in [("two-state", ts), ("5-state", fs), ("4-ring", n_ring(4))]:
        J = held_karp(comp)
        mu, cert = greedy_vstar_tree(comp)
        V = value_star(comp)
        ok = True
        for x in range(comp.n):
            d = tree_cost(comp, mu, x) - J[x]
            if d > cert[x]:
                ok = False
            if comp.e[x] < INF and cert[x] > comp.e[x] - V[x]:
                ok = False
        check(f"greedy cert {name}", ok)

    print("=== end-to-end: two-state downstream + singleton upstream ===")
    # Global graph: states 0,1 as two-state with exits to ω; state 2 → 0 cost 1.
    # Folded downstream: two-state. Folded upstream: singleton with e(2)=1+λ J*(0).
    down = two_state()
    Jd = held_karp(down)
    Exc_d, _ = exact_exc_star(down, Jd)
    lam = F(1, 2)
    e2 = 1 + lam * Jd[0]
    up = Folded(1, [], [e2], lam)
    Ju = held_karp(up)
    Exc_u, _ = exact_exc_star(up, Ju)
    # Global: enumerate in-trees on {0,1,2}
    glob = Folded(3, [(0, 1, F(1, 2)), (1, 0, F(1, 2)), (2, 0, 1)], [F(2), F(2), INF], lam)
    Jg = held_karp(glob)
    Exc_g, _ = exact_exc_star(glob, Jg)
    check("upstream Exc*=0", Exc_u == 0, str(Exc_u))
    check("global J* consistency", Jg[0] == Jd[0] and Jg[2] == Ju[0], str(Jg))
    check("end-to-end lower", max(Exc_d, Exc_u) <= Exc_g, f"max_S={max(Exc_d, Exc_u)} global={Exc_g}")
    # chain bound: Exc_g ≤ Exc_d + λ Exc_u  (S_down then nothing)
    check("end-to-end upper", Exc_g <= Exc_d + lam * Exc_u, f"{Exc_g} vs {Exc_d + lam * Exc_u}")

    print("=== random sandwich (n≤4) ===")
    rng = random.Random(0)
    n_rand = 0
    max_ratio = F(0)
    max_h = 0
    for _ in range(80):
        n = rng.randint(2, 4)
        lam = F(1, 2)
        arcs = []
        for x in range(n):
            y = (x + 1) % n
            arcs.append((x, y, F(rng.randint(1, 6), rng.randint(1, 4))))
            if rng.random() < 0.5:
                z = rng.randrange(n)
                arcs.append((x, z, F(rng.randint(1, 8), rng.randint(1, 4))))
        exits = [F(rng.randint(3, 12)) if rng.random() < 0.7 else INF for _ in range(n)]
        if all(e is INF or e >= INF for e in exits):
            exits[0] = F(8)
        comp = Folded(n, arcs, exits, lam)
        try:
            J = held_karp(comp)
            if any(j >= INF / 2 for j in J):
                continue
            Exc, _ = exact_exc_star(comp, J)
            sstar, hstar, _ = bottleneck(comp, J)
        except RuntimeError:
            continue
        ub = sstar * (1 - lam ** hstar) / (1 - lam) if sstar > 0 else F(0)
        check(f"rand sandwich {n_rand}", sstar <= Exc <= ub + F(0),
              f"s*={sstar} Exc*={Exc} ub={ub}")
        n_rand += 1
        if sstar > 0:
            r = Exc / sstar
            if r > max_ratio:
                max_ratio = r
                max_h = hstar
        if n_rand >= 40:
            break
    print(f"    random max Exc*/s* = {float(max_ratio):.4f} (h*={max_h})")

    print("=== tightness search at λ=9/10, n=4 ===")
    rng = random.Random(1)
    best = (F(0), 0, None)
    factor_hits = 0
    for _ in range(120):
        n = 4
        lam = F(9, 10)
        arcs = []
        for x in range(n):
            arcs.append((x, (x + 1) % n, F(rng.randint(1, 5), rng.randint(2, 6))))
            if rng.random() < 0.6:
                arcs.append((x, rng.randrange(n), F(rng.randint(1, 8), rng.randint(2, 6))))
        exits = [F(rng.randint(4, 20), rng.randint(1, 3)) if rng.random() < 0.6 else INF for _ in range(n)]
        if all(e >= INF / 2 for e in exits):
            exits[0] = F(8)
        comp = Folded(n, arcs, exits, lam)
        try:
            J = held_karp(comp)
            if any(j >= INF / 2 for j in J):
                continue
            Exc, _ = exact_exc_star(comp, J)
            sstar, hstar, _ = bottleneck(comp, J)
        except RuntimeError:
            continue
        if sstar > 0:
            r = Exc / sstar
            fac = (1 - lam ** hstar) / (1 - lam)
            if r > best[0]:
                best = (r, hstar, fac)
            if fac - r < F(1, 50):
                factor_hits += 1
    print(f"    best ratio {float(best[0]):.4f} at h*={best[1]}, factor={float(best[2]) if best[2] else 0:.4f}; near-tight hits={factor_hits}")

    print("=== tightness note: self-loop chain is a V*-excess family, not J* ===")
    # For J*-sandwich, n-ring saturates the lower bound. Upper factor tightness
    # remains open; 5-state ratio 1.73 vs 1.94 at λ=1/2.
    check("open gap acknowledged", True, "upper sandwich factor not shown tight")

    print(f"\n{n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
