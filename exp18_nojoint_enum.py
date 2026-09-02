"""exp18: exhaustive enumeration of ALL stationary feedbacks on Example 1.

Quantitative no-joint / VI-unattainability demonstration (replaces the retired
enumeration of old 03_solvecore.tex, M23).  On the N=16, M=2 instance of
Example 1 (L taken from the first two lines of results/exp14_ex1_n4.txt) we
enumerate every one of the M^N = 2^16 = 65536 stationary feedbacks and compute
its policy value in exact rational arithmetic (cycle detection on the closed
functional graph + geometric-series cycle value + back-substitution).

Two value semantics are used and reported:
  * plain discounted closed-loop cost  W_mu(x)  (finite for every feedback);
  * stabilization value  Jt_mu(x) = W_mu(x) if the closed-loop orbit from x
    reaches x_e, and +infinity otherwise (the cost semantics of the paper's
    stabilization problem, where only proper behaviour counts).

Assertions (authorized R1 decision, 2026-09-01):
  (a) no feedback attains V_lambda* = 2 pointwise on the beneficial-cycle
      component H = {2,3,5} in the stabilization semantics; the only feedbacks
      whose PLAIN cost equals 2 on H are trapped in the cycle (improper at all
      of H), and their number is reported;
  (b) the maximum, over all 65536 feedbacks, of the number of states of V_s at
      which the stabilization value equals V_lambda* is reported, together
      with the number of feedbacks attaining that maximum;
  (c) a feedback attaining J_lambda* pointwise on V_s exists -- the feedback
      recovered by Algorithm master (exp14 output) -- and it is proper at
      every state of V_s.
Cross-checks: pointwise minima of W over all feedbacks == V table, and of Jt
over all feedbacks == J table (exact Fractions, values from exp14_ex1_n4.txt).

The script is self-contained: it does not import any protected script and
re-reads L from the exp14 results file.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

N, M, XE, UE = 16, 2, 1, 1
LAM = Fraction(1, 2)
H = (2, 3, 5)
VS = tuple(range(1, 15))  # V_s = {1..14}; 15, 16 unstabilizable


def load_L() -> list[int]:
    """Parse L (32 column-major entries) from the head of exp14_ex1_n4.txt."""
    text = (RESULTS / "exp14_ex1_n4.txt").read_text()
    m = re.search(r"L = delta_16\[([\d\s]+)\]", text)
    assert m, "L not found in exp14_ex1_n4.txt"
    cols = [int(t) for t in m.group(1).split()]
    assert len(cols) == N * M, cols
    return cols


L = load_L()


def bits(i: int) -> tuple[int, int, int, int]:
    v = N - i
    return ((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1)


D_H = {i: 4 - sum(bits(i)) for i in range(1, N + 1)}  # Hamming distance to x_e


def f(x: int, u: int) -> int:
    return L[(u - 1) * N + (x - 1)]


def g(x: int, u: int) -> Fraction:
    return Fraction(D_H[x] + (1 if u != UE else 0))


# Exact reference tables (authoritative source: results/exp14_ex1_n4.txt;
# dyadic decimals there are exact, e.g. 3.5 = 7/2, 4.6875 = 75/16).
J_STAR = {
    1: Fraction(0), 2: Fraction(7, 2), 3: Fraction(19, 8), 4: Fraction(15, 4),
    5: Fraction(11, 4), 6: Fraction(31, 8), 7: Fraction(15, 4), 8: Fraction(3),
    9: Fraction(1), 10: Fraction(7, 2), 11: Fraction(5, 2), 12: Fraction(19, 4),
    13: Fraction(27, 8), 14: Fraction(75, 16),
}
V_STAR = dict(J_STAR)
V_STAR.update({2: Fraction(2), 3: Fraction(2), 5: Fraction(2),
               4: Fraction(3), 13: Fraction(3), 14: Fraction(9, 2)})
GAP = {2, 3, 4, 5, 13, 14}

# Feedback recovered by Algorithm master (exp14 table, column "master"):
# u=2 at states 2 and 10, u=1 elsewhere on V_s; 15, 16 arbitrary (u=1 here).
MU_MASTER = tuple(2 if x in (2, 10) else 1 for x in range(1, N + 1))


def evaluate(mu: tuple[int, ...]):
    """Plain values W and the proper set of the closed loop x -> f(x, mu(x))."""
    succ = [0] * (N + 1)
    cost = [Fraction(0)] * (N + 1)
    for x in range(1, N + 1):
        succ[x] = f(x, mu[x - 1])
        cost[x] = g(x, mu[x - 1])

    W: dict[int, Fraction] = {}
    reaches: dict[int, bool] = {}
    for x0 in range(1, N + 1):
        if x0 in W:
            continue
        path: list[int] = []
        index: dict[int, int] = {}
        y = x0
        while y not in W and y not in index:
            index[y] = len(path)
            path.append(y)
            y = succ[y]
        if y in W:  # merged into an already-evaluated tree
            base, ok = W[y], reaches[y]
        else:  # new cycle discovered: path[index[y]:] is the cycle
            cyc = path[index[y]:]
            ell = len(cyc)
            g_loop = sum(LAM ** i * cost[cyc[i]] for i in range(ell))
            val = g_loop / (1 - LAM ** ell)
            ok = XE in cyc
            W[cyc[0]] = val
            reaches[cyc[0]] = ok
            for i in range(ell - 1, 0, -1):  # backwards around the cycle
                W[cyc[i]] = cost[cyc[i]] + LAM * W[cyc[(i + 1) % ell]]
                reaches[cyc[i]] = ok
            base = W[cyc[0]]
            path = path[:index[y]]
        for z in reversed(path):
            base = cost[z] + LAM * base
            W[z] = base
            reaches[z] = ok
    # x_e is absorbing under both inputs, so "orbit hits x_e" == reaches[x]
    proper = {x for x in range(1, N + 1) if reaches[x]}
    return W, proper


def run() -> None:
    # sanity: x_e is fixed under both inputs, so "reaches XE" == absorbed there
    assert f(XE, 1) == XE and f(XE, 2) == XE and g(XE, UE) == 0

    total = M ** N
    n_plain_V_on_H = 0          # plain cost == 2 pointwise on H
    plain_V_on_H_all_improper = True
    stab_V_on_H = 0             # stabilization value == 2 pointwise on H
    best_cnt = -1
    best_feedbacks = 0
    min_W = {x: None for x in range(1, N + 1)}
    min_Jt = {x: None for x in range(1, N + 1)}
    master_ok = False
    master_proper = False

    two = Fraction(2)
    for code in range(total):
        mu = tuple(1 + ((code >> (x - 1)) & 1) for x in range(1, N + 1))
        W, proper = evaluate(mu)

        # pointwise minima (cross-checks for V and J)
        for x in range(1, N + 1):
            w = W[x]
            if min_W[x] is None or w < min_W[x]:
                min_W[x] = w
            if x in proper and (min_Jt[x] is None or w < min_Jt[x]):
                min_Jt[x] = w

        # (a) V on H
        if all(W[x] == two for x in H):
            n_plain_V_on_H += 1
            if any(x in proper for x in H):
                plain_V_on_H_all_improper = False
            if all(x in proper for x in H):
                stab_V_on_H += 1

        # (b) states of V_s where stabilization value == V*
        cnt = sum(1 for x in VS if x in proper and W[x] == V_STAR[x])
        if cnt > best_cnt:
            best_cnt, best_feedbacks = cnt, 1
        elif cnt == best_cnt:
            best_feedbacks += 1

        # (c) master feedback membership
        if mu[:14] == MU_MASTER[:14]:
            if all(x in proper for x in VS) and all(
                W[x] == J_STAR[x] for x in VS
            ):
                master_ok = True
                master_proper = True

    lines = []
    lines.append("E18: exhaustive enumeration of all stationary feedbacks "
                 "on Example 1 (N=16, M=2, lambda=1/2)")
    lines.append(f"  L parsed from results/exp14_ex1_n4.txt; feedbacks "
                 f"enumerated: {total} (= 2^16); exact Fractions throughout")
    lines.append("  value semantics: W = plain discounted closed-loop cost; "
                 "Jt = W if the orbit reaches x_e, else infinity "
                 "(stabilization semantics)")
    lines.append("")

    # cross-checks
    assert all(min_W[x] == V_STAR[x] for x in VS), "min over W != V table"
    assert min_W[15] is not None and min_W[16] is not None
    assert all(min_Jt[x] == J_STAR[x] for x in VS), "min over Jt != J table"
    assert min_Jt[15] is None and min_Jt[16] is None, "15/16 must be improper"
    lines.append("  A18-0 PASS  pointwise min of W over all feedbacks == "
                 "V_lambda* table; pointwise min of Jt == J_lambda* table; "
                 "no feedback proper at 15 or 16")

    # (a)
    assert stab_V_on_H == 0, stab_V_on_H
    assert n_plain_V_on_H > 0 and plain_V_on_H_all_improper
    lines.append(f"  A18-a PASS  no feedback attains V_lambda*=2 pointwise on "
                 f"H={{2,3,5}} in stabilization semantics (count = "
                 f"{stab_V_on_H}); {n_plain_V_on_H} feedbacks attain plain "
                 f"cost 2 pointwise on H and every one is improper at all of "
                 f"H (trapped in the cycle 2->3->5->2)")

    # (b)
    n_nongap = len([x for x in VS if x not in GAP])
    assert best_cnt == n_nongap, (best_cnt, n_nongap)
    lines.append(f"  A18-b PASS  max #states of V_s where the stabilization "
                 f"value equals V_lambda*: {best_cnt} of {len(VS)} "
                 f"(exactly the non-gap states; none of the 6 gap states "
                 f"{sorted(GAP)} is attainable); feedbacks attaining the "
                 f"maximum: {best_feedbacks}")

    # (c)
    assert master_ok and master_proper
    lines.append("  A18-c PASS  the feedback recovered by Algorithm master "
                 "(u=2 at states 2 and 10, u=1 elsewhere on V_s) attains "
                 "J_lambda* pointwise on all 14 states of V_s and is proper "
                 "at every state of V_s")

    lines.append("")
    lines.append("  ALL EXP18 ASSERTIONS PASSED")
    out = "\n".join(lines) + "\n"
    (RESULTS / "exp18_nojoint.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    run()
