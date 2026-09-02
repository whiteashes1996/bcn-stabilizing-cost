"""Recover the nine Boolean update rules of the ara operon from its matrix L.

The network file distributed with~\\cite{gao2020} stores only the transition
matrix, so the logical form printed in the paper is recovered here and checked
against the model of Jenkins and Macauley: for each node the support is found by
flipping one variable at a time, and the restricted truth table is matched
against the published rule.  Node and input symbols follow that model, in the
order in which the matrix indexes them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import build_ara_dynamics, inverse_map  # noqa: E402

STATE_NAMES = ["A", "Am", "Ara+", "C", "E", "D", "MS", "MT", "T"]
INPUT_NAMES = ["Ae", "Aem", "Ara-", "Ge"]

# The published rules, as Python predicates over (x, u) bit tuples.
RULES = {
    "A":    (["T", "Ae"],            lambda T, Ae: T and Ae),
    "Am":   (["T", "Aem", "Ae"],     lambda T, Aem, Ae: (Aem and T) or Ae),
    "Ara+": (["Am", "A", "Ara-"],    lambda Am, A, Am_: (Am or A) and Am_),
    "C":    (["Ge"],                 lambda Ge: not Ge),
    "E":    (["MS"],                 lambda MS: MS),
    "D":    (["Ara+", "Ara-"],       lambda Ap, Am_: (not Ap) and Am_),
    "MS":   (["Ara+", "C", "D"],     lambda Ap, C, D: Ap and C and (not D)),
    "MT":   (["Ara+", "C"],          lambda Ap, C: Ap and C),
    "T":    (["MT"],                 lambda MT: MT),
}

# Printed form, for the table of Example ara.
LATEX = {
    "A":    r"$A_e \wedge T$",
    "Am":   r"$(A_{em} \wedge T) \vee A_e$",
    "Ara+": r"$(A_m \vee A) \wedge A^{-}$",
    "C":    r"$\neg G_e$",
    "E":    r"$M_S$",
    "D":    r"$\neg A^{+} \wedge A^{-}$",
    "MS":   r"$A^{+} \wedge C \wedge \neg D$",
    "MT":   r"$A^{+} \wedge C$",
    "T":    r"$M_T$",
}


def _lookup(name: str, xb, ub) -> int:
    if name in STATE_NAMES:
        return xb[STATE_NAMES.index(name)]
    return ub[INPUT_NAMES.index(name)]


def verify(path: str | None = None) -> dict:
    """Check every rule against the transition matrix at all 8192 (x, u)."""
    n, m, N, M, L, f = build_ara_dynamics(path)
    assert (n, m) == (9, 4), (n, m)

    checked = 0
    for x in range(1, N + 1):
        xb = inverse_map(x, n)
        for u in range(1, M + 1):
            ub = inverse_map(u, m)
            yb = inverse_map(f(x, u), n)
            for i, node in enumerate(STATE_NAMES):
                args, rule = RULES[node]
                want = int(bool(rule(*[_lookup(a, xb, ub) for a in args])))
                assert yb[i] == want, (
                    f"rule for {node} disagrees at x={x}, u={u}: "
                    f"matrix {yb[i]}, rule {want}"
                )
                checked += 1
    return {"checked": checked, "nodes": len(STATE_NAMES)}


if __name__ == "__main__":
    print(verify())
