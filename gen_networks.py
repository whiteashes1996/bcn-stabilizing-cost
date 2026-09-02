"""Rule-level generators for the Example 2 real BCN datasets (Step 2 plan §5.2).

For every network the Boolean update rules are transcribed VERBATIM from the
cited literature (table / equation numbers in the comments), the transition
matrix L is emitted under the Gao bit convention (ara_io.inverse_map), and a
battery of literature anchors is asserted before anything is written to
simulations/networks/.  Dual-source networks (ara, lambda switch, T-LGL) are
additionally diffed entry-by-entry against an independently distributed matrix.

Run:  python3 gen_networks.py        (exits non-zero on any anchor failure)

Networks and sources (bib keys refer to 镇定最小代价/references.bib):
  ara operon      Jenkins & Macauley 2017 (jenkins2017araoperon), Fig. 3;
                  BCN usage Gao-Xiang-Lee 2020 ICCA (gaoshuhua2020icca).
                  Matrix already in the Gao repo; rule check via ara_rules.py.
  lambda switch   Thieffry & Thomas 1995 (thieffry1995dynamical); BCN form
                  Laschov & Margaliot 2013 SICON (margaliot2013siam2869);
                  this exact matrix: Zhu-Liu-Lu-Cao 2018 SICON, Example 2
                  (per the header of lambda_switch.py in the Gao repo).
  apoptosis       Chaves 2009 ECC (apoptosis), Section II; BCN usage with
                  TNF as input: Kobayashi & Hiraishi 2013 (J. Appl. Math. /
                  IEICE 2013), Example 1.  [bib entry to be added]
  mammalian cc    Faure, Naldi, Chaouiya & Thieffry 2006 Bioinformatics,
                  Table 1 (CycD is the input, stated in the caption); BCN
                  usage Zhang & Zhang 2015 Automatica (invertibility).
                  [bib entries to be added]
  lac operon      Veliz-Cuba & Stigler 2011 (veliz2011boolean), model H;
                  BCN usage with (Ge, Le, Lem) as inputs and 6 admissible
                  input values: Bof, Fornasini & Valcher 2015 Automatica
                  (bof2015auto21), Section 5.
  T-LGL (16+3)    Zhang et al. 2008 PNAS (pnas-TLGL-29nodes) reduced by
                  Saadatpour et al. 2011 PLoS Comput Biol (T-survial),
                  Table S3; BCN with u1->sFas, u2->Ceramide, u3->MCL1:
                  Gao, Sun, Xiang, Qin & Lee, IEEE Trans. Cybernetics
                  (DOI 10.1109/TCYB.2020.3003552; arXiv:1910.08718),
                  Table II.  [bib entry to be added]
"""
from __future__ import annotations

import os
import sys
from itertools import permutations, product
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import inverse_map  # single source of the bit convention

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORKS_DIR = os.path.join(SIM_DIR, "networks")
REPO_ROOT = os.path.normpath(os.path.join(SIM_DIR, "..", ".."))
GAO_NET_DIR = os.path.join(
    REPO_ROOT, "literature", "04_optimal_cost_stabilization",
    "Gao2020_bcn_opt_dc_repo", "code", "networks",
)
TLGL_DOWNLOAD = os.path.join(
    REPO_ROOT, "literature", "04_optimal_cost_stabilization",
    "sbcn_mmc_T_LGL_原始下载.txt",
)

Bits = Tuple[int, ...]
Step = Callable[[Bits, Bits], Bits]

anchors: List[str] = []


def anchor(tag: str, cond: bool, detail: str = "") -> None:
    assert cond, f"ANCHOR {tag} FAILED  {detail}"
    anchors.append(f"{tag} PASS  {detail}".rstrip())


# ---------------------------------------------------------------------------
# bit convention helpers (Gao): delta_N^1 = all ones, delta_N^N = all zeros
# ---------------------------------------------------------------------------

def index_of(bits: Sequence[int]) -> int:
    """Inverse of ara_io.inverse_map: 0/1 vector -> canonical index (1-based).

    Accumulative-STP rule: i = 1 + sum over zero bits of 2^(n-1-k)."""
    n = len(bits)
    i = 1
    for k, b in enumerate(bits):
        if not b:
            i += 2 ** (n - 1 - k)
    return i


def emit_L(n: int, m: int, step: Step) -> List[int]:
    """Column-major L: entry (u-1)*N + (x-1) holds f(x, u), 1-based indices."""
    N, M = 2 ** n, 2 ** m
    xbits = [tuple(inverse_map(x, n)) for x in range(1, N + 1)]
    ubits = [tuple(inverse_map(u, m)) for u in range(1, M + 1)]
    # sanity of the two helpers on this n
    for probe in (1, N // 2 + 1, N):
        assert index_of(inverse_map(probe, n)) == probe, probe
    L: List[int] = [0] * (N * M)
    for u in range(1, M + 1):
        ub = ubits[u - 1]
        base = (u - 1) * N
        for x in range(1, N + 1):
            L[base + (x - 1)] = index_of(step(xbits[x - 1], ub))
    return L


def synchronous_attractors(step1: Callable[[Bits], Bits], n: int) -> List[List[Bits]]:
    """All attractors (cycles incl. fixpoints) of an autonomous map on {0,1}^n."""
    N = 2 ** n
    nxt = {}
    for x in range(1, N + 1):
        b = tuple(inverse_map(x, n))
        nxt[x] = index_of(step1(b))
    color: Dict[int, int] = {}
    cycles: List[List[int]] = []
    for x0 in range(1, N + 1):
        if x0 in color:
            continue
        path, pos = [], {}
        cur = x0
        while cur not in color and cur not in pos:
            pos[cur] = len(path)
            path.append(cur)
            cur = nxt[cur]
        if cur in pos:  # new cycle found
            cycles.append(path[pos[cur]:])
        for v in path:
            color[v] = 1
    return [[tuple(inverse_map(v, n)) for v in cyc] for cyc in cycles]


def read_network_flex(path: str) -> Tuple[int, int, List[int], int]:
    """Read 3-line (n/m/L) or 4-line (n/m/z/L) network files, skipping '#'."""
    lines: List[str] = []
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    n, m = int(lines[0]), int(lines[1])
    if len(lines) == 3:
        return n, m, [int(t) for t in lines[2].split()], 1
    z = int(lines[2])
    L = [int(t) for t in lines[3].split()]
    return n, m, L, z


def write_network(
    fname: str,
    *,
    name: str,
    n: int,
    m: int,
    L: List[int],
    node_names: Sequence[str],
    input_names: Sequence[str],
    origin: str,
    bcn_usage: str,
    valid_inputs: Sequence[int] | None = None,
    extra: Iterable[str] = (),
) -> str:
    os.makedirs(NETWORKS_DIR, exist_ok=True)
    path = os.path.join(NETWORKS_DIR, fname)
    N, M = 2 ** n, 2 ** m
    assert len(L) == N * M and len(node_names) == n and len(input_names) == m
    head = [
        f"# name: {name}",
        f"# origin: {origin}",
        f"# bcn_usage: {bcn_usage}",
        f"# nodes: {' '.join(node_names)}",
        f"# inputs: {' '.join(input_names)}",
    ]
    if valid_inputs is not None:
        head.append(f"# valid_inputs: {' '.join(str(v) for v in valid_inputs)}")
    head.append("# generated_by: gen_networks.py (bit convention: ara_io.inverse_map)")
    head.extend(f"# {line}" for line in extra)
    with open(path, "w") as fh:
        fh.write("\n".join(head) + "\n")
        fh.write(f"{n}\n{m}\n")
        fh.write(" ".join(str(v) for v in L) + "\n")
    return path


# ===========================================================================
# N1  ara operon  (rule check delegated to ara_rules.verify, dual-source)
# ===========================================================================

def gen_ara() -> None:
    import ara_rules
    res = ara_rules.verify()  # asserts all 8192 transitions vs Jenkins-Macauley
    anchor("A-ara-rules", res["checked"] == 8192 * 9 // 9 * 9,
           f"matrix == rules at all 8192 (x,u), nodes={res['nodes']}")


# ===========================================================================
# N2  lambda switch  (Thieffry-Thomas 1995; BCN form Laschov-Margaliot 2013;
#     matrix from Zhu et al. 2018 Example 2 via the Gao repo)
#     Rules as printed in the pinning-control transcription (ADE 2016, eq. 24;
#     same equations in JFI 2020, Example 4.1):
#       cI'   = ~cro & (cI | cII)
#       cro'  = ~cI & ~cII
#       cII'  = ~cI & (N | cIII) & u
#       cIII' = ~cI & N & u
#       N'    = ~cI & ~cro
# ===========================================================================

LAMBDA_RULES: Dict[str, Callable[..., bool]] = {
    "cI":   lambda v, u: (not v["cro"]) and (v["cI"] or v["cII"]),
    "cro":  lambda v, u: (not v["cI"]) and (not v["cII"]),
    "cII":  lambda v, u: (not v["cI"]) and (v["N"] or v["cIII"]) and u,
    "cIII": lambda v, u: (not v["cI"]) and v["N"] and u,
    "N":    lambda v, u: (not v["cI"]) and (not v["cro"]),
}


def gen_lambda() -> List[str]:
    n, m, L_file, z = read_network_flex(os.path.join(GAO_NET_DIR, "lambda_switch.txt"))
    anchor("A-lam-shape", (n, m, z) == (5, 1, 1), f"n={n} m={m}")
    names = list(LAMBDA_RULES)
    matches: List[Tuple[str, ...]] = []
    for order in permutations(names):
        def step(xb: Bits, ub: Bits, _o=order) -> Bits:
            v = dict(zip(_o, xb))
            return tuple(int(bool(LAMBDA_RULES[nm](v, ub[0]))) for nm in _o)
        if emit_L(5, 1, step) == L_file:
            matches.append(order)
    anchor("A-lam-diff", len(matches) == 1,
           f"unique node order matching the Zhu/Gao matrix: {matches}")
    order = matches[0]
    # decode the two control fixpoints biologically (E6 found xe in {24, 31})
    fx24 = dict(zip(order, inverse_map(24, 5)))
    fx31 = dict(zip(order, inverse_map(31, 5)))
    anchor("A-lam-lysogeny", fx24["cI"] == 1 and fx24["cro"] == 0,
           f"delta_32^24 is the lysogenic profile under order {order}")
    anchor("A-lam-lysis", fx31["cro"] == 1 and fx31["cI"] == 0,
           f"delta_32^31 is on the lytic side")
    return list(order)


# ===========================================================================
# N3  apoptosis (Chaves 2009 ECC, Section II; input u = TNF = X0):
#       IAP' = ~C3a & u ;  C3a' = ~IAP & C8a ;  C8a' = C3a | u
# ===========================================================================

APOP_NODES = ["IAP", "C3a", "C8a"]


def apop_step(xb: Bits, ub: Bits) -> Bits:
    iap, c3a, c8a = xb
    u = ub[0]
    return (int((not c3a) and u), int((not iap) and c8a), int(c3a or u))


def gen_apoptosis() -> str:
    L = emit_L(3, 1, apop_step)
    # Chaves 2009: (0,1,1) is a steady state under BOTH inputs; (0,0,0) under
    # u=0 only; (1,0,1) under u=1 only.  (Section II, Fig. 1a/b discussion.)
    def fps(u: int) -> Set[Bits]:
        out = set()
        for x in range(1, 9):
            b = tuple(inverse_map(x, 3))
            if apop_step(b, (u,)) == b:
                out.add(b)
        return out
    anchor("A-apop-fps0", fps(0) == {(0, 1, 1), (0, 0, 0)}, f"{sorted(fps(0))}")
    anchor("A-apop-fps1", fps(1) == {(0, 1, 1), (1, 0, 1)}, f"{sorted(fps(1))}")
    return write_network(
        "apoptosis.txt", name="apoptosis", n=3, m=1, L=L,
        node_names=APOP_NODES, input_names=["TNF"],
        origin="Chaves 2009 ECC (bib: apoptosis), Section II",
        bcn_usage="Kobayashi & Hiraishi 2013, Example 1 (TNF as control input)",
        extra=["cell death: (IAP,C3a,C8a)=(0,1,1); survival: (0,0,0)@u=0, (1,0,1)@u=1"],
    )


# ===========================================================================
# N4  mammalian cell cycle (Faure et al. 2006, Table 1; input CycD)
#     state order: Rb E2F CycE CycA p27 Cdc20 Cdh1 UbcH10 CycB
# ===========================================================================

MCC_NODES = ["Rb", "E2F", "CycE", "CycA", "p27", "Cdc20", "Cdh1", "UbcH10", "CycB"]


def mcc_step(xb: Bits, ub: Bits) -> Bits:
    Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, Ubc, CycB = xb
    CycD = ub[0]
    return (
        int((not CycD and not CycE and not CycA and not CycB)
            or (p27 and not CycD and not CycB)),                       # Rb
        int((not Rb and not CycA and not CycB)
            or (p27 and not Rb and not CycB)),                         # E2F
        int(E2F and not Rb),                                           # CycE
        int((E2F and not Rb and not Cdc20 and not (Cdh1 and Ubc))
            or (CycA and not Rb and not Cdc20 and not (Cdh1 and Ubc))),  # CycA
        int((not CycD and not CycE and not CycA and not CycB)
            or (p27 and not (CycE and CycA) and not CycB and not CycD)),  # p27
        int(CycB),                                                     # Cdc20
        int((not CycA and not CycB) or Cdc20 or (p27 and not CycB)),   # Cdh1
        int((not Cdh1) or (Cdh1 and Ubc and (Cdc20 or CycA or CycB))),  # UbcH10
        int(not Cdc20 and not Cdh1),                                   # CycB
    )


G0_BITS = (1, 0, 0, 0, 1, 0, 1, 0, 0)  # Rb, p27, Cdh1 active (Faure 2006, Results)


def gen_mcc() -> str:
    L = emit_L(9, 1, mcc_step)
    att0 = synchronous_attractors(lambda b: mcc_step(b, (0,)), 9)
    att1 = synchronous_attractors(lambda b: mcc_step(b, (1,)), 9)
    # Faure 2006 (synchronous): CycD=0 -> single stable state G0 (Rb, p27,
    # Cdh1 ON), reached from every state; CycD=1 -> single cyclic attractor
    # of seven states.
    anchor("A-mcc-g0", att0 == [[G0_BITS]] or (len(att0) == 1 and att0[0] == [G0_BITS]),
           f"CycD=0 attractors: {[len(a) for a in att0]}")
    anchor("A-mcc-cycle7", len(att1) == 1 and len(att1[0]) == 7,
           f"CycD=1 attractors: {[len(a) for a in att1]}")
    return write_network(
        "mammalian_cc.txt", name="mammalian_cc", n=9, m=1, L=L,
        node_names=MCC_NODES, input_names=["CycD"],
        origin="Faure, Naldi, Chaouiya & Thieffry 2006 Bioinformatics, Table 1",
        bcn_usage="Zhang & Zhang 2015 Automatica (CycD as the control input node)",
        extra=[f"G0 quiescent fixpoint under CycD=0: {''.join(map(str, G0_BITS))}"
               " (Rb, p27, Cdh1 active)"],
    )


# ===========================================================================
# N5  lac operon (Veliz-Cuba & Stigler 2011, model H; Bof et al. 2015 eqs)
#     state order: M P B C R Rm A Am L Lm ; inputs Ge Le Lem
#     invalid input pair (Le,Lem) = (1,0)
# ===========================================================================

LAC_NODES = ["M", "P", "B", "C", "R", "Rm", "A", "Am", "L", "Lm"]
LAC_INPUTS = ["Ge", "Le", "Lem"]


def lac_step(xb: Bits, ub: Bits) -> Bits:
    M, P, B, C, R, Rm, A, Am, L_, Lm = xb
    Ge, Le, Lem = ub
    return (
        int(C and not R and not Rm),            # M
        int(M),                                  # P
        int(M),                                  # B
        int(not Ge),                             # C
        int(not A and not Am),                   # R
        int((not A and not Am) or R),            # Rm
        int(L_ and B),                           # A
        int(L_ or Lm),                           # Am
        int(not Ge and P and Le),                # L
        int(not Ge and ((Lem and P) or Le)),     # Lm
    )


def lac_valid_inputs() -> List[int]:
    out = []
    for u in range(1, 9):
        Ge, Le, Lem = inverse_map(u, 3)
        if not (Le == 1 and Lem == 0):
            out.append(u)
    return out


def gen_lac() -> str:
    L = emit_L(10, 3, lac_step)
    valid_u = lac_valid_inputs()
    anchor("A-lac-validU", len(valid_u) == 6, f"valid inputs {valid_u}")
    # valid-state forward invariance: no (R,Rm)/(A,Am)/(L,Lm) pair ever
    # reaches (1,0) from a valid state under a valid input
    def state_ok(b: Bits) -> bool:
        M, P, B, C, R, Rm, A, Am, L_, Lm = b
        return not ((R and not Rm) or (A and not Am) or (L_ and not Lm))
    n_valid = 0
    for x in range(1, 1025):
        b = tuple(inverse_map(x, 10))
        if not state_ok(b):
            continue
        n_valid += 1
        for u in valid_u:
            nb = lac_step(b, tuple(inverse_map(u, 3)))
            assert state_ok(nb), (b, u, nb)
    anchor("A-lac-invariant", n_valid == 432,
           f"{n_valid} valid states closed under the 6 valid inputs")
    # fixpoint census per valid input (Veliz-Cuba & Stigler bistability:
    # exactly one valid input -- no glucose, medium lactose -- has TWO
    # fixpoints, ON and OFF; every other valid input has exactly one)
    census: Dict[int, List[Bits]] = {}
    for u in valid_u:
        ub = tuple(inverse_map(u, 3))
        fps = []
        for x in range(1, 1025):
            b = tuple(inverse_map(x, 10))
            if state_ok(b) and lac_step(b, ub) == b:
                fps.append(b)
        census[u] = fps
    two = [u for u, f in census.items() if len(f) == 2]
    anchor("A-lac-bistable-count",
           len(two) == 1 and all(len(census[u]) == 1 for u in valid_u if u not in two),
           f"fixpoints per valid input: { {u: len(f) for u, f in census.items()} }")
    ub_two = tuple(inverse_map(two[0], 3))
    anchor("A-lac-bistable-input", ub_two == (0, 0, 1),
           f"bistable input decodes to (Ge,Le,Lem)={ub_two} (no glucose, medium lactose)")
    on_off = {b[:3] for b in census[two[0]]}
    anchor("A-lac-onoff", on_off == {(1, 1, 1), (0, 0, 0)},
           "the two fixpoints are ON (M=P=B=1) and OFF (M=P=B=0)")
    return write_network(
        "lac_operon.txt", name="lac_operon", n=10, m=3, L=L,
        node_names=LAC_NODES, input_names=LAC_INPUTS,
        valid_inputs=valid_u,
        origin="Veliz-Cuba & Stigler 2011 JCB (bib: veliz2011boolean), model H",
        bcn_usage="Bof, Fornasini & Valcher 2015 Automatica (bib: bof2015auto21)",
        extra=["(Le,Lem)=(1,0) inadmissible -> 6 valid inputs of 8",
               "(R,Rm),(A,Am),(L,Lm) != (1,0) -> 432 valid states (forward invariant)"],
    )


# ===========================================================================
# N6  T-LGL 16-node reduced network (Gao et al. TCYB, Table II;
#     rules from Saadatpour et al. 2011 Table S3; u1->sFas, u2->Ceramide,
#     u3->MCL1).  State order = Table II row order.
# ===========================================================================

TLGL_NODES = ["CREB", "IFNG", "P2", "GPCR", "SMAD", "Fas", "sFas", "Ceramide",
              "DISC", "Caspase", "FLIP", "BID", "IAP", "MCL1", "S1P", "Apoptosis"]
TLGL_INPUTS = ["u1", "u2", "u3"]


def tlgl_step(xb: Bits, ub: Bits) -> Bits:
    (CREB, IFNG, P2, GPCR, SMAD, Fas, sFas, Cer,
     DISC, Casp, FLIP, BID, IAP, MCL1, S1P, Apo) = xb
    u1, u2, u3 = ub
    return (
        int(IFNG and not Apo),                              # CREB
        int(not (SMAD or P2 or Apo)),                       # IFNG
        int((IFNG or P2) and not Apo),                      # P2
        int(S1P and not Apo),                               # GPCR
        int(GPCR and not Apo),                              # SMAD
        int(not (sFas or Apo)),                             # Fas
        int(S1P and not Apo and u1),                        # sFas
        int((Fas and not (S1P or Apo)) or u2),              # Ceramide
        int((Cer or (Fas and not FLIP)) and not Apo),       # DISC
        int(((BID and not IAP) or DISC) and not Apo),       # Caspase
        int(not (DISC or Apo)),                             # FLIP
        int(not (MCL1 or Apo)),                             # BID
        int(not (BID or Apo)),                              # IAP
        int(not (DISC or Apo) and u3),                      # MCL1
        int(not (Cer or Apo)),                              # S1P
        int(Casp or Apo),                                   # Apoptosis
    )


def gen_tlgl() -> str:
    nf, mf, L_file, z = read_network_flex(TLGL_DOWNLOAD)
    anchor("A-tlgl-shape", (nf, mf, z) == (16, 3, 1), f"n={nf} m={mf} z={z}")
    L = emit_L(16, 3, tlgl_step)
    anchor("A-tlgl-diff", L == L_file,
           "rule-generated L identical to the sbcn_mmc download (524288 entries)")
    xh = 65535  # healthy state: Apoptosis on, all else off (Gao Sec. VII)
    hb = tuple(inverse_map(xh, 16))
    anchor("A-tlgl-xh-bits", hb == tuple([0] * 15 + [1]), f"{hb}")
    for u in range(1, 9):
        ub = tuple(inverse_map(u, 3))
        fixed = tlgl_step(hb, ub) == hb
        anchor(f"A-tlgl-xh-u{u}", fixed == (ub[1] == 0),
               f"(u1,u2,u3)={ub}: xh {'fixed' if fixed else 'not fixed'}")
    x0 = 58834  # diseased initial state of Gao Sec. VII
    b0 = dict(zip(TLGL_NODES, inverse_map(x0, 16)))
    anchor("A-tlgl-x0", tuple(inverse_map(x0, 16)) == tuple(
        int(c) for c in "0001101000101110")
        and b0["Caspase"] == 0 and b0["Apoptosis"] == 0,
        "delta^58834 decodes to 0001101000101110 (Caspase, Apoptosis OFF)")
    return write_network(
        "t_lgl.txt", name="t_lgl", n=16, m=3, L=L,
        node_names=TLGL_NODES, input_names=TLGL_INPUTS,
        origin="Zhang et al. 2008 PNAS -> Saadatpour et al. 2011 PLoS CB Table S3",
        bcn_usage=("Gao, Sun, Xiang, Qin & Lee, IEEE TCYB "
                   "(DOI 10.1109/TCYB.2020.3003552), Table II; "
                   "u1->sFas, u2->Ceramide, u3->MCL1"),
        extra=["healthy target x_e = delta^65535 (Apoptosis on), fixed iff u2=0",
               "diseased initial state of the TCYB paper: delta^58834"],
    )


# ===========================================================================

def run() -> dict:
    gen_ara()
    lam_order = gen_lambda()
    p_apop = gen_apoptosis()
    p_mcc = gen_mcc()
    p_lac = gen_lac()
    p_tlgl = gen_tlgl()
    lines = ["gen_networks: all anchors passed", ""]
    lines += [f"  {a}" for a in anchors]
    lines += ["", f"  lambda switch node order (inferred): {lam_order}",
              f"  written: {p_apop}", f"  written: {p_mcc}",
              f"  written: {p_lac}", f"  written: {p_tlgl}"]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.join(SIM_DIR, "results"), exist_ok=True)
    with open(os.path.join(SIM_DIR, "results", "gen_networks_anchors.txt"), "w") as fh:
        fh.write(text + "\n")
    return {"anchors": len(anchors)}


if __name__ == "__main__":
    run()
