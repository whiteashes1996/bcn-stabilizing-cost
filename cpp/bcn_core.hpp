// bcn_core.hpp -- C++ port of simulations/methods.py under floating(1e-9)
// semantics.  Every function states its methods.py counterpart; the predicate
// layer (lt/close/is_inf), the Jacobi sweep reading, the pi-tight primary
// detector and the SolveCore traversal order are ports with ZERO semantic
// deviation.  Representation-level changes only: dicts -> flat arrays,
// restrict() -> in_R mask.  See notes/rewrite_experiments/step3_example3_cpp_plan.md.
//
// exact() has no C++ counterpart: the exact reference lives in Python
// (Fraction); crosschecks run through exp17_crosscheck.py.
#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bcn {

using u8 = uint8_t;
using u32 = uint32_t;
using i32 = int32_t;
using u64 = uint64_t;
using i64 = int64_t;

constexpr double INFV = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-9;  // floating(1e-9); change here => change methods.py

// ---------------------------------------------------------------------------
// Numeric layer  (methods.NumericContext, floating(1e-9))
// inf branches BEFORE any eps arithmetic, exactly as in Python.
// ---------------------------------------------------------------------------
inline bool is_inf(double v) { return v == INFV || v >= 1e300; }

inline bool lt(double a, double b) {
  if (is_inf(b)) return !is_inf(a);
  if (is_inf(a)) return false;
  return a < b - EPS * std::max({1.0, std::fabs(a), std::fabs(b)});
}

inline bool close(double a, double b) {
  const bool ia = is_inf(a), ib = is_inf(b);
  if (ia || ib) return ia && ib;
  return std::fabs(a - b) <= EPS * std::max({1.0, std::fabs(a), std::fabs(b)});
}

#ifdef __APPLE__
// mach_absolute_time reads the cycle counter directly (~5 ns per call vs
// ~20 ns for steady_clock): with five timing spans per component and mostly
// singleton components, the cheaper clock keeps the unmeasured per-component
// overhead below the 5% stage-sum budget of acceptance B9 (plan section 10).
#include <mach/mach_time.h>
inline double now_s() {
  static const double k = [] {
    mach_timebase_info_data_t ti;
    mach_timebase_info(&ti);
    return 1e-9 * (double)ti.numer / (double)ti.denom;
  }();
  return (double)mach_absolute_time() * k;
}
#else
inline double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
#endif

// ---------------------------------------------------------------------------
// Graph layer  (methods.STG / stg_from_L).  Dense (x,u) representation: a BCN
// arc is uniquely determined by (x,u), which realizes the parallel-arc
// collapse invariant of methods.py section 10.1 by construction.
// States are 0-based internally; 1-based only at I/O boundaries.
// ---------------------------------------------------------------------------
struct Graph {
  u32 N = 0, M = 0;
  std::vector<u32> nxt;      // (x*M + u) -> y
  std::vector<double> cost;  // (x*M + u) -> g(x,u), context-injected double
  std::vector<u32> roff, radj;  // reverse CSR (with multiplicity; BFS dedups)

  void build_rev() {
    roff.assign(N + 1, 0);
    for (u64 a = 0; a < (u64)N * M; ++a) roff[nxt[a] + 1]++;
    for (u32 i = 0; i < N; ++i) roff[i + 1] += roff[i];
    radj.assign((u64)N * M, 0);
    std::vector<u32> cur(roff.begin(), roff.end() - 1);
    for (u32 x = 0; x < N; ++x)
      for (u32 u = 0; u < M; ++u) radj[cur[nxt[(u64)x * M + u]]++] = x;
  }
};

// Step 1 (methods.reverse_reachable): reverse BFS from xe; returns mask+count.
inline u64 reverse_reachable(const Graph& g, u32 xe, std::vector<u8>& in_R) {
  in_R.assign(g.N, 0);
  std::vector<u32> q;
  q.reserve(1024);
  in_R[xe] = 1;
  q.push_back(xe);
  u64 cnt = 1;
  for (size_t h = 0; h < q.size(); ++h) {
    const u32 v = q[h];
    for (u32 a = g.roff[v]; a < g.roff[v + 1]; ++a) {
      const u32 p = g.radj[a];
      if (!in_R[p]) {
        in_R[p] = 1;
        ++cnt;
        q.push_back(p);
      }
    }
  }
  return cnt;
}

// ---------------------------------------------------------------------------
// Step 2 (methods.condense): iterative Tarjan on G|_R; pop order IS the
// reverse topological order (checked explicitly, release builds included).
// ---------------------------------------------------------------------------
struct Cond {
  std::vector<u32> comp_of;   // valid where in_R
  std::vector<u32> off, vtx;  // component CSR in pop order (cid asc = rev topo)
  u32 ncomp = 0;
};

inline void condense(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                     Cond& c) {
  const u32 UNSET = 0xffffffffu;
  const u32 N = g.N, M = g.M;
  std::vector<u32> index(N, UNSET), low(N, 0);
  std::vector<u8> onstack(N, 0);
  std::vector<u32> tstack;
  c.comp_of.assign(N, UNSET);
  c.off.clear();
  c.vtx.clear();
  c.off.push_back(0);
  c.ncomp = 0;
  u32 counter = 0;
  std::vector<std::pair<u32, u32>> work;  // (v, next input index)
  for (u32 root = 0; root < N; ++root) {
    if (!in_R[root] || index[root] != UNSET) continue;
    work.clear();
    work.push_back({root, 0});
    while (!work.empty()) {
      u32 v = work.back().first;
      u32 ui = work.back().second;
      if (ui == 0) {
        index[v] = low[v] = counter++;
        tstack.push_back(v);
        onstack[v] = 1;
      }
      bool descended = false;
      for (u32 u = ui; u < M; ++u) {
        const u32 w = g.nxt[(u64)v * M + u];
        if (!in_R[w]) continue;
        bool dup = false;  // dedup distinct targets (M <= 8: linear scan ok)
        for (u32 u2 = 0; u2 < u; ++u2)
          if (in_R[g.nxt[(u64)v * M + u2]] && g.nxt[(u64)v * M + u2] == w) {
            dup = true;
            break;
          }
        if (dup) continue;
        if (index[w] == UNSET) {
          work.back().second = u + 1;
          work.push_back({w, 0});
          descended = true;
          break;
        }
        if (onstack[w]) low[v] = std::min(low[v], index[w]);
      }
      if (descended) continue;
      if (low[v] == index[v]) {
        const u32 cid = c.ncomp++;
        while (true) {
          const u32 w = tstack.back();
          tstack.pop_back();
          onstack[w] = 0;
          c.comp_of[w] = cid;
          c.vtx.push_back(w);
          if (w == v) break;
        }
        c.off.push_back((u32)c.vtx.size());
      }
      work.pop_back();
      if (!work.empty())
        low[work.back().first] = std::min(low[work.back().first], low[v]);
    }
  }
  // Reverse-topological-order check on every cross-component arc (always on).
  for (u32 x = 0; x < N; ++x) {
    if (!in_R[x]) continue;
    for (u32 u = 0; u < M; ++u) {
      const u32 y = g.nxt[(u64)x * M + u];
      if (!in_R[y]) continue;
      if (c.comp_of[x] != c.comp_of[y] && !(c.comp_of[y] < c.comp_of[x]))
        throw std::runtime_error("Tarjan pop order violated reverse topo");
    }
  }
  if (c.comp_of[xe] != 0)
    throw std::runtime_error("xe component is not the first popped (cid 0)");
}

// ---------------------------------------------------------------------------
// Folded component (methods.fold_component / fold_terminal) with local
// indexing.  S is sorted ascending by global id (matches solve_core's
// sorted(S); the other routines are order-independent).
// Arc order within a state: ascending u -- identical to Python's insertion
// order in STG.out, which fold_component preserves.
// ---------------------------------------------------------------------------
struct Comp {
  std::vector<u32> S;      // sorted global ids (terminal comp: excludes xe)
  std::vector<double> e;   // exit weights, local
  std::vector<u32> aoff;   // per-local arc CSR
  std::vector<u32> a_u;    // input of arc
  std::vector<u32> a_yl;   // local target of arc
  std::vector<double> a_w; // stage cost
  bool terminal = false;

  size_t n() const { return S.size(); }
  u32 na() const { return (u32)a_u.size(); }
};

// Scratch space reused across components (allocation-free inner loop).
struct Workspace {
  std::vector<u32> pos;      // global -> local (epoch-stamped)
  std::vector<u32> pos_ep;
  u32 epoch = 0;
  explicit Workspace(u32 N) : pos(N, 0), pos_ep(N, 0) {}
  void begin(const std::vector<u32>& S) {
    ++epoch;
    for (u32 i = 0; i < (u32)S.size(); ++i) {
      pos[S[i]] = i;
      pos_ep[S[i]] = epoch;
    }
  }
  bool inS(u32 gx) const { return pos_ep[gx] == epoch; }
  u32 loc(u32 gx) const { return pos[gx]; }
};

// Fold component cid.  Non-terminal: exits e(x) = min g + lam*J(y) over arcs
// leaving S with finite J (methods.fold_component).  Terminal: xe plays
// omega, arcs into xe are exits of weight g (methods.fold_terminal).
inline void fold(const Graph& g, const std::vector<u8>& in_R, const Cond& cd,
                 u32 cid, u32 xe, double lam, const std::vector<double>& J,
                 Workspace& ws, Comp& c) {
  c.terminal = (cd.comp_of[xe] == cid);
  c.S.clear();
  for (u32 i = cd.off[cid]; i < cd.off[cid + 1]; ++i) {
    const u32 v = cd.vtx[i];
    if (c.terminal && v == xe) continue;
    c.S.push_back(v);
  }
  std::sort(c.S.begin(), c.S.end());
  ws.begin(c.S);
  const size_t n = c.S.size();
  c.e.assign(n, INFV);
  c.aoff.assign(n + 1, 0);
  c.a_u.clear();
  c.a_yl.clear();
  c.a_w.clear();
  for (size_t i = 0; i < n; ++i) {
    const u32 x = c.S[i];
    for (u32 u = 0; u < g.M; ++u) {
      const u32 y = g.nxt[(u64)x * g.M + u];
      const double w = g.cost[(u64)x * g.M + u];
      if (!in_R[y]) continue;  // restrict() semantics
      if (c.terminal) {
        if (y == xe) {
          if (lt(w, c.e[i])) c.e[i] = w;
        } else if (ws.inS(y)) {
          c.a_u.push_back(u);
          c.a_yl.push_back(ws.loc(y));
          c.a_w.push_back(w);
        }
        // terminal comp has no arcs into other comps (unique sink, cid 0)
      } else {
        if (ws.inS(y)) {
          c.a_u.push_back(u);
          c.a_yl.push_back(ws.loc(y));
          c.a_w.push_back(w);
        } else {
          const double jy = J[y];
          if (is_inf(jy)) continue;
          const double cand = w + lam * jy;
          if (lt(cand, c.e[i])) c.e[i] = cand;
        }
      }
    }
    c.aoff[i + 1] = (u32)c.a_u.size();
  }
}

// ---------------------------------------------------------------------------
// One synchronous (Jacobi) sweep of eq. local-lc (methods.sweep_T).
// RED LINE: out-of-place double-buffering; NEVER make this Gauss-Seidel --
// it would break the sweep-count semantics of solve_easy (Lemma lc-cap).
// pol encoding: -2 none, -1 exit, >=0 index into the component arc arrays.
// ---------------------------------------------------------------------------
inline void sweep_T(const Comp& c, double lam, const std::vector<double>& J,
                    std::vector<double>& Jn, std::vector<int>& poln) {
  const size_t n = c.n();
  for (size_t i = 0; i < n; ++i) {
    double best = c.e[i];
    int act = is_inf(best) ? -2 : -1;
    for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
      const double jy = J[c.a_yl[a]];
      if (is_inf(jy)) continue;
      const double val = c.a_w[a] + lam * jy;
      if (lt(val, best)) {
        best = val;
        act = (int)a;
      }
    }
    Jn[i] = best;
    poln[i] = act;
  }
}

inline bool improved_somewhere(const std::vector<double>& A,
                               const std::vector<double>& B) {
  for (size_t i = 0; i < A.size(); ++i)
    if (lt(A[i], B[i])) return true;
  return false;
}

// Algorithm solveeasy (methods.solve_easy): |S|-1 sweeps + 1 verification
// sweep from J_0 = e.  Under floating() the verdict is an AUDIT SIGNAL only;
// run_master acts on detect_via_pi (below).
struct EasyOut {
  std::vector<double> J;
  std::vector<int> pol;
  bool has_cycle_sweep = false;
  u32 sweeps = 0, strict_sweeps = 0;
};

inline void solve_easy(const Comp& c, double lam, EasyOut& out,
                       std::vector<double>& Jn, std::vector<int>& poln) {
  const size_t n = c.n();
  out.J.assign(c.e.begin(), c.e.end());
  out.pol.assign(n, -2);
  for (size_t i = 0; i < n; ++i)
    if (!is_inf(out.J[i])) out.pol[i] = -1;
  out.sweeps = out.strict_sweeps = 0;
  out.has_cycle_sweep = false;
  Jn.resize(n);
  poln.resize(n);
  const size_t rounds = n > 0 ? n - 1 : 0;
  for (size_t r = 0; r < rounds; ++r) {
    sweep_T(c, lam, out.J, Jn, poln);
    out.sweeps++;
    const bool improved = improved_somewhere(Jn, out.J);
    if (improved) out.strict_sweeps++;
    out.J.swap(Jn);
    out.pol.swap(poln);
    if (!improved) return;  // settled: verdict false, labels = J (adopted)
  }
  sweep_T(c, lam, out.J, Jn, poln);
  out.sweeps++;
  if (improved_somewhere(Jn, out.J)) {
    out.strict_sweeps++;
    out.has_cycle_sweep = true;  // keep pre-test J/pol (Python semantics)
    return;
  }
  out.J.swap(Jn);  // adopt J_test / pol_test (Python semantics)
  out.pol.swap(poln);
}

// ---------------------------------------------------------------------------
// Closed-form policy evaluation on the functional graph
// (methods.evaluate_policy_closed_form): cycle values by the rotated lasso
// sum / (1 - lam^ell), tree values by back-substitution.  For ell <= 1024 the
// O(ell^2) per-rotation sums mirror Python bit for bit; longer cycles (never
// reached in crosscheck instances) use the O(ell) identity
// V(v_r) = c_r + lam V(v_{r+1}) -- documented deviation, values agree
// mathematically.
// ---------------------------------------------------------------------------
struct EvalScratch {
  std::vector<i64> succ;
  std::vector<double> cost;
  std::vector<u8> color;
  std::vector<u32> path;
  std::vector<u32> posv, pos_ep;
  u32 epoch = 0;
};

inline void eval_policy(const Comp& c, const std::vector<int>& pol, double lam,
                        std::vector<double>& V, EvalScratch& s) {
  const size_t n = c.n();
  s.succ.resize(n);
  s.cost.resize(n);
  s.color.assign(n, 0);
  if (s.posv.size() < n) {
    s.posv.assign(n, 0);
    s.pos_ep.assign(n, 0);
    s.epoch = 0;
  }
  V.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const int a = pol[i];
    if (a == -2) {
      s.succ[i] = -1;
      s.cost[i] = INFV;
    } else if (a == -1) {
      s.succ[i] = -1;
      s.cost[i] = c.e[i];
    } else {
      s.succ[i] = (i64)c.a_yl[a];
      s.cost[i] = c.a_w[a];
    }
  }
  for (size_t x0 = 0; x0 < n; ++x0) {
    if (s.color[x0] == 2) continue;
    s.path.clear();
    ++s.epoch;
    i64 cur = (i64)x0;
    while (cur != -1 && s.color[cur] == 0) {
      s.color[cur] = 1;
      s.posv[cur] = (u32)s.path.size();
      s.pos_ep[cur] = s.epoch;
      s.path.push_back((u32)cur);
      cur = s.succ[cur];
    }
    size_t cut = s.path.size();
    if (cur != -1 && s.color[cur] == 1) {  // found a cycle on this path
      const u32 start = s.posv[cur];
      const size_t ell = s.path.size() - start;
      bool anyinf = false;
      for (size_t k = start; k < s.path.size(); ++k)
        if (is_inf(s.cost[s.path[k]])) anyinf = true;
      if (anyinf) {
        for (size_t k = start; k < s.path.size(); ++k) {
          V[s.path[k]] = INFV;
          s.color[s.path[k]] = 2;
        }
      } else if (ell <= 1024) {
        const double denom = 1.0 - std::pow(lam, (double)ell);
        for (size_t r = 0; r < ell; ++r) {
          double sum = 0.0, disc = 1.0;
          for (size_t k = 0; k < ell; ++k) {
            sum += disc * s.cost[s.path[start + (r + k) % ell]];
            disc *= lam;
          }
          V[s.path[start + r]] = sum / denom;
        }
        for (size_t k = start; k < s.path.size(); ++k) s.color[s.path[k]] = 2;
      } else {  // long-cycle fallback (documented deviation)
        const double denom = 1.0 - std::pow(lam, (double)ell);
        double sum = 0.0, disc = 1.0;
        for (size_t k = 0; k < ell; ++k) {
          sum += disc * s.cost[s.path[start + k]];
          disc *= lam;
        }
        V[s.path[start]] = sum / denom;
        for (size_t k = ell - 1; k >= 1; --k) {
          const u32 v = s.path[start + k];
          const u32 vn = s.path[start + (k + 1) % ell];
          V[v] = s.cost[v] + lam * V[vn];
        }
        for (size_t k = start; k < s.path.size(); ++k) s.color[s.path[k]] = 2;
      }
      cut = start;
    }
    for (size_t k = cut; k-- > 0;) {  // back-substitute the tree part
      const u32 v = s.path[k];
      const i64 sc = s.succ[v];
      if (is_inf(s.cost[v]) || (sc != -1 && is_inf(V[sc])))
        V[v] = INFV;
      else
        V[v] = (sc == -1) ? s.cost[v] : s.cost[v] + lam * V[sc];
      s.color[v] = 2;
    }
  }
}

// Policy iteration (methods._policy_iteration): closed-form evaluation plus
// one improvement scan per step; acceptance through lt(), finitely many steps.
inline u32 policy_iteration(const Comp& c, double lam, std::vector<double>& V,
                            std::vector<int>& pol, EvalScratch& s,
                            u32 max_iter = 10000) {
  const size_t n = c.n();
  pol.assign(n, -2);
  for (size_t i = 0; i < n; ++i) {
    if (!is_inf(c.e[i]))
      pol[i] = -1;
    else if (c.aoff[i + 1] > c.aoff[i])
      pol[i] = (int)c.aoff[i];  // first internal arc (Python internal[x][0])
  }
  u32 steps = 0;
  for (u32 it = 0; it < max_iter; ++it) {
    eval_policy(c, pol, lam, V, s);
    ++steps;
    bool stable = true;
    for (size_t i = 0; i < n; ++i) {
      const double cur = V[i];
      double best = cur;
      int best_act = pol[i];
      if (lt(c.e[i], best)) {
        best = c.e[i];
        best_act = -1;
      }
      for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
        const double vy = V[c.a_yl[a]];
        if (is_inf(vy)) continue;
        const double val = c.a_w[a] + lam * vy;
        if (lt(val, best)) {
          best = val;
          best_act = (int)a;
        }
      }
      if (best_act != -2 && lt(best, cur)) {
        pol[i] = best_act;
        stable = false;
      }
    }
    if (stable) break;
  }
  return steps;
}

// Backward BFS over tight arcs (methods._tight_backward_bfs).
// acts[i]: list of tight actions (-1 exit, >=0 arc index).
inline void tight_backward_bfs(const Comp& c,
                               const std::vector<std::vector<int>>& acts,
                               std::vector<u8>& good, std::vector<int>& tree) {
  const size_t n = c.n();
  good.assign(n, 0);
  tree.assign(n, -2);
  std::vector<std::vector<std::pair<u32, int>>> pred(n);
  std::vector<u32> q;
  for (size_t i = 0; i < n; ++i)
    for (int a : acts[i]) {
      if (a == -1) {
        if (!good[i]) {
          good[i] = 1;
          tree[i] = -1;
          q.push_back((u32)i);
        }
      } else {
        pred[c.a_yl[a]].push_back({(u32)i, a});
      }
    }
  for (size_t h = 0; h < q.size(); ++h) {
    const u32 y = q[h];
    for (auto& [x, a] : pred[y])
      if (!good[x]) {
        good[x] = 1;
        tree[x] = a;
        q.push_back(x);
      }
  }
}

// Beneficial-cycle detector via Proposition gap-W(b) (methods.detect_via_pi):
// PI fixed point + EVERY close()-tight Bellman minimizer + backward BFS.
// PRIMARY detector under floating(); compares argmin structure, not
// lambda^|S|-sized sweep residuals.  tight_margin as in Python.
struct DetectOut {
  bool has_cycle = false;
  std::vector<double> V;
  double margin = 0.0;
  bool has_margin = false;
  u32 pi_steps = 0;
};

inline void detect_via_pi(const Comp& c, double lam, DetectOut& out,
                          std::vector<int>& pol, EvalScratch& s,
                          std::vector<std::vector<int>>& acts,
                          std::vector<u8>& good, std::vector<int>& tree) {
  const size_t n = c.n();
  out.pi_steps = policy_iteration(c, lam, out.V, pol, s);
  acts.assign(n, {});
  out.has_margin = false;
  out.margin = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double best = INFV;
    bool any = false;
    // pass 1: raw minimum (Python: best = min(v for v,_ in vals))
    if (!is_inf(c.e[i])) {
      best = c.e[i];
      any = true;
    }
    for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
      const double vy = out.V[c.a_yl[a]];
      if (is_inf(vy)) continue;
      const double val = c.a_w[a] + lam * vy;
      if (!any || val < best) best = val;
      any = true;
    }
    if (!any) continue;
    // pass 2: tight set + margin
    if (!is_inf(c.e[i])) {
      if (close(c.e[i], best))
        acts[i].push_back(-1);
      else {
        const double gap = c.e[i] - best;
        if (!out.has_margin || gap < out.margin) {
          out.margin = gap;
          out.has_margin = true;
        }
      }
    }
    for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
      const double vy = out.V[c.a_yl[a]];
      if (is_inf(vy)) continue;
      const double val = c.a_w[a] + lam * vy;
      if (close(val, best))
        acts[i].push_back((int)a);
      else {
        const double gap = val - best;
        if (!out.has_margin || gap < out.margin) {
          out.margin = gap;
          out.has_margin = true;
        }
      }
    }
  }
  tight_backward_bfs(c, acts, good, tree);
  out.has_cycle = false;
  for (size_t i = 0; i < n; ++i)
    if (!good[i]) {
      out.has_cycle = true;
      break;
    }
}

// ---------------------------------------------------------------------------
// Algorithm core (methods.solve_core): Held-Karp bitmask layered DP.
// H(v,{v}) = e(v);  H(v,W) = min{e(v), min over internal (v,u,y), y in W\{v}:
// g + lam*H(y, W\{v})}.  Subsets ascending by popcount; cap 20 (floating),
// identical to methods.py.  first_arc: -1 exit, >=0 arc index.
// ---------------------------------------------------------------------------
struct CoreOut {
  std::vector<double> J;      // local
  std::vector<int> first_arc; // local
  u64 evals = 0;
};

inline void solve_core(const Comp& c, double lam, CoreOut& out,
                       u32 cap = 20) {
  const u32 n = (u32)c.n();
  if (n > cap) throw std::runtime_error("SolveCore cap exceeded: |S|=" +
                                        std::to_string(n));
  const u64 size = 1ull << n;
  std::vector<double> H(size * n, INFV);
  std::vector<i32> CH(size * n, -2);
  for (u32 i = 0; i < n; ++i) {
    H[((u64)1 << i) * n + i] = c.e[i];
    CH[((u64)1 << i) * n + i] = -1;
  }
  std::vector<std::vector<u32>> by_pop(n + 1);
  for (u64 mask = 1; mask < size; ++mask)
    by_pop[(u32)__builtin_popcountll(mask)].push_back((u32)mask);
  out.evals = n;
  for (u32 k = 2; k <= n; ++k) {
    for (const u32 mask : by_pop[k]) {
      double* row = &H[(u64)mask * n];
      i32* chrow = &CH[(u64)mask * n];
      for (u32 i = 0; i < n; ++i) {
        if (!((mask >> i) & 1)) continue;
        const u32 prev = mask ^ (1u << i);
        double best = c.e[i];
        i32 ch = -1;
        for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
          const u32 j = c.a_yl[a];
          if (!((prev >> j) & 1)) continue;
          const double sub = H[(u64)prev * n + j];
          if (is_inf(sub)) continue;
          const double val = c.a_w[a] + lam * sub;
          if (lt(val, best)) {
            best = val;
            ch = (i32)a;
          }
        }
        row[i] = best;
        chrow[i] = ch;
        out.evals++;
      }
    }
  }
  const u64 full = size - 1;
  out.J.assign(n, INFV);
  out.first_arc.assign(n, -1);
  for (u32 i = 0; i < n; ++i) {
    out.J[i] = H[full * n + i];
    out.first_arc[i] = (CH[full * n + i] == -2) ? -1 : CH[full * n + i];
  }
}

// methods._selection_trapped: does following the first arcs loop inside S?
inline bool selection_trapped(const Comp& c, const std::vector<int>& pol) {
  const size_t n = c.n();
  std::vector<u32> seen_ep(n, 0);
  u32 ep = 0;
  for (size_t x0 = 0; x0 < n; ++x0) {
    ++ep;
    size_t cur = x0;
    while (true) {
      const int a = pol[cur];
      if (a < 0) break;  // none or exit
      if (seen_ep[cur] == ep) return true;
      seen_ep[cur] = ep;
      cur = c.a_yl[a];
    }
  }
  return false;
}

// methods._proper_in_tree: settle states one at a time, cheapest arc into the
// settled part; acyclic hence proper (may exceed J inside a beneficial-cycle
// component; Proposition no-joint).  Pending scanned in ascending local order.
inline void proper_in_tree(const Comp& c, double lam, std::vector<int>& tree) {
  const size_t n = c.n();
  std::vector<double> V(n, INFV);
  std::vector<u8> settled(n, 0);
  tree.assign(n, -2);
  for (size_t round = 0; round < n; ++round) {
    i64 pick = -1;
    double pick_val = INFV;
    int pick_act = -2;
    for (size_t i = 0; i < n; ++i) {
      if (settled[i]) continue;
      double cand = c.e[i];
      int act = -1;
      for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
        const u32 yl = c.a_yl[a];
        if (!settled[yl]) continue;
        const double val = c.a_w[a] + lam * V[yl];
        if (lt(val, cand)) {
          cand = val;
          act = (int)a;
        }
      }
      if (lt(cand, pick_val)) {
        pick = (i64)i;
        pick_val = cand;
        pick_act = act;
      }
    }
    if (pick < 0 || is_inf(pick_val)) break;
    V[pick] = pick_val;
    tree[pick] = pick_act;
    settled[pick] = 1;
  }
}

// ---------------------------------------------------------------------------
// Master result + Algorithm master (methods.run_master, floating branch:
// solve_easy audit first, detect_via_pi primary, PI values as benign labels,
// SolveCore on beneficial components, then Algorithm recover).
// ---------------------------------------------------------------------------
struct CompRecord {
  std::vector<u32> members;  // global, sorted (terminal: excludes xe)
  bool has_cycle = false, sweep_verdict = false, disagreement = false;
  u32 sweeps = 0, strict_sweeps = 0;
  double margin = 0.0;
  bool has_margin = false;
  u64 core_evals = 0;  // 0 for benign components
};

struct MasterOut {
  std::vector<double> J;            // size N, INF outside Vs
  std::vector<i32> mu_u, mu_y;      // -1 absent
  std::vector<u8> in_R;
  u64 Vs_size = 0, bc_count = 0, disagreements = 0;
  std::vector<CompRecord> records;
  double t_reach = 0, t_scc = 0, t_fold = 0, t_sweep = 0, t_detect = 0,
         t_core = 0, t_recover = 0, t_book = 0, t_master = 0;
  // t_book: per-component bookkeeping (label write-back, records) -- kept as
  // its own column so the seven algorithmic stages plus t_book account for
  // t_master to <5% (acceptance B9).
  double min_margin = 0.0;
  bool has_min_margin = false;
};

// Algorithm recover (methods.recover), one component.
inline void recover_component(const Graph& g, const std::vector<u8>& in_R,
                              const Cond& cd, u32 cid, u32 xe, double lam,
                              const std::vector<double>& J, bool has_cycle,
                              const std::vector<int>* first_arc, Workspace& ws,
                              Comp& c, std::vector<std::vector<int>>& acts,
                              std::vector<u8>& good, std::vector<int>& tree,
                              std::vector<i32>& mu_u, std::vector<i32>& mu_y) {
  fold(g, in_R, cd, cid, xe, lam, J, ws, c);
  const size_t n = c.n();
  std::vector<int> comp_pol;
  if (!has_cycle) {
    acts.assign(n, {});
    for (size_t i = 0; i < n; ++i) {
      const double jx = J[c.S[i]];
      if (close(c.e[i], jx)) acts[i].push_back(-1);
      for (u32 a = c.aoff[i]; a < c.aoff[i + 1]; ++a) {
        const double jy = J[c.S[c.a_yl[a]]];
        if (is_inf(jy)) continue;
        if (close(c.a_w[a] + lam * jy, jx)) acts[i].push_back((int)a);
      }
    }
    tight_backward_bfs(c, acts, good, tree);
    comp_pol = tree;
  } else {
    comp_pol = *first_arc;
    if (selection_trapped(c, comp_pol)) proper_in_tree(c, lam, comp_pol);
  }
  for (size_t i = 0; i < n; ++i) {
    const int a = comp_pol[i];
    if (a == -2) continue;
    const u32 x = c.S[i];
    if (a == -1) {
      // resolve the folded exit to the ctx-cheapest concrete arc leaving S
      double bx = INFV;
      i32 bu = -1, by = -1;
      for (u32 u = 0; u < g.M; ++u) {
        const u32 y = g.nxt[(u64)x * g.M + u];
        if (!in_R[y]) continue;
        if (ws.inS(y)) continue;  // y in S (terminal S excludes xe already)
        const double jy = J[y];
        if (is_inf(jy)) continue;
        const double val = g.cost[(u64)x * g.M + u] + lam * jy;
        if (lt(val, bx)) {
          bx = val;
          bu = (i32)u;
          by = (i32)y;
        }
      }
      if (bu >= 0) {
        mu_u[x] = bu;
        mu_y[x] = by;
      }
    } else {
      mu_u[x] = (i32)c.a_u[a];
      mu_y[x] = (i32)c.S[c.a_yl[a]];
    }
  }
}

inline void run_master(const Graph& g, u32 xe, i32 ue, double lam,
                       MasterOut& out) {
  const double t0 = now_s();
  double t = t0;
  out.records.clear();
  out.bc_count = out.disagreements = 0;
  out.has_min_margin = false;
  out.min_margin = 0.0;
  out.t_book = 0.0;
  out.Vs_size = reverse_reachable(g, xe, out.in_R);
  double tb = now_s();
  out.t_reach = tb - t;
  out.J.assign(g.N, INFV);  // N-proportional init: booked (shared boundary)
  out.J[xe] = 0.0;
  out.mu_u.assign(g.N, -1);
  out.mu_y.assign(g.N, -1);
  if (ue >= 0) {
    out.mu_u[xe] = ue;
    out.mu_y[xe] = (i32)xe;
  }
  out.t_book += now_s() - tb;
  if (out.Vs_size == 1) {
    out.t_scc = out.t_fold = out.t_sweep = out.t_detect = out.t_core =
        out.t_recover = 0.0;
    out.t_master = now_s() - t0;
    return;
  }
  t = now_s();
  Cond cd;
  condense(g, out.in_R, xe, cd);
  out.t_scc = now_s() - t;
  out.t_fold = out.t_sweep = out.t_detect = out.t_core = 0.0;

  t = now_s();
  Workspace ws(g.N);
  Comp c;
  EasyOut easy;
  DetectOut det;
  EvalScratch es;
  CoreOut core;
  std::vector<double> Jn;
  std::vector<int> poln, pipol, tree;
  std::vector<std::vector<int>> acts;
  std::vector<u8> good;
  std::vector<u8> cyc_by_cid(cd.ncomp, 0);
  std::vector<std::vector<int>> first_by_cid(cd.ncomp);
  out.t_book += now_s() - t;

  // Adjacent spans share their boundary timestamp so that the whole loop
  // body is covered with zero leak (acceptance B9); only the loop control
  // between iterations stays unmeasured (a few ns per component).
  for (u32 cid = 0; cid < cd.ncomp; ++cid) {
    const u32 sz = cd.off[cid + 1] - cd.off[cid];
    if (cd.comp_of[xe] == cid && sz == 1) continue;  // singleton terminal
    const double t1 = now_s();
    fold(g, out.in_R, cd, cid, xe, lam, out.J, ws, c);
    const double t2 = now_s();
    out.t_fold += t2 - t1;
    solve_easy(c, lam, easy, Jn, poln);
    const double t3 = now_s();
    out.t_sweep += t3 - t2;
    detect_via_pi(c, lam, det, pipol, es, acts, good, tree);
    const double t4 = now_s();
    out.t_detect += t4 - t3;
    const bool verdict = det.has_cycle;
    const bool disagree = (verdict != easy.has_cycle_sweep);
    if (disagree) out.disagreements++;
    cyc_by_cid[cid] = verdict ? 1 : 0;
    const std::vector<double>* labels = &det.V;  // benign: PI fixed point
    if (verdict) {
      out.bc_count++;
      solve_core(c, lam, core);
      labels = &core.J;
      first_by_cid[cid] = core.first_arc;
      t = now_s();
      out.t_core += t - t4;
    } else {
      t = t4;  // book span starts where detect ended (no extra clock call)
    }
    for (size_t i = 0; i < c.n(); ++i)
      if (!is_inf((*labels)[i])) out.J[c.S[i]] = (*labels)[i];
    CompRecord rec;
    rec.members = c.S;
    rec.has_cycle = verdict;
    rec.sweep_verdict = easy.has_cycle_sweep;
    rec.disagreement = disagree;
    rec.sweeps = easy.sweeps;
    rec.strict_sweeps = easy.strict_sweeps;
    rec.margin = det.margin;
    rec.has_margin = det.has_margin;
    rec.core_evals = verdict ? core.evals : 0;
    if (det.has_margin &&
        (!out.has_min_margin || det.margin < out.min_margin)) {
      out.min_margin = det.margin;
      out.has_min_margin = true;
    }
    out.records.push_back(std::move(rec));
    out.t_book += now_s() - t;
  }
  t = now_s();
  for (u32 cid = 0; cid < cd.ncomp; ++cid) {
    const u32 sz = cd.off[cid + 1] - cd.off[cid];
    if (cd.comp_of[xe] == cid && sz == 1) continue;
    recover_component(g, out.in_R, cd, cid, xe, lam, out.J, cyc_by_cid[cid],
                      &first_by_cid[cid], ws, c, acts, good, tree, out.mu_u,
                      out.mu_y);
  }
  out.t_recover = now_s() - t;
  out.t_master = now_s() - t0;
}

// Improper states of a feedback (methods.improper_states, O(N) functional-
// graph variant: proper set = backward reachability of xe along mu edges;
// semantically identical -- a checker, not a compared algorithm).
inline u64 improper_count(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                          const std::vector<i32>& mu_u,
                          const std::vector<i32>& mu_y) {
  const u32 N = g.N;
  std::vector<u32> cnt(N + 1, 0);
  for (u32 x = 0; x < N; ++x)
    if (in_R[x] && x != xe && mu_u[x] >= 0) cnt[(u32)mu_y[x] + 1]++;
  for (u32 i = 0; i < N; ++i) cnt[i + 1] += cnt[i];
  std::vector<u32> adj(cnt[N]);
  {
    std::vector<u32> cur(cnt.begin(), cnt.end() - 1);
    for (u32 x = 0; x < N; ++x)
      if (in_R[x] && x != xe && mu_u[x] >= 0) adj[cur[(u32)mu_y[x]]++] = x;
  }
  std::vector<u8> good(N, 0);
  std::vector<u32> q;
  good[xe] = 1;
  q.push_back(xe);
  for (size_t h = 0; h < q.size(); ++h) {
    const u32 y = q[h];
    for (u32 a = cnt[y]; a < cnt[y + 1]; ++a)
      if (!good[adj[a]]) {
        good[adj[a]] = 1;
        q.push_back(adj[a]);
      }
  }
  u64 bad = 0;
  for (u32 x = 0; x < N; ++x)
    if (in_R[x] && x != xe && !good[x]) bad++;
  return bad;
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

// VI (methods.vi + exp16.vi_until stopping rule): synchronous Bellman on
// G|_R with xe absorbing, V_0 = INF, V_0(xe) = 0; stop at the first k with
// sup-residual <= eps_vi*(1-lam)/(2*lam) (inf-inf counts 0, one-sided inf
// counts INF).  k_bound (exp16): ceil(log(eps*(1-lam)/g_max)/log(lam)) + 5.
struct ViOut {
  std::vector<double> V;  // final V_k (size N, INF outside R)
  u64 k = 0;
  bool converged = false;
  double t = 0.0;
};

inline i64 vi_k_bound(double g_max, double lam, double eps) {
  return (i64)std::ceil(std::log(eps * (1.0 - lam) / std::max(g_max, 1e-12)) /
                        std::log(lam)) +
         5;
}

inline void vi_run(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                   double lam, ViOut& out, double eps_vi = 1e-6,
                   i64 k_cap = -1, double wall_s = 300.0) {
  const double t0 = now_s();
  double g_max = 1.0;
  for (u32 x = 0; x < g.N; ++x)
    if (in_R[x])
      for (u32 u = 0; u < g.M; ++u)
        if (in_R[g.nxt[(u64)x * g.M + u]])
          g_max = std::max(g_max, g.cost[(u64)x * g.M + u]);
  const double thr = eps_vi * (1.0 - lam) / (2.0 * lam);
  if (k_cap < 0) k_cap = std::max<i64>(2 * vi_k_bound(g_max, lam, eps_vi), 64);
  std::vector<double> V(g.N, INFV), Vn(g.N, INFV);
  V[xe] = 0.0;
  out.k = 0;
  out.converged = false;
  while ((i64)out.k < k_cap) {
    double res = 0.0;
    for (u32 x = 0; x < g.N; ++x) {
      if (!in_R[x]) continue;
      if (x == xe) {
        Vn[x] = 0.0;
        continue;
      }
      double best = INFV;
      for (u32 u = 0; u < g.M; ++u) {
        const u32 y = g.nxt[(u64)x * g.M + u];
        if (!in_R[y]) continue;
        const double vy = V[y];
        if (is_inf(vy)) continue;
        const double val = g.cost[(u64)x * g.M + u] + lam * vy;
        if (lt(val, best)) best = val;
      }
      Vn[x] = best;
      const bool ia = is_inf(V[x]), ib = is_inf(best);
      if (ia && ib) {
      } else if (ia != ib)
        res = INFV;
      else
        res = std::max(res, std::fabs(V[x] - best));
    }
    V.swap(Vn);
    out.k++;
    if (res <= thr) {
      out.converged = true;
      break;
    }
    if (now_s() - t0 > wall_s) break;
  }
  out.V.swap(V);
  out.t = now_s() - t0;
}

// Greedy feedback from a value table (methods.greedy_actions + greedy_mu):
// raw min, close()-tight set, smallest input tie-break.
inline void greedy_mu(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                      double lam, const std::vector<double>& V,
                      std::vector<i32>& mu_u, std::vector<i32>& mu_y) {
  mu_u.assign(g.N, -1);
  mu_y.assign(g.N, -1);
  for (u32 x = 0; x < g.N; ++x) {
    if (!in_R[x] || x == xe) continue;
    double best = INFV;
    bool any = false;
    for (u32 u = 0; u < g.M; ++u) {
      const u32 y = g.nxt[(u64)x * g.M + u];
      if (!in_R[y]) continue;
      const double vy = V[y];
      if (is_inf(vy)) continue;
      const double val = g.cost[(u64)x * g.M + u] + lam * vy;
      if (!any || val < best) best = val;
      any = true;
    }
    if (!any) continue;
    for (u32 u = 0; u < g.M; ++u) {  // smallest tight input wins
      const u32 y = g.nxt[(u64)x * g.M + u];
      if (!in_R[y]) continue;
      const double vy = V[y];
      if (is_inf(vy)) continue;
      const double val = g.cost[(u64)x * g.M + u] + lam * vy;
      if (close(val, best)) {
        mu_u[x] = (i32)u;
        mu_y[x] = (i32)y;
        break;
      }
    }
  }
}

// PI baseline (methods.pi): fold R\{xe} with exits into xe, run the same
// closed-form policy iteration.
struct PiOut {
  std::vector<double> V;  // size N, INF outside R, V(xe) = 0
  u32 steps = 0;
  double t = 0.0;
};

inline void pi_run(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                   double lam, PiOut& out) {
  const double t0 = now_s();
  Comp c;
  c.terminal = false;
  c.S.clear();
  for (u32 x = 0; x < g.N; ++x)
    if (in_R[x] && x != xe) c.S.push_back(x);
  const size_t n = c.S.size();
  std::vector<u32> loc(g.N, 0xffffffffu);
  for (size_t i = 0; i < n; ++i) loc[c.S[i]] = (u32)i;
  c.e.assign(n, INFV);
  c.aoff.assign(n + 1, 0);
  c.a_u.clear();
  c.a_yl.clear();
  c.a_w.clear();
  for (size_t i = 0; i < n; ++i) {
    const u32 x = c.S[i];
    for (u32 u = 0; u < g.M; ++u) {
      const u32 y = g.nxt[(u64)x * g.M + u];
      if (!in_R[y]) continue;
      const double w = g.cost[(u64)x * g.M + u];
      if (y == xe) {
        if (lt(w, c.e[i])) c.e[i] = w;
      } else {
        c.a_u.push_back(u);
        c.a_yl.push_back(loc[y]);
        c.a_w.push_back(w);
      }
    }
    c.aoff[i + 1] = (u32)c.a_u.size();
  }
  std::vector<double> Vloc;
  std::vector<int> pol;
  EvalScratch es;
  out.steps = policy_iteration(c, lam, Vloc, pol, es);
  out.V.assign(g.N, INFV);
  out.V[xe] = 0.0;
  for (size_t i = 0; i < n; ++i) out.V[c.S[i]] = Vloc[i];
  out.t = now_s() - t0;
}

// Floyd-type discounted (min,+) matrix powers (methods.floyd):
// W_{k+1} = W_1 (+) (W_1 (x)_lam W_k); xe-column stopping rule as VI.
struct FloydOut {
  std::vector<double> col;  // xe column over R (size N; col[xe]=0)
  u64 k = 0;
  int status = 0;  // 0 ok, 1 cap, 2 skipped (memory)
  double t = 0.0;
};

inline void floyd_run(const Graph& g, const std::vector<u8>& in_R, u32 xe,
                      double lam, FloydOut& out, double eps_vi = 1e-6,
                      i64 k_cap = -1, u32 n_cap = 4200) {
  const double t0 = now_s();
  std::vector<u32> states;
  for (u32 x = 0; x < g.N; ++x)
    if (in_R[x]) states.push_back(x);
  const u32 nR = (u32)states.size();
  if (nR > n_cap) {
    out.status = 2;
    out.t = 0;
    out.k = 0;
    return;
  }
  std::vector<u32> loc(g.N, 0xffffffffu);
  for (u32 i = 0; i < nR; ++i) loc[states[i]] = i;
  const u32 xel = loc[xe];
  double g_max = 1.0;
  // sparse W1 rows: (z, w) with parallel arcs min-collapsed through lt
  std::vector<std::vector<std::pair<u32, double>>> w1(nR);
  for (u32 i = 0; i < nR; ++i) {
    const u32 x = states[i];
    for (u32 u = 0; u < g.M; ++u) {
      const u32 y = g.nxt[(u64)x * g.M + u];
      if (!in_R[y]) continue;
      const double w = g.cost[(u64)x * g.M + u];
      g_max = std::max(g_max, w);
      const u32 j = loc[y];
      bool found = false;
      for (auto& pr : w1[i])
        if (pr.first == j) {
          if (lt(w, pr.second)) pr.second = w;
          found = true;
          break;
        }
      if (!found) w1[i].push_back({j, w});
    }
  }
  const double thr = eps_vi * (1.0 - lam) / (2.0 * lam);
  if (k_cap < 0) k_cap = std::max<i64>(2 * vi_k_bound(g_max, lam, eps_vi), 64);
  std::vector<double> Wk((u64)nR * nR, INFV), Wn((u64)nR * nR, INFV);
  for (u32 i = 0; i < nR; ++i)
    for (auto& [j, w] : w1[i]) Wk[(u64)i * nR + j] = w;
  std::vector<double> col(nR), colp(nR);
  for (u32 i = 0; i < nR; ++i) colp[i] = Wk[(u64)i * nR + xel];
  colp[xel] = 0.0;
  out.k = 1;
  out.status = 1;
  while ((i64)out.k < k_cap) {
    for (u64 a = 0; a < (u64)nR * nR; ++a) Wn[a] = INFV;
    for (u32 i = 0; i < nR; ++i) {
      double* rown = &Wn[(u64)i * nR];
      for (auto& [j, w] : w1[i]) {
        if (!is_inf(w) && (is_inf(rown[j]) || lt(w, rown[j]))) rown[j] = w;
      }
      for (auto& [z, wxz] : w1[i]) {
        const double* rowk = &Wk[(u64)z * nR];
        for (u32 y = 0; y < nR; ++y) {
          const double wzy = rowk[y];
          if (is_inf(wzy)) continue;
          const double cand = wxz + lam * wzy;
          if (is_inf(rown[y]) || lt(cand, rown[y])) rown[y] = cand;
        }
      }
    }
    Wk.swap(Wn);
    out.k++;
    double res = 0.0;
    for (u32 i = 0; i < nR; ++i) {
      col[i] = Wk[(u64)i * nR + xel];
      if (i == xel) col[i] = 0.0;
      const bool ia = is_inf(colp[i]), ib = is_inf(col[i]);
      if (ia && ib) continue;
      if (ia != ib) {
        res = INFV;
        continue;
      }
      res = std::max(res, std::fabs(colp[i] - col[i]));
    }
    col.swap(colp);
    if (res <= thr) {
      out.status = 0;
      break;
    }
  }
  out.col.assign(g.N, INFV);
  for (u32 i = 0; i < nR; ++i) out.col[states[i]] = colp[i];
  out.col[xe] = 0.0;
  out.t = now_s() - t0;
}

// DFS simple-path oracle (methods.dfs_simple_paths): exact minimum discounted
// cost over ALL simple paths x -> xe on the FULL graph; exponential.
struct DfsCtx {
  const Graph* g;
  u32 xe;
  double lam;
  std::vector<u8> onpath;
  u64 calls = 0;
  double deadline = 0.0;
  bool timed_out = false;
};

inline double dfs_rec(DfsCtx& ctx, u32 x) {
  if (x == ctx.xe) return 0.0;
  if (((++ctx.calls) & 1023u) == 0 && now_s() > ctx.deadline) {
    ctx.timed_out = true;
    return INFV;
  }
  ctx.onpath[x] = 1;
  double best = INFV;
  const Graph& g = *ctx.g;
  for (u32 u = 0; u < g.M; ++u) {
    const u32 y = g.nxt[(u64)x * g.M + u];
    if (ctx.onpath[y]) continue;
    const double sub = dfs_rec(ctx, y);
    if (ctx.timed_out) break;
    if (is_inf(sub)) continue;
    const double val = g.cost[(u64)x * g.M + u] + ctx.lam * sub;
    if (lt(val, best)) best = val;
  }
  ctx.onpath[x] = 0;
  return best;
}

struct DfsOut {
  std::vector<double> J;
  bool timed_out = false;
  double t = 0.0;
};

inline void dfs_run(const Graph& g, u32 xe, double lam, DfsOut& out,
                    double timeout_s = 60.0) {
  const double t0 = now_s();
  DfsCtx ctx;
  ctx.g = &g;
  ctx.xe = xe;
  ctx.lam = lam;
  ctx.onpath.assign(g.N, 0);
  ctx.deadline = t0 + timeout_s;
  out.J.assign(g.N, INFV);
  out.J[xe] = 0.0;
  for (u32 x = 0; x < g.N && !ctx.timed_out; ++x) {
    ctx.onpath[x] = 1;  // Python: dfs(x, {x})
    out.J[x] = dfs_rec(ctx, x);
    ctx.onpath[x] = 0;
  }
  out.timed_out = ctx.timed_out;
  out.t = now_s() - t0;
}

// Hamming cost (bcn_data.hamming_cost, Gao bit convention bits(i) = N-i for
// 1-based i, i.e. N-1-x for 0-based x): g(x,u) = d_H(x,xe) + [u != ue].
inline void build_hamming_cost(Graph& g, u32 xe, u32 ue) {
  g.cost.assign((u64)g.N * g.M, 0.0);
  for (u32 x = 0; x < g.N; ++x) {
    const double dh =
        (double)__builtin_popcountll((u64)((g.N - 1 - x) ^ (g.N - 1 - xe)));
    for (u32 u = 0; u < g.M; ++u)
      g.cost[(u64)x * g.M + u] = dh + (u == ue ? 0.0 : 1.0);
  }
}

}  // namespace bcn
