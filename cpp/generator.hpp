// generator.hpp -- synthetic BCN generator with plantable strongly connected
// cores (exp17 plan section 3).  BCN semantics throughout: N=2^n, M=2^m,
// column-major L; acyclic skeleton along a random rank (every arc strictly
// increases rank, xe at top rank).
//
// STRUCTURAL GUARANTEE (fixes the plan's section 3.3 argument, which was
// wrong for >1 planted set: a cross-set cycle can alternate rank-down arcs
// inside two sets with rank-up arcs between them).  Each planted set now
// occupies a CONTIGUOUS rank band and every arc leaving a set jumps ABOVE the
// band top.  Outside the sets all arcs are rank-increasing, so any walk is
// strictly monotone in band-top order once it leaves a set and can never
// return: the nontrivial SCCs are exactly the planted sets.
#pragma once
#include <numeric>
#include <sstream>

#include "bcn_core.hpp"

namespace bcn {

constexpr u64 MASTER_SEED = 20260831ull;

struct SplitMix {
  u64 s;
  explicit SplitMix(u64 seed) : s(seed) {}
  u64 next() {
    s += 0x9E3779B97f4A7C15ull;
    u64 z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  u64 below(u64 k) { return next() % k; }  // determinism > uniformity nicety
};

inline u64 fnv1a(const void* data, size_t len, u64 h = 1469598103934665603ull) {
  const u8* p = (const u8*)data;
  for (size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}
inline u64 fnv1a_str(const std::string& s) { return fnv1a(s.data(), s.size()); }

struct Decoy {
  u32 x, u;   // 0-based
  i64 p, q;   // eps = p/q exact rational
};

struct Instance {
  Graph g;
  u32 xe = 0, ue = 0;  // 0-based
  i64 lam_p = 1, lam_q = 2;
  double lam = 0.5;
  std::vector<std::vector<u32>> cores;    // 0-based members, cycle order
  std::vector<std::vector<u32>> benigns;  // 0-based members, cycle order
  std::vector<Decoy> decoys;
  std::string config;  // graph part only (lambda excluded: seed is
                       // lambda-independent so curve (b) reuses one L)
  u64 seed = 0, l_hash = 0;
  bool benign_dropped = false;  // room fallback triggered (recorded in CSV)
};

struct GenParams {
  u32 n = 4, m = 2;
  std::vector<u32> core_sizes;  // one planted beneficial core per entry
  u32 n_benign = 0, benign_size = 0;
  i64 lam_p = 1, lam_q = 2;
  u32 rep = 0;
};

inline std::string graph_config_str(const GenParams& p) {
  std::ostringstream os;
  os << "n=" << p.n << ";m=" << p.m << ";cores=";
  if (p.core_sizes.empty()) os << "0";
  for (size_t i = 0; i < p.core_sizes.size(); ++i)
    os << (i ? "+" : "") << p.core_sizes[i];
  os << ";benign=" << p.n_benign << "x" << p.benign_size;
  return os.str();
}

// Exit target strictly above the set's band top (xe at top rank always
// qualifies, so the draw needs no rejection loop).
inline u32 above_band_target(SplitMix& rng, const std::vector<u32>& order,
                             u32 band_top, u32 N) {
  const u32 r = band_top + 1 + (u32)rng.below(N - 1 - band_top);
  return order[r];
}

inline void generate(const GenParams& p, Instance& inst) {
  const u32 N = 1u << p.n, M = 1u << p.m;
  inst.config = graph_config_str(p);
  inst.seed = SplitMix(fnv1a_str(inst.config) ^ MASTER_SEED ^ (u64)p.rep).next();
  SplitMix rng(inst.seed);
  inst.g.N = N;
  inst.g.M = M;
  inst.g.nxt.assign((u64)N * M, 0);
  inst.xe = N - 1;  // 1-based N = all-zero state under the Gao convention
  inst.ue = 0;
  inst.lam_p = p.lam_p;
  inst.lam_q = p.lam_q;
  inst.lam = (double)p.lam_p / (double)p.lam_q;
  inst.cores.clear();
  inst.benigns.clear();
  inst.decoys.clear();
  inst.benign_dropped = false;

  // 1. random rank: order[0..N-2] = shuffled non-xe states, order[N-1] = xe
  std::vector<u32> order(N);
  std::iota(order.begin(), order.end(), 0u);
  std::swap(order[inst.xe], order[N - 1]);  // xe to the top slot
  for (u32 i = N - 2; i > 0; --i)
    std::swap(order[i], order[(u32)rng.below(i + 1)]);
  std::vector<u32> rank(N);
  for (u32 r = 0; r < N; ++r) rank[order[r]] = r;

  // 2. acyclic skeleton: every arc jumps to a strictly higher rank
  for (u32 x = 0; x < N; ++x) {
    if (x == inst.xe) {
      for (u32 u = 0; u < M; ++u) inst.g.nxt[(u64)x * M + u] = inst.xe;
      continue;
    }
    for (u32 u = 0; u < M; ++u) {
      const u32 r = rank[x] + 1 + (u32)rng.below(N - 1 - rank[x]);
      inst.g.nxt[(u64)x * M + u] = order[r];
    }
  }

  // 3. planted-set selection: contiguous rank bands placed back to back at a
  //    random offset (membership is random through the rank permutation);
  //    drop benigns if there is no room (fallback, recorded)
  u64 want = 0;
  for (u32 s : p.core_sizes) want += s;
  u32 n_benign = p.n_benign, bsize = p.benign_size;
  if (want + (u64)n_benign * bsize > N - 1) {
    n_benign = 0;
    inst.benign_dropped = (p.n_benign > 0);
  }
  const u64 total = want + (u64)n_benign * bsize;
  if (total > N - 1)
    throw std::runtime_error("planted sets exceed N-1 states");
  std::vector<u32> sizes(p.core_sizes);
  for (u32 b = 0; b < n_benign; ++b) sizes.push_back(bsize);
  u32 cursor =
      total < N - 1 ? (u32)rng.below((u64)(N - 1 - total) + 1) : 0;
  std::vector<std::pair<u32, u32>> bands;  // (band_start, band_top) in ranks
  std::vector<std::vector<u32>> sets;
  for (u32 s : sizes) {
    std::vector<u32> set(order.begin() + cursor, order.begin() + cursor + s);
    bands.push_back({cursor, cursor + s - 1});
    sets.push_back(std::move(set));
    cursor += s;
  }
  for (size_t i = 0; i < p.core_sizes.size(); ++i)
    inst.cores.push_back(sets[i]);
  for (size_t i = p.core_sizes.size(); i < sets.size(); ++i)
    inst.benigns.push_back(sets[i]);

  // 4. wiring: u=0 Hamilton cycle; u=1 chord (M>=4) or coin chord/exit (M=2);
  //    every non-internal arc of a set member must jump ABOVE the band top
  //    (structural guarantee above), resampled where the skeleton arc lands
  //    at or below it.
  for (size_t si = 0; si < sets.size(); ++si) {
    const std::vector<u32>& set = sets[si];
    const u32 band_top = bands[si].second;
    const bool core = si < p.core_sizes.size();
    const u32 s = (u32)set.size();
    auto fix_exit = [&](u32 x, u32 u) {
      if (rank[inst.g.nxt[(u64)x * M + u]] <= band_top)
        inst.g.nxt[(u64)x * M + u] = above_band_target(rng, order, band_top, N);
    };
    auto chord = [&](u32 i, u32 succ) {
      u32 j = i;
      for (int tries = 0; tries < 32; ++tries) {
        j = (u32)rng.below(s);
        if (j != i && (s <= 2 || set[j] != succ)) break;
      }
      if (j == i) j = (i + 1) % s;
      return set[j];
    };
    // Benign sets get SATURATING internal costs g_max = n+1 on every internal
    // arc: staying on any internal cycle then costs (n+1)/(1-lam) > J at
    // every lambda (J < g_max/(1-lam) strictly), so benign sets are provably
    // non-beneficial -- natural Hamming costs are NOT enough on small
    // instances (cheap cycle states + expensive exits flip the verdict; found
    // by the crosscheck grid, recorded in plan section 10).
    const i64 gmax = (i64)p.n + 1;
    for (u32 i = 0; i < s; ++i) {
      const u32 x = set[i];
      const u32 succ = set[(i + 1) % s];
      inst.g.nxt[(u64)x * M + 0] = succ;  // the controllable cycle
      if (core)
        inst.decoys.push_back(
            {x, 0, p.lam_q - p.lam_p, 2 * p.lam_q});  // eps = (1-lam)/2
      else
        inst.decoys.push_back({x, 0, gmax, 1});
      if (M >= 4) {
        inst.g.nxt[(u64)x * M + 1] = chord(i, succ);  // internal density
        if (!core) inst.decoys.push_back({x, 1, gmax, 1});
        for (u32 u = 2; u < M; ++u) fix_exit(x, u);
      } else {  // M == 2
        if (i == 0)
          fix_exit(x, 1);  // c_0 always keeps an exit
        else if (rng.next() & 1u) {
          inst.g.nxt[(u64)x * M + 1] = chord(i, succ);
          if (!core) inst.decoys.push_back({x, 1, gmax, 1});
        } else
          fix_exit(x, 1);
      }
    }
    // self-check: at least one arc leaves the set
    bool has_exit = false;
    for (u32 v : set)
      for (u32 u = 0; u < M; ++u)
        if (rank[inst.g.nxt[(u64)v * M + u]] > band_top) has_exit = true;
    if (!has_exit) throw std::runtime_error("planted set lost all exits");
  }

  // 5. costs: Hamming base + decoy overrides eps = p/q (ratio-of-ints double,
  //    identical to Python float(Fraction(p,q)))
  build_hamming_cost(inst.g, inst.xe, inst.ue);
  for (const Decoy& d : inst.decoys)
    inst.g.cost[(u64)d.x * M + d.u] = (double)d.p / (double)d.q;

  // 6. self-checks: fixpoint, full reverse reachability
  if (inst.g.nxt[(u64)inst.xe * M + inst.ue] != inst.xe)
    throw std::runtime_error("f(xe,ue) != xe");
  inst.g.build_rev();
  std::vector<u8> in_R;
  if (reverse_reachable(inst.g, inst.xe, in_R) != N)
    throw std::runtime_error("skeleton lost full reverse reachability");
  inst.l_hash = fnv1a(inst.g.nxt.data(), inst.g.nxt.size() * sizeof(u32));
}

}  // namespace bcn
