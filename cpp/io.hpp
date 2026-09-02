// io.hpp -- line-based text exchange formats for exp17 (plan sections 5.1 and
// 6.1): instance files (C++ -> Python exact side), result files (C++ -> the
// crosscheck verifier), the t_lgl.txt network loader, and CSV emission.
// No third-party dependencies; states/inputs are 1-based on disk.
#pragma once
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <sstream>

#include "generator.hpp"

namespace bcn {

inline std::string fmt_g17(double v) {
  if (is_inf(v)) return "inf";
  char buf[40];
  snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

// ---------------------------------------------------------------------------
// Instance files
// ---------------------------------------------------------------------------
inline void write_instance(const std::string& path, const Instance& inst) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "# exp17 crosscheck instance\n";
  f << "config " << inst.config << "\n";
  f << "seed " << inst.seed << "\n";
  f << "n " << (u32)__builtin_ctz(inst.g.N) << "\n";
  f << "m " << (u32)__builtin_ctz(inst.g.M) << "\n";
  f << "xe " << inst.xe + 1 << "\n";
  f << "ue " << inst.ue + 1 << "\n";
  f << "lam " << inst.lam_p << " " << inst.lam_q << "\n";
  auto dump_sets = [&](const char* key,
                       const std::vector<std::vector<u32>>& sets) {
    f << key << " " << sets.size() << "\n";
    for (const auto& s : sets) {
      f << s.size();
      for (u32 v : s) f << " " << v + 1;
      f << "\n";
    }
  };
  dump_sets("cores", inst.cores);
  dump_sets("benigns", inst.benigns);
  f << "decoys " << inst.decoys.size() << "\n";
  for (const Decoy& d : inst.decoys)
    f << d.x + 1 << " " << d.u + 1 << " " << d.p << " " << d.q << "\n";
  f << "L";
  for (u32 u = 0; u < inst.g.M; ++u)  // column-major: column (u-1)*N + x
    for (u32 x = 0; x < inst.g.N; ++x)
      f << " " << inst.g.nxt[(u64)x * inst.g.M + u] + 1;
  f << "\n";
}

inline void read_instance(const std::string& path, Instance& inst) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::string tok;
  u32 n = 0, m = 0;
  inst.cores.clear();
  inst.benigns.clear();
  inst.decoys.clear();
  while (f >> tok) {
    if (tok[0] == '#') {
      std::string rest;
      std::getline(f, rest);
      continue;
    }
    if (tok == "config")
      f >> inst.config;
    else if (tok == "seed")
      f >> inst.seed;
    else if (tok == "n")
      f >> n;
    else if (tok == "m")
      f >> m;
    else if (tok == "xe") {
      u32 v;
      f >> v;
      inst.xe = v - 1;
    } else if (tok == "ue") {
      u32 v;
      f >> v;
      inst.ue = v - 1;
    } else if (tok == "lam") {
      f >> inst.lam_p >> inst.lam_q;
    } else if (tok == "cores" || tok == "benigns") {
      size_t k;
      f >> k;
      auto& dst = (tok == "cores") ? inst.cores : inst.benigns;
      dst.resize(k);
      for (auto& s : dst) {
        size_t sz;
        f >> sz;
        s.resize(sz);
        for (auto& v : s) {
          f >> v;
          v -= 1;
        }
      }
    } else if (tok == "decoys") {
      size_t k;
      f >> k;
      inst.decoys.resize(k);
      for (auto& d : inst.decoys) {
        f >> d.x >> d.u >> d.p >> d.q;
        d.x -= 1;
        d.u -= 1;
      }
    } else if (tok == "L") {
      const u32 N = 1u << n, M = 1u << m;
      inst.g.N = N;
      inst.g.M = M;
      inst.g.nxt.assign((u64)N * M, 0);
      for (u32 u = 0; u < M; ++u)
        for (u32 x = 0; x < N; ++x) {
          u32 y;
          f >> y;
          inst.g.nxt[(u64)x * M + u] = y - 1;
        }
    }
  }
  inst.lam = (double)inst.lam_p / (double)inst.lam_q;
  build_hamming_cost(inst.g, inst.xe, inst.ue);
  for (const Decoy& d : inst.decoys)
    inst.g.cost[(u64)d.x * inst.g.M + d.u] = (double)d.p / (double)d.q;
  inst.g.build_rev();
  inst.l_hash = fnv1a(inst.g.nxt.data(), inst.g.nxt.size() * sizeof(u32));
}

// networks/*.txt loader (bcn_data format: '#' header lines, then n, m, then
// one line of N*M column-major 1-based entries).
inline void read_network(const std::string& path, Instance& inst, u32 xe1,
                         u32 ue1, i64 lam_p, i64 lam_q) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::string line;
  std::vector<u64> nums;
  u32 n = 0, m = 0;
  int stage = 0;
  while (std::getline(f, line)) {
    if (!line.empty() && line[0] == '#') continue;
    std::istringstream is(line);
    if (stage == 0) {
      is >> n;
      stage = 1;
    } else if (stage == 1) {
      is >> m;
      stage = 2;
    } else {
      u64 v;
      while (is >> v) nums.push_back(v);
    }
  }
  const u32 N = 1u << n, M = 1u << m;
  if (nums.size() != (u64)N * M)
    throw std::runtime_error("network file: wrong L length");
  inst.g.N = N;
  inst.g.M = M;
  inst.g.nxt.assign((u64)N * M, 0);
  for (u32 u = 0; u < M; ++u)
    for (u32 x = 0; x < N; ++x)
      inst.g.nxt[(u64)x * M + u] = (u32)nums[(u64)u * N + x] - 1;
  inst.xe = xe1 - 1;
  inst.ue = ue1 - 1;
  inst.lam_p = lam_p;
  inst.lam_q = lam_q;
  inst.lam = (double)lam_p / (double)lam_q;
  inst.config = "net:" + path;
  build_hamming_cost(inst.g, inst.xe, inst.ue);
  inst.g.build_rev();
  inst.l_hash = fnv1a(inst.g.nxt.data(), inst.g.nxt.size() * sizeof(u32));
}

// ---------------------------------------------------------------------------
// Result files (consumed by exp17_crosscheck.py)
// ---------------------------------------------------------------------------
inline void write_result(const std::string& path, const Instance& inst,
                         const MasterOut& mo, const ViOut* vi, u64 improper_vi,
                         u64 improper_master, const PiOut* pi) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "META vs " << mo.Vs_size << " bc " << mo.bc_count << " disagree "
    << mo.disagreements << " improper_master " << improper_master
    << " vi_k " << (vi ? (i64)vi->k : -1) << " vi_converged "
    << (vi && vi->converged ? 1 : 0) << " improper_vi "
    << (vi ? (i64)improper_vi : -1) << " pi_steps "
    << (pi ? (i64)pi->steps : -1) << " l_hash " << inst.l_hash << "\n";
  f << "COMPS " << mo.records.size() << "\n";
  for (const auto& r : mo.records) {
    f << "COMP " << r.members.size() << " " << (r.has_cycle ? 1 : 0) << " "
      << (r.sweep_verdict ? 1 : 0) << " " << (r.disagreement ? 1 : 0) << " "
      << r.sweeps << " " << r.strict_sweeps << " "
      << (r.has_margin ? fmt_g17(r.margin) : std::string("none")) << " "
      << r.core_evals << " :";
    for (u32 v : r.members) f << " " << v + 1;
    f << "\n";
  }
  u64 nstate = 0;
  for (u32 x = 0; x < inst.g.N; ++x)
    if (mo.in_R[x]) nstate++;
  f << "STATES " << nstate << "\n";
  for (u32 x = 0; x < inst.g.N; ++x) {
    if (!mo.in_R[x]) continue;
    f << "S " << x + 1 << " " << fmt_g17(mo.J[x]) << " "
      << (mo.mu_u[x] >= 0 ? mo.mu_u[x] + 1 : 0) << " "
      << (mo.mu_y[x] >= 0 ? mo.mu_y[x] + 1 : 0) << "\n";
  }
  if (vi) {
    f << "VI " << nstate << "\n";
    for (u32 x = 0; x < inst.g.N; ++x)
      if (mo.in_R[x]) f << "V " << x + 1 << " " << fmt_g17(vi->V[x]) << "\n";
  } else {
    f << "VI 0\n";
  }
  if (pi) {
    f << "PIV " << nstate << "\n";
    for (u32 x = 0; x < inst.g.N; ++x)
      if (mo.in_R[x]) f << "P " << x + 1 << " " << fmt_g17(pi->V[x]) << "\n";
  } else {
    f << "PIV 0\n";
  }
  f << "END\n";
}

// ---------------------------------------------------------------------------
// CSV emission (header written once per file)
// ---------------------------------------------------------------------------
inline void csv_open(std::ofstream& f, const std::string& path,
                     const std::string& header, const std::string& meta) {
  const bool exists = std::ifstream(path).good();
  f.open(path, std::ios::app);
  if (!f) throw std::runtime_error("cannot open csv " + path);
  if (!exists) {
    f << "# " << meta << "\n";
    f << header << "\n";
  }
}

}  // namespace bcn
