// main.cpp -- exp17 CLI: gen / solve / solve-net / bench.
// All five compared algorithms (master stages, VI, PI, Floyd, DFS) live in
// bcn_core.hpp and are compiled into this single translation unit with one
// set of flags (acceptance standard 1).
#include <cstring>
#include <iostream>
#include <map>
#include <set>

#include "io.hpp"

using namespace bcn;

namespace {

std::string arg_of(int argc, char** argv, const std::string& key,
                   const std::string& dflt = "") {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return dflt;
}
bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (key == argv[i]) return true;
  return false;
}

std::vector<u32> parse_u32_list(const std::string& s) {
  std::vector<u32> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back((u32)std::stoul(item));
  return out;
}

void die(const std::string& msg) {
  std::cerr << "FATAL: " << msg << "\n";
  std::exit(1);
}

void check(bool cond, const std::string& msg) {
  if (!cond) die("bench assertion failed: " + msg);
}

// ---------------------------------------------------------------------------
// One benchmark point: master + VI + PI (+ optional Floyd/DFS on the same
// graph).  Timing starts with the graph in memory (exp16 convention).
// ---------------------------------------------------------------------------
struct PointOut {
  MasterOut mo;
  ViOut vi;
  PiOut pi;
  FloydOut fl;
  DfsOut dfs;
  u64 improper_vi = 0, improper_master = 0;
  bool ran_floyd = false, ran_dfs = false;
  bool floyd_vi_agree = true, dfs_master_agree = true;
};

void run_point(const Instance& inst, PointOut& o, bool run_floyd, bool run_dfs,
               double dfs_timeout, double vi_wall = 300.0) {
  run_master(inst.g, inst.xe, (i32)inst.ue, inst.lam, o.mo);
  o.improper_master =
      improper_count(inst.g, o.mo.in_R, inst.xe, o.mo.mu_u, o.mo.mu_y);
  vi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi, 1e-6, -1, vi_wall);
  std::vector<i32> gu, gy;
  greedy_mu(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi.V, gu, gy);
  o.improper_vi = improper_count(inst.g, o.mo.in_R, inst.xe, gu, gy);
  pi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.pi);
  o.ran_floyd = run_floyd;
  if (run_floyd) {
    floyd_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.fl);
    if (o.fl.status == 0) {
      o.floyd_vi_agree = true;
      for (u32 x = 0; x < inst.g.N; ++x)
        if (o.mo.in_R[x] && !close(o.fl.col[x], o.vi.V[x]))
          o.floyd_vi_agree = false;
    }
  }
  o.ran_dfs = run_dfs;
  if (run_dfs) {
    dfs_run(inst.g, inst.xe, inst.lam, o.dfs, dfs_timeout);
    if (!o.dfs.timed_out) {
      o.dfs_master_agree = true;
      for (u32 x = 0; x < inst.g.N; ++x) {
        const double jm = o.mo.in_R[x] ? o.mo.J[x] : INFV;
        if (!close(o.dfs.J[x], jm)) o.dfs_master_agree = false;
      }
    }
  }
}

// B4 structural assertion: nontrivial SCC records == planted sets; cores
// verdict true, benigns false; disagreements == 0 on dyadic instances.
void assert_planted(const Instance& inst, const MasterOut& mo,
                    const std::string& tag) {
  std::set<std::vector<u32>> want_core, want_benign;
  for (auto s : inst.cores) {
    std::sort(s.begin(), s.end());
    want_core.insert(s);
  }
  for (auto s : inst.benigns) {
    std::sort(s.begin(), s.end());
    want_benign.insert(s);
  }
  std::set<std::vector<u32>> got_core, got_nontrivial;
  for (const auto& r : mo.records) {
    if (r.members.size() > 1) got_nontrivial.insert(r.members);
    if (r.has_cycle) got_core.insert(r.members);
  }
  std::set<std::vector<u32>> want_all = want_core;
  for (const auto& s : want_benign) want_all.insert(s);
  check(got_nontrivial == want_all,
        tag + ": nontrivial SCCs != planted sets");
  check(got_core == want_core, tag + ": beneficial verdicts != planted cores");
  check(mo.bc_count == inst.cores.size(), tag + ": bc_count");
  const bool dyadic = (inst.lam_q & (inst.lam_q - 1)) == 0;
  if (dyadic)
    check(mo.disagreements == 0, tag + ": detector disagreement on dyadic");
}

const char* CSV_HEADER =
    "curve,config,lam_p,lam_q,rep,seed,l_hash,n,N,m,M,core_size,n_cores,"
    "benign_n,benign_size,benign_dropped,t_build,t_reach,t_scc,t_fold,t_sweep,"
    "t_detect,t_core,t_recover,t_book,t_master,t_vi,k_vi,vi_converged,t_pi,k_pi,"
    "bc_count,improper_vi,improper_master,disagreements,min_margin,"
    "t_floyd,k_floyd,floyd_status,floyd_vi_agree,t_dfs,dfs_status,"
    "dfs_master_agree";

std::string csv_meta() {
  return std::string("exp17 bench; Apple clang 17 arm64; flags: ") +
         "-std=c++20 -O2 -DNDEBUG -ffp-contract=off; single-thread; "
         "eps=1e-9 eps_vi=1e-6; timings exclude generation and I/O";
}

void csv_row(std::ofstream& f, const std::string& curve, const Instance& inst,
             const GenParams& p, double t_build, const PointOut& o) {
  const auto& m = o.mo;
  f << curve << "," << inst.config << "," << inst.lam_p << "," << inst.lam_q
    << "," << p.rep << "," << inst.seed << "," << inst.l_hash << "," << p.n
    << "," << inst.g.N << "," << p.m << "," << inst.g.M << ","
    << (p.core_sizes.empty() ? 0 : p.core_sizes[0]) << ","
    << p.core_sizes.size() << "," << p.n_benign << "," << p.benign_size << ","
    << (inst.benign_dropped ? 1 : 0) << "," << fmt_g17(t_build) << ","
    << fmt_g17(m.t_reach) << "," << fmt_g17(m.t_scc) << ","
    << fmt_g17(m.t_fold) << "," << fmt_g17(m.t_sweep) << ","
    << fmt_g17(m.t_detect) << "," << fmt_g17(m.t_core) << ","
    << fmt_g17(m.t_recover) << "," << fmt_g17(m.t_book) << ","
    << fmt_g17(m.t_master) << ","
    << fmt_g17(o.vi.t) << "," << o.vi.k << "," << (o.vi.converged ? 1 : 0)
    << "," << fmt_g17(o.pi.t) << "," << o.pi.steps << "," << m.bc_count << ","
    << o.improper_vi << "," << o.improper_master << "," << m.disagreements
    << "," << (m.has_min_margin ? fmt_g17(m.min_margin) : std::string("na"))
    << ",";
  if (o.ran_floyd)
    f << fmt_g17(o.fl.t) << "," << o.fl.k << "," << o.fl.status << ","
      << (o.fl.status == 0 ? (o.floyd_vi_agree ? 1 : 0) : -1) << ",";
  else
    f << "na,-1,-1,-1,";
  if (o.ran_dfs)
    f << fmt_g17(o.dfs.t) << "," << (o.dfs.timed_out ? "timeout" : "ok") << ","
      << (o.dfs.timed_out ? -1 : (o.dfs_master_agree ? 1 : 0));
  else
    f << "na,na,-1";
  f << "\n";
  f.flush();
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------
int cmd_gen(int argc, char** argv) {
  GenParams p;
  p.n = (u32)std::stoul(arg_of(argc, argv, "--n", "4"));
  p.m = (u32)std::stoul(arg_of(argc, argv, "--m", "2"));
  p.core_sizes = parse_u32_list(arg_of(argc, argv, "--cores", ""));
  p.n_benign = (u32)std::stoul(arg_of(argc, argv, "--benign", "0"));
  p.benign_size = (u32)std::stoul(arg_of(argc, argv, "--bsize", "0"));
  p.lam_p = std::stoll(arg_of(argc, argv, "--lamp", "1"));
  p.lam_q = std::stoll(arg_of(argc, argv, "--lamq", "2"));
  p.rep = (u32)std::stoul(arg_of(argc, argv, "--rep", "0"));
  const std::string out = arg_of(argc, argv, "--out");
  if (out.empty()) die("gen: --out required");
  Instance inst;
  generate(p, inst);
  write_instance(out, inst);
  std::cout << "gen ok seed=" << inst.seed << " l_hash=" << inst.l_hash
            << " benign_dropped=" << (inst.benign_dropped ? 1 : 0) << "\n";
  return 0;
}

int cmd_solve(int argc, char** argv) {
  const std::string in = arg_of(argc, argv, "--inst");
  const std::string out = arg_of(argc, argv, "--out");
  if (in.empty() || out.empty()) die("solve: --inst/--out required");
  const bool baselines = !has_flag(argc, argv, "--no-baselines");
  Instance inst;
  read_instance(in, inst);
  PointOut o;
  run_master(inst.g, inst.xe, (i32)inst.ue, inst.lam, o.mo);
  o.improper_master =
      improper_count(inst.g, o.mo.in_R, inst.xe, o.mo.mu_u, o.mo.mu_y);
  if (baselines) {
    vi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi);
    std::vector<i32> gu, gy;
    greedy_mu(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi.V, gu, gy);
    o.improper_vi = improper_count(inst.g, o.mo.in_R, inst.xe, gu, gy);
    pi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.pi);
  }
  write_result(out, inst, o.mo, baselines ? &o.vi : nullptr, o.improper_vi,
               o.improper_master, baselines ? &o.pi : nullptr);
  std::cout << "solve ok vs=" << o.mo.Vs_size << " bc=" << o.mo.bc_count
            << " improper_master=" << o.improper_master << "\n";
  return 0;
}

int cmd_solve_net(int argc, char** argv) {
  const std::string net = arg_of(argc, argv, "--net");
  const std::string out = arg_of(argc, argv, "--out");
  if (net.empty() || out.empty()) die("solve-net: --net/--out required");
  Instance inst;
  read_network(net, inst, (u32)std::stoul(arg_of(argc, argv, "--xe", "1")),
               (u32)std::stoul(arg_of(argc, argv, "--ue", "1")),
               std::stoll(arg_of(argc, argv, "--lamp", "1")),
               std::stoll(arg_of(argc, argv, "--lamq", "2")));
  PointOut o;
  run_master(inst.g, inst.xe, (i32)inst.ue, inst.lam, o.mo);
  o.improper_master =
      improper_count(inst.g, o.mo.in_R, inst.xe, o.mo.mu_u, o.mo.mu_y);
  vi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi);
  std::vector<i32> gu, gy;
  greedy_mu(inst.g, o.mo.in_R, inst.xe, inst.lam, o.vi.V, gu, gy);
  o.improper_vi = improper_count(inst.g, o.mo.in_R, inst.xe, gu, gy);
  pi_run(inst.g, o.mo.in_R, inst.xe, inst.lam, o.pi);
  write_result(out, inst, o.mo, &o.vi, o.improper_vi, o.improper_master,
               &o.pi);
  std::cout << "solve-net ok vs=" << o.mo.Vs_size << " bc=" << o.mo.bc_count
            << " improper_vi=" << o.improper_vi
            << " improper_master=" << o.improper_master
            << " t_master=" << fmt_g17(o.mo.t_master) << "\n";
  return 0;
}

int cmd_bench(int argc, char** argv) {
  const std::string curve = arg_of(argc, argv, "--curve");
  const std::string out = arg_of(argc, argv, "--out");
  const bool smoke = has_flag(argc, argv, "--smoke");
  if (curve.empty() || out.empty()) die("bench: --curve/--out required");

  auto run_grid = [&](const std::vector<GenParams>& grid, bool floyd, bool dfs,
                      double dfs_timeout) {
    std::ofstream f;
    csv_open(f, out, CSV_HEADER, csv_meta());
    // one full-size warmup (first grid point) to settle pages / clocks
    if (!grid.empty()) {
      Instance w;
      generate(grid.back(), w);
      PointOut o;
      run_point(w, o, false, false, 0.0);
    }
    for (const auto& p : grid) {
      Instance inst;
      const double t0 = now_s();
      generate(p, inst);
      const double t_build = now_s() - t0;
      PointOut o;
      run_point(inst, o, floyd, dfs, dfs_timeout);
      assert_planted(inst, o.mo, curve + "/" + inst.config +
                                     "/rep=" + std::to_string(p.rep));
      csv_row(f, curve, inst, p, t_build, o);
      std::cout << curve << " " << inst.config << " lam=" << inst.lam_p << "/"
                << inst.lam_q << " rep=" << p.rep
                << " t_master=" << fmt_g17(o.mo.t_master)
                << " t_vi=" << fmt_g17(o.vi.t) << " k_vi=" << o.vi.k
                << " bc=" << o.mo.bc_count << "\n";
    }
  };

  const u32 reps = smoke ? 1 : 5;
  if (curve == "a") {
    std::vector<GenParams> grid;
    const std::vector<u32> ns =
        smoke ? std::vector<u32>{10, 12} : std::vector<u32>{10, 12, 14, 16, 18, 20};
    const std::vector<u32> cores = smoke ? std::vector<u32>{12}
                                         : std::vector<u32>{12, 16, 20};
    for (u32 core : cores)
      for (u32 n : ns)
        for (u32 r = 0; r < reps; ++r) {
          GenParams p;
          p.n = n;
          p.m = 2;
          p.core_sizes = {core};
          p.n_benign = 2;
          p.benign_size = 32;
          p.lam_p = 1;
          p.lam_q = 2;
          p.rep = r;
          grid.push_back(p);
        }
    run_grid(grid, false, false, 0.0);
  } else if (curve == "ap") {
    std::vector<GenParams> grid;
    const std::vector<u32> ks =
        smoke ? std::vector<u32>{1, 2} : std::vector<u32>{1, 2, 4, 8};
    for (u32 k : ks)
      for (u32 r = 0; r < reps; ++r) {
        GenParams p;
        p.n = 16;
        p.m = 2;
        p.core_sizes.assign(k, 16);
        p.n_benign = 2;
        p.benign_size = 32;
        p.lam_p = 1;
        p.lam_q = 2;
        p.rep = r;
        grid.push_back(p);
      }
    run_grid(grid, false, false, 0.0);
  } else if (curve == "b") {
    std::vector<GenParams> grid;
    const std::vector<std::pair<i64, i64>> lams =
        smoke ? std::vector<std::pair<i64, i64>>{{1, 2}, {9, 10}}
              : std::vector<std::pair<i64, i64>>{
                    {1, 2}, {9, 10}, {99, 100}, {999, 1000}, {9999, 10000}};
    for (auto [pp, qq] : lams)
      for (u32 r = 0; r < reps; ++r) {
        GenParams p;
        p.n = smoke ? 10 : 14;
        p.m = 2;
        p.core_sizes = {16};
        p.n_benign = 2;
        p.benign_size = 32;
        p.lam_p = pp;
        p.lam_q = qq;
        p.rep = r;
        grid.push_back(p);
      }
    run_grid(grid, false, false, 0.0);
  } else if (curve == "c") {
    std::vector<GenParams> grid;
    const std::vector<u32> ns = smoke
                                    ? std::vector<u32>{4, 6, 8}
                                    : std::vector<u32>{4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (u32 n : ns)
      for (u32 r = 0; r < reps; ++r) {
        GenParams p;
        p.n = n;
        p.m = 2;
        p.core_sizes = {8};
        p.n_benign = 1;
        p.benign_size = 8;
        p.lam_p = 1;
        p.lam_q = 2;
        p.rep = r;
        grid.push_back(p);
      }
    run_grid(grid, true, true, smoke ? 5.0 : 60.0);
  } else if (curve == "tlgl") {
    const std::string net =
        arg_of(argc, argv, "--net", "../networks/t_lgl.txt");
    const std::string dump = arg_of(argc, argv, "--dump", "");
    Instance inst;
    read_network(net, inst, 65535, 3, 1, 2);  // exp16 t_lgl row config
    std::ofstream f;
    csv_open(f, out, CSV_HEADER, csv_meta());
    const u32 nreps = smoke ? 2 : 5;
    PointOut o;
    run_point(inst, o, false, false, 0.0);  // warmup (also sanity numbers)
    std::vector<double> t_masters;
    for (u32 r = 0; r < nreps; ++r) {
      PointOut oo;
      run_point(inst, oo, false, false, 0.0);
      // B8 hard assertions
      check(oo.mo.bc_count == 2, "tlgl bc_count == 2");
      for (const auto& rec : oo.mo.records)
        if (rec.has_cycle)
          check(rec.members.size() == 20, "tlgl core |S| == 20");
      check(oo.improper_vi == 1216, "tlgl improper_vi == 1216");
      check(oo.improper_master == 0, "tlgl improper_master == 0");
      GenParams p;
      p.n = 16;
      p.m = 3;
      p.rep = r;
      csv_row(f, "tlgl", inst, p, 0.0, oo);
      t_masters.push_back(oo.mo.t_master);
      std::cout << "tlgl rep=" << r << " t_master=" << fmt_g17(oo.mo.t_master)
                << " t_vi=" << fmt_g17(oo.vi.t) << " k_vi=" << oo.vi.k
                << " improper_vi=" << oo.improper_vi << "\n";
      if (r == nreps - 1 && !dump.empty())
        write_result(dump, inst, oo.mo, &oo.vi, oo.improper_vi,
                     oo.improper_master, &oo.pi);
    }
    std::sort(t_masters.begin(), t_masters.end());
    const double med = t_masters[t_masters.size() / 2];
    check(med < 2.0, "tlgl t_master median < 2 s");
    std::cout << "tlgl median t_master=" << fmt_g17(med) << "\n";
  } else {
    die("unknown curve " + curve);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bcn gen|solve|solve-net|bench ...\n";
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    if (cmd == "gen") return cmd_gen(argc, argv);
    if (cmd == "solve") return cmd_solve(argc, argv);
    if (cmd == "solve-net") return cmd_solve_net(argc, argv);
    if (cmd == "bench") return cmd_bench(argc, argv);
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
