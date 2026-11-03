"""
数值验证：在 ara operon 网络上检验 §3B 的良定条件 (W) 阈值。

只用 Gao 仓库的网络动态 L（真实），不修改其源码。
我们按"镇定问题"重新设定：挑控制不动点作 x_e，配非负、x_e 处为 0 的代价。

输出：
  - 控制不动点个数与列表
  - 对每个不动点 x_e：可镇定集 |R(x_e)|、镇定深度 D_reach
  - (W) 充分条件阈值 gamma* 与 gamma=0.5/0.6 的对比（含代价异质度敏感性）
"""
from collections import deque
import os

REPO = os.path.join(os.path.dirname(__file__),
                    "..", "literature", "04_optimal_cost_stabilization",
                    "Gao2020_bcn_opt_dc_repo", "code", "networks", "ara_operon.txt")


def read_network(path):
    with open(path) as f:
        n = int(f.readline().strip())
        m = int(f.readline().strip())
        nums = f.readline().strip().split()
        L = [int(x) for x in nums]
    N, M = 2 ** n, 2 ** m
    assert len(L) == M * N, (len(L), M * N)
    return n, m, N, M, L


def next_state(L, N, x, u):
    # x,u 1-indexed; 返回 1-indexed 后继
    return L[(u - 1) * N + (x - 1)]


def inverse_map(i, n):
    # 复刻 Gao ara_operon.py：把 delta_{2^n}^i 转成 0/1 逻辑值
    r = []
    while n > 0:
        if i % 2 == 0:
            r.append(0); i = i // 2
        else:
            r.append(1); i = (i + 1) // 2
        n -= 1
    r.reverse()
    return r


def g_gao(i, k, n=9, m=4):
    X = inverse_map(i, n)
    U = inverse_map(k, m)
    A = [-28, -12, 12, 16, 0, 0, 0, 20, 16]
    B = [-8, 40, 20, 40]
    return sum(a * x for a, x in zip(A, X)) + sum(b * u for b, u in zip(B, U))


def main():
    n, m, N, M, L = read_network(os.path.normpath(REPO))
    print(f"网络: n={n}, m={m}, N={N}, M={M}, len(L)={len(L)}")

    # 1) 控制不动点：存在 u 使 next_state(x,u)=x
    fixpoints = {}
    for x in range(1, N + 1):
        us = [u for u in range(1, M + 1) if next_state(L, N, x, u) == x]
        if us:
            fixpoints[x] = us
    print(f"\n控制不动点个数: {len(fixpoints)}")
    for x, us in fixpoints.items():
        print(f"  x_e={x} (bits={inverse_map(x, n)}), 自保持控制 u={us}")

    # 2) 预构造前向邻接（状态图，边 x->x' 若存在 u）和反向邻接
    fwd = [set() for _ in range(N + 1)]
    rev = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u in range(1, M + 1):
            y = next_state(L, N, x, u)
            fwd[x].add(y)
            rev[y].add(x)

    # 3) 对每个不动点：反向 BFS 求 R(x_e) 与 D_reach
    def reach_stats(xe):
        dist = {xe: 0}
        dq = deque([xe])
        while dq:
            v = dq.popleft()
            for p in rev[v]:
                if p not in dist:
                    dist[p] = dist[v] + 1
                    dq.append(p)
        R = set(dist)
        Dreach = max(dist.values())
        return R, Dreach, dist

    print("\n各不动点的可镇定集与深度:")
    results = {}
    for xe in fixpoints:
        R, Dreach, dist = reach_stats(xe)
        results[xe] = (len(R), Dreach)
        print(f"  x_e={xe}: |R|={len(R)} / N={N},  D_reach={Dreach}")

    # 4) (W) 阈值：gamma > (1 - delta/cmax)^(1/Dreach)
    #    等价: gamma=0.5/0.6 满足充分条件所需的 delta/cmax > 1 - gamma^Dreach
    print("\n(W) 充分条件阈值分析:")
    print("  关系: gamma > (1 - delta/cmax)^(1/D_reach)")
    print("  即固定 gamma 时需要 代价异质比 delta/cmax > 1 - gamma^D_reach")
    for xe, (Rsize, Dreach) in results.items():
        print(f"\n  --- x_e={xe}, D_reach={Dreach}, |R|={Rsize} ---")
        for gamma in (0.5, 0.6, 0.9, 0.99):
            need = 1 - gamma ** Dreach
            print(f"    gamma={gamma}: 需 delta/cmax > {need:.6f}"
                  + ("  (几乎要求均匀代价)" if need > 0.99 else ""))

    # 5) 具体代价示例：用 Gao 的 g 平移成非负 c = g - g_min，
    #    取 x_e 为"最便宜不动点对"(g(x_e,u_e) 最小)，看是否能让 c(x_e,u_e)=0
    print("\n具体代价（Gao g 平移）下的检验:")
    # 全局 g 范围
    all_g = []
    for x in range(1, N + 1):
        for u in range(1, M + 1):
            all_g.append(g_gao(x, u, n, m))
    gmin, gmax = min(all_g), max(all_g)
    print(f"  Gao g 范围: [{gmin}, {gmax}]  (Rmax≈{gmax-gmin})")
    # 不动点对的 g
    fp_pairs = [(x, u, g_gao(x, u, n, m)) for x, us in fixpoints.items() for u in us]
    fp_pairs.sort(key=lambda t: t[2])
    print("  不动点对 (x_e,u_e,g) 按 g 升序前几个:")
    for x, u, gv in fp_pairs[:5]:
        print(f"    x_e={x}, u_e={u}, g={gv}")
    # 取最便宜不动点对为 x_e；c(x,u)=g(x,u)-g(x_e,u_e) 可能为负 -> 需平移到全局 gmin
    xe_best, ue_best, g_best = fp_pairs[0]
    print(f"  选 x_e={xe_best}, u_e={ue_best}, g(x_e,u_e)={g_best}")
    print(f"  若令 c=g-gmin(全局非负): c(x_e,u_e)={g_best-gmin} "
          + ("=0, 合法" if g_best == gmin else "≠0, 该不动点非全局最便宜->需另设代价"))

    # 6) 直接经验性检验 (W)：用一个合法的镇定代价，把无约束折扣 VI 解到收敛，
    #    取贪婪策略，模拟所有状态，检查是否全部镇定到 x_e（= 最优策略 proper = (W) 成立）
    print("\n" + "=" * 60)
    print("经验性检验 (W)：最优(贪婪)策略是否对所有状态镇定")
    print("=" * 60)

    def hamming_cost(xe, ue, w_state=1, w_ctrl=1):
        bx_e = inverse_map(xe, n)
        def c(x, u):
            bx = inverse_map(x, n)
            ds = sum(1 for a, b in zip(bx, bx_e) if a != b)
            dc = 0 if u == ue else w_ctrl
            return w_state * ds + dc
        return c

    def solve_vi(c, gamma, tol=1e-9, maxit=100000):
        V = [0.0] * (N + 1)
        nit = 0
        while True:
            nit += 1
            psi = 0.0
            for x in range(1, N + 1):
                best = min(c(x, u) + gamma * V[next_state(L, N, x, u)]
                           for u in range(1, M + 1))
                psi = max(psi, abs(best - V[x]))
                V[x] = best
            if psi < tol or nit >= maxit:
                break
        pol = [0] * (N + 1)
        for x in range(1, N + 1):
            _, ub = min((c(x, u) + gamma * V[next_state(L, N, x, u)], u)
                        for u in range(1, M + 1))
            pol[x] = ub
        return V, pol, nit

    def check_proper(pol, xe):
        # 沿贪婪策略模拟，检查每个 x 是否在 <=N 步到达 xe 并停留
        proper = 0
        worst_depth = 0
        for x in range(1, N + 1):
            cur, steps, seen = x, 0, set()
            ok = False
            while steps <= N + 1:
                if cur == xe:
                    ok = True
                    break
                if cur in seen:
                    break  # 进入不含 xe 的环 -> 不镇定
                seen.add(cur)
                cur = next_state(L, N, cur, pol[cur])
                steps += 1
            if ok:
                proper += 1
                worst_depth = max(worst_depth, steps)
        return proper, worst_depth

    # 选 x_e=512(全零, D_reach=3), u_e=11
    xe, ue = 512, 11
    print(f"\n代价模型: c(x,u)=Hamming(bits(x),bits(x_e)) + [u!=u_e],  x_e={xe}, u_e={ue}")
    print(f"  delta=1, c_max=9+1=10, delta/cmax=0.1")
    for gamma in (0.3, 0.5, 0.6, 0.9, 0.99):
        c = hamming_cost(xe, ue)
        V, pol, nit = solve_vi(c, gamma)
        proper, wd = check_proper(pol, xe)
        thr_need = 1 - gamma ** 3  # D_reach=3
        certified = 0.1 > thr_need
        print(f"  gamma={gamma}: VI收敛{nit}轮, 镇定状态数={proper}/{N}, "
              f"最坏镇定深度={wd}, (W){'成立' if proper==N else '失败'}; "
              f"阈值需 delta/cmax>{thr_need:.3f} -> 充分条件{'满足' if certified else '不满足(但经验上仍可能成立)'}")

    # 7) 对抗性代价：制造一个"便宜诱饵不动点"，看 (W) 是否会失败（验证条件非平凡）
    print("\n" + "-" * 60)
    print("对抗性代价：诱饵不动点 x_decoy=9 几乎免费，看 (W) 是否失败")
    print("-" * 60)
    xe2, ue2 = 512, 11
    decoy = 9  # 另一不动点（自保持 u=2）
    eps = 0.01
    def adv_cost(x, u):
        if x == xe2 and u == ue2:
            return 0.0
        if x == decoy:      # 诱饵：停在此处几乎免费
            return eps
        return 1.0
    for gamma in (0.3, 0.5, 0.6, 0.9, 0.99):
        V, pol, nit = solve_vi(adv_cost, gamma)
        proper, wd = check_proper(pol, xe2)
        # 诱饵处最优动作是否自保持(锁死)
        decoy_self = (next_state(L, N, decoy, pol[decoy]) == decoy)
        print(f"  gamma={gamma}: 镇定状态数={proper}/{N}, "
              f"诱饵处自锁={'是' if decoy_self else '否'}, "
              f"(W){'成立' if proper==N else '失败(诱饵截留)'}")


if __name__ == "__main__":
    main()
