"""Find Phi^3 oscillon harmonics: fast two-phase approach.
Phase 1: Coarse node-count scan to locate transition Omega ranges.
Phase 2: Precise brentq within each n-constant region for Phi(r_max)=0.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def integrate_phi3(Phi0, Omega, r_max=25.0):
    kappa2 = 1.0 - Omega**2
    if kappa2 <= 0:
        return None

    def rhs(r, y):
        Phi, dPhi = y
        if r < 1e-10:
            d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**3 / 3.0
        else:
            d2Phi = -(2.0 / r) * dPhi - (Omega**2 - 1) * Phi - Phi**3
        return [dPhi, d2Phi]

    sol = solve_ivp(rhs, [1e-10, r_max], [Phi0, 0.0],
                    rtol=1e-10, atol=1e-12, max_step=0.2)
    return sol if sol.success else None


def count_nodes(sol):
    return np.sum(np.diff(np.sign(sol.y[0])) != 0)


def total_energy(sol, Omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)


print("=" * 70)
print("  Phi^3 Oscillon Harmonics via Shooting (optimized)")
print("=" * 70)

R_MAX = 25.0

for Phi0 in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    print(f"\n{'='*50}")
    print(f"  Phi0 = {Phi0:.1f}")
    print(f"{'='*50}")

    # Phase 1: coarse scan
    print("  Phase 1: node count scan...", flush=True)
    grid = np.linspace(0.9999, 0.02, 500)
    scan = []
    for Om in grid:
        sol = integrate_phi3(Phi0, Om, R_MAX)
        if sol is not None:
            scan.append((Om, sol.y[0, -1], count_nodes(sol)))
        else:
            scan.append((Om, np.inf, -1))

    max_nodes = max(s[2] for s in scan)
    print(f"  Max node count observed: {max_nodes}")

    # Find Omega ranges for each node count
    node_ranges = {}
    for Om, val, n in scan:
        if n >= 0:
            if n not in node_ranges:
                node_ranges[n] = [Om, Om]
            else:
                node_ranges[n][0] = min(node_ranges[n][0], Om)
                node_ranges[n][1] = max(node_ranges[n][1], Om)

    for n in sorted(node_ranges.keys()):
        lo, hi = node_ranges[n]
        print(f"    n={n}: Omega in [{lo:.5f}, {hi:.5f}]")

    # Phase 2: find Phi(r_max)=0 within each node-count region
    print(f"\n  Phase 2: finding Phi(r_max)=0 for each harmonic...")
    print(f"  {'n':>4} {'Omega':>12} {'kappa':>8} {'E':>14} {'Phi_end':>12}")
    print(f"  {'-'*52}")

    states = []
    for n_target in sorted(node_ranges.keys()):
        lo, hi = node_ranges[n_target]

        # Finer scan within the node range to find sign changes of Phi_end
        fine_grid = np.linspace(hi, lo, 200)
        fine_vals = []
        for Om in fine_grid:
            sol = integrate_phi3(Phi0, Om, R_MAX)
            if sol is not None and count_nodes(sol) == n_target:
                fine_vals.append((Om, sol.y[0, -1]))

        if len(fine_vals) < 2:
            print(f"  {n_target:4d} {'not enough pts':>12}")
            continue

        # Find sign changes
        found_for_n = False
        for i in range(1, len(fine_vals)):
            Om_prev, v_prev = fine_vals[i-1]
            Om_curr, v_curr = fine_vals[i]
            if v_prev * v_curr < 0:
                try:
                    def shoot_f(Om_try):
                        s = integrate_phi3(Phi0, Om_try, R_MAX)
                        if s is None:
                            return 1e10
                        return s.y[0, -1]

                    Om_star = brentq(shoot_f, Om_prev, Om_curr, xtol=1e-12)
                    sol_star = integrate_phi3(Phi0, Om_star, R_MAX)
                    if sol_star is not None:
                        n_chk = count_nodes(sol_star)
                        Phi_end = sol_star.y[0, -1]
                        if abs(Phi_end) < 0.01:
                            E = total_energy(sol_star, Om_star)
                            kappa = np.sqrt(1.0 - Om_star**2)
                            already = any(abs(Om_star - s[1]) < 0.0001 for s in states)
                            if not already:
                                states.append((n_chk, Om_star, E, kappa))
                                print(f"  {n_chk:4d} {Om_star:12.8f} {kappa:8.4f}"
                                      f" {E:14.6f} {Phi_end:12.2e}")
                                found_for_n = True
                except (ValueError, RuntimeError):
                    pass

        if not found_for_n:
            print(f"  {n_target:4d} {'no root found':>12}")

    states.sort(key=lambda x: x[0])

    if len(states) >= 2:
        print(f"\n  ENERGY SPECTRUM:")
        E0 = states[0][2]
        for n, Om, E, kappa in states:
            print(f"    n={n}: E = {E:.4f}  (E/E0 = {E/E0:.4f})")

    if len(states) >= 3:
        from itertools import combinations
        def koide(m1, m2, m3):
            return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

        energies = sorted([s[2] for s in states])
        best_Q_diff = 999
        for combo in combinations(energies, 3):
            e = sorted(combo)
            Q = koide(*e)
            if abs(Q - 2/3) < best_Q_diff:
                best_Q_diff = abs(Q - 2/3)
                best_Q = Q
                best_e = e

        print(f"\n  KOIDE: Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
        print(f"    E = {best_e[0]:.4f}, {best_e[1]:.4f}, {best_e[2]:.4f}")
        print(f"    Ratios: 1 : {best_e[1]/best_e[0]:.2f} : {best_e[2]/best_e[0]:.2f}")
        print(f"    Target: 1 : 206.8 : 3477.2")
