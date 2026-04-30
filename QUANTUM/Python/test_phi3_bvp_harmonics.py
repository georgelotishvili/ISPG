"""Phi^3 oscillon harmonics via BVP solver, seeded by shooting profiles."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp


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


def count_nodes(Phi_vals):
    return np.sum(np.diff(np.sign(Phi_vals)) != 0)


def find_phi3_bvp(Phi0, n_target, r_max=25.0, Omega_guess=None,
                   r_seed=None, y_seed=None):
    """BVP solver for Phi^3 oscillon, optionally seeded by shooting."""

    def ode(r, y, p):
        Omega = p[0]
        Phi = y[0]
        dPhi = y[1]
        r_safe = np.maximum(r, 1e-8)
        d2Phi = -(2.0 / r_safe) * dPhi - (Omega**2 - 1) * Phi - Phi**3
        d2Phi_origin = -(Omega**2 - 1) * Phi / 3.0 - Phi**3 / 3.0
        d2Phi = np.where(r < 1e-8, d2Phi_origin, d2Phi)
        return np.vstack([dPhi, d2Phi])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    N = 500

    if r_seed is not None and y_seed is not None:
        from scipy.interpolate import interp1d
        r_init = np.linspace(1e-6, r_max, N)
        f0 = interp1d(r_seed, y_seed[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_seed, y_seed[1], fill_value=0.0, bounds_error=False)
        Phi_init = f0(r_init)
        dPhi_init = f1(r_init)
        Phi_init[-1] = 0.0
        Phi_init[0] = Phi0
    else:
        r_init = np.linspace(1e-6, r_max, N)
        kappa_g = np.sqrt(max(0.01, 1.0 - Omega_guess**2))
        envelope = Phi0 / np.cosh(r_init * kappa_g)**2
        if n_target == 0:
            Phi_init = envelope
        else:
            r_core = min(r_max * 0.5, 5.0 / kappa_g)
            Phi_init = envelope * np.cos(n_target * np.pi * r_init / r_core)
        dPhi_init = np.gradient(Phi_init, r_init)

    y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r_init, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=30000, verbose=0)

    if sol.success:
        nodes_found = count_nodes(sol.y[0])
        return sol.p[0], sol, nodes_found
    return None, None, -1


def total_energy(sol, Omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)


print("=" * 70)
print("  Phi^3 Oscillon Harmonics via Shooting-Seeded BVP")
print("=" * 70)

R_MAX = 25.0

for Phi0 in [2.0, 3.0, 5.0, 8.0, 10.0]:
    print(f"\n{'='*60}")
    print(f"  Phi0 = {Phi0:.1f}")
    print(f"{'='*60}")

    # Phase 1: node-count scan
    print("  Scanning node counts...", flush=True)
    grid = np.linspace(0.9995, 0.02, 300)
    node_map = {}
    for Om in grid:
        sol = integrate_phi3(Phi0, Om, R_MAX)
        if sol is not None:
            n = count_nodes(sol.y[0])
            if n not in node_map:
                node_map[n] = []
            node_map[n].append((Om, sol))

    max_n = max(node_map.keys()) if node_map else 0
    print(f"  Max nodes observed: {max_n}")
    for n in sorted(node_map.keys()):
        oms = [x[0] for x in node_map[n]]
        print(f"    n={n}: {len(oms)} points, Omega=[{min(oms):.4f}, {max(oms):.4f}]")

    # Phase 2: BVP solve for each harmonic
    print(f"\n  BVP solutions:")
    print(f"  {'n':>4} {'Omega':>12} {'kappa':>8} {'E':>14} {'nodes':>6}")
    print(f"  {'-'*48}")

    states = []
    for n_target in sorted(node_map.keys()):
        entries = node_map[n_target]
        mid_idx = len(entries) // 2
        Om_mid, sol_mid = entries[mid_idx]

        Om_bvp, sol_bvp, n_found = find_phi3_bvp(
            Phi0, n_target, R_MAX,
            Omega_guess=Om_mid,
            r_seed=sol_mid.t, y_seed=sol_mid.y)

        if Om_bvp is not None and 0.01 < Om_bvp < 0.999:
            E = total_energy(sol_bvp, Om_bvp)
            kappa = np.sqrt(1.0 - Om_bvp**2)
            states.append((n_found, Om_bvp, E, kappa))
            print(f"  {n_found:4d} {Om_bvp:12.8f} {kappa:8.4f} {E:14.6f} {n_found:6d}")
        else:
            for entry_Om, entry_sol in entries:
                Om_bvp, sol_bvp, n_found = find_phi3_bvp(
                    Phi0, n_target, R_MAX,
                    Omega_guess=entry_Om,
                    r_seed=entry_sol.t, y_seed=entry_sol.y)
                if Om_bvp is not None and 0.01 < Om_bvp < 0.999:
                    E = total_energy(sol_bvp, Om_bvp)
                    kappa = np.sqrt(1.0 - Om_bvp**2)
                    states.append((n_found, Om_bvp, E, kappa))
                    print(f"  {n_found:4d} {Om_bvp:12.8f} {kappa:8.4f}"
                          f" {E:14.6f} {n_found:6d}")
                    break
            else:
                print(f"  {n_target:4d} {'FAILED':>12}")

    states.sort(key=lambda x: x[0])

    if len(states) >= 2:
        print(f"\n  ENERGY SPECTRUM:")
        E0 = states[0][2] if states[0][2] > 0 else 1.0
        for n, Om, E, kappa in states:
            print(f"    n={n}: E={E:.6f}, E/E0={E/E0:.4f}, Om={Om:.8f}")

    if len(states) >= 3:
        from itertools import combinations
        def koide(m1, m2, m3):
            return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

        energies = sorted([s[2] for s in states if s[2] > 0])
        if len(energies) >= 3:
            best_Q_diff = 999
            for combo in combinations(energies, 3):
                e = sorted(combo)
                Q = koide(*e)
                if abs(Q - 2/3) < best_Q_diff:
                    best_Q_diff = abs(Q - 2/3)
                    best_Q = Q
                    best_e = e

            print(f"\n  KOIDE Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
            print(f"    E = {best_e[0]:.4f}, {best_e[1]:.4f}, {best_e[2]:.4f}")
            r1 = best_e[1]/best_e[0]
            r2 = best_e[2]/best_e[0]
            print(f"    Ratios: 1 : {r1:.2f} : {r2:.2f}")
            print(f"    Target: 1 : 206.8 : 3477.2")
