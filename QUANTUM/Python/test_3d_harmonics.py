"""Search for 3D oscillon harmonics (excited states with n nodes)."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp

def find_oscillon_harmonic(Phi0, n_nodes, r_max=35.0, Omega_guess=None):
    """Find 3D oscillon solution with exactly n_nodes zero crossings."""

    if Omega_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Omega_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))
        if n_nodes > 0:
            Omega_guess = max(0.05, Omega_guess - 0.1 * n_nodes)

    def ode(r, y, p):
        Omega = p[0]
        Phi = y[0]
        dPhi = y[1]
        r_safe = np.maximum(r, 1e-8)
        d2Phi = -(2.0 / r_safe) * dPhi - (Omega**2 - 1) * Phi - Phi**2
        d2Phi_origin = -(Omega**2 - 1) * Phi / 3.0 - Phi**2 / 3.0
        d2Phi = np.where(r < 1e-8, d2Phi_origin, d2Phi)
        return np.vstack([dPhi, d2Phi])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    r_init = np.linspace(1e-6, r_max, 500)

    kappa_g = np.sqrt(max(0.01, 1.0 - Omega_guess**2))

    if n_nodes == 0:
        Phi_init = Phi0 / np.cosh(r_init * kappa_g)**2
    else:
        envelope = Phi0 / np.cosh(r_init * kappa_g)**2
        r_core = min(r_max * 0.7, 8.0 / kappa_g)
        oscillation = np.cos(n_nodes * np.pi * r_init / r_core)
        Phi_init = envelope * oscillation

    dPhi_init = np.gradient(Phi_init, r_init)
    y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r_init, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)

    if sol.success:
        Phi_sol = sol.y[0]
        sign_changes = np.sum(np.diff(np.sign(Phi_sol)) != 0)
        return sol.p[0], sol, sign_changes
    return None, None, -1


def total_energy_3d(sol, Omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)


print("=" * 70)
print("  3D Oscillon Harmonics Search")
print("  (excited states with n=0,1,2,3... nodes)")
print("=" * 70)

test_Phi0s = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
max_harmonics = 6

for Phi0 in test_Phi0s:
    print(f"\n  Phi0 = {Phi0:.1f}")
    print(f"  {'n':>4} {'Omega':>10} {'kappa':>8} {'E_total':>12} {'nodes':>6} {'status':>10}")
    print(f"  {'-'*52}")

    found = []

    for n_target in range(max_harmonics):
        Om_guesses = []

        kappa_base = np.sqrt(min(Phi0 / 4.2, 0.95))
        Om_base = np.sqrt(max(0.01, 1.0 - kappa_base**2))

        for shift in np.linspace(0.0, 0.8, 20):
            Om_guesses.append(max(0.05, Om_base - shift))
        for Om_try in np.linspace(0.05, 0.99, 30):
            Om_guesses.append(Om_try)

        best_sol = None
        best_Om = None
        best_nodes = -1
        best_diff = 999

        for Om_g in Om_guesses:
            Om, sol, nodes = find_oscillon_harmonic(
                Phi0, n_target, r_max=35.0, Omega_guess=Om_g)

            if Om is not None and 0.01 < Om < 0.999:
                diff = abs(nodes - n_target)
                already_found = any(abs(Om - f[0]) < 0.001 for f in found)

                if nodes == n_target and not already_found:
                    if best_sol is None or diff < best_diff:
                        best_Om = Om
                        best_sol = sol
                        best_nodes = nodes
                        best_diff = diff

        if best_sol is not None:
            E = total_energy_3d(best_sol, best_Om)
            kappa = np.sqrt(1.0 - best_Om**2)
            found.append((best_Om, E, best_nodes))
            print(f"  {n_target:4d} {best_Om:10.6f} {kappa:8.4f} {E:12.4f}"
                  f" {best_nodes:6d}     found")
        else:
            print(f"  {n_target:4d} {'':>10} {'':>8} {'':>12} {'':>6}   not found")

    if len(found) >= 2:
        print(f"\n  Energy ratios (relative to ground state):")
        E0 = found[0][1]
        for i, (Om, E, n) in enumerate(found):
            print(f"    n={n}: E={E:.4f}, E/E0={E/E0:.4f}")

    if len(found) >= 3:
        from itertools import combinations
        def koide(m1, m2, m3):
            return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

        energies = sorted([f[1] for f in found])
        best_Q_diff = 999
        best_Q = None
        for combo in combinations(energies, 3):
            e = sorted(combo)
            Q = koide(*e)
            if abs(Q - 2/3) < best_Q_diff:
                best_Q_diff = abs(Q - 2/3)
                best_Q = Q
                best_e = e

        print(f"\n  Best Koide Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
        print(f"    Using E = {best_e[0]:.4f}, {best_e[1]:.4f}, {best_e[2]:.4f}")
        print(f"    Ratios: 1 : {best_e[1]/best_e[0]:.2f} : {best_e[2]/best_e[0]:.2f}")
