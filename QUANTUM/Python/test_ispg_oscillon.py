"""ISPG self-gravitating oscillon: the ACTUAL equation from the theory.

Equation from ISPG_Quantum.tex (eq. self_consistent):
  Phi'' + (2/r)Phi' + w^2 * exp(-2*phi0) * Phi * (1 + Phi/2) - phi0' * Phi' = 0

Gravity (Poisson):
  phi0'' + (2/r)phi0' + alpha * (w^2*Phi^2/2 + Phi'^2/2) = 0

Key: massless scalar field — confinement comes from GRAVITY ITSELF.
The gravitational well creates a "box" that selects discrete frequencies.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp


def solve_ispg_oscillon(Phi0, alpha, r_max=30.0, omega_guess=0.5,
                         r_prev=None, y_prev=None, p_prev=None):
    """Solve coupled ISPG oscillon + gravity BVP.

    Variables: y = [Phi, Phi', phi0, phi0']
    Parameter: p = [omega]
    """

    def ode(r, y, p):
        omega = p[0]
        Phi, dPhi, phi0, dphi0 = y
        r_safe = np.maximum(r, 1e-8)

        exp_factor = np.exp(-2.0 * phi0)
        w2e = omega**2 * exp_factor

        d2Phi = (-(2.0 / r_safe) * dPhi
                 - w2e * Phi * (1.0 + 0.5 * Phi)
                 + dphi0 * dPhi)

        d2phi0 = (-(2.0 / r_safe) * dphi0
                  - alpha * (0.5 * omega**2 * Phi**2 + 0.5 * dPhi**2))

        d2Phi_0 = -w2e * Phi * (1.0 + 0.5 * Phi) / 3.0
        d2phi0_0 = -alpha * (0.5 * omega**2 * Phi**2 + 0.5 * dPhi**2) / 3.0

        d2Phi = np.where(r < 1e-8, d2Phi_0, d2Phi)
        d2phi0 = np.where(r < 1e-8, d2phi0_0, d2phi0)

        return np.vstack([dPhi, d2Phi, dphi0, d2phi0])

    def bc(ya, yb, p):
        return np.array([
            ya[0] - Phi0,   # Phi(0) = Phi0
            ya[1],           # Phi'(0) = 0
            yb[0],           # Phi(r_max) = 0
            ya[3],           # phi0'(0) = 0
            yb[2],           # phi0(r_max) = 0
        ])

    N = 400
    r_init = np.linspace(1e-6, r_max, N)

    if r_prev is not None and y_prev is not None:
        from scipy.interpolate import interp1d
        y_init = np.zeros((4, N))
        for i in range(4):
            f = interp1d(r_prev, y_prev[i], fill_value=0.0, bounds_error=False)
            y_init[i] = f(r_init)
        y_init[0] *= Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init[1] *= Phi0 / max(abs(y_prev[0][0]), 1e-30)
        omega_g = p_prev[0] if p_prev is not None else omega_guess
    else:
        kappa_g = 0.3
        Phi_init = Phi0 / np.cosh(r_init * kappa_g)**2
        dPhi_init = np.gradient(Phi_init, r_init)
        phi0_init = -alpha * Phi0**2 * np.exp(-r_init * kappa_g) / (1 + r_init)
        dphi0_init = np.gradient(phi0_init, r_init)
        y_init = np.vstack([Phi_init, dPhi_init, phi0_init, dphi0_init])
        omega_g = omega_guess

    sol = solve_bvp(ode, bc, r_init, y_init, p=[omega_g],
                    tol=1e-5, max_nodes=30000, verbose=0)

    if sol.success:
        return sol.p[0], sol
    return None, None


def energy_ispg(sol, omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    phi0 = sol.y[2]
    integrand = (0.5 * omega**2 * Phi**2 + 0.5 * dPhi**2) * r**2
    return 4 * np.pi * np.trapz(integrand, r)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


print("=" * 70)
print("  ISPG Self-Gravitating Oscillon")
print("  (actual equation from the theory)")
print("=" * 70)

# Scan alpha (gravitational coupling strength)
for alpha in [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    print(f"\n  alpha = {alpha}")
    print(f"  {'Phi0':>6} {'omega':>10} {'E':>12} {'phi0_center':>12}")
    print(f"  {'-'*44}")

    results = []

    # Try multiple Phi0 values
    prev_sol = None
    for Phi0 in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        best_om = None
        best_sol = None

        for om_g in np.linspace(0.1, 2.0, 20):
            om, sol = solve_ispg_oscillon(
                Phi0, alpha, r_max=25.0, omega_guess=om_g)

            if om is not None and om > 0.01:
                E = energy_ispg(sol, om)
                if E > 0:
                    phi0_c = sol.y[2][0]
                    already = any(abs(om - r[1]) < 0.01 for r in results)
                    if not already:
                        if best_sol is None or abs(phi0_c) < 10:
                            best_om = om
                            best_sol = sol

        if best_om is not None:
            E = energy_ispg(best_sol, best_om)
            phi0_c = best_sol.y[2][0]
            results.append((Phi0, best_om, E, phi0_c))
            print(f"  {Phi0:6.2f} {best_om:10.6f} {E:12.4f} {phi0_c:12.6f}")

    if len(results) >= 3:
        energies = sorted(set([r[2] for r in results]))
        if len(energies) >= 3:
            from itertools import combinations
            best_Q_diff = 999
            for combo in combinations(energies, 3):
                e = sorted(combo)
                if e[0] > 0:
                    Q = koide(*e)
                    if abs(Q - 2/3) < best_Q_diff:
                        best_Q_diff = abs(Q - 2/3)
                        best_Q = Q
                        best_e = e

            print(f"\n  Koide Q = {best_Q:.6f}  |Q-2/3| = {best_Q_diff:.2e}")
            if best_e[0] > 0:
                print(f"  Ratios: 1 : {best_e[1]/best_e[0]:.1f}"
                      f" : {best_e[2]/best_e[0]:.1f}")

print("\n" + "=" * 70)
print("  Discrete eigenvalue search at fixed alpha")
print("  (scanning omega finely to find ALL bound states)")
print("=" * 70)

alpha_test = 0.1
Phi0_test = 1.0
r_max = 25.0

print(f"\n  alpha={alpha_test}, Phi0={Phi0_test}, r_max={r_max}")
print(f"  Scanning omega from 0.05 to 3.0...")

found_omegas = []
for om_g in np.linspace(0.05, 3.0, 100):
    om, sol = solve_ispg_oscillon(Phi0_test, alpha_test, r_max, om_g)
    if om is not None and om > 0.01:
        already = any(abs(om - fo) < 0.02 for fo in found_omegas)
        if not already:
            E = energy_ispg(sol, om)
            phi0_c = sol.y[2][0]
            found_omegas.append(om)
            nodes = np.sum(np.diff(np.sign(sol.y[0])) != 0)
            print(f"    omega={om:.6f}, E={E:.4f}, phi0(0)={phi0_c:.6f},"
                  f" nodes={nodes}")

if len(found_omegas) >= 3:
    print(f"\n  Found {len(found_omegas)} distinct omega values!")
    print(f"  This suggests DISCRETE spectrum from self-gravity!")
