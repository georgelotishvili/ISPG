"""3D oscillon with Phi^3 nonlinearity — search for harmonics.

Equation: Phi'' + (2/r)Phi' + (Omega^2 - 1)Phi + Phi^3 = 0
This is symmetric under Phi -> -Phi, allowing nodal (excited) solutions.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq

# ---- Part 1: Node scan via shooting ----

def shoot_phi3(Phi0, Omega, r_max=25.0):
    """Integrate Phi^3 oscillon from r=0, return (Phi_end, nodes, sol)."""
    kappa2 = 1.0 - Omega**2
    if kappa2 <= 0:
        return np.inf, -1, None

    def rhs(r, y):
        Phi, dPhi = y
        if r < 1e-10:
            d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**3 / 3.0
        else:
            d2Phi = -(2.0 / r) * dPhi - (Omega**2 - 1) * Phi - Phi**3
        return [dPhi, d2Phi]

    sol = solve_ivp(rhs, [1e-10, r_max], [Phi0, 0.0],
                    rtol=1e-10, atol=1e-12,
                    max_step=0.1, dense_output=True)
    if not sol.success:
        return np.inf, -1, None

    Phi_vals = sol.y[0]
    nodes = np.sum(np.diff(np.sign(Phi_vals)) != 0)
    return sol.y[0, -1], nodes, sol


print("=" * 70)
print("  Phi^3 oscillon: node count scan")
print("=" * 70)

for Phi0 in [0.5, 1.0, 1.5]:
    print(f"\n  Phi0 = {Phi0}")
    print(f"  {'Omega':>8} {'Phi_end':>12} {'nodes':>6}")
    print(f"  {'-'*30}")

    prev_nodes = -1
    transitions = []

    for Omega in np.linspace(0.999, 0.02, 1000):
        Phi_end, nodes, _ = shoot_phi3(Phi0, Omega, r_max=25.0)
        if nodes >= 0 and nodes != prev_nodes and prev_nodes >= 0:
            transitions.append((Omega, prev_nodes, nodes))
        prev_nodes = nodes if nodes >= 0 else prev_nodes

    if transitions:
        print(f"  Transitions found:")
        for Om, n_from, n_to in transitions:
            print(f"    Omega={Om:.5f}: {n_from} -> {n_to} nodes")
    else:
        print(f"  No node transitions found")

    print(f"\n  Detailed node counts:")
    for Om in np.linspace(0.99, 0.05, 20):
        Phi_end, nodes, _ = shoot_phi3(Phi0, Om, r_max=25.0)
        marker = ""
        if nodes >= 0:
            marker = " *" if abs(Phi_end) < 0.05 else ""
        print(f"    Om={Om:.3f}: nodes={nodes:2d}, Phi_end={Phi_end:10.4e}{marker}")

# ---- Part 2: BVP solutions for each branch ----

print("\n" + "=" * 70)
print("  Phi^3 oscillon: BVP solver for ground + excited states")
print("=" * 70)

def find_phi3_bvp(Phi0, n_nodes, r_max=25.0, Omega_guess=0.9,
                   r_prev=None, y_prev=None):
    """BVP solver for Phi^3 oscillon with n_nodes target."""

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
    r_init = np.linspace(1e-6, r_max, N)

    if r_prev is not None and y_prev is not None:
        from scipy.interpolate import interp1d
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        scale = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        Phi_init = f0(r_init) * scale
        dPhi_init = f1(r_init) * scale
    else:
        kappa_g = np.sqrt(max(0.01, 1.0 - Omega_guess**2))
        envelope = Phi0 / np.cosh(r_init * kappa_g)**2
        if n_nodes == 0:
            Phi_init = envelope
        else:
            r_core = min(r_max * 0.6, 6.0 / kappa_g)
            Phi_init = envelope * np.cos(n_nodes * np.pi * r_init / r_core)
        dPhi_init = np.gradient(Phi_init, r_init)

    y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r_init, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)

    if sol.success:
        nodes_found = np.sum(np.diff(np.sign(sol.y[0])) != 0)
        return sol.p[0], sol, nodes_found
    return None, None, -1


def total_energy_phi3(sol, Omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)


for Phi0 in [1.0, 1.5, 2.0]:
    print(f"\n  Phi0 = {Phi0}")
    print(f"  {'n_target':>8} {'Omega':>10} {'E':>12} {'nodes':>6} {'status':>10}")

    found_states = []
    for n_target in range(5):
        best_Om = None
        best_sol = None
        best_nodes = -1

        for Om_g in np.linspace(0.95, 0.05, 40):
            Om, sol, nodes = find_phi3_bvp(Phi0, n_target, r_max=25.0,
                                            Omega_guess=Om_g)
            if Om is not None and 0.01 < Om < 0.999 and nodes == n_target:
                already = any(abs(Om - f[0]) < 0.005 for f in found_states)
                if not already:
                    best_Om = Om
                    best_sol = sol
                    best_nodes = nodes
                    break

        if best_sol is not None:
            E = total_energy_phi3(best_sol, best_Om)
            kappa = np.sqrt(1.0 - best_Om**2)
            found_states.append((best_Om, E, best_nodes))
            print(f"  {n_target:8d} {best_Om:10.6f} {E:12.4f} {best_nodes:6d}      found")
        else:
            print(f"  {n_target:8d} {'':>10} {'':>12} {'':>6}  not found")

    if len(found_states) >= 2:
        print(f"\n  Energy spectrum:")
        E0 = found_states[0][1]
        for Om, E, n in found_states:
            print(f"    n={n}: Omega={Om:.6f}, E={E:.4f}, E/E0={E/E0:.4f}")

    if len(found_states) >= 3:
        from itertools import combinations
        def koide(m1, m2, m3):
            return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

        energies = sorted([f[1] for f in found_states])
        best_Q_diff = 999
        best_Q = None
        for combo in combinations(energies, 3):
            e = sorted(combo)
            Q = koide(*e)
            if abs(Q - 2/3) < best_Q_diff:
                best_Q_diff = abs(Q - 2/3)
                best_Q = Q
                best_e = e

        print(f"\n  Koide Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
        print(f"    Energies: {best_e[0]:.4f}, {best_e[1]:.4f}, {best_e[2]:.4f}")
        print(f"    Ratios: 1 : {best_e[1]/best_e[0]:.2f} : {best_e[2]/best_e[0]:.2f}")
        print(f"    Lepton target: 1 : 206.8 : 3477.2")
