"""Phi^3 oscillon full spectrum: independent BVP calls at each Phi0.
Relationship: Phi0 ~ 4.33 * kappa (from known solutions).
Also search for excited states at each Phi0.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp


def bvp_solve(Phi0, n_nodes, r_max, Omega_guess):
    """Single BVP solve for Phi^3 oscillon."""

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        d2 = -(2.0 / r_safe) * dPhi - (Om**2 - 1) * Phi - Phi**3
        d2_0 = -(Om**2 - 1) * Phi / 3.0 - Phi**3 / 3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    r = np.linspace(1e-6, r_max, 400)
    kappa_g = np.sqrt(max(0.01, 1.0 - Omega_guess**2))
    env = Phi0 / np.cosh(r * kappa_g)**2

    if n_nodes == 0:
        Phi_init = env
    else:
        r_core = min(r_max * 0.5, 4.0 / kappa_g)
        Phi_init = env * np.cos(n_nodes * np.pi * r / r_core)

    dPhi_init = np.gradient(Phi_init, r)
    y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)

    if sol.success:
        nodes = np.sum(np.diff(np.sign(sol.y[0])) != 0)
        return sol.p[0], sol, nodes
    return None, None, -1


def energy(sol, Om):
    r, Phi, dPhi = sol.x, sol.y[0], sol.y[1]
    return 4 * np.pi * np.trapz((0.25 * Om**2 * Phi**2 + 0.25 * dPhi**2) * r**2, r)


print("=" * 70)
print("  Phi^3 Oscillon Spectrum (independent BVP calls)")
print("=" * 70)

# Known: Phi0 ~ 4.33 * kappa for ground state
RATIO = 4.33

all_states = {}

for Phi0 in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    print(f"\n  Phi0 = {Phi0:.1f}", flush=True)
    all_states[Phi0] = []

    kappa_est = Phi0 / RATIO
    Om_est = np.sqrt(max(0.01, 1.0 - kappa_est**2))
    r_max = max(15.0, min(40.0, 12.0 / max(kappa_est, 0.05)))

    for n_target in range(4):
        found = False

        if n_target == 0:
            omega_tries = [Om_est]
            for delta in np.linspace(-0.1, 0.1, 10):
                omega_tries.append(max(0.05, min(0.999, Om_est + delta)))
        else:
            omega_tries = []
            for om in np.linspace(Om_est + 0.01, 0.999, 15):
                omega_tries.append(om)

        for Om_try in omega_tries:
            Om, sol, nodes = bvp_solve(Phi0, n_target, r_max, Om_try)
            if Om is not None and 0.01 < Om < 0.999 and nodes == n_target:
                E = energy(sol, Om)
                kappa = np.sqrt(1 - Om**2)
                already = any(abs(Om - s[1]) < 0.002 for s in all_states[Phi0])
                if not already and E > 0:
                    all_states[Phi0].append((n_target, Om, E, kappa))
                    print(f"    n={n_target}: Om={Om:.6f}, kappa={kappa:.4f},"
                          f" E={E:.4f}, r_max={r_max:.0f}")
                    found = True
                    break

        if not found and n_target < 2:
            print(f"    n={n_target}: not found")

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

for Phi0, states in all_states.items():
    if not states:
        continue
    print(f"\n  Phi0 = {Phi0:.1f}:")
    if states:
        E0 = states[0][2]
        for n, Om, E, kappa in states:
            print(f"    n={n}: E={E:.4f} (E/E0={E/E0:.3f}), Om={Om:.6f}")

        if len(states) >= 3:
            from itertools import combinations
            def koide(m1, m2, m3):
                return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2
            es = sorted([s[2] for s in states])
            for combo in combinations(es, 3):
                e = sorted(combo)
                Q = koide(*e)
                if abs(Q - 2/3) < 0.05:
                    print(f"    ** KOIDE Q={Q:.6f}, E={e[0]:.2f},{e[1]:.2f},{e[2]:.2f}")
