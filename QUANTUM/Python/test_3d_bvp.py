"""3D oscillon via BVP solver with continuation method."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp

def find_3d_oscillon_bvp(Phi0, r_max=50.0, Omega_guess=None,
                          r_prev=None, y_prev=None):
    """Solve 3D oscillon BVP with optional previous solution as seed."""

    if Omega_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Omega_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

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

    if r_prev is not None and y_prev is not None:
        r_init = np.linspace(1e-6, r_max, max(300, len(r_prev)))
        from scipy.interpolate import interp1d
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        Phi_init = f0(r_init) * (Phi0 / max(y_prev[0][0], 1e-30))
        dPhi_init = f1(r_init) * (Phi0 / max(y_prev[0][0], 1e-30))
        y_init = np.vstack([Phi_init, dPhi_init])
    else:
        r_init = np.linspace(1e-6, r_max, 300)
        kappa_g = np.sqrt(max(0.01, 1.0 - Omega_guess**2))
        Phi_init = Phi0 / np.cosh(r_init * kappa_g)**2
        dPhi_init = -2.0 * Phi0 * kappa_g * np.sinh(r_init * kappa_g) / np.cosh(r_init * kappa_g)**3
        y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r_init, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)

    if sol.success:
        return sol.p[0], sol
    return None, None

def total_energy_3d(sol, Omega):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)

print("=" * 65)
print("  3D Oscillon via BVP solver (continuation method)")
print("=" * 65)
print(f"  {'Phi0':>6} {'Omega':>10} {'kappa':>8} {'E_total':>12} {'E/E_ref':>10}"
      f" {'Phi_end':>10}")

results = []
E_ref = None

# Phase 1: find seed solution at Phi0=0.30 (known to work)
print("\n  --- Phase 1: seed solution ---")
seed_Phi0 = 0.30
r_max_seed = 35.0
Om_seed, sol_seed = find_3d_oscillon_bvp(seed_Phi0, r_max=r_max_seed,
                                          Omega_guess=0.96)
if Om_seed is not None:
    E_seed = total_energy_3d(sol_seed, Om_seed)
    E_ref = E_seed
    results.append((seed_Phi0, Om_seed, E_seed, np.sqrt(1-Om_seed**2)))
    print(f"  {seed_Phi0:6.2f} {Om_seed:10.6f} {np.sqrt(1-Om_seed**2):8.4f}"
          f" {E_seed:12.4f} {1.0:10.2f} {sol_seed.y[0,-1]:10.2e}")
    prev_Om = Om_seed
    prev_r = sol_seed.x
    prev_y = sol_seed.y
    prev_rmax = r_max_seed
else:
    print("  SEED FAILED!")
    sys.exit(1)

# Phase 2: continue downward (smaller Phi0 -> barely bound, wider)
print("\n  --- Phase 2: continue to small Phi0 ---")
prev_Om_down = Om_seed
prev_r_down = sol_seed.x
prev_y_down = sol_seed.y

for Phi0 in np.arange(0.25, 0.01, -0.05):
    r_max = r_max_seed

    Om, sol = find_3d_oscillon_bvp(Phi0, r_max=r_max,
                                    Omega_guess=prev_Om_down,
                                    r_prev=prev_r_down, y_prev=prev_y_down)
    if Om is not None and 0.01 < Om < 0.999:
        kappa = np.sqrt(1.0 - Om**2)
        E = total_energy_3d(sol, Om)
        if E > 0:
            results.append((Phi0, Om, E, kappa))
            print(f"  {Phi0:6.2f} {Om:10.6f} {kappa:8.4f} {E:12.4f}"
                  f" {E/E_ref:10.2f} {sol.y[0,-1]:10.2e}")
            prev_Om_down = Om
            prev_r_down = sol.x
            prev_y_down = sol.y
        else:
            print(f"  {Phi0:6.2f} E<=0")
    else:
        print(f"  {Phi0:6.2f} NO CONVERGENCE")

# Phase 3: continue upward (larger Phi0 -> deeply bound, compact)
print("\n  --- Phase 3: continue to large Phi0 ---")
prev_Om_up = Om_seed
prev_r_up = sol_seed.x
prev_y_up = sol_seed.y

for Phi0 in np.arange(0.35, 4.01, 0.01):
    r_max = r_max_seed

    Om, sol = find_3d_oscillon_bvp(Phi0, r_max=r_max,
                                    Omega_guess=prev_Om_up,
                                    r_prev=prev_r_up, y_prev=prev_y_up)
    if Om is not None and 0.01 < Om < 0.999:
        kappa = np.sqrt(1.0 - Om**2)
        E = total_energy_3d(sol, Om)
        if E > 0:
            results.append((Phi0, Om, E, kappa))
            print(f"  {Phi0:6.2f} {Om:10.6f} {kappa:8.4f} {E:12.4f}"
                  f" {E/E_ref:10.2f} {sol.y[0,-1]:10.2e}")
            prev_Om_up = Om
            prev_r_up = sol.x
            prev_y_up = sol.y
        else:
            print(f"  {Phi0:6.2f} E<=0")
    else:
        print(f"  {Phi0:6.2f} NO CONVERGENCE")
        break

# Sort by Phi0
results.sort(key=lambda x: x[0])

print("\n" + "=" * 65)
print("  SUMMARY: E(Phi0) curve")
print("=" * 65)
print(f"  {'Phi0':>6} {'Omega':>10} {'E_total':>12} {'E/E_ref':>10}")
for r in results:
    print(f"  {r[0]:6.2f} {r[1]:10.6f} {r[2]:12.4f} {r[2]/E_ref:10.2f}")

if len(results) >= 3:
    from itertools import combinations
    m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
    def koide(m1, m2, m3):
        return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

    print(f"\n  Koide check (best triple):")
    best_Q_diff = 1.0
    best = None
    for combo in combinations(results, 3):
        energies = sorted([c[2] for c in combo])
        Q = koide(*energies)
        if abs(Q - 2/3) < best_Q_diff:
            best_Q_diff = abs(Q - 2/3)
            best = combo
            best_Q = Q

    if best:
        energies = sorted([c[2] for c in best])
        print(f"    Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
        print(f"    Ratios: 1 : {energies[1]/energies[0]:.1f}"
              f" : {energies[2]/energies[0]:.1f}")
        print(f"    Target: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")
        for c in sorted(best, key=lambda x: x[2]):
            print(f"      Phi0={c[0]:.3f}, Om={c[1]:.6f}, E={c[2]:.4f}")
