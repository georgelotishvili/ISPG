"""
ISPG Oscillon: Total Energy Approach to Koide

Key insight: particle mass = total energy of the oscillon, NOT just hbar*omega.
The total energy depends on BOTH amplitude AND frequency.

For the 1D sech^2 oscillon:
  E(Omega) ~ (1 - Omega^2)^{3/2} * (4*Omega^2 + 1)

For Omega -> 1 (barely bound): E -> 0  (light particle)
For Omega ~ 0.5 (deeply bound): E ~ max (heavy particle)

This gives mass ratios MUCH larger than the frequency ratios.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq
from itertools import product


# ============================================================
#  1D ANALYTICAL MODEL
# ============================================================

def energy_1d(Omega):
    """Total energy of the 1D sech^2 oscillon (arbitrary units)."""
    if Omega >= 1.0 or Omega <= 0.0:
        return 0.0
    beta2 = 1.0 - Omega**2
    beta3 = beta2**1.5
    return beta3 * (4.0 * Omega**2 + 1.0)


def koide_ratio(m1, m2, m3):
    """Q = (m1+m2+m3) / (sqrt(m1)+sqrt(m2)+sqrt(m3))^2"""
    s = m1 + m2 + m3
    sq = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    return s / sq**2


print("=" * 60)
print("  1D Oscillon: Energy vs Frequency")
print("=" * 60)
print(f"  {'Omega':>8}  {'E(Omega)':>12}  {'E_ratio_to_0.99':>18}")
E_ref = energy_1d(0.99)
for Om in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
    E = energy_1d(Om)
    ratio = E / E_ref if E_ref > 0 else 0
    print(f"  {Om:8.3f}  {E:12.6f}  {ratio:18.1f}")


# ============================================================
#  SCAN: Find (Omega_1, Omega_2, Omega_3) satisfying Koide
# ============================================================

print("\n" + "=" * 60)
print("  Scanning for Koide-satisfying triples (1D model)")
print("=" * 60)

# Lepton mass ratios for reference
m_e = 0.51099895   # MeV
m_mu = 105.6583755
m_tau = 1776.86
Q_actual = koide_ratio(m_e, m_mu, m_tau)
print(f"  Actual lepton Q = {Q_actual:.8f}")
print(f"  Actual ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")

# Strategy: Omega_tau (heavy) is small, Omega_e (light) is close to 1
# Scan over (Omega_tau, Omega_mu, Omega_e) and check Koide

best_Q_diff = 1.0
best_triple = None
best_masses = None

Omega_grid = np.concatenate([
    np.linspace(0.05, 0.70, 100),
    np.linspace(0.70, 0.95, 100),
    np.linspace(0.95, 0.9999, 200),
])

energies = np.array([energy_1d(Om) for Om in Omega_grid])
valid = energies > 1e-15

Omega_valid = Omega_grid[valid]
E_valid = energies[valid]

# For efficiency, precompute and scan
N = len(Omega_valid)
print(f"  Scanning {N} Omega values...")

# We need m_tau >> m_mu >> m_e, so Omega_tau < Omega_mu < Omega_e
# and E(Omega_tau) >> E(Omega_mu) >> E(Omega_e)
# Try all combinations where E ratios are roughly right

best_results = []

for i_tau in range(N):
    E_tau = E_valid[i_tau]
    if E_tau < 0.1:
        continue
    for i_mu in range(i_tau + 1, N):
        E_mu = E_valid[i_mu]
        ratio_tau_mu = E_tau / E_mu
        if ratio_tau_mu < 5 or ratio_tau_mu > 50:
            continue
        for i_e in range(i_mu + 1, N):
            E_e = E_valid[i_e]
            if E_e < 1e-10:
                continue
            ratio_mu_e = E_mu / E_e
            if ratio_mu_e < 50 or ratio_mu_e > 500:
                continue
            Q = koide_ratio(E_e, E_mu, E_tau)
            Q_diff = abs(Q - 2.0/3.0)
            if Q_diff < 0.01:
                best_results.append((
                    Q_diff, Q,
                    Omega_valid[i_tau], Omega_valid[i_mu], Omega_valid[i_e],
                    E_tau, E_mu, E_e
                ))

best_results.sort(key=lambda x: x[0])

print(f"\n  Found {len(best_results)} triples with |Q - 2/3| < 0.01")
if best_results:
    print(f"\n  Top 10 matches:")
    print(f"  {'Q':>10} {'Om_tau':>8} {'Om_mu':>8} {'Om_e':>8}"
          f" {'E_tau/E_e':>10} {'E_mu/E_e':>10}")
    for res in best_results[:10]:
        Q_diff, Q, Om_t, Om_m, Om_e, E_t, E_m, E_e_val = res
        print(f"  {Q:10.7f} {Om_t:8.4f} {Om_m:8.4f} {Om_e:8.4f}"
              f" {E_t/E_e_val:10.1f} {E_m/E_e_val:10.1f}")

    # Best match details
    res = best_results[0]
    Q_diff, Q, Om_t, Om_m, Om_e, E_t, E_m, E_e_val = res
    print(f"\n  BEST MATCH:")
    print(f"    Q = {Q:.8f}  (target 0.666667, diff = {Q_diff:.2e})")
    print(f"    Omega_tau  = {Om_t:.6f},  E_tau  = {E_t:.8f}")
    print(f"    Omega_mu   = {Om_m:.6f},  E_mu   = {E_m:.8f}")
    print(f"    Omega_e    = {Om_e:.6f},  E_e    = {E_e_val:.8f}")
    print(f"    Mass ratios: 1 : {E_m/E_e_val:.1f} : {E_t/E_e_val:.1f}")
    print(f"    Target:      1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")


# ============================================================
#  REFINED SEARCH with continuous optimization
# ============================================================

print("\n" + "=" * 60)
print("  Refined optimization: fix mass ratios, find best Koide")
print("=" * 60)

def find_Omega_for_energy(E_target, Omega_range=(0.01, 0.999)):
    """Find Omega such that energy_1d(Omega) = E_target."""
    # E(Omega) has a maximum, so there may be two solutions
    # Left branch: Omega < Omega_max (deeply bound)
    # Right branch: Omega > Omega_max (barely bound)
    Om_max = minimize_scalar(lambda x: -energy_1d(x),
                              bounds=(0.01, 0.999), method='bounded').x
    E_max = energy_1d(Om_max)
    if E_target > E_max:
        return None, None

    solutions = []
    # Left branch
    try:
        Om_left = brentq(lambda x: energy_1d(x) - E_target,
                         0.01, Om_max, xtol=1e-14)
        solutions.append(Om_left)
    except:
        pass
    # Right branch
    try:
        Om_right = brentq(lambda x: energy_1d(x) - E_target,
                          Om_max, 0.9999, xtol=1e-14)
        solutions.append(Om_right)
    except:
        pass
    return solutions


# Find Omega that gives maximum energy
Om_max = minimize_scalar(lambda x: -energy_1d(x),
                          bounds=(0.01, 0.999), method='bounded').x
E_max = energy_1d(Om_max)
print(f"  Maximum energy at Omega = {Om_max:.6f}, E_max = {E_max:.8f}")

# Now: fix E_tau = some value, compute E_mu = E_tau/16.82, E_e = E_tau/3477
# Check if all three have valid Omega, and compute Koide

print(f"\n  Scanning E_tau and checking Koide:")
print(f"  {'E_tau':>10} {'Q':>10} {'Om_tau':>10} {'Om_mu':>10} {'Om_e':>10}"
      f" {'branch_t':>8} {'branch_m':>8} {'branch_e':>8}")

for E_tau_frac in np.linspace(0.1, 0.99, 50):
    E_tau = E_max * E_tau_frac
    E_mu = E_tau * m_mu / m_tau   # = E_tau / 16.82
    E_e = E_tau * m_e / m_tau     # = E_tau / 3477

    sols_tau = find_Omega_for_energy(E_tau)
    sols_mu = find_Omega_for_energy(E_mu)
    sols_e = find_Omega_for_energy(E_e)

    if not sols_tau or not sols_mu or not sols_e:
        continue

    for Om_t in (sols_tau if sols_tau else []):
        for Om_m in (sols_mu if sols_mu else []):
            for Om_e in (sols_e if sols_e else []):
                E_t = energy_1d(Om_t)
                E_m = energy_1d(Om_m)
                E_ev = energy_1d(Om_e)
                Q = koide_ratio(E_ev, E_m, E_t)
                b_t = "L" if Om_t < Om_max else "R"
                b_m = "L" if Om_m < Om_max else "R"
                b_e = "L" if Om_e < Om_max else "R"
                if abs(Q - 2.0/3.0) < 0.05:
                    print(f"  {E_tau:10.6f} {Q:10.7f} {Om_t:10.6f}"
                          f" {Om_m:10.6f} {Om_e:10.6f}"
                          f" {b_t:>8} {b_m:>8} {b_e:>8}")


# ============================================================
#  3D OSCILLON TOTAL ENERGY
# ============================================================

print("\n" + "=" * 60)
print("  3D Oscillon: Total Energy")
print("=" * 60)

def oscillon_3d_rhs(r, y, Omega):
    Phi, dPhi = y
    if r < 1e-10:
        d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**2 / 3.0
    else:
        d2Phi = -(2.0 / r) * dPhi - (Omega**2 - 1) * Phi - Phi**2
    return [dPhi, d2Phi]

def integrate_oscillon_3d(Phi0, Omega, r_max=80.0):
    sol = solve_ivp(
        lambda r, y: oscillon_3d_rhs(r, y, Omega),
        [1e-6, r_max],
        [Phi0, 0.0],
        method='RK45',
        max_step=max(0.02, r_max / 5000),
        rtol=1e-11,
        atol=1e-13,
        dense_output=True
    )
    return sol

def find_3d_oscillon(Phi0, r_max=80.0):
    """Shooting for Phi(r_max)=0, scanning all roots, ground-state selection."""

    def tail_value(Omega):
        sol = integrate_oscillon_3d(Phi0, Omega, r_max)
        return sol.y[0, -1]

    grid = np.linspace(0.03, 0.997, 200)
    fvals = np.empty(len(grid))
    for idx, Om in enumerate(grid):
        try:
            fvals[idx] = tail_value(Om)
        except Exception:
            fvals[idx] = np.nan

    candidates = []
    for i in range(1, len(grid)):
        if (np.isfinite(fvals[i-1]) and np.isfinite(fvals[i])
                and fvals[i-1] * fvals[i] < 0):
            try:
                Om_star = brentq(tail_value, grid[i-1], grid[i], xtol=1e-12)
                sol = integrate_oscillon_3d(Phi0, Om_star, r_max)
                r_half = r_max * 0.7
                core_mask = sol.t < r_half
                has_node = np.any(sol.y[0, core_mask] < -abs(Phi0) * 1e-4)
                candidates.append((1 if has_node else 0, Om_star, sol))
            except (ValueError, RuntimeError):
                continue

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]

def total_energy_3d(sol, Omega):
    """Compute total time-averaged energy of 3D oscillon."""
    r = sol.t
    Phi = sol.y[0]
    dPhi = sol.y[1]

    # Time-averaged energy density:
    # <T00> = 1/4 * omega^2 * Phi^2 + 1/4 * (dPhi/dr)^2
    # (factor 1/2 from time-averaging, 1/2 from T00 definition)
    # Total: E = 4*pi * integral of <T00> * r^2 dr

    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    E = 4.0 * np.pi * np.trapz(integrand, r)
    return E

print(f"\n  {'Phi0':>6} {'Omega':>10} {'E_total':>12} {'E/E_ref':>10}")

results_3d = []
E_ref_3d = None

for Phi0 in np.concatenate([
    np.linspace(0.05, 0.5, 10),
    np.linspace(0.5, 2.0, 15),
    np.linspace(2.0, 4.0, 10),
]):
    Omega_3d, sol_3d = find_3d_oscillon(Phi0, r_max=80.0)
    if Omega_3d is not None and Omega_3d > 0.01:
        E_3d = total_energy_3d(sol_3d, Omega_3d)
        if E_3d > 0:
            if E_ref_3d is None:
                E_ref_3d = E_3d
            results_3d.append((Phi0, Omega_3d, E_3d))
            print(f"  {Phi0:6.2f} {Omega_3d:10.6f} {E_3d:12.6f}"
                  f" {E_3d/E_ref_3d:10.1f}")

# Check Koide for 3D oscillons
if len(results_3d) >= 3:
    from itertools import combinations as comb3
    print(f"\n  Checking Koide for all 3D triples...")
    best_Q_3d = 1.0
    best_triple_3d = None
    for combo in comb3(results_3d, 3):
        energies = [c[2] for c in combo]
        energies.sort()
        Q = koide_ratio(*energies)
        if abs(Q - 2.0/3.0) < best_Q_3d:
            best_Q_3d = abs(Q - 2.0/3.0)
            best_triple_3d = combo
            best_Q_val = Q

    if best_triple_3d:
        print(f"\n  BEST 3D Koide match: Q = {best_Q_val:.8f}")
        energies = sorted([c[2] for c in best_triple_3d])
        print(f"    Energies: {energies}")
        print(f"    Ratios: 1 : {energies[1]/energies[0]:.1f}"
              f" : {energies[2]/energies[0]:.1f}")
        print(f"    Target:  1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")
        for c in sorted(best_triple_3d, key=lambda x: x[2]):
            print(f"      Phi0={c[0]:.3f}, Omega={c[1]:.6f}, E={c[2]:.6f}")
