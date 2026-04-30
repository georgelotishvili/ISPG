"""Resonance condition: find Phi0 where Omega * R_eff = n*pi.
Then add gravitational time dilation and recheck.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


def bvp_phi3(Phi0, r_max, Omega_guess, r_prev=None, y_prev=None):
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

    N = 400
    r_init = np.linspace(1e-6, r_max, N)

    if r_prev is not None and y_prev is not None:
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        scale = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        Phi_init = f0(r_init) * scale
        dPhi_init = f1(r_init) * scale
    else:
        kappa = np.sqrt(max(0.01, 1.0 - Omega_guess**2))
        Phi_init = Phi0 / np.cosh(r_init * kappa)**2
        dPhi_init = np.gradient(Phi_init, r_init)

    y_init = np.vstack([Phi_init, dPhi_init])
    sol = solve_bvp(ode, bc, r_init, y_init, p=[Omega_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)

    if sol.success and sol.p[0] < 1.0 and sol.p[0] > 0.01:
        return sol.p[0], sol
    return None, None


def compute_properties(sol, Om):
    """Compute energy, effective radius, and resonance ratio."""
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]

    energy_density = (0.25 * Om**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    E_total = 4 * np.pi * np.trapz(energy_density, r)

    cumulative = 4 * np.pi * np.cumsum(
        np.diff(r) * 0.5 * (energy_density[:-1] + energy_density[1:]))
    cumulative = np.insert(cumulative, 0, 0.0)

    R_half = np.interp(0.5 * E_total, cumulative, r)
    R_90 = np.interp(0.9 * E_total, cumulative, r)

    kappa = np.sqrt(1 - Om**2)

    return {
        'E': E_total,
        'R_half': R_half,
        'R_90': R_90,
        'R_kappa': 1.0 / kappa,
        'kappa': kappa,
        'resonance_half': Om * R_half,
        'resonance_90': Om * R_90,
        'resonance_kappa': Om / kappa,
    }


# ============================================================
# Phase 1: Compute ground state oscillon properties
# ============================================================
print("=" * 70)
print("  Phase 1: Phi^3 oscillon properties + resonance ratios")
print("=" * 70)

SEED_PHI0 = 1.0
SEED_OM = 0.97
R_MAX = 25.0

Om0, sol0 = bvp_phi3(SEED_PHI0, R_MAX, SEED_OM)
if Om0 is None:
    print("  SEED FAILED!")
    sys.exit(1)

results = []
props0 = compute_properties(sol0, Om0)
results.append((SEED_PHI0, Om0, sol0, props0))

# Continue upward
prev_Om, prev_r, prev_y = Om0, sol0.x, sol0.y
for Phi0 in np.arange(1.02, 5.01, 0.02):
    Om, sol = bvp_phi3(Phi0, R_MAX, prev_Om,
                        r_prev=prev_r, y_prev=prev_y)
    if Om is not None:
        props = compute_properties(sol, Om)
        results.append((Phi0, Om, sol, props))
        prev_Om, prev_r, prev_y = Om, sol.x, sol.y
    else:
        break

# Continue downward from seed
prev_Om, prev_r, prev_y = Om0, sol0.x, sol0.y
for Phi0 in np.arange(0.98, 0.29, -0.02):
    Om, sol = bvp_phi3(Phi0, R_MAX, prev_Om,
                        r_prev=prev_r, y_prev=prev_y)
    if Om is not None:
        props = compute_properties(sol, Om)
        results.append((Phi0, Om, sol, props))
        prev_Om, prev_r, prev_y = Om, sol.x, sol.y
    else:
        break

results.sort(key=lambda x: x[0])

print(f"\n  {'Phi0':>6} {'Om':>8} {'kappa':>6} {'E':>10} {'R_half':>7}"
      f" {'R_90':>7} {'Om*Rh':>7} {'Om*R90':>7} {'Om/k':>7}")

for Phi0, Om, sol, p in results:
    print(f"  {Phi0:6.2f} {Om:8.4f} {p['kappa']:6.3f} {p['E']:10.4f}"
          f" {p['R_half']:7.3f} {p['R_90']:7.3f}"
          f" {p['resonance_half']:7.3f} {p['resonance_90']:7.3f}"
          f" {p['resonance_kappa']:7.3f}")

# ============================================================
# Phase 2: Find resonance points (where Om*R = n*pi)
# ============================================================
print("\n" + "=" * 70)
print("  Phase 2: Resonance points (Om * R_eff = n * pi)")
print("=" * 70)

for ratio_name, ratio_key in [('Om*R_half', 'resonance_half'),
                                ('Om*R_90', 'resonance_90'),
                                ('Om/kappa', 'resonance_kappa')]:
    print(f"\n  Using {ratio_name}:")
    ratios = [(r[0], r[1], r[3][ratio_key], r[3]['E']) for r in results]

    for n in range(1, 10):
        target = n * np.pi
        for i in range(1, len(ratios)):
            r_prev_val = ratios[i-1][2]
            r_curr_val = ratios[i][2]
            if (r_prev_val - target) * (r_curr_val - target) < 0:
                frac = (target - r_prev_val) / (r_curr_val - r_prev_val)
                Phi0_interp = ratios[i-1][0] + frac * (ratios[i][0] - ratios[i-1][0])
                Om_interp = ratios[i-1][1] + frac * (ratios[i][1] - ratios[i-1][1])
                E_interp = ratios[i-1][3] + frac * (ratios[i][3] - ratios[i-1][3])
                print(f"    n={n}: Phi0={Phi0_interp:.4f}, Om={Om_interp:.4f},"
                      f" E={E_interp:.4f}")

# ============================================================
# Phase 3: Resonance with gravitational time dilation
# ============================================================
print("\n" + "=" * 70)
print("  Phase 3: Resonance with gravitational time dilation")
print("  (proper_frequency * proper_radius = n * pi)")
print("=" * 70)

for alpha_label, alpha in [("weak (0.01)", 0.01),
                            ("moderate (0.1)", 0.1),
                            ("strong (0.3)", 0.3),
                            ("very strong (0.5)", 0.5),
                            ("extreme (0.8)", 0.8)]:
    print(f"\n  Coupling alpha = {alpha_label}:")

    grav_ratios = []
    for Phi0, Om, sol, p in results:
        E = p['E']
        R = p['R_90']
        if R < 0.01:
            continue

        grav_potential = alpha * E / R
        g_tt = max(0.01, 1.0 - 2 * grav_potential)
        Om_proper = Om / np.sqrt(g_tt)
        R_proper = R * np.sqrt(1 + grav_potential * 0.5)

        grav_ratio = Om_proper * R_proper
        grav_ratios.append((Phi0, Om, E, grav_ratio, Om_proper, R_proper))

    resonance_pts = []
    for n in range(1, 15):
        target = n * np.pi
        for i in range(1, len(grav_ratios)):
            r_prev_val = grav_ratios[i-1][3]
            r_curr_val = grav_ratios[i][3]
            if (r_prev_val - target) * (r_curr_val - target) < 0:
                frac = (target - r_prev_val) / (r_curr_val - r_prev_val)
                Phi0_interp = grav_ratios[i-1][0] + frac * (grav_ratios[i][0] - grav_ratios[i-1][0])
                E_interp = grav_ratios[i-1][2] + frac * (grav_ratios[i][2] - grav_ratios[i-1][2])
                resonance_pts.append((n, Phi0_interp, E_interp))
                print(f"    n={n}: Phi0={Phi0_interp:.4f}, E={E_interp:.4f}")

    if len(resonance_pts) >= 3:
        from itertools import combinations
        def koide(m1, m2, m3):
            return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

        energies = [p[2] for p in resonance_pts]
        best_Q_diff = 999
        best_Q = None
        best_triple = None
        for combo in combinations(range(len(energies)), 3):
            e = sorted([energies[c] for c in combo])
            if e[0] > 0:
                Q = koide(*e)
                if abs(Q - 2/3) < best_Q_diff:
                    best_Q_diff = abs(Q - 2/3)
                    best_Q = Q
                    best_triple = [resonance_pts[c] for c in combo]

        if best_triple:
            es = sorted([t[2] for t in best_triple])
            print(f"\n    KOIDE Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
            print(f"    E = {es[0]:.4f}, {es[1]:.4f}, {es[2]:.4f}")
            print(f"    Ratios: 1 : {es[1]/es[0]:.2f} : {es[2]/es[0]:.2f}")
            print(f"    Target: 1 : 206.8 : 3477.2")
            for t in sorted(best_triple, key=lambda x: x[2]):
                print(f"      n={t[0]}, Phi0={t[1]:.4f}, E={t[2]:.4f}")
