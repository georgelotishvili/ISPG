"""Phi^3 oscillon: BVP continuation from small Phi0 upward.
Ground state first, then attempt excited states via continuation."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


def bvp_phi3(Phi0, r_max, Omega_guess, r_prev=None, y_prev=None):
    """BVP for Phi'' + (2/r)Phi' + (Om^2-1)Phi + Phi^3 = 0."""

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y[0], y[1]
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

    if sol.success:
        nodes = np.sum(np.diff(np.sign(sol.y[0])) != 0)
        return sol.p[0], sol, nodes
    return None, None, -1


def energy(sol, Om):
    r, Phi, dPhi = sol.x, sol.y[0], sol.y[1]
    return 4 * np.pi * np.trapz((0.25 * Om**2 * Phi**2 + 0.25 * dPhi**2) * r**2, r)


R_MAX = 25.0
SEED_PHI0 = 1.0
SEED_OMEGA = 0.97

# ---- Ground state: continuation from Phi0=1.0 (both directions) ----
print("=" * 60)
print("  Phi^3 ground state (n=0) via continuation")
print("=" * 60)
print(f"  {'Phi0':>6} {'Omega':>10} {'kappa':>8} {'E':>12}")

Om0, sol0, n0 = bvp_phi3(SEED_PHI0, R_MAX, SEED_OMEGA)
if Om0 is None or Om0 >= 1.0:
    print("  SEED FAILED!")
    sys.exit(1)

E0_val = energy(sol0, Om0)
print(f"  {SEED_PHI0:6.2f} {Om0:10.6f} {np.sqrt(1-Om0**2):8.4f} {E0_val:12.4f}")

ground_states = [(SEED_PHI0, Om0, E0_val)]

# Continue downward
prev_Om, prev_r, prev_y = Om0, sol0.x, sol0.y
for Phi0 in np.arange(SEED_PHI0 - 0.02, 0.29, -0.02):
    Om, sol, nodes = bvp_phi3(Phi0, R_MAX, prev_Om,
                               r_prev=prev_r, y_prev=prev_y)
    if Om is not None and 0.01 < Om < 0.999 and nodes == 0:
        E = energy(sol, Om)
        ground_states.append((Phi0, Om, E))
        print(f"  {Phi0:6.2f} {Om:10.6f} {np.sqrt(1-Om**2):8.4f} {E:12.4f}")
        prev_Om, prev_r, prev_y = Om, sol.x, sol.y
    else:
        print(f"  {Phi0:6.2f} down-stop")
        break

# Continue upward
prev_Om, prev_r, prev_y = Om0, sol0.x, sol0.y
for Phi0 in np.arange(SEED_PHI0 + 0.02, 15.01, 0.02):
    Om, sol, nodes = bvp_phi3(Phi0, R_MAX, prev_Om,
                               r_prev=prev_r, y_prev=prev_y)
    if Om is not None and 0.01 < Om < 0.999 and nodes == 0:
        E = energy(sol, Om)
        kappa = np.sqrt(1 - Om**2)
        ground_states.append((Phi0, Om, E))
        if Phi0 < 3.05 or Phi0 % 0.5 < 0.025:
            print(f"  {Phi0:6.2f} {Om:10.6f} {kappa:8.4f} {E:12.4f}")
        prev_Om, prev_r, prev_y = Om, sol.x, sol.y
    else:
        print(f"  {Phi0:6.2f} NO CONVERGENCE (stopped)")
        break

# ---- Now try excited state n=1 ----
print("\n" + "=" * 60)
print("  Phi^3 first excited state (n=1)")
print("=" * 60)

excited_1 = []
# Use ground state profile at large Phi0, add a node
for Phi0_try in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    gs = [g for g in ground_states if abs(g[0] - Phi0_try) < 0.15]
    if not gs:
        continue
    gs_Phi0, gs_Om, gs_E = gs[0]

    found = False
    for Om_shift in np.linspace(0.01, 0.5, 20):
        Om_try = min(gs_Om + Om_shift, 0.999)

        r_test = np.linspace(1e-6, R_MAX, 400)
        kappa_t = np.sqrt(max(0.01, 1.0 - Om_try**2))
        env = Phi0_try / np.cosh(r_test * kappa_t)**2
        r_node = 3.0 / kappa_t
        Phi_init = env * np.cos(np.pi * r_test / r_node)

        def ode(r, y, p):
            Om = p[0]
            Phi, dPhi = y[0], y[1]
            r_safe = np.maximum(r, 1e-8)
            d2 = -(2.0 / r_safe) * dPhi - (Om**2 - 1) * Phi - Phi**3
            d2_0 = -(Om**2 - 1) * Phi / 3.0 - Phi**3 / 3.0
            d2 = np.where(r < 1e-8, d2_0, d2)
            return np.vstack([dPhi, d2])
        def bc(ya, yb, p):
            return np.array([ya[0] - Phi0_try, ya[1], yb[0]])

        dPhi_init = np.gradient(Phi_init, r_test)
        y_init = np.vstack([Phi_init, dPhi_init])

        sol = solve_bvp(ode, bc, r_test, y_init, p=[Om_try],
                        tol=1e-6, max_nodes=20000, verbose=0)
        if sol.success:
            nodes = np.sum(np.diff(np.sign(sol.y[0])) != 0)
            if nodes == 1:
                Om_ex = sol.p[0]
                E_ex = energy(sol, Om_ex)
                kappa = np.sqrt(1 - Om_ex**2)
                excited_1.append((Phi0_try, Om_ex, E_ex))
                print(f"  Phi0={Phi0_try:.1f}: Om={Om_ex:.6f}, kappa={kappa:.4f},"
                      f" E={E_ex:.4f}, n=1")
                found = True
                break

    if not found:
        print(f"  Phi0={Phi0_try:.1f}: n=1 not found")

# ---- Summary ----
print("\n" + "=" * 60)
print("  SPECTRUM SUMMARY")
print("=" * 60)

if ground_states:
    print(f"\n  Ground state (n=0): {len(ground_states)} solutions")
    print(f"    Phi0 range: [{ground_states[0][0]:.1f}, {ground_states[-1][0]:.1f}]")
    print(f"    E range: [{ground_states[0][2]:.2f}, {ground_states[-1][2]:.2f}]")
    print(f"    Omega range: [{ground_states[-1][1]:.4f}, {ground_states[0][1]:.4f}]")

if excited_1:
    print(f"\n  First excited (n=1): {len(excited_1)} solutions")
    for Phi0, Om, E in excited_1:
        gs_match = [g for g in ground_states if abs(g[0] - Phi0) < 0.15]
        if gs_match:
            E0 = gs_match[0][2]
            print(f"    Phi0={Phi0:.1f}: E_1/E_0 = {E/E0:.4f}"
                  f" (E_0={E0:.2f}, E_1={E:.2f})")

    if len(excited_1) >= 1 and len(ground_states) >= 1:
        print(f"\n  Koide check (using n=0 and n=1 at same Phi0):")
        for Phi0, Om1, E1 in excited_1:
            gs = [g for g in ground_states if abs(g[0] - Phi0) < 0.15]
            if gs:
                E0 = gs[0][2]
                if E0 > 0 and E1 > 0 and E0 != E1:
                    e_pair = sorted([E0, E1])
                    ratio = e_pair[1] / e_pair[0]
                    print(f"    Phi0={Phi0:.1f}: E0={E0:.2f}, E1={E1:.2f},"
                          f" ratio={ratio:.2f}")
