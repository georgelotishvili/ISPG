"""E_3D vs E_1D Koide: full 3D oscillon energy for cavity eigenvalues.

Goal: replace E_1D(Om) with E_3D(Om) to fix the 23% mass ratio discrepancy.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from itertools import combinations


ALPHA = 0.5


def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)


def solve_osc(Phi0, r_max=60.0, Om_guess=None,
              r_prev=None, y_prev=None, p_prev=None):
    if Om_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Om_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = nl_func(Phi)
        d2 = -(2.0/r_safe)*dPhi - (Om**2 - 1)*Phi - NL
        d2_0 = -(Om**2 - 1)*Phi/3.0 - NL/3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None:
        N_pts = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N_pts)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r)*sc, f1(r)*sc])
        Om_guess = p_prev[0] if p_prev is not None else Om_guess
    else:
        r = np.linspace(1e-6, r_max, 500)
        kg = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kg)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-5, max_nodes=30000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None


def energy_3d(sol, Om):
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    return 4.0 * np.pi * np.trapz((0.5*Om**2*Phi**2 + 0.5*dPhi**2) * r**2, r)


def cavity_eigs(r_bg, Phi_bg, l_val, N=2000):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, _ = eigsh(H, k=min(20, N-2), which='SM')
    bound = evals[evals < 1.0]
    return np.sqrt(np.maximum(np.sort(bound), 0))


def E_1d(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*Om**2 + 1)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86

print("=" * 70)
print("  E_3D vs E_1D: Full 3D oscillon energy")
print("=" * 70)

print("\n--- Building E_3D(Omega) curve (small steps) ---\n")

e3d_data = []
prev_sol = None

for Phi0 in np.arange(0.05, 5.01, 0.01):
    if prev_sol:
        Om, sol = solve_osc(Phi0, r_prev=prev_sol.x, y_prev=prev_sol.y,
                             p_prev=prev_sol.p)
    else:
        Om, sol = solve_osc(Phi0)

    if Om is None:
        if prev_sol is None:
            continue
        else:
            print(f"  Continuation broke at Phi0={Phi0:.2f}")
            break
    prev_sol = sol
    E = energy_3d(sol, Om)
    e3d_data.append((Phi0, Om, E))

    if abs(Phi0 - round(Phi0*5)/5) < 0.005:
        print(f"  Phi0={Phi0:.2f}  Om={Om:.6f}  E_3D={E:.2f}  E_1D={E_1d(Om):.6f}")

if not e3d_data:
    print("  FAILED to build curve!")
    sys.exit(1)

Om_arr = np.array([x[1] for x in e3d_data])
E3d_arr = np.array([x[2] for x in e3d_data])
Phi0_arr = np.array([x[0] for x in e3d_data])

print(f"\n  Curve: {len(e3d_data)} points")
print(f"  Phi0: {Phi0_arr[0]:.2f} ... {Phi0_arr[-1]:.2f}")
print(f"  Omega: {Om_arr[-1]:.4f} ... {Om_arr[0]:.4f}")

E3d_of_Om = interp1d(Om_arr[::-1], E3d_arr[::-1], fill_value='extrapolate')

print(f"\n--- Cavity spectrum at various Phi0 ---\n")

best_1d = (999, None)
best_3d = (999, None)

for Phi0_test in np.arange(0.5, min(Phi0_arr[-1], 4.0), 0.1):
    idx = np.argmin(np.abs(Phi0_arr - Phi0_test))
    if abs(Phi0_arr[idx] - Phi0_test) > 0.02:
        continue

    Om_bg = Om_arr[idx]
    sol_test = None
    for P, O, E in e3d_data:
        if abs(P - Phi0_arr[idx]) < 0.005:
            Om_bg_solve, sol_test = solve_osc(P, r_prev=prev_sol.x,
                                               y_prev=prev_sol.y, p_prev=prev_sol.p)
            if Om_bg_solve:
                Om_bg = Om_bg_solve
            break

    if sol_test is None:
        Om_bg, sol_test = solve_osc(Phi0_test)
    if sol_test is None:
        continue

    all_modes = {}
    for l_val in range(4):
        eigs = cavity_eigs(sol_test.x, sol_test.y[0], l_val, N=1500)
        for n, om in enumerate(eigs):
            if om > 0.001:
                all_modes[(n, l_val)] = om

    if len(all_modes) < 3:
        continue

    mlist = sorted(all_modes.items())
    for combo in combinations(range(len(mlist)), 3):
        keys = [mlist[i][0] for i in combo]
        oms = sorted([mlist[i][1] for i in combo])

        E1d_v = sorted([E_1d(o) for o in oms], reverse=True)
        if E1d_v[-1] > 1e-15:
            Q1 = koide(*E1d_v)
            if abs(Q1-2/3) < best_1d[0]:
                best_1d = (abs(Q1-2/3), (Phi0_test, keys, oms, E1d_v, Q1))

        om_min, om_max = Om_arr.min(), Om_arr.max()
        ok = all(om_min <= o <= om_max for o in oms)
        if ok:
            E3d_v = sorted([float(E3d_of_Om(o)) for o in oms], reverse=True)
            if all(e > 0 for e in E3d_v) and E3d_v[-1] > 1e-10:
                Q3 = koide(*E3d_v)
                if abs(Q3-2/3) < best_3d[0]:
                    best_3d = (abs(Q3-2/3), (Phi0_test, keys, oms, E3d_v, Q3))

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)

print(f"\n  Target: Q={koide(m_e, m_mu, m_tau):.8f}"
      f"  Ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")

print(f"\n  BEST with E_1D:")
if best_1d[1]:
    P, keys, oms, Ev, Q = best_1d[1]
    r0 = min(Ev)
    print(f"    Phi0 = {P:.2f}")
    print(f"    Modes: {keys}")
    print(f"    Omegas: {[f'{o:.6f}' for o in oms]}")
    print(f"    Q = {Q:.10f}  |Q-2/3| = {abs(Q-2/3):.2e}")
    print(f"    Ratios: 1 : {Ev[1]/r0:.1f} : {Ev[0]/r0:.1f}")

print(f"\n  BEST with E_3D:")
if best_3d[1]:
    P, keys, oms, Ev, Q = best_3d[1]
    r0 = min(Ev)
    print(f"    Phi0 = {P:.2f}")
    print(f"    Modes: {keys}")
    print(f"    Omegas: {[f'{o:.6f}' for o in oms]}")
    print(f"    E_3D: {[f'{e:.2f}' for e in Ev]}")
    print(f"    Q = {Q:.10f}  |Q-2/3| = {abs(Q-2/3):.2e}")
    print(f"    Ratios: 1 : {Ev[1]/r0:.1f} : {Ev[0]/r0:.1f}")
else:
    print("    No valid E_3D triple found (omega out of range)")
    print("    Need to extend continuation to lower Omega values")
