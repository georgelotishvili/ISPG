"""
Self-consistency test (#7): Find Φ₀ where Ω_bg = ω_tau.

The background oscillon has frequency Ω_bg.
The ground state cavity mode (tau) has frequency ω_tau.
Currently Ω_bg = 0.866, ω_tau = 0.664 — they don't match.

If a self-consistent point exists, it would give the "true"
cavity spectrum and potentially shift θ₀ to 2/9.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d

ALPHA = 0.5
PI = np.pi

def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))

def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)

def solve_osc(Phi0, r_max=80.0, r_prev=None, y_prev=None, p_prev=None):
    Om_guess = np.sqrt(max(0.01, 1.0 - min(Phi0/4.2, 0.95)))
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
        N = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r)*sc, f1(r)*sc])
        Om_guess = p_prev[0]
    else:
        r = np.linspace(1e-6, r_max, 500)
        kg = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kg)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])
    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-6, max_nodes=50000, verbose=0)
    if sol.success and 0.001 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None

def cavity_ground(r_bg, Phi_bg, l_val=0, N=3000):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, _ = eigsh(H, k=min(6, N-2), which='SM')
    bound = evals < 1.0
    if np.any(bound):
        return np.sqrt(np.sort(evals[bound]))
    return np.array([])

def E_1D(omega):
    k2 = 1.0 - omega**2
    if k2 <= 0:
        return 0.0
    return k2**1.5 * (4*omega**2 + 1)

def koide_Q(m1, m2, m3):
    s = m1 + m2 + m3
    sr = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    return s / sr**2

print("=" * 80)
print("  SELF-CONSISTENCY TEST: Ω_bg vs ω_tau across full Φ₀ range")
print("=" * 80)

# Phase 1: Wide scan
print(f"\n  {'Φ₀':>6} {'Ω_bg':>8} {'ω_τ':>8} {'ω_μ':>8} {'gap':>8} {'ratio':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

prev = None
results = []
phi0_values = list(np.arange(0.1, 0.5, 0.1)) + \
              list(np.arange(0.5, 3.0, 0.1)) + \
              list(np.arange(3.0, 6.0, 0.2)) + \
              list(np.arange(6.0, 12.0, 0.5))

for Phi0 in phi0_values:
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om is None:
        print(f"  {Phi0:6.1f}  --- BVP failed ---")
        continue
    prev = sol

    omegas_l0 = cavity_ground(sol.x, sol.y[0], l_val=0, N=3000)
    om_tau = omegas_l0[0] if len(omegas_l0) > 0 else None
    om_mu = omegas_l0[1] if len(omegas_l0) > 1 else None

    if om_tau is not None:
        gap = Om - om_tau
        ratio = om_tau / Om
        results.append((Phi0, Om, om_tau, om_mu, gap, ratio))
        mu_str = f"{om_mu:8.4f}" if om_mu is not None else "    ---"
        print(f"  {Phi0:6.2f} {Om:8.4f} {om_tau:8.4f} {mu_str} {gap:8.4f} {ratio:8.4f}")
    else:
        print(f"  {Phi0:6.2f} {Om:8.4f}   (no bound state)")

# Analysis
print(f"\n{'='*80}")
print(f"  ANALYSIS")
print(f"{'='*80}")

if len(results) > 2:
    phi0s = [r[0] for r in results]
    gaps = [r[4] for r in results]
    ratios = [r[5] for r in results]

    min_gap = min(results, key=lambda x: abs(x[4]))
    print(f"\n  Minimum gap: Φ₀={min_gap[0]:.2f}, gap={min_gap[4]:.4f}")
    print(f"    Ω_bg={min_gap[1]:.4f}, ω_tau={min_gap[2]:.4f}")

    max_ratio = max(results, key=lambda x: x[5])
    print(f"  Maximum ratio ω_τ/Ω_bg: Φ₀={max_ratio[0]:.2f}, ratio={max_ratio[5]:.4f}")

    if any(g <= 0 for g in gaps):
        print(f"\n  *** CROSSING FOUND! Self-consistent point exists! ***")
        for i in range(1, len(results)):
            if results[i-1][4] * results[i][4] < 0:
                p1, p2 = results[i-1], results[i]
                phi0_cross = p1[0] + (p2[0]-p1[0]) * p1[4]/(p1[4]-p2[4])
                print(f"  Between Φ₀={p1[0]:.2f} and {p2[0]:.2f}")
                print(f"  Interpolated crossing: Φ₀ ≈ {phi0_cross:.2f}")
    else:
        print(f"\n  No crossing found. Gap is always positive (Ω_bg > ω_tau).")
        print(f"  The gap {'decreases' if gaps[-1] < gaps[0] else 'increases'} with Φ₀.")

        # Check l=2 mode at the point of minimum gap
        p = min_gap
        # Find the solution at this Φ₀
        Om_check, sol_check = solve_osc(p[0], r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
        if Om_check:
            omegas_l2 = cavity_ground(sol_check.x, sol_check.y[0], l_val=2, N=3000)
            if len(omegas_l2) > 0:
                print(f"\n  At minimum gap (Φ₀={p[0]:.2f}):")
                print(f"    ω_e(l=2) = {omegas_l2[0]:.6f}")
                m_tau = E_1D(p[2])
                m_mu = E_1D(p[3]) if p[3] else 0
                m_e = E_1D(omegas_l2[0])
                if m_e > 0 and m_mu > 0:
                    Q = koide_Q(m_tau, m_mu, m_e)
                    print(f"    Q = {Q:.6f} (target: {2/3:.6f})")

# Phase 2: Check what happens to the ratio for large Φ₀
print(f"\n{'='*80}")
print(f"  TREND: ω_τ/Ω_bg ratio vs Φ₀")
print(f"{'='*80}")
for r in results[::max(1, len(results)//20)]:
    bar = "#" * int(r[5] * 50)
    print(f"  Φ₀={r[0]:5.1f}: ω_τ/Ω_bg = {r[5]:.4f} |{bar}")
