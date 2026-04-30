"""
2D optimization: find (Φ₀, η) where Q = 2/3 AND θ₀ = 2/9 simultaneously.
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

def d2nl_func(Phi):
    return 2*ALPHA*np.exp(-ALPHA*Phi) - ALPHA**2*Phi*np.exp(-ALPHA*Phi)

def solve_osc(Phi0, r_max=60.0, r_prev=None, y_prev=None, p_prev=None):
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
                    tol=1e-6, max_nodes=40000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None

def cavity_eigs(r_bg, Phi_bg, l_val, N=2000, k_return=4, delta_V=None):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    if delta_V is not None:
        f_dV = interp1d(delta_V[0], delta_V[1], fill_value=0, bounds_error=False)
        V = V + f_dV(r)
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, evecs = eigsh(H, k=min(k_return, N-2), which='SM')
    bound = evals < 1.0
    order = np.argsort(evals[bound])
    return r, np.sqrt(np.maximum(evals[bound][order], 0)), evecs[:, np.where(bound)[0][order]], dr

def E_1D(omega):
    k2 = 1.0 - omega**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*omega**2 + 1)

def koide_Q(m1, m2, m3):
    s = m1 + m2 + m3
    sr = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    return s / sr**2

def koide_angle(m1, m2, m3):
    sr = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    M = sr / 3.0
    c0 = (np.sqrt(m1)/M - 1) / np.sqrt(2)
    return np.arccos(np.clip(c0, -1, 1))

print("=" * 80)
print("  2D OPTIMIZATION: (Φ₀, η) → Q = 2/3 AND θ₀ = 2/9")
print("=" * 80)

expt_tau_e = 1776.86 / 0.511
expt_mu_e = 105.658 / 0.511

# Build oscillon chain
prev = None
for p in np.arange(0.05, 2.25, 0.05):
    if prev:
        Om, sol = solve_osc(p, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(p)
    if Om: prev = sol

# Scan grid
print(f"\n  {'Φ₀':>6} {'η':>6} {'Q':>8} {'θ₀':>8} {'|ΔQ|':>10} {'|Δθ|':>10} {'cost':>10} {'τ/e':>8}")

best = None
best_cost = 1e10

for Phi0_10 in range(230, 250, 1):
    Phi0 = Phi0_10 / 100.0
    Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    if Om is None: continue
    prev = sol

    # Get unperturbed tau eigenfunction for Hartree
    r0, om0, ev0, dr0 = cavity_eigs(sol.x, sol.y[0], 0, N=2000)
    if len(om0) < 2: continue

    f_Phi = interp1d(sol.x, sol.y[0], fill_value=0, bounds_error=False)
    Phi_r0 = f_Phi(r0)
    d2F = d2nl_func(Phi_r0)
    u_tau = ev0[:, 0]
    norm_tau = trapezoid(u_tau**2, r0)
    psi_tau_sq = (u_tau / np.maximum(r0, 1e-10))**2 / norm_tau

    for eta_10 in range(0, 20):
        eta = eta_10 / 100.0
        dV_vals = -0.5 * d2F * psi_tau_sq * eta
        delta_V = (r0, dV_vals)

        _, om0h, _, _ = cavity_eigs(sol.x, sol.y[0], 0, N=2000, delta_V=delta_V)
        _, om2h, _, _ = cavity_eigs(sol.x, sol.y[0], 2, N=2000, delta_V=delta_V)

        if len(om0h) < 2 or len(om2h) < 1 or om2h[0] >= 0.9999:
            continue

        m_t = E_1D(om0h[0])
        m_m = E_1D(om0h[1])
        m_e = E_1D(om2h[0])
        if m_e <= 0: continue

        Q = koide_Q(m_t, m_m, m_e)
        te = m_t / m_e
        theta = koide_angle(te, m_m/m_e, 1.0)

        dQ = abs(Q - 2/3)
        dtheta = abs(theta - 2/9)
        cost = (dQ / (2/3))**2 + (dtheta / (2/9))**2

        if cost < best_cost:
            best_cost = cost
            best = (Phi0, eta, Q, theta, dQ, dtheta, cost, te, m_m/m_e,
                    om0h[0], om0h[1], om2h[0])
            print(f"  {Phi0:6.2f} {eta:6.2f} {Q:8.5f} {theta:8.5f} "
                  f"{dQ:10.6f} {dtheta:10.6f} {cost:10.2e} {te:8.1f} *")

if best:
    print(f"\n{'='*80}")
    print(f"  BEST JOINT FIT")
    print(f"{'='*80}")
    Phi0, eta, Q, theta, dQ, dtheta, cost, te, me_r, wt, wm, we = best
    print(f"  Φ₀  = {Phi0:.2f}")
    print(f"  η   = {eta:.2f}")
    print(f"  Q   = {Q:.6f} (target: {2/3:.6f}, error: {dQ/(2/3)*100:.3f}%)")
    print(f"  θ₀  = {theta:.6f} (target: {2/9:.6f}, error: {dtheta/(2/9)*100:.3f}%)")
    print(f"  ω_τ = {wt:.6f}")
    print(f"  ω_μ = {wm:.6f}")
    print(f"  ω_e = {we:.6f}")
    print(f"  τ/e = {te:.1f} (expt: {expt_tau_e:.1f}, error: {abs(te-expt_tau_e)/expt_tau_e*100:.1f}%)")
    print(f"  μ/e = {me_r:.1f} (expt: {expt_mu_e:.1f}, error: {abs(me_r-expt_mu_e)/expt_mu_e*100:.1f}%)")

    # With 4/π correction
    te_corr = te * 4/PI
    me_corr = me_r * 4/PI
    print(f"\n  With 4/π correction:")
    print(f"  τ/e = {te_corr:.1f} (expt: {expt_tau_e:.1f}, error: {abs(te_corr-expt_tau_e)/expt_tau_e*100:.1f}%)")
    print(f"  μ/e = {me_corr:.1f} (expt: {expt_mu_e:.1f}, error: {abs(me_corr-expt_mu_e)/expt_mu_e*100:.1f}%)")

    # Compare with η=0
    print(f"\n  Comparison with η=0 (no Hartree):")
    print(f"  η=0: Q=0.66674, θ₀=0.21668, τ/e=2741")
    print(f"  η={eta:.2f}: Q={Q:.5f}, θ₀={theta:.5f}, τ/e={te:.0f}")
