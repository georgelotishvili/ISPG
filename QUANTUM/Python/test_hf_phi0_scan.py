"""
Φ₀ scan with different Hartree-Fock prescriptions.

Tests three approaches:
A. l=0 bare, l=2 gets δV from τ+μ (semiclassical: heavy modes unaffected)
B. Full channel-separated HF (l=0←e, l=2←τ+μ)
C. Same δV both channels (old approach, for comparison)

Scans Φ₀ from 2.30 to 2.50 to find where Q=2/3 and θ₀=2/9 are optimal.
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

def cavity_solve(r_bg, Phi_bg, l_val, N=2500, delta_V_func=None):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    V = 1.0 - dnl_func(Phi) + l_val*(l_val+1)/r**2
    if delta_V_func is not None:
        V = V + delta_V_func(r)
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, evecs = eigsh(H, k=min(6, N-2), which='SM')
    bound = evals < 1.0
    order = np.argsort(evals[bound])
    omegas = np.sqrt(np.maximum(evals[bound][order], 0))
    vecs = evecs[:, np.where(bound)[0][order]]
    return r, omegas, vecs, dr

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

def compute_delta_V(u_vec, r_grid, norm, l_val, omega, r_ref, d2F_ref):
    u_interp = interp1d(r_grid, u_vec, fill_value=0, bounds_error=False)
    u_on_ref = u_interp(r_ref)
    psi_sq = u_on_ref**2 / (np.maximum(r_ref, 1e-10)**2 * norm)
    eta = (2 * l_val + 1) / (8 * PI * omega)
    return -0.5 * d2F_ref * psi_sq * eta

# ===============================================================
print("=" * 80)
print("  Φ₀ SCAN WITH HARTREE-FOCK PRESCRIPTIONS")
print("=" * 80)

# Build oscillon chain
prev = None
for p in np.arange(0.05, 2.25, 0.05):
    Om, sol = solve_osc(p, r_prev=prev.x if prev else None,
                        y_prev=prev.y if prev else None,
                        p_prev=prev.p if prev else None)
    if Om: prev = sol

# Scan Φ₀
phi0_vals = np.arange(2.30, 2.55, 0.01)

approaches = {
    'Bare':     {'dV_l0': 'none', 'dV_l2': 'none'},
    'A: l2←τμ': {'dV_l0': 'none', 'dV_l2': 'tau+mu'},
    'B: HF':    {'dV_l0': 'elec', 'dV_l2': 'tau+mu'},
    'C: same':  {'dV_l0': 'tau+mu', 'dV_l2': 'tau+mu'},
}

for approach_name, config in approaches.items():
    print(f"\n{'='*80}")
    print(f"  Approach: {approach_name}")
    print(f"  l=0 ← {config['dV_l0']},  l=2 ← {config['dV_l2']}")
    print(f"{'='*80}")
    print(f"  {'Φ₀':>6} {'Q':>10} {'θ₀':>10} {'|ΔQ|':>10} {'|Δθ|':>10} "
          f"{'cost':>10} {'τ/e':>8} {'μ/e':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} "
          f"{'─'*10} {'─'*8} {'─'*8}")

    best_cost = 1e10
    best_result = None

    for Phi0 in phi0_vals:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
        if Om is None: continue
        prev = sol

        r_bg = sol.x
        Phi_bg = sol.y[0]
        f_Phi = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)

        # Bare modes
        r0, om0, ev0, _ = cavity_solve(r_bg, Phi_bg, 0)
        r2, om2, ev2, _ = cavity_solve(r_bg, Phi_bg, 2)

        if len(om0) < 2 or len(om2) < 1 or om2[0] >= 0.9999:
            continue

        # Reference grid for δV
        N_ref = 3000
        r_ref = np.linspace(0.01, r_bg[-1]*0.85, N_ref)
        Phi_ref = f_Phi(r_ref)
        d2F_ref = d2nl_func(Phi_ref)

        u_tau = ev0[:, 0]; norm_tau = trapezoid(u_tau**2, r0)
        u_mu  = ev0[:, 1]; norm_mu  = trapezoid(u_mu**2, r0)
        u_e   = ev2[:, 0]; norm_e   = trapezoid(u_e**2, r2)

        dV_tau = compute_delta_V(u_tau, r0, norm_tau, 0, om0[0], r_ref, d2F_ref)
        dV_mu  = compute_delta_V(u_mu,  r0, norm_mu,  0, om0[1], r_ref, d2F_ref)
        dV_e   = compute_delta_V(u_e,   r2, norm_e,   2, om2[0], r_ref, d2F_ref)

        # Build channel-specific δV
        dV_l0_arr = np.zeros_like(r_ref)
        dV_l2_arr = np.zeros_like(r_ref)

        if config['dV_l0'] == 'tau+mu':
            dV_l0_arr = dV_tau + dV_mu
        elif config['dV_l0'] == 'elec':
            dV_l0_arr = dV_e

        if config['dV_l2'] == 'tau+mu':
            dV_l2_arr = dV_tau + dV_mu

        dV_l0_f = None if np.allclose(dV_l0_arr, 0) else \
                  interp1d(r_ref, dV_l0_arr, fill_value=0, bounds_error=False)
        dV_l2_f = None if np.allclose(dV_l2_arr, 0) else \
                  interp1d(r_ref, dV_l2_arr, fill_value=0, bounds_error=False)

        _, om0_h, _, _ = cavity_solve(r_bg, Phi_bg, 0, delta_V_func=dV_l0_f)
        _, om2_h, _, _ = cavity_solve(r_bg, Phi_bg, 2, delta_V_func=dV_l2_f)

        if len(om0_h) < 2 or len(om2_h) < 1 or om2_h[0] >= 0.9999:
            continue

        m_t = E_1D(om0_h[0])
        m_m = E_1D(om0_h[1])
        m_e = E_1D(om2_h[0])
        if m_e <= 0: continue

        Q = koide_Q(m_t, m_m, m_e)
        theta = koide_angle(m_t/m_e, m_m/m_e, 1.0)
        dQ = abs(Q - 2/3)
        dth = abs(theta - 2/9)
        cost = (dQ / (2/3))**2 + (dth / (2/9))**2

        te = m_t / m_e
        me_r = m_m / m_e

        marker = " *" if cost < best_cost else ""
        if cost < best_cost:
            best_cost = cost
            best_result = (Phi0, Om, Q, theta, dQ, dth, te, me_r, om0_h, om2_h)

        print(f"  {Phi0:6.2f} {Q:10.7f} {theta:10.7f} {dQ:10.2e} {dth:10.2e} "
              f"{cost:10.2e} {te:8.1f} {me_r:8.1f}{marker}")

    if best_result:
        Phi0, Om, Q, theta, dQ, dth, te, me_r, om0h, om2h = best_result
        print(f"\n  BEST for {approach_name}:")
        print(f"    Φ₀ = {Phi0:.2f}, Ω_bg = {Om:.6f}")
        print(f"    Q = {Q:.7f} (|ΔQ| = {dQ:.2e}, {dQ/(2/3)*100:.4f}%)")
        print(f"    θ₀ = {theta:.7f} (|Δθ| = {dth:.2e}, {dth/(2/9)*100:.4f}%)")
        print(f"    τ/e = {te:.1f} (expt: 3477.2)")
        print(f"    μ/e = {me_r:.1f} (expt: 206.8)")
        print(f"    τ/e × 4/π = {te*4/PI:.1f}")
        print(f"    μ/e × 4/π = {me_r*4/PI:.1f}")

print(f"\n{'='*80}")
print("  DONE")
print("=" * 80)
