"""
Quantum Hartree: compute η from first principles (QFT normalization).

The mode expansion of the quantum field gives:
  ⟨δΦ²⟩ = Σ_n (2n_occ + 1)/(2ω_n) × |ψ_n(r)|²/(4π)

For vacuum (n_occ=0) zero-point fluctuations:
  ⟨0|δΦ²|0⟩_tau = 1/(2ω_τ) × |u_τ(r)|²/(r² × 4π × norm_τ)

This gives η_QM = 1/(8πω_τ) which we compare with η*=0.074.
If they match, θ₀ = 2/9 is a PREDICTION, not a fit!
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

def cavity_full(r_bg, Phi_bg, l_val, N=3000, delta_V_func=None):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
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

print("=" * 80)
print("  QUANTUM HARTREE: η from QFT zero-point fluctuations")
print("=" * 80)

# Build background
prev = None
for p in np.arange(0.05, 2.40, 0.05):
    Om, sol = solve_osc(p, r_prev=prev.x if prev else None,
                        y_prev=prev.y if prev else None,
                        p_prev=prev.p if prev else None)
    if Om: prev = sol

Phi0 = 2.35
Om_bg, sol_bg = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
r_bg = sol_bg.x
Phi_bg = sol_bg.y[0]
print(f"  Background: Φ₀={Phi0}, Ω_bg={Om_bg:.6f}")

# Get all cavity modes
r0, om_l0, ev_l0, dr0 = cavity_full(r_bg, Phi_bg, 0, N=3000)
r2, om_l2, ev_l2, dr2 = cavity_full(r_bg, Phi_bg, 2, N=3000)

print(f"\n  Bare spectrum:")
print(f"    ω_tau   = {om_l0[0]:.6f}  (l=0, n=0)")
print(f"    ω_muon  = {om_l0[1]:.6f}  (l=0, n=1)")
print(f"    ω_elec  = {om_l2[0]:.6f}  (l=2, n=0)")

# Compute quantum η for each mode
print(f"\n{'='*80}")
print(f"  QUANTUM η COMPUTATION")
print(f"{'='*80}")

f_Phi = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)

# For each mode, compute the QFT-normalized contribution
# η_n = 1/(8π ω_n) for vacuum zero-point (l=0 modes)
# η_n = (2l+1)/(8π ω_n) for l>0 (summed over m)

modes = [
    ("tau  (l=0,n=0)", 0, om_l0[0], ev_l0[:, 0], r0, 0),
    ("muon (l=0,n=1)", 0, om_l0[1], ev_l0[:, 1], r0, 0),
    ("elec (l=2,n=0)", 2, om_l2[0], ev_l2[:, 0], r2, 2),
]

print(f"\n  {'Mode':>20} {'ω':>8} {'η_vac':>10} {'η_1part':>10} {'(2l+1)':>6}")

eta_vac_total = 0.0
eta_1part_total = 0.0

for name, l, omega, u_vec, r_grid, l_val in modes:
    norm = trapezoid(u_vec**2, r_grid)
    degeneracy = 2*l_val + 1
    eta_vac = degeneracy / (8 * PI * omega)
    eta_1part = degeneracy * 3 / (8 * PI * omega)
    eta_vac_total += eta_vac
    eta_1part_total += eta_1part
    print(f"  {name:>20} {omega:8.4f} {eta_vac:10.4f} {eta_1part:10.4f} {degeneracy:6d}")

# l=1 translational mode (Goldstone)
# Not included because it's a zero mode with different physics
print(f"\n  {'TOTAL':>20} {'':>8} {eta_vac_total:10.4f} {eta_1part_total:10.4f}")

print(f"\n  η* (empirical, gives θ₀=2/9) = 0.074")
print(f"  η_vacuum (all modes)          = {eta_vac_total:.4f}")
print(f"  η_vacuum (tau only)           = {1/(8*PI*om_l0[0]):.4f}")
print(f"  η_particle (tau, n=1)         = {1/(4*PI*om_l0[0]):.4f}")

# The "right" η depends on what modes are occupied.
# In the oscillon, the background IS the condensate, so modes
# start in the vacuum state. The relevant η is the VACUUM zero-point.

print(f"\n{'='*80}")
print(f"  ITERATIVE HARTREE-FOCK WITH QUANTUM η")
print(f"{'='*80}")

# Method: use the QFT-determined η (not fitted) and iterate
# δV = -½ F'' × Σ_modes (2l+1)/(8πω_n) × |u_n(r)|²/(r² × norm_n)

def compute_hartree_dV(r_grid, r_bg, Phi_bg, modes_data):
    """Compute total Hartree δV from all mode zero-point fluctuations."""
    f_Phi = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi_on_grid = f_Phi(r_grid)
    d2F = d2nl_func(Phi_on_grid)

    dV = np.zeros_like(r_grid)
    for l_val, omega, u_vec, r_mode, norm in modes_data:
        degeneracy = 2 * l_val + 1
        eta = degeneracy / (8 * PI * omega)
        u_interp = interp1d(r_mode, u_vec, fill_value=0, bounds_error=False)
        u_on_grid = u_interp(r_grid)
        psi_sq = u_on_grid**2 / (np.maximum(r_grid, 1e-10)**2 * norm)
        dV += -0.5 * d2F * psi_sq * eta
    return dV

# Iteration loop
print(f"\n  Iterating (quantum η, all modes' zero-point):")
print(f"  {'iter':>4} {'ω_τ':>8} {'ω_μ':>8} {'ω_e':>10} {'Q':>8} {'θ₀':>8} {'Δθ':>8}")

dV_func = None
for iteration in range(8):
    # Solve cavity with current δV
    r0i, om0i, ev0i, _ = cavity_full(r_bg, Phi_bg, 0, N=3000, delta_V_func=dV_func)
    r2i, om2i, ev2i, _ = cavity_full(r_bg, Phi_bg, 2, N=3000, delta_V_func=dV_func)

    if len(om0i) < 2 or len(om2i) < 1:
        print(f"  {iteration:4d}  --- lost modes ---")
        break

    m_t = E_1D(om0i[0])
    m_m = E_1D(om0i[1])
    m_e = E_1D(om2i[0])
    if m_e <= 0:
        print(f"  {iteration:4d}  --- electron unbound ---")
        break

    Q = koide_Q(m_t, m_m, m_e)
    theta = koide_angle(m_t/m_e, m_m/m_e, 1.0)
    dtheta = theta - 2/9
    print(f"  {iteration:4d} {om0i[0]:8.4f} {om0i[1]:8.4f} {om2i[0]:10.6f} "
          f"{Q:8.5f} {theta:8.5f} {dtheta:+8.5f}")

    # Compute new δV from current eigenfunctions
    modes_data = [
        (0, om0i[0], ev0i[:, 0], r0i, trapezoid(ev0i[:, 0]**2, r0i)),
        (0, om0i[1], ev0i[:, 1], r0i, trapezoid(ev0i[:, 1]**2, r0i)),
        (2, om2i[0], ev2i[:, 0], r2i, trapezoid(ev2i[:, 0]**2, r2i)),
    ]

    # Build δV on a reference grid
    r_ref = np.linspace(0.01, r_bg[-1]*0.85, 3000)
    dV_values = compute_hartree_dV(r_ref, r_bg, Phi_bg, modes_data)
    dV_interp = interp1d(r_ref, dV_values, fill_value=0, bounds_error=False)
    dV_func = dV_interp

# Also try: tau-only quantum η
print(f"\n  Tau-only quantum η iteration:")
print(f"  {'iter':>4} {'ω_τ':>8} {'ω_μ':>8} {'ω_e':>10} {'Q':>8} {'θ₀':>8} {'Δθ':>8}")

dV_func = None
for iteration in range(8):
    r0i, om0i, ev0i, _ = cavity_full(r_bg, Phi_bg, 0, N=3000, delta_V_func=dV_func)
    r2i, om2i, ev2i, _ = cavity_full(r_bg, Phi_bg, 2, N=3000, delta_V_func=dV_func)

    if len(om0i) < 2 or len(om2i) < 1:
        print(f"  {iteration:4d}  --- lost modes ---")
        break

    m_t = E_1D(om0i[0])
    m_m = E_1D(om0i[1])
    m_e = E_1D(om2i[0])
    if m_e <= 0:
        print(f"  {iteration:4d}  --- electron unbound ---")
        break

    Q = koide_Q(m_t, m_m, m_e)
    theta = koide_angle(m_t/m_e, m_m/m_e, 1.0)
    dtheta = theta - 2/9
    print(f"  {iteration:4d} {om0i[0]:8.4f} {om0i[1]:8.4f} {om2i[0]:10.6f} "
          f"{Q:8.5f} {theta:8.5f} {dtheta:+8.5f}")

    # Tau-only Hartree
    modes_data = [
        (0, om0i[0], ev0i[:, 0], r0i, trapezoid(ev0i[:, 0]**2, r0i)),
    ]
    r_ref = np.linspace(0.01, r_bg[-1]*0.85, 3000)
    dV_values = compute_hartree_dV(r_ref, r_bg, Phi_bg, modes_data)
    dV_interp = interp1d(r_ref, dV_values, fill_value=0, bounds_error=False)
    dV_func = dV_interp

# Summary
print(f"\n{'='*80}")
print(f"  SUMMARY")
print(f"{'='*80}")
eta_tau_vac = 1/(8*PI*om_l0[0])
eta_all_vac = eta_vac_total
print(f"  η*(empirical)        = 0.074   (gives θ₀ = 2/9)")
print(f"  η_QM(tau, vacuum)    = {eta_tau_vac:.4f}  (= 1/(8πω_τ))")
print(f"  η_QM(all, vacuum)    = {eta_all_vac:.4f}  (= Σ (2l+1)/(8πω_n))")
print(f"  η_QM(tau, 1 particle)= {2*eta_tau_vac:.4f}  (= 1/(4πω_τ))")
print(f"  Ratio η*/η_tau_vac   = {0.074/eta_tau_vac:.3f}")
print(f"  Ratio η*/η_all_vac   = {0.074/eta_all_vac:.3f}")
