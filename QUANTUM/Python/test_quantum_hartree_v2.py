"""
Test specific combinations of quantum Hartree η to find which
physical scenario gives θ₀ = 2/9.

Key: the l=2 electron mode's 5-fold degeneracy makes its
zero-point contribution huge. But its OVERLAP with the
l=0 potential is weighted differently.
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
print("  QUANTUM HARTREE: Which modes' zero-point gives θ₀ = 2/9?")
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
f_Phi = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)

# Get bare eigenfunctions
r0, om_l0, ev_l0, _ = cavity_full(r_bg, Phi_bg, 0, N=3000)
r2, om_l2, ev_l2, _ = cavity_full(r_bg, Phi_bg, 2, N=3000)

# Prepare mode data: (name, l, ω, u_vec, r_grid, norm)
u_tau = ev_l0[:, 0]; norm_tau = trapezoid(u_tau**2, r0)
u_mu = ev_l0[:, 1]; norm_mu = trapezoid(u_mu**2, r0)
u_e = ev_l2[:, 0]; norm_e = trapezoid(u_e**2, r2)

# For each mode, compute its δV contribution on a common grid
r_ref = np.linspace(0.01, r_bg[-1]*0.85, 3000)
Phi_ref = f_Phi(r_ref)
d2F_ref = d2nl_func(Phi_ref)

def mode_dV_on_ref(u_vec, r_mode, norm, l_val, omega):
    """δV from one mode's vacuum zero-point on r_ref grid."""
    u_interp = interp1d(r_mode, u_vec, fill_value=0, bounds_error=False)
    u_on_ref = u_interp(r_ref)
    psi_sq = u_on_ref**2 / (np.maximum(r_ref, 1e-10)**2 * norm)
    degeneracy = 2 * l_val + 1
    eta = degeneracy / (8 * PI * omega)
    return -0.5 * d2F_ref * psi_sq * eta

dV_tau = mode_dV_on_ref(u_tau, r0, norm_tau, 0, om_l0[0])
dV_mu = mode_dV_on_ref(u_mu, r0, norm_mu, 0, om_l0[1])
dV_e = mode_dV_on_ref(u_e, r2, norm_e, 2, om_l2[0])

# Spatial overlap analysis
print(f"\n  Mode δV contributions at r=0:")
print(f"    tau:  max|δV| = {np.max(np.abs(dV_tau)):.6f}")
print(f"    muon: max|δV| = {np.max(np.abs(dV_mu)):.6f}")
print(f"    elec: max|δV| = {np.max(np.abs(dV_e)):.6f}")
print(f"    (elec is large because degeneracy 2l+1 = 5)")

# Test various PHYSICAL scenarios
scenarios = [
    ("bare (η=0)",                   np.zeros_like(r_ref)),
    ("τ vacuum only",                dV_tau),
    ("τ + μ vacuum",                 dV_tau + dV_mu),
    ("τ + μ + e vacuum (all)",       dV_tau + dV_mu + dV_e),
    ("τ + 0.35×μ vacuum",           dV_tau + 0.35*dV_mu),
    ("τ vacuum × 1.235 (= η*)",     dV_tau * 1.235),
    ("τ + e vacuum (no μ)",          dV_tau + dV_e),
    ("e vacuum only",                dV_e),
]

# Also: l=2 without degeneracy factor
dV_e_no_deg = mode_dV_on_ref(u_e, r2, norm_e, 0, om_l2[0])
scenarios.append(("e vacuum (no degeneracy)", dV_e_no_deg))
scenarios.append(("τ + e(no deg)",            dV_tau + dV_e_no_deg))

print(f"\n  {'Scenario':>35} {'ω_e':>10} {'Q':>8} {'θ₀':>8} {'Δθ/Δθ_max':>10} {'η_eff':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

for name, dV_total in scenarios:
    if np.all(dV_total == 0):
        dV_func = None
    else:
        dV_interp = interp1d(r_ref, dV_total, fill_value=0, bounds_error=False)
        dV_func = dV_interp

    r0i, om0i, ev0i, _ = cavity_full(r_bg, Phi_bg, 0, N=3000, delta_V_func=dV_func)
    r2i, om2i, ev2i, _ = cavity_full(r_bg, Phi_bg, 2, N=3000, delta_V_func=dV_func)

    if len(om0i) < 2 or len(om2i) < 1 or om2i[0] >= 0.9999:
        print(f"  {name:>35}  --- electron lost ---")
        continue

    m_t = E_1D(om0i[0])
    m_m = E_1D(om0i[1])
    m_e = E_1D(om2i[0])
    if m_e <= 0:
        print(f"  {name:>35}  --- electron unbound ---")
        continue

    Q = koide_Q(m_t, m_m, m_e)
    theta = koide_angle(m_t/m_e, m_m/m_e, 1.0)
    dtheta = theta - 2/9
    frac = (theta - 0.21668) / (2/9 - 0.21668)  # fraction of needed shift

    # Effective η
    omega_e_bare = 0.999037
    omega_e_new = om2i[0]
    eta_eff = 0.074 * (omega_e_bare - omega_e_new) / (omega_e_bare - 0.998995) if abs(omega_e_bare - 0.998995) > 1e-10 else 0

    print(f"  {name:>35} {om2i[0]:10.6f} {Q:8.5f} {theta:8.5f} "
          f"{frac:10.1%} {eta_eff:8.3f}")

# Key comparison
print(f"\n{'='*80}")
print(f"  KEY FINDINGS")
print(f"{'='*80}")
eta_tau = 1/(8*PI*om_l0[0])
eta_mu = 1/(8*PI*om_l0[1])
eta_e = 5/(8*PI*om_l2[0])
print(f"  η_τ(vac) = 1/(8πω_τ)    = {eta_tau:.4f}")
print(f"  η_μ(vac) = 1/(8πω_μ)    = {eta_mu:.4f}")
print(f"  η_e(vac) = 5/(8πω_e)    = {eta_e:.4f}")
print(f"  η* (gives θ₀=2/9)       = 0.074")
print(f"  η_τ + η_μ               = {eta_tau + eta_mu:.4f}")
print(f"  η_τ + 0.35×η_μ          = {eta_tau + 0.35*eta_mu:.4f}")
print(f"  η_τ × 1.235             = {eta_tau*1.235:.4f}")

# The physics: modes contribute to the l=0 effective potential
# proportional to their RADIAL overlap with the electron's
# sensitive region. The electron lives near the oscillon edge.
# Let's compute these overlaps.
print(f"\n  Radial overlap with electron eigenfunction:")

u_e_interp = interp1d(r2, u_e**2/norm_e, fill_value=0, bounds_error=False)
weight_e = u_e_interp(r_ref) / np.maximum(r_ref, 1e-10)**2

u_tau_sq = interp1d(r0, u_tau**2/norm_tau, fill_value=0, bounds_error=False)(r_ref) / np.maximum(r_ref, 1e-10)**2
u_mu_sq = interp1d(r0, u_mu**2/norm_mu, fill_value=0, bounds_error=False)(r_ref) / np.maximum(r_ref, 1e-10)**2

overlap_tau_e = trapezoid(u_tau_sq * weight_e, r_ref)
overlap_mu_e = trapezoid(u_mu_sq * weight_e, r_ref)

print(f"  ⟨ψ_τ²|ψ_e²⟩ = {overlap_tau_e:.6f}")
print(f"  ⟨ψ_μ²|ψ_e²⟩ = {overlap_mu_e:.6f}")
print(f"  Ratio μ/τ   = {overlap_mu_e/overlap_tau_e:.4f}")
print(f"  ⟨Expected: η* = η_τ × (1 + {overlap_mu_e/overlap_tau_e:.2f} × η_μ/η_τ)⟩")
