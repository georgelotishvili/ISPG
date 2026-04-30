"""
Hartree self-consistency: modes backreact on the background.

Idea: the cavity modes modify the effective potential. This shifts
the eigenfrequencies, potentially moving θ₀ from 0.217 to 2/9.

Method:
1. Solve background oscillon → Φ₀(r), Ω_bg
2. Compute cavity eigenfunctions u_n(r)
3. The mode backreaction modifies the potential:
   V_eff → V_eff + δV(r) where δV ~ -Σ c_n |ψ_n|²
4. Re-diagonalize with the modified potential
5. Check if the new spectrum gives θ₀ closer to 2/9

The key physics: the tau mode (ground state) has the largest
amplitude and deepens the potential well. This shifts all modes
to lower frequencies, but the electron (barely bound) is
affected the most.
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
    """Second derivative of the nonlinearity, for Hartree correction."""
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

def cavity_eigs(r_bg, Phi_bg, l_val, N=3000, k_return=5, delta_V=None):
    """Compute cavity eigenvalues with optional potential correction δV."""
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    if delta_V is not None:
        f_dV = interp1d(r_bg, delta_V, fill_value=0, bounds_error=False)
        V = V + f_dV(r)
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, evecs = eigsh(H, k=min(k_return, N-2), which='SM')
    bound = evals < 1.0
    order = np.argsort(evals[bound])
    return r, np.sqrt(np.maximum(evals[bound][order], 0)), evecs[:, np.where(bound)[0][order]], dr

def E_1D(omega):
    k2 = 1.0 - omega**2
    if k2 <= 0:
        return 0.0
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

# Build oscillon
print("=" * 80)
print("  HARTREE SELF-CONSISTENCY: Mode backreaction on the background")
print("=" * 80)

prev = None
for Phi0 in np.arange(0.05, 2.40, 0.05):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om:
        prev = sol

Om_bg, sol_bg = solve_osc(2.35, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
print(f"  Background: Φ₀=2.35, Ω_bg={Om_bg:.6f}")

r_bg = sol_bg.x
Phi_bg = sol_bg.y[0]
f_Phi = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)

# Step 1: Unperturbed spectrum
print(f"\n--- Step 1: Unperturbed spectrum ---")
r0, om0, ev0, dr0 = cavity_eigs(r_bg, Phi_bg, 0, N=3000)
r2, om2, ev2, dr2 = cavity_eigs(r_bg, Phi_bg, 2, N=3000)
print(f"  ω_tau  = {om0[0]:.6f}")
print(f"  ω_mu   = {om0[1]:.6f}")
print(f"  ω_e    = {om2[0]:.6f}")

m_t = E_1D(om0[0])
m_m = E_1D(om0[1])
m_e = E_1D(om2[0])
Q0 = koide_Q(m_t, m_m, m_e)
theta0 = koide_angle(m_t/m_e, m_m/m_e, 1.0)
print(f"  Q = {Q0:.6f}, θ₀ = {theta0:.6f} (2/9 = {2/9:.6f})")

# Step 2: Compute mode profiles (ψ = u/r)
print(f"\n--- Step 2: Mode backreaction (Hartree correction) ---")

# The Hartree correction to the potential comes from the
# second-order nonlinear term:
# δV(r) = -½ d²F_NL/dΦ² × <δΦ²>
# where <δΦ²> = Σ |ψ_n(r)|² × occupation
#
# For the tau mode (ground state, always occupied):
# δV_tau(r) = -½ d²F_NL/dΦ²|_{Φ_bg} × |ψ_tau(r)|²

Phi_on_r0 = f_Phi(r0)
d2F = d2nl_func(Phi_on_r0)

# tau eigenfunction ψ = u/r
u_tau = ev0[:, 0]
norm_tau = trapezoid(u_tau**2, r0)
psi_tau = u_tau / np.maximum(r0, 1e-10) / np.sqrt(norm_tau)

# Scan different "occupation strengths" η (coupling constant)
# Physical: η = amplitude² of the mode occupation
print(f"\n  Scanning Hartree coupling strength η:")
print(f"  {'η':>8} {'ω_τ':>8} {'ω_μ':>8} {'ω_e':>10} {'Q':>8} {'θ₀':>8} {'Δθ':>8} {'τ/e':>8}")

expt_tau_e = 1776.86 / 0.511

for eta in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    # δV on the r0 grid
    delta_V_on_r0 = -0.5 * d2F * psi_tau**2 * eta

    # Build δV on the background grid for passing to cavity_eigs
    f_dV = interp1d(r0, delta_V_on_r0, fill_value=0, bounds_error=False)
    delta_V_bg = f_dV(r_bg)

    r0h, om0h, _, _ = cavity_eigs(r_bg, Phi_bg, 0, N=3000, delta_V=delta_V_bg)
    r2h, om2h, _, _ = cavity_eigs(r_bg, Phi_bg, 2, N=3000, delta_V=delta_V_bg)

    if len(om0h) >= 2 and len(om2h) >= 1 and om2h[0] < 0.9999:
        m_t = E_1D(om0h[0])
        m_m = E_1D(om0h[1])
        m_e = E_1D(om2h[0])
        if m_e > 0:
            Q = koide_Q(m_t, m_m, m_e)
            te = m_t / m_e
            theta = koide_angle(te, m_m/m_e, 1.0)
            d_theta = theta - 2/9
            print(f"  {eta:8.2f} {om0h[0]:8.4f} {om0h[1]:8.4f} {om2h[0]:10.6f} "
                  f"{Q:8.5f} {theta:8.5f} {d_theta:+8.5f} {te:8.1f}")
    else:
        print(f"  {eta:8.2f}  --- l=2 mode lost or insufficient modes ---")

# Step 3: Try adding ALL mode backreactions
print(f"\n--- Step 3: All modes' backreaction ---")

# Include tau + muon backreaction
u_mu = ev0[:, 1]
norm_mu = trapezoid(u_mu**2, r0)
psi_mu = u_mu / np.maximum(r0, 1e-10) / np.sqrt(norm_mu)

# l=2 electron mode
Phi_on_r2 = f_Phi(r2)
d2F_r2 = d2nl_func(Phi_on_r2)
u_e = ev2[:, 0]
norm_e = trapezoid(u_e**2, r2)
psi_e = u_e / np.maximum(r2, 1e-10) / np.sqrt(norm_e)

print(f"\n  Individual mode contributions to δV at r=0:")
print(f"  tau:  max|δV| = {np.max(np.abs(d2F * psi_tau**2)):.6f}")
print(f"  muon: max|δV| = {np.max(np.abs(d2F * psi_mu**2)):.6f}")
print(f"  elec: max|δV| = {np.max(np.abs(d2F_r2 * psi_e**2)):.6f}")

# Combined: tau + muon (both l=0)
for eta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    dV_combined = -0.5 * d2F * (psi_tau**2 + psi_mu**2) * eta
    f_dV = interp1d(r0, dV_combined, fill_value=0, bounds_error=False)
    delta_V_bg = f_dV(r_bg)

    r0h, om0h, _, _ = cavity_eigs(r_bg, Phi_bg, 0, N=3000, delta_V=delta_V_bg)
    r2h, om2h, _, _ = cavity_eigs(r_bg, Phi_bg, 2, N=3000, delta_V=delta_V_bg)

    if len(om0h) >= 2 and len(om2h) >= 1 and om2h[0] < 0.9999:
        m_t = E_1D(om0h[0])
        m_m = E_1D(om0h[1])
        m_e = E_1D(om2h[0])
        if m_e > 0:
            Q = koide_Q(m_t, m_m, m_e)
            te = m_t / m_e
            theta = koide_angle(te, m_m/m_e, 1.0)
            d_theta = theta - 2/9
            print(f"  η={eta:5.1f} (tau+mu): ω_e={om2h[0]:.6f}, Q={Q:.5f}, "
                  f"θ₀={theta:.5f} (Δ={d_theta:+.5f}), τ/e={te:.1f}")

# Step 4: What η gives θ₀ = 2/9?
print(f"\n--- Step 4: Find η that gives θ₀ = 2/9 ---")
from scipy.optimize import brentq

def theta_vs_eta(eta):
    dV = -0.5 * d2F * psi_tau**2 * eta
    f_dV = interp1d(r0, dV, fill_value=0, bounds_error=False)
    dV_bg = f_dV(r_bg)

    _, om0h, _, _ = cavity_eigs(r_bg, Phi_bg, 0, N=3000, delta_V=dV_bg)
    _, om2h, _, _ = cavity_eigs(r_bg, Phi_bg, 2, N=3000, delta_V=dV_bg)

    if len(om0h) >= 2 and len(om2h) >= 1 and om2h[0] < 0.9999:
        m_t = E_1D(om0h[0])
        m_m = E_1D(om0h[1])
        m_e = E_1D(om2h[0])
        if m_e > 0:
            te = m_t / m_e
            theta = koide_angle(te, m_m/m_e, 1.0)
            return theta - 2/9
    return None

# Evaluate at several points to find bracket
results_eta = []
for eta in np.arange(0, 15, 0.5):
    val = theta_vs_eta(eta)
    if val is not None:
        results_eta.append((eta, val))
        if len(results_eta) >= 2 and results_eta[-2][1] * results_eta[-1][1] < 0:
            print(f"  Sign change between η={results_eta[-2][0]:.1f} and η={results_eta[-1][0]:.1f}")
            try:
                eta_star = brentq(lambda e: theta_vs_eta(e),
                                  results_eta[-2][0], results_eta[-1][0],
                                  xtol=0.01)
                print(f"  *** η* = {eta_star:.3f} gives θ₀ = 2/9! ***")

                # Compute full spectrum at η*
                dV = -0.5 * d2F * psi_tau**2 * eta_star
                f_dV2 = interp1d(r0, dV, fill_value=0, bounds_error=False)
                dV_bg2 = f_dV2(r_bg)

                _, om0f, _, _ = cavity_eigs(r_bg, Phi_bg, 0, N=3000, delta_V=dV_bg2)
                _, om2f, _, _ = cavity_eigs(r_bg, Phi_bg, 2, N=3000, delta_V=dV_bg2)

                m_t = E_1D(om0f[0])
                m_m = E_1D(om0f[1])
                m_e = E_1D(om2f[0])
                Q = koide_Q(m_t, m_m, m_e)
                te = m_t / m_e
                me_r = m_m / m_e
                theta = koide_angle(te, me_r, 1.0)

                print(f"\n  At η* = {eta_star:.3f}:")
                print(f"  ω_tau  = {om0f[0]:.6f}")
                print(f"  ω_mu   = {om0f[1]:.6f}")
                print(f"  ω_e    = {om2f[0]:.6f}")
                print(f"  Q      = {Q:.6f} (target: {2/3:.6f})")
                print(f"  θ₀     = {theta:.6f} (target: {2/9:.6f})")
                print(f"  τ/e    = {te:.1f} (expt: {expt_tau_e:.1f})")
                print(f"  μ/e    = {me_r:.1f} (expt: 206.8)")
                print(f"  τ/e error = {abs(te-expt_tau_e)/expt_tau_e*100:.2f}%")
            except:
                print(f"  (brentq failed)")
            break

if not any(r[1] < 0 for r in results_eta if r[1] is not None):
    print(f"  No sign change found. θ₀ never reaches 2/9 in this range.")
    print(f"  Trend: θ₀ - 2/9 = ", end="")
    for eta, val in results_eta[::max(1,len(results_eta)//5)]:
        print(f"{val:+.4f}(η={eta:.0f}) ", end="")
    print()
