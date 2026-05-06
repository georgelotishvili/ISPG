"""
Self-Consistent Hartree-Fock test for the ISPG cavity spectrum.

Tests / explores three candidate features:
1. Iterative HF with channel separation, to see whether Q=2/3 AND
   θ₀=2/9 close simultaneously beyond first order
2. Proper l=2 degeneracy via channel separation (no self-interaction)
3. An analytical η_eff from overlap integrals after convergence

Current outcome: the run improves θ₀ (toward 2/9) but does NOT
improve Q (Q goes from bare 0.6667 toward 0.6654, worse than
the first-order prescription); the lepton mass ratios in E1D
form remain significantly off (τ/e ≈ 2517 vs 3477, μ/e ≈ 147 vs
207). Therefore the channel-separated HF scheme does not jointly
close (Q, θ₀) and the lepton mass ratios; the joint closure
beyond first order remains an open numerical task. The
post-convergence η_eff produced here is a different
mathematical object from the ad-hoc overlap-weighted η_eff
in test_quantum_hartree_v2.py (first-order weight vs converged
self-consistent value).

Key physics: channel-separated backreaction (proper HF prescription)
  - l=0 channel sees vacuum fluctuations from l≠0 modes (electron l=2)
  - l=2 channel sees vacuum fluctuations from l≠2 modes (tau + muon l=0)
  - Self-interaction excluded by angular channel
  - QFT vacuum η = (2l+1)/(8πω) updated self-consistently each iteration
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

# --------------- Nonlinearity F(Φ) = Φ(1 - e^{-αΦ}) ---------------

def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))

def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)

def d2nl_func(Phi):
    return 2*ALPHA*np.exp(-ALPHA*Phi) - ALPHA**2*Phi*np.exp(-ALPHA*Phi)

# --------------- Background oscillon solver ---------------

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

# --------------- Cavity eigenvalue solver ---------------

def cavity_solve(r_bg, Phi_bg, l_val, N=3000, delta_V_func=None):
    """Solve cavity eigenvalue problem with optional δV correction.
    Returns (r_grid, omegas, eigenvectors, dr)."""
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
    evals, evecs = eigsh(H, k=min(8, N-2), which='SM')
    bound = evals < 1.0
    order = np.argsort(evals[bound])
    omegas = np.sqrt(np.maximum(evals[bound][order], 0))
    vecs = evecs[:, np.where(bound)[0][order]]
    return r, omegas, vecs, dr

# --------------- Physics functions ---------------

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

# --------------- Hartree-Fock δV computation ---------------

def compute_delta_V(u_vec, r_grid, norm, l_val, omega, r_ref, d2F_ref):
    """Compute δV contribution from one mode's vacuum zero-point on r_ref grid.
    η = (2l+1)/(8πω) is the QFT vacuum zero-point with angular degeneracy."""
    u_interp = interp1d(r_grid, u_vec, fill_value=0, bounds_error=False)
    u_on_ref = u_interp(r_ref)
    psi_sq = u_on_ref**2 / (np.maximum(r_ref, 1e-10)**2 * norm)
    eta = (2 * l_val + 1) / (8 * PI * omega)
    return -0.5 * d2F_ref * psi_sq * eta, eta

# ===============================================================
#                          MAIN
# ===============================================================

print("=" * 80)
print("  SELF-CONSISTENT HARTREE-FOCK: Channel-Separated Backreaction")
print("  l=0 ← δV from l=2 (electron vacuum)")
print("  l=2 ← δV from l=0 (tau + muon vacuum)")
print("=" * 80)

# --- Build background oscillon ---
print("\n[1] Building background oscillon (Φ₀ = 2.35)...")
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
print(f"    Ω_bg = {Om_bg:.6f}")

# --- Common reference grid ---
N_ref = 4000
r_ref = np.linspace(0.01, r_bg[-1]*0.85, N_ref)
Phi_ref = f_Phi(r_ref)
d2F_ref = d2nl_func(Phi_ref)

# --- Bare cavity modes (iteration 0) ---
print("\n[2] Bare cavity spectrum (no Hartree)...")
r0, om0_bare, ev0_bare, _ = cavity_solve(r_bg, Phi_bg, 0)
r2, om2_bare, ev2_bare, _ = cavity_solve(r_bg, Phi_bg, 2)

print(f"    ω_τ  = {om0_bare[0]:.6f}  (l=0, n=0)")
print(f"    ω_μ  = {om0_bare[1]:.6f}  (l=0, n=1)")
if len(om2_bare) > 0:
    print(f"    ω_e  = {om2_bare[0]:.6f}  (l=2, n=0)")

m_t = E_1D(om0_bare[0])
m_m = E_1D(om0_bare[1])
m_e = E_1D(om2_bare[0])
Q_bare = koide_Q(m_t, m_m, m_e)
theta_bare = koide_angle(m_t/m_e, m_m/m_e, 1.0)
print(f"    Q    = {Q_bare:.7f}  (target: {2/3:.7f})")
print(f"    θ₀   = {theta_bare:.6f}  (target: {2/9:.6f})")
print(f"    |ΔQ| = {abs(Q_bare - 2/3):.2e}")
print(f"    |Δθ| = {abs(theta_bare - 2/9):.2e}")

# ===============================================================
# [3] SELF-CONSISTENT HF ITERATION
# ===============================================================
print("\n" + "=" * 80)
print("  [3] SELF-CONSISTENT HARTREE-FOCK ITERATION")
print("=" * 80)

om_tau = om0_bare[0]
om_mu  = om0_bare[1]
om_e   = om2_bare[0]

u_tau = ev0_bare[:, 0]; norm_tau = trapezoid(u_tau**2, r0)
u_mu  = ev0_bare[:, 1]; norm_mu  = trapezoid(u_mu**2, r0)
u_e   = ev2_bare[:, 0]; norm_e   = trapezoid(u_e**2, r2)

MAX_ITER = 20
TOL = 1e-8

print(f"\n  {'Iter':>4} {'ω_τ':>10} {'ω_μ':>10} {'ω_e':>12} "
      f"{'Q':>10} {'θ₀':>10} {'|ΔQ|':>10} {'|Δθ|':>10} {'Δω_e':>10}")
print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*12} "
      f"{'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

# Print bare (iter 0)
print(f"  {0:4d} {om_tau:10.6f} {om_mu:10.6f} {om_e:12.8f} "
      f"{Q_bare:10.7f} {theta_bare:10.7f} {abs(Q_bare-2/3):10.2e} "
      f"{abs(theta_bare-2/9):10.2e} {'---':>10}")

history = [(om_tau, om_mu, om_e, Q_bare, theta_bare)]

for iteration in range(1, MAX_ITER + 1):
    # Compute δV from each mode on the reference grid
    dV_tau, eta_tau = compute_delta_V(u_tau, r0, norm_tau, 0, om_tau, r_ref, d2F_ref)
    dV_mu,  eta_mu  = compute_delta_V(u_mu,  r0, norm_mu,  0, om_mu,  r_ref, d2F_ref)
    dV_e,   eta_e   = compute_delta_V(u_e,   r2, norm_e,   2, om_e,   r_ref, d2F_ref)

    # Channel-separated HF potentials:
    # l=0 sees only l≠0 backreaction (electron)
    # l=2 sees only l≠2 backreaction (tau + muon)
    dV_for_l0 = dV_e
    dV_for_l2 = dV_tau + dV_mu

    dV_l0_func = interp1d(r_ref, dV_for_l0, fill_value=0, bounds_error=False)
    dV_l2_func = interp1d(r_ref, dV_for_l2, fill_value=0, bounds_error=False)

    # Solve eigenvalue problems with channel-separated potentials
    r0_new, om0_new, ev0_new, _ = cavity_solve(r_bg, Phi_bg, 0, delta_V_func=dV_l0_func)
    r2_new, om2_new, ev2_new, _ = cavity_solve(r_bg, Phi_bg, 2, delta_V_func=dV_l2_func)

    if len(om0_new) < 2 or len(om2_new) < 1:
        print(f"  {iteration:4d}  *** Mode lost! Stopping. ***")
        break

    om_tau_new = om0_new[0]
    om_mu_new  = om0_new[1]
    om_e_new   = om2_new[0]

    # Convergence check
    d_om_e = abs(om_e_new - om_e)
    d_om_max = max(abs(om_tau_new - om_tau), abs(om_mu_new - om_mu), d_om_e)

    # Update modes
    om_tau, om_mu, om_e = om_tau_new, om_mu_new, om_e_new
    r0, u_tau = r0_new, ev0_new[:, 0]; norm_tau = trapezoid(u_tau**2, r0)
    u_mu = ev0_new[:, 1]; norm_mu = trapezoid(u_mu**2, r0)
    r2, u_e = r2_new, ev2_new[:, 0]; norm_e = trapezoid(u_e**2, r2)

    # Compute observables
    m_t = E_1D(om_tau)
    m_m = E_1D(om_mu)
    m_e_val = E_1D(om_e)
    if m_e_val <= 0:
        print(f"  {iteration:4d}  *** Electron unbound! ***")
        break

    Q = koide_Q(m_t, m_m, m_e_val)
    theta = koide_angle(m_t/m_e_val, m_m/m_e_val, 1.0)
    history.append((om_tau, om_mu, om_e, Q, theta))

    print(f"  {iteration:4d} {om_tau:10.6f} {om_mu:10.6f} {om_e:12.8f} "
          f"{Q:10.7f} {theta:10.7f} {abs(Q-2/3):10.2e} "
          f"{abs(theta-2/9):10.2e} {d_om_e:10.2e}")

    if d_om_max < TOL:
        print(f"\n  *** Converged at iteration {iteration}! ***")
        break

# ===============================================================
# [4] FINAL RESULTS
# ===============================================================
print("\n" + "=" * 80)
print("  [4] CONVERGED HARTREE-FOCK RESULTS")
print("=" * 80)

om_tau_f, om_mu_f, om_e_f, Q_f, theta_f = history[-1]
m_t = E_1D(om_tau_f)
m_m = E_1D(om_mu_f)
m_e_val = E_1D(om_e_f)
te = m_t / m_e_val
me_r = m_m / m_e_val

expt_tau_e = 1776.86 / 0.511
expt_mu_e = 105.658 / 0.511

print(f"\n  Converged ω values:")
print(f"    ω_τ = {om_tau_f:.8f}")
print(f"    ω_μ = {om_mu_f:.8f}")
print(f"    ω_e = {om_e_f:.8f}")

print(f"\n  Koide parameters:")
print(f"    Q    = {Q_f:.8f}  (target: {2/3:.8f})")
print(f"    θ₀   = {theta_f:.8f}  (target: {2/9:.8f})")
print(f"    |ΔQ| = {abs(Q_f - 2/3):.2e}  ({abs(Q_f - 2/3)/(2/3)*100:.4f}%)")
print(f"    |Δθ| = {abs(theta_f - 2/9):.2e}  ({abs(theta_f - 2/9)/(2/9)*100:.4f}%)")

print(f"\n  Mass ratios (E_1D):")
print(f"    τ/e  = {te:.1f}  (expt: {expt_tau_e:.1f}, error: {abs(te-expt_tau_e)/expt_tau_e*100:.2f}%)")
print(f"    μ/e  = {me_r:.1f}  (expt: {expt_mu_e:.1f}, error: {abs(me_r-expt_mu_e)/expt_mu_e*100:.2f}%)")
print(f"    τ/μ  = {m_t/m_m:.2f}  (expt: {1776.86/105.658:.2f})")

te_corr = te * 4/PI
me_corr = me_r * 4/PI
print(f"\n  Mass ratios with 4/π correction:")
print(f"    τ/e  = {te_corr:.1f}  (expt: {expt_tau_e:.1f}, error: {abs(te_corr-expt_tau_e)/expt_tau_e*100:.2f}%)")
print(f"    μ/e  = {me_corr:.1f}  (expt: {expt_mu_e:.1f}, error: {abs(me_corr-expt_mu_e)/expt_mu_e*100:.2f}%)")

# ===============================================================
# [5] COMPARISON: HF vs non-HF approaches
# ===============================================================
print("\n" + "=" * 80)
print("  [5] COMPARISON: Channel-Separated HF vs Old Approaches")
print("=" * 80)

# Re-compute bare for comparison
r0b, om0b, ev0b, _ = cavity_solve(r_bg, Phi_bg, 0)
r2b, om2b, ev2b, _ = cavity_solve(r_bg, Phi_bg, 2)

approaches = {}

# A: Bare (no Hartree)
m_t_b = E_1D(om0b[0]); m_m_b = E_1D(om0b[1]); m_e_b = E_1D(om2b[0])
Q_b = koide_Q(m_t_b, m_m_b, m_e_b)
th_b = koide_angle(m_t_b/m_e_b, m_m_b/m_e_b, 1.0)
approaches['A: Bare (η=0)'] = (Q_b, th_b, m_t_b/m_e_b, m_m_b/m_e_b)

# B: Non-HF with empirical η=0.074 (old approach, same δV both channels)
u_tau_b = ev0b[:, 0]; norm_tau_b = trapezoid(u_tau_b**2, r0b)
dV_tau_b, _ = compute_delta_V(u_tau_b, r0b, norm_tau_b, 0, om0b[0], r_ref, d2F_ref)
dV_emp = dV_tau_b * (0.074 / (1/(8*PI*om0b[0])))  # scale to η=0.074
dV_emp_func = interp1d(r_ref, dV_emp, fill_value=0, bounds_error=False)
_, om0_emp, _, _ = cavity_solve(r_bg, Phi_bg, 0, delta_V_func=dV_emp_func)
_, om2_emp, _, _ = cavity_solve(r_bg, Phi_bg, 2, delta_V_func=dV_emp_func)
if len(om0_emp) >= 2 and len(om2_emp) >= 1:
    m_t_e = E_1D(om0_emp[0]); m_m_e = E_1D(om0_emp[1]); m_e_e = E_1D(om2_emp[0])
    if m_e_e > 0:
        Q_e = koide_Q(m_t_e, m_m_e, m_e_e)
        th_e = koide_angle(m_t_e/m_e_e, m_m_e/m_e_e, 1.0)
        approaches['B: η=0.074 (old, same δV)'] = (Q_e, th_e, m_t_e/m_e_e, m_m_e/m_e_e)

# C: Channel-separated HF (converged)
approaches['C: HF converged'] = (Q_f, theta_f, te, me_r)

print(f"\n  {'Approach':>30} {'Q':>10} {'θ₀':>10} {'|ΔQ|':>10} {'|Δθ|':>10} {'τ/e':>8} {'μ/e':>8}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
for name, (Q, th, te_v, me_v) in approaches.items():
    print(f"  {name:>30} {Q:10.7f} {th:10.7f} {abs(Q-2/3):10.2e} "
          f"{abs(th-2/9):10.2e} {te_v:8.1f} {me_v:8.1f}")
print(f"  {'Target':>30} {2/3:10.7f} {2/9:10.7f} {'0':>10} {'0':>10} "
      f"{expt_tau_e:8.1f} {expt_mu_e:8.1f}")

# ===============================================================
# [6] ANALYTICAL η AND OVERLAP INTEGRALS
# ===============================================================
print("\n" + "=" * 80)
print("  [6] ANALYTICAL η FROM QFT VACUUM + OVERLAP INTEGRALS")
print("=" * 80)

# Use converged HF eigenfunctions
dV_tau_final, eta_tau_f = compute_delta_V(u_tau, r0, norm_tau, 0, om_tau_f, r_ref, d2F_ref)
dV_mu_final,  eta_mu_f  = compute_delta_V(u_mu,  r0, norm_mu,  0, om_mu_f,  r_ref, d2F_ref)
dV_e_final,   eta_e_f   = compute_delta_V(u_e,   r2, norm_e,   2, om_e_f,   r_ref, d2F_ref)

print(f"\n  QFT vacuum η values (converged):")
print(f"    η_τ = 1/(8πω_τ)   = {eta_tau_f:.6f}  (ω_τ = {om_tau_f:.6f})")
print(f"    η_μ = 1/(8πω_μ)   = {eta_mu_f:.6f}  (ω_μ = {om_mu_f:.6f})")
print(f"    η_e = 5/(8πω_e)   = {eta_e_f:.6f}  (ω_e = {om_e_f:.6f})")

# Overlap integrals: how much each mode's δV overlaps with the electron
u_e_on_ref = interp1d(r2, u_e, fill_value=0, bounds_error=False)(r_ref)
psi_e_sq_ref = u_e_on_ref**2 / (np.maximum(r_ref, 1e-10)**2 * norm_e)

# The eigenvalue shift from each mode:
# δω²_e ≈ ∫ δV_n(r) × |u_e(r)|² dr / ∫ |u_e(r)|² dr
u_e_sq_ref = u_e_on_ref**2
norm_ue_ref = trapezoid(u_e_sq_ref, r_ref)

shift_from_tau = trapezoid(dV_tau_final * u_e_sq_ref, r_ref) / norm_ue_ref
shift_from_mu  = trapezoid(dV_mu_final  * u_e_sq_ref, r_ref) / norm_ue_ref
shift_total = shift_from_tau + shift_from_mu

print(f"\n  Eigenvalue shifts on electron (δω²_e from perturbation theory):")
print(f"    From τ:     δω²_e(τ) = {shift_from_tau:.6e}")
print(f"    From μ:     δω²_e(μ) = {shift_from_mu:.6e}")
print(f"    Total:      δω²_e    = {shift_total:.6e}")
print(f"    Ratio μ/τ:  {shift_from_mu/shift_from_tau:.4f}")

# Cross-check: actual ω shift from HF
om_e_bare_sq = om2_bare[0]**2
om_e_hf_sq = om_e_f**2
actual_shift = om_e_hf_sq - om_e_bare_sq
print(f"\n  Cross-check:")
print(f"    Perturbation theory:  δω² = {shift_total:.6e}")
print(f"    Actual HF:            δω² = {actual_shift:.6e}")
print(f"    Agreement: {abs(shift_total - actual_shift)/abs(actual_shift)*100:.2f}% difference")

# Overlap integrals (spatial overlap of mode densities)
u_tau_on_ref = interp1d(r0, u_tau, fill_value=0, bounds_error=False)(r_ref)
u_mu_on_ref  = interp1d(r0, u_mu,  fill_value=0, bounds_error=False)(r_ref)

psi_tau_sq = u_tau_on_ref**2 / (np.maximum(r_ref, 1e-10)**2 * norm_tau)
psi_mu_sq  = u_mu_on_ref**2  / (np.maximum(r_ref, 1e-10)**2 * norm_mu)

# Raw overlaps: ∫ d²F × |ψ_n|² × |u_e|² dr
I_tau_e = trapezoid(d2F_ref * psi_tau_sq * u_e_sq_ref, r_ref)
I_mu_e  = trapezoid(d2F_ref * psi_mu_sq  * u_e_sq_ref, r_ref)

print(f"\n  Spatial overlap integrals:")
print(f"    I(τ,e) = ∫ d²F × |ψ_τ|² × |u_e|² dr = {I_tau_e:.6e}")
print(f"    I(μ,e) = ∫ d²F × |ψ_μ|² × |u_e|² dr = {I_mu_e:.6e}")
print(f"    Overlap ratio I(μ,e)/I(τ,e) = {I_mu_e/I_tau_e:.4f}")

# Effective η: the single η that reproduces the total shift
# δω² = -½ × η_eff × I_ref / norm_ue
# where I_ref = ∫ d²F × |ψ_τ|² × |u_e|² dr (using tau as reference shape)
eta_eff_from_tau = eta_tau_f * (1 + (eta_mu_f / eta_tau_f) * (I_mu_e / I_tau_e))
print(f"\n  Effective η (using tau as reference):")
print(f"    η_eff = η_τ × (1 + (η_μ/η_τ) × I(μ)/I(τ))")
print(f"          = {eta_tau_f:.4f} × (1 + {eta_mu_f/eta_tau_f:.4f} × {I_mu_e/I_tau_e:.4f})")
print(f"          = {eta_eff_from_tau:.6f}")
print(f"    η*(empirical from old approach) = 0.074")

# Contribution of l=2 → l=0 backreaction
u_tau_on_ref2 = interp1d(r0, u_tau, fill_value=0, bounds_error=False)(r_ref)
u_tau_sq_ref = u_tau_on_ref2**2
norm_ut_ref = trapezoid(u_tau_sq_ref, r_ref)

shift_l0_from_e = trapezoid(dV_e_final * u_tau_sq_ref, r_ref) / norm_ut_ref

u_mu_on_ref2 = interp1d(r0, u_mu, fill_value=0, bounds_error=False)(r_ref)
u_mu_sq_ref = u_mu_on_ref2**2
norm_um_ref = trapezoid(u_mu_sq_ref, r_ref)

shift_l0_mu_from_e = trapezoid(dV_e_final * u_mu_sq_ref, r_ref) / norm_um_ref

print(f"\n  Backreaction of electron (l=2) on l=0 channel:")
print(f"    δω²_τ from e: {shift_l0_from_e:.6e}")
print(f"    δω²_μ from e: {shift_l0_mu_from_e:.6e}")
print(f"    (These shifts change τ and μ masses, affecting Q)")

# ===============================================================
# [7] SUMMARY TABLE
# ===============================================================
print("\n" + "=" * 80)
print("  [7] SUMMARY: Before and After Self-Consistent HF")
print("=" * 80)

om_t0, om_m0, om_e0 = om0_bare[0], om0_bare[1], om2_bare[0]
m_t0, m_m0, m_e0 = E_1D(om_t0), E_1D(om_m0), E_1D(om_e0)
Q0 = koide_Q(m_t0, m_m0, m_e0)
th0 = koide_angle(m_t0/m_e0, m_m0/m_e0, 1.0)

m_tf, m_mf, m_ef = E_1D(om_tau_f), E_1D(om_mu_f), E_1D(om_e_f)
Q_final = koide_Q(m_tf, m_mf, m_ef)
th_final = koide_angle(m_tf/m_ef, m_mf/m_ef, 1.0)

print(f"\n  {'Metric':>25} {'Bare':>14} {'HF converged':>14} {'Experiment':>14}")
print(f"  {'─'*25} {'─'*14} {'─'*14} {'─'*14}")
print(f"  {'Koide Q':>25} {Q0:14.7f} {Q_final:14.7f} {2/3:14.7f}")
print(f"  {'|Q − 2/3|':>25} {abs(Q0-2/3):14.2e} {abs(Q_final-2/3):14.2e} {'—':>14}")
print(f"  {'θ₀':>25} {th0:14.7f} {th_final:14.7f} {2/9:14.7f}")
print(f"  {'|θ₀ − 2/9|':>25} {abs(th0-2/9):14.2e} {abs(th_final-2/9):14.2e} {'—':>14}")
print(f"  {'ω_τ':>25} {om_t0:14.6f} {om_tau_f:14.6f} {'—':>14}")
print(f"  {'ω_μ':>25} {om_m0:14.6f} {om_mu_f:14.6f} {'—':>14}")
print(f"  {'ω_e':>25} {om_e0:14.8f} {om_e_f:14.8f} {'—':>14}")
print(f"  {'τ/e (E_1D)':>25} {m_t0/m_e0:14.1f} {m_tf/m_ef:14.1f} {expt_tau_e:14.1f}")
print(f"  {'μ/e (E_1D)':>25} {m_m0/m_e0:14.1f} {m_mf/m_ef:14.1f} {expt_mu_e:14.1f}")
print(f"  {'τ/e × 4/π':>25} {m_t0/m_e0*4/PI:14.1f} {m_tf/m_ef*4/PI:14.1f} {expt_tau_e:14.1f}")
print(f"  {'μ/e × 4/π':>25} {m_m0/m_e0*4/PI:14.1f} {m_mf/m_ef*4/PI:14.1f} {expt_mu_e:14.1f}")

# Key question: did HF improve BOTH Q and θ₀?
improved_Q = abs(Q_final - 2/3) < abs(Q0 - 2/3)
improved_theta = abs(th_final - 2/9) < abs(th0 - 2/9)
print(f"\n  Q improved:  {'YES' if improved_Q else 'NO'} "
      f"({abs(Q0-2/3):.2e} → {abs(Q_final-2/3):.2e})")
print(f"  θ₀ improved: {'YES' if improved_theta else 'NO'} "
      f"({abs(th0-2/9):.2e} → {abs(th_final-2/9):.2e})")

if improved_Q and improved_theta:
    print(f"\n  *** BOTH Q AND θ₀ IMPROVED! Self-consistent HF works! ***")
elif improved_theta and not improved_Q:
    print(f"\n  θ₀ improved but Q degraded slightly.")
    print(f"  This is the same pattern as first-order Hartree.")
    print(f"  The channel separation changes the balance.")

# ===============================================================
# [8] NON-HF COMPARISON (same δV both channels, for reference)
# ===============================================================
print("\n" + "=" * 80)
print("  [8] NON-HF REFERENCE: Same δV for both channels (old approach)")
print("=" * 80)

# Use converged HF eigenfunctions but apply SAME δV to both channels
dV_all = dV_tau_final + dV_mu_final + dV_e_final
dV_no_self_l2 = dV_tau_final + dV_mu_final  # no electron self-interaction

configs = [
    ("τ only (QFT η)", dV_tau_final, dV_tau_final),
    ("τ + μ (QFT η)", dV_tau_final + dV_mu_final, dV_tau_final + dV_mu_final),
    ("τ + μ + e (all, QFT η)", dV_all, dV_all),
    ("τ + μ → l=2, e → l=0 (HF)", dV_e_final, dV_no_self_l2),
]

print(f"\n  {'Config':>35} {'Q':>10} {'θ₀':>10} {'|ΔQ|':>10} {'|Δθ|':>10}")
print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for name, dV_l0, dV_l2 in configs:
    dV_l0_f = interp1d(r_ref, dV_l0, fill_value=0, bounds_error=False)
    dV_l2_f = interp1d(r_ref, dV_l2, fill_value=0, bounds_error=False)

    _, om0_c, _, _ = cavity_solve(r_bg, Phi_bg, 0, delta_V_func=dV_l0_f)
    _, om2_c, _, _ = cavity_solve(r_bg, Phi_bg, 2, delta_V_func=dV_l2_f)

    if len(om0_c) >= 2 and len(om2_c) >= 1 and om2_c[0] < 0.9999:
        mt = E_1D(om0_c[0]); mm = E_1D(om0_c[1]); me = E_1D(om2_c[0])
        if me > 0:
            Qc = koide_Q(mt, mm, me)
            thc = koide_angle(mt/me, mm/me, 1.0)
            print(f"  {name:>35} {Qc:10.7f} {thc:10.7f} "
                  f"{abs(Qc-2/3):10.2e} {abs(thc-2/9):10.2e}")
        else:
            print(f"  {name:>35}  --- electron unbound ---")
    else:
        print(f"  {name:>35}  --- mode lost ---")

print(f"  {'Target':>35} {2/3:10.7f} {2/9:10.7f} {'0':>10} {'0':>10}")

print("\n" + "=" * 80)
print("  DONE")
print("=" * 80)
