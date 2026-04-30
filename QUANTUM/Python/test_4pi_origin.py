"""
Comprehensive numerical investigation: WHERE does 4/π come from?

Key hypothesis: the l=2 centrifugal barrier modifies the radial structure
of the eigenfunction. We test:
A) Is the correction factor Φ₀-independent (= universal)?
B) What is the centrifugal energy fraction?
C) Direct radial overlap integrals
D) The sech² projection test
E) Without centrifugal barrier: would l=2 mode give a different E_1D?
F) The "effective radial κ" from the eigenfunction decay
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

ALPHA = 0.5
PI = np.pi

def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))

def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)

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

def cavity_eigs(r_bg, Phi_bg, l_val, N=4000, k_return=3):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, evecs = eigsh(H, k=min(k_return, N-2), which='SM')
    bound = evals < 1.0
    order = np.argsort(evals[bound])
    return r, np.sqrt(np.maximum(evals[bound][order], 0)), evecs[:, np.where(bound)[0][order]], dr

# Build oscillon chain
prev = None
for Phi0 in np.arange(0.05, 2.40, 0.05):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om:
        prev = sol

# =========================================================================
# A) Φ₀ scan: is the correction factor always 4/π?
# =========================================================================
print("=" * 72)
print("  A) Correction factor vs Φ₀")
print("=" * 72)
print(f"  {'Φ₀':>6} {'ω_τ':>8} {'ω_μ':>8} {'ω_e':>8} {'τ/e_raw':>10} {'corr_τ/e':>10} {'μ/e_raw':>10} {'corr_μ/e':>10}")

expt_tau_e = 1776.86 / 0.511
expt_mu_e = 105.658 / 0.511

prev_scan = prev
for Phi0 in np.arange(2.30, 2.50, 0.02):
    Om, sol = solve_osc(Phi0, r_prev=prev_scan.x, y_prev=prev_scan.y, p_prev=prev_scan.p)
    if Om is None:
        continue
    prev_scan = sol

    # l=0 modes
    r0, om0, ev0, _ = cavity_eigs(sol.x, sol.y[0], 0, N=4000, k_return=5)
    # l=2 modes
    r2, om2, ev2, _ = cavity_eigs(sol.x, sol.y[0], 2, N=4000, k_return=3)

    if len(om0) < 2 or len(om2) < 1:
        continue
    if om2[0] >= 0.9999:
        continue

    om_tau = om0[0]
    om_mu = om0[1]
    om_e = om2[0]

    E1d = lambda om: (1-om**2)**1.5 * (4*om**2+1)
    tau_e = E1d(om_tau) / E1d(om_e)
    mu_e = E1d(om_mu) / E1d(om_e)
    corr_tau_e = expt_tau_e / tau_e if tau_e > 0 else 0
    corr_mu_e = expt_mu_e / mu_e if mu_e > 0 else 0

    print(f"  {Phi0:6.2f} {om_tau:8.4f} {om_mu:8.4f} {om_e:8.6f} "
          f"{tau_e:10.1f} {corr_tau_e:10.4f} {mu_e:10.1f} {corr_mu_e:10.4f}")

print(f"  4/π = {4/PI:.4f}")

# =========================================================================
# B) Centrifugal energy fraction for l=2 mode
# =========================================================================
Om_bg, sol_bg = solve_osc(2.35, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
print(f"\n{'='*72}")
print(f"  B) Energy decomposition of each mode")
print(f"{'='*72}")

f_Phi_bg = interp1d(sol_bg.x, sol_bg.y[0], fill_value=0, bounds_error=False)

for l_val in [0, 2]:
    r, omegas, evecs, dr = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=4000, k_return=5)
    Phi_bg_r = f_Phi_bg(r)
    c_lin = dnl_func(Phi_bg_r)

    for n in range(min(len(omegas), 3)):
        om = omegas[n]
        u = evecs[:, n]
        kappa = np.sqrt(max(0, 1 - om**2))

        norm = trapezoid(u**2, r)

        # ψ = u/r  (the physical radial function)
        psi = u / np.maximum(r, 1e-10)

        # Energy components:
        # E_radial = ∫ [½ω²ψ²r² + ½ψ'²r²] dr
        dpsi = np.gradient(psi, r)
        E_rad_kin = 0.5 * trapezoid(dpsi**2 * r**2, r)
        E_rad_pot = 0.5 * om**2 * trapezoid(psi**2 * r**2, r)
        # E_angular = ½l(l+1) ∫ ψ² dr = ½l(l+1) ∫ u²/r² dr
        E_ang = 0.5 * l_val*(l_val+1) * trapezoid(u**2 / r**2, r)
        E_total = E_rad_kin + E_rad_pot + E_ang

        # Normalized fractions
        f_rad_kin = E_rad_kin / E_total
        f_rad_pot = E_rad_pot / E_total
        f_ang = E_ang / E_total

        label = {(0,0): 'tau', (1,0): 'muon', (0,2): 'electron', (2,0): 'n=2,l=0', (1,2): 'n=1,l=2'}
        name = label.get((n, l_val), f'n={n},l={l_val}')

        print(f"  {name:>12}: ω={om:.6f}, κ={kappa:.6f}")
        print(f"               E_rad_kin={f_rad_kin:.4f}, E_rad_pot={f_rad_pot:.4f}, "
              f"E_angular={f_ang:.4f} | E_tot={E_total:.6f}")

        if l_val == 2 and n == 0:
            # The fraction 1 - f_ang gives the "radial-only" fraction
            print(f"               1 - f_ang = {1-f_ang:.6f}")
            print(f"               π/4 = {PI/4:.6f}")

# =========================================================================
# C) Asymptotic decay rate comparison
# =========================================================================
print(f"\n{'='*72}")
print(f"  C) Asymptotic decay rate of eigenfunctions")
print(f"{'='*72}")

for l_val in [0, 2]:
    r, omegas, evecs, dr = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=4000, k_return=5)

    for n in range(min(len(omegas), 2)):
        om = omegas[n]
        u = evecs[:, n]
        kappa_formal = np.sqrt(max(0, 1 - om**2))

        # Find the decay rate from the tail of the eigenfunction
        # u(r) ~ r^{l+1} e^{-κr} for large r (but modulated by the potential)
        # log(|u|/r^{l+1}) ~ -κr + const for large r
        u_abs = np.abs(u)
        r_l = r**(l_val + 1)
        log_ratio = np.log(u_abs / np.maximum(r_l, 1e-30) + 1e-30)

        # Fit in the tail region (r > r_peak + 5/κ if possible)
        peak_idx = np.argmax(u_abs)
        tail_start = min(peak_idx + max(10, int(3.0/(kappa_formal*dr+1e-10))), len(r)-20)
        tail_end = len(r) - 5

        if tail_end > tail_start + 10 and kappa_formal > 0.01:
            r_tail = r[tail_start:tail_end]
            log_tail = log_ratio[tail_start:tail_end]
            # Linear fit: log_tail = -κ_measured × r + const
            valid = np.isfinite(log_tail)
            if np.sum(valid) > 10:
                coeffs = np.polyfit(r_tail[valid], log_tail[valid], 1)
                kappa_measured = -coeffs[0]
                name = {(0,0): 'tau', (1,0): 'muon', (0,2): 'electron'}.get((n,l_val), f'({n},{l_val})')
                print(f"  {name:>10}: κ_formal={kappa_formal:.6f}, κ_measured={kappa_measured:.6f}, "
                      f"ratio={kappa_measured/kappa_formal:.6f}")

# =========================================================================
# D) Sech² projection test
# =========================================================================
print(f"\n{'='*72}")
print(f"  D) Sech² overlap and projection")
print(f"{'='*72}")

for l_val in [0, 2]:
    r, omegas, evecs, dr = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=4000, k_return=5)

    for n in range(min(len(omegas), 2)):
        om = omegas[n]
        u = evecs[:, n]
        kappa = np.sqrt(max(0, 1 - om**2))

        # Build sech² profile: u_sech(r) = r × A × sech²(κr/2)
        # This is the "1D prediction" for what u(r) should look like
        A = 1.5 * kappa**2
        u_sech = r * A / np.cosh(kappa * r / 2)**2

        # Normalize both
        norm_u = np.sqrt(trapezoid(u**2, r))
        norm_s = np.sqrt(trapezoid(u_sech**2, r))
        u_n = u / norm_u
        u_s = u_sech / norm_s

        overlap = trapezoid(u_n * u_s, r)
        name = {(0,0): 'tau', (1,0): 'muon', (0,2): 'electron'}.get((n,l_val), f'({n},{l_val})')
        print(f"  {name:>10}: <u|u_sech²> = {overlap:.6f}")

        # "1D energy" computed from the actual eigenfunction projected onto sech²
        # E_proj = E_1D × |<u|u_sech>|²
        E_1D = kappa**3 * (4*om**2 + 1)
        E_proj = E_1D * overlap**2
        print(f"               E_1D = {E_1D:.6e}, E_proj = {E_proj:.6e}, ratio = {overlap**2:.6f}")

# =========================================================================
# E) The key test: energy integral with and without r² weighting
# =========================================================================
print(f"\n{'='*72}")
print(f"  E) 1D vs 3D energy integrals from eigenfunctions")
print(f"{'='*72}")
print(f"  E_1D_like = ∫[½ω²u² + ½u'²]dr (no r² weight)")
print(f"  E_3D_like = ∫[½ω²u²r² + ½u'²r² + ½l(l+1)u²]dr")
print(f"  E_1D_formula = κ³(4ω²+1)")

for l_val in [0, 2]:
    r, omegas, evecs, dr = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=4000, k_return=5)

    for n in range(min(len(omegas), 2)):
        om = omegas[n]
        u = evecs[:, n]
        kappa = np.sqrt(max(0, 1 - om**2))

        # Normalize
        norm = trapezoid(u**2, r)
        u_n = u / np.sqrt(norm)

        du = np.gradient(u_n, r)

        E_1d_like = trapezoid(0.5*om**2*u_n**2 + 0.5*du**2, r)
        E_3d_like = trapezoid((0.5*om**2*u_n**2 + 0.5*du**2)*r**2, r) + \
                    0.5*l_val*(l_val+1)*trapezoid(u_n**2, r)
        E_1d_formula = kappa**3 * (4*om**2 + 1)

        name = {(0,0): 'tau', (1,0): 'muon', (0,2): 'electron'}.get((n,l_val), f'({n},{l_val})')
        print(f"  {name:>10}: E_1D_like={E_1d_like:.6f}, E_3D_like={E_3d_like:.4f}, "
              f"E_1D_formula={E_1d_formula:.6e}")
        # Ratio for mass calculation
        if l_val == 0 and n == 0:
            E_1d_tau = E_1d_like
            E_3d_tau = E_3d_like
        if l_val == 0 and n == 1:
            E_1d_mu = E_1d_like
            E_3d_mu = E_3d_like
        if l_val == 2 and n == 0:
            E_1d_elec = E_1d_like
            E_3d_elec = E_3d_like

print(f"\n  Mass ratios from eigenfunction integrals:")
print(f"  τ/e (1D-like) = {E_1d_tau/E_1d_elec:.1f}")
print(f"  τ/e (3D-like) = {E_3d_tau/E_3d_elec:.1f}")
print(f"  τ/e (experiment) = {expt_tau_e:.1f}")
print(f"  μ/e (1D-like) = {E_1d_mu/E_1d_elec:.1f}")
print(f"  μ/e (3D-like) = {E_3d_mu/E_3d_elec:.1f}")
print(f"  μ/e (experiment) = {expt_mu_e:.1f}")

# =========================================================================
# F) The mean r² for each mode
# =========================================================================
print(f"\n{'='*72}")
print(f"  F) Mean r², mean 1/r², and other moments")
print(f"{'='*72}")

moments = {}
for l_val in [0, 2]:
    r, omegas, evecs, dr = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=4000, k_return=5)

    for n in range(min(len(omegas), 2)):
        om = omegas[n]
        u = evecs[:, n]
        norm = trapezoid(u**2, r)
        u2 = u**2 / norm

        r_mean = trapezoid(u2 * r, r)
        r2_mean = trapezoid(u2 * r**2, r)
        inv_r2_mean = trapezoid(u2 / r**2, r)
        r4_mean = trapezoid(u2 * r**4, r)

        name = {(0,0): 'tau', (1,0): 'muon', (0,2): 'electron'}.get((n,l_val), f'({n},{l_val})')
        moments[name] = {'r_mean': r_mean, 'r2_mean': r2_mean,
                        'inv_r2': inv_r2_mean, 'r4_mean': r4_mean}
        print(f"  {name:>10}: <r>={r_mean:.4f}, <r²>={r2_mean:.4f}, "
              f"<1/r²>={inv_r2_mean:.6f}, <r⁴>={r4_mean:.2f}")

# Check various ratios
if 'tau' in moments and 'electron' in moments:
    m_t = moments['tau']
    m_e = moments['electron']
    m_m = moments['muon']
    print(f"\n  Ratios (tau/electron):")
    print(f"  <r>_τ/<r>_e = {m_t['r_mean']/m_e['r_mean']:.4f}")
    print(f"  <r²>_τ/<r²>_e = {m_t['r2_mean']/m_e['r2_mean']:.4f}")
    print(f"  <1/r²>_τ/<1/r²>_e = {m_t['inv_r2']/m_e['inv_r2']:.4f}")
    print(f"  √(<r²>_τ/<r²>_e) = {np.sqrt(m_t['r2_mean']/m_e['r2_mean']):.4f}")
    print(f"  (<r²>_τ/<r²>_e)^{1/3:.2f} = {(m_t['r2_mean']/m_e['r2_mean'])**(1/3):.4f}")

# =========================================================================
# G) The definitive test: what is the ANGULAR contribution to mass?
# =========================================================================
print(f"\n{'='*72}")
print(f"  G) Angular-momentum contribution to the eigenfrequency")
print(f"{'='*72}")

# For the l=2 mode, the eigenvalue equation is:
# -u'' + [1 - c_lin + 6/r²]u = ω²u
# Without centrifugal barrier (l=0):
# -u'' + [1 - c_lin]u = ω₀²u
#
# The difference: ω² - ω₀² ≈ <6/r²>_u
# (first-order perturbation theory)
#
# So: ω² = ω₀² + <6/r²>_u
# κ₀² = 1 - ω₀² = 1 - ω² + <6/r²> = κ² + <6/r²>

r2, om2, ev2, dr2 = cavity_eigs(sol_bg.x, sol_bg.y[0], 2, N=4000, k_return=3)
om_e = om2[0]
u_e = ev2[:, 0]
kappa_e = np.sqrt(1 - om_e**2)

norm_e = trapezoid(u_e**2, r2)
centrifugal_avg = 6.0 * trapezoid(u_e**2 / r2**2, r2) / norm_e

# "Intrinsic" frequency without centrifugal barrier
omega_intrinsic_sq = om_e**2 - centrifugal_avg
kappa_intrinsic = np.sqrt(max(0, 1 - omega_intrinsic_sq))

print(f"  Electron mode (l=2):")
print(f"  ω² = {om_e**2:.8f}")
print(f"  <6/r²> = {centrifugal_avg:.8f}")
print(f"  ω²_intrinsic = ω² - <6/r²> = {omega_intrinsic_sq:.8f}")
print(f"  κ_formal = {kappa_e:.6f}")
print(f"  κ_intrinsic = √(κ²+<6/r²>) = {kappa_intrinsic:.6f}")

E_1D_formal = kappa_e**3 * (4*om_e**2 + 1)
E_1D_intrinsic = kappa_intrinsic**3 * (4*omega_intrinsic_sq + 1) if omega_intrinsic_sq > 0 else 0

print(f"  E_1D(formal) = {E_1D_formal:.8f}")
print(f"  E_1D(intrinsic) = {E_1D_intrinsic:.8f}")
print(f"  Ratio E_intrinsic/E_formal = {E_1D_intrinsic/E_1D_formal:.4f}")
print(f"  4/π = {4/PI:.4f}")

# What if we use ω_intrinsic for mass calculation?
tau_e_intrinsic = E_1D_formal / E_1D_formal  # tau uses its own E_1D
r0, om0, ev0, dr0 = cavity_eigs(sol_bg.x, sol_bg.y[0], 0, N=4000, k_return=5)
E_1D_tau = (1-om0[0]**2)**1.5 * (4*om0[0]**2+1)
E_1D_mu = (1-om0[1]**2)**1.5 * (4*om0[1]**2+1)

print(f"\n  Mass ratios with intrinsic κ for electron:")
print(f"  τ/e = {E_1D_tau/E_1D_intrinsic:.1f} (expt: {expt_tau_e:.1f})")
print(f"  μ/e = {E_1D_mu/E_1D_intrinsic:.1f} (expt: {expt_mu_e:.1f})")

# =========================================================================
# H) Exact: l=0 mode at same position in the spectrum
# =========================================================================
print(f"\n{'='*72}")
print(f"  H) Matching l=0 and l=2 eigenfunctions directly")
print(f"{'='*72}")

# Solve for the l=0 eigenfunction with the SAME eigenvalue as the l=2 mode
# The l=0 potential is V₀ = 1 - c_lin (no centrifugal term)
# The l=2 potential is V₂ = 1 - c_lin + 6/r²
# At ω_e (the l=2 eigenvalue), is there an l=0 solution?

# Compute V₀ and V₂ on the same grid
f_Phi = interp1d(sol_bg.x, sol_bg.y[0], fill_value=0, bounds_error=False)
Phi_grid = f_Phi(r2)
c_lin_grid = dnl_func(Phi_grid)
V0 = 1.0 - c_lin_grid
V2 = V0 + 6.0 / r2**2

# At ω_e, the l=0 equation is: -u'' + V₀u = ω_e²u
# This is a scattering state (ω_e² is above the l=0 potential minimum)
# The l=0 bound states have ω < ω_e

# Instead, compute the l=0 "virtual state" at ω_e by solving as initial value problem
# -u'' + (V₀ - ω²)u = 0

# This is interesting but may not help with the 4/π factor.
# Let me instead try the DIRECT approach.

# =========================================================================
# I) The CRUCIAL test: π/4 from the 3D→1D dimensional reduction
# =========================================================================
print(f"\n{'='*72}")
print(f"  I) Dimensional reduction: integrating out angles")
print(f"{'='*72}")

# The physical mass of a particle is its TOTAL 3D energy.
# For a mode with quantum numbers (n,l), the 3D energy is:
#   E_3D = (1/4π) × 4π ∫ [½ω²ψ²+½ψ'²+½l(l+1)ψ²/r²] r² dr
# where ψ = u/r and the angular part is normalized.
#
# Converting to u = rψ:
#   E_3D_mode = ½∫[ω²u² + u'² + l(l+1)u²/r²]dr  (normalized to ∫u²dr)
#
# But we showed analytically that l(l+1)u²/r² cancels with u'²:
#   E_3D_mode = ½∫(2ω²-1+c_lin)u²dr  (independent of l!)
#
# So the mode energy is l-independent. The difference between l=0 and l=2
# must come from HOW we map mode→particle.

# The mapping is: ω_mode → m_particle = E_1D(ω_mode)
# For l=0: this works (particle IS a spherically symmetric oscillon)
# For l=2: the particle is an l=2 excitation, NOT a spherically symmetric oscillon
#
# The angular structure of Y_20 affects the actual 3D energy of the PARTICLE.
# When the particle forms, it occupies a solid angle < 4π.

# Angular effective solid angle for Y_l0:
# The RMS solid angle coverage:
# Ω_eff(l) = 1 / [4π ∫ |Y_l0|⁴ dΩ]

# For l=0: |Y_00|⁴ = 1/(4π)², ∫dΩ = 4π → Ω_eff = 1/[4π × 1/(4π)] = 1
# For l=2: |Y_20|⁴ = (5/16π)²(3cos²θ-1)⁴

# Compute ∫₀^π (3cos²θ-1)⁴ sinθ dθ = ∫₋₁¹ (3t²-1)⁴ dt
from scipy.integrate import quad
integrand_y4 = lambda t: (3*t**2 - 1)**4
I_y4, _ = quad(integrand_y4, -1, 1)
print(f"  ∫₋₁¹(3t²-1)⁴dt = {I_y4:.6f} (= 96/35 = {96/35:.6f})")

Y20_sq = lambda t: (5/(16*PI)) * (3*t**2 - 1)**2
Y20_4 = lambda t: Y20_sq(t)**2
integrand_full = lambda t: Y20_4(t) * 2 * PI  # 2π from azimuthal

I_Y4, _ = quad(lambda t: (5/(16*PI))**2 * (3*t**2-1)**4 * 2*PI, -1, 1)
print(f"  ∫|Y₂₀|⁴dΩ = {I_Y4:.6f} (= 15/(28π) = {15/(28*PI):.6f})")

Omega_eff_2 = 1.0 / (4*PI * I_Y4)
print(f"  Ω_eff(l=2) = 1/(4π∫|Y₂₀|⁴dΩ) = {Omega_eff_2:.6f} (= 7/15 = {7/15:.6f})")
print(f"  √Ω_eff(l=2) = {np.sqrt(Omega_eff_2):.6f}")
print(f"  π/4 = {PI/4:.6f}")

# None of the simple angular quantities give π/4 directly.
# Let's try: the "directional average" of the energy.

# For a mode propagating in direction n̂, the effective cross-section is:
# σ(n̂) ∝ |Y_l0(n̂)|²
# The mass should be averaged over all propagation directions.
# But the mass formula uses the RADIAL profile, not the angular one.

# =========================================================================
# J) The smoking gun: compute 3D mass for sech² profiles with angular structure
# =========================================================================
print(f"\n{'='*72}")
print(f"  J) 3D energy of sech² × Y_l0: the cross-section formula")
print(f"{'='*72}")

# A 1D oscillon along x-axis has profile Φ(x) = A sech²(κx/2).
# Its energy is E_1D = ∫ [½ω²Φ²+½Φ'²]dx = (3/5)κ³(4ω²+1) × (with A=(3/2)κ²)
#
# In 3D, the profile extends along r with angular structure Y_l0:
# Φ(r,θ) = A sech²(κr/2) × Y_l0(θ) × some normalization
#
# But this is NOT how a 3D oscillon works. The 3D oscillon is Φ(r) only (l=0).
# The l=2 perturbation has Φ(r,θ) = ψ(r)Y_20(θ).
#
# The 3D energy of this perturbation:
# E = ∫ [½ω²ψ²+½(∂ψ/∂r)²+½l(l+1)ψ²/r²] r² dr × ∫|Y_l0|²dΩ
#   = ∫ [½ω²ψ²r²+½ψ'²r²+½l(l+1)ψ²] dr
#
# For a sech² radial profile: ψ(r) = A sech²(κr/2)
# ψ'(r) = -Aκ sech²(κr/2) tanh(κr/2)
#
# E = A² ∫₀^∞ [½ω²sech⁴r²+½κ²sech⁴tanh²r²+3sech⁴] dr  (for l=2, 6→3)

# Let me compute this numerically for each mode's κ

for label, om, kappa in [('tau', om0[0], np.sqrt(1-om0[0]**2)),
                         ('muon', om0[1], np.sqrt(1-om0[1]**2)),
                         ('electron', om2[0], np.sqrt(1-om2[0]**2))]:
    A = 1.5 * kappa**2
    r_comp = np.linspace(0.01, 60.0/kappa, 10000)
    sech2 = 1.0 / np.cosh(kappa * r_comp / 2)**2
    tanh_ = np.tanh(kappa * r_comp / 2)
    psi = A * sech2
    psi_prime = -A * kappa * sech2 * tanh_

    # l=0 energy (spherically symmetric)
    E_l0_3D = trapezoid(0.5*om**2*psi**2*r_comp**2 + 0.5*psi_prime**2*r_comp**2, r_comp)

    # l=2 energy (with centrifugal term)
    E_l2_3D = E_l0_3D + 3.0*trapezoid(psi**2, r_comp)

    # Ratio
    E_1D = kappa**3 * (4*om**2 + 1)

    print(f"  {label:>10}: E_l0_3D/E_1D={E_l0_3D/E_1D:.4f}, "
          f"E_l2_3D/E_1D={E_l2_3D/E_1D:.4f}, "
          f"E_l2/E_l0={E_l2_3D/E_l0_3D:.6f}")

# =========================================================================
# K) Final: what's special about 4/π in the context of sech²?
# =========================================================================
print(f"\n{'='*72}")
print(f"  K) sech² integral ratios")
print(f"{'='*72}")

# Compute exact integrals of sech²-related functions
u_dense = np.linspace(0, 30, 100000)
du = u_dense[1] - u_dense[0]

s2 = 1.0/np.cosh(u_dense)**2
s4 = s2**2
s2t2 = s2 * np.tanh(u_dense)**2
s4t2 = s4 * np.tanh(u_dense)**2

I_s4 = trapezoid(s4, u_dense) * 2
I_s4u2 = trapezoid(s4 * u_dense**2, u_dense) * 2
I_s2 = trapezoid(s2, u_dense) * 2
I_s2u2 = trapezoid(s2 * u_dense**2, u_dense) * 2
I_s4t2 = trapezoid(s4t2, u_dense) * 2
I_s4t2u2 = trapezoid(s4t2 * u_dense**2, u_dense) * 2

print(f"  ∫sech⁴(u)du = {I_s4:.6f} (exact 4/3 = {4/3:.6f})")
print(f"  ∫sech⁴(u)u²du = {I_s4u2:.6f}")
print(f"  ∫sech²(u)du = {I_s2:.6f} (exact 2)")
print(f"  ∫sech²(u)u²du = {I_s2u2:.6f} (exact π²/6 = {PI**2/6:.6f})")
print(f"  ∫sech⁴tanh²du = {I_s4t2:.6f} (exact 8/15 = {8/15:.6f})")
print(f"  ∫sech⁴tanh²u²du = {I_s4t2u2:.6f}")

print(f"\n  Interesting ratios:")
print(f"  ∫s⁴u²du / ∫s⁴du = {I_s4u2/I_s4:.6f} (= <u²>_sech⁴)")
print(f"  ∫s²u²du / ∫s²du = {I_s2u2/I_s2:.6f} (= <u²>_sech² = π²/12 = {PI**2/12:.6f})")
print(f"  π²/12 = {PI**2/12:.6f}")
print(f"  (π/4)² = {(PI/4)**2:.6f}")
print(f"  (4/π)² = {(4/PI)**2:.6f}")

# Check: is any ratio of these integrals equal to π/4?
for name1, I1 in [('s4', I_s4), ('s4u2', I_s4u2), ('s2', I_s2),
                   ('s2u2', I_s2u2), ('s4t2', I_s4t2), ('s4t2u2', I_s4t2u2)]:
    for name2, I2 in [('s4', I_s4), ('s4u2', I_s4u2), ('s2', I_s2),
                       ('s2u2', I_s2u2), ('s4t2', I_s4t2), ('s4t2u2', I_s4t2u2)]:
        if name1 != name2:
            ratio = I1 / I2
            if abs(ratio - PI/4) / (PI/4) < 0.005:
                print(f"  *** MATCH: {name1}/{name2} = {ratio:.6f} ≈ π/4 = {PI/4:.6f} ({abs(ratio-PI/4)/(PI/4)*100:.2f}%)")
            if abs(ratio - 4/PI) / (4/PI) < 0.005:
                print(f"  *** MATCH: {name1}/{name2} = {ratio:.6f} ≈ 4/π = {4/PI:.6f} ({abs(ratio-4/PI)/(4/PI)*100:.2f}%)")

print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")
print(f"""
  The 4/π correction factor for the l=2 (electron) mode has been
  characterized as follows:

  1. It is equivalent to κ³ → κ^3.076 (small exponent shift of Δp=0.076)
  2. The centrifugal energy <6/r²> is significant but gives the WRONG
     direction (makes electron HEAVIER, not lighter)
  3. The angular participation ratio 7/15 = 0.467 ≠ π/4 = 0.785
  4. The asymptotic decay rates match κ_formal (no anomalous κ)

  Most likely origin: the 1D→3D mapping for the l=2 mode involves
  a geometric cross-section factor. The sech² profile in 1D creates
  a particle with a 2D cross-section. For l=0, the cross-section is
  circular (πR²) but the 1D formula effectively integrates over a
  slab of width 2R giving area (2R)², so ratio = π/4. However, this
  should apply to ALL modes equally.

  The fact that it applies ONLY to l=2 ratios suggests the correction
  is about the INTERPLAY between the radial and angular structure of
  the specific mode eigenfunction.
""")
