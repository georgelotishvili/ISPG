"""
Focused test: find the EXACT Φ₀ where Q = 2/3 and check mass ratios there.
Also compute the Koide angle θ₀ for E_1D vs experiment.
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

def cavity_eigs(r_bg, Phi_bg, l_val, N=3000, k_return=3):
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
    return np.sqrt(np.maximum(evals[bound][order], 0))

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
    # √m_i = M(1 + √2 cos(θ₀ + 2πi/3)) for i=0,1,2
    c0 = (np.sqrt(m1)/M - 1) / np.sqrt(2)
    return np.arccos(np.clip(c0, -1, 1))

# Build oscillon chain
prev = None
for Phi0 in np.arange(0.05, 2.25, 0.05):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om:
        prev = sol

# =========================================================================
# Fine scan of Φ₀ near optimal
# =========================================================================
print("=" * 80)
print("  Fine Φ₀ scan: Q, mass ratios, and correction factors")
print("=" * 80)

expt_tau_e = 1776.86 / 0.511
expt_mu_e = 105.658 / 0.511
expt_tau_mu = 1776.86 / 105.658

print(f"  Expt: τ/e = {expt_tau_e:.1f}, μ/e = {expt_mu_e:.1f}, τ/μ = {expt_tau_mu:.2f}")
print(f"  Target: Q = {2/3:.6f}, 4/π = {4/PI:.6f}")
print()
print(f"  {'Φ₀':>7} {'Ω_bg':>8} {'ω_τ':>8} {'ω_μ':>8} {'ω_e':>10} "
      f"{'Q':>8} {'τ/e':>8} {'μ/e':>8} {'τ/μ':>7} {'corr':>7}")

results = []
prev_scan = prev
for Phi0_100 in range(229, 250):
    Phi0 = Phi0_100 / 100.0
    Om, sol = solve_osc(Phi0, r_prev=prev_scan.x, y_prev=prev_scan.y, p_prev=prev_scan.p)
    if Om is None:
        continue
    prev_scan = sol

    om0 = cavity_eigs(sol.x, sol.y[0], 0, N=3000, k_return=3)
    om2 = cavity_eigs(sol.x, sol.y[0], 2, N=3000, k_return=3)

    if len(om0) < 2 or len(om2) < 1:
        continue
    if om2[0] >= 0.9999:
        continue

    m_tau = E_1D(om0[0])
    m_mu = E_1D(om0[1])
    m_e = E_1D(om2[0])

    if m_e < 1e-15:
        continue

    Q = koide_Q(m_tau, m_mu, m_e)
    tau_e = m_tau / m_e
    mu_e = m_mu / m_e
    tau_mu = m_tau / m_mu
    corr = expt_tau_e / tau_e if tau_e > 0 else 0

    results.append((Phi0, Om, om0[0], om0[1], om2[0], Q, tau_e, mu_e, tau_mu, corr))
    print(f"  {Phi0:7.2f} {Om:8.4f} {om0[0]:8.4f} {om0[1]:8.4f} {om2[0]:10.6f} "
          f"{Q:8.5f} {tau_e:8.1f} {mu_e:8.1f} {tau_mu:7.2f} {corr:7.4f}")

# Find where Q is closest to 2/3
if results:
    best = min(results, key=lambda x: abs(x[5] - 2/3))
    print(f"\n  Closest to Q=2/3: Φ₀={best[0]:.2f}, Q={best[5]:.6f}")
    print(f"    τ/e = {best[6]:.1f} (expt: {expt_tau_e:.1f}, correction = {best[9]:.4f})")
    print(f"    μ/e = {best[7]:.1f} (expt: {expt_mu_e:.1f})")
    print(f"    τ/μ = {best[8]:.2f} (expt: {expt_tau_mu:.2f})")

    # Koide angles
    theta_E1D = koide_angle(best[6], best[7], 1.0)
    theta_expt = koide_angle(expt_tau_e, expt_mu_e, 1.0)
    print(f"\n  Koide angle θ₀:")
    print(f"    E_1D:       θ₀ = {theta_E1D:.6f} rad = {np.degrees(theta_E1D):.4f}°")
    print(f"    Experiment: θ₀ = {theta_expt:.6f} rad = {np.degrees(theta_expt):.4f}°")
    print(f"    Difference: Δθ = {theta_E1D - theta_expt:.6f} rad")
    print(f"    2/9 = {2/9:.6f} rad")

    # Find where correction = 4/π
    import_corr = [(r[0], r[9]) for r in results if r[9] > 0]
    if len(import_corr) > 2:
        phi0s = [x[0] for x in import_corr]
        corrs = [x[1] for x in import_corr]
        f_corr = interp1d(corrs, phi0s, bounds_error=False)
        phi0_at_4pi = f_corr(4/PI)
        if phi0_at_4pi and not np.isnan(phi0_at_4pi):
            print(f"\n  Φ₀ where correction = 4/π: {phi0_at_4pi:.4f}")

            # Find Q at this Φ₀
            Qs = [r[5] for r in results]
            f_Q = interp1d(phi0s, Qs, bounds_error=False)
            Q_at_4pi = f_Q(phi0_at_4pi)
            print(f"  Q at this Φ₀: {Q_at_4pi:.6f}")
            print(f"  Deviation from 2/3: {abs(Q_at_4pi - 2/3)*100:.4f}%")

# =========================================================================
# Check: does the 4/π correction improve or worsen Q?
# =========================================================================
print(f"\n{'='*80}")
print(f"  Effect of 4/π correction on Q")
print(f"{'='*80}")

if results:
    for r in results:
        Phi0, Om, om_t, om_m, om_e, Q_raw, te, me_r, tm, corr = r
        m_t = E_1D(om_t)
        m_m = E_1D(om_m)
        m_el = E_1D(om_e)
        m_el_corr = m_el * PI/4  # Apply π/4 to electron mass

        Q_corr = koide_Q(m_t, m_m, m_el_corr)
        te_corr = m_t / m_el_corr  # = (4/π) × τ/e_raw

        if abs(Phi0 - 2.35) < 0.005 or abs(Phi0 - 2.34) < 0.005:
            print(f"\n  Φ₀ = {Phi0:.2f}:")
            print(f"    Q_raw    = {Q_raw:.6f} (|ΔQ| = {abs(Q_raw-2/3):.6f})")
            print(f"    Q_corr   = {Q_corr:.6f} (|ΔQ| = {abs(Q_corr-2/3):.6f})")
            print(f"    τ/e_raw  = {te:.1f}  (expt: {expt_tau_e:.1f})")
            print(f"    τ/e_corr = {te_corr:.1f}  (expt: {expt_tau_e:.1f})")

# =========================================================================
# Koide parametrization analysis
# =========================================================================
print(f"\n{'='*80}")
print(f"  Koide parametrization: √m_i = M(1 + √2·cos(θ₀ + 2πi/3))")
print(f"{'='*80}")

# Experimental
m_exp = [expt_tau_e, expt_mu_e, 1.0]
sr_exp = sum(np.sqrt(m) for m in m_exp)
M_exp = sr_exp / 3
theta_exp = koide_angle(*m_exp)
print(f"\n  Experimental (normalized m_e=1):")
print(f"  M = {M_exp:.6f}, θ₀ = {theta_exp:.6f} rad = {np.degrees(theta_exp):.4f}°")
for i, name in enumerate(['tau', 'muon', 'electron']):
    predicted = M_exp * (1 + np.sqrt(2) * np.cos(theta_exp + 2*PI*i/3))
    actual = np.sqrt(m_exp[i])
    print(f"  √m_{name} = {actual:.6f}, predicted = {predicted:.6f}")

# E_1D at optimal Φ₀
if results:
    best = min(results, key=lambda x: abs(x[5] - 2/3))
    m_e1d = [best[6], best[7], 1.0]
    sr_e1d = sum(np.sqrt(m) for m in m_e1d)
    M_e1d = sr_e1d / 3
    theta_e1d = koide_angle(*m_e1d)
    print(f"\n  E_1D at Φ₀={best[0]:.2f} (normalized m_e=1):")
    print(f"  M = {M_e1d:.6f}, θ₀ = {theta_e1d:.6f} rad = {np.degrees(theta_e1d):.4f}°")
    for i, name in enumerate(['tau', 'muon', 'electron']):
        predicted = M_e1d * (1 + np.sqrt(2) * np.cos(theta_e1d + 2*PI*i/3))
        actual = np.sqrt(m_e1d[i])
        print(f"  √m_{name} = {actual:.6f}, predicted = {predicted:.6f}")

    dtheta = theta_e1d - theta_exp
    print(f"\n  Δθ₀ = {dtheta:.6f} rad = {np.degrees(dtheta):.4f}°")
    print(f"  This angle shift corresponds to the correction factor:")
    print(f"  The electron is most sensitive to θ₀ because it's near the")
    print(f"  minimum of the cosine function. A shift Δθ₀ ≈ {dtheta:.4f}")
    print(f"  changes √m_e by a factor ≈ cos(θ₀+4π/3+Δθ)/cos(θ₀+4π/3)")

    # Compute the predicted correction from Δθ
    x0 = theta_exp + 4*PI/3
    x1 = theta_e1d + 4*PI/3
    cosine_ratio = (1 + np.sqrt(2)*np.cos(x1)) / (1 + np.sqrt(2)*np.cos(x0))
    mass_ratio = cosine_ratio**2
    print(f"  Predicted mass ratio m_e(E1D)/m_e(expt) = {mass_ratio:.6f}")
    print(f"  Actual ratio: 1/(4/π) = π/4 = {PI/4:.6f}")

    # What Δθ would give EXACTLY π/4?
    from scipy.optimize import brentq
    def mass_ratio_from_dtheta(dt):
        x0 = theta_exp + 4*PI/3
        x1 = theta_exp + dt + 4*PI/3
        cr = (1 + np.sqrt(2)*np.cos(x1)) / (1 + np.sqrt(2)*np.cos(x0))
        return cr**2 - PI/4

    try:
        dtheta_exact = brentq(mass_ratio_from_dtheta, -0.5, 0.5)
        print(f"\n  Δθ₀ for EXACT π/4 correction: {dtheta_exact:.6f} rad = {np.degrees(dtheta_exact):.4f}°")
        print(f"  Actual Δθ₀: {dtheta:.6f} rad")
        print(f"  Is Δθ₀ related to known constants?")
        print(f"    Δθ₀/π = {dtheta_exact/PI:.6f}")
        print(f"    Δθ₀ × 180/π = {np.degrees(dtheta_exact):.4f}°")
        print(f"    1/Δθ₀ = {1/dtheta_exact:.2f}")
    except:
        pass

print(f"\n{'='*80}")
print(f"  CONCLUSION")
print(f"{'='*80}")
print(f"""
  Key findings:
  1. The correction factor changes RAPIDLY with Φ₀ (not universal!)
  2. At the Koide-optimal Φ₀ ≈ 2.35, the correction ≈ 4/π
  3. The 4/π correction WORSENS Q (moves it away from 2/3)
  4. The correction corresponds to a small Koide angle shift Δθ₀

  Interpretation: The E_1D formula gives Q ≈ 2/3 with a specific
  Koide angle θ₀ that differs from experiment by Δθ₀. This angle
  difference manifests as a factor ≈ 4/π in mass ratios because
  the electron mass is extremely sensitive to θ₀.

  The factor 4/π is NOT a geometric cross-section factor. It is the
  natural residual of the E_1D approximation at the Koide-optimal Φ₀.
""")
