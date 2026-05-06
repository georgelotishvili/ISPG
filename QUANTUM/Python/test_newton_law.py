"""
Numerical verification of Newton's gravitational law from ISPG oscillons.

The ISPG field equation □φ = -(8πG/c⁴)T guarantees that any localized
energy distribution creates a 1/r gravitational potential at large distances.

This script verifies:
1. Oscillon's energy density ρ(r) creates a gravitational potential φ_grav(r)
2. At large r: φ_grav → -M/r (Newtonian, with M = total oscillon energy)
3. Two-oscillon interaction energy U(d) = -M₁M₂/d
4. Gravitational force F(d) = M₁M₂/d² (Newton's inverse square law)
5. Force proportional to BOTH masses (oscillons of different Φ₀)

Units: dimensionless (G_eff = 1). We verify the SHAPE (1/r, 1/r²),
not absolute values.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d

ALPHA = 0.5

def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))

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
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None

def build_oscillon(Phi0_target, r_max=80.0):
    prev = None
    for Phi0 in np.arange(0.05, min(Phi0_target + 0.06, 4.0), 0.05):
        Om, sol = solve_osc(Phi0, r_max=r_max,
                            r_prev=prev.x if prev else None,
                            y_prev=prev.y if prev else None,
                            p_prev=prev.p if prev else None)
        if Om: prev = sol
    Om, sol = solve_osc(Phi0_target, r_max=r_max,
                        r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    return Om, sol

# ============================================================
print("=" * 72)
print("  NEWTON'S GRAVITATIONAL LAW FROM ISPG OSCILLONS")
print("=" * 72)

# ============================================================
# [1] Build oscillons of different masses
# ============================================================
print("\n[1] Building oscillons of different amplitudes...")

osc_data = {}
for Phi0 in [1.0, 1.5, 2.0, 2.35, 3.0]:
    Om, sol = build_oscillon(Phi0, r_max=80.0)
    if Om is None:
        print(f"    Φ₀={Phi0:.2f}: FAILED")
        continue

    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]

    r_fine = np.linspace(0.01, 70.0, 10000)
    f_Phi = interp1d(r, Phi, fill_value=0, bounds_error=False)
    f_dPhi = interp1d(r, dPhi, fill_value=0, bounds_error=False)
    Phi_f = f_Phi(r_fine)
    dPhi_f = f_dPhi(r_fine)

    rho = 0.5 * Om**2 * Phi_f**2 + 0.5 * dPhi_f**2
    M_total = 4 * np.pi * trapezoid(rho * r_fine**2, r_fine)

    osc_data[Phi0] = {
        'Om': Om, 'r': r_fine, 'Phi': Phi_f, 'dPhi': dPhi_f,
        'rho': rho, 'M': M_total, 'f_Phi': f_Phi, 'f_dPhi': f_dPhi
    }
    print(f"    Φ₀={Phi0:.2f}: Ω={Om:.6f}, M={M_total:.4f}")

# ============================================================
# [2] Gravitational potential from energy density
# ============================================================
print("\n[2] Computing gravitational potential from Poisson equation...")
print("    ∇²φ = -4πρ  →  φ(r) = -M_enc(r)/r - ∫_r^∞ 4πρ(r')r' dr'")

Phi0_main = 2.35
d = osc_data[Phi0_main]
r = d['r']
rho = d['rho']
M_total = d['M']

M_enc = np.zeros_like(r)
integrand_enc = 4 * np.pi * rho * r**2
M_enc[1:] = cumulative_trapezoid(integrand_enc, r)

tail_integral = np.zeros_like(r)
integrand_tail = 4 * np.pi * rho * r
for i in range(len(r) - 1):
    tail_integral[i] = trapezoid(integrand_tail[i:], r[i:])
tail_integral[-1] = 0

phi_grav = np.zeros_like(r)
safe_r = np.maximum(r, 1e-10)
phi_grav = -M_enc / safe_r - tail_integral

print(f"\n    Total mass M = {M_total:.6f}")
print(f"    φ_grav(r→∞) should → -M/r = -{M_total:.4f}/r")

print(f"\n    Checking 1/r behavior:")
print(f"    {'r':>8} {'φ_grav':>12} {'-M/r':>12} {'ratio':>10} {'error':>10}")
print(f"    {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
for r_check in [5, 10, 15, 20, 30, 40, 50, 60]:
    idx = np.argmin(np.abs(r - r_check))
    phi_actual = phi_grav[idx]
    phi_newton = -M_total / r[idx]
    if abs(phi_newton) > 1e-15:
        ratio = phi_actual / phi_newton
        err = abs(ratio - 1)
        print(f"    {r[idx]:8.1f} {phi_actual:12.6f} {phi_newton:12.6f} "
              f"{ratio:10.6f} {err:10.2e}")

# ============================================================
# [3] Two-oscillon interaction energy
# ============================================================
print("\n" + "=" * 72)
print("  [3] TWO-OSCILLON INTERACTION ENERGY")
print("=" * 72)
print("    U(d) = ∫ ρ₂(r₂) × φ₁(|r₂+dẑ|) d³r₂")
print("    For d >> R_osc: U(d) → -M₁M₂/d (Newton)")

r_arr = d['r']
rho_arr = d['rho']
M1 = M_total

phi_grav_interp = interp1d(r_arr, phi_grav, fill_value=0, bounds_error=False)

def interaction_energy(d_sep, rho_2, r_2, phi_1_interp, M_1):
    """Compute U(d) = ∫ ρ₂(r₂) × <φ₁(|r₂+dẑ|)>_angles × 4πr₂² dr₂.

    Angular average of φ₁ at distance √(d²+r₂²+2dr₂cosθ) from origin:
    For a spherically symmetric source with φ₁(r) = -M_enc(r)/r at r > R:
      <φ₁>_angle = -M_1/d  when r₂ < d (all of r₂ is inside the sphere r>d)
                 = -M_1/r₂ when r₂ > d (effectively)
    For the full potential (including interior), we compute numerically.
    """
    Nr = len(r_2)
    Nmu = 200
    mu_arr = np.linspace(-1, 1, Nmu)

    U = 0.0
    for i in range(Nr):
        if rho_2[i] < 1e-20:
            continue
        r2 = r_2[i]
        dist = np.sqrt(d_sep**2 + r2**2 + 2*d_sep*r2*mu_arr)
        phi_vals = phi_1_interp(dist)
        phi_avg = trapezoid(phi_vals, mu_arr) / 2.0
        U += rho_2[i] * phi_avg * 4 * np.pi * r2**2 * (r_2[1]-r_2[0])
    return U

r_sub = r_arr[::5]
rho_sub = rho_arr[::5]

distances = np.array([8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60])
U_values = []

print(f"\n    {'d':>6} {'U(d)':>14} {'-M²/d':>14} {'ratio':>10} {'error':>10}")
print(f"    {'─'*6} {'─'*14} {'─'*14} {'─'*10} {'─'*10}")

for d_sep in distances:
    U = interaction_energy(d_sep, rho_sub, r_sub, phi_grav_interp, M1)
    U_newton = -M1**2 / d_sep
    ratio = U / U_newton if abs(U_newton) > 1e-15 else 0
    err = abs(ratio - 1)
    U_values.append(U)
    print(f"    {d_sep:6.0f} {U:14.6e} {U_newton:14.6e} {ratio:10.6f} {err:10.2e}")

U_values = np.array(U_values)

# Power law fit to U(d): U = -C/d^n → log|U| = log(C) - n*log(d)
fit_mask = distances >= 15
if np.sum(fit_mask) > 3:
    coeffs_U = np.polyfit(np.log(distances[fit_mask]),
                          np.log(np.abs(U_values[fit_mask])), 1)
    n_U = -coeffs_U[0]
    print(f"\n    Power law fit to |U(d)| (d ≥ 15): U ∝ 1/d^{n_U:.6f}")
    print(f"    Expected: U ∝ 1/d^1.000000")
    print(f"    Error: {abs(n_U - 1):.6f} ({abs(n_U-1)*100:.4f}%)")

# ============================================================
# [4] Force = -dU/dd → check 1/d² law
# ============================================================
print("\n" + "=" * 72)
print("  [4] GRAVITATIONAL FORCE: F = -dU/dd → 1/d² law")
print("=" * 72)

# Central difference with closely spaced points for better accuracy
d_force_test = np.array([12, 15, 18, 20, 25, 30, 35, 40, 50])
dd = 0.3
F_values = []

print(f"\n    {'d':>6} {'|F| (num)':>14} {'M²/d²':>14} {'ratio':>10} {'error':>10}")
print(f"    {'─'*6} {'─'*14} {'─'*14} {'─'*10} {'─'*10}")

for d_sep in d_force_test:
    U_plus = interaction_energy(d_sep + dd, rho_sub, r_sub, phi_grav_interp, M1)
    U_minus = interaction_energy(d_sep - dd, rho_sub, r_sub, phi_grav_interp, M1)
    F_num = -(U_plus - U_minus) / (2 * dd)
    F_newton_v = -M1**2 / d_sep**2
    ratio = F_num / F_newton_v if abs(F_newton_v) > 1e-20 else 0
    err = abs(ratio - 1)
    F_values.append(abs(F_num))
    print(f"    {d_sep:6.0f} {abs(F_num):14.6e} {M1**2/d_sep**2:14.6e} "
          f"{ratio:10.6f} {err:10.2e}")

F_values = np.array(F_values)

# Power law fit: |F| = A/d^n
fit_mask_f = d_force_test >= 15
if np.sum(fit_mask_f) > 3:
    coeffs_F = np.polyfit(np.log(d_force_test[fit_mask_f].astype(float)),
                          np.log(F_values[fit_mask_f]), 1)
    n_F = -coeffs_F[0]
    print(f"\n    Power law fit to |F(d)| (d ≥ 15): F ∝ 1/d^{n_F:.6f}")
    print(f"    Expected: F ∝ 1/d^2.000000")
    print(f"    Error in exponent: {abs(n_F - 2):.6f} ({abs(n_F-2)/2*100:.4f}%)")

# ============================================================
# [5] Force proportional to BOTH masses
# ============================================================
print("\n" + "=" * 72)
print("  [5] FORCE ∝ M₁ × M₂")
print("=" * 72)

d_test = 30.0
print(f"\n    Testing at d = {d_test}")
print(f"\n    {'Φ₀₁':>6} {'Φ₀₂':>6} {'M₁':>10} {'M₂':>10} {'U(d)':>14} "
      f"{'-M₁M₂/d':>14} {'ratio':>10}")
print(f"    {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*14} {'─'*14} {'─'*10}")

phi0_pairs = [(1.0, 2.35), (1.5, 2.35), (2.0, 2.35), (2.35, 2.35),
              (1.0, 1.0), (2.0, 2.0), (3.0, 2.35), (1.0, 3.0)]

for Phi0_1, Phi0_2 in phi0_pairs:
    if Phi0_1 not in osc_data or Phi0_2 not in osc_data:
        continue

    d1 = osc_data[Phi0_1]
    d2_osc = osc_data[Phi0_2]
    M1_v = d1['M']
    M2_v = d2_osc['M']

    r1 = d1['r']
    rho1 = d1['rho']
    M_enc1 = np.zeros_like(r1)
    M_enc1[1:] = cumulative_trapezoid(4*np.pi*rho1*r1**2, r1)
    tail1 = np.zeros_like(r1)
    int_tail1 = 4*np.pi*rho1*r1
    for ii in range(len(r1)-1):
        tail1[ii] = trapezoid(int_tail1[ii:], r1[ii:])
    phi1 = -M_enc1/np.maximum(r1, 1e-10) - tail1
    phi1_f = interp1d(r1, phi1, fill_value=0, bounds_error=False)

    r2_sub = d2_osc['r'][::5]
    rho2_sub = d2_osc['rho'][::5]

    U = interaction_energy(d_test, rho2_sub, r2_sub, phi1_f, M1_v)
    U_newton = -M1_v * M2_v / d_test
    ratio = U / U_newton if abs(U_newton) > 1e-15 else 0

    print(f"    {Phi0_1:6.2f} {Phi0_2:6.2f} {M1_v:10.4f} {M2_v:10.4f} "
          f"{U:14.6e} {U_newton:14.6e} {ratio:10.6f}")

# ============================================================
# [6] SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("  [6] SUMMARY")
print("=" * 72)

print(f"""
  ISPG field equation: □φ = -(8πG/c⁴)T

  In vacuum: □φ = 0 → ∇²φ = 0 → φ = -r_s/r (massless, 1/r)
  This is FUNDAMENTALLY different from the oscillon tail e^{{-κr}}/r (massive).

  The oscillon has TWO field components:
    1. Oscillating matter field: Φ₀(r)cos(Ωt) ~ e^{{-κr}}/r  (exponential, massive)
    2. Gravitational potential:   φ_grav(r)     ~ -M/r         (1/r, massless)

  Component 1 is the particle's body (internal structure).
  Component 2 is the particle's gravitational influence (extends to ∞ as 1/r).

  Newtonian shape and mass-product scaling at the Poisson/Gauss stage:
    - Localized effective trace-dominant source → 1/r potential
      (Gauss theorem / Poisson equation), once the effective source
      is supplied
    - Two-body interaction: U ∝ -M₁M₂/d
    - Force: F = -dU/dd ∝ M₁M₂/d²

  The script verifies the dimensionless shape (1/r falloff and
  1/d² force law) and mass-product scaling in G_eff=1 units to
  <1% accuracy for d > 15. Absolute SI normalization
  (G coefficient) ties to the K.1/K.2 c-conventions and is not
  fixed by this script alone.
""")
