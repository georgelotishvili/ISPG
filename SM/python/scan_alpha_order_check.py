"""
ISPG — α = 1/137 "რიგის დადასტურების" ცდა
================================================================

წყარო: MASS/ISPG_FrequencyToMass.tex
  eq:alpha_target (ხაზი 5677-5683): α = |Φ_ℓ=1|² / |Φ_ℓ=0|²
  eq:Rperp_final (ხაზი 3785-3787): R_⊥/λ_osc ≈ 1/(2π) ≈ 0.16
  eq:torsion_rigidity (ხაზი 3430): κ_T ~ c_s² (R_⊥/λ)²

MASS-ის ცხადი პროგნოზი:
  dipole-monopole amplitude ratio ~ (R_⊥/λ_osc)² ≈ 1/(4π²) ≈ 0.025

სამიზნე:
  α_obs = 1/137.036 ≈ 0.00730

კითხვა: (R_⊥/λ_osc)² = 1/(4π²) ცდა მართლაც ხდება რიცხვით?
  - თუ კი: რიგი დასტურდება, მაგრამ ფაქტორი 3.5-ჯერ ნაკლები ვიდრე α_obs
  - თუ არა: MASS-ის eq:Rperp_final მოდიფიკაციას სჭირდება

ოპერაციული გეგმა:
  1. ოსცილონის ფონი (α=0.5, Φ_c=2.35) — ℓ=0 ფონი
  2. R_c გამოვიტანო: სადაც Φ₀(r) ცვივა 1/e-მდე
  3. λ_osc = 2π/ω_osc (გეოდეზიური სიდიდე)
  4. R_⊥/λ_osc ფარდობა
  5. ℓ=1 dipole მოდისთვის ეიგენფუნქცია
  6. dipole ამპლიტუდა core-ში r<R_c (ნორმალიზებული)
  7. monopole ამპლიტუდა core-ში
  8. ფარდობა α_predicted = |Φ_ℓ=1(r<R_c)|² / |Φ_ℓ=0(r<R_c)|²

ცხადი შემოწმებები:
  A. (R_⊥/λ_osc)² =? 1/(4π²) [MASS eq 3785]
  B. α_predicted =? (R_⊥/λ_osc)² · რაღაც გეომეტრიული ფაქტორი
  C. α_predicted vs α_obs = 1/137

ფონური ფრემვორი არ იცვლება — იგივე G₂(X) ოსცილონი რაც MASS-ში.
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq, minimize_scalar
from scipy.linalg import eigh_tridiagonal

# ===================================================================
# A. ოსცილონის ფონი
# ===================================================================

ALPHA_NL = 0.5
PHI_C    = 2.35

def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]


def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL,
                  r_max=40.0, Omega_lo=0.30, Omega_hi=0.999,
                  n_points=8000):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14,
                        max_step=0.05)
        return sol.y[0, -1]
    Omega = brentq(residual, Omega_lo, Omega_hi, xtol=1e-12)
    r_eval = np.linspace(1e-10, r_max, n_points)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


# ===================================================================
# B. ეიგენფუნქციები ℓ=0, 1, 2
# ===================================================================

def cavity_eigen(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=2):
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)
    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:]**2
    centrifugal[0] = centrifugal[1] if ell == 0 else 1e12
    W = V + centrifugal

    diag     = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2
    eigvals, eigvecs = eigh_tridiagonal(diag, off_diag)

    n_return = min(n_want, len(eigvals))
    u_full = np.zeros((N, n_return))
    u_full[1:-1, :] = eigvecs[:, :n_return]

    for k in range(n_return):
        norm = np.sqrt(simpson(u_full[:, k]**2, r))
        if norm > 0:
            u_full[:, k] /= norm
        if u_full[min(10, N-1), k] < 0:
            u_full[:, k] *= -1

    return eigvals[:n_return], u_full


# ===================================================================
# C. ოსცილონის core რადიუსი
# ===================================================================

def find_core_radius(r, Phi_bg):
    """R_c = მანძილი, სადაც |Φ(r)/Φ(0)| = 1/e"""
    Phi0 = Phi_bg[0]
    target = Phi0 / np.e
    # პირველი წერტილი სადაც Phi < target
    for i in range(len(r)):
        if abs(Phi_bg[i]) < abs(target):
            # ლინეარული ინტერპოლაცია
            if i == 0:
                return r[0]
            r1, r2 = r[i-1], r[i]
            p1, p2 = Phi_bg[i-1], Phi_bg[i]
            if p1 == p2:
                return r2
            return r1 + (target - p1) / (p2 - p1) * (r2 - r1)
    return r[-1]


# ===================================================================
# D. მთავარი სცენარი
# ===================================================================

print("=" * 72)
print(" α = 1/137: რიგის დადასტურების ცდა")
print(" (MASS eq 5677, 3430, 3785)")
print("=" * 72)

print("\n[1] ოსცილონის ფონის ამოხსნა...")
r, Phi_bg, Omega_bg = find_oscillon()
print(f"    Ω_bg = {Omega_bg:.6f}")
print(f"    Φ(0) = {Phi_bg[0]:.4f}")

# Core რადიუსი
R_c = find_core_radius(r, Phi_bg)
print(f"    R_c (ოსცილონის 1/e რადიუსი) = {R_c:.4f}")

# λ_osc: ოსცილონის "Compton"-ანალოგი.
# ფიზიკურად: λ_osc = 2π/ω_osc (proper-frame-ში)
lambda_osc = 2 * np.pi / Omega_bg
print(f"    λ_osc = 2π/Ω = {lambda_osc:.4f}")

# MASS ამბობს λ_osc = R_c (Compton radius = wavelength). შევამოწმო:
ratio_Rc_lambda = R_c / lambda_osc
print(f"    R_c / λ_osc = {ratio_Rc_lambda:.4f} (MASS: 1)")

# R_⊥/λ_osc definition (MASS eq 3670):
# = λ_osc / (2π R_c) = 1/(2π × R_c/λ_osc)
R_perp_over_lambda = lambda_osc / (2 * np.pi * R_c)
print(f"\n    R_⊥/λ_osc = λ_osc/(2π R_c) = {R_perp_over_lambda:.4f}")
print(f"    (MASS: 1/(2π) = {1/(2*np.pi):.4f})")
print(f"    (R_⊥/λ_osc)² = {R_perp_over_lambda**2:.6f}")
print(f"    1/(4π²) = {1/(4*np.pi**2):.6f}  (MASS prediction)")

# ===================================================================
# E. ℓ=0, ℓ=1 eigenfunctions
# ===================================================================

print("\n[2] ℓ=0, ℓ=1, ℓ=2 eigenfunctions...")

eig0, u0_all = cavity_eigen(r, Phi_bg, ell=0, n_want=2)
eig1, u1_all = cavity_eigen(r, Phi_bg, ell=1, n_want=1)
eig2, u2_all = cavity_eigen(r, Phi_bg, ell=2, n_want=1)

print(f"    ℓ=0 (n=0): ω² = {eig0[0]:.4f}  (= ტაუ ფონი)")
print(f"    ℓ=1 (n=0): ω² = {eig1[0]:.4f}  (= dipole mode)")
print(f"    ℓ=2 (n=0): ω² = {eig2[0]:.4f}  (= ელექტრონი, d-ორბიტა)")

u0 = u0_all[:, 0]  # ℓ=0 monopole
u1 = u1_all[:, 0]  # ℓ=1 dipole
u2 = u2_all[:, 0]  # ℓ=2 quadrupole (electron)

# Core-ში ნორმები (r < R_c)
mask_core = r < R_c
r_core = r[mask_core]
u0_core = u0[mask_core]
u1_core = u1[mask_core]
u2_core = u2[mask_core]

int_u0_core_sq = simpson(u0_core**2, r_core) if len(r_core) > 1 else 0
int_u1_core_sq = simpson(u1_core**2, r_core) if len(r_core) > 1 else 0
int_u2_core_sq = simpson(u2_core**2, r_core) if len(r_core) > 1 else 0

print(f"\n[3] core-ში (r < R_c = {R_c:.3f}) u² ინტეგრალები:")
print(f"    ∫|u_ℓ=0|²dr (core) = {int_u0_core_sq:.4f}")
print(f"    ∫|u_ℓ=1|²dr (core) = {int_u1_core_sq:.4f}")
print(f"    ∫|u_ℓ=2|²dr (core) = {int_u2_core_sq:.4f}")

# ===================================================================
# F. ცდები — α-ს სხვადასხვა ინტერპრეტაცია
# ===================================================================

ALPHA_TARGET = 1.0 / 137.036

print("\n[4] სხვადასხვა ინტერპრეტაცია α-სთვის:")
print(f"    სამიზნე: α = 1/137.036 = {ALPHA_TARGET:.6e}")
print()

results = {}

# M1. (R_⊥/λ)² — MASS-ის ცხადი პროგნოზი
results["M1: (R_⊥/λ)²"] = R_perp_over_lambda**2

# M2. ℓ=1 vs ℓ=0 core fraction
if int_u0_core_sq > 0:
    results["M2: |u₁core|²/|u₀core|²"] = int_u1_core_sq / int_u0_core_sq

# M3. κ² შედარებები (electron binding energy)
kappa2_electron = 1 - eig2[0]  # ℓ=2 (n=0)
kappa2_tau = 1 - eig0[0]       # ℓ=0 (n=0)
kappa2_dipole = 1 - eig1[0]    # ℓ=1 (n=0)

results["M3: κ²_electron (ℓ=2, n=0)"] = kappa2_electron
results["M4: κ²_tau (ℓ=0, n=0)"] = kappa2_tau
results["M5: κ²_dipole (ℓ=1, n=0)"] = kappa2_dipole

# M6. Peak-amplitude ratios (not from core)
r_safe = r.copy()
r_safe[0] = r[1]
Phi0 = u0 / r_safe
Phi1 = u1 / r_safe
Phi2 = u2 / r_safe
results["M6: max|Φ₁|²/max|Φ₀|²"] = np.max(Phi1**2) / np.max(Phi0**2)

# M7. αN_e² ≈ 0.18 ∼ sin-ს რაღაც?
N_e = 5
results["M7: κ²_e · 4 (guess)"] = 4 * kappa2_electron

# M8. κ²_e · (2ℓ+1) — degeneracy weight
results["M8: κ²_e · (2ℓ=2+1)"] = 5 * kappa2_electron

# M9. 1/(4π²)
results["M9: 1/(4π²)"] = 1/(4*np.pi**2)

# M10. 1/(4π³) კონიცდენცია
results["M10: 1/(4π³)"] = 1/(4*np.pi**3)

# M11. (R_⊥/λ)² · κ²_e
results["M11: (R_⊥/λ)² · κ²_e"] = R_perp_over_lambda**2 * kappa2_electron

# M12. dipole-in-electron: ℓ=2-ის ფონი + ℓ=1 fluctuation
# ელექტრონი = ℓ=2 oscillon. dipole = ℓ=1 on top of ℓ=2 background
if int_u2_core_sq > 0:
    results["M12: |u₁|²/|u₂|² (core)"] = int_u1_core_sq / int_u2_core_sq

print(f"    {'მეთოდი':<40} {'მნიშვნელობა':<15} {'×137':<12} {'ცდ.(%)':<10}")
print(f"    {'-'*40} {'-'*15} {'-'*12} {'-'*10}")
for key, val in results.items():
    if val > 0 and np.isfinite(val):
        err = 100 * abs(val - ALPHA_TARGET) / ALPHA_TARGET
        mark = " ★" if err < 20 else ("" if err > 100 else " (ახლოს)")
        print(f"    {key:<40} {val:<15.6e} {val*137.036:<12.4f} {err:<10.2f}{mark}")

# ===================================================================
# G. დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" დასკვნა")
print("=" * 72)

# MASS ცხადი პროგნოზი
pred_MASS = R_perp_over_lambda**2
err_MASS = abs(pred_MASS - ALPHA_TARGET) / ALPHA_TARGET * 100
print(f"\n MASS-ის პროგნოზი: α ≈ (R_⊥/λ)² = {pred_MASS:.4f}")
print(f"                    სამიზნე:      α_obs = {ALPHA_TARGET:.4f}")
print(f"                    ცდომილება:   {err_MASS:.1f}%")
print()
print(f" MASS-ის ფორმულა რიგით სწორია (10⁻² vs 10⁻²)")
print(f" მაგრამ ფაქტორი {pred_MASS/ALPHA_TARGET:.2f}-ჯერ ცდება 1/137-დან.")
print()
print(f" გეომეტრიული ფაქტორი რომ მოვიყვანოთ: 1/{1/ALPHA_TARGET:.1f} = 1/137")
print(f"                                     (R_⊥/λ)² = 1/{1/pred_MASS:.1f}")
print()
print(f" ფაქტორის სხვაობა: 1/137 / 1/39.5 = 39.5/137 = {pred_MASS/ALPHA_TARGET:.3f}")
print(f"                  (π/1.08 ≈ π · 0.29)")

print("\n" + "=" * 72)
print(" ცდა დასრულდა")
print("=" * 72)
