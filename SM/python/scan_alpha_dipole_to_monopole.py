"""
ISPG — α = 1/137 ცდა: dipole-to-monopole ამპლიტუდის ფარდობა
================================================================

წყარო: MASS/ISPG_FrequencyToMass.tex eq:alpha_target (ხაზი 5677-5683):

    α = |Φ_{ℓ=1}|² / |Φ_{ℓ=0}|²  =?  1/137.036

ფიზიკური ინტერპრეტაცია (sec:charge, ხაზი 5495-5688):
   ℓ=0 მოდი = monopole (ცენტრალური, "გრავიტაციული" ზონა)
   ℓ=1 მოდი = dipole (ტრანსვერსული ასიმეტრია → ელექტრული მუხტი)
   α = ელექტრომაგნიტური კუპლინგი = dipole-ის ფარდობითი "სიძლიერე"

მეთოდი:
   1. 3D პულსონის ფონი Φ(r), α_NL=0.5, Φ_c=2.35 (იგივე რაც
      scan_three_axes_pulson_3D_spectrum-ში).
   2. კავიტის ეიგენპრობლემა ℓ=0 და ℓ=1 მოდებისთვის:
        −u'' + [V(r) + ℓ(ℓ+1)/r²] u = ω² u
   3. საწყისი (n=0) მოდის ეიგენფუნქცია u_ℓ(r) — ნორმირებული.
   4. რამდენიმე ფარდობის ცდა (eq:alpha_target-ის ფიზიკური ვარიანტები):
        A. max|Φ|² ფარდობა (peak amplitude)
        B. ∫|u|²dr ფარდობა (L² norm — ტრივიალური თუ ორივე ნორმ=1)
        C. ∫|Φ|² r² dr (volumeric ამპლიტუდა)
        D. ∫|u'|² dr (კინეტიკური, ∝ ემისიის სიმძლავრე)
        E. (ცენტრის ახლოს ამპლიტუდის ფარდობა)
   5. შევადარო 1/137.036 = 0.00729735.

სტატუსი: ცდა. "ერთი ველი" ბიოლოგიას პატივს ვცემს — ახალი სტრუქტურა
არ შემოვაქვს, მხოლოდ უკვე არსებული G₂(X) ოსცილონის ჩარჩოში ვმუშაობ.
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

# ===================================================================
# A. პულსონის ფონი (იგივე იდენტური, რაც Phase 8-ში)
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
                  n_points=6000):
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
# B. კავიტის ეიგენფუნქციები (ℓ=0, ℓ=1) — eigenvectors დაბრუნდება
# ===================================================================

def cavity_eigen(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=3):
    """
    აბრუნებს (eigenvalues[:n_want], eigenvectors[:, :n_want])
    eigenvectors[:, k] = u_k(r) 3D BVP-სთვის: u(r) = r·Φ(r).
    სასაზღვრო პირობა: u(0)=0, u(r_max)=0 (Dirichlet).
    """
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)
    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:]**2
    centrifugal[0] = centrifugal[1]
    W = V + centrifugal

    diag     = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2
    eigvals, eigvecs = eigh_tridiagonal(diag, off_diag)

    # შევავსოთ საზღვრული ნულებით u(0)=u(r_max)=0
    n_return = min(n_want, len(eigvals))
    u_full = np.zeros((N, n_return))
    u_full[1:-1, :] = eigvecs[:, :n_return]

    # ნორმირება ∫|u|² dr = 1
    for k in range(n_return):
        norm = np.sqrt(simpson(u_full[:, k]**2, r))
        u_full[:, k] /= norm
        # ნიშნის კონვენცია: u'(0) > 0 (პირველი ბიჯი დადებითი)
        if u_full[1, k] < 0:
            u_full[:, k] *= -1

    return eigvals[:n_return], u_full


# ===================================================================
# C. ფონის ამოხსნა
# ===================================================================

print("=" * 72)
print(" ISPG — α = 1/137 ცდა: dipole/monopole ამპლიტუდის ფარდობა")
print(" წყარო: MASS eq:alpha_target (ISPG_FrequencyToMass.tex:5677-5683)")
print("=" * 72)

print("\n[1] პულსონის ფონის ამოხსნა (α=0.5, Φ_c=2.35)...")
r, Phi_bg, Omega_bg = find_oscillon()
print(f"    Ω_bg = {Omega_bg:.6f}")
print(f"    r ∈ [0, {r[-1]:.1f}],  N = {len(r)}")
print(f"    Φ(0) = {Phi_bg[0]:.4f},  max|Φ| = {np.max(np.abs(Phi_bg)):.4f}")

# ===================================================================
# D. ℓ=0 და ℓ=1 მოდების ამოხსნა
# ===================================================================

print("\n[2] კავიტის ეიგენფუნქციები ℓ=0, 1-ისთვის...")

eig0, u0_all = cavity_eigen(r, Phi_bg, ell=0, n_want=3)
eig1, u1_all = cavity_eigen(r, Phi_bg, ell=1, n_want=3)

print(f"\n    ℓ=0 eigenvalues: {eig0}")
print(f"    ℓ=1 eigenvalues: {eig1}")

# ბმული თუ არაა?
bound0 = eig0 < 1.0
bound1 = eig1 < 1.0
n_bound0 = np.sum(bound0)
n_bound1 = np.sum(bound1)
print(f"\n    ℓ=0: {n_bound0} ბმული მდგომარეობა (ω²<1)")
print(f"    ℓ=1: {n_bound1} ბმული მდგომარეობა (ω²<1)")

if n_bound1 == 0:
    print("\n    ⚠ ℓ=1-ზე ბმული მდგომარეობა არ არსებობს — α-ცდა გაუქმდა.")
    print("    dipole-to-monopole ფარდობა უბმული მდგომარეობაზე ფიზიკურად ბუნდოვანია.")
    # continue anyway with lowest eigenvalue (unbound mode) just to get a number
    print("    ვცდით უმდაბლეს eigenvalue-ით (არა-ბმული), მხოლოდ რიცხვის გასაჩვენებლად.")

# საწყისი (n=0) მოდები
u0 = u0_all[:, 0]  # ℓ=0, n=0 — monopole (ტაუ-თან ასოცირებული)
u1 = u1_all[:, 0]  # ℓ=1, n=0 — dipole (ელექტრომაგნეტური)

om2_0 = eig0[0]
om2_1 = eig1[0]

print(f"\n    ℓ=0 (n=0):  ω² = {om2_0:.6f}   ω = {np.sqrt(max(om2_0,0)):.6f}")
print(f"    ℓ=1 (n=0):  ω² = {om2_1:.6f}   ω = {np.sqrt(max(om2_1,0)):.6f}")

# ===================================================================
# E. α-ფარდობის რამდენიმე ფიზიკური ინტერპრეტაცია
# ===================================================================

print("\n[3] α = |Φ_{ℓ=1}|² / |Φ_{ℓ=0}|² — რამდენიმე ინტერპრეტაცია")
print("    (ორივე u ნორმირებულია ∫|u|²dr=1)")

# Φ(r) = u(r)/r
r_safe = r.copy()
r_safe[0] = r[1]  # გავაუქმოთ 1/0
Phi0 = u0 / r_safe
Phi1 = u1 / r_safe

# სხვადასხვა ფარდობის ტესტი
alpha_target = 1.0 / 137.036

results = {}

# A. max|Φ|² ფარდობა (peak amplitude)
max_Phi0_sq = np.max(Phi0**2)
max_Phi1_sq = np.max(Phi1**2)
results["A_peak_amp_sq"] = max_Phi1_sq / max_Phi0_sq

# B. ∫|u|² dr (ტრივიალური = 1/1 რადგან ნორმირებულია — სწორი check)
int_u0_sq = simpson(u0**2, r)
int_u1_sq = simpson(u1**2, r)
results["B_u_L2_ratio"] = int_u1_sq / int_u0_sq

# C. ∫|Φ|² r² dr (volumeric, standard quantum mech)
int_Phi0_sq_vol = simpson(Phi0**2 * r**2, r)
int_Phi1_sq_vol = simpson(Phi1**2 * r**2, r)
results["C_Phi_volumeric"] = int_Phi1_sq_vol / int_Phi0_sq_vol

# D. ∫|u'|² dr (კინეტიკური ენერგია ∝ emission power ratio)
du0 = np.gradient(u0, r)
du1 = np.gradient(u1, r)
int_du0_sq = simpson(du0**2, r)
int_du1_sq = simpson(du1**2, r)
results["D_kinetic_ratio"] = int_du1_sq / int_du0_sq

# E. ∫|Φ|² dr (no volume weight)
int_Phi0_sq = simpson(Phi0**2, r)
int_Phi1_sq = simpson(Phi1**2, r)
results["E_Phi_plain"] = int_Phi1_sq / int_Phi0_sq

# F. max|u|² ფარდობა (peak u)
max_u0_sq = np.max(u0**2)
max_u1_sq = np.max(u1**2)
results["F_u_peak_sq"] = max_u1_sq / max_u0_sq

# G. cross-coupling integral ⟨u₀|V|u₁⟩ (dipole matrix element — rough)
# ეს უფრო elaborate dipole radiation amplitude-ს ჰგავს
overlap = simpson(u0 * u1 * Phi_bg, r)
results["G_dipole_overlap"] = overlap**2  # normalized cross-section-like

print(f"\n    სამიზნე: α = 1/137.036 = {alpha_target:.6e}")
print(f"\n    {'მეთოდი':<25} {'რიცხვი':<20} {'×137':<15} {'log₁₀':<10}")
print(f"    {'-'*25} {'-'*20} {'-'*15} {'-'*10}")
for key, val in results.items():
    log10_val = np.log10(abs(val)) if abs(val) > 1e-300 else -np.inf
    if val > 0:
        times_137 = val * 137.036
    else:
        times_137 = float('nan')
    print(f"    {key:<25} {val:<20.6e} {times_137:<15.4f} {log10_val:<10.3f}")

# ===================================================================
# F. ცხრილად — რომელი მიახლოებით 1/137-ია?
# ===================================================================

print("\n[4] ცდომილება 1/137.036-დან:")
print(f"    {'მეთოდი':<25} {'ცდომილება (%)':<15}")
print(f"    {'-'*25} {'-'*15}")
for key, val in results.items():
    if val > 0:
        err_pct = 100 * abs(val - alpha_target) / alpha_target
        close = "← SUSPICIOUSLY CLOSE!" if err_pct < 20 else ""
        print(f"    {key:<25} {err_pct:<15.2f} {close}")

# ===================================================================
# G. ფონური კონტროლი — რას ველი?
# ===================================================================

print("\n[5] დამატებითი ინფორმაცია:")
print(f"    ℓ=1 (n=0) ω² = {om2_1:.6f}  (ბმული: ω²<1 → {'კი' if om2_1 < 1 else 'არა'})")
print(f"    centrifugal barrier ℓ=1-ზე: ℓ(ℓ+1)/r² = 2/r² ⇒ ექსკლუზიური რეგიონი r < √2")
print(f"    Φ₁(r=0) უნდა იყოს ~0 (რადგან u∝r^(ℓ+1) r→0-ზე); max|Φ₁| = {np.max(np.abs(Phi1)):.4f}")
print(f"    Φ₀(r=0) ~ u'₀(0)/1; max|Φ₀| = {np.max(np.abs(Phi0)):.4f}")

print("\n" + "=" * 72)
print(" დასრულდა")
print("=" * 72)
