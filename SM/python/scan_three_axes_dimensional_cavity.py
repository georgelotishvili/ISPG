"""
ISPG — Phase 11: ლეპტონი = n-განზომილებიანი რხევა
================================================================

მომხმარებლის ახალი ინტერპრეტაცია:
   ელექტრონი = 1D რხევა  (მხოლოდ 1 მიმართულება)
   მიუონი    = 2D რხევა  (2 მიმართულება ერთდროულად)
   ტაუ       = 3D რხევა  (სამივე მიმართულება)

ტესტი:
   ყოველი ლეპტონი თავის განზომილებიანი კავიტში ცხოვრობს:
   1D: −u'' + V(r) u = ω² u   (r ∈ [0, R], u(0)=u(R)=0)
   2D: რადიალური: −u'' − (1/r)u' + [V+ℓ²/r²] u = ω² u
   3D: −u'' − (2/r)u' + [V+ℓ(ℓ+1)/r²] u = ω² u

პოტენციალი V(r) = (1 − αΦ) e^{−αΦ} იგივე პულსონის ფონიდან.

მასის ფორმულა:
   m_i ∝ κ²_i = 1 − ω²_i (ბმულობის ენერგია)
   ან m_i ∝ ω_i

საცდელი: რა ფარდობები ვიპოვოთ?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

ALPHA_NL = 0.5
PHI_C = 2.35

def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]

def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL, r_max=40.0):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14, max_step=0.05)
        return sol.y[0, -1]
    Omega = brentq(residual, 0.30, 0.999, xtol=1e-12)
    r_eval = np.linspace(1e-10, r_max, 6000)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval, rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


def cavity_eigenvalues_nD(r, Phi_bg, alpha=ALPHA_NL, dim=3, ell=0, n_want=10):
    """
    n-განზომილებიანი რადიალური სქრედინგერი:
      -u'' - ((n-1)/r) u' + V(r) u + L_eff u = ω² u
    სადაც L_eff = ℓ(ℓ+n-2)/r² (ზოგადი n-სთვის)
    1D: no angular, dim=1, ℓ=0, L_eff = 0
    2D: L_eff = ℓ²/r²
    3D: L_eff = ℓ(ℓ+1)/r²

    ვიყენებთ u = r^((n-1)/2) ψ ტრანსფორმაციას, რომ მივიდეთ სქრედინგერის
    სტანდარტულ ფორმაზე:
      -ψ'' + [V + L_eff + (n-1)(n-3)/(4r²)] ψ = ω² ψ

    Dirichlet საზღვრები: ψ(0) = ψ(r_max) = 0.
    """
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)

    # კუთხოვანი + განზომილებიანი შესწორება
    if dim == 1:
        L_eff = np.zeros(N)
        dim_correction = np.zeros(N)
    else:
        # ზოგადი n-D: ℓ(ℓ+n-2)/r²
        L_ang_coeff = ell * (ell + dim - 2)
        L_eff = np.zeros(N)
        L_eff[1:] = L_ang_coeff / r[1:]**2
        L_eff[0] = L_eff[1]

        # განზომილების ფაქტორი
        dc_coeff = (dim - 1) * (dim - 3) / 4.0
        dim_correction = np.zeros(N)
        if abs(dc_coeff) > 1e-12:
            dim_correction[1:] = dc_coeff / r[1:]**2
            dim_correction[0] = dim_correction[1]

    W = V + L_eff + dim_correction

    diag = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2
    eigenvalues, _ = eigh_tridiagonal(diag, off_diag)
    return eigenvalues[:min(n_want, len(eigenvalues))]


print("=" * 72)
print(" Phase 11 — ლეპტონი = n-განზომილებიანი რხევა")
print("=" * 72)

print("\n[1] პულსონის ფონი...")
r, Phi, Omega = find_oscillon()
print(f"    Ω = {Omega:.6f}")

# ===================================================================
# ყოველი განზომილებისთვის ფუნდამენტი (ყველაზე დაბალი მდგომარეობა)
# ===================================================================

print("\n" + "-" * 72)
print(" [2] ფუნდამენტური მდგომარეობა ყოველ განზომილებაში")
print("-" * 72)

# 1D (ელექტრონი): მხოლოდ ℓ=0, dim=1
evals_1d = cavity_eigenvalues_nD(r, Phi, dim=1, ell=0, n_want=5)
bound_1d = evals_1d[evals_1d < 1.0]

# 2D (მიუონი): ℓ=0,1,2,... dim=2
all_2d = []
for ell in range(5):
    ev = cavity_eigenvalues_nD(r, Phi, dim=2, ell=ell, n_want=5)
    for n_idx, e in enumerate(ev[ev < 1.0]):
        all_2d.append((ell, n_idx, float(e), 1 if ell == 0 else 2))
all_2d.sort(key=lambda s: s[2])

# 3D (ტაუ): ℓ=0,1,2,... dim=3
all_3d = []
for ell in range(5):
    ev = cavity_eigenvalues_nD(r, Phi, dim=3, ell=ell, n_want=5)
    for n_idx, e in enumerate(ev[ev < 1.0]):
        all_3d.append((ell, n_idx, float(e), 2*ell+1))
all_3d.sort(key=lambda s: s[2])

print(f"\n   1D კავიტი (ელექტრონი):")
print(f"   {'n':>4} {'ω²':>10} {'κ²':>10}")
for n_idx, ev in enumerate(bound_1d):
    print(f"   {n_idx:>4} {ev:>10.6f} {1-ev:>10.6f}")

print(f"\n   2D კავიტი (მიუონი) — ენერგიით დალაგ.:")
print(f"   {'ℓ':>3} {'n':>3} {'ω²':>10} {'κ²':>10} {'დეგენ.':>6}")
for ell, n_idx, ev, deg in all_2d[:8]:
    print(f"   {ell:>3} {n_idx:>3} {ev:>10.6f} {1-ev:>10.6f} {deg:>6}")

print(f"\n   3D კავიტი (ტაუ) — ენერგიით დალაგ.:")
print(f"   {'ℓ':>3} {'n':>3} {'ω²':>10} {'κ²':>10} {'2ℓ+1':>6}")
for ell, n_idx, ev, deg in all_3d[:8]:
    print(f"   {ell:>3} {n_idx:>3} {ev:>10.6f} {1-ev:>10.6f} {deg:>6}")

# ===================================================================
# [3] ფუნდამენტების ფარდობა — პირდაპირი მასა?
# ===================================================================

print("\n" + "-" * 72)
print(" [3] მასის ფარდობა — n-D ფუნდამენტების შედარება")
print("-" * 72)

# ყველაზე ღრმა (ფუნდამენტი) ყოველ განზომილებაში
om2_1d = bound_1d[0] if len(bound_1d) > 0 else None
om2_2d = all_2d[0][2] if len(all_2d) > 0 else None
om2_3d = all_3d[0][2] if len(all_3d) > 0 else None

k2_1d = 1 - om2_1d
k2_2d = 1 - om2_2d
k2_3d = 1 - om2_3d

print(f"\n   ფუნდამენტები:")
print(f"   1D (ე):  ω² = {om2_1d:.6f}, κ² = {k2_1d:.6f}")
print(f"   2D (μ):  ω² = {om2_2d:.6f}, κ² = {k2_2d:.6f}")
print(f"   3D (τ):  ω² = {om2_3d:.6f}, κ² = {k2_3d:.6f}")

# დაკვირვებული
m_e = 0.51099895
m_mu = 105.6583755
m_tau = 1776.86

obs_mu_e = m_mu / m_e
obs_tau_e = m_tau / m_e

print(f"\n   დაკვირვებული:")
print(f"   m_μ/m_e = {obs_mu_e:.3f}")
print(f"   m_τ/m_e = {obs_tau_e:.1f}")

print(f"\n   სხვადასხვა ფორმულა:")
print(f"   ───────────────────")
# m ∝ κ²
r1 = k2_2d / k2_1d
r2 = k2_3d / k2_1d
print(f"   m ∝ κ²:")
print(f"     m_μ/m_e = κ²_2D/κ²_1D = {r1:.3f}  (დაკვ. {obs_mu_e:.2f}, ცდომ. {abs(r1-obs_mu_e)/obs_mu_e*100:.1f}%)")
print(f"     m_τ/m_e = κ²_3D/κ²_1D = {r2:.3f}  (დაკვ. {obs_tau_e:.1f}, ცდომ. {abs(r2-obs_tau_e)/obs_tau_e*100:.1f}%)")

# m ∝ ω
r3 = np.sqrt(om2_2d) / np.sqrt(om2_1d)
r4 = np.sqrt(om2_3d) / np.sqrt(om2_1d)
print(f"\n   m ∝ ω:")
print(f"     m_μ/m_e = {r3:.3f}  (დაკვ. {obs_mu_e:.2f}, ცდომ. {abs(r3-obs_mu_e)/obs_mu_e*100:.1f}%)")
print(f"     m_τ/m_e = {r4:.3f}  (დაკვ. {obs_tau_e:.1f}, ცდომ. {abs(r4-obs_tau_e)/obs_tau_e*100:.1f}%)")

# m ∝ 1/κ²  (შებრუნებული: ღრმა → მძიმე)
r5 = k2_1d / k2_2d
r6 = k2_1d / k2_3d
print(f"\n   m ∝ 1/κ²  (შებრუნებული):")
print(f"     m_μ/m_e = κ²_1D/κ²_2D = {r5:.3f}")
print(f"     m_τ/m_e = κ²_1D/κ²_3D = {r6:.3f}")

# m ∝ ω² · something
print(f"\n   m ∝ κ² · (განზომილება):")
r7 = (k2_2d * 2) / (k2_1d * 1)
r8 = (k2_3d * 3) / (k2_1d * 1)
print(f"     m_μ/m_e = 2·κ²_2D/κ²_1D = {r7:.3f}")
print(f"     m_τ/m_e = 3·κ²_3D/κ²_1D = {r8:.3f}")

# ===================================================================
# [4] თუ N_i ∝ κ² · dim
# ===================================================================

print("\n" + "-" * 72)
print(" [4] Mathieu ინდექსი N-ის ძიება")
print("-" * 72)

from scipy.special import mathieu_b

print(f"\n   დავუშვათ N_e = 5 (პაპიერზე):")
print(f"   ─────────────────────")
# N_i = 5 × (κ²_i/κ²_1D)
N_mu_v1 = 5 * (k2_2d / k2_1d)
N_tau_v1 = 5 * (k2_3d / k2_1d)
print(f"   V1: N = 5·(κ²_n/κ²_1D)")
print(f"     N_μ = {N_mu_v1:.2f}  (ელოდა 72, ცდომ. {abs(N_mu_v1-72)/72*100:.1f}%)")
print(f"     N_τ = {N_tau_v1:.2f}  (ელოდა 295, ცდომ. {abs(N_tau_v1-295)/295*100:.1f}%)")

# N_i = 5 × dim × (κ²_i/κ²_1D)
N_mu_v2 = 5 * 2 * (k2_2d / k2_1d)
N_tau_v2 = 5 * 3 * (k2_3d / k2_1d)
print(f"\n   V2: N = 5·dim·(κ²_n/κ²_1D)")
print(f"     N_μ = {N_mu_v2:.2f}  (ელოდა 72, ცდომ. {abs(N_mu_v2-72)/72*100:.1f}%)")
print(f"     N_τ = {N_tau_v2:.2f}  (ელოდა 295, ცდომ. {abs(N_tau_v2-295)/295*100:.1f}%)")

# N_i = dim² × რაღაც
print(f"\n   V3: N = 5·dim² ბუნებრივი (ბალისტიკური):")
print(f"     N_μ = 5·2² = 20")
print(f"     N_τ = 5·3² = 45")

# დაკვირვება: 72/5 ≈ 14.4, 295/5 = 59
# 14.4 ≈ 2² × 3.6?  59 ≈ 3² × 6.5?

# თუ κ²_i ∝ dim² (დიდი დონე ღრმა)
print(f"\n   [ფაქტი] dim-ის ზრდით κ² ზრდება ბუნებრივია")
print(f"     κ²_1D / 1² = {k2_1d:.6f}")
print(f"     κ²_2D / 2² = {k2_2d/4:.6f}")
print(f"     κ²_3D / 3² = {k2_3d/9:.6f}")

# ===================================================================
# [5] Mathieu-ის თვისებებით შემოწმება
# ===================================================================

print("\n" + "-" * 72)
print(" [5] Mathieu b_N ფარდობებით შემოწმება")
print("-" * 72)

q = 1.853
b5 = float(mathieu_b(5, q))
b72 = float(mathieu_b(72, q))
b295 = float(mathieu_b(295, q))

print(f"\n   Mathieu b_N(q=1.853):")
print(f"     b_5   = {b5:.4f}")
print(f"     b_72  = {b72:.4f}     b_72/b_5  = {b72/b5:.3f}  (μ/e={obs_mu_e:.2f})")
print(f"     b_295 = {b295:.4f}    b_295/b_5 = {b295/b5:.1f}  (τ/e={obs_tau_e:.1f})")

# ===================================================================
# [6] დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" დასკვნა — n-განზომილებიანი ჰიპოთეზა")
print("=" * 72)

# საუკეთესო შედეგი
best_mu_err = min(abs(r1-obs_mu_e)/obs_mu_e,
                  abs(r3-obs_mu_e)/obs_mu_e,
                  abs(r5-obs_mu_e)/obs_mu_e,
                  abs(r7-obs_mu_e)/obs_mu_e) * 100
best_tau_err = min(abs(r2-obs_tau_e)/obs_tau_e,
                   abs(r4-obs_tau_e)/obs_tau_e,
                   abs(r6-obs_tau_e)/obs_tau_e,
                   abs(r8-obs_tau_e)/obs_tau_e) * 100

print(f"""
   რა ვნახეთ:
   ──────────
   თუ ყოველი ლეპტონი თავის n-D კავიტში ცხოვრობს (n = 1, 2, 3):

   1D (ელექტრ.):   κ² = {k2_1d:.4f}
   2D (მიუონი):    κ² = {k2_2d:.4f}   ე.ი. {k2_2d/k2_1d:.2f}× უფრო ღრმა
   3D (ტაუ):       κ² = {k2_3d:.4f}   ე.ი. {k2_3d/k2_1d:.2f}× უფრო ღრმა

   ფარდობები ↔ დაკვირვება:
   m_μ/m_e: საუკეთესო ცდომილება {best_mu_err:.1f}%
   m_τ/m_e: საუკეთესო ცდომილება {best_tau_err:.1f}%

   თუ ცდომილება < 5%-ია, ჰიპოთეზა **დადასტურებულია**.
   თუ დიდია, ან მასის ფორმულა სხვაა, ან დიმენსიური ანზაცი არ მუშაობს.
""")

print("=" * 72)
print(" Phase 11 — დასრულდა")
print("=" * 72)
