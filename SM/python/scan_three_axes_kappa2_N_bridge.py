"""
ISPG — Phase 9: κ² → N ხიდი (3D → 1D Mathieu)
================================================================

Phase 8-ის დიდი მინიშნება:
   κ²_tau / κ²_e = 292.4   (3D კავიტი)
   N_tau = 295             (Mathieu ინდექსი)
   292 ≈ 295 (ცდომილება 0.9%)

ჰიპოთეზა:
   3D პულსონის κ² = 1 − ω² "ბმულობის ენერგია" არის ის,
   რაც განსაზღვრავს, რომელი Mathieu ზოლში (N-ზე) ჩაჯდება
   ეფექტური ოსცილაცია.

   κ²_i / κ²_e = N_i / N_e

   N_e = 5, ე.ი. N_i = 5 · (κ²_i / κ²_e)

ტესტი:
   A. κ²-ფარდობების გამოთვლა სამი ბმული მდგომარეობისთვის
   B. თუ N_τ/N_e = κ²_τ/κ²_e, შემდეგ m_τ/m_e = b_{N_τ}/b_{N_e}
   C. 59-ის ახსნა: 59 = κ²_τ/κ²_e × (N_e/5)?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
from scipy.special import mathieu_b

ALPHA_NL = 0.5
PHI_C = 2.35
Q = 1.853

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

def cavity_eigenvalues(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=20):
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)
    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:]**2
    centrifugal[0] = centrifugal[1]
    W = V + centrifugal
    diag = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2
    eigenvalues, _ = eigh_tridiagonal(diag, off_diag)
    return eigenvalues[:min(n_want, len(eigenvalues))]


print("=" * 72)
print(" Phase 9 — κ² → N ხიდი (3D კავიტი → 1D Mathieu ინდექსი)")
print("=" * 72)

print("\n[1] პულსონის ამოხსნა...")
r, Phi, Omega = find_oscillon()

# ბმული მდგომარეობები
e_evals = cavity_eigenvalues(r, Phi, ell=2, n_want=3)
mu_evals = cavity_eigenvalues(r, Phi, ell=0, n_want=3)

om2_tau = mu_evals[0]   # ℓ=0, n=0
om2_mu  = mu_evals[1]   # ℓ=0, n=1
om2_e   = e_evals[0]    # ℓ=2, n=0

k2_tau = 1 - om2_tau
k2_mu = 1 - om2_mu
k2_e = 1 - om2_e

print(f"\n    ტაუ   (ℓ=0,n=0): ω² = {om2_tau:.6f}, κ² = {k2_tau:.6f}")
print(f"    მიუონი (ℓ=0,n=1): ω² = {om2_mu:.6f},  κ² = {k2_mu:.6f}")
print(f"    ელექტრ.(ℓ=2,n=0): ω² = {om2_e:.6f},  κ² = {k2_e:.6f}")

# ===================================================================
# A. κ² ფარდობები
# ===================================================================

print("\n" + "-" * 72)
print(" A. κ² ფარდობები — 3D ხიდი")
print("-" * 72)

ratio_tau_e = k2_tau / k2_e
ratio_mu_e = k2_mu / k2_e
ratio_tau_mu = k2_tau / k2_mu

print(f"\n    κ²_τ / κ²_e = {ratio_tau_e:.4f}")
print(f"    κ²_μ / κ²_e = {ratio_mu_e:.4f}")
print(f"    κ²_τ / κ²_μ = {ratio_tau_mu:.4f}")

# ===================================================================
# B. ჰიპოთეზა: N = 5 × (κ²/κ²_e)
# ===================================================================

print("\n" + "-" * 72)
print(" B. N = 5·(κ²_i/κ²_e) — Mathieu ინდექსის წარმოშობა?")
print("-" * 72)

N_e_assumed = 5
N_tau_from_k2 = N_e_assumed * ratio_tau_e
N_mu_from_k2 = N_e_assumed * ratio_mu_e

print(f"\n    N_e (ფიქს.) = {N_e_assumed}")
print(f"    N_τ (პროგნ.) = 5 × {ratio_tau_e:.4f} = {N_tau_from_k2:.3f}")
print(f"    N_μ (პროგნ.) = 5 × {ratio_mu_e:.4f} = {N_mu_from_k2:.3f}")

print(f"\n    დაკვირვება (სტატიიდან):")
print(f"    N_τ = 295,  N_μ = 72")

err_tau = abs(N_tau_from_k2 - 295) / 295 * 100
err_mu = abs(N_mu_from_k2 - 72) / 72 * 100

print(f"\n    ცდომილება:")
print(f"      N_τ: {err_tau:.2f}%")
print(f"      N_μ: {err_mu:.2f}%")

# ===================================================================
# C. 59-ის წარმოშობა
# ===================================================================

print("\n" + "-" * 72)
print(" C. რას ნიშნავს κ²_τ/κ²_e = 292.4?")
print("-" * 72)

ratio_tau_index = ratio_tau_e
ratio_tau_over_e = ratio_tau_index / N_e_assumed

print(f"""
   3D-ის გამოთვლა: κ²_τ/κ²_e = {ratio_tau_index:.4f}

   ეს რიცხვი ახლოსაა selected N_τ = 295-თან,
   არა ფარდობა N_τ/N_e = 59-თან.

   ნორმალიზებული ფარდობა:
      N_τ/N_e ≈ {ratio_tau_index:.4f} / 5 = {ratio_tau_over_e:.4f}
      selected 59-თან ცდომილება {abs(ratio_tau_over_e - 59)/59*100:.2f}%

   დასკვნა: κ² ფარდობა არის τ-ინდექსის სკალის ძლიერი მინიშნება,
   არა ზუსტი 59-ის პირდაპირი გამოყვანა.
""")

close = abs(ratio_tau_index - 295) / 295 * 100
print(f"   გამოთვლა: {ratio_tau_index:.4f} → selected N_τ=295-თან ცდომილება {close:.2f}%")

# ===================================================================
# D. რამდენად ახლოა 292-სა და 295-ს შორის? რა აკლია?
# ===================================================================

print("\n" + "-" * 72)
print(" D. 292 ≈ 295 — რატომ არ არის ზუსტი?")
print("-" * 72)

print(f"""
   ცდომილება selected N_τ=295-თან: {abs(ratio_tau_index - 295)/295*100:.2f}%
   შესაძლო მიზეზები:
     1. ALPHA_NL = 0.5 ფიტირებული პარამეტრია — არ ზუსტი
     2. Φ_C = 2.35 — ფიტირებული
     3. 3D რადიალური ODE რეგულარიზაცია r=0-ზე
     4. κ² ფარდობა დონე-ზუსტი, selected N_τ=295 კი შესაძლოა
        დამატებით 3D→1D/Floquet შესწორებას ითხოვდეს

   თუ ეს 0.87% ცდომილება ფიზიკური არ არის, მაშინ ზუსტი 295
   არ გამოდის ამ static κ² დიაგნოსტიკიდან.
   59 არის selected ratio 295/5; raw 3D ფარდობა იძლევა
   {ratio_tau_over_e:.4f}.

   თუ სრული ანალიზი დარჩება N_τ≈292-ზე, მაშინ 295 რჩება
   MASS/Mathieu selected index-ად, ხოლო bridge ღიაა.
""")

# ===================================================================
# E. მიუონის წინასწარი მოლოდინი
# ===================================================================

print("-" * 72)
print(" E. მიუონი: N_μ ≈ 72?")
print("-" * 72)

print(f"\n   N_μ (3D-დან) = 5 × {ratio_mu_e:.4f} = {N_mu_from_k2:.2f}")
print(f"   N_μ (ემპირ.) = 72")
err_mu2 = abs(N_mu_from_k2 - 72) / 72 * 100
print(f"   ცდომილება: {err_mu2:.2f}%")

if err_mu2 > 10:
    print(f"""
   !! დიდი ცდომილება მიუონზე !!
   ე.ი. მიუონი ℓ=0,n=1 არ არის ზუსტი მიკუთვნება.
   ან 3D პულსონი ვერ ხატავს ამ ფარდობას.
""")
elif err_mu2 < 5:
    print(f"   ✓ მიუონიც სწორ ადგილასაა!")

# ===================================================================
# F. მოდელი: 5·(κ²_i/κ²_e) → b_{N_i}/b_{N_e} → m_i/m_e
# ===================================================================

print("-" * 72)
print(" F. ზუსტი ტესტი: მასა სრულად 3D-დან?")
print("-" * 72)

N_tau_int = int(round(N_tau_from_k2))
N_mu_int = int(round(N_mu_from_k2))

b5 = float(mathieu_b(5, Q))
b_tau = float(mathieu_b(N_tau_int, Q))
b_mu = float(mathieu_b(N_mu_int, Q))

m_tau_over_e_pred = b_tau / b5
m_mu_over_e_pred = b_mu / b5

m_tau_obs = 1776.86 / 0.51099895
m_mu_obs = 105.6583755 / 0.51099895

print(f"\n   3D გამოთვლა:  N_τ ≈ {N_tau_int}, N_μ ≈ {N_mu_int}")
print(f"   Mathieu:       b_{N_tau_int}/b_5 = {m_tau_over_e_pred:.1f}")
print(f"                  b_{N_mu_int}/b_5 = {m_mu_over_e_pred:.2f}")
print(f"   დაკვირვება:    m_τ/m_e = {m_tau_obs:.1f}")
print(f"                  m_μ/m_e = {m_mu_obs:.2f}")

err_tau_mass = abs(m_tau_over_e_pred - m_tau_obs) / m_tau_obs * 100
err_mu_mass = abs(m_mu_over_e_pred - m_mu_obs) / m_mu_obs * 100
print(f"\n   ცდომილება მასაში:")
print(f"     m_τ/m_e: {err_tau_mass:.2f}%")
print(f"     m_μ/m_e: {err_mu_mass:.2f}%")

# ===================================================================
# G. დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" G. დასკვნა")
print("=" * 72)

print(f"""
   **ახალი ჰიპოთეზა რომელიც ახლა შემოწმდა:**
   ─────────────────────────────────────────
   3D პულსონის ბმული მდგომარეობა (ℓ, n) →
   κ² = 1−ω² ფარდობა →
   1D Mathieu ინდექსი N (ფარდობით) →
   Mathieu b_N(q) → მასა

   რიცხვითი შედეგი:
   ─────────────────
   κ²_τ/κ²_e = {ratio_tau_e:.2f}       (selected N_τ = 295,
                                      ცდომ. {abs(ratio_tau_e-295)/295*100:.2f}%)
   κ²_μ/κ²_e = {ratio_mu_e:.2f}        (selected N_μ = 72,
                                      ცდომ. {abs(ratio_mu_e-72)/72*100:.1f}%)

   Phase 9 ჰიპოთეზა N_i = 5·(κ²_i/κ²_e):
     N_τ pred = {5*ratio_tau_e:.0f} (vs selected 295,
                  ცდომ. {abs(5*ratio_tau_e-295)/295*100:.1f}%)
     N_μ pred = {5*ratio_mu_e:.0f} (vs selected 72,
                  ცდომ. {abs(5*ratio_mu_e-72)/72*100:.1f}%)
   ე.ი. ეს ჰიპოთეზა (`5·κ-ratio`) ჩავარდა.

   სასარგებლო შედეგი:
     κ²_τ/κ²_e = {ratio_tau_e:.2f} ≈ selected N_τ = 295
     0.87%-ით — ეს არის τ-ინდექსის სკალის რიცხვითი მინიშნება
     (Phase 10-ში გამოყენებული `(2ℓ+1)·κ²/κ²_e` ფორმულით).
   ეს არ არის 59-ის (=295/5) პირდაპირი გამოყვანა.
""")

print("=" * 72)
print(" Phase 9 — დასრულდა")
print("=" * 72)
