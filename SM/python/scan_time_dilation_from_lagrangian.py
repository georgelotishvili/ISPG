"""
ISPG — Phase 14: დროის ფაქტორი ISPG ლაგრანჟიანიდან
================================================================

ცდა: Phase 13-ში ფიტით ვიპოვე c ≈ 1/113 კოეფიციენტი.
ახლა შევამოწმოთ — ISPG-ის ლაგრანჟიანიდან ბუნებრივად გამოდის ეს რიცხვი?

ქცევა:
   1. პულსონის ენერგიული სიმჭიდროვე ρ(r):
      ρ(r) = (1/4)·[ω²·Φ² + (dΦ/dr)²] + V_nonlin(Φ)
      (დროით-გასაშუალებული ოსცილაცია)

   2. ფლატ-ფონი ნიუტონური პოტენციალი Φ_N(r):
      ∇²Φ_N = 4π·G·ρ
      სფერული სიმეტრიით.

   3. დროის შესწორება თითოეულ მოდში:
      F_mode = ⟨1 + Φ_N⟩_|u|²  (სუსტი ველი)
      ან
      F_mode = ⟨exp(Φ_N)⟩_|u|²

   4. ფარდობა F_τ/F_e — ემთხვევა თუ არა Phase 13-ის 1.0089?
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
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


def cavity_eigenpair(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_idx=0):
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
    eigenvalues, eigenvectors = eigh_tridiagonal(diag, off_diag)
    u_full = np.zeros(N)
    u_full[1:-1] = eigenvectors[:, n_idx]
    norm = np.sqrt(np.trapz(u_full**2, r))
    u_full /= norm
    return float(eigenvalues[n_idx]), u_full


print("=" * 72)
print(" Phase 14 — დროის ფაქტორი ISPG ლაგრანჟიანიდან")
print("=" * 72)

print("\n[1] პულსონი + კავიტი...")
r, Phi, Omega = find_oscillon()
print(f"    Ω = {Omega:.6f}, Φ_c = {Phi[0]:.4f}")

om2_tau, u_tau = cavity_eigenpair(r, Phi, ell=0, n_idx=0)
om2_mu,  u_mu  = cavity_eigenpair(r, Phi, ell=0, n_idx=1)
om2_e,   u_e   = cavity_eigenpair(r, Phi, ell=2, n_idx=0)

# ===================================================================
# [2] ენერგიული სიმჭიდროვე ρ(r)
# ===================================================================

print("\n" + "-" * 72)
print(" [2] პულსონის ენერგიული სიმჭიდროვე ρ(r)")
print("-" * 72)

# dΦ/dr რიცხვითად
dPhi_dr = np.gradient(Phi, r)

# დროით-გასაშუალებული: ⟨cos²(ωt)⟩ = 1/2
# ρ = (1/4)[ω²Φ² + (dΦ/dr)²] + V_nonlin
# V_nonlin = ∫ [Φ - Φe^{-αΦ}] dΦ = (1/2)Φ² - (1-αΦ-...)/α²
# სიმარტივისთვის — კინეტიკური ნაწილი:
rho_kinetic = 0.25 * (Omega**2 * Phi**2 + dPhi_dr**2)

# ნონ-ლინეარული ნაწილი (პოტენციალი)
# V(Φ) = (1/2)Φ² - (e^{-αΦ}(αΦ + 1) - 1)/α² ... რთული ფორმულა
# მარტივად: უშუალოდ მთავარი კვადრატიული + ნონ-ლინ.
V_nonlin = 0.5 * Phi**2 * (1 - np.exp(-ALPHA_NL * Phi))

rho = rho_kinetic + V_nonlin

print(f"\n   ρ(r=0)    = {rho[0]:.4f}")
print(f"   ρ(r≈1)    = {rho[np.argmin(np.abs(r-1.0))]:.4f}")
print(f"   ρ(r≈5)    = {rho[np.argmin(np.abs(r-5.0))]:.4f}")
print(f"   ρ(r≈10)   = {rho[np.argmin(np.abs(r-10.0))]:.4f}")

# მთლიანი მასა
M_total = np.trapz(4 * np.pi * r**2 * rho, r)
print(f"\n   M_total = ∫4π r²ρ dr = {M_total:.4f}  (ნატურალური ერთეულებში)")

# ===================================================================
# [3] ნიუტონური პოტენციალი Φ_N(r) — Poisson-ის ამოხსნა
# ===================================================================

print("\n" + "-" * 72)
print(" [3] ნიუტონური პოტენციალი Φ_N(r)")
print("-" * 72)

# სფერულ სიმეტრიაში:
# Φ_N(r) = -4πG [ (1/r) ∫₀^r r'² ρ(r') dr' + ∫_r^∞ r' ρ(r') dr' ]
# ნატურ. ერთ. (G=1):

# კუმულირებული მასა შიგნით r-ით
integrand1 = r**2 * rho
M_inside = cumulative_trapezoid(integrand1, r, initial=0)

# გარე ინტეგრალი (r-დან r_max-მდე)
integrand2 = r * rho
outer_int = np.zeros_like(r)
for i in range(len(r)):
    outer_int[i] = np.trapz(integrand2[i:], r[i:])

# Φ_N (G=1 ნატურალურ ერთეულებში)
Phi_N = np.zeros_like(r)
Phi_N[1:] = -4 * np.pi * (M_inside[1:] / r[1:] + outer_int[1:])
Phi_N[0] = Phi_N[1]

print(f"\n   Φ_N(r=0)  = {Phi_N[0]:.6f}  (ცენტრი, მაქსიმალური)")
print(f"   Φ_N(r=1)  = {Phi_N[np.argmin(np.abs(r-1.0))]:.6f}")
print(f"   Φ_N(r=5)  = {Phi_N[np.argmin(np.abs(r-5.0))]:.6f}")
print(f"   Φ_N(r=10) = {Phi_N[np.argmin(np.abs(r-10.0))]:.6f}")
print(f"   Φ_N(r=20) = {Phi_N[np.argmin(np.abs(r-20.0))]:.6f}  (გარეთ, → 0)")

# ===================================================================
# [4] ⟨Φ_N⟩ თითოეული მოდისთვის
# ===================================================================

print("\n" + "-" * 72)
print(" [4] ⟨Φ_N⟩ — დროის დილატაცია თითოეული მოდისთვის")
print("-" * 72)

def avg_weighted(u, field, r):
    num = np.trapz(u**2 * field, r)
    den = np.trapz(u**2, r)
    return num / den

Phi_N_tau = avg_weighted(u_tau, Phi_N, r)
Phi_N_mu  = avg_weighted(u_mu,  Phi_N, r)
Phi_N_e   = avg_weighted(u_e,   Phi_N, r)

print(f"\n   ⟨Φ_N⟩_τ  = {Phi_N_tau:.6f}")
print(f"   ⟨Φ_N⟩_μ  = {Phi_N_mu:.6f}")
print(f"   ⟨Φ_N⟩_e  = {Phi_N_e:.6f}")

# დროის ფაქტორი: F = 1 + Φ_N (სუსტი ველი)
F_tau = 1 + Phi_N_tau
F_mu  = 1 + Phi_N_mu
F_e   = 1 + Phi_N_e

print(f"\n   F_τ = 1 + ⟨Φ_N⟩_τ = {F_tau:.6f}")
print(f"   F_μ = 1 + ⟨Φ_N⟩_μ = {F_mu:.6f}")
print(f"   F_e = 1 + ⟨Φ_N⟩_e = {F_e:.6f}")

ratio_FE = F_tau / F_e
ratio_FM = F_mu / F_e
print(f"\n   F_τ/F_e = {ratio_FE:.6f}")
print(f"   F_μ/F_e = {ratio_FM:.6f}")

# ===================================================================
# [5] Phase 13-თან შედარება
# ===================================================================

print("\n" + "-" * 72)
print(" [5] Phase 13-თან შედარება")
print("-" * 72)

k2_tau = 1 - om2_tau
k2_e   = 1 - om2_e
base_ratio = k2_tau / k2_e
target = 295.0
needed_correction = target / base_ratio

print(f"""
   Phase 13-ში ვიპოვეთ:
     N_τ ბაზური = {base_ratio:.4f}
     საჭირო კორექცია = {needed_correction:.5f}
     (Phase 13-ის ფიტი: F_τ/F_e = 1.0098)

   Phase 14-ში გამოთვლილი (ლაგრანჟიანიდან):
     F_τ/F_e = {ratio_FE:.6f}
""")

# რა ფაქტორში გვაქვს განსხვავება?
if abs(ratio_FE - 1) > 1e-6:
    phys_factor = (needed_correction - 1) / (ratio_FE - 1)
    print(f"   (needed - 1) / (calculated - 1) = {phys_factor:.4f}")
    print(f"   ე.ი. ფიზიკური ფაქტორი უნდა იყოს ~{phys_factor:.2f}× ნაკლები")

# შესწორებული N_τ ლაგრანჟიანიდან
N_tau_corrected = base_ratio * ratio_FE
err_lag = abs(N_tau_corrected - 295) / 295 * 100
print(f"\n   N_τ ლაგრანჟიანიდან = {base_ratio:.3f} × {ratio_FE:.6f} = {N_tau_corrected:.3f}")
print(f"   295-თან ცდომილება: {err_lag:.2f}%")

# ===================================================================
# [6] ალტერნატიული: F = exp(Φ_N)
# ===================================================================

print("\n" + "-" * 72)
print(" [6] ალტერნატიული: F = exp(Φ_N) ძლიერი ველი")
print("-" * 72)

# ცალი წერტილი-მიხედვით ავწონოთ
F_exp = np.exp(Phi_N)
F_exp_tau = avg_weighted(u_tau, F_exp, r)
F_exp_mu  = avg_weighted(u_mu,  F_exp, r)
F_exp_e   = avg_weighted(u_e,   F_exp, r)

ratio_exp = F_exp_tau / F_exp_e
N_tau_exp = base_ratio * ratio_exp
err_exp = abs(N_tau_exp - 295) / 295 * 100

print(f"\n   F = ⟨exp(Φ_N)⟩:")
print(f"     F_τ = {F_exp_tau:.6f}, F_e = {F_exp_e:.6f}")
print(f"     F_τ/F_e = {ratio_exp:.6f}")
print(f"     N_τ = {N_tau_exp:.3f}  (295-ცდომ. {err_exp:.2f}%)")

# ===================================================================
# [7] სიმძიმის ცდა: Φ_N·scale ფაქტორი
# ===================================================================

print("\n" + "-" * 72)
print(" [7] ოპტიმალური 'სკალარი-გრავიტაცია' კუპლინგი")
print("-" * 72)

# ISPG-ში შეიძლება სკალარული ველი გრავიტაციას სპეციალური
# კუპლინგით ვუკავშირდეს: F = 1 + g_s · Φ_N
# სადაც g_s არის "ISPG გრავიტაციული კუპლინგი"

from scipy.optimize import brentq as br
def err_fn(g_s):
    F_t = 1 + g_s * Phi_N_tau
    F_el = 1 + g_s * Phi_N_e
    N = base_ratio * F_t/F_el
    return N - 295.0

try:
    g_s_fit = br(err_fn, 0.001, 1000)
    print(f"\n   ოპტიმალური g_s = {g_s_fit:.4f}")
    print(f"   ე.ი. F = 1 + {g_s_fit:.2f}·Φ_N")
    print(f"   F_τ/F_e = {(1 + g_s_fit * Phi_N_tau)/(1 + g_s_fit * Phi_N_e):.6f}")
    print(f"   N_τ = 295.000 ✓")

    # Phase 13-ის კოეფიციენტი 1/113 იყო Φ-ზე
    # ახლა g_s არის Φ_N-ზე (რომელიც თვითონ ~Φ-ის ფორმითაა)
    # ფარდობა
    ratio_gs_to_Phi_N = abs(Phi_N_tau) / 1.362  # Phi_N to Phi ratio
    print(f"\n   Φ_N(avg) / Φ(avg) = {Phi_N_tau/1.362:.4f}")
    print(f"   ე.ი. g_s = {g_s_fit:.4f} × {Phi_N_tau/1.362:.4f} ≈ {g_s_fit*Phi_N_tau/1.362:.4f}")

except Exception as e:
    print(f"\n   ოპტიმიზაცია ვერ გამოვიდა: {e}")

# ===================================================================
# [8] დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" [8] დასკვნა")
print("=" * 72)

print(f"""
   რა აღმოვაჩინეთ:
   ───────────────
   1. ISPG-ის ლაგრანჟიანიდან პირდაპირ გამოთვლილი ნიუტონური
      პოტენციალი Φ_N პულსონში — მთლიანად მცირეა:
        Φ_N(0) = {Phi_N[0]:.4f}, Φ_N(∞) → 0

   2. ⟨Φ_N⟩_τ = {Phi_N_tau:.4f}, ⟨Φ_N⟩_e = {Phi_N_e:.4f}
      განსხვავება: ΔΦ_N ≈ {abs(Phi_N_tau - Phi_N_e):.4f}

   3. სუსტი ველის ფაქტორი F_τ/F_e = {ratio_FE:.4f}
      ეს იძლევა N_τ = {N_tau_corrected:.2f}
      (295 ცდომ. {err_lag:.2f}%)

   4. მოსახმარი კოეფიციენტი Phase 13-ში იყო ~1/113.
      ლაგრანჟიანის ბუნებრივი კოეფიციენტი არის {abs(Phi_N_tau):.3f}.

   შედეგი:
   ────────
   {'ლაგრანჟიანი ბუნებრივად იძლევა საჭირო შესწორებას!' if err_lag < 1 else 'ლაგრანჟიანი ვერ იძლევა — სჭირდება დამატებითი კუპლინგი g_s'}

   თუ ცდომილება > 5%: ISPG-ის ფრენკვენცია-გრავიტაცია კუპლინგი
   სტანდარტულ ნიუტონურზე სუსტია. ცალკე მექანიზმი უნდა არსებობდეს.
""")

print("=" * 72)
print(" Phase 14 — დასრულდა")
print("=" * 72)
