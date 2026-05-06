"""
ISPG — Phase 10: κ² × (2ℓ+1) კომბინაცია
================================================================

Phase 9-ში ვნახეთ:
   κ²_τ/κ²_e = 292.4 ≈ 295 = N_τ  (აბსოლუტური, არა ფარდობა)

ეს ახლოა, მაგრამ არ ასახავს დეგენერაციას:
   ტაუ (ℓ=0): 1 მდგომარეობა
   ელექტრ. (ℓ=2): 5 მდგომარეობა (2ℓ+1)

ჰიპოთეზა:
   N_ეფექტური = κ²_i × (2ℓ_ref+1) / (2ℓ_i+1)

   სადაც ref = ელექტრონი (ℓ=2) → ფაქტორი 5/(2ℓ_i+1)

ტესტი:
   N_τ/N_e = κ²_τ/(κ²_e · 5) · 5 = κ²_τ/κ²_e
   N_τ/N_e × 5 = 59 × 5 = 295 → ე.ი. N = 5·κ²/κ²_e მუშაობს ტაუზე

   მიუონი — რომელი ℓ? სხვადასხვა მიკუთვნების ტესტი.
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
print(" Phase 10 — κ² და დეგენერაცია (2ℓ+1)")
print("=" * 72)

print("\n[1] პულსონი...")
r, Phi, Omega = find_oscillon()

# ყველა ℓ ≤ 3 მდგომარეობა
all_states = []
for ell in range(4):
    evals = cavity_eigenvalues(r, Phi, ell=ell, n_want=5)
    bound = evals[evals < 1.0]
    for n_idx, ev in enumerate(bound):
        all_states.append({
            'ell': ell,
            'n': n_idx,
            'omega2': float(ev),
            'kappa2': 1.0 - float(ev),
            'deg': 2*ell + 1,
        })

all_states.sort(key=lambda s: s['omega2'])

print("\n   ყველა ბმული მდგომარეობა (ენერგიით დალაგებული):")
print(f"   {'idx':>4} {'ℓ':>3} {'n':>3} {'ω²':>10} {'κ²':>10} {'2ℓ+1':>5} {'κ²·(2ℓ+1)':>12}")
for i, s in enumerate(all_states):
    weighted = s['kappa2'] * s['deg']
    print(f"   {i:>4} {s['ell']:>3} {s['n']:>3} {s['omega2']:>10.6f} "
          f"{s['kappa2']:>10.6f} {s['deg']:>5} {weighted:>12.6f}")

# ===================================================================
# A. რომელი კომბინაცია იძლევა 5, 72, 295-ს?
# ===================================================================

print("\n" + "-" * 72)
print(" A. რომელი მიკუთვნება იძლევა (5, 72, 295)?")
print("-" * 72)

# ელექტრონი: ℓ=2, n=0 (κ² ყველაზე მცირე)
e_state = next(s for s in all_states if s['ell'] == 2 and s['n'] == 0)
print(f"\n   ელექტრ. = (ℓ=2, n=0): κ² = {e_state['kappa2']:.6f}, (2ℓ+1) = 5")

# ტაუ: ℓ=0, n=0 (ყველაზე ღრმა)
tau_state = next(s for s in all_states if s['ell'] == 0 and s['n'] == 0)
print(f"   ტაუ    = (ℓ=0, n=0): κ² = {tau_state['kappa2']:.6f}, (2ℓ+1) = 1")

# მიუონი — ვცადოთ სხვადასხვა მიკუთვნება
print(f"\n   მიუონის შესაძლო მიკუთვნებები:")
for s in all_states:
    if s == e_state or s == tau_state:
        continue
    # N_μ ფორმულის ვარიანტები
    # V1: N = κ²/κ²_e (აბსოლუტური)
    N_abs = s['kappa2'] / e_state['kappa2']
    # V2: N = κ²·(2ℓ+1) / κ²_e·(2ℓ_e+1)
    N_weighted = (s['kappa2'] * s['deg']) / (e_state['kappa2'] * e_state['deg'])
    # V3: N = κ²/(κ²_e · (2ℓ+1)/(2ℓ_e+1))
    N_ratio = (s['kappa2'] / e_state['kappa2']) * (e_state['deg'] / s['deg'])

    print(f"\n     (ℓ={s['ell']}, n={s['n']}): κ² = {s['kappa2']:.6f}")
    print(f"       V1: κ²/κ²_e = {N_abs:.2f}          (72-თან ცდომილ. {abs(N_abs-72)/72*100:.1f}%)")
    print(f"       V2: weighted = {N_weighted:.2f}     (72-თან ცდომილ. {abs(N_weighted-72)/72*100:.1f}%)")
    print(f"       V3: ratio × 5 = {N_ratio:.2f}       (72-თან ცდომილ. {abs(N_ratio-72)/72*100:.1f}%)")

# ===================================================================
# B. ტაუზე გადამოწმება სამივე ფორმულით
# ===================================================================

print("\n" + "-" * 72)
print(" B. ტაუ — (ℓ=0, n=0) სხვადასხვა ფორმულით")
print("-" * 72)

N_tau_V1 = tau_state['kappa2'] / e_state['kappa2']
N_tau_V2 = (tau_state['kappa2'] * tau_state['deg']) / (e_state['kappa2'] * e_state['deg'])
N_tau_V3 = (tau_state['kappa2'] / e_state['kappa2']) * (e_state['deg'] / tau_state['deg'])

print(f"\n   V1 (κ²/κ²_e):              N_τ = {N_tau_V1:.2f}    (295-თან ცდომილ. {abs(N_tau_V1-295)/295*100:.2f}%)")
print(f"   V2 (weighted):              N_τ = {N_tau_V2:.2f}     (295-თან ცდომილ. {abs(N_tau_V2-295)/295*100:.2f}%)")
print(f"   V3 (ratio × 5):             N_τ = {N_tau_V3:.2f}   (295-თან ცდომილ. {abs(N_tau_V3-295)/295*100:.2f}%)")

# ===================================================================
# C. ყველაზე ახლოს რომელი ფორმულაა?
# ===================================================================

print("\n" + "-" * 72)
print(" C. საუკეთესო ფორმულა")
print("-" * 72)

print(f"""
   V1 (κ²/κ²_e) მარტივი ფარდობა:
     N_τ = 292.4  →  295 (ცდომილება 0.9%)
     N_μ — ცდომილება იცვლება მიკუთვნების მიხედვით

   **ფუნდამენტური მინიშნება:**
   ─────────────────────────────
   ნორმალიზაცია: N_e = 5. ეს "5" ჩანს ელექტრონის ℓ=2 დეგენერაციიდან!
   2ℓ+1 = 5 ზუსტად ↔ N_e = 5 ↔ 5 = ელექტრონის Mathieu ინდექსი

   ე.ი. **N = (2ℓ+1) · (რადიალური ფაქტორი)**-ტიპის კავშირი

   ტაუ (ℓ=0, n=0): 2ℓ+1 = 1, და κ² მაქსიმალური (0.559)
     → რადიალური ფაქტორი უნდა იყოს ~295

   ელექტრ. (ℓ=2, n=0): 2ℓ+1 = 5, და κ² მინიმალური (0.0019)
     → რადიალური ფაქტორი = 1 (n=0)
     → N_e = 5 × 1 = 5  ✓

   ფორმულა:
     N = (2ℓ+1) × f(n, κ²)
""")

# ===================================================================
# D. f(n, κ²) ფუნქციის ძიება
# ===================================================================

print("-" * 72)
print(" D. რადიალური ფაქტორი f(n, κ²)")
print("-" * 72)

# N_e = 5 = 5 × f(n=0, κ²=0.0019) → f = 1
# N_τ = 295 = 1 × f(n=0, κ²=0.559) → f = 295
# N_μ = 72 = (2ℓ+1) × f(n, κ²)
#   თუ μ = (ℓ=0, n=1): 72 = 1 × f(1, 0.058) → f = 72
#   თუ μ = (ℓ=1, n=0): 72 = 3 × f(0, κ²_μ) → f = 24

print(f"\n   თუ N_i = (2ℓ_i+1) × f_i:")
print(f"   ────────────────────────────")
print(f"   ელექტრ. (ℓ=2, n=0): f_e = 5/5 = 1, κ² = {e_state['kappa2']:.6f}")
print(f"   ტაუ    (ℓ=0, n=0): f_τ = 295/1 = 295, κ² = {tau_state['kappa2']:.6f}")

# იდენტიფიცირება: არის თუ არა f_τ/f_e = κ²_τ/κ²_e?
ratio_f = 295.0  # if f_e = 1
ratio_k2 = tau_state['kappa2'] / e_state['kappa2']
print(f"\n   f_τ/f_e = {ratio_f}")
print(f"   κ²_τ/κ²_e = {ratio_k2:.4f}")
print(f"   ცდომილება: {abs(ratio_f - ratio_k2)/ratio_f*100:.2f}%")

print(f"""
   **შედეგი:**
   f ∝ κ²!  კერძოდ, f_i = κ²_i/κ²_e

   ე.ი. N_i = (2ℓ_i + 1) × κ²_i / κ²_e · (1/(2ℓ_e+1)) · (2ℓ_e+1)
         ე.ი. N_i = (2ℓ_i+1) · κ²_i/κ²_e  (ელექტრონით ნორმალიზ.)

   ან უფრო სუფთად: N_i ∝ (2ℓ_i+1) · κ²_i
""")

# ===================================================================
# E. ფინალური ტესტი სრული ფორმულით
# ===================================================================

print("-" * 72)
print(" E. სრული ფორმულა: N_i = K · (2ℓ_i+1) · κ²_i")
print("-" * 72)

# ნორმალიზაცია ელექტრონზე: N_e = 5 = K × 5 × κ²_e
# → K = 1/κ²_e
K = 1.0 / e_state['kappa2']
print(f"\n   K = 1/κ²_e = {K:.3f}")

print(f"\n   ყველა ℓ-ზე N-ის გამოთვლა:")
print(f"   {'ℓ':>3} {'n':>3} {'(2ℓ+1)':>6} {'κ²':>10} {'N = K·(2ℓ+1)·κ²':>16}")
for s in all_states:
    N_pred = K * s['deg'] * s['kappa2']
    marker = ""
    if abs(N_pred - 5) / 5 < 0.05: marker = "  ← 5 (ელექტრონი)"
    if abs(N_pred - 72) / 72 < 0.1: marker = "  ← 72?"
    if abs(N_pred - 295) / 295 < 0.05: marker = "  ← 295 (ტაუ)"
    print(f"   {s['ell']:>3} {s['n']:>3} {s['deg']:>6} {s['kappa2']:>10.6f} {N_pred:>16.2f}{marker}")

# ===================================================================
# F. დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" F. დასკვნა")
print("=" * 72)

N_tau_pred = K * tau_state['deg'] * tau_state['kappa2']
ratio_pred = N_tau_pred / 5.0
mu_state = next((s for s in all_states if s['ell'] == 0 and s['n'] == 1), None)

print(f"""
   ფორმულა:  N_i = (1/κ²_e) × (2ℓ_i+1) × κ²_i

   ელექტრ. (ℓ=2, n=0): N = 5 × 1 = 5 ✓ (ნორმალიზაცია)
   ტაუ    (ℓ=0, n=0): N = {N_tau_pred:.2f}  (selected 295,
                                   ცდომ. {abs(N_tau_pred - 295)/295*100:.2f}%)

   3D კავიტიდან მიღებული ფარდობა:
   ──────────────────
   N_τ/N_e = {N_tau_pred:.2f} / 5 = {ratio_pred:.2f}
            ≈ 59 (selected ratio 295/5; ცდომილება {abs(ratio_pred - 59)/59*100:.2f}%)

   ე.ი. ზუსტი 59 = 295/5 არის selected integer ratio.
   3D კავიტიდან მისი პირდაპირი გამოყვანა (კერძოდ, ცდომილების
   0.87%-ის გასწორება) ღია რჩება.

   მიუონი დარჩა გადაუჭრელად:
     (ℓ=0, n=1) → N = {K * mu_state['deg'] * mu_state['kappa2'] if mu_state else float('nan'):.2f}  (selected 72)
""")

print("=" * 72)
print(" Phase 10 — დასრულდა")
print("=" * 72)
