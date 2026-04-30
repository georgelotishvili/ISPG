"""
ISPG — Phase 8: 3D პულსონის სპექტრი — 59-ის ძიება
================================================================

Phase 7-ში ვნახეთ: 1D Mathieu b_N(q) ∝ N² ზუსტად, 59 არ გამოდის.
ახლა ვცდით 3D პულსონის სტრუქტურას:

   — პულსონის Φ(r) პროფილი (3D სფერული, α=0.5, Φ₀=2.35)
   — კავიტიის ეიგენფუნქციები ყოველ ℓ-ზე: −u'' + V_eff u = ω² u
   — ვეძებთ: ყველა ბმული მდგომარეობა (ℓ, n), ენერგიით დალაგებული

კითხვები:
   A. რამდენი ბმული მდგომარეობაა მთლიანად?
   B. რა არის (ℓ_max, n_max)? ჯამი ≈ 295? ან ≈ 59?
   C. ელექტრონი (2,0), მიუონი (0,1), ტაუ (0,0) — რამდენი ℓ-ის ყველა
      მოდი "ელექტრონამდე" (ω² < ω²_e)?
   D. სტრუქტურული მთვლელი: ∑(2ℓ+1) ბმული მდგომარეობებზე
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

# ===================================================================
# A. პულსონის პროფილი (იგივე რაც ispg_oscillon_simulation.py-ში)
# ===================================================================

ALPHA_NL = 0.5
PHI_C    = 2.35
OMEGA_EXP = 0.866

def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]


def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL,
                  r_max=40.0, Omega_lo=0.30, Omega_hi=0.999):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14,
                        max_step=0.05)
        return sol.y[0, -1]
    Omega = brentq(residual, Omega_lo, Omega_hi, xtol=1e-12)
    r_eval = np.linspace(1e-10, r_max, 6000)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


# ===================================================================
# B. კავიტიის ეიგენვალუები ბევრ ℓ-ზე
# ===================================================================

def cavity_eigenvalues(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=20):
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
    eigenvalues, eigenvectors = eigh_tridiagonal(diag, off_diag)
    n_return = min(n_want, len(eigenvalues))
    return eigenvalues[:n_return]


# ===================================================================
# მთავარი სკანი
# ===================================================================

print("=" * 72)
print(" Phase 8 — 3D პულსონის სპექტრი (ℓ, n) — 59-ის ძიება")
print("=" * 72)

print("\n[1] პულსონის პროფილის ამოხსნა...")
r, Phi, Omega = find_oscillon()
print(f"    Ω = {Omega:.6f}")
print(f"    r ∈ [0, {r[-1]:.1f}],  N_points = {len(r)}")

print("\n[2] კავიტიის ეიგენვალუები ℓ = 0..30-ზე...")
print("    V_eff(∞) = 1 → ბმული: ω² < 1")

ELL_MAX = 30
N_PER_ELL = 30

all_states = []   # (ell, n, omega2)
for ell in range(ELL_MAX + 1):
    evals = cavity_eigenvalues(r, Phi, ell=ell, n_want=N_PER_ELL)
    bound = evals[evals < 1.0]
    for n_idx, ev in enumerate(bound):
        all_states.append((ell, n_idx, ev))

all_states.sort(key=lambda s: s[2])

print(f"\n    ჯამი ბმული მდგომარეობა: {len(all_states)}")
print(f"    ℓ სპექტრი:")
for ell in range(ELL_MAX + 1):
    n_bound = sum(1 for e, n, ev in all_states if e == ell)
    if n_bound > 0:
        print(f"      ℓ = {ell:2d}:  {n_bound} ბმული მდგომარეობა")

# ===================================================================
# C. დეგენერაციით (2ℓ+1) მთვლელი
# ===================================================================

print("\n" + "-" * 72)
print(" C. დეგენერაციით მთვლელი: ∑ (2ℓ+1)")
print("-" * 72)

total_deg = sum(2 * s[0] + 1 for s in all_states)
print(f"\n    ∑ (2ℓ+1) ყველა ბმული მდგომარეობაზე = {total_deg}")
print(f"    გასაცოცხი რიცხვები: 59, 72, 295, 5, 14, 72+5=77")

# ვცდით ცალკე ℓ-ების ლიმიტებს
print(f"\n    ზემოდან (ω²<threshold):")
for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    below = [s for s in all_states if s[2] < threshold]
    total = sum(2 * s[0] + 1 for s in below)
    n_states = len(below)
    print(f"      ω² < {threshold:.2f}:  {n_states:3d} მდგომარეობა, "
          f"∑(2ℓ+1) = {total:4d}")

# ===================================================================
# D. ელექტრონამდე — რამდენი მდგომარეობა?
# ===================================================================

print("\n" + "-" * 72)
print(" D. ელექტრონის მდგომარეობის ენერგია")
print("-" * 72)

# Paper: ელექტრონი = (ℓ=2, n=0); მიუონი = (ℓ=0, n=1); ტაუ = (ℓ=0, n=0)
e_state = [s for s in all_states if s[0] == 2 and s[1] == 0]
mu_state = [s for s in all_states if s[0] == 0 and s[1] == 1]
tau_state = [s for s in all_states if s[0] == 0 and s[1] == 0]

if e_state and mu_state and tau_state:
    print(f"\n    ტაუ   (ℓ=0, n=0):  ω² = {tau_state[0][2]:.6f}, ω = {np.sqrt(tau_state[0][2]):.6f}")
    print(f"    მიუონი (ℓ=0, n=1):  ω² = {mu_state[0][2]:.6f}, ω = {np.sqrt(mu_state[0][2]):.6f}")
    print(f"    ელექტრ. (ℓ=2, n=0): ω² = {e_state[0][2]:.6f}, ω = {np.sqrt(e_state[0][2]):.6f}")

    e_energy = e_state[0][2]
    below_e = [s for s in all_states if s[2] < e_energy]
    total_below_e = sum(2 * s[0] + 1 for s in below_e)
    print(f"\n    მდგომარეობები ელექტრონზე ქვემოთ (ω² < {e_energy:.4f}):")
    print(f"      ცალკე: {len(below_e)}")
    print(f"      ∑(2ℓ+1) დეგენერაციით: {total_below_e}")

# ===================================================================
# E. ω² → m რა პროპორციონალობა?
# ===================================================================

print("\n" + "-" * 72)
print(" E. m_τ/m_e ფარდობა 3D სპექტრიდან")
print("-" * 72)

if e_state and mu_state and tau_state:
    # Paper's formula: m ∝ E_1D = ω² κ (binding energy-like)
    # or m ∝ ω (ISPG_Quantum)
    # or m ∝ κ² = 1 - ω²
    om2_e, om2_mu, om2_tau = e_state[0][2], mu_state[0][2], tau_state[0][2]
    om_e, om_mu, om_tau = np.sqrt(om2_e), np.sqrt(om2_mu), np.sqrt(om2_tau)
    k2_e = 1 - om2_e
    k2_mu = 1 - om2_mu
    k2_tau = 1 - om2_tau

    # დაკვირვებული
    obs_mu_e = 105.6583755 / 0.51099895
    obs_tau_e = 1776.86 / 0.51099895

    print(f"\n    დაკვირვებული: m_μ/m_e = {obs_mu_e:.3f}, m_τ/m_e = {obs_tau_e:.1f}")
    print(f"\n    სხვადასხვა მასის ფორმულის ტესტი:")
    print(f"    ─────────────────────────────────")
    print(f"    m ∝ ω:        m_τ/m_e = {om_tau/om_e:.4f}   (დაკვირვ.: {obs_tau_e:.1f})")
    print(f"    m ∝ ω²:       m_τ/m_e = {om2_tau/om2_e:.4f}   (დაკვირვ.: {obs_tau_e:.1f})")
    print(f"    m ∝ κ²=1−ω²:  m_τ/m_e = {k2_tau/k2_e:.4f}   (დაკვირვ.: {obs_tau_e:.1f})")
    print(f"    m ∝ κ:         m_τ/m_e = {np.sqrt(k2_tau)/np.sqrt(k2_e):.4f}   (დაკვირვ.: {obs_tau_e:.1f})")
    print(f"    m ∝ ω²·κ:     m_τ/m_e = {(om2_tau*np.sqrt(k2_tau))/(om2_e*np.sqrt(k2_e)):.4f}   (დაკვირვ.: {obs_tau_e:.1f})")
    print(f"    m ∝ κ⁴:       m_τ/m_e = {k2_tau**2/k2_e**2:.4f}   (დაკვირვ.: {obs_tau_e:.1f})")

# ===================================================================
# F. 3D დონეების დათვლა ტაუმდე
# ===================================================================

print("\n" + "-" * 72)
print(" F. 3D დონეების დათვლა ტაუ-ზე")
print("-" * 72)

if e_state and tau_state:
    tau_energy = tau_state[0][2]
    # "ტაუს ქვემოთ" — სინამდვილეში არაფერია, ტაუ თავად ყველაზე ღრმა მდგომარეობაა
    # მაგრამ რამდენი მდგომარეობაა ელექტრონსა და ტაუს შორის?
    e_energy = e_state[0][2]
    between = [s for s in all_states if tau_energy <= s[2] <= e_energy]
    total_between = sum(2 * s[0] + 1 for s in between)
    print(f"\n    მდგომარეობები ელექტრონი-ტაუ შუალედში:")
    print(f"      ცალკე: {len(between)}")
    print(f"      ∑(2ℓ+1): {total_between}")

# ===================================================================
# G. 59-ის ძიება პირდაპირ
# ===================================================================

print("\n" + "-" * 72)
print(" G. 59-ის პირდაპირი ძიება სპექტრის ინდექსებში")
print("-" * 72)

# სრული დეგენერაციებით სიახლოვის შემოწმება
print(f"\n    სულ = {len(all_states)}, ∑(2ℓ+1) = {total_deg}")

# სხვადასხვა ამონარჩევის "რაოდენობა"
print(f"\n    სკანი თრეშოლდებით:")
targets = [5, 59, 72, 77, 295]
for threshold_pct in np.linspace(0.1, 1.0, 50):
    below = [s for s in all_states if s[2] < threshold_pct]
    n = len(below)
    n_deg = sum(2*s[0]+1 for s in below)
    for t in targets:
        if n == t or n_deg == t:
            print(f"      ω² < {threshold_pct:.3f}: {n} ცალი, ∑(2ℓ+1) = {n_deg}  ← {t}!")

# ცალკე ელექტრონზე დაბალი ℓ-ების რაოდენობა
print(f"\n    ცალკე ℓ-ებზე ბმული მდგომარეობების ჯამი:")
cumul = 0
cumul_deg = 0
for ell in range(ELL_MAX + 1):
    n_ell = sum(1 for s in all_states if s[0] == ell)
    cumul += n_ell
    cumul_deg += n_ell * (2*ell + 1)
    mark = ""
    if cumul == 59 or cumul_deg == 59: mark += " ← 59!"
    if cumul == 72 or cumul_deg == 72: mark += " ← 72!"
    if cumul == 295 or cumul_deg == 295: mark += " ← 295!"
    if cumul == 5 or cumul_deg == 5: mark += " ← 5!"
    print(f"      ℓ ≤ {ell:2d}:  N_ცალკე = {cumul:3d},  ∑(2ℓ+1) = {cumul_deg:4d}{mark}")

# ===================================================================
# H. შედეგი
# ===================================================================

print("\n" + "=" * 72)
print(" H. შედეგი")
print("=" * 72)

print(f"""
   რა ვნახეთ:
   ────────────
   3D პულსონის ბმული მდგომარეობის ჯამი:
     — ცალკე (n, ℓ) წყვილების რაოდენობა: {len(all_states)}
     — დეგენერაციით ∑(2ℓ+1): {total_deg}

   ელექტრონი, მიუონი, ტაუ — გავასრულეთ:
     — ტაუ   (ℓ=0, n=0): ყველაზე ღრმა
     — მიუონი (ℓ=0, n=1): მეორე ℓ=0 მდგომარეობა
     — ელექტრ. (ℓ=2, n=0): ℓ=2 ფუნდამენტი

   59-ის კავშირი:
     ── თუ რიცხვი 59 სპექტრის სტრუქტურული ინდექსია, იგი ჩანს
        სულ დეგენერაციებში ან დონეების რაოდენობაში.
""")

print("=" * 72)
print(" Phase 8 — დასრულდა")
print("=" * 72)
