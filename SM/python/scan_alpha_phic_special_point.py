"""
ISPG — Phase 15: სპეციალური (α, Φ_c) წერტილის ძიება
================================================================

Phase 10-ში: (α=0.5, Φ_c=2.35) → N_τ = 292.4, ცდომ. 0.9%
ეს მახეა: α და Φ_c ფიტებია. სად არის სპეციალური წერტილი?

საძიებო კითხვები:
  A. არის თუ არა (α*, Φ_c*), სადაც N_τ = ზუსტად 295?
  B. თუ კი — დამოკიდებულია თუ არა ეს წერტილი
     (i) პულსონის მოქმედების ექსტრემუმზე?
     (ii) ენერგიის მინიმუმზე?
     (iii) თვითთანმიმდევრობის პირობაზე?
  C. თუ არა — 0.9% ცდომ. ფიზიკური ხმაურია?

მეთოდი:
  (α, Φ_c) ბადე: α ∈ [0.3, 0.8], Φ_c ∈ [1.8, 2.8]
  თითოეულზე:
     - ვპოულობთ პულსონი Ω
     - ვითვლით კავიტის ყველა ℓ=0,1,2 მდგომარეობას
     - ვითვლით N_τ = (κ²_τ × 5) / (κ²_e × 1) · 5 = 5·κ²_τ/κ²_e [თუ e=ℓ=2]
     - N_τ = 295 ხაზის ძიება
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

def oscillon_rhs(r, y, Omega, alpha):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]

def find_oscillon(Phi_c, alpha, r_max=40.0):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-10, atol=1e-12, max_step=0.1)
        return sol.y[0, -1]
    try:
        Omega = brentq(residual, 0.30, 0.999, xtol=1e-10)
    except Exception:
        return None, None, None
    r_eval = np.linspace(1e-10, r_max, 3000)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0], Omega

def cavity_spectrum(r, Phi_bg, alpha, ell):
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
    evals, _ = eigh_tridiagonal(diag, off_diag)
    bound = evals[evals < 1.0]
    return bound

def pulson_energy(r, Phi, Omega, alpha):
    """პულსონის მთლიანი ენერგია (Lagrangian H = T + V)"""
    dPhi = np.gradient(Phi, r)
    kin = 0.5 * dPhi**2
    osc = 0.5 * Omega**2 * Phi**2
    # უნო: V_nl = (1-(1+αΦ)e^{-αΦ}) / α² — არის პოტენციალი
    V_nl = (1.0 - (1.0 + alpha*Phi) * np.exp(-alpha*Phi)) / alpha**2
    return np.trapz(4*np.pi*r**2 * (kin + osc + V_nl), r)


print("=" * 72)
print(" Phase 15 — (α, Φ_c) სპეციალური წერტილის ძიება")
print("=" * 72)

# ===================================================================
# [1] ბადის სკანი
# ===================================================================

print("\n[1] (α, Φ_c) ბადის სკანი...")

alphas = np.linspace(0.35, 0.75, 15)
phics = np.linspace(1.9, 2.7, 15)

# შენახვა
N_tau_grid = np.full((len(alphas), len(phics)), np.nan)
N_mu_grid = np.full((len(alphas), len(phics)), np.nan)
E_grid = np.full((len(alphas), len(phics)), np.nan)
kappa_tau_grid = np.full((len(alphas), len(phics)), np.nan)
kappa_e_grid = np.full((len(alphas), len(phics)), np.nan)

for i, a in enumerate(alphas):
    for j, pc in enumerate(phics):
        r, Phi, Omega = find_oscillon(pc, a)
        if r is None:
            continue

        # ენერგია
        try:
            E_grid[i, j] = pulson_energy(r, Phi, Omega, a)
        except Exception:
            pass

        # ℓ=0 (ტაუ) და ℓ=2 (ელექტრონი)
        ev0 = cavity_spectrum(r, Phi, a, 0)
        ev2 = cavity_spectrum(r, Phi, a, 2)
        ev1 = cavity_spectrum(r, Phi, a, 1)

        if len(ev0) == 0 or len(ev2) == 0:
            continue

        kappa2_tau = 1.0 - ev0[0]  # ℓ=0, n=0
        kappa2_e = 1.0 - ev2[0]    # ℓ=2, n=0

        if kappa2_e <= 0:
            continue

        K = 1.0 / kappa2_e
        N_tau = K * 1 * kappa2_tau   # (2·0+1)·κ²_τ/κ²_e
        N_tau_grid[i, j] = N_tau
        kappa_tau_grid[i, j] = kappa2_tau
        kappa_e_grid[i, j] = kappa2_e

        # მიუონი — საუკეთესო ℓ=1 მიკუთვნება (ან ℓ=0 n=1)
        if len(ev1) > 0:
            kappa2_mu = 1.0 - ev1[0]
            N_mu_grid[i, j] = K * 3 * kappa2_mu

    print(f"   α = {a:.3f} დასრულდა ({len(phics)} Φ_c-ს სკანირება)")

# ===================================================================
# [2] სად არის N_τ = 295 ხაზი?
# ===================================================================

print("\n" + "-" * 72)
print(" [2] N_τ = 295 ხაზის მახვილი წერტილები")
print("-" * 72)

# ცდომილება 295-თან
err_grid = np.abs(N_tau_grid - 295.0) / 295.0 * 100

# იპოვე საუკეთესო
min_idx = np.unravel_index(np.nanargmin(err_grid), err_grid.shape)
i_best, j_best = min_idx
print(f"\n   საუკეთესო ფიტი:")
print(f"     α     = {alphas[i_best]:.4f}")
print(f"     Φ_c   = {phics[j_best]:.4f}")
print(f"     N_τ   = {N_tau_grid[i_best, j_best]:.3f}")
print(f"     ცდომ. = {err_grid[i_best, j_best]:.3f}%")
print(f"     κ²_τ  = {kappa_tau_grid[i_best, j_best]:.6f}")
print(f"     κ²_e  = {kappa_e_grid[i_best, j_best]:.6f}")
print(f"     E(pulson) = {E_grid[i_best, j_best]:.3f}")

# ===================================================================
# [3] სად არის ენერგიის მინიმუმი?
# ===================================================================

print("\n" + "-" * 72)
print(" [3] პულსონის ენერგიის მინიმუმი")
print("-" * 72)

# ყოველი α-თვის, იპოვე Φ_c, რომელიც მინიმიზირებს ენერგიას
print(f"\n   {'α':>8} {'Φ_c* (Emin)':>13} {'E_min':>10} {'N_τ (ამ წერტ.)':>18}")
for i, a in enumerate(alphas):
    row = E_grid[i, :]
    if np.all(np.isnan(row)):
        continue
    j_min = np.nanargmin(row)
    N_at_min = N_tau_grid[i, j_min]
    print(f"   {a:>8.4f} {phics[j_min]:>13.4f} {row[j_min]:>10.3f} "
          f"{N_at_min:>18.3f}")

# ===================================================================
# [4] სპეციფიური წერტილი: (0.5, 2.35)
# ===================================================================

print("\n" + "-" * 72)
print(" [4] მიმდინარე წერტილი (α=0.5, Φ_c=2.35)")
print("-" * 72)

# ზუსტი წერტილი
r, Phi, Omega = find_oscillon(2.35, 0.5)
ev0 = cavity_spectrum(r, Phi, 0.5, 0)
ev2 = cavity_spectrum(r, Phi, 0.5, 2)
K_ref = 1.0 / (1 - ev2[0])
N_tau_ref = K_ref * (1 - ev0[0])
E_ref = pulson_energy(r, Phi, Omega, 0.5)

print(f"\n   Ω       = {Omega:.6f}")
print(f"   κ²_τ    = {1-ev0[0]:.6f}")
print(f"   κ²_e    = {1-ev2[0]:.6f}")
print(f"   N_τ     = {N_tau_ref:.3f}")
print(f"   ცდომ. 295-თან: {abs(N_tau_ref-295)/295*100:.3f}%")
print(f"   E       = {E_ref:.3f}")

# ===================================================================
# [5] ქვებრუნების კონსტრუქცია: სპეციალური წერტილი ფიზიკურია?
# ===================================================================

print("\n" + "-" * 72)
print(" [5] სპეციალური წერტილი ფიზიკურია?")
print("-" * 72)

alpha_spec = alphas[i_best]
phic_spec = phics[j_best]

# დავაზუსტოთ რეფრაქციით
alphas_fine = np.linspace(max(0.3, alpha_spec - 0.05), min(0.8, alpha_spec + 0.05), 11)
phics_fine = np.linspace(max(1.5, phic_spec - 0.05), min(3.0, phic_spec + 0.05), 11)

print(f"\n   დაზუსტება ({alphas_fine[0]:.3f}..{alphas_fine[-1]:.3f}) ×"
      f" ({phics_fine[0]:.3f}..{phics_fine[-1]:.3f})")

best_fine = None
for a in alphas_fine:
    for pc in phics_fine:
        r, Phi, Omega = find_oscillon(pc, a)
        if r is None:
            continue
        ev0 = cavity_spectrum(r, Phi, a, 0)
        ev2 = cavity_spectrum(r, Phi, a, 2)
        if len(ev0) == 0 or len(ev2) == 0:
            continue
        k2_tau = 1 - ev0[0]
        k2_e = 1 - ev2[0]
        if k2_e <= 0:
            continue
        N_tau = k2_tau / k2_e
        err = abs(N_tau - 295)
        if best_fine is None or err < best_fine['err']:
            best_fine = dict(alpha=a, phic=pc, N_tau=N_tau, err=err,
                             Omega=Omega, k2_tau=k2_tau, k2_e=k2_e)

if best_fine:
    print(f"\n   საუკეთესო (დაზუსტებული):")
    print(f"     α*   = {best_fine['alpha']:.5f}")
    print(f"     Φ_c* = {best_fine['phic']:.5f}")
    print(f"     N_τ  = {best_fine['N_tau']:.4f}  (ცდომ. {best_fine['err']/295*100:.4f}%)")

# ===================================================================
# [6] დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" [6] დასკვნა")
print("=" * 72)

# მონაცემები
min_err = err_grid[i_best, j_best]
current_err = abs(N_tau_ref - 295) / 295 * 100

print(f"""
   ძიების შედეგი:
   ──────────────
   (α*, Φ_c*) = ({alphas[i_best]:.3f}, {phics[j_best]:.3f}) → N_τ = {N_tau_grid[i_best,j_best]:.2f}
   მიმდინარე   = (0.500, 2.350)          → N_τ = {N_tau_ref:.2f}
   სხვაობა:    {abs(alphas[i_best]-0.5):.3f} × {abs(phics[j_best]-2.35):.3f}

   ცდომილება 295-თან:
     მიმდინარე:  {current_err:.3f}%
     საუკეთესო:  {min_err:.3f}%

   ენერგიული მინიმუმის დამთხვევა N_τ = 295-ზე?
     [იხ. ცხრილი [3]] — თუ დიახ, ფორმულა ფიზიკურია
     თუ არა → 0.9% ცდომ. დარჩება ხმაურად
""")

print("=" * 72)
print(" Phase 15 — დასრულდა")
print("=" * 72)
