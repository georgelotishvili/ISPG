"""
ISPG — ლეპტონური ჰიერარქიის გამოყვანა პირველი პრინციპებიდან
===============================================================
ჰიპოთეზა (sm_particle_scan.py-ს შედეგი):
  √³N_g = √³N_1 + (g-1)·Δ,   Δ ≈ 2.47

ინტერპრეტაცია:  N ∝ R³ (3D მოცულობა).
               R_g = R_1·(1 + (g-1)·η),  η ≈ 1.44

ტესტი:
  1. ISPG-ის ოსცილონზე გავხსნათ რადიალური Schrödinger-ი
  2. ყოველი n=0,1,2,... მდგომარეობისთვის ვიპოვოთ R_n = ⟨r²⟩^(1/2)
  3. შევამოწმოთ: R_1/R_0, R_2/R_1 ≈ 2.44?
  4. η_predicted გამოვითვალოთ → შევადაროთ 1.44-ს

წარმატების კრიტერიუმი:
  🟢 η_pred ≈ 1.44 ± 0.10  →  ფორმულა ISPG-ის ფიზიკიდან გამოყვანილი
  🟡 η_pred ≠ 1.44, მაგრამ ხაზობრივი კანონი მუშაობს  → კორექცია
  🔴 R_n არ არის ხაზობრივი →  ფორმულა ოსცილონის ფიზიკიდან არ გამოდის
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

ALPHA_NL = 0.5
PHI_C    = 2.35
ETA_OBS  = 1.44   # ემპირიული სამიზნე


def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    Phi, dPhi = y
    if r < 1e-12:
        d2 = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2 = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2]


def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL, r_max=40.0,
                  Omega_lo=0.30, Omega_hi=0.999, N_grid=4000):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14, max_step=0.05)
        return sol.y[0, -1]
    Omega = brentq(residual, Omega_lo, Omega_hi, xtol=1e-12)
    r_eval = np.linspace(1e-10, r_max, N_grid)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


def radial_eigenmodes(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=8):
    """
    გახსნა:  −u'' + [V(r) + ℓ(ℓ+1)/r²] u = ω² u
             u(r) = r·R(r)  (რედიუსული ფუნქცია)
    Dirichlet BC-ებით.  უბრუნებს ω², u_n(r) ყოველი n-თვის.
    """
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V  = (1.0 - aP) * np.exp(-aP)
    cen = np.zeros(N)
    cen[1:] = ell * (ell + 1) / r[1:]**2
    cen[0]  = cen[1]
    W = V + cen
    diag = 2.0 / dr**2 + W[1:-1]
    off  = -np.ones(N - 3) / dr**2
    evals, evecs = eigh_tridiagonal(diag, off)
    n = min(n_want, len(evals))
    return evals[:n], evecs[:, :n]


def expectation_r(r_interior, u_n):
    """⟨r²⟩ ფიზიკური ქცევის (ψ = u/r) რადიუსის მოცულობა:
       ⟨r²⟩ = ∫ ψ² r² · (4π r² dr) / ∫ ψ² · 4π r² dr
            = ∫ u_n² · r² dr / ∫ u_n² dr       (რადგან ψ=u/r, d³r = 4πr² dr,
                                                  |ψ|² d³r = u² dr)
    """
    norm = np.trapz(u_n**2, r_interior)
    if norm <= 0:
        return np.nan
    r2_exp = np.trapz(u_n**2 * r_interior**2, r_interior) / norm
    return np.sqrt(r2_exp)


def main():
    print("=" * 74)
    print("  ფაზა A — ლეპტონური ჰიერარქიის გამოყვანა")
    print("=" * 74)

    # --- ოსცილონის პროფილი ---
    print("\n[1] ოსცილონის პროფილის გახსნა (shooting)...")
    r, Phi, Omega = find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL,
                                   r_max=40.0, N_grid=6000)
    print(f"    Φ₀ = {PHI_C},  α = {ALPHA_NL}")
    print(f"    Ω = {Omega:.8f}")

    r_in = r[1:-1]  # ინტერიული წერტილები (მატრიცას იყენებს)

    # --- ყველა (ℓ, n) ბმული მოდი (ω² < 1) ---
    print("\n[1b] ყველა ბმული (ℓ, n) მოდი (ω² < 1):")
    print("    ℓ   n      ω²            ω          R = √⟨r²⟩")
    print("   ─────────────────────────────────────────────────")
    bound_modes = []
    for ell in range(4):
        evals_l, evecs_l = radial_eigenmodes(r, Phi, alpha=ALPHA_NL,
                                              ell=ell, n_want=6)
        for n in range(len(evals_l)):
            if evals_l[n] < 1.0:  # ბმული
                R = expectation_r(r_in, evecs_l[:, n])
                ω2 = evals_l[n]
                ω  = np.sqrt(ω2)
                bound_modes.append((ell, n, ω2, ω, R))
                print(f"    {ell}   {n}   {ω2:.6f}    {ω:.5f}    {R:.4f}")
    # ვინახავ paper-ის იდენტიფიკაციას:
    #   τ=(0,0),  μ=(0,1),  e=(2,0),  grav=(1,0)
    paper_ids = {(0, 0): 'τ', (0, 1): 'μ', (2, 0): 'e', (1, 0): 'grav'}
    paper_R = {}
    for (ell, n, ω2, ω, R) in bound_modes:
        if (ell, n) in paper_ids:
            paper_R[paper_ids[(ell, n)]] = R

    print("\n[1c] paper-ის ლეპტონური იდენტიფიკაცია:")
    for nm in ['e', 'μ', 'τ']:
        if nm in paper_R:
            print(f"    {nm} = (ℓ,n)={'(2,0)' if nm=='e' else ('(0,1)' if nm=='μ' else '(0,0)')}  →  R = {paper_R[nm]:.4f}")

    # --- ლოგიკური ტესტი paper-ის იდენტიფიკაციისთვის ---
    if all(k in paper_R for k in ['e', 'μ', 'τ']):
        print("\n[1d] კუბური ფარდობა paper-ის R-ებიდან:")
        Re, Rmu, Rtau = paper_R['e'], paper_R['μ'], paper_R['τ']
        print(f"    R_e = {Re:.4f},  R_μ = {Rmu:.4f},  R_τ = {Rtau:.4f}")
        print(f"    (R_μ/R_e)³ = {(Rmu/Re)**3:.2f}   (obs N_μ/N_e = 14.4)")
        print(f"    (R_τ/R_e)³ = {(Rtau/Re)**3:.2f}   (obs N_τ/N_e = 59.0)")
        print(f"    (R_τ/R_μ)³ = {(Rtau/Rmu)**3:.2f}   (obs N_τ/N_μ = 4.1)")

    # --- რადიალური მოდები ---
    print("\n[2] რადიალური მოდების გამოთვლა (ℓ=0)...")
    print("    n | ω²           ω          R_n = √⟨r²⟩    R_n/R_0")
    print("   ───┼──────────────────────────────────────────────────")

    n_want = 6
    evals, evecs = radial_eigenmodes(r, Phi, alpha=ALPHA_NL,
                                      ell=0, n_want=n_want)
    R_list = []
    for n in range(n_want):
        u_n = evecs[:, n]
        R = expectation_r(r_in, u_n)
        omega2 = evals[n]
        omega = np.sqrt(abs(omega2)) if omega2 >= 0 else float('nan')
        R_list.append(R)
        R_ratio = R / R_list[0] if R_list[0] > 0 else 1.0
        print(f"    {n} | {omega2:+.6f}  {omega:.5f}    {R:.4f}        "
              f"{R_ratio:.4f}")

    # --- ხაზობრივი ფიტი R_n = R_0·(1 + n·η_pred) ---
    R_arr = np.array(R_list)
    n_arr = np.arange(n_want)
    # ოფსეტი ფიქსირდება R_0 = R_arr[0] — ფიტი ცალი η-თი
    if R_arr[0] > 0:
        slope = np.mean((R_arr[1:] / R_arr[0] - 1) / n_arr[1:])
        eta_pred_simple = slope
    else:
        eta_pred_simple = float('nan')

    # --- სრული least-squares linear fit R_n ~ a + b·n ---
    A = np.vstack([np.ones_like(n_arr), n_arr]).T.astype(float)
    coefs, *_ = np.linalg.lstsq(A, R_arr, rcond=None)
    a_fit, b_fit = coefs
    eta_pred_fit = b_fit / a_fit

    print("\n[3] ხაზობრივი ფიტი:  R_n = a + b·n")
    print(f"    a = {a_fit:.4f}  (R_0 ფიტი)")
    print(f"    b = {b_fit:.4f}")
    print(f"    η_pred = b/a = {eta_pred_fit:.4f}")
    print(f"    η_obs  =      {ETA_OBS:.4f}  (ლეპტონებიდან, 1.88% ცდ.)")
    print(f"    ფარდობითი ცდ. η_pred-η_obs = "
          f"{abs(eta_pred_fit - ETA_OBS) / ETA_OBS * 100:.1f}%")

    # --- რეზიდუალები ---
    print(f"\n    რეზიდუალი R_n − (a + b·n):")
    for n, R in enumerate(R_list):
        pred = a_fit + b_fit * n
        print(f"      n={n}: R={R:.4f}, pred={pred:.4f}, "
              f"ცდ. = {(R-pred)/R*100:+.2f}%")

    # --- პრედიქცია N_g = (R_g / R_0)^3 · N_e ---
    print("\n[4] N-ის პრედიქცია:  N_g = N_e · (R_g/R_0)³")
    print(f"    N_e (assumed g=1, n=0) = 5")
    N_list = [5.0]
    for n in range(1, n_want):
        R_ratio = R_list[n] / R_list[0] if R_list[0] > 0 else 1.0
        N_pred = 5.0 * R_ratio**3
        N_list.append(N_pred)
        print(f"    n={n}:  R_n/R_0 = {R_ratio:.4f},  "
              f"N_pred = 5·(R_n/R_0)³ = {N_pred:.1f}")

    # --- დაკვირვებულთან შედარება ---
    obs_leptons = {0: 5, 1: 72, 2: 295}
    print("\n[5] პრედიქცია vs დაკვირვებული ლეპტონი:")
    print(f"    n  |  N_pred      N_obs      ცდომ.")
    print(f"   ────┼──────────────────────────────────")
    for n, N_obs in obs_leptons.items():
        N_p = N_list[n]
        err = abs(N_p - N_obs) / N_obs * 100
        flag = "✓" if err < 10 else ("◐" if err < 30 else "✗")
        print(f"    {n}  |  {N_p:7.1f}     {N_obs:4d}      {err:.1f}%  {flag}")

    # --- ფიგურა ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(r, Phi, 'b-', lw=1.5)
    ax.set_xlabel("r"); ax.set_ylabel("Φ(r)")
    ax.set_title(f"ოსცილონის პროფილი (Ω={Omega:.4f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_want))
    for n in range(n_want):
        u_n = evecs[:, n]
        norm_factor = np.max(np.abs(u_n))
        if norm_factor > 0:
            ax.plot(r_in, u_n / norm_factor, color=colors[n],
                    lw=1.2, label=f"n={n}")
    ax.set_xlabel("r"); ax.set_ylabel("u_n(r) (ნორმირებული)")
    ax.set_title("რადიალური ეიგენმოდები (ℓ=0)")
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[2]
    ax.plot(n_arr, R_arr, 'o-', color='darkblue', markersize=9,
            label=f"R_n (computed)")
    n_fine = np.linspace(0, n_want-1, 50)
    ax.plot(n_fine, a_fit + b_fit * n_fine, 'r--', lw=1.5,
            label=f"fit: a={a_fit:.2f}, η={eta_pred_fit:.2f}")
    # ემპირიული ხაზი η=1.44
    ax.plot(n_fine, R_arr[0] * (1 + n_fine * ETA_OBS), 'g:',
            lw=1.5, label=f"target: η=1.44")
    ax.set_xlabel("n (რადიალური)")
    ax.set_ylabel("R_n = √⟨r²⟩")
    ax.set_title(f"η_pred = {eta_pred_fit:.3f}  vs  η_obs = 1.44")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT / "lepton_hierarchy_derivation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    ფიგურა → {path}")

    # --- ვერდიქტი ---
    print("\n" + "=" * 74)
    print("  ვერდიქტი")
    print("=" * 74)
    err_eta = abs(eta_pred_fit - ETA_OBS) / ETA_OBS * 100
    if err_eta < 15:
        verdict = "🟢 ძლიერი — η გამოყვანილია ISPG-ის ოსცილონის ფიზიკიდან"
    elif err_eta < 40:
        verdict = "🟡 ნაწილობრივი — ხაზობრივი კანონი არსებობს, η-ს ცდომ. საჭიროა კორექცია"
    else:
        verdict = "🔴 ვერ გამოიყვანებოდა — დამატებითი სტრუქტურა საჭიროა"
    print(f"\n  η_pred = {eta_pred_fit:.4f},  η_obs = {ETA_OBS:.4f},  ცდ. = {err_eta:.1f}%")
    print(f"  {verdict}")
    print("=" * 74)


if __name__ == "__main__":
    main()
