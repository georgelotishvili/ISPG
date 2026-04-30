"""
ISPG — Phase 16: Floquet (დროის-საშუალო) კავიტი
================================================================

საკითხი: ადრინდელ ფაზებში ვიყენებდით სტატიკურ პოტენციალს:
   V_eff(r) = (1 − αΦ(r)) · exp(−αΦ(r))

მაგრამ **პულსონი ირხევა დროში**: Φ(r, t) = Φ(r) · cos(Ωt)
ე.ი. რეალურად:
   V(r, t) = (1 − αΦ(r)cos(Ωt)) · exp(−αΦ(r)cos(Ωt))

ჰიპოთეზა (Phase 15-ის შემდეგ):
   დროის-საშუალო ⟨V⟩_t შეიძლება ოდნავ განსხვავდებოდეს სტატიკურისგან,
   რაც გამოისახება κ²_τ/κ²_e-ში 0.9%-ის რიგზე.

ტესტი:
   A. ⟨V⟩_t = (1/T) ∫ V(r, t) dt
   B. კავიტის საკუთრივ-მნიშვნელობები ⟨V⟩_t-ით
   C. ახალი N_τ = κ²_τ/κ²_e ↔ 295?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

ALPHA = 0.5
PHI_C = 2.35


def oscillon_rhs(r, y, Omega, alpha=ALPHA):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]


def find_oscillon(Phi_c=PHI_C, alpha=ALPHA, r_max=40.0):
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


def V_static(Phi, alpha=ALPHA):
    """სტატიკური პოტენციალი (ადრინდელი ფაზები)"""
    aP = alpha * Phi
    return (1.0 - aP) * np.exp(-aP)


def V_time_average(Phi, alpha=ALPHA, N_t=2048):
    """
    დროის-საშუალო პოტენციალი:
        ⟨V⟩ = (1/T) ∫₀ᵀ (1 − αΦcos(Ωt)) · exp(−αΦcos(Ωt)) dt
    """
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    cos_t = np.cos(t)
    V_sum = np.zeros_like(Phi)
    for ct in cos_t:
        aP = alpha * Phi * ct
        V_sum += (1.0 - aP) * np.exp(-aP)
    return V_sum / N_t


def V_time_var(Phi, alpha=ALPHA, N_t=2048):
    """
    დროის-დისპერსია — ფლოკეს მეორე-წესრიგის შესწორების სამაგრი
    """
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    cos_t = np.cos(t)
    V_avg = V_time_average(Phi, alpha, N_t)
    V_sq = np.zeros_like(Phi)
    for ct in cos_t:
        aP = alpha * Phi * ct
        V_t = (1.0 - aP) * np.exp(-aP)
        V_sq += (V_t - V_avg) ** 2
    return V_sq / N_t


def cavity_eigenvalues(r, V_r, ell=0, n_want=5):
    """-u'' + [V + ℓ(ℓ+1)/r²] u = ω² u"""
    N = len(r)
    dr = r[1] - r[0]
    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:] ** 2
    centrifugal[0] = centrifugal[1]
    W = V_r + centrifugal
    diag = 2.0 / dr ** 2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr ** 2
    evals, _ = eigh_tridiagonal(diag, off_diag)
    return evals[:min(n_want, len(evals))]


print("=" * 72)
print(" Phase 16 — Floquet (დროის-საშუალო) კავიტი")
print("=" * 72)

# ===================================================================
# [1] პულსონი
# ===================================================================

print("\n[1] პულსონი...")
r, Phi, Omega = find_oscillon()
print(f"   Ω       = {Omega:.6f}")
print(f"   Φ(0)    = {Phi[0]:.4f}")
print(f"   α       = {ALPHA},  Φ_c = {PHI_C}")

# ===================================================================
# [2] სტატიკური vs დროის-საშუალო პოტენციალი
# ===================================================================

print("\n" + "-" * 72)
print(" [2] V(r) სტატიკური vs დროის-საშუალო")
print("-" * 72)

V_stat = V_static(Phi)
V_avg = V_time_average(Phi)
V_var = V_time_var(Phi)

print(f"\n   {'r':>6} {'V_static':>12} {'V_avg':>12} {'ΔV/V':>10} {'σ_V':>12}")
for idx in [0, 50, 100, 200, 500, 1000]:
    if idx >= len(r):
        continue
    Vs = V_stat[idx]
    Va = V_avg[idx]
    sig = np.sqrt(V_var[idx])
    delta = (Va - Vs) / abs(Vs) * 100 if Vs != 0 else 0
    print(f"   {r[idx]:>6.2f} {Vs:>12.6f} {Va:>12.6f} {delta:>9.3f}% {sig:>12.6f}")

# ===================================================================
# [3] ორივე პოტენციალის სპექტრი
# ===================================================================

print("\n" + "-" * 72)
print(" [3] კავიტის სპექტრი — სტატიკური vs დროის-საშუალო")
print("-" * 72)

print(f"\n   {'ℓ':>3} {'n':>3} {'ω²_stat':>10} {'κ²_stat':>10} "
      f"{'ω²_avg':>10} {'κ²_avg':>10} {'Δκ²/κ²':>10}")

results = {}
for ell in [0, 1, 2]:
    ev_s = cavity_eigenvalues(r, V_stat, ell=ell, n_want=3)
    ev_a = cavity_eigenvalues(r, V_avg, ell=ell, n_want=3)
    ev_s = ev_s[ev_s < 1.0]
    ev_a = ev_a[ev_a < 1.0]
    for n_idx in range(min(len(ev_s), len(ev_a))):
        k2_s = 1.0 - ev_s[n_idx]
        k2_a = 1.0 - ev_a[n_idx]
        delta = (k2_a - k2_s) / k2_s * 100 if k2_s != 0 else 0
        print(f"   {ell:>3} {n_idx:>3} {ev_s[n_idx]:>10.6f} {k2_s:>10.6f} "
              f"{ev_a[n_idx]:>10.6f} {k2_a:>10.6f} {delta:>9.3f}%")
        results[(ell, n_idx)] = (k2_s, k2_a)

# ===================================================================
# [4] N_τ შედარება
# ===================================================================

print("\n" + "-" * 72)
print(" [4] N_τ = κ²_τ/κ²_e შედარება")
print("-" * 72)

k2_e_s, k2_e_a = results[(2, 0)]
k2_tau_s, k2_tau_a = results[(0, 0)]

N_tau_s = k2_tau_s / k2_e_s
N_tau_a = k2_tau_a / k2_e_a

err_s = abs(N_tau_s - 295) / 295 * 100
err_a = abs(N_tau_a - 295) / 295 * 100

print(f"\n   სტატიკური:         N_τ = {N_tau_s:.4f}  (ცდომ. 295-თან {err_s:.3f}%)")
print(f"   დროის-საშუალო:     N_τ = {N_tau_a:.4f}  (ცდომ. 295-თან {err_a:.3f}%)")
print(f"   ცვლილება:          ΔN_τ = {N_tau_a - N_tau_s:+.3f}")

if err_a < err_s:
    print(f"\n   ✓ **Floquet-მა გაააახლოვა 295-ს** ({err_s:.3f}% → {err_a:.3f}%)")
    improvement = (err_s - err_a) / err_s * 100
    print(f"     გაუმჯობესება: {improvement:.1f}%")
elif err_a > err_s:
    print(f"\n   ✗ Floquet-მა **გაიტანა** 295-დან ({err_s:.3f}% → {err_a:.3f}%)")
else:
    print(f"\n   — Floquet-ს ეფექტი არ ჰქონდა")

# ===================================================================
# [5] ∫V_var/Ω² — შესწორების რიგი
# ===================================================================

print("\n" + "-" * 72)
print(" [5] ფლოკეს შესწორების რიგი")
print("-" * 72)

# Kapitza-ტიპის შესწორება: V_eff ≈ ⟨V⟩ + σ²_V/(2Ω²) დიდი Ω-ზე
# აქ Ω ≈ 0.87 — არც "სწრაფი", არც "ნელი". ვნახოთ მაინც ორდერი:
kapitza_corr = np.mean(V_var) / (2 * Omega ** 2)
print(f"\n   ⟨σ²_V⟩            = {np.mean(V_var):.6f}")
print(f"   ⟨σ²_V⟩/(2Ω²)      = {kapitza_corr:.6f}")
print(f"   ⟨V_stat⟩          = {np.mean(V_stat):.6f}")
print(f"   ფარდობა:          {kapitza_corr/np.mean(np.abs(V_stat))*100:.2f}%")

# ===================================================================
# [6] ახალი N-ფორმულა სრული დეგენერაციით
# ===================================================================

print("\n" + "-" * 72)
print(" [6] N_i = (1/κ²_e) · (2ℓ+1) · κ²_i ფორმულით")
print("-" * 72)

print(f"\n   {'ℓ':>3} {'n':>3} {'N_static':>12} {'N_Floquet':>12} "
      f"{'ცდომ_stat':>12} {'ცდომ_Floq':>12}")

targets = {5: "e", 72: "μ", 295: "τ"}
for (ell, n_idx), (k2_s, k2_a) in results.items():
    N_s = (2 * ell + 1) * k2_s / k2_e_s
    N_a = (2 * ell + 1) * k2_a / k2_e_a

    best_s = min(targets, key=lambda t: abs(N_s - t))
    best_a = min(targets, key=lambda t: abs(N_a - t))
    err_best_s = abs(N_s - best_s) / best_s * 100
    err_best_a = abs(N_a - best_a) / best_a * 100

    print(f"   {ell:>3} {n_idx:>3} {N_s:>12.3f} {N_a:>12.3f} "
          f"{err_best_s:>11.3f}% {err_best_a:>11.3f}%")

# ===================================================================
# [7] დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" [7] დასკვნა")
print("=" * 72)

print(f"""
   რა აღმოვაჩინეთ:
   ──────────────
   სტატიკური:  N_τ = {N_tau_s:.3f}  (ცდომ. {err_s:.3f}%)
   Floquet:    N_τ = {N_tau_a:.3f}  (ცდომ. {err_a:.3f}%)

   Kapitza-ტიპის შესწორება: {kapitza_corr/np.mean(np.abs(V_stat))*100:.2f}% ⟨V⟩-სთან

   ინტერპრეტაცია:
   ──────────────""")

if err_a < 0.1:
    print(f"""   ✓✓ სრული წარმატება — Floquet დახურავს 0.9% ხვრელს!
   59-ის ფიზიკური ახსნა: სტატიკური κ²_τ/κ²_e + დროის-საშუალო კორექცია = 295
   **ახალი პარამეტრი არ არის საჭირო.**""")
elif err_a < err_s:
    print(f"""   ⚠ ნაწილობრივი წარმატება — Floquet-მა გააფართოვა სწორი მიმართულებით,
   მაგრამ არ დახურა სრულად. შესაძლოა:
     (a) სრული Floquet (არა მხოლოდ დროის-საშუალო) გვჭირდება
     (b) Bessel-სერიის მაღალი რიგი გასათვალისწინებელი
     (c) ფონური პულსონის დინამიკა უფრო რთულია""")
else:
    print(f"""   ✗ Floquet-მა არ იმუშავა მარტივი დროის-საშუალოთი.
   ალტერნატივა: სრული Floquet-ოპერატორი ან უფრო მარტივი ფიზიკური ეფექტი.""")

print("\n" + "=" * 72)
print(" Phase 16 — დასრულდა")
print("=" * 72)
