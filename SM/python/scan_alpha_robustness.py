"""
ISPG — α = 4 · κ²_electron რობუსტობის ცდა
================================================================

წინა ცდაში (scan_alpha_order_check.py) აღმოჩნდა რომ Φ_c=2.35, α_NL=0.5
ფონზე, ელექტრონის binding energy κ²_e = 1-ω²_e = 0.00191 მისცემს:

    α_predicted = 4 × κ²_e = 0.00765
    α_observed = 1/137.036 = 0.00730
    ცდომილება: 4.8%

ცდა: ფიზიკურია თუ კოინციდენცია?

თუ α = 4 · κ²_e **ფიზიკური კავშირია**, მაშინ:
  α_obs / κ²_e (რაც ფიქსირებულია = 137^{-1})
  უნდა იყოს ~4 სხვადასხვა ფონზე, **და** κ²_e-ც მცირე ცვლილებებზე
  რჩებოდეს ~ 1/(4·137) = 0.00183.

თუ κ²_e Φ_c-ის/α_NL-ის ფუნქციაა და "4" ცვლილებს, მაშინ კოინციდენცია.

სკანი:
  Φ_c ∈ [1.5, 3.5] (ცხრილად 7 მნიშვნელობა)
  α_NL ∈ [0.3, 0.7] (ცხრილად 5 მნიშვნელობა)

შედეგი თითოეული წერტილისთვის:
  (ω²_electron, κ²_e, α_obs/κ²_e) ცხრილი.

თუ ფარდობა α_obs/κ²_e ≈ 4 მხოლოდ Φ_c=2.35-ზე → კოინციდენცია
თუ ~4 ფართო რეგიონში → ფიზიკური
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal


def oscillon_rhs(r, y, Omega, alpha):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]


def find_oscillon(Phi_c, alpha, r_max=50.0, n_points=5000):
    def residual(Omega):
        try:
            sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                            [1e-10, r_max], [Phi_c, 0.0],
                            method='RK45', rtol=1e-11, atol=1e-13,
                            max_step=0.05)
            return sol.y[0, -1]
        except Exception:
            return np.nan

    # ძიების რეგიონი
    try:
        Omega = brentq(residual, 0.30, 0.999, xtol=1e-11)
    except (ValueError, RuntimeError):
        return None, None, None

    r_eval = np.linspace(1e-10, r_max, n_points)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-11, atol=1e-13)
    return sol.t, sol.y[0], Omega


def ell2_lowest_eig(r, Phi_bg, alpha, ell=2):
    """ℓ=2 (n=0) ეიგენვალუს დაბრუნება"""
    N = len(r)
    dr = r[1] - r[0]
    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)
    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:]**2
    centrifugal[0] = 1e12

    W = V + centrifugal
    diag     = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2
    eigvals = eigh_tridiagonal(diag, off_diag, eigvals_only=True,
                               select='i', select_range=(0, 0))
    return eigvals[0]


print("=" * 76)
print(" α = 4·κ²_e რობუსტობის ცდა: (Φ_c, α_NL) სკანი")
print("=" * 76)

ALPHA_TARGET = 1.0 / 137.036
print(f"\n სამიზნე: α = 1/137.036 = {ALPHA_TARGET:.6e}")
print(f" ჰიპოთეზა: α = 4 · κ²_electron")
print(f" ე.ი. κ²_e უნდა იყოს ≈ 1/(4·137) = {ALPHA_TARGET/4:.6e}")
print(f"      ω²_e უნდა იყოს ≈ 1 - {ALPHA_TARGET/4:.4f} = {1-ALPHA_TARGET/4:.6f}")

Phi_c_list = [1.8, 2.0, 2.15, 2.25, 2.35, 2.5, 2.8]
alpha_NL_list = [0.4, 0.45, 0.5, 0.55, 0.6]

print("\n" + "-" * 76)
print(f" {'Φ_c':>6} {'α_NL':>6} {'Ω_bg':>10} {'ω²_e':>10} {'κ²_e':>10} {'α=4κ²':>10} {'ცდ.(%)':>8} {'α/κ²':>8}")
print("-" * 76)

results = []
for alpha_NL in alpha_NL_list:
    for Phi_c in Phi_c_list:
        r, Phi_bg, Omega = find_oscillon(Phi_c, alpha_NL)
        if r is None:
            print(f" {Phi_c:>6.2f} {alpha_NL:>6.2f}  {'fail':>9}")
            continue

        try:
            omega2_e = ell2_lowest_eig(r, Phi_bg, alpha_NL, ell=2)
        except Exception as e:
            print(f" {Phi_c:>6.2f} {alpha_NL:>6.2f}  {'eig_fail':>9}")
            continue

        kappa2_e = 1.0 - omega2_e
        if kappa2_e <= 0 or not np.isfinite(kappa2_e):
            # ℓ=2 არ არის bound
            print(f" {Phi_c:>6.2f} {alpha_NL:>6.2f} {Omega:>10.5f} {omega2_e:>10.5f}  unbound")
            continue

        alpha_pred = 4 * kappa2_e
        err = 100 * abs(alpha_pred - ALPHA_TARGET) / ALPHA_TARGET
        ratio = ALPHA_TARGET / kappa2_e  # "რეალური ფაქტორი"

        mark = " ★" if err < 10 else ""
        print(f" {Phi_c:>6.2f} {alpha_NL:>6.2f} {Omega:>10.5f} {omega2_e:>10.5f} "
              f"{kappa2_e:>10.6f} {alpha_pred:>10.6f} {err:>8.2f} {ratio:>8.3f}{mark}")

        results.append((Phi_c, alpha_NL, Omega, omega2_e, kappa2_e, alpha_pred, err, ratio))

# ===================================================================
# ანალიზი
# ===================================================================

print("\n" + "=" * 76)
print(" ანალიზი")
print("=" * 76)

# ratio = α_obs/κ²_e ცვალების დიაპაზონი
if len(results) >= 3:
    ratios = [r[7] for r in results]
    ratios_arr = np.array(ratios)
    print(f"\n α_obs/κ²_e ფარდობა სკანზე (n={len(ratios)} წერტილი):")
    print(f"   მინ. = {ratios_arr.min():.3f}")
    print(f"   მაქს. = {ratios_arr.max():.3f}")
    print(f"   საშ. = {ratios_arr.mean():.3f}")
    print(f"   σ    = {ratios_arr.std():.3f}")
    print(f"   ცვალობა (max-min)/mean = {(ratios_arr.max()-ratios_arr.min())/ratios_arr.mean()*100:.1f}%")

    print(f"\n κ²_e ცვალების რიცხვი:")
    k2s = np.array([r[4] for r in results])
    print(f"   მინ. = {k2s.min():.6f}")
    print(f"   მაქს. = {k2s.max():.6f}")
    print(f"   სხვაობა {k2s.max()/k2s.min():.2f}-ჯერ")

    print(f"\n ნახვა:")
    if (ratios_arr.max() - ratios_arr.min()) / ratios_arr.mean() < 0.10:
        print(f"   ✓ ფარდობა α/κ²_e რჩება კონსტანტად (<10% ცვალობა) → **ფიზიკური**")
        print(f"     ფაქტორი ≈ {ratios_arr.mean():.2f}")
    elif k2s.max() / k2s.min() > 2 and (ratios_arr.max() - ratios_arr.min()) / ratios_arr.mean() < 0.2:
        print(f"   ~ ფარდობა ცოტა ცვლადი, მაგრამ κ²_e ბევრად ცვლადი → **ნაწილობრივ ფიზიკური**")
    else:
        print(f"   ✗ ფარდობა ცვალები, κ²_e-ც ცვლადი → **კოინციდენცია Φ_c=2.35-ზე**")

print("\n" + "=" * 76)
