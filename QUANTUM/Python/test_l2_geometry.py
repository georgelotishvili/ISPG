"""Compute the l-dependent geometric correction factor.

The 4/pi correction works for l=0 modes (tau, muon) vs l=2 (electron).
But l=2 has angular structure (Y_2^0), so its "effective cross-section"
differs from a circle.

The correction factor for each mode involves the angular overlap integral:
  f(l) = integral of |Y_l^m|^2 weighted by the background potential.

For the energy ratio, the relevant quantity is how the angular distribution
of each mode affects the 1D -> 3D volume conversion.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.special import sph_harm
from scipy.integrate import dblquad, quad
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

PI = np.pi
ALPHA = 0.5

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


print("=" * 70)
print("  l=2 Geometric Correction Factor")
print("=" * 70)

print("\n--- Angular integrals for spherical harmonics ---\n")

for l in range(4):
    m = 0
    def integrand_Y2(theta):
        Y = sph_harm(m, l, 0, theta).real
        return Y**2 * np.sin(theta)

    norm, _ = quad(integrand_Y2, 0, PI)
    norm *= 2 * PI
    print(f"  l={l}: integral |Y_{l}^0|^2 dOmega = {norm:.6f} (should be 1)")

    def integrand_Y4(theta):
        Y = sph_harm(m, l, 0, theta).real
        return Y**4 * np.sin(theta)

    y4, _ = quad(integrand_Y4, 0, PI)
    y4 *= 2 * PI
    print(f"         integral |Y_{l}^0|^4 dOmega = {y4:.6f}")
    print(f"         effective solid angle = 1/y4 = {1/y4:.4f} sr"
          f" (full sphere = {4*PI:.4f})")
    eff_fraction = 1.0 / (y4 * 4 * PI)
    print(f"         fraction of sphere = {eff_fraction:.4f}")
    print()

print("--- Angular RMS radius for each l ---\n")

for l in range(4):
    m = 0
    def integrand_sin2(theta):
        Y = sph_harm(m, l, 0, theta).real
        return Y**2 * np.sin(theta)**3

    sin2_avg, _ = quad(integrand_sin2, 0, PI)
    sin2_avg *= 2 * PI

    def integrand_cos2(theta):
        Y = sph_harm(m, l, 0, theta).real
        return Y**2 * np.cos(theta)**2 * np.sin(theta)

    cos2_avg, _ = quad(integrand_cos2, 0, PI)
    cos2_avg *= 2 * PI

    print(f"  l={l}: <sin^2 theta> = {sin2_avg:.6f}"
          f"  <cos^2 theta> = {cos2_avg:.6f}")

print("\n--- Geometric correction factors ---\n")

y4_values = {}
for l in range(4):
    m = 0
    def integrand_Y4_l(theta, l_val=l):
        Y = sph_harm(0, l_val, 0, theta).real
        return Y**4 * np.sin(theta)
    y4, _ = quad(integrand_Y4_l, 0, PI)
    y4 *= 2 * PI
    y4_values[l] = y4

g0 = y4_values[0]
print(f"  Reference: l=0 has |Y_0^0|^4 integral = {g0:.6f}")
print(f"  (Y_0^0 = 1/sqrt(4pi) = {1/np.sqrt(4*PI):.6f})")
print(f"  g0 = (1/4pi)^2 * 4pi = 1/(4pi) = {1/(4*PI):.6f}")
print()

for l in range(4):
    ratio = y4_values[l] / y4_values[0]
    print(f"  l={l}: g(l)/g(0) = {ratio:.6f}")

print("\n--- Apply l-dependent correction to mass ratios ---\n")

om_tau = 0.6639
om_mu = 0.9704
om_e = 0.9990

def E_1d(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*Om**2 + 1)

E_t = E_1d(om_tau)
E_m = E_1d(om_mu)
E_el = E_1d(om_e)

print(f"  Raw E_1D: tau={E_t:.6f}  mu={E_m:.6f}  e={E_el:.6e}")
print(f"  Raw ratios: 1 : {E_m/E_el:.1f} : {E_t/E_el:.0f}")

g_ratio_l0_l2 = y4_values[0] / y4_values[2]
print(f"\n  g(l=0)/g(l=2) = {g_ratio_l0_l2:.6f}")
print(f"  4/pi = {4/PI:.6f}")
print(f"  4/pi * g(l=0)/g(l=2) = {4/PI * g_ratio_l0_l2:.6f}")

corrections = {}
for name, label in [("simple 4/pi", None), ("Y4 ratio", "y4"),
                     ("sqrt Y4 ratio", "sqy4")]:
    if label is None:
        corr_tau = 4/PI
        corr_mu = 4/PI
        corr_e = 1.0
    elif label == "y4":
        corr_tau = 1.0 / y4_values[0]
        corr_mu = 1.0 / y4_values[0]
        corr_e = 1.0 / y4_values[2]
    elif label == "sqy4":
        corr_tau = 1.0 / np.sqrt(y4_values[0])
        corr_mu = 1.0 / np.sqrt(y4_values[0])
        corr_e = 1.0 / np.sqrt(y4_values[2])

    m_t = E_t * corr_tau
    m_m = E_m * corr_mu
    m_el = E_el * corr_e

    Q = koide(m_t, m_m, m_el)
    r_mu = m_m / m_el
    r_tau = m_t / m_el

    err_mu = abs(r_mu - m_mu/m_e) / (m_mu/m_e) * 100
    err_tau = abs(r_tau - m_tau/m_e) / (m_tau/m_e) * 100

    print(f"\n  {name}:")
    print(f"    corrections: tau={corr_tau:.4f}  mu={corr_mu:.4f}  e={corr_e:.4f}")
    print(f"    ratios: 1 : {r_mu:.1f} : {r_tau:.0f}")
    print(f"    Q = {Q:.8f}  |Q-2/3| = {abs(Q-2/3):.2e}")
    print(f"    mu/e error: {err_mu:.2f}%  tau/e error: {err_tau:.2f}%")
    print(f"    Target: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.0f}")

print("\n--- Find optimal correction factor for l=2 ---\n")

target_mu_e = m_mu / m_e
target_tau_e = m_tau / m_e

needed_factor_tau = target_tau_e / (E_t / E_el)
needed_factor_mu = target_mu_e / (E_m / E_el)

print(f"  To match tau/e exactly: need factor = {needed_factor_tau:.6f}")
print(f"  To match mu/e exactly: need factor = {needed_factor_mu:.6f}")
print(f"  4/pi = {4/PI:.6f}")
print(f"  tau factor / (4/pi) = {needed_factor_tau / (4/PI):.6f}")
print(f"  mu factor / (4/pi) = {needed_factor_mu / (4/PI):.6f}")

print(f"\n  If tau and mu use factor 4/pi, electron needs:")
f_e_for_tau = (E_t / E_el) * (4/PI) / target_tau_e
f_e_for_mu = (E_m / E_el) * (4/PI) / target_mu_e
print(f"    From tau/e: electron correction = {f_e_for_tau:.6f}")
print(f"    From mu/e: electron correction = {f_e_for_mu:.6f}")

print(f"\n  Interpretation:")
print(f"    l=0 modes (tau, mu): correction = 4/pi = {4/PI:.4f}")
print(f"    l=2 mode (electron): correction = ~{(f_e_for_tau+f_e_for_mu)/2:.4f}")
print(f"    Ratio l2/l0 = ~{(f_e_for_tau+f_e_for_mu)/2 / (4/PI):.4f}")

angular_ratio = y4_values[2] / y4_values[0]
print(f"\n    g(l=2)/g(l=0) from Y4 integrals = {angular_ratio:.4f}")
print(f"    sqrt(g(l=2)/g(l=0)) = {np.sqrt(angular_ratio):.4f}")
print(f"    (2l+1)/(2*0+1) = {(2*2+1)/(2*0+1):.4f}")
print(f"    1/(2l+1) ratio: (1/5)/(1/1) = {1/5:.4f}")

print(f"\n--- Final: best l-dependent correction ---\n")

for f_l2_factor in [1.0, angular_ratio, np.sqrt(angular_ratio),
                     1/5, 1/3, 0.8, needed_factor_tau/(4/PI),
                     needed_factor_mu/(4/PI)]:
    corr_l0 = 4/PI
    corr_l2 = (4/PI) * f_l2_factor

    m_t = E_t * corr_l0
    m_m = E_m * corr_l0
    m_el = E_el * corr_l2

    Q = koide(m_t, m_m, m_el)
    r_mu = m_m / m_el
    r_tau = m_t / m_el

    err_mu = abs(r_mu - m_mu/m_e) / (m_mu/m_e) * 100
    err_tau = abs(r_tau - m_tau/m_e) / (m_tau/m_e) * 100

    print(f"  f_l2={f_l2_factor:.4f}: 1:{r_mu:.1f}:{r_tau:.0f}"
          f"  Q={Q:.6f}  err_mu={err_mu:.2f}%  err_tau={err_tau:.2f}%")
