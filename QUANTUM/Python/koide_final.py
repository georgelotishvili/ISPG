"""
ISPG Oscillon: Precision Koide Computation

Goal: find three oscillon frequencies (Omega_e, Omega_mu, Omega_tau)
whose TOTAL ENERGIES reproduce the lepton mass ratios AND Koide Q = 2/3.

Mass = total energy of oscillon:
  E(Omega) = (1 - Omega^2)^{3/2} * (4*Omega^2 + 1)   [1D model]

For 3D: numerical integration of the oscillon profile.
"""

import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar
from scipy.integrate import solve_ivp


# Physical lepton masses (MeV)
m_e = 0.51099895
m_mu = 105.6583755
m_tau = 1776.86

Q_exp = (m_e + m_mu + m_tau) / (np.sqrt(m_e) + np.sqrt(m_mu) + np.sqrt(m_tau))**2
print(f"Experimental Koide ratio: Q = {Q_exp:.10f}")
print(f"Target: 2/3 = {2/3:.10f}")
print(f"Lepton ratios: 1 : {m_mu/m_e:.2f} : {m_tau/m_e:.2f}")
print()


# ============================================================
#  1D MODEL: E(Omega) = beta^3 * (4*Omega^2 + 1)
# ============================================================

def E_1d(Omega):
    """1D oscillon total energy (dimensionless)."""
    if Omega <= 0 or Omega >= 1:
        return 0.0
    b2 = 1.0 - Omega**2
    return b2**1.5 * (4.0 * Omega**2 + 1.0)


def E_1d_deriv(Omega):
    """dE/dOmega for Newton's method."""
    b2 = 1.0 - Omega**2
    b = np.sqrt(b2)
    t1 = -3.0 * Omega * b * (4.0 * Omega**2 + 1.0)
    t2 = b2**1.5 * 8.0 * Omega
    return t1 + t2


Om_peak = minimize_scalar(lambda x: -E_1d(x), bounds=(0.01, 0.99),
                           method='bounded').x
E_peak = E_1d(Om_peak)
print(f"1D energy peak: Omega = {Om_peak:.8f}, E = {E_peak:.8f}")


def find_Omega_for_E(E_target, branch='R'):
    """
    Find Omega such that E_1d(Omega) = E_target.
    branch='L': Omega < Om_peak (deeply bound)
    branch='R': Omega > Om_peak (barely bound)
    """
    if E_target > E_peak or E_target <= 0:
        return None
    if branch == 'R':
        return brentq(lambda x: E_1d(x) - E_target, Om_peak, 0.999999, xtol=1e-15)
    else:
        return brentq(lambda x: E_1d(x) - E_target, 0.001, Om_peak, xtol=1e-15)


# Strategy: parameterize by Omega_tau, compute Omega_mu and Omega_e
# from the requirement that E ratios = mass ratios.
# Then check Koide.

def koide(m1, m2, m3):
    return (m1 + m2 + m3) / (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3))**2


# The energy function is universal up to a scale factor.
# If E_tau = E_1d(Om_tau), then:
#   E_mu = E_tau * (m_mu / m_tau)
#   E_e  = E_tau * (m_e  / m_tau)
# We need all three to be achievable (E < E_peak).

print("\n" + "="*70)
print("  1D MODEL: Exact lepton mass ratios")
print("="*70)
print(f"  {'Om_tau':>10} {'Om_mu':>10} {'Om_e':>10}"
      f" {'E_tau':>10} {'E_mu':>10} {'E_e':>12}"
      f" {'Q':>12} {'|Q-2/3|':>12}")

best_result = None

for Om_tau in np.linspace(0.05, Om_peak - 0.001, 500):
    E_tau = E_1d(Om_tau)
    E_mu = E_tau * m_mu / m_tau
    E_e = E_tau * m_e / m_tau

    if E_mu > E_peak or E_e > E_peak:
        continue

    Om_mu = find_Omega_for_E(E_mu, 'R')
    Om_e = find_Omega_for_E(E_e, 'R')

    if Om_mu is None or Om_e is None:
        continue

    Q = koide(E_e, E_mu, E_tau)
    Q_diff = abs(Q - 2.0/3.0)

    if best_result is None or Q_diff < best_result[0]:
        best_result = (Q_diff, Q, Om_tau, Om_mu, Om_e, E_tau, E_mu, E_e)

# Also scan right branch for tau
for Om_tau in np.linspace(Om_peak + 0.001, 0.98, 500):
    E_tau = E_1d(Om_tau)
    E_mu = E_tau * m_mu / m_tau
    E_e = E_tau * m_e / m_tau

    if E_mu > E_peak or E_e > E_peak:
        continue

    Om_mu = find_Omega_for_E(E_mu, 'R')
    Om_e = find_Omega_for_E(E_e, 'R')

    if Om_mu is None or Om_e is None:
        continue

    Q = koide(E_e, E_mu, E_tau)
    Q_diff = abs(Q - 2.0/3.0)

    if best_result is None or Q_diff < best_result[0]:
        best_result = (Q_diff, Q, Om_tau, Om_mu, Om_e, E_tau, E_mu, E_e)

Q_diff, Q, Om_t, Om_m, Om_e, E_t, E_m, E_ev = best_result
print(f"  {Om_t:10.6f} {Om_m:10.6f} {Om_e:10.6f}"
      f" {E_t:10.6f} {E_m:10.6f} {E_ev:12.8f}"
      f" {Q:12.10f} {Q_diff:12.2e}")

# Note: Q here is just the lepton Koide ratio because we fixed the mass ratios!
# The real question: does the 1D energy function REPRODUCE Q=2/3 independently?

print(f"\n  NOTE: With fixed mass ratios, Q = Q_lepton = {Q_exp:.10f}")
print(f"  This is a CONSISTENCY CHECK, not a prediction.")


# ============================================================
#  THE REAL TEST: Can E(Omega) produce Q=2/3 WITHOUT fixing ratios?
# ============================================================

print("\n" + "="*70)
print("  1D MODEL: Optimize for Q = 2/3 (free mass ratios)")
print("="*70)

def objective_koide(params):
    """Minimize |Q - 2/3| over (Om_tau, Om_mu, Om_e)."""
    Om1, Om2, Om3 = params
    if Om1 <= 0.01 or Om1 >= 0.999:
        return 10.0
    if Om2 <= 0.01 or Om2 >= 0.999:
        return 10.0
    if Om3 <= 0.01 or Om3 >= 0.999:
        return 10.0
    E1, E2, E3 = E_1d(Om1), E_1d(Om2), E_1d(Om3)
    if E1 <= 0 or E2 <= 0 or E3 <= 0:
        return 10.0
    Q = koide(E1, E2, E3)
    return (Q - 2.0/3.0)**2


# Multi-start optimization
print("\n  Optimizing with multiple starting points...")
best_opt = None

np.random.seed(42)
for _ in range(500):
    x0 = np.random.uniform(0.02, 0.98, 3)
    x0.sort()
    try:
        res = minimize(objective_koide, x0, method='Nelder-Mead',
                       options={'xatol': 1e-14, 'fatol': 1e-20, 'maxiter': 5000})
        if best_opt is None or res.fun < best_opt.fun:
            best_opt = res
    except:
        pass

Om_opt = sorted(best_opt.x)
E_opt = [E_1d(Om) for Om in Om_opt]
Q_opt = koide(*E_opt)
ratios_opt = [E / E_opt[0] for E in E_opt]

print(f"\n  BEST OPTIMIZATION RESULT:")
print(f"    Q = {Q_opt:.12f}  (target = {2/3:.12f})")
print(f"    |Q - 2/3| = {abs(Q_opt - 2/3):.2e}")
print(f"    Omega values: {Om_opt[0]:.8f}, {Om_opt[1]:.8f}, {Om_opt[2]:.8f}")
print(f"    Energies:     {E_opt[0]:.8f}, {E_opt[1]:.8f}, {E_opt[2]:.8f}")
print(f"    Mass ratios:  1 : {ratios_opt[1]:.2f} : {ratios_opt[2]:.2f}")
print(f"    Lepton target: 1 : {m_mu/m_e:.2f} : {m_tau/m_e:.2f}")

# Oscillon parameters for each particle
print(f"\n  Oscillon parameters:")
for name, Om, E in zip(['electron', 'muon', 'tau'], Om_opt, E_opt):
    beta2 = 1 - Om**2
    A = 1.5 * beta2
    B = 0.5 * np.sqrt(beta2)
    print(f"    {name:10s}: Omega={Om:.8f}, A={A:.6f}, B={B:.6f}, E={E:.8f}")


# ============================================================
#  CONSTRAINED: Q=2/3 AND mass ratios close to leptons
# ============================================================

print("\n" + "="*70)
print("  1D MODEL: Q=2/3 AND lepton-like ratios simultaneously")
print("="*70)

r_mu_e = m_mu / m_e    # 206.77
r_tau_e = m_tau / m_e   # 3477.23

def objective_combined(params):
    """Minimize |Q-2/3| + penalty for wrong mass ratios."""
    Om1, Om2, Om3 = sorted(params)
    if Om1 <= 0.01 or Om3 >= 0.999:
        return 1e6
    E1, E2, E3 = E_1d(Om1), E_1d(Om2), E_1d(Om3)
    if E1 <= 0 or E2 <= 0 or E3 <= 0:
        return 1e6
    # E3 > E2 > E1 (tau > mu > e)
    Es = sorted([E1, E2, E3])
    Q = koide(*Es)
    # Mass ratio penalties
    r1 = Es[1] / Es[0]  # should be ~207
    r2 = Es[2] / Es[0]  # should be ~3477
    penalty_Q = (Q - 2.0/3.0)**2 * 1e6
    penalty_r1 = (np.log(r1) - np.log(r_mu_e))**2
    penalty_r2 = (np.log(r2) - np.log(r_tau_e))**2
    return penalty_Q + penalty_r1 + penalty_r2


best_comb = None
np.random.seed(123)
for _ in range(500):
    x0 = np.random.uniform(0.02, 0.998, 3)
    try:
        res = minimize(objective_combined, x0, method='Nelder-Mead',
                       options={'xatol': 1e-14, 'fatol': 1e-20, 'maxiter': 5000})
        if best_comb is None or res.fun < best_comb.fun:
            best_comb = res
    except:
        pass

Om_comb = sorted(best_comb.x)
E_comb = sorted([E_1d(Om) for Om in Om_comb])
Q_comb = koide(*E_comb)
ratios_comb = [E / E_comb[0] for E in E_comb]

print(f"\n  COMBINED OPTIMIZATION:")
print(f"    Q = {Q_comb:.12f}  (target = {2/3:.12f})")
print(f"    |Q - 2/3| = {abs(Q_comb - 2/3):.2e}")
print(f"    Mass ratios:  1 : {ratios_comb[1]:.2f} : {ratios_comb[2]:.2f}")
print(f"    Lepton target: 1 : {r_mu_e:.2f} : {r_tau_e:.2f}")
print(f"    Omega: [{Om_comb[0]:.8f}, {Om_comb[1]:.8f}, {Om_comb[2]:.8f}]")
print(f"    Energies: [{E_comb[0]:.10f}, {E_comb[1]:.8f}, {E_comb[2]:.8f}]")

# x = alpha/lambda0 from Koide proof
if E_comb[2] > 0 and E_comb[0] > 0:
    # In rank-1 picture: lambda_1 = lambda_0 + alpha, lambda_2=lambda_3=lambda_0
    # But our three masses are all different.
    # x_eff = (E_tau - E_mu) / E_e  (rough estimate)
    x_eff = (E_comb[2] - E_comb[1]) / E_comb[0]
    x_theory = 33 + 24*np.sqrt(2)
    print(f"\n    Effective x = (E_tau - E_mu)/E_e = {x_eff:.2f}")
    print(f"    Theoretical x = 33 + 24*sqrt(2) = {x_theory:.2f}")


# ============================================================
#  3D FULL COMPUTATION
# ============================================================

print("\n" + "="*70)
print("  3D MODEL: Precision oscillon energy computation")
print("="*70)

def oscillon_3d_rhs(r, y, Omega):
    Phi, dPhi = y
    if r < 1e-10:
        d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**2 / 3.0
    else:
        d2Phi = -(2.0/r) * dPhi - (Omega**2 - 1) * Phi - Phi**2
    return [dPhi, d2Phi]

def find_3d_oscillon(Phi0, r_max=100.0):
    def tail_value(Omega):
        sol = solve_ivp(lambda r, y: oscillon_3d_rhs(r, y, Omega),
                        [1e-6, r_max], [Phi0, 0.0],
                        method='RK45', max_step=0.02,
                        rtol=1e-11, atol=1e-13)
        return sol.y[0, -1]

    try:
        Om_lo, Om_hi = 0.01, 0.999
        signs = []
        Om_tests = np.linspace(0.05, 0.995, 60)
        for Om_test in Om_tests:
            signs.append((Om_test, tail_value(Om_test)))

        for i in range(len(signs)-1):
            if signs[i][1] * signs[i+1][1] < 0:
                Om_lo = signs[i][0]
                Om_hi = signs[i+1][0]
                break

        Omega = brentq(tail_value, Om_lo, Om_hi, xtol=1e-12)
        sol = solve_ivp(lambda r, y: oscillon_3d_rhs(r, y, Omega),
                        [1e-6, r_max], [Phi0, 0.0],
                        method='RK45', max_step=0.01,
                        rtol=1e-12, atol=1e-14, dense_output=True)
        return Omega, sol
    except:
        return None, None

def energy_3d(sol, Omega):
    r = sol.t
    Phi = sol.y[0]
    dPhi = sol.y[1]
    integrand = (0.25 * Omega**2 * Phi**2 + 0.25 * dPhi**2) * r**2
    return 4.0 * np.pi * np.trapz(integrand, r)


# Fine grid of Phi0
print(f"\n  {'Phi0':>6} {'Omega':>10} {'E_3d':>14} {'E_ratio':>10}")

results_3d = []
for Phi0 in np.concatenate([
    np.linspace(0.02, 0.10, 5),
    np.linspace(0.10, 0.50, 20),
    np.linspace(0.50, 0.65, 20),
]):
    Omega_3d, sol_3d = find_3d_oscillon(Phi0, r_max=100.0)
    if Omega_3d is not None and Omega_3d > 0.01:
        E = energy_3d(sol_3d, Omega_3d)
        if E > 0 and E < 1e8:
            results_3d.append((Phi0, Omega_3d, E))
            print(f"  {Phi0:6.3f} {Omega_3d:10.6f} {E:14.6f}"
                  f" {E/results_3d[0][2]:10.1f}")

# Find best Koide triple among 3D results
if len(results_3d) >= 3:
    from itertools import combinations
    best_3d = None
    for combo in combinations(results_3d, 3):
        Es = sorted([c[2] for c in combo])
        Q = koide(*Es)
        Qd = abs(Q - 2.0/3.0)
        if best_3d is None or Qd < best_3d[0]:
            best_3d = (Qd, Q, combo)

    Qd, Q, combo = best_3d
    Es = sorted([c[2] for c in combo])
    print(f"\n  BEST 3D KOIDE:")
    print(f"    Q = {Q:.10f}  (|Q-2/3| = {Qd:.2e})")
    print(f"    Mass ratios: 1 : {Es[1]/Es[0]:.2f} : {Es[2]/Es[0]:.2f}")
    for c in sorted(combo, key=lambda x: x[2]):
        print(f"      Phi0={c[0]:.4f}, Omega={c[1]:.8f}, E={c[2]:.6f}")

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"  Experimental Koide: Q = {Q_exp:.10f}")
print(f"  1D best Q=2/3 (free ratios): Q = {Q_opt:.10f}")
print(f"  1D mass ratios: 1 : {ratios_opt[1]:.1f} : {ratios_opt[2]:.1f}")
print(f"  Lepton ratios:  1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")
