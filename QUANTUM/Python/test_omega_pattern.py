"""Reverse approach:
1. Find exact Omega triplet that gives Koide Q=2/3 AND correct mass ratios
2. Check if those Omegas fit a resonance pattern Om_n = f(n)
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.optimize import minimize


def E_1D(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0:
        return 0.0
    return k2**1.5 * (4 * Om**2 + 1)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
Q_exp = koide(m_e, m_mu, m_tau)
r_mu = m_mu / m_e
r_tau = m_tau / m_e

print("=" * 70)
print("  Step 1: Find Omega triplet matching lepton masses")
print(f"  Q_exp = {Q_exp:.10f}")
print(f"  Ratios: 1 : {r_mu:.2f} : {r_tau:.2f}")
print("=" * 70)

def cost(params):
    Om_e, Om_mu, Om_tau = params
    if Om_e <= 0 or Om_e >= 1: return 1e10
    if Om_mu <= 0 or Om_mu >= 1: return 1e10
    if Om_tau <= 0 or Om_tau >= 1: return 1e10
    if Om_tau >= Om_mu: return 1e10
    if Om_mu >= Om_e: return 1e10

    E_e = E_1D(Om_e)
    E_mu = E_1D(Om_mu)
    E_tau = E_1D(Om_tau)
    if E_e <= 0 or E_mu <= 0 or E_tau <= 0: return 1e10

    Q = koide(E_e, E_mu, E_tau)
    r1 = E_mu / E_e
    r2 = E_tau / E_e

    return (Q - Q_exp)**2 * 1e8 + (np.log(r1/r_mu))**2 + (np.log(r2/r_tau))**2

best_result = None
best_cost = 1e10

for Om_tau_init in np.linspace(0.3, 0.85, 20):
    for Om_mu_init in np.linspace(Om_tau_init + 0.05, 0.98, 15):
        for Om_e_init in np.linspace(Om_mu_init + 0.01, 0.9999, 10):
            res = minimize(cost, [Om_e_init, Om_mu_init, Om_tau_init],
                           method='Nelder-Mead',
                           options={'xatol': 1e-14, 'fatol': 1e-14,
                                    'maxiter': 10000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x

Om_e, Om_mu, Om_tau = best_result
E_e = E_1D(Om_e)
E_mu = E_1D(Om_mu)
E_tau = E_1D(Om_tau)
Q = koide(E_e, E_mu, E_tau)

print(f"\n  Optimal Omega values:")
print(f"    Electron: Om_e   = {Om_e:.12f}   kappa_e   = {np.sqrt(1-Om_e**2):.12f}")
print(f"    Muon:     Om_mu  = {Om_mu:.12f}   kappa_mu  = {np.sqrt(1-Om_mu**2):.12f}")
print(f"    Tau:      Om_tau = {Om_tau:.12f}   kappa_tau = {np.sqrt(1-Om_tau**2):.12f}")

print(f"\n  Energy (mass):")
print(f"    E_e   = {E_e:.12f}")
print(f"    E_mu  = {E_mu:.12f}")
print(f"    E_tau = {E_tau:.12f}")

print(f"\n  Check:")
print(f"    Q = {Q:.10f}  (target: {Q_exp:.10f})")
print(f"    Ratios: 1 : {E_mu/E_e:.2f} : {E_tau/E_e:.2f}")
print(f"    Target: 1 : {r_mu:.2f} : {r_tau:.2f}")

# Derived quantities
k_e = np.sqrt(1 - Om_e**2)
k_mu = np.sqrt(1 - Om_mu**2)
k_tau = np.sqrt(1 - Om_tau**2)

print(f"\n  Kappa values:")
print(f"    kappa_e   = {k_e:.10f}")
print(f"    kappa_mu  = {k_mu:.10f}")
print(f"    kappa_tau = {k_tau:.10f}")
print(f"    kappa ratios: 1 : {k_mu/k_e:.4f} : {k_tau/k_e:.4f}")

print(f"\n  Omega * R (Om/kappa):")
print(f"    e:   {Om_e/k_e:.6f}")
print(f"    mu:  {Om_mu/k_mu:.6f}")
print(f"    tau: {Om_tau/k_tau:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("  Step 2: Check if Omegas fit resonance pattern Om_n = f(n)")
print("=" * 70)

# Try: kappa_n = a / n^p → Om_n = sqrt(1 - (a/n^p)^2)
print("\n  Test: kappa = a / n^p")
for n_e in range(2, 15):
    for n_mu in range(1, n_e):
        for n_tau in range(1, n_mu):
            if n_tau == n_mu or n_mu == n_e:
                continue
            # From kappa = a * n^(-p), fit a and p using two points
            # log(kappa) = log(a) - p*log(n)
            # Using tau and e: two equations, two unknowns
            log_ratio = np.log(k_tau / k_e)
            log_n_ratio = np.log(n_tau / n_e)  # negative
            if abs(log_n_ratio) < 0.01:
                continue
            p = -log_ratio / log_n_ratio
            a = k_e * n_e**p

            # Check muon
            k_mu_pred = a / n_mu**p
            if abs(k_mu_pred - k_mu) / k_mu < 0.01:
                Om_pred = [np.sqrt(1 - (a/n**p)**2) for n in [n_e, n_mu, n_tau]
                           if (a/n**p)**2 < 1]
                if len(Om_pred) == 3:
                    print(f"    MATCH: n_tau={n_tau}, n_mu={n_mu}, n_e={n_e}")
                    print(f"      a={a:.6f}, p={p:.6f}")
                    print(f"      kappa_mu: pred={k_mu_pred:.8f},"
                          f" actual={k_mu:.8f},"
                          f" err={abs(k_mu_pred-k_mu)/k_mu*100:.4f}%")

# Try: Om_n = sqrt(1 - (a/n)^b)
print("\n  Test: Om = sqrt(1 - (a/n)^b)")
for n_e in range(2, 15):
    for n_mu in range(1, n_e):
        for n_tau in range(1, n_mu):
            if n_tau == n_mu or n_mu == n_e:
                continue
            k2_e = 1 - Om_e**2
            k2_tau = 1 - Om_tau**2
            log_ratio = np.log(k2_tau / k2_e)
            log_n_ratio = np.log(n_tau / n_e)
            if abs(log_n_ratio) < 0.01:
                continue
            b = -log_ratio / log_n_ratio
            a_b = k2_e * n_e**b
            a = a_b**(1/b) if b > 0 else 0

            if a <= 0 or a > 100:
                continue

            k2_mu_pred = (a / n_mu)**b
            if k2_mu_pred >= 1:
                continue

            err = abs(k2_mu_pred - (1-Om_mu**2)) / (1-Om_mu**2)
            if err < 0.02:
                print(f"    MATCH: n_tau={n_tau}, n_mu={n_mu}, n_e={n_e}")
                print(f"      a={a:.8f}, b={b:.8f}")
                Om_pred_mu = np.sqrt(1 - (a/n_mu)**b)
                print(f"      Om_mu: pred={Om_pred_mu:.10f},"
                      f" actual={Om_mu:.10f},"
                      f" err={err*100:.4f}%")

# Try simple: kappa_n = C / n  (hydrogen-like)
print("\n  Test: kappa = C/n (hydrogen-like)")
for n_e in range(2, 20):
    for n_mu in range(1, n_e):
        C_e = k_e * n_e
        C_mu = k_mu * n_mu
        if abs(C_e - C_mu) / C_e < 0.03:
            C_avg = (C_e + C_mu) / 2
            for n_tau in range(1, n_mu):
                k_tau_pred = C_avg / n_tau
                if abs(k_tau_pred - k_tau) / k_tau < 0.03:
                    print(f"    MATCH: n_tau={n_tau}, n_mu={n_mu}, n_e={n_e}")
                    print(f"      C = {C_avg:.6f}")
                    print(f"      kappa: pred=[{C_avg/n_tau:.6f},"
                          f" {C_avg/n_mu:.6f}, {C_avg/n_e:.6f}]")
                    print(f"      kappa: actual=[{k_tau:.6f},"
                          f" {k_mu:.6f}, {k_e:.6f}]")

# Try: kappa_n = C / n^2 (Balmer-like)
print("\n  Test: kappa = C/n^2 (Balmer-like)")
for n_e in range(2, 20):
    for n_mu in range(1, n_e):
        C_e = k_e * n_e**2
        C_mu = k_mu * n_mu**2
        if abs(C_e - C_mu) / C_e < 0.05:
            C_avg = (C_e + C_mu) / 2
            for n_tau in range(1, n_mu):
                k_tau_pred = C_avg / n_tau**2
                if abs(k_tau_pred - k_tau) / k_tau < 0.05:
                    print(f"    MATCH: n_tau={n_tau}, n_mu={n_mu}, n_e={n_e}")
                    print(f"      C = {C_avg:.6f}")

# Print key ratios to look for patterns
print("\n  Key ratios for pattern recognition:")
print(f"    kappa_tau/kappa_mu = {k_tau/k_mu:.8f}")
print(f"    kappa_mu/kappa_e  = {k_mu/k_e:.8f}")
print(f"    kappa_tau/kappa_e  = {k_tau/k_e:.8f}")
print(f"    Om_e/Om_mu        = {Om_e/Om_mu:.8f}")
print(f"    Om_mu/Om_tau      = {Om_mu/Om_tau:.8f}")
print(f"    Om_e/Om_tau       = {Om_e/Om_tau:.8f}")
print(f"    (1-Om_e^2)/(1-Om_mu^2) = {(1-Om_e**2)/(1-Om_mu**2):.8f}")
print(f"    (1-Om_mu^2)/(1-Om_tau^2) = {(1-Om_mu**2)/(1-Om_tau**2):.8f}")
