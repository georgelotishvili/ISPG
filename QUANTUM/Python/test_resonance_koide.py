"""Hybrid: 3D resonance selects Omega, 1D energy gives mass.
Also try: 3D energy directly, and various resonance definitions.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from itertools import combinations


def E_1D(Om):
    """1D oscillon energy formula."""
    k2 = 1.0 - Om**2
    if k2 <= 0:
        return 0.0
    return k2**1.5 * (4 * Om**2 + 1)


def koide(m1, m2, m3):
    return (m1 + m2 + m3) / (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3))**2


# Resonance-selected Omega values from Phase 2 of test_resonance.py
resonance_sets = {
    'Om*R_half': [
        (1, 0.9725),
        (2, 0.4972),  # actually this has Om=0.9938; the Phi0 was 0.4972
    ],
    'Om*R_90': [
        (1, 0.8978),
        (2, 0.9700),
        (3, 0.9866),
        (4, 0.9939),
        (5, 0.9992),
    ],
    'Om/kappa': [
        (1, 0.9529),
        (2, 0.9876),
        (3, 0.9944),
        (4, 0.9968),
        (5, 0.9980),
        (6, 0.9986),
        (7, 0.9989),
        (8, 0.9992),
    ],
}

# Also try general: Omega_n = n/sqrt(n^2 + c^2) for various c
# This corresponds to Om*R = n*pi with R = c/kappa

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86

print("=" * 70)
print("  Hybrid: 3D resonance selects Omega, 1D formula gives mass")
print("=" * 70)

for name, omegas in resonance_sets.items():
    if len(omegas) < 3:
        continue
    print(f"\n  Resonance definition: {name}")
    print(f"  {'n':>4} {'Omega':>10} {'E_1D':>14} {'kappa':>8}")

    for n, Om in omegas:
        E = E_1D(Om)
        kappa = np.sqrt(1 - Om**2)
        print(f"  {n:4d} {Om:10.6f} {E:14.8f} {kappa:8.4f}")

    Es = [(n, Om, E_1D(Om)) for n, Om in omegas if E_1D(Om) > 0]
    if len(Es) < 3:
        continue

    print(f"\n  Koide check (all triples):")
    best_Q_diff = 999
    best = None
    for combo in combinations(Es, 3):
        e = sorted([c[2] for c in combo])
        Q = koide(*e)
        diff = abs(Q - 2/3)
        if diff < best_Q_diff:
            best_Q_diff = diff
            best_Q = Q
            best = combo
            best_e = e

    if best:
        print(f"    Best Q = {best_Q:.8f}  |Q-2/3| = {best_Q_diff:.2e}")
        print(f"    E = {best_e[0]:.8f}, {best_e[1]:.8f}, {best_e[2]:.8f}")
        if best_e[0] > 0:
            print(f"    Ratios: 1 : {best_e[1]/best_e[0]:.1f}"
                  f" : {best_e[2]/best_e[0]:.1f}")
        print(f"    Target: 1 : 206.8 : 3477.2")
        for c in sorted(best, key=lambda x: x[2]):
            print(f"      n={c[0]}, Om={c[1]:.6f}, E={c[2]:.8f}")

# ============================================================
# Parametric scan: Om_n = n*pi / sqrt(n^2*pi^2 + C^2)
# ============================================================
print("\n" + "=" * 70)
print("  Parametric: Om_n = n*pi / sqrt(n^2*pi^2 + C^2)")
print("  Scanning C to find best Koide Q")
print("=" * 70)

best_overall_Q_diff = 999
best_overall = None

for C in np.linspace(0.5, 20.0, 2000):
    omegas = []
    for n in range(1, 12):
        Om = n * np.pi / np.sqrt(n**2 * np.pi**2 + C**2)
        if Om < 0.999:
            omegas.append((n, Om, E_1D(Om)))

    if len(omegas) < 3:
        continue

    for combo in combinations(omegas, 3):
        e = sorted([c[2] for c in combo])
        if e[0] > 1e-15:
            Q = koide(*e)
            diff = abs(Q - 2/3)
            if diff < best_overall_Q_diff:
                best_overall_Q_diff = diff
                best_overall_Q = Q
                best_overall_C = C
                best_overall = combo
                best_overall_e = e

if best_overall:
    print(f"\n  BEST Koide Q = {best_overall_Q:.10f}")
    print(f"  |Q - 2/3| = {best_overall_Q_diff:.2e}")
    print(f"  C = {best_overall_C:.4f}")
    print(f"  E = {best_overall_e[0]:.10f}, {best_overall_e[1]:.10f},"
          f" {best_overall_e[2]:.10f}")
    if best_overall_e[0] > 0:
        print(f"  Ratios: 1 : {best_overall_e[1]/best_overall_e[0]:.1f}"
              f" : {best_overall_e[2]/best_overall_e[0]:.1f}")
    print(f"  Target: 1 : 206.8 : 3477.2")
    for c in sorted(best_overall, key=lambda x: x[2]):
        print(f"    n={c[0]}, Om={c[1]:.8f}, E_1D={c[2]:.10f}")

# ============================================================
# Also try: Om_n = sqrt(1 - (a/n)^b) for various a, b
# ============================================================
print("\n" + "=" * 70)
print("  General: Om_n = sqrt(1 - (a/n)^b)")
print("  Scanning a, b for best Koide Q")
print("=" * 70)

best2_Q_diff = 999
best2 = None

for a in np.linspace(0.1, 5.0, 200):
    for b in np.linspace(0.5, 4.0, 100):
        omegas = []
        for n in range(1, 10):
            val = (a / n)**b
            if val < 1:
                Om = np.sqrt(1 - val)
                omegas.append((n, Om, E_1D(Om)))

        if len(omegas) < 3:
            continue

        for combo in combinations(omegas, 3):
            e = sorted([c[2] for c in combo])
            if e[0] > 1e-15:
                Q = koide(*e)
                diff = abs(Q - 2/3)
                if diff < best2_Q_diff:
                    best2_Q_diff = diff
                    best2_Q = Q
                    best2_a = a
                    best2_b = b
                    best2 = combo
                    best2_e = e

if best2:
    print(f"\n  BEST Koide Q = {best2_Q:.10f}")
    print(f"  |Q - 2/3| = {best2_Q_diff:.2e}")
    print(f"  a = {best2_a:.4f}, b = {best2_b:.4f}")
    print(f"  E = {best2_e[0]:.10f}, {best2_e[1]:.10f},"
          f" {best2_e[2]:.10f}")
    if best2_e[0] > 0:
        print(f"  Ratios: 1 : {best2_e[1]/best2_e[0]:.1f}"
              f" : {best2_e[2]/best2_e[0]:.1f}")
    print(f"  Target: 1 : 206.8 : 3477.2")
    for c in sorted(best2, key=lambda x: x[2]):
        print(f"    n={c[0]}, Om={c[1]:.8f}, E_1D={c[2]:.10f}")

    print(f"\n  Full spectrum at a={best2_a:.4f}, b={best2_b:.4f}:")
    for n in range(1, 10):
        val = (best2_a / n)**best2_b
        if val < 1:
            Om = np.sqrt(1 - val)
            E = E_1D(Om)
            kappa = np.sqrt(1 - Om**2)
            print(f"    n={n}: Om={Om:.8f}, kappa={kappa:.6f}, E={E:.10f}")
