"""
Phase R: Consistency audit of the swirl-source completion
=========================================================

Goals:
  1. Quantify how much the finite activation law A_vort(omega) changes the
     galaxy MOND profile relative to the coherent limit A_vort = 1.
  2. Check strong-field suppression of the source fraction.
  3. Make explicit the domain-of-validity condition needed for GR recovery in
     non-galactic systems.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import G, Msun, Omega_tr_conj, a0, c, r_M
from source import g_newton

SEP = "=" * 78


def g_simple_mond(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def omega_from_circular_state(g, r):
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.sqrt(np.maximum(g / r, 0.0))


def a_vort(omega):
    omega = np.asarray(omega, dtype=float)
    return omega / (omega + Omega_tr_conj)


def solve_g_with_activation(g_n, r, tol=1e-13, max_iter=200):
    g_n = np.asarray(g_n, dtype=float)
    r = np.asarray(r, dtype=float)
    g = g_simple_mond(g_n)

    for _ in range(max_iter):
        omega = omega_from_circular_state(g, r)
        A = a_vort(omega)
        g_new = 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * A * a0 * g_n))
        if np.max(np.abs(g_new - g) / np.maximum(g_new, 1e-300)) < tol:
            return g_new, A
        g = g_new

    return g, a_vort(omega_from_circular_state(g, r))


def print_galaxy_audit():
    print(SEP)
    print("  1. Galaxy profile audit")
    print(SEP)

    xi = np.geomspace(0.05, 100.0, 800)
    r = xi * r_M
    g_n = g_newton(xi)
    g_coh = g_simple_mond(g_n)
    g_act, A = solve_g_with_activation(g_n, r)

    rel_g = np.abs(g_act - g_coh) / g_coh
    v_coh = np.sqrt(r * g_coh)
    v_act = np.sqrt(r * g_act)
    rel_v = np.abs(v_act - v_coh) / v_coh

    print(f"  Omega_tr = a0/c = {Omega_tr_conj:.4e} s^-1")
    print(f"  max relative change in g(r)      = {np.max(rel_g):.3e}")
    print(f"  max relative change in v_circ(r) = {np.max(rel_v):.3e}")
    print()
    print("    r/r_M    A_vort    g_act/a0   g_simple/a0   rel.dg     rel.dv")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        idx = np.argmin(np.abs(xi - factor))
        reldv = abs(v_act[idx] - v_coh[idx]) / v_coh[idx]
        print(
            f"    {factor:5.1f}   "
            f"{A[idx]:8.4f}   "
            f"{g_act[idx]/a0:9.3f}   "
            f"{g_coh[idx]/a0:11.3f}   "
            f"{rel_g[idx]:7.3e}   "
            f"{reldv:7.3e}"
        )

    print()
    print("  Result:")
    print("    In the fiducial spiral-galaxy regime the finite activation law")
    print("    changes the coherent MOND profile only weakly, because A_vort")
    print("    stays very close to unity throughout the disk/halo region.")


def point_mass_g(M, r):
    return G * M / r**2


def print_strong_field_examples():
    print("\n" + SEP)
    print("  2. Strong-field source fraction")
    print(SEP)

    AU = 1.495978707e11
    R_earth = 6.371e6
    M_earth = 5.9722e24
    R_ns = 1.2e4
    M_ns = 1.4 * Msun

    cases = [
        ("Earth orbit (1 AU)", point_mass_g(Msun, AU), np.sqrt(point_mass_g(Msun, AU) / AU)),
        ("Mercury orbit", point_mass_g(Msun, 5.79e10), np.sqrt(point_mass_g(Msun, 5.79e10) / 5.79e10)),
        ("LEO around Earth", point_mass_g(M_earth, R_earth + 4.0e5), np.sqrt(point_mass_g(M_earth, R_earth + 4.0e5) / (R_earth + 4.0e5))),
        ("Neutron-star surface", point_mass_g(M_ns, R_ns), np.sqrt(point_mass_g(M_ns, R_ns) / R_ns)),
    ]

    print("    case                    g/a0         A_naive     Sigma/T^m    delta_g/a0")
    for name, g_here, omega_here in cases:
        A_here = a_vort(omega_here)
        sigma_ratio = A_here * a0 / g_here
        delta_g_over_a0 = A_here
        print(
            f"    {name:20s} "
            f"{g_here/a0:10.3e}   "
            f"{A_here:8.4f}   "
            f"{sigma_ratio:10.3e}   "
            f"{delta_g_over_a0:10.3e}"
        )

    print()
    print("  Reading:")
    print("    The source fraction Sigma/T^m is tiny in strong fields because it")
    print("    scales as A_vort * a0/g.")
    print("    But if one naively set A_vort ~ 1 in every bound system, the")
    print("    residual absolute acceleration would remain of order a0.")


def print_domain_statement():
    print("\n" + SEP)
    print("  3. Domain-of-validity condition")
    print(SEP)
    print(
        """
  Therefore the swirl-source completion must be read as a GALAXY-SCALE
  coarse-grained effective sector.

  The activation factor A_vort is not the Kepler frequency of an arbitrary
  small bound system. It is the macroscopic phase-order parameter of a mature
  coherent vortex of the space medium.

  Consequences:
  - Mature spiral galaxies: A_vort ~ 1  -> MOND active.
  - Systems without a mature macroscopic vortex: A_vort ~ 0  -> GR recovered.

  So Solar-System consistency requires interpreting A_vort as a macroscopic
  coherent-vortex order parameter, not as a universal local orbital factor.
        """
    )


def print_summary():
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        """
  The new swirl-source completion passes the galaxy-profile audit:
  finite A_vort changes the fiducial spiral-galaxy MOND curve only weakly.

  The strong-field source fraction is small as a0/g, but exact GR recovery in
  non-galactic systems requires the intended effective reading:
  the sector is activated only by mature macroscopic vortices.

  So the consistency picture is:
  - galaxy regime: good,
  - coherent MOND limit: recovered,
  - non-galactic regime: requires the macroscopic-vortex interpretation of
    A_vort.
        """
    )


if __name__ == "__main__":
    print_galaxy_audit()
    print_strong_field_examples()
    print_domain_statement()
    print_summary()
