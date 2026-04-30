"""
Phase I (refined): vibrational loading and rotational MOND
==========================================================

Goal:
  Turn the ontology into a clean constitutive law with a uniqueness theorem:

    tail = vibration = pressure deficit

  Strong local deficit -> medium is more loaded -> harder to rotate
  Weak local deficit   -> medium is less loaded -> easier to rotate

Key output:
  If the rotationally free fraction of the medium is

      Z_rot = 1 / (1 + g/a0),

  then ISPG gives exactly

      mu(x) = x / (1 + x),   x = g/a0.

The strengthening in this file is that Z_rot is no longer presented as the
"simplest" fraction. It is uniquely selected once one demands:

  1. the loading variable is the first-power deficit amplitude L = g/a0,
  2. the load/free odds ratio is additive under independent deficit loading,
  3. the transition point L = 1 splits the medium 50/50.

This is still not a full action-level theorem, but it removes the earlier
"picked by hand" weakness of the loading fraction itself.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import G, M_gal, Msun, a0, c, eps, kpc, lambda_H, r_M, r_s

SEP = "=" * 72


def mu_simple(x):
    x = np.asarray(x, dtype=float)
    return x / (1.0 + x)


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def X_from_g(g):
    grad_phi = 2.0 * np.asarray(g, dtype=float) / c**2
    return 0.5 * grad_phi**2


X0 = X_from_g(a0)


def load_linear(g):
    return np.asarray(g, dtype=float) / a0


def load_quadratic(g):
    return (np.asarray(g, dtype=float) / a0) ** 2


def z_rot_from_linear_loading(g):
    return 1.0 / k_rot_from_linear_loading(g)


def k_rot_from_linear_loading(g):
    L = load_linear(g)
    return 1.0 + L


def load_to_free_odds_ratio(g):
    """Odds ratio R := f_load / f_free selected by additive linear loading."""
    return load_linear(g)


def z_rot_from_quadratic_loading(g):
    L = load_quadratic(g)
    return 1.0 / (1.0 + L)


def print_intro():
    print(SEP)
    print("  PHASE I (REFINED): Vibrational Loading -> MOND")
    print(SEP)
    print()
    print("  ISPG ontology used here:")
    print("    1. tail = vibration")
    print("    2. vibration = pressure deficit")
    print("    3. pressure deficit slows local clocks and makes the medium")
    print("       harder to refill / harder to spin")
    print()
    print("  Therefore the rotational response should be controlled by")
    print("  the local pressure-deficit loading, not by bare frame-dragging.")
    print()
    print(f"  a0       = {a0:.3e} m/s^2")
    print(f"  r_M      = {r_M / kpc:.2f} kpc")
    print(f"  r_s      = {r_s:.3e} m")
    print(f"  eps      = {eps:.3e}")
    print(f"  lambda_H = {lambda_H / (1e3 * kpc):.3e} Mpc")


def print_derivation():
    print("\n" + SEP)
    print("  STEP 1: The loading variable")
    print(SEP)
    print(
        """
  In weak field ISPG:
      g = (c^2 / 2) |grad phi|

  Define the MOND-scale gradient:
      |grad phi|_0 = 2 a0 / c^2

  Then
      X  = (1/2) |grad phi|^2
      X0 = (1/2) |grad phi|_0^2

  The dimensionless loading can be built in two obvious ways:

      linear loading:     L1 = |grad phi| / |grad phi|_0 = sqrt(X/X0) = g/a0
      quadratic loading:  L2 = X / X0 = (g/a0)^2

  The question is: which loading controls rotational stickiness?
        """
    )

    print("\n" + SEP)
    print("  STEP 2: Uniqueness of the free fraction")
    print(SEP)
    print(
        """
  Let

      f_free  = rotationally free fraction,
      f_load  = rotationally loaded fraction,
      R(L)    := f_load / f_free .

  The physical loading variable is the first-power deficit amplitude

      L = g/a0 = sqrt(X/X0).

  Strengthening assumption:
  independent pressure-deficit loadings add in the ODDS ratio, not directly
  in the fraction itself. Therefore

      R(L_a + L_b) = R(L_a) + R(L_b),
      R(0) = 0,
      R(L) monotone.

  The regular monotone solution is

      R(L) = kappa * L.

  Fix the normalization at the MOND transition:

      L = 1  ->  f_free = f_load = 1/2  ->  R(1) = 1,

  so kappa = 1 and therefore

      R(L) = L.

  Since

      f_free = 1 / (1 + R),

  the rotationally free fraction is uniquely selected as

      Z_rot = f_free = 1 / (1 + L)
            = 1 / (1 + g/a0)

  Interpret the total inward support g as two channels:

      g_h = Z_rot * g              (rotational / MOND channel)
      g_N = (1 - Z_rot) * g        (static Bernoulli channel)

  Then immediately:

      g_N / g = (g/a0) / (1 + g/a0) = mu(g/a0)

  so

      mu(x) = x / (1 + x),    x = g/a0

  and also

      g_h / g_N = a0 / g

  which is exactly the MOND closure relation used before.
        """
    )

    print("\n" + SEP)
    print("  STEP 3: Why linear loading is the right one")
    print(SEP)
    print(
        """
  The medium becomes sticky because the static deficit itself is large:

      lower pressure -> slower local time -> slower refilling / more hysteresis

  That is a first-power effect in the deficit amplitude.
  So the natural control variable is L1 ~ |grad phi| ~ g,
  not the energy density X ~ g^2.

  If one incorrectly uses quadratic loading, one gets

      mu_quad(x) = x^2 / (1 + x^2)

  which is not the observed simple MOND law.
        """
    )


def run_numerics():
    print("\n" + SEP)
    print("  STEP 4: Numerical verification")
    print(SEP)

    r = np.logspace(np.log10(0.03 * r_M), np.log10(100.0 * r_M), 600)
    g_n = G * M_gal / r**2
    g_tot = g_total_from_simple_mu(g_n)
    x_tot = g_tot / a0

    X = X_from_g(g_tot)
    z_rot = z_rot_from_linear_loading(g_tot)
    z_rot_x = 1.0 / (1.0 + np.sqrt(X / X0))
    g_h = z_rot * g_tot
    g_n_back = (1.0 - z_rot) * g_tot
    mu_back = g_n_back / g_tot
    mu_target = mu_simple(x_tot)

    mu_quad = load_quadratic(g_tot) / (1.0 + load_quadratic(g_tot))

    rel_err_gn = np.max(np.abs(g_n_back - g_n) / g_n)
    abs_err_mu = np.max(np.abs(mu_back - mu_target))
    abs_err_z = np.max(np.abs(z_rot - z_rot_x))
    abs_err_quad = np.max(np.abs(mu_quad - mu_target))

    print(f"  max relative error in reconstructed g_N  = {rel_err_gn:.3e}")
    print(f"  max absolute error in mu(x)              = {abs_err_mu:.3e}")
    print(f"  max absolute error in Z_rot(X)           = {abs_err_z:.3e}")
    print(f"  max absolute error of quadratic loading  = {abs_err_quad:.3e}")

    print("\n  Sample radii:")
    print("    r/r_M    g_N/a0     g/a0      Z_rot      mu(x)    g_h/g_N")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(r / r_M - factor))
        gh_over_gn = g_h[idx] / g_n[idx]
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx] / a0:8.3f}   "
            f"{g_tot[idx] / a0:8.3f}   "
            f"{z_rot[idx]:8.3f}   "
            f"{mu_back[idx]:8.3f}   "
            f"{gh_over_gn:8.3f}"
        )

    idx_rm = np.argmin(np.abs(r - r_M))
    print("\n  At r = r_M (defined by g_N = a0):")
    print(f"    g/a0                   = {g_tot[idx_rm] / a0:.6f}")
    print(f"    Z_rot                  = {z_rot[idx_rm]:.6f}")
    print(f"    omega_space/Omega_star = {np.sqrt(z_rot[idx_rm]):.6f}")
    print("    Rotational channel supplies about 38% of total support here.")


def print_rotational_parametrization():
    print("\n" + SEP)
    print("  STEP 5: Rotational-sector parametrization")
    print(SEP)
    print(
        """
  Keep the scalar ISPG sector unchanged and parameterize the EXTRA
  weak-field rotational sector by

      S = (1 / 16 pi G) int d^4x sqrt(-g) [ R + X + Z_rot(X/X0) I_rot ] + S_m

  where
      X        = (1/2) g^{mu nu} d_mu phi d_nu phi
      X0       = (1/2) (2 a0 / c^2)^2
      K_rot(y) = 1 + sqrt(y)
      Z_rot(y) = 1 / K_rot(y)

  and I_rot is a positive invariant built from the rotational sector.
  In weak field it should reduce to something like

      I_rot ~ |curl A|^2 / c^4

  with A the gravitomagnetic potential.

  Here K_rot is the rotational stiffness of the medium and Z_rot is
  the rotationally free fraction. The Hubble scale is already encoded
  through a0 inside X0. The Einstein-Hilbert term already
  contains the baseline GR frame-dragging sector; the new term is only
  the additional bound-structure rotational response.

  Why this is the right shape:
      strong field:  y >> 1  ->  K_rot ~ sqrt(y) ~ g/a0,   Z_rot ~ a0/g -> 0
      weak field:    y << 1  ->  K_rot -> 1,               Z_rot -> 1

  So:
      strong field -> extra MOND rotation switches off
      weak field   -> extra MOND rotation is unsuppressed
        """
    )


def print_consistency_checks():
    print("\n" + SEP)
    print("  STEP 6: Strong-field suppression")
    print(SEP)

    g_earth_orbit = G * Msun / (1.496e11) ** 2
    g_earth_surface = 9.81
    g_neutron_star = G * 1.4 * Msun / (1.0e4) ** 2

    for name, g_here in [
        ("Earth orbit", g_earth_orbit),
        ("Earth surface", g_earth_surface),
        ("neutron star surface", g_neutron_star),
    ]:
        z_here = z_rot_from_linear_loading(g_here)
        print(
            f"  {name:20s}: g/a0 = {g_here / a0:10.3e}, "
            f"Z_rot = {z_here:10.3e}"
        )

    print("\n  The rotational modification is negligible in strong fields.")


def print_summary():
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        """
  1. The correct loading variable is
         L = g/a0 = sqrt(X/X0)

  2. The rotationally free fraction of the medium is uniquely selected by
     additive odds-ratio loading:
         R = f_load/f_free = g/a0
         Z_rot = 1 / (1 + g/a0)

  3. Splitting total support into static + rotational channels gives
         g_N = (1 - Z_rot) g
         g_h = Z_rot g

  4. Therefore
         mu(x) = g_N/g = x/(1+x),   x = g/a0

  5. In X-language the same law is
         K_rot(X) = 1 + sqrt(X/X0)
         Z_rot(X) = 1 / K_rot(X)

  6. This is a clean constitutive bridge from
         tail -> vibration -> pressure deficit -> rotational stickiness
     to the MOND interpolating function.

  Program role:
     This script isolates the constitutive bridge between rotational
     loading and the MOND interpolating function.
     The later axisymmetric PDE check shows that an additive
     operator correction weighted only by Z_rot gives at most O(1)
     enhancement of the GR rotational response. Therefore the full
     bound-structure closure must include the source-side response too.
     The corresponding source-side completion is
         S_rot = [Z_rot / (1 - Z_rot)] S_N ,
     which reproduces the exact MOND closure without eps suppression.
     The remaining derivation task is to obtain the same odds-ratio law
         R = g/a0
     from the covariant rotational sector instead of imposing it at the
     coarse-grained constitutive level.
        """
    )


if __name__ == "__main__":
    print_intro()
    print_derivation()
    run_numerics()
    print_rotational_parametrization()
    print_consistency_checks()
    print_summary()
