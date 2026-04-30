"""
Phase K: Axisymmetric A_phi(R,z) PDE for the rotational MOND sector
===================================================================

Goal:
  Derive the cylindrical PDE for the azimuthal gravitomagnetic potential
  A_phi(R,z), assuming axisymmetry and the rotational-sector weak-field equation

      D_GR[A]_i + [curl(Z_rot B_g)]_i = (16piG/c^2) J_i .

Key question:
  Does the new rotational term merely renormalize the GR operator,
  or can it by itself bridge the old amplitude gap?
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, eps, r_M, kpc
from source import g_newton

SEP = "=" * 78


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def z_rot(g):
    g = np.asarray(g, dtype=float)
    return 1.0 / (1.0 + g / a0)


def print_derivation():
    print(SEP)
    print("  PHASE K: Axisymmetric Rotational PDE")
    print(SEP)
    print(
        r"""
  Assume axisymmetry and a purely azimuthal vector potential

      A = A_phi(R,z) e_phi .

  Then the gravitomagnetic field is poloidal:

      B_R = -∂_z A_phi
      B_z = (1/R) ∂_R (R A_phi)

  Therefore the phi-component of curl(Z_rot B_g) is

      [curl(Z_rot B_g)]_phi
      = ∂_z (Z_rot B_R) - ∂_R (Z_rot B_z)
      = - ∂_z [ Z_rot ∂_z A_phi ]
        - ∂_R [ Z_rot (1/R) ∂_R (R A_phi) ] .

  So the weak-field stationary axisymmetric equation is

      D_GR[A_phi]
      - ∂_z [ Z_rot ∂_z A_phi ]
      - ∂_R [ Z_rot (1/R) ∂_R (R A_phi) ]
      = (16piG/c^2) J_phi .
        """
    )


def print_flux_function_form():
    print("\n" + SEP)
    print("  Flux-function form")
    print(SEP)
    print(
        r"""
  Define the flux function

      psi(R,z) = R A_phi(R,z) .

  Then

      B_R = -(1/R) ∂_z psi
      B_z =  (1/R) ∂_R psi

  and the extra MOND operator becomes

      [curl(Z_rot B_g)]_phi
      = -(1/R) [ Z_rot Δ_* psi + (∂_R Z_rot)(∂_R psi) + (∂_z Z_rot)(∂_z psi) ]

  where

      Δ_* psi = ∂_R^2 psi - (1/R) ∂_R psi + ∂_z^2 psi

  is the standard Grad-Shafranov operator.

  This is the cleanest axisymmetric form of the rotational extension.
        """
    )


def print_slow_variation_limit():
    print("\n" + SEP)
    print("  Slow-variation limit")
    print(SEP)
    print(
        r"""
  If Z_rot varies slowly compared with A_phi, then gradient terms of Z_rot
  are subleading and

      [curl(Z_rot B_g)]_phi ≈ - Z_rot (∇^2 - 1/R^2) A_phi .

  So the total equation becomes approximately

      [1 + Z_rot(R,z)] L_GR[A_phi] ≈ source ,

  where

      L_GR[A_phi] = - (∇^2 - 1/R^2) A_phi .

  This immediately shows a crucial point:

      operator enhancement <= 2

  because

      0 <= Z_rot <= 1 .

  Therefore, if the source on the right-hand side remains the ordinary
  GR mass-current source J_phi, the new term by itself can only amplify
  the response by an O(1) factor.
        """
    )


def run_midplane_check():
    print("\n" + SEP)
    print("  Midplane regime check for a disk galaxy")
    print(SEP)

    xi = np.geomspace(0.05, 50.0, 300)
    g_n = g_newton(xi)
    g_tot = g_total_from_simple_mu(g_n)
    z = z_rot(g_tot)
    operator_gain = 1.0 + z

    print("    r/r_M    g_N/a0     g/a0      Z_rot    1+Z_rot")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx] / a0:8.3f}   "
            f"{g_tot[idx] / a0:8.3f}   "
            f"{z[idx]:8.3f}   "
            f"{operator_gain[idx]:8.3f}"
        )

    print()
    print(f"  Maximum operator gain in this model: {np.max(operator_gain):.3f}")
    print(f"  Old perturbative amplitude gap: 1/eps = {1.0 / eps:.3e}")

    idx_rm = np.argmin(np.abs(xi - 1.0))
    print()
    print(f"  At r ~ r_M:")
    print(f"    Z_rot         = {z[idx_rm]:.3f}")
    print(f"    operator gain = {operator_gain[idx_rm]:.3f}")


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        r"""
  Honest conclusion:

  1. The axisymmetric PDE is clean and well defined.
  2. The MOND term has the right qualitative behavior:
         strong field -> off
         weak field   -> on
  3. But if it only appears as an additive correction to the GR operator,
     its direct effect is only O(1), because Z_rot <= 1.
  4. So this by itself does NOT explain a huge enhancement over the old
     frame-dragging source.

  Therefore the next logical step is sharper:

      the new rotational physics must enter not only as operator
      renormalization, but as a genuinely new source / constitutive
      response tied to pattern rotation or pressure redistribution itself.

  In simple words:
      changing how easily space twists is not enough;
      we probably also need a new way for twisting to be sourced.
        """
    )


if __name__ == "__main__":
    print_derivation()
    print_flux_function_form()
    print_slow_variation_limit()
    run_midplane_check()
    print_interpretation()
