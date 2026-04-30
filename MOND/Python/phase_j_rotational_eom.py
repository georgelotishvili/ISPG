"""
Phase J: Weak-field rotational field equation from the rotational-sector action
=============================================================================

Purpose:
  Starting from the rotational-sector extension

      S = (1/16piG) ∫ sqrt(-g) [ R + X + Z_rot(X/X0) I_rot ] + S_m

  derive the weak-field stationary equations for:
    1. the rotational sector A_i  (or g_{0i})
    2. the scalar field phi       (backreaction from rotation)

This is still an effective derivation, not a final theorem of the full theory.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import G, M_gal, Msun, a0, c, kpc, lambda_H, r_M

SEP = "=" * 76


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def X_from_g(g):
    grad_phi = 2.0 * np.asarray(g, dtype=float) / c**2
    return 0.5 * grad_phi**2


X0 = X_from_g(a0)


def z_rot_from_y(y):
    y = np.asarray(y, dtype=float)
    return 1.0 / (1.0 + np.sqrt(y))


def dz_dy(y):
    y = np.asarray(y, dtype=float)
    return -1.0 / (2.0 * np.sqrt(y) * (1.0 + np.sqrt(y)) ** 2)


def z_rot_from_g(g):
    return 1.0 / (1.0 + np.asarray(g, dtype=float) / a0)


def print_intro():
    print(SEP)
    print("  PHASE J: Rotational EOM from Candidate Action")
    print(SEP)
    print()
    print("  Candidate effective action:")
    print("    S = (1/16piG) ∫ sqrt(-g) [ R + X + Z_rot(X/X0) I_rot ] + S_m")
    print()
    print("  with")
    print("    X      = (1/2) g^{mu nu} d_mu phi d_nu phi")
    print("    X0     = (1/2) (2 a0 / c^2)^2")
    print("    Z_rot  = 1 / (1 + sqrt(X/X0))")
    print("    I_rot  ~ |curl A|^2 / (2 c^4)   in the weak-field stationary limit")
    print()
    print("  Interpretation:")
    print("    R term              -> baseline GR frame-dragging")
    print("    Z_rot I_rot term    -> extra MOND rotational channel")
    print("    strong field        -> Z_rot << 1, extra term turns off")
    print("    weak field          -> Z_rot ~ 1, extra term is active")


def print_variation_wrt_A():
    print("\n" + SEP)
    print("  1. Variation with respect to A_i")
    print(SEP)
    print(
        r"""
  In the stationary weak-field limit take

      I_rot = |B_g|^2 / (2 c^4),     B_g = curl A

  Then the extra rotational action is

      S_rot = (1 / 16piG) ∫ d^3x  Z_rot(X/X0) |B_g|^2 / (2 c^4)

  Vary A:

      delta B_g = curl(delta A)

  so

      delta S_rot
      = (1 / 16piG c^4) ∫ d^3x  Z_rot B_g · curl(delta A)
      = (1 / 16piG c^4) ∫ d^3x  [curl(Z_rot B_g)] · delta A

  after integration by parts and dropping boundary terms.

  Therefore the extra Euler-Lagrange contribution is

      curl(Z_rot B_g)

  and the full stationary weak-field vector equation is schematically

      D_GR[A]_i + [curl(Z_rot B_g)]_i = (16piG / c^2) J_i

  where:
      D_GR[A]_i   = the usual GR / Einstein-Hilbert g_{0i} operator
      J_i         = matter angular-momentum current density

  In Coulomb gauge (div A = 0) and when Z_rot varies slowly:

      D_GR[A]_i - Z_rot nabla^2 A_i ≈ (16piG / c^2) J_i

  So the MOND rotational channel is switched on by the factor Z_rot.
        """
    )


def print_variation_wrt_phi():
    print("\n" + SEP)
    print("  2. Variation with respect to phi")
    print(SEP)
    print(
        r"""
  Because Z_rot depends on X, and X depends on grad(phi), the rotational
  sector feeds back into the scalar equation.

      X = (1/2) |grad phi|^2       in the static weak-field limit

  So

      delta X = grad(phi) · grad(delta phi)

  and

      delta S_rot
      = (1 / 16piG X0) ∫ d^3x  I_rot Z_rot'(X/X0) delta X
      = -(1 / 16piG X0)
        ∫ d^3x  div[ I_rot Z_rot'(X/X0) grad(phi) ] delta phi

  Therefore the scalar field equation becomes schematically

      Box(phi) = source_matter
                 + (1 / X0) div[ I_rot Z_rot'(X/X0) grad(phi) ]

  Since

      Z_rot(y)  = 1 / (1 + sqrt(y))
      Z_rot'(y) = -1 / [ 2 sqrt(y) (1 + sqrt(y))^2 ]  < 0

  the sign is important:
      rotation deepens the low-pressure state instead of erasing it.

  So the rotational sector and scalar pressure sector reinforce each other
  in the MOND regime.
        """
    )


def run_numbers():
    print("\n" + SEP)
    print("  3. Numerical regime check")
    print(SEP)

    r = np.logspace(np.log10(0.03 * r_M), np.log10(100.0 * r_M), 600)
    g_n = G * M_gal / r**2
    g_tot = g_total_from_simple_mu(g_n)
    X = X_from_g(g_tot)
    y = X / X0
    z = z_rot_from_y(y)
    zp = dz_dy(y)

    print(f"  lambda_H = {lambda_H / (1e3 * kpc):.3e} Mpc")
    print(f"  X0       = {X0:.3e} m^-2")
    print("  Hubble scale enters through a0 inside X0.")

    print("\n  Sample radii:")
    print("    r/r_M    g/a0      Z_rot        Z_rot'(X/X0)      comment")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(r / r_M - factor))
        comment = (
            "Newtonian"
            if g_tot[idx] / a0 > 3.0
            else "transition"
            if g_tot[idx] / a0 > 0.3
            else "deep MOND"
        )
        print(
            f"    {factor:5.1f}   "
            f"{g_tot[idx] / a0:8.3f}   "
            f"{z[idx]:10.3e}   "
            f"{zp[idx]:14.3e}   "
            f"{comment}"
        )

    print("\n  Strong-field suppression examples:")
    for name, g_here in [
        ("Earth orbit", G * Msun / (1.496e11) ** 2),
        ("MOND radius", a0),
        ("10 r_M galaxy", g_total_from_simple_mu(a0 / 100.0)),
    ]:
        z_here = z_rot_from_g(g_here)
        print(f"    {name:12s}: g/a0 = {np.asarray(g_here).item() / a0:9.3e}, Z_rot = {np.asarray(z_here).item():9.3e}")


def print_summary():
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        r"""
  Weak-field rotational field equation:

      D_GR[A]_i + [curl(Z_rot B_g)]_i = (16piG / c^2) J_i

  Scalar backreaction:

      Box(phi) = source_matter
                 + (1 / X0) div[ I_rot Z_rot'(X/X0) grad(phi) ]

  Physical reading:
      1. GR frame-dragging remains as the baseline rotational sector.
      2. MOND enters as an extra nonlinear rotational response.
      3. The response is controlled by Z_rot = 1 / (1 + g/a0).
      4. In strong fields Z_rot -> 0, so the extra term disappears.
      5. In weak fields Z_rot -> 1, so the extra term becomes active.

  This is the first clean field-equation-level form of the rotational-sector
  parametrization. The remaining derivation task is to show dynamically why
  the medium selects exactly this Z_rot(X/X0).
        """
    )


if __name__ == "__main__":
    print_intro()
    print_variation_wrt_A()
    print_variation_wrt_phi()
    run_numbers()
    print_summary()
