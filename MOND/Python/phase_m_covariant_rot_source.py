"""
Phase M: Covariant rotational source tensor
===========================================

Goal:
  Package the source-side MOND completion into a covariant effective tensor.

Physical idea:
  The vortex does not merely modify the operator; it contributes an
  additional stress/source in the plane transverse to the local vorticity axis.

Definitions:
  u^mu        : matter/medium 4-velocity
  h_{mu nu}   : spatial projector orthogonal to u^mu
  omega^mu    : vorticity pseudovector
  P^perp_{mu nu} = h_{mu nu} - omegahat_mu omegahat_nu

The rotational stress tensor is written as

  T^(rot)_{mu nu} = (1/2) chi(X,omega) T^(m) P^perp_{mu nu}

with

  chi(X,omega) = A_vort(omega) * sqrt(X0 / X)

and in the coherent galactic-vortex regime A_vort ~ 1, so

  T^(rot) / T^(m) = sqrt(X0/X) = a0/g .

This reproduces the source-side MOND closure.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, c, r_M
from source import g_newton

SEP = "=" * 78


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def X_from_g(g):
    grad_phi = 2.0 * np.asarray(g, dtype=float) / c**2
    return 0.5 * grad_phi**2


X0 = X_from_g(a0)


def chi_from_g(g):
    return a0 / np.asarray(g, dtype=float)


def z_rot(g):
    return 1.0 / (1.0 + np.asarray(g, dtype=float) / a0)


def print_covariant_setup():
    print(SEP)
    print("  PHASE M: Covariant Rotational Source Tensor")
    print(SEP)
    print(
        r"""
  Start from the source-side completion

      T_rot = (a0/g) T_N

  and rewrite it covariantly.

  Let

      u^mu u_mu = -c^2
      h_{mu nu} = g_{mu nu} + u_mu u_nu / c^2

  be the usual spatial projector orthogonal to the matter flow.

  Let the local vorticity pseudovector be

      omega^mu = (1/2) epsilon^{mu nu alpha beta} u_nu nabla_alpha u_beta

  and define the unit vorticity direction omegahat^mu.

  Then the plane transverse to the vortex axis is projected by

      P^perp_{mu nu} = h_{mu nu} - omegahat_mu omegahat_nu .
        """
    )


def print_tensor_form():
    print("\n" + SEP)
    print("  1. Tensor form")
    print(SEP)
    print(
        r"""
  Effective rotational stress:

      T^(rot)_{mu nu}
      = (1/2) chi(X,omega) T^(m) P^perp_{mu nu}

  where

      X  = (1/2) g^{mu nu} d_mu phi d_nu phi
      X0 = (1/2) (2 a0 / c^2)^2

      chi(X,omega) = A_vort(omega) sqrt(X0 / X)

  where Phase Q closes the activation factor as

      A_vort(omega) = omega / (omega + Omega_tr),
      Omega_tr      = a0 / c,

  so

      0 <= A_vort <= 1,
      A_vort ~ 0   for no coherent vortex,
      A_vort ~ 1   for a mature coherent galactic vortex.

  Why this form is natural:

  - sqrt(X0/X) = a0/g gives the required MOND loading ratio.
  - P^perp_{mu nu} puts the stress in the swirl plane, not along the axis.
  - The factor 1/2 is chosen because the transverse projector has trace 2.
        """
    )


def print_trace_check():
    print("\n" + SEP)
    print("  2. Trace check")
    print(SEP)
    print(
        r"""
  Since

      g^{mu nu} h_{mu nu} = 3,
      g^{mu nu} omegahat_mu omegahat_nu = 1,

  the transverse projector has trace

      g^{mu nu} P^perp_{mu nu} = 2.

  Therefore

      T^(rot) = g^{mu nu} T^(rot)_{mu nu}
              = (1/2) chi T^(m) [g^{mu nu} P^perp_{mu nu}]
              = chi T^(m).

  So in the coherent-vortex limit A_vort -> 1,

      T^(rot) = sqrt(X0/X) T^(m) = (a0/g) T^(m) .

  This is exactly the source-side MOND completion we wanted.
        """
    )


def print_rest_frame_example():
    print("\n" + SEP)
    print("  3. Rest-frame example")
    print(SEP)

    # Simple Minkowski rest frame, vortex axis = z
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    u = np.array([1.0, 0.0, 0.0, 0.0])  # use c=1 convention only for this algebraic demo
    h = g + np.outer(u, u)
    omegahat = np.array([0.0, 0.0, 0.0, 1.0])
    p_perp = h - np.outer(omegahat, omegahat)
    trace = np.trace(g @ p_perp)

    print("  In the local rest frame with vortex axis along z:")
    print(f"    h_mu_nu trace        = {np.trace(g @ h):.1f}")
    print(f"    P_perp trace         = {trace:.1f}")
    print("    P_perp acts only in the x-y swirl plane.")
    print()
    print("  Matrix form of P_perp:")
    for row in p_perp:
        print("   ", " ".join(f"{x:5.1f}" for x in row))


def print_scalar_equation():
    print("\n" + SEP)
    print("  4. Scalar equation with the new covariant source")
    print(SEP)
    print(
        r"""
  The scalar equation becomes

      Box(phi) = -(8piG/c^4) [ T^(m) + T^(rot) ]

  hence

      Box(phi) = -(8piG/c^4) [ 1 + chi(X,omega) ] T^(m)

  and in the coherent-vortex galaxy regime

      Box(phi) = -(8piG/c^4) [ 1 + a0/g ] T^(m) .

  In the nonrelativistic limit this implies

      g = g_N + (a0/g) g_N ,

  i.e. the simple MOND closure exactly.
        """
    )


def run_numerics():
    print("\n" + SEP)
    print("  5. Numerical check")
    print(SEP)

    xi = np.geomspace(0.01, 100.0, 600)
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    chi = chi_from_g(g)
    z = z_rot(g)

    # In coherent-vortex regime, chi = a0/g = Z/(1-Z)
    rel_id = np.max(np.abs(chi - z / (1.0 - z)) / chi)
    closure = np.max(np.abs(g - g_n * (1.0 + chi)) / g)

    print(f"  max relative identity error   = {rel_id:.3e}")
    print(f"  max relative closure error    = {closure:.3e}")
    print()
    print("    r/r_M    g_N/a0     g/a0      chi=a0/g    Z/(1-Z)")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx]/a0:8.3f}   "
            f"{g[idx]/a0:8.3f}   "
            f"{chi[idx]:10.3f}   "
            f"{(z[idx]/(1.0-z[idx])):10.3f}"
        )

    idx_rm = np.argmin(np.abs(xi - 1.0))
    print("\n  At r ~ r_M:")
    print(f"    g/a0         = {g[idx_rm]/a0:.6f}")
    print(f"    chi          = {chi[idx_rm]:.6f}")
    print("    So the new trace source is already O(1) there.")


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        """
  This is the cleanest covariant version so far of what you were saying:

  - the vortex creates a transverse stress in the swirl plane
  - that stress has a nonzero trace
  - the trace acts as a new direct source in the scalar equation
  - in the galaxy vortex regime its strength is a0/g times the ordinary source

  So MOND appears not because tiny frame-dragging gets amplified,
  but because the vortex contributes a genuinely new pressure-deficit source.

  Program role:
  this script records the covariant source completion in a form that can
  be carried into the scalar equation.
  Phase O now supplies an explicit action-based rotational realization
  of this same tensor via a coarse-grained swirl-phase sector.
  The remaining derivation task is to obtain that sector directly from
  the fundamental ISPG ontology rather than insert it algebraically.
        """
    )


if __name__ == "__main__":
    print_covariant_setup()
    print_tensor_form()
    print_trace_check()
    print_rest_frame_example()
    print_scalar_equation()
    run_numerics()
    print_interpretation()
