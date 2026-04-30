"""
Phase N: Anisotropic vortex-fluid interpretation of the new source
==================================================================

Goal:
  Reinterpret the covariant rotational source as the stress tensor
  of an anisotropic vortex fluid.

Standard anisotropic fluid form:

  T_{mu nu}
  = rho u_mu u_nu / c^2
    + p_perp h_{mu nu}
    + (p_par - p_perp) s_mu s_nu

where s^mu is a unit spacelike direction.

For a vortex, choose s^mu = omegahat^mu (vortex axis).
Then the transverse swirl-plane stress is

  T^(vort)_{mu nu} = p_perp ( h_{mu nu} - omegahat_mu omegahat_nu )
                   = p_perp P^perp_{mu nu} .

Matching to Phase M:

  p_perp = (1/2) chi(X,omega) T^(m),   p_par = 0,   rho_vort = 0

which reproduces the rotational MOND source tensor exactly.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, c
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


def print_formulas():
    print(SEP)
    print("  PHASE N: Anisotropic Vortex Fluid")
    print(SEP)
    print(
        r"""
  Use the standard anisotropic-fluid stress tensor

      T_{mu nu}
      = rho u_mu u_nu / c^2
        + p_perp h_{mu nu}
        + (p_par - p_perp) s_mu s_nu

  with

      h_{mu nu} = g_{mu nu} + u_mu u_nu / c^2 .

  For a vortex fluid:

      s_mu = omegahat_mu

  so

      T^(vort)_{mu nu}
      = rho_vort u_mu u_nu / c^2
        + p_perp h_{mu nu}
        + (p_par - p_perp) omegahat_mu omegahat_nu .

  Choose the minimal MOND vortex fluid:

      rho_vort = 0,
      p_par    = 0,
      p_perp   = (1/2) chi(X,omega) T^(m) .

  Then

      T^(vort)_{mu nu}
      = (1/2) chi T^(m) [ h_{mu nu} - omegahat_mu omegahat_nu ]
      = (1/2) chi T^(m) P^perp_{mu nu} ,

  which is exactly the Phase M tensor.
        """
    )


def print_action_language():
    print("\n" + SEP)
    print("  Effective action language")
    print(SEP)
    print(
        r"""
  Interpreted as an effective matter component, the new source can be
  described as an anisotropic vortex fluid with on-shell pressures

      p_perp(X,omega,T^(m)) = (1/2) A_vort(omega) sqrt(X0/X) T^(m),
      p_par  = 0.

  In standard fluid EFT language, this corresponds to adding a new matter
  sector S_vort whose metric variation yields

      T^(vort)_{mu nu}
      = rho_vort u_mu u_nu / c^2
        + p_perp h_{mu nu}
        + (p_par - p_perp) omegahat_mu omegahat_nu .

  Phase O then gives an explicit mesoscopic realization of this same tensor:
  a fast swirl phase Theta in the transverse plane whose azimuthal average
  yields (1/2) chi T^(m) P^perp_{mu nu}.
        """
    )


def print_trace_and_scalar_source():
    print("\n" + SEP)
    print("  Trace and scalar sourcing")
    print(SEP)
    print(
        r"""
  For the minimal vortex fluid:

      rho_vort = 0,   p_par = 0,   p_perp = (1/2) chi T^(m)

  the trace is

      T^(vort) = -rho_vort c^2 + 2 p_perp + p_par
               = chi T^(m).

  Therefore the scalar equation becomes

      Box(phi) = -(8piG/c^4) [ T^(m) + T^(vort) ]
               = -(8piG/c^4) [ 1 + chi ] T^(m).

  In the coherent-vortex regime chi = a0/g, so

      Box(phi) = -(8piG/c^4) [ 1 + a0/g ] T^(m),

  giving the exact simple MOND closure.
        """
    )


def run_numbers():
    print("\n" + SEP)
    print("  Numerical check")
    print(SEP)

    xi = np.geomspace(0.01, 100.0, 600)
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    chi = chi_from_g(g)
    p_perp_over_T = 0.5 * chi
    trace_ratio = 2.0 * p_perp_over_T  # because p_par = rho = 0

    closure = np.max(np.abs(g - g_n * (1.0 + trace_ratio)) / g)
    trace_err = np.max(np.abs(trace_ratio - chi))

    print(f"  max trace reconstruction error = {trace_err:.3e}")
    print(f"  max MOND closure error         = {closure:.3e}")
    print()
    print("    r/r_M    g_N/a0     g/a0      chi      p_perp/T^m   T_vort/T^m")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx]/a0:8.3f}   "
            f"{g[idx]/a0:8.3f}   "
            f"{chi[idx]:8.3f}   "
            f"{p_perp_over_T[idx]:12.3f}   "
            f"{trace_ratio[idx]:11.3f}"
        )


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        """
  This is useful because it reframes the new MOND source as something
  very concrete:

  - not a mysterious extra tensor
  - but the stress tensor of a vortex fluid with transverse pressure

  In simple words:
  the rotating space medium behaves like a fluid that develops
  pressure in the swirl plane, while remaining soft along the vortex axis.
  That transverse pressure has a trace, and that trace is exactly what
  the scalar equation sees as an extra pressure-deficit source.

  Program role:
  we now have three levels of the same idea:

    1. intuitive source picture,
    2. covariant tensor picture,
    3. anisotropic-vortex-fluid picture,
    4. explicit swirl-phase action picture.

  What still remains deeper is not the existence of an action anymore,
  but deriving that action directly from the fundamental ISPG ontology
  without inserting the swirl phase sector by hand.
        """
    )


if __name__ == "__main__":
    print_formulas()
    print_action_language()
    print_trace_and_scalar_source()
    run_numbers()
    print_interpretation()
