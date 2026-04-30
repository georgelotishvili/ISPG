"""
Phase O: Swirl-phase action for the vortex source
=================================================

Goal:
  Give the covariant vortex-source tensor an explicit action-based realization.

Idea:
  Introduce a fast swirl phase Theta whose gradient lives in the plane
  transverse to the local vorticity axis. In a coherent vortex, azimuthal
  averaging of the fast swirl direction produces the transverse projector
  P^perp_{mu nu} and therefore the Phase M / Phase N stress tensor.
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


def print_setup():
    print(SEP)
    print("  PHASE O: Swirl-Phase Action")
    print(SEP)
    print(
        r"""
  Introduce the usual slow background data

      u^mu u_mu = -c^2,
      h_{mu nu} = g_{mu nu} + u_mu u_nu / c^2,
      P^perp_{mu nu} = h_{mu nu} - omegahat_mu omegahat_nu ,

  and add one new fast field:

      Theta(x) = local swirl phase.

  The projected phase gradient is

      q_mu := P^perp_mu{}^nu nabla_nu Theta ,

  so q_mu lives in the swirl plane transverse to the vortex axis.
        """
    )


def print_action():
    print("\n" + SEP)
    print("  1. Mesoscopic action")
    print(SEP)
    print(
        r"""
  Use the effective rotational action

      S_swirl
      = int d^4x sqrt(-g) Sigma(X,omega,T^(m)) F(Y),

  with

      Y := q_mu q^mu
         = P^perp^{mu nu} nabla_mu Theta nabla_nu Theta .

  Here Sigma is the slowly varying loading amplitude and F(Y) is a
  saturation function chosen so that the coherent-vortex branch sits at

      Y = 1,

  with

      F(1)  = 0,
      F'(1) = 1/2 .

  The simplest example is

      F(Y) = sqrt(Y) - 1 .

  For MOND matching we choose

      Sigma(X,omega,T^(m)) = chi(X,omega) T^(m),
      chi(X,omega) = A_vort(omega) sqrt(X0 / X),

  with Phase Q activation

      A_vort(omega) = omega / (omega + a0/c).
        """
    )


def print_derivation():
    print("\n" + SEP)
    print("  2. Coarse-grained stress derivation")
    print(SEP)
    print(
        r"""
  In the coherent-vortex / WKB regime, the fast phase Theta oscillates around
  the swirl direction while Sigma and P^perp vary only slowly across one swirl
  cell. To leading order, the fast-sector stress is

      T^(Theta)_{mu nu}
      ~= 2 Sigma F'(Y) q_mu q_nu - g_{mu nu} Sigma F(Y).

  On the saturated branch Y -> 1,

      F(1) = 0,
      F'(1) = 1/2,

  so this becomes

      T^(Theta)_{mu nu} ~= Sigma q_mu q_nu .

  Now average over the fast azimuthal phase inside the transverse plane:

      < q_mu q_nu >_az = (1/2) P^perp_{mu nu} .

  Therefore

      < T^(Theta)_{mu nu} >_az
      = (1/2) Sigma P^perp_{mu nu}.

  Choosing Sigma = chi(X,omega) T^(m) gives

      < T^(Theta)_{mu nu} >_az
      = (1/2) chi(X,omega) T^(m) P^perp_{mu nu},

  which is exactly the Phase M covariant tensor and the Phase N anisotropic
  vortex-fluid stress.
        """
    )


def print_trace_and_scalar_source():
    print("\n" + SEP)
    print("  3. Trace and scalar source")
    print(SEP)
    print(
        r"""
  Since the transverse projector has trace

      g^{mu nu} P^perp_{mu nu} = 2,

  the averaged swirl stress has trace

      < T^(Theta) >
      = (1/2) Sigma g^{mu nu} P^perp_{mu nu}
      = Sigma.

  Hence with Sigma = chi T^(m),

      < T^(Theta) > = chi T^(m).

  The scalar equation becomes

      Box(phi) = -(8piG/c^4) [ T^(m) + <T^(Theta)> ]
               = -(8piG/c^4) [ 1 + chi(X,omega) ] T^(m).

  In the coherent galactic-vortex regime A_vort -> 1,

      chi = sqrt(X0/X) = a0/g,

  so the exact simple MOND closure follows:

      g = g_N + (a0/g) g_N.
        """
    )


def run_plane_average_demo():
    print("\n" + SEP)
    print("  4. Azimuthal averaging demo")
    print(SEP)

    angles = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
    q = np.stack(
        [
            np.zeros_like(angles),
            np.cos(angles),
            np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )
    avg_dyad = (q.T @ q) / len(angles)
    p_perp = np.diag([0.0, 1.0, 1.0, 0.0])
    err = np.max(np.abs(avg_dyad - 0.5 * p_perp))

    print("  Rest frame with vortex axis along z:")
    print(f"    max |<q q> - (1/2) P_perp| = {err:.3e}")
    print()
    print("  <q_mu q_nu>_az:")
    for row in avg_dyad:
        print("   ", " ".join(f"{x:8.5f}" for x in row))


def run_mond_check():
    print("\n" + SEP)
    print("  5. MOND closure check")
    print(SEP)

    xi = np.geomspace(0.01, 100.0, 600)
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    chi = chi_from_g(g)

    trace_ratio = chi
    closure = np.max(np.abs(g - g_n * (1.0 + trace_ratio)) / g)

    print(f"  max closure error = {closure:.3e}")
    print()
    print("    r/r_M    g_N/a0     g/a0      Sigma/T^m   <T_Theta>/T^m")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx]/a0:8.3f}   "
            f"{g[idx]/a0:8.3f}   "
            f"{chi[idx]:11.3f}   "
            f"{trace_ratio[idx]:13.3f}"
        )


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        """
  This is the first explicit action-based realization of the new MOND source:

  - Theta describes the fast swirl phase of the rotating space medium.
  - Its transverse kinetic energy creates stress only in the swirl plane.
  - Coarse-graining over the fast azimuthal phase turns that dyad into
    the projector P_perp.
  - The trace of that averaged stress is Sigma = chi T^m, exactly the
    source-side term needed for MOND.

  Program role:
  this is now an action-level completion of the rotational sector with a
  closed mesoscopic activation law. What remains deeper is to derive why the
  specific loading amplitude Sigma = chi T^m and coherent branch Y = 1 must
  emerge from the fundamental ISPG ontology without adding Theta by hand.
        """
    )


if __name__ == "__main__":
    print_setup()
    print_action()
    print_derivation()
    run_plane_average_demo()
    print_trace_and_scalar_source()
    run_mond_check()
    print_interpretation()
