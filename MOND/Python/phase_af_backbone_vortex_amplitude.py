"""
Phase AF: Backbone-derived local vortex amplitude equation
==========================================================

Goal:
  Replace the old primary MOND transport ansatz

      phi_h / tau_rel = Omega_tr * phi_N

  by a local amplitude equation obtained directly from the sourced scalar
  equation plus the rotational backreaction term already recorded in
  Appendix 8.

Key output:
  The primary closure variable is not Omega_tr * tau_rel anymore, but the
  projected source ratio

      Upsilon_rot := S_rot^(proj) / S_N^(proj),

  so the local vortex mode obeys

      a_v'' + 3H a_v' + c^2 k_B^2 a_v
      = Upsilon_rot * c^2 k_B^2 a_N.

  In the fast local-equilibrium limit this gives

      a_v = Upsilon_rot a_N,
      g_v = Upsilon_rot g_N.

  Therefore the MOND closure condition is now

      Upsilon_rot = a0 / g,

  which is a source-side condition derived from the backbone scalar equation,
  not a transport ansatz inserted by hand.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.special import jn_zeros

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import H0, a0, c, kpc, r_M
from source import g_newton

SEP = "=" * 78
BESSEL_ZERO = jn_zeros(0, 1)[0]


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def y_from_g(g):
    """Dimensionless loading y = X / X0 = (g/a0)^2."""
    g = np.asarray(g, dtype=float)
    return (g / a0) ** 2


def z_rot(y):
    y = np.asarray(y, dtype=float)
    return 1.0 / (1.0 + np.sqrt(y))


def dz_dy(y):
    y = np.asarray(y, dtype=float)
    return -1.0 / (2.0 * np.sqrt(y) * (1.0 + np.sqrt(y)) ** 2)


def upsilon_from_local_backreaction(y, i_hat_rot):
    """
    Local-cell slow-coefficient approximation:

      S_rot^(proj) / S_N^(proj) = Upsilon_rot = - I_hat_rot * Z_rot'(y),

    where I_hat_rot is the dimensionless rotational invariant measured in
    units of the MOND loading scale used in Appendix 8.
    """
    y = np.asarray(y, dtype=float)
    i_hat_rot = np.asarray(i_hat_rot, dtype=float)
    return -i_hat_rot * dz_dy(y)


def upsilon_mond(g):
    """Exact MOND source ratio required by g_v = (a0/g) g_N."""
    g = np.asarray(g, dtype=float)
    return a0 / g


def required_i_hat_for_mond(y):
    """Dimensionless invariant needed so Upsilon_rot reproduces MOND exactly."""
    y = np.asarray(y, dtype=float)
    return (1.0 / np.sqrt(y)) / (-dz_dy(y))


def local_bessel_scales(xi):
    xi = np.asarray(xi, dtype=float)
    r_cell = xi * r_M
    k_b = BESSEL_ZERO / r_cell
    omega_b = c * k_b
    tau_spatial = 3.0 * H0 / (c**2 * k_b**2)
    return k_b, omega_b, tau_spatial


def print_foundation():
    print(SEP)
    print("  PHASE AF: Backbone-Derived Local Vortex Amplitude Equation")
    print(SEP)
    print(
        r"""
  STARTING POINTS ALREADY PRESENT IN THE THEORY:

  (1) Backbone scalar equation (Appendix 1):

      Box(phi) = -(8 pi G / c^4) T .

  (2) Rotational scalar backreaction (Appendix 8, weak-field schematic form):

      Box(phi)
      = source_matter
        + (1 / X0) div[ I_rot * Z_rot'(X/X0) * grad(phi) ] .

  The old primary closure law was

      phi_h / tau_rel = Omega_tr * phi_N ,

  which is a transport ansatz. The goal here is to replace it by an
  amplitude equation obtained by splitting and projecting the backbone PDE.
        """
    )


def print_split_derivation():
    print("\n" + SEP)
    print("  1. Split the Scalar Field into Static + Vortex Parts")
    print(SEP)
    print(
        r"""
  Write

      phi = phi_N + phi_v ,

  where phi_N is the static Bernoulli / Newtonian part and phi_v is the
  additional rotational-vortex contribution.

  Let phi_N solve

      d^2 phi_N/dt^2 + 3H dphi_N/dt - c^2 nabla^2 phi_N = S_N .

  Subtracting from the full scalar equation gives the exact schematic
  equation for the vortex part:

      d^2 phi_v/dt^2 + 3H dphi_v/dt - c^2 nabla^2 phi_v = S_rot[phi_N, phi_v, A] .

  To leading order around the static background, the rotational source is
  evaluated on phi_N:

      S_rot^(back)
      := (1 / X0) div[ I_rot * Z_rot'(X_N/X0) * grad(phi_N) ] .
        """
    )


def print_local_projection():
    print("\n" + SEP)
    print("  2. Local Cell Projection Replaces the Old Ansatz")
    print(SEP)
    print(
        r"""
  In a local mature-vortex cell, assume:

  - the m=0 axisymmetric Bessel mode dominates,
  - coefficient gradients are slow across the cell,
  - phi_v and phi_N are projected on the same local mode shape psi_B.

  Then

      phi_N = a_N(t) psi_B(x),
      phi_v = a_v(t) psi_B(x),
      -nabla^2 psi_B = k_B^2 psi_B,
      k_B = 2.4048 / r_cell .

  In the slow-coefficient limit,

      S_rot^(back)
      ~= [ - I_hat_rot * Z_rot'(y_N) ] S_N
      := Upsilon_rot * S_N ,

  where

      y_N := X_N / X0,
      I_hat_rot := dimensionless local rotational invariant,
      Upsilon_rot := S_rot^(proj) / S_N^(proj).

  Projecting on psi_B gives the local amplitude equation

      a_v'' + 3H a_v' + c^2 k_B^2 a_v
      = Upsilon_rot * c^2 k_B^2 a_N .

  This is the new primary closure law.
  The theory now has to derive Upsilon_rot, not postulate Omega_tr * tau_rel.
        """
    )


def print_equilibrium_reading():
    print("\n" + SEP)
    print("  3. Fast Local Equilibrium")
    print(SEP)
    print(
        r"""
  Because c^2 k_B^2 >> H^2 on galactic cells, the local mode reaches
  spatial equilibrium much faster than cosmological evolution.

  Therefore the instantaneous mature-vortex branch satisfies

      a_v = Upsilon_rot a_N ,

  and hence

      g_v = Upsilon_rot g_N .

  So the old ratio g_v/g_N is no longer inserted via the transport ansatz:
  it is the projected source ratio of the backbone rotational backreaction.

  MOND exactness now means simply

      Upsilon_rot = a0 / g .
        """
    )


def run_regime_check():
    print("\n" + SEP)
    print("  4. Regime Check and Required Invariant Profile")
    print(SEP)

    xi = np.array([0.3, 1.0, 3.0, 10.0], dtype=float)
    g_n = g_newton(xi)
    g_tot = g_total_from_simple_mu(g_n)
    y_tot = y_from_g(g_tot)
    z_tot = z_rot(y_tot)

    # Natural O(1) baseline: I_hat_rot = 1.
    upsilon_unit = upsilon_from_local_backreaction(y_tot, np.ones_like(y_tot))
    upsilon_target = upsilon_mond(g_tot)
    i_hat_req = required_i_hat_for_mond(y_tot)

    k_b, omega_b, tau_spatial = local_bessel_scales(xi)

    print(
        "    r/r_M    g/a0      y=X/X0    Z_rot     "
        "Upsilon(I=1)  Upsilon_MOND   I_hat_req"
    )
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{g_tot[i] / a0:7.3f}   "
            f"{y_tot[i]:8.3f}   "
            f"{z_tot[i]:7.3f}   "
            f"{upsilon_unit[i]:11.3e}   "
            f"{upsilon_target[i]:11.3e}   "
            f"{i_hat_req[i]:10.3f}"
        )

    print("\n  Local Bessel time scale:")
    print("    r/r_M    k_B (m^-1)      c k_B (s^-1)   tau_spatial")
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{k_b[i]:11.3e}   "
            f"{omega_b[i]:11.3e}   "
            f"{tau_spatial[i] / 86400.0:11.3f} d"
        )

    print(
        """
  Reading:
  - With I_hat_rot = O(1), the backbone backreaction already gives a definite
    source ratio Upsilon_rot, but it is smaller than the exact MOND target.
  - Therefore the next job is no longer to justify an ansatz, but to derive
    the actual nonlinear rotational invariant I_hat_rot selected by the mature
    vortex branch.
  - For the Appendix-8 choice Z_rot = 1 / (1 + sqrt(y)), exact MOND would
    require

        I_hat_rot^MOND = 2 (1 + g/a0)^2 = 2 K_rot^2 .

    Phase AG shows this is uniquely selected by impedance matching:
    among I = alpha K^n A, n=2 is the only power for which K^{n-2}
    is constant, so Upsilon depends on activation alone.
        """
    )

    return {
        "xi": xi,
        "g_tot": g_tot,
        "y_tot": y_tot,
        "z_tot": z_tot,
        "upsilon_unit": upsilon_unit,
        "upsilon_target": upsilon_target,
        "i_hat_req": i_hat_req,
        "tau_spatial": tau_spatial,
    }


def print_summary(results):
    idx_rm = int(np.argmin(np.abs(results["xi"] - 1.0)))
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        f"""
  Stage-1 closure result:

  1. The old primary MOND law
         phi_h / tau_rel = Omega_tr phi_N
     is replaced by the projected backbone amplitude equation
         a_v'' + 3H a_v' + c^2 k_B^2 a_v
         = Upsilon_rot c^2 k_B^2 a_N .

  2. The local mature-vortex equilibrium is
         g_v = Upsilon_rot g_N .

  3. Therefore the central closure task is now precise:
         derive Upsilon_rot from the nonlinear rotating-medium branch.

  4. At r ~ r_M:
     - g/a0                = {results['g_tot'][idx_rm] / a0:.6f}
     - Upsilon(I_hat=1)    = {results['upsilon_unit'][idx_rm]:.6e}
     - Upsilon_MOND        = {results['upsilon_target'][idx_rm]:.6e}
     - I_hat_rot required  = {results['i_hat_req'][idx_rm]:.6f}

  This closes the first conceptual gap:
  the theory no longer needs the transport ansatz as its primary closure
  equation. What remains is to derive the nonlinear source ratio Upsilon_rot
  itself and show how the mature vortex branch reaches the MOND value.
        """
    )


if __name__ == "__main__":
    print_foundation()
    print_split_derivation()
    print_local_projection()
    print_equilibrium_reading()
    results = run_regime_check()
    print_summary(results)
