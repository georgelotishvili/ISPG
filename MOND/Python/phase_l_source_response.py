"""
Phase L: Source-side constitutive response for rotational MOND
==============================================================

Motivation:
  Phase K showed that modifying only the rotational operator is too weak:
  it gives at most O(1) enhancement of the GR frame-dragging response.

Source-side closure:
  The rotating medium contributes a NEW effective source term directly
  in the pressure/scalar channel, not just an operator correction in g_{0i}.

Core constitutive split:
  loaded fraction  f_load = 1 - Z_rot
  free fraction    f_free = Z_rot

If the static Bernoulli source is produced by the loaded fraction and the
centrifugal/rotational source is produced by the free fraction under the same
orbital forcing, then

    S_rot / S_N = f_free / f_load = Z_rot / (1 - Z_rot) = a0 / g .

This gives the MOND closure exactly.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, eps, r_M
from source import g_newton

SEP = "=" * 76


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def z_rot(g):
    g = np.asarray(g, dtype=float)
    return 1.0 / (1.0 + g / a0)


def f_load(g):
    return 1.0 - z_rot(g)


def source_ratio(g):
    z = z_rot(g)
    return z / (1.0 - z)


def print_setup():
    print(SEP)
    print("  PHASE L: Source-Side Rotational Response")
    print(SEP)
    print(
        """
  Previous result:
    operator-only rotational modification is too weak.

  Source-side closure:
    let the rotating medium generate an EXTRA effective source in the
    scalar/pressure equation.

  Physical picture:
    - loaded fraction of the medium is stuck to the static pressure well
    - free fraction of the medium participates in rotational redistribution
    - both fractions feel the same orbital forcing
    - therefore source strengths scale like free/loaded fractions
        """
    )


def print_derivation():
    print("\n" + SEP)
    print("  1. Constitutive source split")
    print(SEP)
    print(
        r"""
  Define

      Z_rot = 1 / (1 + g/a0)

  Then

      f_free  = Z_rot
      f_load  = 1 - Z_rot

  Assume:

      S_N   = static Bernoulli source from the loaded fraction
      S_rot = centrifugal rotational source from the free fraction

  If both are driven by the same local orbital forcing scale, then

      S_rot / S_N = f_free / f_load
                  = Z_rot / (1 - Z_rot)
                  = a0 / g

  Therefore

      S_rot = (a0 / g) S_N .

  Since the Newtonian channel gives g_N, the rotational channel gives

      g_h = (a0 / g) g_N

  and the total field satisfies

      g = g_N + g_h
        = g_N + (a0 / g) g_N .

  So

      g^2 = g g_N + a0 g_N ,

  which is exactly the simple MOND algebra.
        """
    )


def print_field_equation_form():
    print("\n" + SEP)
    print("  2. Effective field equation form")
    print(SEP)
    print(
        r"""
  Write the scalar equation schematically as

      Box(phi) = S_N + S_rot

  and identify

      S_rot = [ Z_rot / (1 - Z_rot) ] S_N .

  Equivalently,

      Box(phi) = S_N / (1 - Z_rot)

  or in effective-trace language

      Box(phi) = -(8piG/c^4) [ T_eff^(N) + T_eff^(rot) ]
      T_eff^(rot) = [ Z_rot / (1 - Z_rot) ] T_eff^(N) .

  In this script we record the source-side constitutive completion
  suggested by the medium split. The remaining derivation task is to
  obtain the same relation from the microscopic rotational sector.
        """
    )


def run_numerics():
    print("\n" + SEP)
    print("  3. Numerical check")
    print(SEP)

    xi = np.geomspace(0.01, 100.0, 600)
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    z = z_rot(g)
    load = f_load(g)
    ratio = source_ratio(g)
    g_h = ratio * g_n

    closure_err = np.max(np.abs(g - (g_n + g_h)) / g)
    mu_num = g_n / g
    mu_ana = (g / a0) / (1.0 + g / a0)
    mu_err = np.max(np.abs(mu_num - mu_ana))

    print(f"  max relative closure error = {closure_err:.3e}")
    print(f"  max absolute mu error      = {mu_err:.3e}")
    print(f"  old perturbative gap       = 1/eps = {1.0/eps:.3e}")
    print("  new source route: no explicit eps suppression appears")

    print("\n  Sample radii:")
    print("    r/r_M    g_N/a0     g/a0     Z_rot   free/load   g_h/g_N")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx]/a0:8.3f}   "
            f"{g[idx]/a0:8.3f}   "
            f"{z[idx]:7.3f}   "
            f"{(z[idx]/load[idx]):9.3f}   "
            f"{(g_h[idx]/g_n[idx]):8.3f}"
        )

    idx_rm = np.argmin(np.abs(xi - 1.0))
    print("\n  At r ~ r_M:")
    print(f"    g/a0         = {g[idx_rm]/a0:.6f}")
    print(f"    Z_rot        = {z[idx_rm]:.6f}")
    print(f"    free/load    = {z[idx_rm]/load[idx_rm]:.6f}")
    print(f"    g_h/g_N      = {g_h[idx_rm]/g_n[idx_rm]:.6f}")


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        """
  This source-side closure does something important:

  1. It bypasses the old eps-suppressed frame-dragging amplitude.
  2. It keeps the strong-field suppression automatically, because
         Z_rot -> 0  when g >> a0.
  3. It gives the exact MOND closure relation immediately.

  Role in the programme:

  - The remaining derivation task is to show from first principles why
    the rotational source tracks exactly the free/load ratio.
  - The same ratio must be embedded in the covariant rotational sector.

  Compared with the operator-only route, this is the source-side
  closure that produces O(1) MOND directly without asking a tiny
  GR frame-dragging source to become huge.
        """
    )


if __name__ == "__main__":
    print_setup()
    print_derivation()
    print_field_equation_form()
    run_numerics()
    print_interpretation()
