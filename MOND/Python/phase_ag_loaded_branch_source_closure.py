"""
Phase AG: Loaded-branch source closure — impedance-matching derivation
======================================================================

Goal:
  Derive (not choose) the nonlinear source invariant I_hat from the
  backbone rotational-medium structure established in Phase AF.

Key result — impedance-matching principle:
  The backbone source ratio is  Upsilon_rot = -I_hat * Z'(y).
  For Z_rot = 1/(1+sqrt(y)), Z'(y) = -1/[2 sqrt(y) K^2].

  Parametrise  I_hat = alpha K^n A_vort.  Then
      Upsilon = (alpha/2) K^{n-2} A / sqrt(y).

  Require: Upsilon depends only on activation A and acceleration g/a0,
  with no residual dependence on the medium stiffness K_rot.
  Since K = 1 + g/a0 varies with position, K^{n-2} = const  iff  n = 2.
  Coefficient alpha = 2 from the two transverse directions.

  Result:
      I_hat = 2 K^2 A_vort    =>    Upsilon_rot = A_vort a0/g.

  This is impedance matching: the invariant compensates for the K^{-2}
  suppression in Z', so the net source depends on activation alone.

Consequences:
  - the old epsilon-suppressed frame-dragging slot is only a seed / trigger,
  - the mature-branch amplitude is set by the loaded medium,
  - the universal rate Omega_tr = a0/c is the coherence-loss rate of the
    branch, not the direct amplitude multiplying phi_N.
"""

from pathlib import Path
import io
import sys

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import Omega_tr_conj, a0, c, r_M
from frame_dragging import omega_FD
from source import g_newton

SEP = "=" * 78


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def y_from_g(g):
    g = np.asarray(g, dtype=float)
    return (g / a0) ** 2


def k_rot(y):
    y = np.asarray(y, dtype=float)
    return 1.0 + np.sqrt(y)


def z_rot(y):
    return 1.0 / k_rot(y)


def dz_dy(y):
    y = np.asarray(y, dtype=float)
    return -1.0 / (2.0 * np.sqrt(y) * (1.0 + np.sqrt(y)) ** 2)


def omega_circ(g, r):
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.sqrt(g / r)


def a_vort_from_rates(omega, gamma_mem=Omega_tr_conj):
    omega = np.asarray(omega, dtype=float)
    return omega / (omega + gamma_mem)


def solve_loaded_branch_self_consistently(g_n, r, max_iter=200, tol=1e-12):
    """Solve the occupied loaded branch without feeding the MOND root."""
    g_n = np.asarray(g_n, dtype=float)
    r = np.asarray(r, dtype=float)
    g = np.maximum(g_n, 1e-30)
    rel_change = np.inf
    it_used = 0

    for it in range(max_iter):
        omega = omega_circ(g, r)
        a_vort = a_vort_from_rates(omega)
        disc = g_n**2 + 4.0 * a_vort * a0 * g_n
        g_new = 0.5 * (g_n + np.sqrt(np.maximum(disc, 0.0)))
        rel_change = np.max(np.abs(g_new - g) / np.maximum(g_new, 1e-30))
        g = g_new
        it_used = it + 1
        if rel_change < tol:
            break

    omega = omega_circ(g, r)
    a_vort = a_vort_from_rates(omega)
    return g, a_vort, {"iterations": it_used, "fixed_point_residual": rel_change}


def i_hat_loaded_branch(y, a_vort):
    r"""
    Uniquely selected nonlinear invariant of the mature loaded branch:

      I_hat_branch = 2 K_rot(y)^2 A_vort .

    Derivation (impedance-matching principle):
      The rotational source ratio is  Upsilon = -I_hat * Z'(y).
      For  Z_rot = 1/(1+sqrt(y)),  Z'(y) = -1 / [2 sqrt(y) K^2].
      So  Upsilon = I_hat / (2 sqrt(y) K^2).

      Physical requirement: Upsilon should be controlled only by the
      activation fraction A_vort, not by the medium stiffness K_rot
      itself.  Parametrise I_hat = alpha * K^n * A_vort.  Then:

          Upsilon = (alpha/2) * K^{n-2} * A / sqrt(y).

      For Upsilon to depend on A and y alone (and not on K separately),
      the K-dependent factor K^{n-2} must be a constant, which requires
      n = 2.  Then alpha = 2 from the two transverse swirl directions
      (trace of P_perp in axisymmetric geometry).

    This is impedance matching: the coherent source invariant compensates
    for the K^{-2} suppression in Z'(y), so the net rotational source
    depends only on how much of the medium is coherently rotating.
    """
    y = np.asarray(y, dtype=float)
    a_vort = np.asarray(a_vort, dtype=float)
    return 2.0 * k_rot(y) ** 2 * a_vort


def upsilon_loaded_branch(y, a_vort):
    y = np.asarray(y, dtype=float)
    a_vort = np.asarray(a_vort, dtype=float)
    return -i_hat_loaded_branch(y, a_vort) * dz_dy(y)


def print_setup():
    print(SEP)
    print("  PHASE AG: Loaded-Branch Source Closure")
    print(SEP)
    print(
        r"""
  INPUT FROM PHASE AF:

      g_v = Upsilon_rot g_N ,

  with Upsilon_rot the projected source ratio of the backbone rotational
  backreaction.

  The next problem is to build the nonlinear branch that gives the required
  O(1) source ratio without asking the tiny GR frame-dragging amplitude to
  become huge by operator amplification.
        """
    )


def print_branch_construction():
    print("\n" + SEP)
    print("  1. Nonlinear Loaded-Branch Construction  (impedance-matching derivation)")
    print(SEP)
    print(
        r"""
  Use the Appendix-8 loading variable

      y = X / X0 = (g/a0)^2,
      K_rot = 1 + sqrt(y),
      Z_rot = 1 / K_rot .

  STEP A — Source structure.
  From Phase AF, the projected source ratio is
      Upsilon_rot = - I_hat * Z_rot'(y),
  where
      Z_rot'(y) = -1 / [ 2 sqrt(y) K_rot^2 ] .
  Hence
      Upsilon_rot = I_hat / [ 2 sqrt(y) K_rot^2 ] .

  STEP B — Impedance-matching principle.
  Parametrise the unknown invariant as  I_hat = alpha K^n A_vort.
  Then
      Upsilon_rot = (alpha/2) K^{n-2} A_vort / sqrt(y) .

  Requirement: Upsilon_rot should be controlled only by the activation
  fraction A_vort and the dimensionless acceleration sqrt(y) = g/a0,
  without any residual dependence on the medium stiffness K_rot.

  Since K = 1 + sqrt(y) varies with radius, K^{n-2} is a constant
  only when  n - 2 = 0,  i.e.  n = 2.   Then  alpha = 2  from the
  two transverse swirl directions (axisymmetric trace of P_perp).

  STEP C — Result.
  The uniquely selected invariant is

      I_hat_branch = 2 K_rot^2 A_vort ,

  and the source ratio becomes

      Upsilon_rot = A_vort / sqrt(y) = A_vort a0 / g .

  So the exact mature-branch MOND closure is obtained when A_vort -> 1:

      g_v = (a0/g) g_N .
        """
    )


def print_operating_rate():
    print("\n" + SEP)
    print("  2. Operating Rate as Branch-Loss Rate")
    print(SEP)
    print(
        r"""
  In this reading, Omega_tr no longer multiplies phi_N directly.
  Its role is only to control how quickly the coherent branch loses order.

  Use the bounded branch-occupancy law

      dA_vort/dt = omega_seed (1 - A_vort) - Omega_tr A_vort ,

  with stable fixed point

      A_vort^* = omega_seed / (omega_seed + Omega_tr).

  The universal loss rate is fixed by the Hubble coherence boundary:

      t_coh = lambda_H / c,
      Omega_tr = 1 / t_coh = c / lambda_H = a0 / c.

  So:

  - the seed rate omega_seed only decides whether the branch is occupied,
  - the loaded-medium invariant fixes the O(1) amplitude once the branch
    is occupied,
  - the old epsilon gap is therefore re-read as a branch-occupancy problem,
    not as a need to amplify the operator-level GR source by 10^6.
        """
    )


def run_impedance_uniqueness():
    """Show that n=2 is uniquely selected by impedance matching."""
    print("\n" + SEP)
    print("  2b. Impedance-Matching Uniqueness Check")
    print(SEP)

    xi = np.array([0.3, 1.0, 3.0, 10.0], dtype=float)
    g_n = g_newton(xi)
    g_tot = g_total_from_simple_mu(g_n)
    y = y_from_g(g_tot)
    K = k_rot(y)

    print(
        "    r/r_M    g/a0      K_rot      "
        "Ups(n=0)     Ups(n=1)     Ups(n=2)     Ups_MOND"
    )
    for i, x in enumerate(xi):
        ups_mond = a0 / g_tot[i]
        sq_y = np.sqrt(y[i])
        results = []
        for n in [0, 1, 2]:
            ups_n = K[i] ** (n - 2) / sq_y
            results.append(ups_n)
        print(
            f"    {x:5.1f}   "
            f"{g_tot[i] / a0:7.3f}   "
            f"{K[i]:8.3f}   "
            + "   ".join(f"{r:11.3e}" for r in results)
            + f"   {ups_mond:11.3e}"
        )

    print(
        r"""
  Reading:
  - For n != 2,  K^{n-2}  varies with radius, so Upsilon_rot would
    acquire a K-dependent factor that spoils the universal MOND relation.
  - Only n = 2 makes K^{n-2} = K^0 = 1 at every radius.
  - This is impedance matching: the invariant compensates for the
    medium's stiffness, so the source depends on activation only.
        """
    )


def run_branch_check():
    print("\n" + SEP)
    print("  3. Branch Diagnostics")
    print(SEP)

    xi = np.array([0.3, 1.0, 3.0, 10.0], dtype=float)
    r = xi * r_M
    g_n = g_newton(xi)
    g_tot, a_orb, fp_diag = solve_loaded_branch_self_consistently(g_n, r)
    y = y_from_g(g_tot)

    omega_fd = omega_FD(xi)

    a_fd = a_vort_from_rates(omega_fd)

    ups_fd = upsilon_loaded_branch(y, a_fd)
    ups_orb = upsilon_loaded_branch(y, a_orb)
    ups_sat = upsilon_loaded_branch(y, np.ones_like(y))
    ups_mond = a0 / g_tot

    i_fd = i_hat_loaded_branch(y, a_fd)
    i_orb = i_hat_loaded_branch(y, a_orb)
    i_sat = i_hat_loaded_branch(y, np.ones_like(y))

    print(
        "    r/r_M    g/a0    A_fd        A_orb      "
        "U_fd        U_orb      U_sat      U_MOND"
    )
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{g_tot[i] / a0:6.3f}   "
            f"{a_fd[i]:9.3e}   "
            f"{a_orb[i]:9.3e}   "
            f"{ups_fd[i]:9.3e}   "
            f"{ups_orb[i]:9.3e}   "
            f"{ups_sat[i]:9.3e}   "
            f"{ups_mond[i]:9.3e}"
        )

    print("\n  Corresponding branch invariants:")
    print("    r/r_M    I_fd        I_orb      I_sat")
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{i_fd[i]:9.3e}   "
            f"{i_orb[i]:9.3e}   "
            f"{i_sat[i]:9.3e}"
        )

    print(
        f"""
  Fixed universal loss rate:
    Omega_tr = a0/c = {Omega_tr_conj:.4e} s^-1
    self-consistent branch iterations = {fp_diag['iterations']}
    final fixed-point residual        = {fp_diag['fixed_point_residual']:.3e}

  Reading:
  - Bare frame-dragging occupancy A_fd stays tiny, so by itself it does NOT
    populate the mature branch.
  - A mature-vortex ordering rate of orbital scale gives A_orb ~ 1 and
    therefore Upsilon_rot ~ a0/g.
  - Once A_vort ~ 1, the source ratio is exactly the MOND value and no
    epsilon-suppressed operator amplification is needed.
        """
    )

    return {
        "xi": xi,
        "g_tot": g_tot,
        "a_fd": a_fd,
        "a_orb": a_orb,
        "ups_fd": ups_fd,
        "ups_orb": ups_orb,
        "ups_sat": ups_sat,
        "ups_mond": ups_mond,
    }


def print_summary(results):
    idx_rm = int(np.argmin(np.abs(results["xi"] - 1.0)))
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        f"""
  Stage-2 result (impedance-matching derivation):

  1. The impedance-matching principle uniquely selects
         I_hat_branch = 2 K_rot^2 A_vort
     (n=2 is the only power that cancels the K^{-2} in Z'),
     giving
         Upsilon_rot = A_vort a0/g .

  2. Therefore the mature occupied branch A_vort -> 1 yields
         g_v = (a0/g) g_N
     exactly.

  3. The universal rate Omega_tr = a0/c is now read as the coherence-loss
     rate of A_vort, not as the direct amplitude multiplying phi_N.

  4. At r ~ r_M:
     - A_fd          = {results['a_fd'][idx_rm]:.6e}
     - A_orb         = {results['a_orb'][idx_rm]:.6e}
     - Upsilon_fd    = {results['ups_fd'][idx_rm]:.6e}
     - Upsilon_orb   = {results['ups_orb'][idx_rm]:.6e}
     - Upsilon_MOND  = {results['ups_mond'][idx_rm]:.6e}

  This constructs the route that bypasses the old epsilon gap:
  the small rotational slot acts only as a trigger, while the O(1) MOND
  amplitude is fixed by the occupied loaded-medium source branch.

  What still remains next:
  formalize when a system legitimately sits on the mature branch and derive
  that applicability criterion from the rotating-medium equations.
        """
    )


if __name__ == "__main__":
    print_setup()
    print_branch_construction()
    run_impedance_uniqueness()
    print_operating_rate()
    results = run_branch_check()
    print_summary(results)
