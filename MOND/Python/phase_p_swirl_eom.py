"""
Phase P: Euler-Lagrange equations of the swirl-phase action
===========================================================

Purpose:
  Derive the field equations of the mesoscopic swirl-phase sector and show
  explicitly why the coherent-vortex branch acts as a source-side MOND
  completion rather than as a direct operator renormalization of Box(phi).
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


def F_of_Y(y):
    y = np.asarray(y, dtype=float)
    return np.sqrt(y) - 1.0


def dF_dY(y):
    y = np.asarray(y, dtype=float)
    return 0.5 / np.sqrt(y)


def print_intro():
    print(SEP)
    print("  PHASE P: Swirl-Phase Euler-Lagrange Equations")
    print(SEP)
    print(
        r"""
  Start from the effective total action

      S_tot[g,phi,Theta]
      = S_ISPG[g,phi] + S_m[g,psi] + S_swirl[g,phi,Theta],

  with

      S_swirl = int d^4x sqrt(-g) Sigma(X,omega,T^(m)) F(Y),

      X = (1/2) g^{mu nu} nabla_mu phi nabla_nu phi,
      q_mu = P^perp_mu{}^nu nabla_nu Theta,
      Y = q_mu q^mu.

  For the MOND branch:

      Sigma = chi(X,omega) T^(m),
      chi    = A_vort(omega) sqrt(X0 / X),
      A_vort = omega / (omega + a0/c),
      X0     = (1/2) (2 a0 / c^2)^2.

  Use the simple saturation function

      F(Y) = sqrt(Y) - 1,

  so that

      F(1) = 0,      F'(1) = 1/2.
        """
    )


def print_theta_equation():
    print("\n" + SEP)
    print("  1. Variation with respect to Theta")
    print(SEP)
    print(
        r"""
  Since Theta enters only through Y,

      delta Y = 2 q_mu P^perp^{mu nu} nabla_nu(delta Theta),

  hence

      delta S_swirl
      = int d^4x sqrt(-g) 2 Sigma F'(Y)
        q_mu P^perp^{mu nu} nabla_nu(delta Theta).

  After integration by parts:

      delta S_swirl
      = - int d^4x sqrt(-g)
        nabla_mu[2 Sigma F'(Y) P^perp^{mu nu} nabla_nu Theta] delta Theta.

  Therefore the Theta Euler-Lagrange equation is

      nabla_mu[2 Sigma F'(Y) P^perp^{mu nu} nabla_nu Theta] = 0.

  On the coherent saturated branch Y -> 1 this simplifies to

      nabla_mu[Sigma P^perp^{mu nu} nabla_nu Theta] = 0,

  because 2 F'(1) = 1.
        """
    )


def print_metric_variation():
    print("\n" + SEP)
    print("  2. Leading WKB metric variation")
    print(SEP)
    print(
        r"""
  At leading order in the slow-fast separation, treat Sigma and P^perp_{mu nu}
  as approximately constant across one swirl cell while varying the fast phase.
  Then

      T^(Theta)_{mu nu}
      ~= 2 Sigma F'(Y) q_mu q_nu - g_{mu nu} Sigma F(Y).

  On the saturated branch Y -> 1,

      T^(Theta)_{mu nu} ~= Sigma q_mu q_nu,

  because F(1) = 0 and 2 F'(1) = 1.

  Azimuthal averaging inside the swirl plane gives

      < q_mu q_nu >_az = (1/2) P^perp_{mu nu},

  so the coarse-grained stress becomes

      < T^(Theta)_{mu nu} >_az
      = (1/2) Sigma P^perp_{mu nu}.

  Its trace is

      < T^(Theta) > = Sigma,

  because g^{mu nu} P^perp_{mu nu} = 2.
        """
    )


def print_phi_variation():
    print("\n" + SEP)
    print("  3. Direct variation with respect to phi")
    print(SEP)
    print(
        r"""
  The field phi enters S_swirl through the loading amplitude Sigma(X,...).

      delta Sigma = Sigma_X delta X,
      delta X = nabla_mu(phi) nabla^mu(delta phi),

  therefore

      delta S_swirl
      = int d^4x sqrt(-g) Sigma_X F(Y) nabla_mu(phi) nabla^mu(delta phi)

      = - int d^4x sqrt(-g)
          nabla_mu[ Sigma_X F(Y) nabla^mu phi ] delta phi .

  So the direct scalar contribution is

      Delta_phi^(swirl)
      = - nabla_mu[ Sigma_X F(Y) nabla^mu phi ].

  Key point:
  on the coherent branch Y -> 1 we have F(1) = 0, hence

      Delta_phi^(swirl) -> 0.

  So the coherent swirl sector does NOT mainly act by renormalizing the
  differential operator of phi. Its dominant effect survives instead through
  the stress trace <T^(Theta)> = Sigma.

  This is exactly why this completion is source-side rather than operator-side.
        """
    )


def print_scalar_equation():
    print("\n" + SEP)
    print("  4. Scalar sourcing and MOND closure")
    print(SEP)
    print(
        r"""
  In the coherent-vortex regime, the scalar equation is sourced by the total
  trace:

      Box(phi) = -(8piG/c^4) [ T^(m) + <T^(Theta)> ].

  Since <T^(Theta)> = Sigma, choosing

      Sigma = chi(X,omega) T^(m)

  gives

      Box(phi) = -(8piG/c^4) [1 + chi(X,omega)] T^(m).

  In the mature galactic-vortex limit A_vort -> 1:

      chi = sqrt(X0/X) = a0/g,

  hence

      Box(phi) = -(8piG/c^4) [1 + a0/g] T^(m),

  which implies in the weak field

      g = g_N + (a0/g) g_N.

  Therefore the simple MOND closure comes from the trace source of the
  coarse-grained swirl stress, with the activation amplitude fixed by the
  hysteretic phase-ordering law of Phase Q.
        """
    )


def run_branch_check():
    print("\n" + SEP)
    print("  5. Saturation-branch check")
    print(SEP)

    y = np.array([0.90, 0.99, 1.00, 1.01, 1.10])
    f = F_of_Y(y)
    stress_coeff = 2.0 * dF_dY(y)

    print("    Y        F(Y)        2 F'(Y)      meaning")
    for yi, fi, si in zip(y, f, stress_coeff):
        meaning = "coherent branch" if np.isclose(yi, 1.0) else "near branch"
        print(f"  {yi:6.2f}   {fi:10.6f}   {si:10.6f}   {meaning}")

    print()
    print("  At Y = 1 exactly:")
    print(f"    direct phi coefficient F(1)   = {F_of_Y(1.0):.6f}")
    print(f"    stress coefficient 2F'(1)     = {2.0 * dF_dY(1.0):.6f}")
    print("  So direct operator renormalization vanishes while the stress stays finite.")


def run_mond_check():
    print("\n" + SEP)
    print("  6. MOND closure check")
    print(SEP)

    xi = np.geomspace(0.01, 100.0, 600)
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    sigma_over_T = chi_from_g(g)
    closure = np.max(np.abs(g - g_n * (1.0 + sigma_over_T)) / g)

    print(f"  max closure error = {closure:.3e}")
    print()
    print("    r/r_M    g_N/a0     g/a0      Sigma/T^m = a0/g")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{g_n[idx]/a0:8.3f}   "
            f"{g[idx]/a0:8.3f}   "
            f"{sigma_over_T[idx]:15.3f}"
        )


def print_summary():
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        """
  The swirl-phase sector now has explicit equations of motion:

  1. Theta equation:
       nabla_mu[2 Sigma F'(Y) P_perp^{mu nu} nabla_nu Theta] = 0

  2. Leading coarse-grained stress:
       <T^(Theta)_{mu nu}> = (1/2) Sigma P_perp_{mu nu}

  3. Direct phi correction:
       Delta_phi^(swirl) = -nabla_mu[Sigma_X F(Y) nabla^mu phi]

  4. On the coherent branch Y = 1:
       F(1) = 0      -> direct phi operator correction vanishes
       2F'(1) = 1    -> stress remains finite

  Therefore the coherent vortex acts as a direct new source channel.
  MOND comes from the trace of the swirl stress, not from amplifying the old
  weak rotational operator, and its activation amplitude is fixed by
  A_vort = omega / (omega + a0/c).
        """
    )


if __name__ == "__main__":
    print_intro()
    print_theta_equation()
    print_metric_variation()
    print_phi_variation()
    print_scalar_equation()
    run_branch_check()
    run_mond_check()
    print_summary()
