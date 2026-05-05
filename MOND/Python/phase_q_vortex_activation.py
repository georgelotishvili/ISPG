"""
Phase Q: Hysteretic derivation of the vortex activation factor
==============================================================

Purpose:
  Replace the ad hoc activation factor A_vort(omega) by a minimal
  hysteretic phase-ordering law tied to the already established coherence
  rate Omega_tr = a0 / c. Phase S then shows that this same law is the
  coarse-grained coherent-fraction equation of the source-tail ensemble.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import Omega_tr_conj, a0, c, lambda_H, r_M
from source import g_newton

SEP = "=" * 78


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def omega_circ(g, r):
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.sqrt(g / r)


def A_vort_from_rates(omega, gamma_mem=Omega_tr_conj):
    omega = np.asarray(omega, dtype=float)
    return omega / (omega + gamma_mem)


def sigma_over_T(g, omega):
    return A_vort_from_rates(omega) * a0 / np.asarray(g, dtype=float)


def print_setup():
    print(SEP)
    print("  PHASE Q: Vortex Activation from Hysteretic Coherence")
    print(SEP)
    print(
        r"""
  Define the local transverse swirl-direction unit vector

      n^A = (cos Theta, sin Theta),

  and the coarse-grained phase-order parameter

      A_vort := | < n^A > |,

  with

      0 <= A_vort <= 1.

  Physical meaning:

      A_vort = 0   -> no coherent vortex order
      A_vort = 1   -> mature phase-locked vortex
        """
    )


def print_kinetic_derivation():
    print("\n" + SEP)
    print("  1. Minimal phase-ordering kinetics")
    print(SEP)
    print(
        r"""
  Two competing local processes determine A_vort:

  (i)  Alignment:
       ordered macroscopic circulation sweeps the swirl phase through a cell
       at the coarse-grained vortex ordering rate

           omega := sqrt(omega_mu omega^mu)

       and tends to align the local swirl direction.

  (ii) Forgetting:
       the hysteretic medium loses phase memory at the macroscopic
       decoherence rate Gamma_mem.

  The minimal bounded local balance law is therefore

      u^mu nabla_mu A_vort
      = omega (1 - A_vort) - Gamma_mem A_vort .

  This is the minimal bounded two-state Markov law built from the two rates
  (omega, Gamma_mem) that keeps 0 <= A_vort <= 1 and has the correct limits:

      omega -> 0        => A_vort decays to 0,
      Gamma_mem -> 0    => A_vort grows to 1.
        """
    )


def print_rate_identification():
    print("\n" + SEP)
    print("  2. Identifying the forgetting rate")
    print(SEP)
    print(
        r"""
  The forgetting rate should not introduce a new galaxy scale, but it can be
  strengthened beyond a bare "no-new-scale" choice.

  The source-tail phase can remain coherent only while the tail packet stays
  inside one Hubble-coherent domain of size

      lambda_H = 2 pi c / H.

  Tail packets propagate ballistically at speed c in the backbone scalar
  medium, so the longest memory time is the coherence-crossing time

      tau_mem = lambda_H / c.

  Therefore the slowest allowed forgetting rate is fixed by the boundary
  spectral gap itself:

      Gamma_mem = 1 / tau_mem = c / lambda_H = a0 / c = Omega_tr.

  In a steady local vortex cell (u^mu nabla_mu A_vort = 0), the activation
  factor becomes

      A_vort(omega)
      = omega / (omega + Omega_tr)
      = 1 / (1 + Omega_tr / omega).

  This is no longer an arbitrary free function:
  it is fixed by the competition between coarse-grained vortex ordering and
  ballistic loss of phase memory across the largest coherent domain.
  Phase S derives the same law again from the coherent/incoherent packet
  balance of the source-tail ensemble.
        """
    )


def print_swirl_action_closure():
    print("\n" + SEP)
    print("  3. Closed swirl-source coefficient")
    print(SEP)
    print(
        r"""
  The swirl-source coefficient becomes

      Sigma(X,omega,T^(m))
      = A_vort(omega) sqrt(X0 / X) T^(m)

      = [ omega / (omega + a0/c) ] [ a0 / g ] T^(m).

  Hence the coarse-grained vortex stress is

      <T^(Theta)_{mu nu}>_az
      = (1/2) [ omega / (omega + a0/c) ] [ a0 / g ] T^(m) P^perp_{mu nu}.

  In the mature galactic regime omega >> a0/c this reduces to the earlier
  coherent-vortex limit

      <T^(Theta)_{mu nu}>_az
      -> (1/2) (a0/g) T^(m) P^perp_{mu nu}.
        """
    )


def run_numbers():
    print("\n" + SEP)
    print("  4. Galaxy-scale numerical check")
    print(SEP)

    xi = np.geomspace(0.05, 100.0, 600)
    r = xi * r_M
    g_n = g_newton(xi)
    g = g_total_from_simple_mu(g_n)
    omega = omega_circ(g, r)
    A = A_vort_from_rates(omega)
    sigma_ratio = sigma_over_T(g, omega)

    print(f"  Omega_tr = a0/c = {Omega_tr_conj:.4e} s^-1")
    print()
    print("    r/r_M    omega/Omega_tr    A_vort      a0/g      Sigma/T^m")
    for factor in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {factor:5.1f}   "
            f"{omega[idx]/Omega_tr_conj:14.3f}   "
            f"{A[idx]:8.4f}   "
            f"{(a0/g[idx]):8.3f}   "
            f"{sigma_ratio[idx]:11.3f}"
        )

    idx_rm = np.argmin(np.abs(xi - 1.0))
    idx_outer = np.argmin(np.abs(xi - 30.0))
    print("\n  Interpretation:")
    print(f"    At r ~ r_M,     A_vort = {A[idx_rm]:.6f}  (mature vortex)")
    print(f"    At r ~ 30 r_M,  A_vort = {A[idx_outer]:.6f}  (still significant)")


def print_interpretation():
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        """
  This closes the last free factor at the mesoscopic level:

  - The swirl-phase action already supplied the tensor structure.
  - The hysteretic ordering law now supplies the activation amplitude.
  - No new galaxy-tuned scale is introduced.

  Physical picture:
  each local vortex cell is continuously ordered by the galaxy's macroscopic
  coherent circulation and continuously disordered by the medium's forgetting.
  Their competition fixes how much of the swirl stress survives after
  coarse-graining.

  Domain note:
  omega must be read as the ordering rate of a mature macroscopic vortex,
  not as the Kepler frequency of an arbitrary small bound system.

  Program role:
  this is now a mesoscopic mean-field law with an explicit tail-ensemble
  derivation (Phase S), though not yet a full microscopic derivation from the
  underlying oscillon ensemble. But it is no longer an arbitrary function.
        """
    )


if __name__ == "__main__":
    print_setup()
    print_kinetic_derivation()
    print_rate_identification()
    print_swirl_action_closure()
    run_numbers()
    print_interpretation()
