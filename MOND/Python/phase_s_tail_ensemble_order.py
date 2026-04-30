"""
Phase S: Tail-ensemble derivation of the vortex order parameter
===============================================================

Purpose:
  Derive the mesoscopic activation law for A_vort from a coarse-grained
  ensemble of hysteretic tail packets, rather than introducing the kinetic
  equation directly.
"""

from pathlib import Path
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, lambda_H

SEP = "=" * 78
RNG = np.random.default_rng(12345)


def A_steady(omega, gamma):
    omega = np.asarray(omega, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    return omega / (omega + gamma)


def euler_solution(omega, gamma, t, A0=0.0):
    rate = omega + gamma
    Aeq = A_steady(omega, gamma)
    return Aeq + (A0 - Aeq) * np.exp(-rate * t)


def monte_carlo_check(omega, gamma, n_packets=15000, steps=400):
    dt = 0.05 / max(omega + gamma, 1e-12)
    phases = RNG.uniform(0.0, 2.0 * np.pi, size=n_packets)

    for _ in range(steps):
        aligned = phases == 0.0
        incoherent = ~aligned

        p_align = min(omega * dt, 1.0)
        p_forget = min(gamma * dt, 1.0)

        align_draw = RNG.random(n_packets)
        forget_draw = RNG.random(n_packets)

        phases[incoherent & (align_draw < p_align)] = 0.0
        phases[aligned & (forget_draw < p_forget)] = RNG.uniform(
            0.0, 2.0 * np.pi, size=np.count_nonzero(aligned & (forget_draw < p_forget))
        )

    Z = np.mean(np.exp(1j * phases))
    return np.abs(Z)


def print_setup():
    print(SEP)
    print("  PHASE S: Tail-Ensemble Order Parameter")
    print(SEP)
    print(
        r"""
  Let each source contributor emit a hysteretic tail packet in the transverse
  swirl plane. Associate to packet a the unit phasor

      u_a = exp(i theta_a).

  The macroscopic vortex order parameter is the ensemble mean

      Z := (1/N) sum_a u_a,
      A_vort := |Z|.

  Therefore:

      A_vort = 0   -> random tail phases, no coherent vortex
      A_vort = 1   -> fully phase-locked macroscopic vortex
        """
    )


def print_packet_balance():
    print("\n" + SEP)
    print("  1. Coherent/incoherent packet balance")
    print(SEP)
    print(
        r"""
  Coarse-grain the ensemble into two sectors:

      N_coh   = number of phase-locked tail packets
      N_inc   = number of incoherent packets
      N       = N_coh + N_inc.

  In a mature source vortex:

  (i)  ordered circulation recruits incoherent packets into the coherent
       sector at the source-tail ordering rate omega;

  (ii) hysteretic forgetting randomizes coherent packets at the universal
       memory-loss rate Gamma_mem.

  Therefore

      dN_coh/dt = omega N_inc - Gamma_mem N_coh
                = omega (N - N_coh) - Gamma_mem N_coh.

  Define

      A_vort := N_coh / N.

  Then

      dA_vort/dt = omega (1 - A_vort) - Gamma_mem A_vort.

  This is exactly the mesoscopic kinetic law used in Phase Q.
        """
    )


def print_steady_state():
    print("\n" + SEP)
    print("  2. Stationary solution")
    print(SEP)
    print(
        r"""
  Setting dA_vort/dt = 0 gives

      A_vort
      = omega / (omega + Gamma_mem).

  The forgetting rate is fixed by the same coherence boundary.
  A tail packet can preserve phase only while it remains inside one coherent
  domain of size lambda_H = 2 pi c/H.  Because packet propagation is ballistic
  at speed c, the longest memory time is tau_mem = lambda_H/c, hence the
  boundary spectral gap is

      Gamma_mem = Omega_tr = a0 / c,

  one obtains

      A_vort(omega) = omega / (omega + a0/c).

  So the activation factor is the stationary coherent fraction of the
  source-tail ensemble.
        """
    )


def print_domain():
    print("\n" + SEP)
    print("  3. Domain interpretation")
    print(SEP)
    print(
        r"""
  This derivation explains the consistency domain immediately.

  The rate omega is not the orbital frequency of an arbitrary test body.
  It is the coarse-grained ordering rate of a SOURCE-TAIL ENSEMBLE that is
  large enough to sustain a macroscopic coherent vortex.

  Therefore:

  - spiral galaxy: many overlapping ordered tails -> omega > 0 and A_vort ~ 1
  - no mature macroscopic source vortex: no coherent packet reservoir
    -> effectively omega ~ 0 and A_vort ~ 0

  This is why the same formula can be active in galaxies while remaining
  inactive in systems outside its coarse-grained applicability domain.
        """
    )


def run_checks():
    print("\n" + SEP)
    print("  4. Analytic and Monte Carlo checks")
    print(SEP)

    gamma = 1.0
    omegas = [0.01, 0.1, 1.0, 10.0, 100.0]
    t = 10.0

    print("    omega/Gamma   A_steady    A(t=10)     MC estimate")
    for omega in omegas:
        a_eq = A_steady(omega, gamma)
        a_t = euler_solution(omega, gamma, t)
        a_mc = monte_carlo_check(omega, gamma)
        print(f"    {omega/gamma:10.2f}   {a_eq:8.4f}   {a_t:8.4f}   {a_mc:11.4f}")


def print_summary():
    print("\n" + SEP)
    print("  FINAL RESULT")
    print(SEP)
    print(
        """
  Phase Q is now rooted in an explicit source-tail ensemble picture:

  - A_vort is the coherent fraction of the tail ensemble.
  - Ordered circulation pumps packets into the coherent sector.
  - Hubble-regulated forgetting removes them from it.
  - The steady coherent fraction is exactly
        A_vort = omega / (omega + a0/c).

  This makes the macroscopic reading of A_vort natural rather than ad hoc.
        """
    )


if __name__ == "__main__":
    print_setup()
    print_packet_balance()
    print_steady_state()
    print_domain()
    run_checks()
    print_summary()
