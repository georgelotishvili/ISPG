"""
Phase AH: C_eff as an emergent branch quantity
==============================================

Goal:
  Turn the mature-macroscopic-vortex applicability statement into an explicit
  branch criterion.

Inputs from earlier stages:

  Phase AG:
      g_v = A_vort * (a0/g) * g_N

  Therefore the quadratic closure can be written as

      g^2 = g g_N + C_eff a0 g_N ,

  with

      C_eff = A_vort

  on the local branch, up to finite-lag and finite-ensemble corrections.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.special import jn_zeros

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import Gyr, H0, Omega_tr_conj, a0, c, r_M
from frame_dragging import omega_FD
from source import g_newton

SEP = "=" * 78
BESSEL_ZERO = jn_zeros(0, 1)[0]


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def omega_orb(g, r):
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.sqrt(np.maximum(g / r, 0.0))


def a_vort(omega_seed, gamma_mem=Omega_tr_conj):
    omega_seed = np.asarray(omega_seed, dtype=float)
    return omega_seed / (omega_seed + gamma_mem)


def tau_spatial(xi):
    xi = np.asarray(xi, dtype=float)
    r_cell = xi * r_M
    k_b = BESSEL_ZERO / r_cell
    return 3.0 * H0 / np.maximum((c * k_b) ** 2, 1e-300)


def delta_spatial(omega_seed, xi):
    return np.asarray(omega_seed, dtype=float) * tau_spatial(xi)


def delta_ensemble(a_branch, n_cell):
    a_branch = np.asarray(a_branch, dtype=float)
    n_cell = np.asarray(n_cell, dtype=float)
    return np.sqrt(np.maximum(a_branch * (1.0 - a_branch), 0.0) / np.maximum(n_cell, 1e-300))


def ceff_branch(omega_seed):
    """
    Exact branch coefficient at the local source-law level:

        C_eff = A_vort = omega_seed / (omega_seed + Omega_tr) .
    """
    return a_vort(omega_seed)


def ceff_asymptotic_mature(omega_seed):
    ratio = Omega_tr_conj / np.maximum(np.asarray(omega_seed, dtype=float), 1e-300)
    return 1.0 - ratio


def print_foundation():
    print(SEP)
    print("  PHASE AH: C_eff as an Emergent Branch Quantity")
    print(SEP)
    print(
        r"""
  From Phase AG:

      g_v = A_vort * (a0/g) * g_N .

  Insert this into

      g = g_N + g_v

  to get

      g = g_N + A_vort (a0/g) g_N
      => g^2 = g g_N + A_vort a0 g_N .

  Comparing with the standard closure form

      g^2 = g g_N + C_eff a0 g_N

  gives the branch identity

      C_eff = A_vort .

  So C_eff is not a hand-inserted number anymore; it is the occupancy of the
  coherent source branch.
        """
    )


def print_corrections():
    print("\n" + SEP)
    print("  1. Correction Structure")
    print(SEP)
    print(
        r"""
  Two corrections remain around the ideal local branch:

  (i) finite spatial-lag correction

      delta_sp ~ omega_seed * tau_sp ,
      tau_sp = 3H / (c^2 k_B^2),   k_B = 2.4048 / r_cell .

  This measures whether the local mode can track the evolving ordered branch.

  (ii) finite ensemble-fluctuation correction

      delta_N ~ sqrt( A_vort (1 - A_vort) / N_cell ) ,

  where N_cell is the number of overlapping source-tail packets in one local
  Bessel cell. This is the standard binomial / mean-field fluctuation size of
  the coherent fraction.

  Therefore the local branch coefficient is

      C_eff = A_vort + O(delta_sp) + O(delta_N).
        """
    )


def run_branch_audit():
    print("\n" + SEP)
    print("  2. Spiral-Galaxy and Sparse-System Branch Audit")
    print(SEP)

    xi = np.array([0.3, 1.0, 3.0, 10.0], dtype=float)
    r = xi * r_M
    g_n = g_newton(xi)
    g_tot = g_total_from_simple_mu(g_n)

    omega_seed_orb = omega_orb(g_tot, r)
    omega_seed_fd = omega_FD(xi)

    c_orb = ceff_branch(omega_seed_orb)
    c_fd = ceff_branch(omega_seed_fd)
    d_sp_orb = delta_spatial(omega_seed_orb, xi)
    d_sp_fd = delta_spatial(omega_seed_fd, xi)

    print(
        "    r/r_M    g/a0      C_eff(orb)   C_eff(FD)   "
        "delta_sp(orb)  delta_sp(FD)"
    )
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{g_tot[i]/a0:7.3f}   "
            f"{c_orb[i]:11.6f}   "
            f"{c_fd[i]:10.3e}   "
            f"{d_sp_orb[i]:13.3e}   "
            f"{d_sp_fd[i]:11.3e}"
        )

    ratio_orb = omega_seed_orb / Omega_tr_conj
    ratio_fd = omega_seed_fd / Omega_tr_conj
    print("\n  Ordering-to-loss ratio omega_seed / Omega_tr:")
    print("    r/r_M    Q_orb       Q_FD")
    for i, x in enumerate(xi):
        print(
            f"    {x:5.1f}   "
            f"{ratio_orb[i]:9.3e}   "
            f"{ratio_fd[i]:9.3e}"
        )

    print(
        """
  Reading:
  - orbital-scale collective ordering gives C_eff very close to 1 and
    delta_sp << 1 across the galactic MOND window;
  - bare frame-dragging gives C_eff << 1, so it cannot by itself occupy the
    MOND branch;
  - therefore the distinction between galaxy and non-galaxy systems is not
    strong-field suppression alone, but whether a macroscopic coherent source
    ensemble exists to supply the collective ordering branch.
        """
    )

    return {
        "xi": xi,
        "g_tot": g_tot,
        "c_orb": c_orb,
        "c_fd": c_fd,
        "d_sp_orb": d_sp_orb,
        "d_sp_fd": d_sp_fd,
        "ratio_orb": ratio_orb,
        "ratio_fd": ratio_fd,
    }


def run_ensemble_thresholds():
    print("\n" + SEP)
    print("  3. Ensemble Thresholds for a Mature Branch")
    print(SEP)

    a_ref = np.array([0.90, 0.99, 0.999], dtype=float)
    deltas = np.array([0.1, 0.03, 0.01, 0.003], dtype=float)

    print("  Minimum N_cell required so delta_N < tolerance:")
    print("    A_vort    tol       N_cell,min")
    for a_here in a_ref:
        for tol in deltas:
            n_min = a_here * (1.0 - a_here) / (tol ** 2)
            print(f"    {a_here:6.3f}   {tol:6.3f}   {n_min:11.3f}")

    n_scan = np.array([1, 3, 10, 30, 100, 300, 1e3, 1e6], dtype=float)
    a_gal = 0.999
    delta_n = delta_ensemble(a_gal, n_scan)

    print("\n  Example for a near-saturated branch A_vort = 0.999:")
    print("    N_cell    delta_N")
    for n_here, d_here in zip(n_scan, delta_n):
        print(f"    {n_here:7.0f}   {d_here:8.3e}")

    print(
        """
  Reading:
  - once the local Bessel cell contains many overlapping source tails,
    the coherent fraction becomes sharply defined;
  - for N_cell <= O(1), the branch is too noisy to be treated as a stable
    coarse-grained vortex source;
  - this is the mathematical content of the old phrase
    "mature macroscopic vortex".
        """
    )

    return {"n_scan": n_scan, "delta_n": delta_n}


def print_criterion(results):
    idx_rm = int(np.argmin(np.abs(results["xi"] - 1.0)))
    print("\n" + SEP)
    print("  FINAL CRITERION")
    print(SEP)
    print(
        f"""
  Mature-macroscopic-vortex applicability criterion:

  A system sits on the MOND branch only if all three hold on the relevant
  local Bessel cell:

  (C1) occupancy:
       Q_occ := omega_seed / Omega_tr >> 1
       so
       A_vort = Q_occ / (1 + Q_occ) ~ 1

  (C2) adiabatic tracking:
       delta_sp := omega_seed tau_sp << 1

  (C3) ensemble coarse-graining:
       delta_N := sqrt(A_vort(1-A_vort)/N_cell) << 1
       equivalently N_cell >> 1 .

  Then

      C_eff = A_vort + O(delta_sp) + O(delta_N)
            = 1 - Omega_tr/omega_seed + small corrections.

  At r ~ r_M for the galactic collective-ordering branch:
  - C_eff(orb)    = {results['c_orb'][idx_rm]:.6f}
  - Q_occ(orb)    = {results['ratio_orb'][idx_rm]:.3e}
  - delta_sp(orb) = {results['d_sp_orb'][idx_rm]:.3e}

  At the same radius for the bare frame-dragging branch:
  - C_eff(FD)     = {results['c_fd'][idx_rm]:.3e}
  - Q_occ(FD)     = {results['ratio_fd'][idx_rm]:.3e}

  So:
  - mature spiral galaxies satisfy the occupied-branch conditions and have
    C_eff ~ 1,
  - sparse or non-macroscopic systems fall back to the tiny bare-FD branch
    and recover the GR limit.
        """
    )


if __name__ == "__main__":
    print_foundation()
    print_corrections()
    results = run_branch_audit()
    run_ensemble_thresholds()
    print_criterion(results)
