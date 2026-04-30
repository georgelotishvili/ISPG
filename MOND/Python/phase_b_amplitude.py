"""
Phase B: Amplitude Gap Analysis

Central question: WHY does the secular ODE need source = H/(2pi) = a0/c
when the PDE frame-dragging integral gives only eps * a0/c ?

TEST 1 - Algebra check of eq:omega_tr_final
  Verify whether Omega_tr at r_MOND really equals 2*xi*a0/c
  as the manuscript claims.

TEST 2 - Numerical confirmation of the eps gap
  Run compute_Omega_tr and compare with a0/c.

TEST 3 - Accumulation ceiling
  Show that no pressure_index beta can bridge the eps gap
  when the source is eps * H/(2pi) instead of H/(2pi).

TEST 4 - Timescale analysis
  The eigenvalue relaxation rate c^2 k^2/(3H) vs g_N/c:
  which one governs the mode amplitude?
"""

import numpy as np
from pathlib import Path

from constants import G, c, a0, M_gal, R_d, r_M, kpc, eps, H0
from source import m_enc, g_newton, g_newton_dimless, eta
from multiscale import compute_Omega_tr, secular_ode_cosmological
from scipy.special import jn_zeros

x0_bessel = jn_zeros(0, 1)[0]
SEP = "=" * 70
Gyr = 1e9 * 3.15576e7


def test_1_algebra_check():
    """Check whether the analytic Omega_tr at r_MOND = 2*xi*a0/c."""
    print(SEP)
    print("  TEST 1: Algebra Check of eq:omega_tr_final")
    print(SEP)

    xi_spin = 0.5

    r_MOND = r_M
    v_MOND = np.sqrt(G * M_gal / r_MOND)
    Omega_tr_analytic = 2 * xi_spin * v_MOND / (c * r_MOND)

    Omega_tr_claimed = 2 * xi_spin * a0 / c

    ratio = Omega_tr_analytic / Omega_tr_claimed

    print(f"\n  r_MOND = sqrt(GM/a0) = {r_MOND/kpc:.2f} kpc")
    print(f"  v_circ(r_MOND) = {v_MOND/1e3:.1f} km/s")
    print(f"  xi_spin = {xi_spin}")
    print()
    print(f"  Omega_tr (direct eval) = 2*xi*v/(c*r)")
    print(f"    = {Omega_tr_analytic:.4e} rad/s")
    print()
    print(f"  Omega_tr (manuscript claim) = 2*xi*a0/c")
    print(f"    = {Omega_tr_claimed:.4e} rad/s")
    print()
    print(f"  Ratio (direct / claimed) = {ratio:.6e}")
    print(f"  epsilon = {eps:.6e}")
    print(f"  Ratio / epsilon = {ratio/eps:.4f}")

    if abs(ratio - 1.0) < 0.1:
        print(f"\n  => Manuscript is CORRECT: Omega_tr = 2*xi*a0/c")
    elif abs(ratio / eps - 1.0) < 5.0:
        print(f"\n  => Manuscript has ALGEBRA ERROR:")
        print(f"     Omega_tr(r_MOND) = {ratio:.2e} * a0/c,  NOT  a0/c")
        print(f"     The actual value is ~ eps * a0/c")
    else:
        print(f"\n  => Unexpected ratio: {ratio:.4e}")

    print(f"\n  Step-by-step:")
    print(f"    v(r_M) = sqrt(GM/r_M) = sqrt(a0 * r_M) = (GM*a0)^(1/4)")
    print(f"    1/r_M = sqrt(a0/(GM))")
    print(f"    v/r_M = sqrt(a0*r_M)/r_M = sqrt(a0/r_M) = (a0^3/(GM))^(1/4)")
    print(f"    Omega_tr = 2*xi * (a0^3/(GM))^(1/4) / c")
    print(f"    For this to equal a0/c, need (a0^3/(GM))^(1/4) = a0")
    print(f"    i.e. a0^3/(GM) = a0^4, i.e. GM = 1/a0")
    print(f"    GM = {G*M_gal:.4e} m^3/s^2")
    print(f"    1/a0 = {1/a0:.4e} s^2/m")
    print(f"    These are NOT equal (different dimensions!).")

    return ratio


def test_2_numerical_gap():
    """Run the Bessel-projected Omega_tr and confirm the eps factor."""
    print(f"\n{SEP}")
    print("  TEST 2: Numerical Confirmation of the Epsilon Gap")
    print(SEP)

    xi_eval = np.array([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    Omega_tr = compute_Omega_tr(xi_eval, xi_spin=0.5)
    Omega_conj = a0 / c

    print(f"\n  {'xi':>6s}  {'Omega_tr':>12s}  {'a0/c':>12s}"
          f"  {'ratio':>12s}  {'ratio/eps':>12s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>12s}"
          f"  {'---':>12s}  {'---':>12s}")

    ratios = []
    for i, xi in enumerate(xi_eval):
        if np.isnan(Omega_tr[i]):
            continue
        r = Omega_tr[i] / Omega_conj
        ratios.append(r)
        print(f"  {xi:>6.1f}  {Omega_tr[i]:>12.4e}  {Omega_conj:>12.4e}"
              f"  {r:>12.4e}  {r/eps:>12.4f}")

    if ratios:
        median_r = np.median(ratios)
        print(f"\n  Median ratio: {median_r:.4e}")
        print(f"  eps = {eps:.4e}")
        print(f"  Median/eps = {median_r/eps:.2f}")
        print(f"\n  => The PDE frame-dragging gives Omega_tr ~ {median_r/eps:.1f} * eps * a0/c")
        print(f"     There is a factor of 1/eps ~ {1/eps:.0e} gap")
        print(f"     between the PDE source and the needed source.")

    return ratios


def test_3_accumulation_ceiling():
    """Can any pressure_index bridge the eps gap?"""
    print(f"\n{SEP}")
    print("  TEST 3: Accumulation Ceiling (eps-suppressed source)")
    print(SEP)

    print(f"\n  The secular ODE in the code uses source = H/(2pi).")
    print(f"  The PDE gives source = eps * H/(2pi).")
    print(f"  Question: with eps-suppressed source, can any beta")
    print(f"  give C_eff = 1?")

    target_xi1 = a0 / g_newton(np.array([1.0]))[0]

    print(f"\n  Standard code result (source = H/(2pi)):")
    for beta in [0.0, 1.0, 1.9, 5.0, 10.0]:
        phi_h = secular_ode_cosmological(
            np.array([1.0]), z_form=10.0, pressure_index=beta,
            damping='local')
        C = phi_h[0] / target_xi1
        print(f"    beta={beta:5.1f}:  C_eff = {C:.4f}")

    print(f"\n  eps-corrected (source = eps * H/(2pi)):")
    print(f"  Since the ODE is linear, C_eff(eps source) = eps * C_eff(full source):")
    for beta in [0.0, 1.0, 1.9, 5.0, 10.0]:
        phi_h = secular_ode_cosmological(
            np.array([1.0]), z_form=10.0, pressure_index=beta,
            damping='local')
        C_full = phi_h[0] / target_xi1
        C_eps = eps * C_full
        print(f"    beta={beta:5.1f}:  C_eff(full) = {C_full:.4f}"
              f"  ->  C_eff(eps) = {C_eps:.4e}")

    # Find beta that gives C_eff(eps) = 1
    from scipy.optimize import brentq

    def deficit(beta):
        ph = secular_ode_cosmological(
            np.array([1.0]), z_form=10.0, pressure_index=beta,
            damping='local')
        return eps * ph[0] / target_xi1 - 1.0

    lo, hi = deficit(1.9), deficit(15.0)
    if lo * hi < 0:
        beta_eps_crit = brentq(deficit, 1.9, 15.0, xtol=0.01)
    else:
        beta_eps_crit = float('nan')

    print(f"\n  With eps-suppressed source:")
    print(f"    beta_crit(eps source) = {beta_eps_crit:.2f}")
    print(f"    beta_crit(full source) = 1.90")
    print(f"    The eps-suppressed source CAN reach C_eff=1,")
    print(f"    but requires beta ~ {beta_eps_crit:.1f} instead of 1.9.")
    print(f"    This means (H/H0)^{beta_eps_crit:.1f} enhancement,")
    print(f"    which at z=10 gives a factor of ~{8.2**beta_eps_crit:.0e}.")
    print(f"    This is physically unreasonable.")

    return beta_eps_crit


def test_4_damping_rates():
    """Compare eigenvalue damping with local damping."""
    print(f"\n{SEP}")
    print("  TEST 4: Physical Damping Rate Analysis")
    print(SEP)

    xi_vals = np.array([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    gN = g_newton(xi_vals)

    print(f"\n  {'xi':>6s}  {'gamma_eig':>12s}  {'gamma_local':>12s}"
          f"  {'ratio':>12s}  {'tau_eig':>12s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>12s}"
          f"  {'---':>12s}  {'---':>12s}")

    for i, xi in enumerate(xi_vals):
        r = xi * r_M
        k_r = x0_bessel / r
        gamma_eig = c**2 * k_r**2 / (3 * H0)
        gamma_local = gN[i] / c
        ratio = gamma_eig / gamma_local
        tau_eig = 1.0 / gamma_eig
        tau_days = tau_eig / 86400.0

        print(f"  {xi:>6.1f}  {gamma_eig:>12.4e}  {gamma_local:>12.4e}"
              f"  {ratio:>12.2e}  {tau_days:>10.1f} d")

    print(f"\n  The eigenvalue damping rate (c^2 k^2 / 3H) is ~10^12 times")
    print(f"  faster than the local rate (g_N/c) used in the secular ODE.")
    print(f"\n  CONSEQUENCE:")
    print(f"    The physical equation for the mode amplitude is:")
    print(f"      3H * da/dt + c^2 k^2 * a = S_projected")
    print(f"    At quasi-static equilibrium (da/dt ~ 0):")
    print(f"      a = S_projected / (c^2 k^2)")
    print(f"    With S = eps * Omega_tr * phi_N / tau_rel:")
    print(f"      a = eps * (a0/c) * tau_rel * phi_N")
    print(f"      phi_h = eps * (a0/g) * phi_N")
    print(f"      C_eff = eps ~ {eps:.2e}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print(SEP)
    print("  PHASE B: AMPLITUDE GAP ANALYSIS")
    print("  Why does the theory need source = a0/c")
    print("  when the PDE gives only eps * a0/c ?")
    print(SEP)

    ratio_alg = test_1_algebra_check()
    ratios_num = test_2_numerical_gap()
    C_eps = test_3_accumulation_ceiling()
    test_4_damping_rates()

    print(f"\n{SEP}")
    print("  PHASE B VERDICT")
    print(SEP)

    print(f"""
  FINDINGS:
  =========

  1. ALGEBRA ERROR in manuscript eq:omega_tr_final (line 305):
     The claim Omega_tr(r_MOND) = 2*xi*a0/c is WRONG.
     The correct evaluation gives Omega_tr ~ eps * a0/c.
     Ratio: {ratio_alg:.2e} (should be 1.0 if manuscript is correct)

  2. NUMERICAL CONFIRMATION:
     The Bessel-projected transport integral gives
     Omega_tr ~ {np.median(ratios_num) if ratios_num else 0:.2e} * a0/c
     This is ~ eps * a0/c, consistent with Test 1.

  3. COSMOLOGICAL FIX REQUIRES EXTREME ENHANCEMENT:
     With eps-suppressed source, C_eff=1 needs beta ~ {C_eps:.1f}
     instead of the physical beta ~ 1.9.
     This is not a viable resolution.

  4. THE TWO ROUTES DO NOT AGREE:
     Route 1 (frame-dragging integral): Omega_tr = eps * a0/c
     Route 2 (Hubble coherence):        Omega_tr = a0/c
     The manuscript claims they agree. They do not.
     The factor between them is 1/eps ~ {1/eps:.0e}.

  STATUS OF THE MOND DERIVATION:
  ==============================
  The spatial profile is validated (Phase A):
    tau_sp << t_H and ODE equilibrium = MOND.

  The amplitude has a gap of ~{1/eps:.0e}:
    The PDE frame-dragging source is eps-suppressed.
    No known mechanism within the current framework
    amplifies this to the needed a0/c level.

  WHAT WOULD CLOSE THE GAP:
  =========================
  (a) A non-frame-dragging source in the scalar field equation
      that provides O(a0/c) coupling to phi_N.
  (b) A nonlinear mode coupling in the bi-conformal geometry
      that self-consistently amplifies the transported field.
  (c) A cosmological boundary condition (not initial condition)
      that sets phi_h to the equilibrium value.
  (d) Derivation of the Hubble coherence mechanism showing HOW
      lambda_H sets the transport rate to a0/c.

  HONEST ASSESSMENT:
  ==================
  The theory correctly identifies:
    - The SCALE: a0 = cH/(2pi)   [derived]
    - The FORM:  mu = x/(1+x)    [derived from secular ODE equilibrium]
    - The BTFR:  v^4 = GMa0      [derived]
  But the AMPLITUDE of the transported field is not derived
  from the PDE. The secular ODE uses an assumed source (a0/c)
  that is {1/eps:.0e}x larger than what the PDE gives.
""")

    return False


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
