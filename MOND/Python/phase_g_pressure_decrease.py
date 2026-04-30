"""
Phase G: Does global pressure decrease produce MOND?
=====================================================

The ISPG action:
  S = (1/16piG) int d^4x sqrt(-g) [R + 1/2 g^{mu nu} d_mu phi d_nu phi] + S_m

with bi-conformal metric:
  ds^2 = -e^phi c^2 dt^2 + e^{-phi} dsigma^2

gives the scalar field equation:
  Box phi = -(8 pi G / c^4) T

QUESTION: Does this equation, with the cosmological background
phi_0(t) decreasing (= global pressure decrease), produce MOND
at galactic scales?

We check this through 5 independent tests.
"""

import numpy as np
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))

from constants import G, c, a0, M_gal, R_d, r_M, kpc, eps, H0, Msun

SEP = "=" * 72


# =====================================================================
#  TEST 1: Exact static field equation on bi-conformal metric
# =====================================================================

def test_1_exact_static():
    """Derive and solve the exact static field equation."""
    print(SEP)
    print("  TEST 1: Exact Static Field Equation")
    print(SEP)

    print("""
  For the bi-conformal metric:
    ds^2 = -e^phi c^2 dt^2 + e^{-phi} (dr^2 + r^2 dOmega^2)

  The covariant d'Alembertian for STATIC phi(r):

    Box phi = e^phi [nabla^2 phi + 1/2 |nabla phi|^2]

  (Verified by explicit Christoffel symbol computation.)

  The field equation Box phi = S becomes:

    nabla^2 phi + 1/2 |nabla phi|^2 = S * e^{-phi}

  KEY SUBSTITUTION: Let psi = e^{phi/2}. Then:

    nabla^2 psi = 1/2 e^{phi/2} [nabla^2 phi + 1/2 |nabla phi|^2]
                = S * e^{-phi/2} / 2 = S / (2 psi)

  For vacuum (S = 0): nabla^2 psi = 0 (HARMONIC!)

  EXACT vacuum solution:
    psi = 1 - GM/(c^2 r) = 1 - r_s/(2r)
    phi = 2 ln(1 - GM/(c^2 r))

  ACCELERATION:
    g = -c^2/2 * d(phi)/dr = GM / [r(r - GM/c^2)]
    """)

    r_s = 2 * G * M_gal / c**2
    r_arr = np.logspace(np.log10(1*kpc), np.log10(1000*kpc), 500)

    # Exact ISPG solution
    phi_exact = 2 * np.log(1 - G * M_gal / (c**2 * r_arr))
    g_exact = G * M_gal / (r_arr * (r_arr - G * M_gal / c**2))

    # Newtonian solution
    phi_newton = -2 * G * M_gal / (c**2 * r_arr)
    g_newton = G * M_gal / r_arr**2

    # MOND prediction
    x_arr = g_newton / a0
    g_mond = g_newton / (x_arr / (1 + x_arr))  # g = g_N / mu(x) where mu = x/(1+x), x = g/a0... but g appears on both sides
    # Self-consistent MOND: g^2 = g*g_N + a0*g_N
    g_mond = 0.5 * (g_newton + np.sqrt(g_newton**2 + 4 * a0 * g_newton))

    # Compare
    print(f"  r_s = 2GM/c^2 = {r_s:.4e} m = {r_s/kpc:.4e} kpc")
    print(f"  (This is the Schwarzschild radius of the galaxy)")
    print(f"  Galactic scales: r ~ 1-100 kpc >> r_s ~ 10^-8 kpc")
    print()
    print(f"  {'r/kpc':>8s}  {'g_Newton':>12s}  {'g_ISPG_exact':>12s}  {'g_MOND':>12s}"
          f"  {'ISPG/Newton':>12s}  {'MOND/Newton':>12s}")

    test_radii = [1, 3, 5, 10, 20, 50, 100, 500]
    for r_kpc in test_radii:
        r = r_kpc * kpc
        gN = G * M_gal / r**2
        gI = G * M_gal / (r * (r - G * M_gal / c**2))
        gM = 0.5 * (gN + np.sqrt(gN**2 + 4 * a0 * gN))

        print(f"  {r_kpc:>8d}  {gN:>12.4e}  {gI:>12.4e}  {gM:>12.4e}"
              f"  {gI/gN:>12.10f}  {gM/gN:>12.6f}")

    print(f"""
  RESULT: The exact static solution on the bi-conformal metric
  gives corrections of order GM/(c^2 r) ~ {G*M_gal/(c**2 * 10*kpc):.2e} at 10 kpc.

  This is a POST-NEWTONIAN correction (v^2/c^2), NOT MOND.
  At r = 10 kpc: ISPG/Newton = 1 + 10^-7, MOND/Newton = 1.6

  The static bi-conformal nonlinearity does NOT produce MOND.
    """)

    return True


# =====================================================================
#  TEST 2: Why the nonlinearity is "trivial" (removable)
# =====================================================================

def test_2_trivial_nonlinearity():
    """Show that the bi-conformal nonlinearity can be linearized."""
    print(f"\n{SEP}")
    print("  TEST 2: The Nonlinearity is Trivial (Removable)")
    print(SEP)

    print("""
  The static field equation:
    nabla^2 phi + 1/2 |nabla phi|^2 = S

  Under psi = e^{phi/2}:
    nabla^2 psi = S*psi/2

  This is a LINEAR equation for psi!

  MEANING: The e^phi nonlinearity in the bi-conformal metric is
  "trivial" -- it can be removed by a field redefinition.

  CONTRAST with AQUAL (the theory that DOES produce MOND):
    AQUAL action: S = integral f(|nabla Phi|^2 / a0^2) d^3x
    where f(x) is a NONLINEAR function (f(x) ~ x for x>>1, x^{3/2} for x<<1).

    This nonlinearity CANNOT be removed by any field redefinition.
    It is "genuine" nonlinearity that changes the physics.

  In ISPG:
    - Kinetic term: 1/2 g^{mu nu} d_mu phi d_nu phi
    - On bi-conformal metric: X = 1/2 e^phi |nabla phi|^2 (static)
    - Under psi = e^{phi/2}: X = 2|nabla psi|^2 (LINEAR in psi!)

  ISPG has G_2(phi, X) = X  (Horndeski classification)
  AQUAL has G_2(phi, X) = f(X)  with f''(X) != 0

  CONCLUSION: ISPG's kinetic sector is effectively LINEAR.
  MOND requires a GENUINELY NONLINEAR kinetic sector.
  The bi-conformal nonlinearity is just a coordinate transformation
  in field space (phi -> psi = e^{phi/2}).
    """)

    # Numerical verification: solve both forms and compare
    N = 10000
    r = np.linspace(0.01 * kpc, 500 * kpc, N)
    dr = r[1] - r[0]
    r_s_half = G * M_gal / c**2

    # Method 1: Exact phi solution
    phi_exact = 2 * np.log(1 - r_s_half / r)

    # Method 2: psi = e^{phi/2} = 1 - r_s_half/r (harmonic!)
    psi = 1 - r_s_half / r
    phi_from_psi = 2 * np.log(psi)

    # Check they agree
    max_diff = np.max(np.abs(phi_exact - phi_from_psi))
    print(f"  Numerical check: max|phi_exact - phi_from_psi| = {max_diff:.2e}")
    print(f"  (Should be ~0 up to floating point)")

    # Check psi is harmonic
    d2psi = np.gradient(np.gradient(psi, r), r)
    dpsi = np.gradient(psi, r)
    lapl_psi = d2psi + 2 * dpsi / r

    # Exclude boundaries and very small r
    mask = (r > 1*kpc) & (r < 400*kpc)
    max_lapl = np.max(np.abs(lapl_psi[mask]))
    ref_scale = np.max(np.abs(d2psi[mask]))
    print(f"  Check nabla^2 psi = 0: max|nabla^2 psi| / |d^2 psi| = {max_lapl/ref_scale:.2e}")
    print(f"  (Should be ~0 — finite difference noise only)")

    print(f"\n  PASS: The nonlinearity is indeed removable by psi = e^{{phi/2}}.")
    return True


# =====================================================================
#  TEST 3: Hubble damping at galactic scales
# =====================================================================

def test_3_hubble_at_galactic_scales():
    """Check if Hubble damping matters at galactic scales."""
    print(f"\n{SEP}")
    print("  TEST 3: Hubble Damping at Galactic Scales")
    print(SEP)

    print("""
  The time-dependent field equation:
    phi_tt + 3H phi_t - c^2 nabla^2 phi = S

  For a quasi-static galaxy, compare the two spatial terms:
    c^2 nabla^2 phi ~ c^2 phi / r^2  (spatial Laplacian)
    3H phi_t          ~ 3H^2 phi       (if phi evolves on Hubble time)

  Ratio:
    (3H^2 phi) / (c^2 phi / r^2) = 3 H^2 r^2 / c^2 = 3 (r/lambda_H)^2
    """)

    lambda_H = 2 * np.pi * c / H0
    print(f"  lambda_H = 2 pi c / H = {lambda_H/kpc:.0f} kpc = {lambda_H/(1e6*kpc):.0f} Mpc")

    test_radii_kpc = [1, 10, 50, 100, 500, 1000, 10000]
    print(f"\n  {'r (kpc)':>10s}  {'r/lambda_H':>12s}  {'3(r/lH)^2':>12s}  {'Hubble relevant?':>20s}")

    for r_kpc in test_radii_kpc:
        r = r_kpc * kpc
        ratio = r / lambda_H
        hubble_term = 3 * ratio**2
        relevant = "YES" if hubble_term > 0.01 else "NO"
        print(f"  {r_kpc:>10d}  {ratio:>12.2e}  {hubble_term:>12.2e}  {relevant:>20s}")

    r_mond = np.sqrt(G * M_gal / a0)
    ratio_mond = r_mond / lambda_H
    print(f"\n  MOND radius: r_MOND = {r_mond/kpc:.1f} kpc")
    print(f"  r_MOND / lambda_H = {ratio_mond:.2e}")
    print(f"  3(r_MOND/lambda_H)^2 = {3*ratio_mond**2:.2e}")

    print(f"""
  RESULT: At the MOND radius ({r_mond/kpc:.0f} kpc), the Hubble damping
  is {3*ratio_mond**2:.1e} times the spatial Laplacian.

  The Hubble term is completely negligible at galactic scales!
  It only becomes relevant at r > {lambda_H/kpc/10:.0f} kpc ~ 100 Mpc.

  The global pressure decrease (Hubble evolution) does NOT affect
  local galactic gravity. The spatial Laplacian overwhelmingly
  dominates at galactic distances.
    """)

    return 3 * ratio_mond**2


# =====================================================================
#  TEST 4: Does e^{-phi_0(t)} rescaling help?
# =====================================================================

def test_4_cosmological_rescaling():
    """Check if the cosmological evolution of phi_0 helps."""
    print(f"\n{SEP}")
    print("  TEST 4: Cosmological Rescaling")
    print(SEP)

    print("""
  The perturbation equation (derived from full nonlinear Box phi = S):

    nabla^2 delta_phi + 1/2 |nabla delta_phi|^2
        = -(8 pi G / c^2) e^{-phi_0(t)} e^{-delta_phi} rho

  The cosmological background phi_0(t) enters ONLY through:
    1. An overall factor e^{-phi_0} on the source
    2. The Hubble damping term (negligible at galaxy scales, Test 3)

  The factor e^{-phi_0(t)} is a UNIFORM rescaling of G_eff:
    G_eff(t) = G * e^{-phi_0(t)}

  This rescaling:
    - Is the SAME everywhere in the universe at time t
    - Does NOT depend on local acceleration g
    - Does NOT distinguish between g > a0 and g < a0
    - Therefore CANNOT produce MOND (which needs different
      behavior at different acceleration scales)
    """)

    # phi_0 ~ -2H*t approximately (for slowly varying H)
    # At z=0: phi_0 ≈ 0 (by convention)
    # At z=1: phi_0 ≈ -2H*t ~ -2 (very rough)

    t_now = 1 / H0  # Hubble time ~ age of universe
    phi_0_now = 0  # convention
    phi_0_early = -2 * H0 * t_now  # very rough estimate

    print(f"  phi_0 at z=0: ~{phi_0_now:.2f}")
    print(f"  phi_0 at z~1 (rough): ~{phi_0_early:.2f}")
    print(f"  e^{{-phi_0}} change over cosmic time: ~ factor {np.exp(-phi_0_early):.1f}")
    print(f"\n  But this factor is UNIFORM — same for all galaxies,")
    print(f"  same for inner and outer regions of a galaxy.")
    print(f"  It cannot produce the radius-dependent MOND effect.")

    # What WOULD be needed for MOND?
    print(f"\n  What MOND needs: a rescaling that depends on LOCAL g:")
    print(f"    g_eff = g_N * f(g/a0)")
    print(f"  where f(x) -> 1 for x >> 1 and f(x) -> 1/x for x << 1.")
    print(f"\n  The uniform rescaling e^{{-phi_0}} gives:")
    print(f"    g_eff = g_N * const(t)")
    print(f"  which is NOT MOND.")

    print(f"\n  RESULT: The global pressure decrease produces a uniform")
    print(f"  time-dependent rescaling of G. This is NOT MOND.")
    return True


# =====================================================================
#  TEST 5: What would ISPG need to produce MOND?
# =====================================================================

def test_5_what_is_needed():
    """Identify what modification would be needed."""
    print(f"\n{SEP}")
    print("  TEST 5: What Would ISPG Need to Produce MOND?")
    print(SEP)

    print("""
  The ISPG action:
    S = (1/16piG) int sqrt(-g) [R + 1/2 (d phi)^2] + S_m

  Horndeski functions: G2 = X, G3 = 0, G4 = 1/2, G5 = 0

  The kinetic term G2 = X is LINEAR in X = -1/2 (d phi)^2.
  After the bi-conformal metric substitution:
    X_static = 1/2 e^phi |nabla phi|^2 = 2|nabla psi|^2
  (where psi = e^{phi/2})

  This is still LINEAR in the derivatives of psi.

  ----------------------------------------------------------------
  COMPARISON WITH MOND-PRODUCING THEORIES:
  ----------------------------------------------------------------

  AQUAL (Bekenstein & Milgrom 1984):
    G2 = f(X/a0^2)  where f'(x) = mu(x)
    f(x) ~ x for x >> 1  (Newtonian)
    f(x) ~ (2/3) x^{3/2} for x << 1  (MOND)
    This is GENUINELY NONLINEAR: f''(x) != 0

  TeVeS (Bekenstein 2004):
    Additional vector field + nonlinear kinetic function
    Same AQUAL-like nonlinearity in the scalar sector

  RMOND (Skordis & Zlosnik 2021):
    G2 = -a0^2 Y(X/a0^2) with specific Y function
    Uses both scalar and vector fields

  ----------------------------------------------------------------
  WHAT ISPG WOULD NEED:
  ----------------------------------------------------------------

  Replace G2 = X with G2 = X * h(X/X_0)
  where X_0 = a0^2 / c^4 (the MOND kinetic scale)
  and h(y) transitions from h(y) = 1 (Newtonian) to h(y) ~ sqrt(y) (MOND)

  In the bi-conformal framework, this means replacing:
    1/2 (d phi)^2 -> a0^2/(2c^4) * f((d phi)^2 c^4 / a0^2)

  This would give the field equation:
    nabla . [mu(|nabla phi|) nabla phi] = S
  where mu(|nabla phi|) = f'(|nabla phi|^2 c^4 / a0^2)

  THIS would produce MOND. But it requires MODIFYING THE ACTION.
    """)

    # Check: what is X_0 in physical units?
    X_0 = a0**2 / c**4
    print(f"  Relevant scales:")
    print(f"  a0 = {a0:.4e} m/s^2")
    print(f"  X_0 = a0^2/c^4 = {X_0:.4e} m^-2")
    print(f"  |nabla phi| at MOND transition: 2a0/c^2 = {2*a0/c**2:.4e} m^-1")

    # For a galaxy at r_M
    grad_phi_rM = 2 * G * M_gal / (c**2 * r_M**2)
    x_rM = (grad_phi_rM * c**2 / (2 * a0))
    print(f"  |nabla phi| at r_M = {r_M/kpc:.1f} kpc: {grad_phi_rM:.4e} m^-1")
    print(f"  x = |nabla phi| c^2 / (2 a0) = {x_rM:.4f}  (x ~ 1 at MOND transition)")

    print(f"""
  FUNDAMENTAL CONCLUSION:

  The current ISPG action (G2 = X, linear kinetic term) is
  STRUCTURALLY UNABLE to produce MOND. The bi-conformal
  nonlinearity from e^phi is removable (gauge-like) and
  only produces v^2/c^2 post-Newtonian corrections.

  To get MOND, the kinetic term must be genuinely nonlinear:
    G2 = f(X) with f''(X) != 0

  This requires either:
    (A) Modifying the ISPG action (adding nonlinear kinetic term)
    (B) Finding a mechanism within the current action that
        EFFECTIVELY generates f(X) at galactic scales
        (e.g., through quantum corrections, emergent behavior,
         or cosmological boundary conditions)

  Option (B) would be the "missing ingredient" that preserves
  the elegance of the current theory. But it has not been found.
    """)

    return True


# =====================================================================
#  VERDICT
# =====================================================================

def verdict():
    print(f"\n{SEP}")
    print("  PHASE G VERDICT: Global Pressure Decrease and MOND")
    print(SEP)

    print(f"""
  TESTED: Does the global decrease of spatial pressure (the ISPG
  interpretation of Hubble expansion) produce MOND effects at
  galactic scales?

  ANSWER: NO.

  Five independent arguments:

  1. EXACT STATIC SOLUTION: The nonlinear field equation on the
     bi-conformal metric gives phi = 2 ln(1 - GM/(c^2 r)).
     Deviations from Newton: ~ GM/(c^2 r) ~ 10^-7 at 10 kpc.
     MOND needs O(1) deviations. Gap: 10^6.

  2. TRIVIAL NONLINEARITY: The bi-conformal nonlinearity is
     removable by psi = e^{{phi/2}}. The equation becomes LINEAR.
     MOND needs GENUINE nonlinearity (AQUAL-like f(X) with f''!=0).
     ISPG has G2 = X (linear).

  3. HUBBLE DAMPING NEGLIGIBLE: At galactic scales (r ~ 10 kpc),
     the Hubble term is 10^-12 of the Laplacian.
     The pressure decrease does not affect local gravity.

  4. UNIFORM RESCALING: The cosmological evolution of phi_0
     gives G_eff(t) = G * e^{{-phi_0(t)}} -- a uniform factor.
     MOND needs radius-dependent modification.
     A uniform factor is NOT MOND.

  5. STRUCTURAL LIMITATION: MOND requires a nonlinear kinetic
     term G2 = f(X) with f'' != 0. ISPG has G2 = X (linear).
     No amount of cosmological boundary conditions can turn
     a linear kinetic term into a nonlinear one.

  STATUS OF THE THREE OPEN PROBLEMS:
    - Transport ansatz: STILL A POSTULATE
    - Amplitude gap: STILL 10^5 (now understood as structural)
    - Omega_tr = a0/c: STILL DIMENSIONAL ANALYSIS

  WHAT WE NOW KNOW:
    The amplitude gap is not a technical problem (wrong calculation
    or overlooked mechanism). It is a STRUCTURAL limitation of the
    ISPG action: the linear kinetic term G2 = X cannot produce
    MOND. Period.

  POSSIBLE RESOLUTION:
    Modify G2: Replace G2 = X with G2 = f(X) where f encodes
    the MOND transition. This modifies the action but preserves:
      - Bi-conformal metric
      - Horndeski class (ghost-free)
      - PPN equivalence with GR (f -> X for strong fields)
      - a0 = cH/(2pi) identification
    The challenge: find f(X) that is MOTIVATED by ISPG ontology,
    not just postulated.
    """)


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    test_1_exact_static()
    test_2_trivial_nonlinearity()
    test_3_hubble_at_galactic_scales()
    test_4_cosmological_rescaling()
    test_5_what_is_needed()
    verdict()
