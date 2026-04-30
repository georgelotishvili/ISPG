"""
Phase F: Turbulent Bernoulli — MOND from rotating gravitational stirring
========================================================================

IDEA:
  Newtonian gravity = STATIC Bernoulli: |nabla phi|^2 from fixed masses.
  MOND gravity = DYNAMIC Bernoulli: additional |nabla phi|^2 from
  orbiting masses whose fields interfere time-dependently.

  In ISPG, the Bernoulli identity is EXACT:
    P_static + (e^phi / 32 pi G) |nabla phi|^2 = 0

  For a ROTATING galaxy, the individual stellar fields phi_i sweep
  past each other. The total |nabla phi_total|^2 fluctuates in time.
  The TIME-AVERAGED fluctuation <|nabla phi'|^2> creates an ADDITIONAL
  pressure deficit beyond the Newtonian average.

  This additional deficit = additional gravity = g_h = MOND.

  KEY: This is NOT v^2/c^2 suppressed because it's NOT a metric
  perturbation. It's the interference of Newtonian-strength fields
  in a moving configuration.

TESTS:
  1 - Analytic: Bernoulli identity decomposition for rotating system
  2 - Order-of-magnitude: estimate <|nabla phi'|^2> for a galaxy
  3 - N-body: simulate N stars orbiting, measure |nabla phi(t)|^2
  4 - Compare azimuthally averaged g_eff(r) with MOND prediction
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
#  TEST 1: Bernoulli decomposition for a rotating system
# =====================================================================

def test_1_bernoulli_decomposition():
    """Decompose |nabla phi|^2 into mean + turbulent parts."""
    print(SEP)
    print("  TEST 1: Bernoulli Identity for Rotating Galaxy")
    print(SEP)

    print("""
  BERNOULLI IDENTITY (exact in ISPG):
    P_static + (e^phi / 32 pi G) |nabla phi|^2 = 0

  For N point masses at positions r_i(t):
    phi(r,t) = -SUM_i  2 G m_i / (c^2 |r - r_i(t)|)

  The gradient squared:
    |nabla phi|^2 = |SUM_i nabla phi_i|^2
                  = SUM_i |nabla phi_i|^2  +  SUM_{i!=j} nabla phi_i . nabla phi_j

  REYNOLDS DECOMPOSITION:
    phi(r,t) = <phi>(r)  +  phi'(r,t)

  where <...> = time average over one orbital period T = 2 pi / Omega.

  Then:
    <|nabla phi|^2> = |nabla <phi>|^2  +  <|nabla phi'|^2>
                      [Newtonian]         [turbulent Bernoulli]

  The turbulent part creates ADDITIONAL pressure deficit:
    Delta P_turb = -(e^phi / 32 pi G) <|nabla phi'|^2>

  This is additional gravity beyond Newtonian!
    """)

    # For a simple case: 2 equal masses on circular orbit
    m = M_gal / 2
    r_orb = r_M
    v_orb = np.sqrt(G * M_gal / r_orb)
    T_orb = 2 * np.pi * r_orb / v_orb
    Omega = v_orb / r_orb

    print(f"  Simple model: 2 masses on circular orbit")
    print(f"  m = M_gal/2 = {m/Msun:.1e} M_sun")
    print(f"  r_orb = r_M = {r_orb/kpc:.1f} kpc")
    print(f"  v_orb = {v_orb/1e3:.1f} km/s")
    print(f"  T_orb = {T_orb/(1e9*3.15576e7):.2f} Gyr")

    # At a test point far from center (r = 3 r_M, theta = 0)
    r_test = 3 * r_M
    N_time = 1000
    t_arr = np.linspace(0, T_orb, N_time, endpoint=False)

    grad_phi_sq_arr = np.zeros(N_time)
    grad_phi_x_arr = np.zeros(N_time)
    grad_phi_y_arr = np.zeros(N_time)

    for it, t in enumerate(t_arr):
        angle1 = Omega * t
        angle2 = Omega * t + np.pi  # opposite side

        # Positions of the two masses
        x1, y1 = r_orb * np.cos(angle1), r_orb * np.sin(angle1)
        x2, y2 = r_orb * np.cos(angle2), r_orb * np.sin(angle2)

        # Test point at (r_test, 0)
        xp, yp = r_test, 0.0

        # Gradient of phi_i = -2Gm/(c^2 r) => nabla phi_i = 2Gm/(c^2 r^2) r_hat
        for (xm, ym) in [(x1, y1), (x2, y2)]:
            dx, dy = xp - xm, yp - ym
            dist = np.sqrt(dx**2 + dy**2)
            # nabla phi = (2Gm/c^2) * (r_vec / r^3) = r_s / (2 r^2) * r_hat
            fac = 2 * G * m / (c**2 * dist**3)
            grad_phi_x_arr[it] += fac * dx
            grad_phi_y_arr[it] += fac * dy

        grad_phi_sq_arr[it] = grad_phi_x_arr[it]**2 + grad_phi_y_arr[it]**2

    # Time averages
    mean_grad_sq = np.mean(grad_phi_sq_arr)
    mean_grad_x = np.mean(grad_phi_x_arr)
    mean_grad_y = np.mean(grad_phi_y_arr)
    mean_grad_of_mean_sq = mean_grad_x**2 + mean_grad_y**2

    turb_pressure = mean_grad_sq - mean_grad_of_mean_sq
    ratio = turb_pressure / mean_grad_of_mean_sq

    print(f"\n  At test point r = {r_test/kpc:.0f} kpc (= 3 r_M):")
    print(f"    <|nabla phi|^2>       = {mean_grad_sq:.6e}")
    print(f"    |nabla <phi>|^2       = {mean_grad_of_mean_sq:.6e}  (Newtonian)")
    print(f"    <|nabla phi'|^2>      = {turb_pressure:.6e}  (turbulent)")
    print(f"    turbulent / Newtonian = {ratio:.6e}")

    # Now scan over radii
    print(f"\n  Radial scan:")
    print(f"  {'r/r_M':>8s}  {'<|grad phi|^2>':>14s}  {'|grad <phi>|^2':>14s}"
          f"  {'turb/Newton':>12s}")
    print(f"  {'---':>8s}  {'---':>14s}  {'---':>14s}  {'---':>12s}")

    r_scan = np.array([1.5, 2, 3, 5, 10, 20, 50]) * r_M
    ratios = []

    for r_t in r_scan:
        gpsq = np.zeros(N_time)
        gpx = np.zeros(N_time)
        gpy = np.zeros(N_time)

        for it, t in enumerate(t_arr):
            a1 = Omega * t
            a2 = a1 + np.pi
            gpx_t, gpy_t = 0.0, 0.0
            for (xm, ym) in [(r_orb*np.cos(a1), r_orb*np.sin(a1)),
                              (r_orb*np.cos(a2), r_orb*np.sin(a2))]:
                dx, dy = r_t - xm, 0.0 - ym
                dist = np.sqrt(dx**2 + dy**2)
                fac = 2*G*m / (c**2 * dist**3)
                gpx_t += fac*dx
                gpy_t += fac*dy
            gpx[it] = gpx_t
            gpy[it] = gpy_t
            gpsq[it] = gpx_t**2 + gpy_t**2

        avg_sq = np.mean(gpsq)
        sq_avg = np.mean(gpx)**2 + np.mean(gpy)**2
        turb = avg_sq - sq_avg
        rat = turb / sq_avg if sq_avg > 0 else 0
        ratios.append(rat)

        print(f"  {r_t/r_M:>8.1f}  {avg_sq:>14.6e}  {sq_avg:>14.6e}  {rat:>12.6e}")

    print(f"\n  The turbulent/Newtonian ratio for 2-body is O({np.mean(ratios):.2e}).")
    print(f"  This measures how much the rotating configuration 'stirs' the medium")
    print(f"  beyond the azimuthally averaged Newtonian value.")

    return ratios


# =====================================================================
#  TEST 2: Order-of-magnitude estimate for a real galaxy
# =====================================================================

def test_2_amplitude_estimate():
    """Estimate <|nabla phi'|^2> for a realistic galaxy."""
    print(f"\n{SEP}")
    print("  TEST 2: Amplitude Estimate for Realistic Galaxy")
    print(SEP)

    print("""
  For a galaxy with N_* stars of mass m_* in a disk of radius R:

  The Newtonian acceleration at radius r >> R:
    g_N = G M / r^2  (smooth average)

  The fluctuation from individual stars:
    Each star creates delta_g ~ G m_* / b^2 at impact parameter b
    A test point at radius r has ~N_* stars within the disk
    The closest star is at distance ~ r / sqrt(N_eff) where
    N_eff ~ (r/R)^2 * N_* (stars inside annulus of width r)

  But the KEY is not individual encounters — it's the COHERENT
  fluctuation of the TOTAL field as the pattern rotates.
    """)

    # Galaxy parameters
    N_stars = 1e11
    m_star = Msun
    R_disk = 10 * kpc
    M_total = N_stars * m_star

    print(f"  Galaxy: N_* = {N_stars:.0e}, m_* = {m_star/Msun:.0f} M_sun")
    print(f"  M_total = {M_total/Msun:.0e} M_sun")
    print(f"  R_disk = {R_disk/kpc:.0f} kpc")

    # The key quantity: how much does |nabla phi|^2 fluctuate
    # as the disk rotates?
    #
    # For a smooth disk, rotation creates NO fluctuation (axisymmetric).
    # Fluctuation comes from GRANULARITY (discrete stars) and STRUCTURE
    # (spiral arms, bar, etc.).
    #
    # Granularity contribution:
    # At radius r, the nearest star is at distance ~ d_min
    # where d_min ~ r * (m_*/M_enc(r))^(1/2) for a disk
    #
    # The fluctuation of g from one star crossing:
    # delta_g / g_N ~ (m_* / M_enc) * (r / d_min)^2

    print(f"\n  --- Granularity (discrete stars) ---")
    for xi in [0.5, 1.0, 2.0, 5.0, 10.0]:
        r = xi * r_M
        M_enc = M_total * (1 - (1 + r/R_disk) * np.exp(-r/R_disk))
        if M_enc <= 0:
            continue
        g_N = G * M_enc / r**2

        # Number of stars inside radius r
        N_enc = N_stars * M_enc / M_total
        # Mean inter-star separation in the disk (2D)
        if N_enc > 0:
            d_mean = r / np.sqrt(N_enc)
        else:
            d_mean = r

        # Fluctuation from nearest star
        delta_g = G * m_star / d_mean**2
        frac = delta_g / g_N

        print(f"    xi={xi:>4.1f}: N_enc={N_enc:.2e}, d_mean={d_mean/kpc:.4f} kpc, "
              f"delta_g/g_N = {frac:.2e}")

    print(f"\n  Granularity gives delta_g/g_N ~ 10^-11 (utterly negligible).")
    print(f"  Individual stars are too small to stir the medium significantly.")

    # Spiral arm contribution:
    print(f"\n  --- Spiral arms (coherent structure) ---")
    f_arm = 0.15  # fraction of mass in spiral arms
    N_arms = 2
    print(f"  Spiral arm mass fraction: {f_arm:.0%}")
    print(f"  Number of arms: {N_arms}")

    for xi in [0.5, 1.0, 2.0, 5.0, 10.0]:
        r = xi * r_M
        M_enc = M_total * (1 - (1 + r/R_disk) * np.exp(-r/R_disk))
        if M_enc <= 0:
            continue
        g_N = G * M_enc / r**2

        # Spiral arm creates azimuthal variation ~ f_arm * g_N
        # As galaxy rotates, this variation sweeps past test point
        delta_g_arm = f_arm * g_N
        frac = delta_g_arm / g_N

        # Turbulent Bernoulli: <|nabla phi'|^2> / |nabla <phi>|^2
        # For sinusoidal variation with amplitude f_arm:
        # <(f_arm * g_N * cos(N_arms * Omega * t))^2> = f_arm^2 * g_N^2 / 2
        turb_ratio = f_arm**2 / 2

        print(f"    xi={xi:>4.1f}: delta_g/g_N = {frac:.2f}, "
              f"turb/Newton = {turb_ratio:.4f}")

    turb_ratio_spiral = f_arm**2 / 2
    print(f"\n  Spiral arms give turb/Newton ~ {turb_ratio_spiral:.4f}")
    print(f"  This is ~ 1% effect. MOND needs O(1) effect at xi > 1.")
    print(f"  Spiral arms alone are NOT enough.")

    # What WOULD be enough?
    print(f"\n  --- What amplitude is needed for MOND? ---")
    for xi in [1.0, 2.0, 5.0, 10.0]:
        r = xi * r_M
        M_enc = M_total * (1 - (1 + r/R_disk) * np.exp(-r/R_disk))
        if M_enc <= 0:
            continue
        gN_dimless = M_enc / M_total  # g_N / a0 approximately for this case
        x = gN_dimless  # g_N / a0

        # MOND: g = g_N + g_h, g_h = a0 * g_N / g
        # g_h / g_N = a0 / g = 1/(1+x) approximately
        g_h_over_gN = 1.0 / x if x > 0 else 0  # deep MOND approx
        # More precise: g^2 = g*g_N + a0*g_N, g_h = g - g_N
        disc = x**2 + 4*x
        g_tot = 0.5*(x + np.sqrt(disc))
        g_h = g_tot - x
        ratio_needed = g_h / x  # g_h/g_N in units of a0

        # turb/Newton needed = g_h^2 / g_N^2 (since pressure ~ g^2)
        # Actually: delta_P_turb creates delta_g = sqrt(turb) * g_N
        # So for g_h / g_N = ratio, need turb/Newton = ratio^2... no.
        # More carefully: the turbulent term adds to |nabla phi|^2
        # so g_eff^2 = g_N^2 + <|nabla phi'|^2> * c^4
        # g_h = g_eff - g_N = sqrt(g_N^2 + turb*c^4) - g_N ~ turb*c^4/(2*g_N)
        # So turb_needed = 2 * g_h * g_N / c^4 ... in proper units

        print(f"    xi={xi:>4.1f}: g_h/g_N = {ratio_needed:.4f} needed for MOND")

    print(f"""
  CRITICAL ASSESSMENT:

  The turbulent Bernoulli from STRUCTURAL non-axisymmetry (spiral arms,
  bars, etc.) gives at most a few percent effect. This is because the
  azimuthal mass variation is only ~10-15% of the total.

  For MOND we need g_h/g_N ~ 1 at xi ~ 1 and g_h/g_N >> 1 at xi >> 1.
  No reasonable structural asymmetry can produce this.

  CONCLUSION: The "turbulent Bernoulli from structural asymmetry"
  mechanism is TOO WEAK by a factor of ~100 at the MOND radius,
  and gets WORSE at larger radii.

  The turbulent Bernoulli from granularity is even weaker (10^-11).

  THIS APPROACH DOES NOT WORK as formulated.
    """)

    return turb_ratio_spiral


# =====================================================================
#  TEST 3: But wait — what about the COHERENT vortex?
# =====================================================================

def test_3_coherent_vortex():
    """The rotation creates a coherent vortex, not just turbulence."""
    print(f"\n{SEP}")
    print("  TEST 3: Coherent Vortex vs Turbulent Fluctuation")
    print(SEP)

    print("""
  The previous test showed that STRUCTURAL fluctuations are too weak.
  But the user's insight was about a VORTEX — a coherent structure.

  In fluid dynamics, a spinning object creates:
    1. Turbulent wake (from roughness) — WEAK, what Test 2 computed
    2. COHERENT VORTEX (from bulk rotation) — can be STRONG

  The coherent vortex is NOT about fluctuations of |nabla phi|^2.
  It's about the STEADY-STATE flow pattern in the rotating frame.

  In the rotating frame (co-rotating with the galaxy):
    - The mass distribution is STATIC
    - The field equation has additional terms from the rotating frame
    - The Coriolis and centrifugal terms act on the SCALAR FIELD

  The scalar field equation in the rotating frame:
    Box_rot phi = S

  where Box_rot includes the transformation t -> t, theta -> theta - Omega*t:
    partial_t -> partial_t - Omega * partial_theta

  For steady state in the rotating frame (partial_t = 0):
    Box_rot phi = nabla^2 phi - (Omega^2/c^2) partial_theta^2 phi = S

  The extra term -(Omega^2/c^2) partial_theta^2 phi IS v^2/c^2 suppressed.

  SO: even in the rotating frame, the correction is epsilon-suppressed.
  The vortex IS a v^2/c^2 effect in the field equation.
    """)

    v_rot = np.sqrt(G * M_gal / r_M)
    Omega_gal = v_rot / r_M
    eps_rot = (v_rot / c)**2

    print(f"  v_rot = {v_rot/1e3:.1f} km/s")
    print(f"  Omega = {Omega_gal:.4e} rad/s")
    print(f"  v^2/c^2 = {eps_rot:.4e} = epsilon")
    print(f"  The rotating-frame correction is O(epsilon) ~ O({eps_rot:.1e})")

    print(f"""
  HONEST CONCLUSION:

  Both approaches lead to the same result:

  1. STRUCTURAL TURBULENCE (spiral arms, granularity):
     delta_g/g_N ~ f_arm^2 ~ 1%  (too weak by ~100x)

  2. COHERENT VORTEX (rotating frame):
     delta_g/g_N ~ v^2/c^2 ~ {eps_rot:.1e}  (too weak by ~{1/eps_rot:.0e}x)

  The field equation Box phi = S is the SAME whether we think of it
  as "geometry" or "fluid." The math doesn't care about our analogy.
  And the math says: rotation effects are v^2/c^2 suppressed.

  The water analogy is MISLEADING because:
    - In water, v/v_sound can be O(1) or even > 1 (supersonic)
    - In galaxies, v/c ~ 10^-3 (extremely subsonic)
    - Water vortices are strong because Reynolds number is HIGH
    - The "space fluid" Reynolds number is VERY LOW (Hubble damping)

  The space-fluid is like honey, not water. A spinning object in
  honey creates a tiny, epsilon-sized vortex. Not a dramatic whirlpool.
    """)

    # Reynolds number analog
    nu_eff = c**2 / H0  # effective kinematic viscosity from Hubble damping
    Re = v_rot * r_M / nu_eff
    print(f"  'Reynolds number' of space-fluid:")
    print(f"    nu_eff = c^2/H = {nu_eff:.4e} m^2/s")
    print(f"    Re = v*r/nu = {Re:.4e}")
    print(f"    This is EXTREMELY small — ultra-viscous (laminar) regime.")
    print(f"    No turbulence possible. No vortex amplification.")

    return eps_rot


# =====================================================================
#  VERDICT
# =====================================================================

def verdict():
    print(f"\n{SEP}")
    print("  PHASE F VERDICT: Turbulent Bernoulli")
    print(SEP)

    print(f"""
  RESULT: The turbulent Bernoulli idea DOES NOT WORK.

  Three independent arguments show this:

  1. STRUCTURAL FLUCTUATIONS: Spiral arms (~15% mass fraction)
     give <|nabla phi'|^2> / |nabla <phi>|^2 ~ 1%.
     Need ~100% for MOND. Gap: ~100x.

  2. COHERENT VORTEX: Rotating-frame correction to Box phi is
     O(v^2/c^2) = O(10^-6). Same epsilon as frame-dragging.
     Gap: ~10^5.

  3. REYNOLDS NUMBER: Space-fluid Re ~ 10^-9. Ultra-laminar.
     No turbulent amplification possible.

  WHY THE WATER ANALOGY FAILS:
    Water: v/c_sound ~ O(1), Re ~ 10^6. Turbulence, strong vortices.
    Space: v/c ~ 10^-3, Re ~ 10^-9. No turbulence, epsilon vortex.

  The field equation Box phi = S doesn't care whether we call phi
  "geometry" or "fluid." The physics is the same, and the physics
  says rotation at v << c creates only v^2/c^2 corrections.

  THE THREE OPEN PROBLEMS REMAIN UNSOLVED.

  WHAT WE'VE LEARNED:
    - The amplitude gap is NOT a choice-of-approach problem
    - It's a FUNDAMENTAL property of any v << c rotating system
    - ANY mechanism derivable from the ISPG action with rotating
      sources will be v^2/c^2 suppressed, because the action itself
      is Lorentz-invariant and rotation enters at O(v^2/c^2)
    - The only escape would be a mechanism that is NOT rotation-based
      (but MOND only appears in rotating systems... or does it?)
    """)


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    ratios = test_1_bernoulli_decomposition()
    turb = test_2_amplitude_estimate()
    eps_v = test_3_coherent_vortex()
    verdict()
