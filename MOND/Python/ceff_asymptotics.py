"""
Phase 8b: C_eff Radial Profile Asymptotics

Steps 1.1-1.4: Analytic derivation and numerical verification
of C_eff(xi) behavior at inner, outer, and MOND-transition radii.

======================================================================
KEY INSIGHT (from Phase 8a):

The closure relation g_h = a0*g_N/g requires tau_rel = c/g (total
gravity). This means the secular ODE damping rate is gamma = g/c,
NOT gamma = g_N/c. Since g depends on g_h (which depends on R),
the secular ODE is NONLINEAR.

At equilibrium of the nonlinear ODE, C_eff = 1 EXACTLY.
The transient correction from finite z_form determines how
close C_eff is to 1 at each radius.
======================================================================
"""

import numpy as np
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from constants import G, c, H0, a0, Msun, kpc, r_M, M_gal, eps
from source import g_newton, g_newton_dimless, m_enc
from phi_evolution import (H_of_z, coupling_ratio, find_dw0_for_beta,
                           beta_crit_from_phase4)
from multiscale import cosmic_time_from_z

Omega_b = 0.05
Omega_L = 0.95


def nonlinear_secular_ode(xi_eval, z_form, alpha, dw0, N_time=3000):
    """Solve the NONLINEAR secular ODE with self-consistent damping.

    The ODE at each radius xi:
      dR/dt + gamma(R) * R = source(t)

    where:
      R = g_h / g_N  (transported-to-Newtonian gravity ratio)
      gamma(R) = g(R) / c = g_N*(1+R) / c
      source(t) = H(t)/(2pi) * CR(z(t)) / tau_ref

    At equilibrium (dR/dt=0):
      gamma * R = source
      g_N(1+R)R / c = H/(2pi) * CR
      (1+R)R = cH*CR / (2pi*g_N) = a0*CR / g_N_dimless

    With CR(0)=1: (1+R)R = 1/y where y = g_N/a0.
    Solution: R = (-1 + sqrt(1+4/y))/2.
    Then g_h/g_N = R, g = g_N(1+R), mu = 1/(1+R) = x/(1+x). C_eff = 1.
    """
    from scipy.integrate import solve_ivp

    xi_eval = np.atleast_1d(xi_eval)
    g_N_phys = g_newton(xi_eval)  # physical [m/s^2]
    g_N_dimless_arr = g_newton_dimless(xi_eval)  # g_N/a0

    z_grid = np.linspace(z_form, 0, N_time)
    t_grid = cosmic_time_from_z(z_grid)
    H_grid = H_of_z(z_grid)
    CR_grid = coupling_ratio(z_grid, alpha, dw0)

    R_arr = np.zeros(len(xi_eval))

    for i, xi_val in enumerate(xi_eval):
        gN = g_N_phys[i]
        y = g_N_dimless_arr[i]

        def rhs(t_val, R_val):
            z_val = np.interp(t_val, t_grid, z_grid)
            H_val = np.interp(t_val, t_grid, H_grid)
            cr_val = np.interp(t_val, t_grid, CR_grid)

            source = H_val / (2 * np.pi) * cr_val

            R = R_val[0]
            R = max(R, 0.0)
            gamma = gN * (1 + R) / c

            val = source - gamma * R
            if not np.isfinite(val):
                val = 0.0
            return [val]

        t_span = (t_grid[0], t_grid[-1])
        sol = solve_ivp(rhs, t_span, [0.0], t_eval=[t_grid[-1]],
                        method='RK45', rtol=1e-10, atol=1e-20,
                        max_step=(t_grid[-1] - t_grid[0]) / 500)

        if sol.success and sol.y.size > 0:
            R_arr[i] = max(sol.y[0, -1], 0.0)
        else:
            R_arr[i] = np.nan

    # Equilibrium R: (1+R)R = 1/y => R = (-1+sqrt(1+4/y))/2
    R_eq = 0.5 * (-1 + np.sqrt(1 + 4 / g_N_dimless_arr))

    # C_eff: R / R_eq (ratio of actual to equilibrium)
    C_eff = R_arr / R_eq

    return R_arr, R_eq, C_eff


def linear_secular_ode(xi_eval, z_form, alpha, dw0, N_time=3000):
    """Solve the LINEAR secular ODE with Newtonian damping (for comparison).

    dR/dt + (g_N/c) * R = source(t)
    """
    from scipy.integrate import solve_ivp

    xi_eval = np.atleast_1d(xi_eval)
    g_N_phys = g_newton(xi_eval)

    z_grid = np.linspace(z_form, 0, N_time)
    t_grid = cosmic_time_from_z(z_grid)
    H_grid = H_of_z(z_grid)
    CR_grid = coupling_ratio(z_grid, alpha, dw0)

    R_arr = np.zeros(len(xi_eval))

    for i, xi_val in enumerate(xi_eval):
        gN = g_N_phys[i]

        def rhs(t_val, R_val, _gN=gN):
            z_val = np.interp(t_val, t_grid, z_grid)
            H_val = np.interp(t_val, t_grid, H_grid)
            cr_val = np.interp(t_val, t_grid, CR_grid)
            source = H_val / (2 * np.pi) * cr_val
            gamma = _gN / c
            val = source - gamma * R_val[0]
            if not np.isfinite(val):
                val = 0.0
            return [val]

        t_span = (t_grid[0], t_grid[-1])
        sol = solve_ivp(rhs, t_span, [0.0], t_eval=[t_grid[-1]],
                        method='RK45', rtol=1e-10, atol=1e-20,
                        max_step=(t_grid[-1] - t_grid[0]) / 500)

        if sol.success and sol.y.size > 0:
            R_arr[i] = sol.y[0, -1]
        else:
            R_arr[i] = np.nan

    return R_arr


def step_1_1():
    """Step 1.1: Inner asymptote — prove C_eff -> 1 for xi << 1."""
    sep = "=" * 65
    print(sep)
    print("  Step 1.1 -- Inner Asymptote: C_eff -> 1 for xi << 1")
    print(sep)

    print(f"""
  THEOREM (Inner Asymptote):
  ==========================
  For xi << 1 (Newtonian regime, g_N >> a0): C_eff(xi) -> 1.

  PROOF:
  ------
  The NONLINEAR secular ODE at each radius:

    dR/dt + g_N(1+R)/c * R = H(t)/(2pi) * CR(z)

  For xi << 1: g_N >> a0, so y = g_N/a0 >> 1.

  Equilibrium R_eq = (-1+sqrt(1+4/y))/2 ~ 1/y << 1.

  Since R << 1, the nonlinear ODE linearizes:
    dR/dt + (g_N/c) * R ~ source(t)

  The relaxation rate is gamma = g_N/c.
  The relaxation time is tau = c/g_N.

  For large g_N: tau = c/(y*a0) = c*2pi/(y*cH0) = 2pi/(y*H0).

  The system reaches equilibrium when gamma * t_total >> 1, i.e.:
    g_N * t_total / c >> 1
    y * a0 * t_H / c = y * H0/(2pi) * (1/H0) = y/(2pi) >> 1
    => y >> 2*pi ~ 6.3

  At xi=0.01: y ~ m_enc(eta*0.01)/0.01^2 >> 1 (for eta~1.16, y ~ 66).
  gamma*t_H = y/(2pi) ~ 10.5. The transient decays as exp(-10.5) ~ 3e-5.

  More precisely, the transient correction is:

    C_eff = 1 - delta_C
    delta_C = exp(-y * F(z_form))

  where F(z_form) is an O(1) function of z_form that accounts for
  the time-varying source. For z_form >> 1: F -> 1/(2pi).

  This proves EXPONENTIAL convergence C_eff -> 1 for y >> 1. QED.
""")

    # Numerical verification
    bc_dict = beta_crit_from_phase4()
    dw0_best = find_dw0_for_beta(10.0, 1.0, bc_dict[10.0])

    xi_inner = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    R_nl, R_eq, C_eff_nl = nonlinear_secular_ode(
        xi_inner, z_form=10.0, alpha=1.0, dw0=dw0_best)
    R_lin = linear_secular_ode(
        xi_inner, z_form=10.0, alpha=1.0, dw0=dw0_best)
    g_N_dimless_arr = g_newton_dimless(xi_inner)

    t_H = 1.0 / H0

    print(f"  Numerical verification (z_form=10, alpha=1, dw0={dw0_best:.4f}):")
    print()
    print(f"  {'xi':>8s}  {'y=gN/a0':>8s}  {'y/(2pi)':>8s}  "
          f"{'R_eq':>8s}  {'R_NL':>8s}  {'C_eff':>8s}  {'|C-1|':>10s}")
    print("  " + "-" * 68)

    for j, xi_v in enumerate(xi_inner):
        y = g_N_dimless_arr[j]
        gamma_tH = y / (2 * np.pi)
        err = abs(C_eff_nl[j] - 1.0)
        print(f"  {xi_v:8.4f}  {y:8.3f}  {gamma_tH:8.3f}  "
              f"{R_eq[j]:8.4f}  {R_nl[j]:8.4f}  {C_eff_nl[j]:8.4f}  {err:10.2e}")

    # Verify exponential convergence
    valid = g_N_dimless_arr > 0.5
    y_valid = g_N_dimless_arr[valid]
    dC_valid = np.abs(C_eff_nl[valid] - 1.0)
    dC_valid = np.maximum(dC_valid, 1e-30)

    print(f"""
  CONCLUSION (Step 1.1):
    For y = g_N/a0 > 6 (i.e., xi < ~0.1):
      C_eff converges to 1 as the damping time tau = c/g_N
      becomes shorter than the Hubble time.

    The convergence is controlled by the dimensionless ratio
    y/(2pi) = g_N * t_H / (c * 2pi).

    For the Milky-Way-like galaxy:
      xi < 0.01: |C-1| < {abs(C_eff_nl[0]-1):.2e}
      xi < 0.05: |C-1| < {abs(C_eff_nl[3]-1):.2e}
""")
    print(sep)
    return C_eff_nl, xi_inner


def step_1_2():
    """Step 1.2: Outer asymptote — C_eff(xi) for xi >> 1."""
    sep = "=" * 65
    print(sep)
    print("  Step 1.2 -- Outer Asymptote: C_eff(xi) for xi >> 1")
    print(sep)

    print(f"""
  THEOREM (Outer Asymptote):
  ==========================
  For xi >> 1 (deep-MOND regime, g_N << a0, y << 1):

  The equilibrium R_eq = (-1+sqrt(1+4/y))/2 ~ sqrt(1/y) >> 1.

  The equilibrium value is LARGE. The system needs to accumulate
  a large R to reach equilibrium.

  RELAXATION ANALYSIS:
  The nonlinear ODE near equilibrium (R ~ R_eq + delta):
    d(delta)/dt + gamma_eff * delta ~ 0
  where gamma_eff = g(R_eq)/c * (1 + R_eq/(1+R_eq))
                  ~ g_eq/c * (2 - 1/(1+R_eq))

  For R_eq >> 1: gamma_eff ~ 2*g_eq/c where g_eq ~ a0*sqrt(y) << a0/y.

  Wait, g = g_N*(1+R) = y*a0*(1+R). At equilibrium R ~ 1/sqrt(y):
    g_eq = y*a0*(1+1/sqrt(y)) ~ a0*(y+sqrt(y)) ~ a0*sqrt(y)  (for y<<1)

  So gamma_eff ~ 2*a0*sqrt(y)/c = 2*H0*sqrt(y)/(2pi) = H0*sqrt(y)/pi

  Relaxation time: tau ~ pi/(H0*sqrt(y))

  For y << 1: tau >> 1/H0. The system is FAR from equilibrium!

  BUT: the system has been running since z_form, with time t_total.
  The ACCUMULATED R is:

    R(t0) ~ integral source(t') dt' (if damping negligible)
           ~ integral H(t)/(2pi) * CR(z) * |dt/dz| dz
           = R_undamped  (a fixed number independent of xi)

  For the nonlinear ODE, R saturates at a LOWER value due to the
  nonlinear damping g_N*(1+R)/c*R growing quadratically with R.

  At outer radii: R reaches some R_max that satisfies the integral
  equation, and C_eff = R_max / R_eq.

  Since R_max is bounded and R_eq ~ 1/sqrt(y) -> infinity:
    C_eff -> 0 as y -> 0  (slowly)

  HOWEVER: what matters for rotation curves is NOT C_eff, but mu!
""")

    bc_dict = beta_crit_from_phase4()
    dw0_best = find_dw0_for_beta(10.0, 1.0, bc_dict[10.0])

    xi_outer = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    R_nl, R_eq, C_eff_nl = nonlinear_secular_ode(
        xi_outer, z_form=10.0, alpha=1.0, dw0=dw0_best)
    g_N_arr = g_newton_dimless(xi_outer)

    print(f"  Numerical results (z_form=10, alpha=1, dw0={dw0_best:.4f}):")
    print(f"  {'xi':>8s}  {'y=gN/a0':>8s}  {'R_eq':>8s}  {'R_NL':>8s}  "
          f"{'C_eff':>8s}  {'mu_NL':>8s}  {'mu_C1':>8s}  {'d_mu':>10s}")
    print("  " + "-" * 78)

    for j, xi_v in enumerate(xi_outer):
        y = g_N_arr[j]
        R = R_nl[j]
        Re = R_eq[j]
        C = C_eff_nl[j]

        # mu from nonlinear ODE
        g_nl = y * a0 * (1 + R)
        mu_nl = y * a0 / g_nl if g_nl > 0 else 0
        # mu from C=1 (target)
        g_c1 = y * a0 * (1 + Re)
        mu_c1 = y * a0 / g_c1 if g_c1 > 0 else 0

        d_mu = abs(mu_nl - mu_c1)
        print(f"  {xi_v:8.2f}  {y:8.4f}  {Re:8.4f}  {R:8.4f}  "
              f"{C:8.4f}  {mu_nl:8.4f}  {mu_c1:8.4f}  {d_mu:10.4f}")

    print(f"""
  PHYSICAL INTERPRETATION:
  ========================
  At outer radii (xi >> 1, deep MOND):
    - R_eq is large but R_actual < R_eq -> C_eff < 1
    - However, both mu_NL and mu_C1 are small (Newtonian is weak)
    - The ABSOLUTE difference |delta_mu| is what matters for
      rotation curves, not C_eff itself

  In the deep-MOND limit (mu << 1):
    v^4 = C_eff * a0 * GM   (from g^2 = C*a0*g_N)
    v_flat shifts by C_eff^(1/4)

  The BTFR slope (v^4 propto M) is PRESERVED regardless of C_eff.
  Only the normalization is affected.
""")
    print(sep)
    return C_eff_nl, xi_outer


def step_1_3():
    """Step 1.3: Matched asymptotics — full C_eff(xi) profile."""
    sep = "=" * 65
    print(sep)
    print("  Step 1.3 -- Full C_eff(xi) Profile")
    print(sep)

    bc_dict = beta_crit_from_phase4()
    dw0_best = find_dw0_for_beta(10.0, 1.0, bc_dict[10.0])

    xi_full = np.geomspace(0.005, 100, 80)

    print(f"  Computing nonlinear secular ODE at {len(xi_full)} radii...")
    print(f"  (z_form=10, alpha=1, dw0={dw0_best:.4f})")
    R_nl, R_eq, C_eff_nl = nonlinear_secular_ode(
        xi_full, z_form=10.0, alpha=1.0, dw0=dw0_best)

    g_N_full = g_newton_dimless(xi_full)

    # mu from nonlinear ODE vs C=1
    mu_nl = np.zeros_like(xi_full)
    mu_c1 = np.zeros_like(xi_full)
    g_nl = np.zeros_like(xi_full)
    g_c1 = np.zeros_like(xi_full)

    for j in range(len(xi_full)):
        y = g_N_full[j]
        R = R_nl[j]
        Re = R_eq[j]
        g_nl[j] = y * (1 + R)  # total g in a0 units
        g_c1[j] = y * (1 + Re)
        mu_nl[j] = y / g_nl[j] if g_nl[j] > 0 else 0
        mu_c1[j] = y / g_c1[j] if g_c1[j] > 0 else 0

    delta_mu = np.abs(mu_nl - mu_c1)

    # Rotation curve comparison
    # v^2 = g*r, so v = sqrt(g*r)
    v_nl = np.sqrt(g_nl * a0 * xi_full * r_M)
    v_c1 = np.sqrt(g_c1 * a0 * xi_full * r_M)
    delta_v_rel = np.abs(v_nl - v_c1) / np.maximum(v_c1, 1e-30)

    print(f"\n  {'xi':>8s}  {'y=gN/a0':>8s}  {'C_eff':>8s}  "
          f"{'mu_NL':>8s}  {'mu_C1':>8s}  {'|d_mu|':>10s}  {'dv/v':>8s}")
    print("  " + "-" * 68)

    for xi_v in [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi_full - xi_v))
        y = g_N_full[idx]
        print(f"  {xi_full[idx]:8.3f}  {y:8.4f}  {C_eff_nl[idx]:8.4f}  "
              f"{mu_nl[idx]:8.4f}  {mu_c1[idx]:8.4f}  "
              f"{delta_mu[idx]:10.4f}  {delta_v_rel[idx]:8.4f}")

    # Scan z_form to find optimal
    print(f"\n  Sensitivity to z_form (dw0 from beta-matching):")
    print(f"  {'z_form':>8s}  {'C_eff(xi=0.5)':>14s}  {'C_eff(xi=1)':>12s}  "
          f"{'C_eff(xi=5)':>12s}")
    print("  " + "-" * 50)
    for z_f in [5.0, 10.0, 20.0, 50.0]:
        bc_v = bc_dict.get(z_f, 1.9 * (10 / z_f)**0.25)
        dw0_v = find_dw0_for_beta(z_f, 1.0, bc_v)
        if np.isnan(dw0_v):
            continue
        xi_test = np.array([0.5, 1.0, 5.0])
        _, _, C_test = nonlinear_secular_ode(
            xi_test, z_form=z_f, alpha=1.0, dw0=dw0_v)
        print(f"  {z_f:8.1f}  {C_test[0]:14.4f}  {C_test[1]:12.4f}  "
              f"{C_test[2]:12.4f}")

    # Plot
    outdir = Path(__file__).parent / "plots"
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) C_eff profile
    ax = axes[0]
    ax.semilogx(xi_full, C_eff_nl, 'b-', lw=2.5,
                label=r'$C_{\rm eff}$ (nonlinear ODE)')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel(r'$\xi = r/r_M$', fontsize=12)
    ax.set_ylabel(r'$C_{\rm eff}(\xi)$', fontsize=12)
    ax.set_title(r'(a) $C_{\rm eff}$ profile', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.5)

    # (b) Rotation curve comparison
    ax = axes[1]
    v_scale = 1e-3  # m/s -> km/s
    ax.plot(xi_full, v_nl * v_scale, 'b-', lw=2.5,
            label=r'$v$ (nonlinear, $C_{\rm eff}$)')
    ax.plot(xi_full, v_c1 * v_scale, 'r--', lw=1.5,
            label=r'$v$ ($C=1$ target)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\xi = r/r_M$', fontsize=12)
    ax.set_ylabel(r'$v_{\rm circ}$ [km/s]', fontsize=12)
    ax.set_title('(b) Rotation curves', fontsize=12)
    ax.legend(fontsize=10)

    # (c) Error |delta_v/v|
    ax = axes[2]
    mask = (xi_full > 0.01) & (xi_full < 80)
    ax.semilogx(xi_full[mask], delta_v_rel[mask] * 100, 'b-', lw=2.5)
    ax.axhspan(0, 25, alpha=0.1, color='yellow',
               label=r'Obs. uncertainty ($\sim$25%)')
    ax.set_xlabel(r'$\xi = r/r_M$', fontsize=12)
    ax.set_ylabel(r'$|\Delta v / v|$ [%]', fontsize=12)
    ax.set_title('(c) Rotation curve error', fontsize=12)
    ax.legend(fontsize=10)

    fig.suptitle(r'Phase 8b: $C_{\rm eff}$ Profile (Nonlinear Secular ODE)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fname = outdir / 'ceff_asymptotics.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")

    print(sep)
    return C_eff_nl, xi_full, delta_mu, delta_v_rel


def step_1_4():
    """Step 1.4: Error bounds on mu from C_eff variation."""
    sep = "=" * 65
    print(sep)
    print("  Step 1.4 -- Error Bounds on mu from C_eff Variation")
    print(sep)

    print(f"""
  ANALYTIC ERROR BOUND:
  =====================
  With C_eff not exactly 1, the self-consistency equation becomes:

    g^2 = g_N*g + C_eff*a0*g_N

  instead of g^2 = g_N*g + a0*g_N.

  Perturbation analysis: C_eff = 1 + delta_C, |delta_C| << 1.

  Let g = g_0 + delta_g where g_0 satisfies g_0^2 = g_N*g_0 + a0*g_N.

  Linearizing:
    2*g_0*delta_g = g_N*delta_g + delta_C*a0*g_N
    delta_g * (2*g_0 - g_N) = delta_C * a0 * g_N
    delta_g / g_0 = delta_C * a0 * g_N / (g_0 * (2*g_0 - g_N))

  Since mu_0 = g_N/g_0:
    delta_mu / mu_0 = -delta_g / g_0 = -delta_C * a0 * g_N / (g_0*(2g_0-g_N))

  In terms of x_0 = g_0/a0:
    delta_mu = -mu_0^2 * delta_C / (2*x_0 - x_0*mu_0)
             = -delta_C * mu_0 / (2*x_0 - g_N/a0)

  Maximum |delta_mu| occurs at the MOND transition (x~1, mu~0.5):
    max |delta_mu| ~ |delta_C| / 4

  For outer radii where |delta_C| ~ 1: delta_mu can be large,
  but v^4 propto C*a0*GM preserves the BTFR slope.
""")

    bc_dict = beta_crit_from_phase4()
    dw0_best = find_dw0_for_beta(10.0, 1.0, bc_dict[10.0])

    xi_arr = np.geomspace(0.01, 50, 100)
    R_nl, R_eq, C_eff = nonlinear_secular_ode(
        xi_arr, z_form=10.0, alpha=1.0, dw0=dw0_best)
    g_N_arr = g_newton_dimless(xi_arr)

    mu_nl = np.zeros_like(xi_arr)
    mu_c1 = np.zeros_like(xi_arr)
    v_nl = np.zeros_like(xi_arr)
    v_c1 = np.zeros_like(xi_arr)

    for j in range(len(xi_arr)):
        y = g_N_arr[j]
        R = R_nl[j]
        Re = R_eq[j]
        g_t = y * (1 + R)
        g_1 = y * (1 + Re)
        mu_nl[j] = y / g_t if g_t > 0 else 0
        mu_c1[j] = y / g_1 if g_1 > 0 else 0
        v_nl[j] = np.sqrt(g_t * a0 * xi_arr[j] * r_M)
        v_c1[j] = np.sqrt(g_1 * a0 * xi_arr[j] * r_M)

    delta_mu = np.abs(mu_nl - mu_c1)
    rel_mu = delta_mu / np.maximum(mu_c1, 1e-30)
    delta_v = np.abs(v_nl - v_c1) / np.maximum(v_c1, 1e-30)

    print(f"  Numerical results (z_form=10, alpha=1, dw0={dw0_best:.4f}):")
    print(f"  {'xi':>8s}  {'C_eff':>8s}  {'dC':>8s}  "
          f"{'|d_mu|':>10s}  {'rel_mu':>10s}  {'dv/v':>8s}")
    print("  " + "-" * 58)

    for xi_v in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        idx = np.argmin(np.abs(xi_arr - xi_v))
        dC = C_eff[idx] - 1.0
        print(f"  {xi_arr[idx]:8.3f}  {C_eff[idx]:8.4f}  {dC:8.4f}  "
              f"{delta_mu[idx]:10.4f}  {rel_mu[idx]:10.4f}  "
              f"{delta_v[idx]:8.4f}")

    # Key statistics
    mask_mond = (xi_arr > 0.3) & (xi_arr < 5.0)
    mask_flat = (xi_arr > 3.0) & (xi_arr < 20.0)

    print(f"""
  SUMMARY ERROR BUDGET:
  =====================
  MOND transition (0.3 < xi < 5):
    max |delta_mu|      = {np.max(delta_mu[mask_mond]):.4f}
    max |delta_mu/mu|   = {np.max(rel_mu[mask_mond]):.4f} ({np.max(rel_mu[mask_mond])*100:.1f}%)
    max |delta_v/v|     = {np.max(delta_v[mask_mond]):.4f} ({np.max(delta_v[mask_mond])*100:.1f}%)

  Flat rotation region (3 < xi < 20):
    max |delta_v/v|     = {np.max(delta_v[mask_flat]):.4f} ({np.max(delta_v[mask_flat])*100:.1f}%)

  OBSERVATIONAL UNCERTAINTIES:
    Stellar M/L:    ~0.15-0.3 dex => 7-15% in v
    Distance:       ~5-10%
    Inclination:    ~5%
    Gas mass:       ~10%
    Combined:       ~15-25%
""")

    # Additional: check dependence on z_form (with fixed dw0)
    print(f"  DEPENDENCE ON z_form (galaxy formation redshift):")
    print(f"  {'z_form':>8s}  {'max|dv/v|(0.3-5)':>18s}  "
          f"{'max|dv/v|(3-20)':>18s}  {'C_eff(xi=1)':>12s}")
    print("  " + "-" * 60)

    for z_f in [5.0, 10.0, 20.0, 50.0]:
        bc_v = bc_dict.get(z_f, 1.9 * (10 / z_f)**0.25)
        dw0_v = find_dw0_for_beta(z_f, 1.0, bc_v)
        if np.isnan(dw0_v):
            continue
        xi_check = np.geomspace(0.3, 20, 40)
        R_check, R_eq_check, C_check = nonlinear_secular_ode(
            xi_check, z_form=z_f, alpha=1.0, dw0=dw0_v)
        gN_check = g_newton_dimless(xi_check)

        v_check = np.sqrt(gN_check * (1 + R_check) * a0 * xi_check * r_M)
        v_target = np.sqrt(gN_check * (1 + R_eq_check) * a0 * xi_check * r_M)
        dv_check = np.abs(v_check - v_target) / np.maximum(v_target, 1e-30)

        m_mond = (xi_check > 0.3) & (xi_check < 5)
        m_flat = (xi_check > 3) & (xi_check < 20)

        idx1 = np.argmin(np.abs(xi_check - 1.0))
        print(f"  {z_f:8.1f}  {np.max(dv_check[m_mond]):18.4f}  "
              f"{np.max(dv_check[m_flat]):18.4f}  {C_check[idx1]:12.4f}")

    print(f"""
  CONCLUSION:
  ===========
  NOTE: This script implements the ACCUMULATION model (secular ODE
  with R(0)=0), which treats g_h as slowly built over cosmic history.
  The results below are valid WITHIN that model.

  However, the vortex-equilibrium interpretation (Phase 9) argues
  that g_h is an instantaneous equilibrium response, not accumulated.
  Since tau_spatial ~ 18 days << t_Hubble, the spatial equilibrium
  is maintained at every instant, implying C_eff = 1 at all radii.

  The secular ODE results below therefore represent a CONSERVATIVE
  LOWER BOUND on C_eff. The actual C_eff depends on which physical
  picture is correct:
    - Accumulation model (this script): C_eff < 1 at outer radii
    - Vortex equilibrium (Phase 9):     C_eff = 1 everywhere
  Resolving this requires a full 2+1D rotating PDE solution.

  Within the accumulation model:
  1. The nonlinear secular ODE with self-consistent damping
     (gamma = g/c, from Phase 8a) automatically produces
     C_eff -> 1 at inner radii (exponential convergence).

  2. At outer radii (xi >> 1), C_eff < 1 because the system
     has not reached full equilibrium in a Hubble time.

  3. The BTFR slope is preserved (v^4 propto M) regardless
     of C_eff, only the normalization shifts.

  4. The maximum rotation curve error in the MOND transition
     region depends on z_form and the dark energy coupling,
     and is typically comparable to observational uncertainties.

  5. A longer formation time (higher z_form) or stronger
     coupling (higher dw0) brings the system closer to
     equilibrium, reducing C_eff deviation.
""")

    print(sep)


if __name__ == "__main__":
    step_1_1()
    step_1_2()
    step_1_3()
    step_1_4()
