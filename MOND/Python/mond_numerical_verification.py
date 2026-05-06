"""
MOND predictions and self-consistency checks for ISPG theory (Appendix 12).

Computes:
  - Self-consistent MOND rotation curve for an exponential disk galaxy
  - Interpolating function mu(x) = x/(1+x) from the two-channel equation
  - BTFR normalization (a0_pred vs a0_obs)
  - Bessel-projected transport integrals (instantaneous FD coupling)
  - a0(z) redshift evolution
  - Dimensionless epsilon for the master equation

The operating MOND amplitude is supplied here by the mature
vortex-equilibrium / coherence-boundary closure (I.1/I.2). The
time-dependent PDE computation remains the future theorem-level
strengthening, with source-kernel/profile realization tracked as the
I.5 numerical refinement.

Reference: MAIN/12. ISPG_MOND.tex, Secs. 5-8
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros, i0e, i1e, k0e, k1e
from scipy.integrate import quad, solve_bvp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --------------- Physical constants (SI) ---------------
G     = 6.67430e-11      # m^3 kg^-1 s^-2
c     = 2.99792458e8     # m/s
H0    = 67.4e3 / 3.0857e22  # s^-1 (67.4 km/s/Mpc)
Msun  = 1.98848e30       # kg
kpc   = 3.08568e19       # m

a0_pred = c * H0 / (2 * np.pi)  # predicted MOND scale
a0_obs  = 1.2e-10               # observed (McGaugh 2016)

# --------------- Galaxy model ---------------
M_gal = 1.0e11 * Msun    # total baryonic mass
R_d   = 3.5 * kpc         # disk scale length (typical Milky Way)

r_M   = np.sqrt(G * M_gal / a0_pred)  # MOND transition radius
r_s   = 2 * G * M_gal / c**2          # Schwarzschild radius
eps   = 3 * np.pi * r_s / r_M         # dimensionless Hubble parameter


def enclosed_mass_exp_disk(r):
    """Enclosed mass for a thin exponential disk (Eq. from Freeman 1970)."""
    y = r / (2 * R_d)
    return M_gal * (1 - (1 + r / R_d) * np.exp(-r / R_d))


def g_newtonian(r):
    """Newtonian gravitational acceleration for the exponential disk."""
    M_enc = enclosed_mass_exp_disk(r)
    return G * M_enc / r**2


def v_newtonian(r):
    """Newtonian circular velocity."""
    return np.sqrt(r * g_newtonian(r))


# --------------- MOND self-consistent solution ---------------
def solve_mond_self_consistent(r_arr, a0):
    """
    Solve g = g_N + g_h with g_h/g_N = a0/g.
    This is equivalent to: g^2 - g*g_N - a0*g_N = 0.
    """
    g_N = np.array([g_newtonian(r) for r in r_arr])
    discriminant = g_N**2 + 4 * a0 * g_N
    g_total = 0.5 * (g_N + np.sqrt(discriminant))
    mu = g_N / g_total
    x = g_total / a0
    return g_N, g_total, mu, x


# --------------- Bessel-projected transport integrals ---------------
def compute_transport_integrals(r0, a0):
    """
    Compute Omega_tr and tau_rel from the Bessel-projected integrals
    (Appendix 12, Eqs. omega_tr_exact -- denominator_exact).

    r0: outer boundary of the Bessel domain (~ galactic scale radius).
    Returns: Omega_tr, tau_rel, and the product Omega_tr * tau_rel.
    """
    x0 = jn_zeros(0, 1)[0]  # 2.4048...
    k_r = x0 / r0

    tau_rel = 1.0 / (c * k_r)  # = r0 / (x0 * c)

    def integrand_num(r):
        if r < 1e-3 * kpc:
            return 0.0
        v_phi = v_newtonian(r)
        J_ang = 0.5 * enclosed_mass_exp_disk(r) * v_phi * r
        omega_fd = 2 * G * J_ang / (c**2 * r**3)
        phi_N = -G * enclosed_mass_exp_disk(r) / (c**2 * r)
        return r * j0(k_r * r) * omega_fd * k_r * phi_N

    def integrand_den(r):
        if r < 1e-3 * kpc:
            return 0.0
        g_loc = g_newtonian(r)
        phi_N = -G * enclosed_mass_exp_disk(r) / (c**2 * r)
        return r * j0(k_r * r) * phi_N / tau_rel

    num, _ = quad(integrand_num, 0.01 * kpc, r0, limit=200)
    den, _ = quad(integrand_den, 0.01 * kpc, r0, limit=200)

    if abs(den) < 1e-50:
        return np.nan, tau_rel, np.nan

    Omega_tr = num / den
    return Omega_tr, tau_rel, Omega_tr * tau_rel


# --------------- Rotation curve comparison ---------------
def rotation_curve_comparison(r_arr, a0):
    """Compute Newtonian, MOND, and observed-style rotation curves."""
    g_N, g_tot, mu, x = solve_mond_self_consistent(r_arr, a0)
    v_N = np.sqrt(r_arr * g_N)
    v_MOND = np.sqrt(r_arr * g_tot)
    v_flat_pred = (a0 * G * M_gal)**0.25
    return v_N, v_MOND, v_flat_pred


# --------------- BTFR check ---------------
def check_btfr(a0):
    """Verify the baryonic Tully-Fisher relation v^4 = G*M*a0."""
    v_flat = (a0 * G * M_gal)**0.25
    v_flat_kms = v_flat / 1e3
    return v_flat_kms


# =============== MAIN COMPUTATION ===============
if __name__ == "__main__":
    print("=" * 65)
    print("MOND Predictions — ISPG Theory (Appendix 12)")
    print("=" * 65)

    # --- Basic parameters ---
    print(f"\n--- Galaxy model ---")
    print(f"  M_gal    = {M_gal/Msun:.1e} M_sun")
    print(f"  R_d      = {R_d/kpc:.1f} kpc")
    print(f"  r_M      = {r_M/kpc:.1f} kpc  (MOND transition radius)")
    print(f"  r_s      = {r_s:.3e} m  (Schwarzschild radius)")
    print(f"  epsilon  = {eps:.3e}  (dimensionless Hubble parameter)")
    print(f"  Fiducial disk: R_d = 3.5 kpc "
          f"(MW-like; matches PDF figures in MOND/Python/PDF/)")

    print(f"\n--- MOND scale ---")
    print(f"  a0 (predicted) = {a0_pred:.4e} m/s^2")
    print(f"  a0 (observed)  = {a0_obs:.4e} m/s^2")
    print(f"  ratio          = {a0_pred/a0_obs:.3f}")

    # --- BTFR ---
    v_btfr = check_btfr(a0_pred)
    print(f"\n--- Baryonic Tully-Fisher ---")
    print(f"  v_flat (predicted) = {v_btfr:.1f} km/s")
    print(f"  v_flat (observed, MW-like) ~ 220 km/s")

    # --- Self-consistent MOND solution ---
    r_arr = np.geomspace(0.5 * kpc, 100 * kpc, 500)
    g_N, g_tot, mu, x = solve_mond_self_consistent(r_arr, a0_pred)

    print(f"\n--- Interpolating function check ---")
    mu_pred = x / (1 + x)
    rms_err = np.sqrt(np.mean((mu - mu_pred)**2))
    print(f"  mu(x) vs x/(1+x): RMS residual = {rms_err:.2e}")
    print(f"  (Algebraic self-consistency of the two-channel equation)")

    # Bessel-projected transport integrals: instantaneous FD coupling.
    # These pointwise integrals diagnose the epsilon-suppressed local kernel;
    # the mature vortex-equilibrium closure supplies the operating MOND branch.
    print(f"\n--- Bessel-projected transport integrals (instantaneous) ---")
    test_radii = [5, 10, 20, 40]
    for r0_kpc in test_radii:
        r0 = r0_kpc * kpc
        Omega_tr, tau_rel_val, product = compute_transport_integrals(r0, a0_pred)
        g_at_r0 = g_newtonian(r0)
        a0_over_g = a0_pred / g_at_r0
        print(f"  r0 = {r0_kpc:2d} kpc: "
              f"instantaneous = {product:.2e}, "
              f"required a0/g_N = {a0_over_g:.2e}" if not np.isnan(product)
              else f"  r0 = {r0_kpc:2d} kpc: computation failed")
    print(f"  => Pointwise finite-difference / Bessel coupling is")
    print(f"     epsilon-suppressed; the full MOND amplitude is")
    print(f"     supplied by the mature vortex-equilibrium operating")
    print(f"     closure (I.1/I.2). Source-kernel/profile realization")
    print(f"     refinement remains the I.5 numerical task.")

    # --- Rotation curves ---
    v_N, v_MOND, v_flat = rotation_curve_comparison(r_arr, a0_pred)

    print(f"\n--- Rotation curve ---")
    r_check = [5, 10, 20, 50, 100]
    print(f"  {'r (kpc)':>10s}  {'v_N (km/s)':>12s}  {'v_MOND (km/s)':>14s}  {'mu':>8s}  {'x=g/a0':>10s}")
    for rc in r_check:
        idx = np.argmin(np.abs(r_arr / kpc - rc))
        print(f"  {r_arr[idx]/kpc:10.1f}  {v_N[idx]/1e3:12.1f}  {v_MOND[idx]/1e3:14.1f}  "
              f"{mu[idx]:8.4f}  {x[idx]:10.3f}")

    # --- Redshift evolution: dual-background comparison ---
    print(f"\n--- Redshift evolution of a0 ---")
    z_vals = [0, 0.5, 1.0, 2.0, 3.0, 5.0]

    print(f"\n  ISPG comparison background"
          f" (baryons-only manuscript table):")
    ispg_table = {
        0.0: 1.00,
        0.5: 1.07,
        1.0: 1.16,
        2.0: 1.39,
        3.0: 1.63,
        5.0: 2.10,
    }
    for z in z_vals:
        Hz_over_H0 = ispg_table[z]
        print(f"    z = {z:.1f}: H(z)/H0 = {Hz_over_H0:.3f}, "
              f"a0(z)/a0(0) = {Hz_over_H0:.3f}, "
              f"v_flat shift = {Hz_over_H0**0.25:.3f}")

    print(f"\n  Standard LambdaCDM comparison"
          f" (Omega_m=0.315, Omega_L=0.685):")
    Omega_m, Omega_L = 0.315, 0.685
    for z in z_vals:
        Hz = H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_L)
        a0z = c * Hz / (2 * np.pi)
        print(f"    z = {z:.1f}: H(z)/H0 = {Hz/H0:.3f}, "
              f"a0(z)/a0(0) = {a0z/a0_pred:.3f}, "
              f"v_flat shift = {(a0z/a0_pred)**0.25:.3f}")

    # =============== PLOTS ===============
    outdir = Path(__file__).parent / "PDF"
    outdir.mkdir(exist_ok=True)

    # --- Plot 1: Rotation curve ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(r_arr / kpc, v_N / 1e3, 'b--', lw=1.5, label='Newtonian')
    ax.plot(r_arr / kpc, v_MOND / 1e3, 'r-', lw=2,
            label=r'ISPG MOND: $\mu=x/(1+x)$')
    ax.axhline(v_flat / 1e3, color='gray', ls=':', lw=1,
               label=f'$v_{{\\rm flat}}=(a_0 GM)^{{1/4}}={v_flat/1e3:.0f}$ km/s')
    ax.set_xlabel('$r$ (kpc)', fontsize=13)
    ax.set_ylabel('$v_{\\rm circ}$ (km/s)', fontsize=13)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 300)
    ax.legend(fontsize=11)
    ax.set_title('Rotation curve: exponential disk with ISPG-MOND', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_mond_rotation_numerical.pdf', dpi=150)
    plt.close(fig)

    # --- Plot 2: Interpolating function ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x_plot = np.geomspace(0.01, 100, 500)
    mu_simple = x_plot / (1 + x_plot)
    mu_standard = x_plot / np.sqrt(1 + x_plot**2)
    ax.plot(x_plot, mu_simple, 'r-', lw=2,
            label=r'ISPG: $\mu = x/(1+x)$')
    ax.plot(x_plot, mu_standard, 'b--', lw=1.5,
            label=r'Standard: $\mu = x/\sqrt{1+x^2}$')
    ax.set_xscale('log')
    ax.set_xlabel('$x = g/a_0$', fontsize=13)
    ax.set_ylabel(r'$\mu(x)$', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.axvline(1, color='gray', ls=':', lw=0.8, label='$x=1$ (MOND transition)')
    ax.legend(fontsize=11)
    ax.set_title('MOND interpolating function', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_mond_interpolation_numerical.pdf', dpi=150)
    plt.close(fig)

    # --- Plot 3: mu from self-consistent solution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, mu, 'ro', ms=2, label='Self-consistent solution')
    x_fine = np.geomspace(min(x), max(x), 500)
    ax1.plot(x_fine, x_fine / (1 + x_fine), 'k-', lw=1.5,
             label=r'$\mu = x/(1+x)$')
    ax1.set_xscale('log')
    ax1.set_xlabel('$x = g/a_0$', fontsize=13)
    ax1.set_ylabel(r'$\mu(x)$', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_title(r'$\mu(x)$ from self-consistent equation', fontsize=13)

    residual = mu - x / (1 + x)
    ax2.plot(r_arr / kpc, residual, 'r-', lw=1.5)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('$r$ (kpc)', fontsize=13)
    ax2.set_ylabel(r'$\mu_{\rm num} - x/(1+x)$', fontsize=13)
    ax2.set_title('Residual (should be $\\sim 0$)', fontsize=13)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.tight_layout()
    fig.savefig(outdir / 'fig_mond_mu_verification.pdf', dpi=150)
    plt.close(fig)

    # --- Plot 4: a0(z) evolution ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    z_arr = np.linspace(0, 5, 200)
    Hz_arr = H0 * np.sqrt(Omega_m * (1 + z_arr)**3 + Omega_L)
    a0z_arr = c * Hz_arr / (2 * np.pi)
    ax.plot(z_arr, a0z_arr / a0_pred, 'r-', lw=2, label=r'$\Lambda$CDM')
    z_md = np.linspace(0, 5, 200)
    Hz_md = H0 * (1 + z_md)**1.5
    ax.plot(z_md, Hz_md / H0, 'b:', lw=1.5, label='Matter domination')
    ax.axhline(1, color='gray', ls='--', lw=1, label='Standard MOND ($a_0$ = const)')
    ax.fill_between([0.5, 3], 0, 6, alpha=0.1, color='orange',
                    label='JWST/Euclid window')
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'$a_0(z)/a_0(0)$', fontsize=13)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(r'Falsifiable prediction: $a_0(z) = cH(z)/(2\pi)$', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'fig_a0_evolution_numerical.pdf', dpi=150)
    plt.close(fig)

    print(f"\n--- Output ---")
    print(f"  Plots saved to {outdir}/")
    print(f"    fig_mond_rotation_numerical.pdf")
    print(f"    fig_mond_interpolation_numerical.pdf")
    print(f"    fig_mond_mu_verification.pdf")
    print(f"    fig_a0_evolution_numerical.pdf")
    print(f"\n{'=' * 65}")
    print("DONE.")
