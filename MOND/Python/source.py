"""
Baryonic source profiles for the MOND PDE computation.

Thin exponential disk in dimensionless coordinates xi = r / r_M.
All functions accept and return numpy arrays.

Key relations (equatorial plane, azimuthally symmetric):
  eta  = r_M / R_d                          (scale ratio)
  f(xi)      = eta^2 exp(-eta xi)           (dimensionless Poisson source)
  m_enc(xi)  = 1 - (1 + eta xi) exp(-eta xi)  (enclosed mass / M_gal)
  g_N(xi)    = a0 m_enc(xi) / xi^2          (Newtonian acceleration)
  v_circ(xi) = sqrt(xi r_M g_N(xi))         (circular velocity)
  Omega_hat  = Omega r_M / c                (dimensionless angular velocity)

Reference: ISPG_MOND.tex Sec. 11, line 1443-1448.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from constants import (
    G, c, a0, M_gal, R_d, h_d, r_M, kpc, eps,
    xi_min, xi_max,
)

# =====================================================================
# Derived ratio
# =====================================================================
eta = r_M / R_d   # typically ~ 1.16

# =====================================================================
# Source and mass profiles  (all take dimensionless xi)
# =====================================================================

def f_source(xi):
    """Dimensionless baryonic source on the equatorial plane.

    Satisfies  -hat{nabla}^2 u_N = f  with hat{nabla}^2 the
    dimensionless cylindrical Laplacian (1/xi d/dxi (xi d/dxi)).
    Normalization: int_0^inf f(xi) xi dxi = 1.
    """
    return eta**2 * np.exp(-eta * xi)


def f_source_3d(xi, zeta):
    """Full 3D dimensionless source f(xi, zeta).

    zeta = z / r_M.  Vertical profile: sech^2-like with scale h_d.
    The normalization is set so that integrating over zeta recovers
    the equatorial-plane source when convolved appropriately.
    """
    return (eta**2 * np.exp(-eta * xi)
            * (r_M / (2 * h_d)) * np.exp(-np.abs(zeta) * r_M / h_d))


def m_enc(xi):
    """Dimensionless enclosed mass fraction M_enc(r) / M_gal.

    Thin exponential disk (Freeman 1970).
    """
    y = eta * xi
    return 1.0 - (1.0 + y) * np.exp(-y)


def g_newton(xi):
    """Newtonian gravitational acceleration g_N(xi) [m/s^2].

    g_N = a0 * m_enc(xi) / xi^2, using GM = a0 r_M^2.
    """
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    mask = xi > 0
    out[mask] = a0 * m_enc(xi[mask]) / xi[mask]**2
    return out


def g_newton_dimless(xi):
    """Dimensionless Newtonian acceleration g_N / a0."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    mask = xi > 0
    out[mask] = m_enc(xi[mask]) / xi[mask]**2
    return out


def v_circ(xi):
    """Circular velocity [m/s] at dimensionless radius xi."""
    return np.sqrt(np.maximum(xi * r_M * g_newton(xi), 0.0))


def v_circ_flat():
    """BTFR asymptotic flat velocity [m/s]."""
    return (a0 * G * M_gal)**0.25


def omega_dimless(xi):
    """Dimensionless angular velocity Omega_hat = Omega * r_M / c.

    Omega = v_circ / (xi * r_M), so  Omega_hat = v_circ / (xi * c).
    """
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    mask = xi > 0
    v = v_circ(xi[mask])
    out[mask] = v / (xi[mask] * c)
    return out


# =====================================================================
# Diagnostics and plotting
# =====================================================================

def print_diagnostics(xi):
    """Print source profile diagnostics at selected radii."""
    sep = "=" * 65
    print(sep)
    print("  Step 1.2 -- Baryonic Source Diagnostics")
    print(sep)

    print(f"\n  eta = r_M / R_d = {eta:.4f}")
    print(f"  (r_M = {r_M/kpc:.2f} kpc, R_d = {R_d/kpc:.1f} kpc)")

    # Mass normalization check
    from scipy.integrate import quad
    mass_check, _ = quad(lambda x: f_source(x) * x, 0, 500)
    print(f"\n  Source normalization: int_0^inf f(xi) xi dxi = {mass_check:.6f}  (should be 1)")

    print(f"\n  {'xi':>8s}  {'r (kpc)':>8s}  {'m_enc':>8s}  {'g_N (m/s2)':>12s}  "
          f"{'g_N/a0':>10s}  {'v (km/s)':>10s}  {'Omega_hat':>12s}")
    print("  " + "-" * 82)
    for x in xi:
        r_kpc = x * r_M / kpc
        me = m_enc(x)
        gn = g_newton(x)
        gn_d = g_newton_dimless(x)
        vc = v_circ(x)
        oh = omega_dimless(x)
        print(f"  {x:8.3f}  {r_kpc:8.2f}  {me:8.5f}  {gn:12.4e}  "
              f"{gn_d:10.4f}  {vc/1e3:10.2f}  {oh:12.4e}")

    # Asymptotic checks
    print(f"\n  --- Asymptotic checks ---")
    xi_inner = 0.01
    gn_inner = g_newton_dimless(xi_inner)
    gn_inner_analytic = eta**2 / 2  # m_enc ~ (eta*xi)^2/2 for small xi
    print(f"  xi={xi_inner}: g_N/a0 = {gn_inner:.4f},  analytic (eta^2/2) = {gn_inner_analytic:.4f}")

    xi_outer = 100.0
    gn_outer = g_newton_dimless(xi_outer)
    gn_outer_analytic = 1.0 / xi_outer**2  # m_enc -> 1 for large xi
    print(f"  xi={xi_outer}: g_N/a0 = {gn_outer:.6e},  analytic (1/xi^2) = {gn_outer_analytic:.6e}")

    v_flat = v_circ_flat()
    print(f"\n  BTFR flat velocity: v_flat = {v_flat/1e3:.2f} km/s")

    print(f"\n{sep}")


def make_plots(outdir=None):
    """Generate diagnostic plots for all source profiles."""
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    xi = np.geomspace(xi_min, xi_max, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- (a) Source f(xi) ---
    ax = axes[0, 0]
    ax.semilogy(xi, f_source(xi), 'b-', lw=2)
    ax.set_xlabel(r'$\xi = r/r_M$', fontsize=12)
    ax.set_ylabel(r'$f(\xi)$', fontsize=12)
    ax.set_title(r'(a) Baryonic source $f(\xi) = \eta^2 e^{-\eta\xi}$', fontsize=12)
    ax.set_xscale('log')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlim(xi_min, xi_max)

    # --- (b) Enclosed mass ---
    ax = axes[0, 1]
    ax.semilogx(xi, m_enc(xi), 'r-', lw=2)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$m_{\rm enc}(\xi) = M_{\rm enc}/M$', fontsize=12)
    ax.set_title(r'(b) Enclosed mass fraction', fontsize=12)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(0, 1.1)

    # --- (c) Newtonian acceleration ---
    ax = axes[0, 2]
    ax.loglog(xi, g_newton_dimless(xi), 'g-', lw=2, label=r'$g_N/a_0$')
    ax.loglog(xi, 1.0/xi**2, 'k:', lw=1, alpha=0.5, label=r'$1/\xi^2$ (point mass)')
    ax.axhline(1.0, color='orange', ls='--', lw=1, label=r'$g_N = a_0$ (MOND boundary)')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$g_N / a_0$', fontsize=12)
    ax.set_title(r'(c) Newtonian acceleration', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(xi_min, xi_max)

    # --- (d) Rotation curve ---
    ax = axes[1, 0]
    v = v_circ(xi) / 1e3
    v_flat = v_circ_flat() / 1e3
    ax.semilogx(xi, v, 'b-', lw=2, label='Newtonian')
    ax.axhline(v_flat, color='gray', ls='--', lw=1,
               label=f'$v_{{\\rm flat}} = {v_flat:.0f}$ km/s')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$v_{\rm circ}$ (km/s)', fontsize=12)
    ax.set_title(r'(d) Newtonian rotation curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(0, None)

    # --- (e) Dimensionless angular velocity ---
    ax = axes[1, 1]
    Oh = omega_dimless(xi)
    ax.loglog(xi, Oh, 'm-', lw=2, label=r'$\hat\Omega(\xi)$')
    Oh_kep = Oh[0] * (xi[0]/xi)**1.5  # Keplerian ~ xi^{-3/2}
    ax.loglog(xi, Oh_kep, 'k:', lw=1, alpha=0.5,
              label=r'$\propto \xi^{-3/2}$ (Keplerian)')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\hat\Omega = \Omega\,r_M/c$', fontsize=12)
    ax.set_title(r'(e) Dimensionless angular velocity', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)

    # --- (f) alpha(xi) = eps * Omega_hat * xi^2  (mode damping ratio) ---
    ax = axes[1, 2]
    alpha = eps * Oh * xi**2
    ax.loglog(xi, alpha, 'r-', lw=2)
    ax.axhline(1.0, color='k', ls='--', lw=1, label=r'$\alpha = 1$ (MOND transition)')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\alpha(\xi) = \varepsilon\hat\Omega\xi^2$', fontsize=12)
    ax.set_title(r'(f) Mode damping ratio $\alpha(\xi)$', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)
    # Where does alpha = 1?
    from scipy.interpolate import interp1d
    log_alpha = interp1d(np.log10(xi), np.log10(alpha), kind='linear')
    try:
        from scipy.optimize import brentq
        xi_alpha1 = 10**brentq(lambda lx: log_alpha(lx), np.log10(xi_min), np.log10(xi_max))
        ax.axvline(xi_alpha1, color='r', ls=':', lw=0.8)
        ax.annotate(f'$\\xi \\approx {xi_alpha1:.0e}$',
                    xy=(xi_alpha1, 1.0), fontsize=9, color='r',
                    xytext=(xi_alpha1*0.1, 5),
                    arrowprops=dict(arrowstyle='->', color='r'))
    except ValueError:
        pass  # alpha=1 outside domain

    fig.suptitle('Step 1.2: Baryonic Source Profiles (exponential disk)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fname = outdir / 'step1_2_source_profiles.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")
    return fname


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    xi_table = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
    print_diagnostics(xi_table)
    make_plots()
