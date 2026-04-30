"""
Frame-dragging profiles for the MOND PDE computation.

Physical quantities (equatorial plane, theta = pi/2):
  J_enc(r)    = enclosed angular momentum of the disk
  omega_FD(r) = 2 G J_enc / (c^2 r^3)   [Lense-Thirring]
  delta_FD(xi)= 2 omega_FD(xi r_M) r_M / c   [dimensionless, eq:delta_FD]
  alpha(xi)   = eps * Omega_hat * xi^2   [mode damping ratio, eq:alpha]

Reference: ISPG_MOND.tex Secs. 6.1, 9.1, 11.2; eqs. (frame_dragging),
           (delta_FD), (alpha).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from constants import G, c, a0, M_gal, R_d, r_M, kpc, eps, xi_min, xi_max
from source import eta, m_enc, g_newton, v_circ, omega_dimless


# =====================================================================
# Enclosed angular momentum
# =====================================================================

def _build_J_enc_table(xi_arr):
    """Compute enclosed angular momentum J_enc(xi) by numerical integration.

    J_enc(r) = integral_0^r  Sigma(r') v_circ(r') r'^2  2pi r' dr' / (2pi R_d^2)
             = (M_gal / R_d^2) integral_0^r exp(-r'/R_d) v_circ(r') r'^2 dr'

    In dimensionless xi = r/r_M:
    J_enc(xi) = M_gal r_M eta^2 integral_0^xi exp(-eta xi') v_circ(xi') xi'^2 dxi'
    """
    vc = v_circ(xi_arr)    # [m/s]

    integrand = eta**2 * np.exp(-eta * xi_arr) * vc * xi_arr**2
    # Prepend xi=0 point (integrand = 0 there)
    xi_ext = np.concatenate(([0.0], xi_arr))
    int_ext = np.concatenate(([0.0], integrand))

    J_cumul = cumulative_trapezoid(int_ext, xi_ext, initial=0.0)
    J_enc = M_gal * r_M * J_cumul[1:]   # skip the prepended zero, [kg m^2/s]
    return J_enc


# =====================================================================
# Frame-dragging angular velocity
# =====================================================================

def omega_FD(xi_arr):
    """Lense-Thirring frame-dragging angular velocity [rad/s].

    omega_FD(r) = 2 G J_enc(r) / (c^2 r^3)
    At the equatorial plane (sin^2 theta = 1).
    """
    xi_arr = np.asarray(xi_arr, dtype=float)
    J_enc = _build_J_enc_table(xi_arr)
    r = xi_arr * r_M
    w = np.zeros_like(xi_arr)
    mask = r > 0
    w[mask] = 2 * G * J_enc[mask] / (c**2 * r[mask]**3)
    return w


def delta_FD(xi_arr):
    """Dimensionless frame-dragging coupling (eq:delta_FD).

    delta_FD(xi) = 2 omega_FD(xi r_M) r_M / c.
    """
    return 2 * omega_FD(xi_arr) * r_M / c


def alpha_mode(xi_arr):
    """Mode damping ratio (eq:alpha).

    alpha(xi) = eps * Omega_hat(xi) * xi^2.
    When alpha >= 1, the mode is Hubble-damped (MOND regime).
    """
    xi_arr = np.asarray(xi_arr, dtype=float)
    Oh = omega_dimless(xi_arr)
    return eps * Oh * xi_arr**2


# =====================================================================
# Diagnostics
# =====================================================================

def run_diagnostics():
    """Print frame-dragging profile diagnostics."""
    sep = "=" * 65
    print(sep)
    print("  Step 3.1 -- Frame-Dragging Profiles")
    print(sep)

    xi = np.geomspace(xi_min, xi_max, 2000)
    wFD = omega_FD(xi)
    dFD = delta_FD(xi)
    alpha = alpha_mode(xi)

    print(f"\n  {'xi':>8s}  {'r (kpc)':>8s}  {'omega_FD':>12s}  {'delta_FD':>12s}  {'alpha':>12s}")
    print("  " + "-" * 58)
    sel = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        print(f"  {xi[idx]:8.4f}  {xi[idx]*r_M/kpc:8.2f}  {wFD[idx]:12.4e}  "
              f"{dFD[idx]:12.4e}  {alpha[idx]:12.4e}")

    # Keplerian asymptotic check: omega_FD ~ xi^{-5/2}
    print(f"\n  --- Asymptotic scaling (large xi) ---")
    xi_test = [10.0, 30.0, 100.0]
    for i in range(len(xi_test) - 1):
        idx1 = np.argmin(np.abs(xi - xi_test[i]))
        idx2 = np.argmin(np.abs(xi - xi_test[i+1]))
        slope = np.log(wFD[idx2]/wFD[idx1]) / np.log(xi[idx2]/xi[idx1])
        print(f"  d(log omega_FD)/d(log xi) between xi={xi_test[i]:.0f} "
              f"and {xi_test[i+1]:.0f}: {slope:.2f}  (Keplerian: -5/2 = -2.50)")

    # alpha scaling
    print(f"\n  --- alpha(xi) = eps * Omega_hat * xi^2 ---")
    print(f"  For Keplerian Omega ~ xi^{{-3/2}}: alpha ~ eps * xi^{{1/2}}")
    print(f"  alpha = 1 requires xi ~ eps^{{-2}} = {eps**(-2):.2e}")

    # Extrapolate where alpha = 1 using power-law from domain boundary
    alpha_100 = alpha[np.argmin(np.abs(xi - 100.0))]
    # alpha ~ eps * Omega0 * xi^{1/2}, so alpha=1 at xi = 100*(1/alpha_100)^2
    xi_est = 100.0 * (1.0 / alpha_100)**2
    print(f"  alpha(xi=100) = {alpha_100:.4e}")
    print(f"  Extrapolated xi for alpha=1: ~{xi_est:.2e}")
    print(f"  This is {xi_est*r_M/kpc:.0e} kpc -- far outside the galaxy.")

    # delta_FD typical values
    print(f"\n  --- delta_FD scale ---")
    print(f"  delta_FD(xi=1) = {dFD[np.argmin(np.abs(xi-1.0))]:.4e}")
    print(f"  eps * delta_FD(xi=1) = {eps * dFD[np.argmin(np.abs(xi-1.0))]:.4e}")
    print(f"  (This is the effective coupling strength in the master eq.)")

    print(f"\n{sep}")
    return xi, wFD, dFD, alpha


def make_plots(xi, wFD, dFD, alpha, outdir=None):
    """Generate frame-dragging diagnostic plots."""
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) omega_FD
    ax = axes[0, 0]
    ax.loglog(xi, wFD, 'b-', lw=2)
    # Keplerian reference
    idx_ref = np.argmin(np.abs(xi - 10.0))
    kep_ref = wFD[idx_ref] * (xi[idx_ref]/xi)**2.5
    ax.loglog(xi, kep_ref, 'k:', lw=1, alpha=0.5, label=r'$\propto \xi^{-5/2}$ (Keplerian)')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\omega_{\rm FD}$ (rad/s)', fontsize=12)
    ax.set_title(r'(a) Frame-dragging angular velocity', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)

    # (b) delta_FD
    ax = axes[0, 1]
    ax.loglog(xi, dFD, 'r-', lw=2)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\delta_{\rm FD}(\xi)$', fontsize=12)
    ax.set_title(r'(b) Dimensionless frame-dragging $\delta_{\rm FD}$', fontsize=12)
    ax.set_xlim(xi_min, xi_max)

    # (c) alpha
    ax = axes[1, 0]
    ax.loglog(xi, alpha, 'g-', lw=2)
    ax.axhline(1.0, color='k', ls='--', lw=1.5, label=r'$\alpha = 1$')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\alpha(\xi) = \varepsilon \hat\Omega \xi^2$', fontsize=12)
    ax.set_title(r'(c) Mode damping ratio', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)
    # Annotate max alpha
    alpha_max = np.max(alpha)
    ax.annotate(f'max $\\alpha = {alpha_max:.1e}$',
                xy=(xi[np.argmax(alpha)], alpha_max),
                xytext=(1, alpha_max*10), fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    # (d) eps * delta_FD (effective coupling)
    ax = axes[1, 1]
    eff_coupling = eps * dFD
    ax.loglog(xi, eff_coupling, 'm-', lw=2, label=r'$\varepsilon\,\delta_{\rm FD}$')
    ax.loglog(xi, alpha, 'g--', lw=1.5, label=r'$\alpha = \varepsilon\hat\Omega\xi^2$')
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel('Dimensionless coupling', fontsize=12)
    ax.set_title(r'(d) Effective coupling strengths', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)

    fig.suptitle('Step 3.1: Frame-Dragging Profiles', fontsize=14, y=1.01)
    fig.tight_layout()
    fname = outdir / 'step3_1_frame_dragging.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")
    return fname


if __name__ == "__main__":
    xi, wFD, dFD, alpha = run_diagnostics()
    make_plots(xi, wFD, dFD, alpha)
