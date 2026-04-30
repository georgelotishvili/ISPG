"""Generate publication-quality PDF figures for the ISPG_MOND manuscript."""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'text.usetex': False,
    'figure.dpi': 300,
})

from constants import G, c, H0, a0, Msun, kpc, r_M, M_gal
from radial_profile import self_consistent_g, g_newton_dimless
from phi_evolution import (coupling_ratio, beta_eff, H_of_z,
                           find_dw0_for_beta, beta_crit_from_phase4,
                           Omega_b, Omega_L)

PDF_DIR = Path(__file__).parent.parent / "PDF"
PDF_DIR.mkdir(exist_ok=True)


def fig_mu_verification():
    """Figure 1: mu(x) = x/(1+x) verification."""
    from chebyshev import cheb_matrices
    from newtonian import solve_newtonian

    s, xi, U_N, D1_raw = solve_newtonian()
    _, D1, D2 = cheb_matrices()

    g_eff, g_N = self_consistent_g(xi, C=1.0)
    mu = g_N / np.maximum(g_eff, 1e-30)
    x = g_eff

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    mask = (x > 0.01) & (x < 100) & (xi > 0.05) & (xi < 80)
    x_th = np.geomspace(0.01, 100, 500)

    ax1.plot(x[mask], mu[mask], 'b-', lw=2.5, label='Numerical (spectral)')
    ax1.plot(x_th, x_th / (1 + x_th), 'r--', lw=1.5,
             label=r'$x/(1+x)$')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$x = g/a_0$')
    ax1.set_ylabel(r'$\mu(x) = g_N/g$')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.set_title(r'(a) Interpolating function')

    mask2 = (xi > 0.05) & (xi < 80)
    ax2.loglog(xi[mask2], g_N[mask2], 'b-', lw=2, label=r'$g_N/a_0$')
    ax2.loglog(xi[mask2], g_eff[mask2], 'r-', lw=2, label=r'$g/a_0$')
    g_h = g_eff - g_N
    ax2.loglog(xi[mask2], g_h[mask2], 'g--', lw=2, label=r'$g_h/a_0$')
    ax2.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax2.set_xlabel(r'$\xi = r/r_M$')
    ax2.set_ylabel(r'Gravity / $a_0$')
    ax2.legend(fontsize=9)
    ax2.set_title(r'(b) Gravity profiles')

    fig.tight_layout()
    fname = PDF_DIR / 'fig_mond_numerical_mu.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_gap_a():
    """Figure 2: Gap A — coupling ratio and allowed parameter region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bc_dict = beta_crit_from_phase4()

    # (a) beta_eff vs dw0
    dw0_arr = np.linspace(0.01, 0.5, 200)
    for z_f, color, ls in [(10, '#2166ac', '-'), (50, '#4dac26', '--'),
                            (100, '#d01c8b', ':')]:
        betas = [beta_eff(z_f, 1.0, d) for d in dw0_arr]
        ax1.plot(dw0_arr, betas, ls=ls, lw=2, color=color,
                 label=rf'$z_f = {z_f}$')
        if z_f in bc_dict:
            ax1.axhline(bc_dict[z_f], color=color, ls=':', lw=0.8, alpha=0.5)

    ax1.set_xlabel(r'$\delta w_0$')
    ax1.set_ylabel(r'$\beta_{\rm eff}$')
    ax1.legend()
    ax1.set_ylim(-1, 5)
    ax1.set_title(r'(a) $\beta_{\rm eff}$ vs $\delta w_0$ ($\alpha=1$)')

    # (b) Allowed parameter region
    from phi_evolution import _find_exact_dw0

    z_forms_plot = [5, 10, 20, 50, 100, 200]
    colors_cm = plt.cm.viridis(np.linspace(0.15, 0.9, len(z_forms_plot)))
    alpha_scan = np.arange(0.3, 5.1, 0.3)

    for z_f, color in zip(z_forms_plot, colors_cm):
        dw0s = []
        alphas_valid = []
        for a_val in alpha_scan:
            d = _find_exact_dw0(z_f, a_val)
            if np.isfinite(d) and 0.0005 < d < 5:
                dw0s.append(d)
                alphas_valid.append(a_val)
        if dw0s:
            ax2.semilogy(alphas_valid, dw0s, 'o-', color=color, lw=1.5,
                         ms=3, label=rf'$z_f={z_f}$')

    ax2.axhspan(0.01, 0.1, alpha=0.15, color='#ffffb2',
                label=r'Predicted $\delta w_0$')
    ax2.axhline(0.01, color='#fd8d3c', ls=':', lw=0.8)
    ax2.axhline(0.1, color='#fd8d3c', ls=':', lw=0.8)
    ax2.axvspan(1.0, 2.0, alpha=0.08, color='#a6cee3',
                label=r'Physical $\alpha$')

    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\delta w_0$ for $C=1$')
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    ax2.set_ylim(5e-4, 5)
    ax2.set_xlim(0, 5)
    ax2.set_title('(b) Allowed parameter region')

    fig.tight_layout()
    fname = PDF_DIR / 'fig_mond_gap_a.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def fig_poisson():
    """Figure 3: Poisson redistribution verification."""
    from chebyshev import cheb_matrices
    from newtonian import solve_newtonian

    s, xi, U_N, D1_raw = solve_newtonian()
    _, D1, D2 = cheb_matrices()

    g_eff, g_N = self_consistent_g(xi, C=1.0)
    g_h = g_eff - g_N

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    mask = (xi > 0.05) & (xi < 80)

    # (a) g_h * g = a0 * g_N check
    product = g_h * g_eff
    ax1.loglog(xi[mask], product[mask], 'b-', lw=2.5,
               label=r'$g_h \cdot g\, /\, a_0^2$')
    ax1.loglog(xi[mask], g_N[mask], 'r--', lw=1.5,
               label=r'$g_N / a_0$')
    ax1.set_xlabel(r'$\xi = r/r_M$')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title(r'(a) Verify $g_h \cdot g = a_0\,g_N$')

    # (b) Effective source rho_h
    dUh_ds = -xi**2 * g_h
    N_pts = len(s)
    U_h = np.zeros(N_pts)
    for i in range(N_pts - 2, -1, -1):
        ds_step = s[i+1] - s[i]
        U_h[i] = U_h[i+1] - 0.5 * (dUh_ds[i] + dUh_ds[i+1]) * ds_step

    laplacian_U_h = -(1.0 / xi**2) * (D2 @ U_h)

    ax2.plot(xi[mask], laplacian_U_h[mask], 'b-', lw=2.5)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$\nabla^2\varphi_h$ (eff.\ source $\rho_h$)')
    ax2.set_title(r'(b) $\rho_h \geq 0$ at all radii')

    fig.tight_layout()
    fname = PDF_DIR / 'fig_mond_poisson_proof.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    print("Generating PDF figures for manuscript...")
    fig_mu_verification()
    fig_gap_a()
    fig_poisson()
    print("Done.")
