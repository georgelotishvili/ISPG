"""
Strategy B: Two-channel transport model — algebraic coefficient scan.

Model:  g = g_N + g_h,  where  g_h / g_N = C1 C2 (a0 / g).
        tau_rel = C1 c/g,   Omega_tr = C2 a0/c.

Self-consistency:  g^2 - g_N g - C a0 g_N = 0,  C = C1 C2.
Solution:          g = [g_N + sqrt(g_N^2 + 4 C a0 g_N)] / 2.
Interpolation:     mu(x) = g_N / g = x / (x + C),   x = g / a0.
Self-consistent:   C1 = C2 = 1  =>  mu = x/(1+x).

Reference: ISPG_MOND.tex Sec. 5-6; plan Steps 2.1-2.3.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from constants import a0, r_M, kpc, xi_min, xi_max
from source import g_newton_dimless, m_enc


# =====================================================================
# Core solver
# =====================================================================

def solve_two_channel(xi, C):
    """Solve the self-consistency equation g = g_N(1 + C a0/g).

    Parameters
    ----------
    xi : array, dimensionless radii
    C  : float, product C1*C2

    Returns
    -------
    g_dimless : g / a0
    mu        : g_N / g
    x         : g / a0  (same as g_dimless)
    """
    g_N = g_newton_dimless(xi)           # g_N / a0

    discriminant = g_N**2 + 4 * C * g_N  # (g_N/a0)^2 + 4C(g_N/a0)
    g_total = 0.5 * (g_N + np.sqrt(discriminant))   # g / a0

    mu = g_N / g_total
    x = g_total
    return g_total, mu, x


def mu_target(x):
    """Target interpolating function mu = x/(1+x)."""
    return x / (1.0 + x)


# =====================================================================
# Step 2.1: C1 = C2 = 1 test
# =====================================================================

def step_2_1():
    """Report mu(x) for C1 = C2 = 1."""
    sep = "=" * 65
    print(sep)
    print("  Step 2.1 -- Self-Consistency Equation (C1 = C2 = 1)")
    print(sep)

    xi = np.geomspace(xi_min, xi_max, 2000)
    g, mu, x = solve_two_channel(xi, C=1.0)
    mu_exact = mu_target(x)

    residual = mu - mu_exact
    rms = np.sqrt(np.mean(residual**2))
    max_err = np.max(np.abs(residual))

    print(f"\n  C = C1*C2 = 1.0")
    print(f"\n  mu(x) vs x/(1+x):")
    print(f"  RMS residual:  {rms:.2e}")
    print(f"  Max residual:  {max_err:.2e}")

    print(f"\n  {'xi':>8s}  {'g/a0':>10s}  {'mu (num)':>10s}  {'x/(1+x)':>10s}  {'residual':>10s}")
    print("  " + "-" * 54)
    sel = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        print(f"  {xi[idx]:8.4f}  {x[idx]:10.4f}  {mu[idx]:10.6f}  "
              f"{mu_exact[idx]:10.6f}  {residual[idx]:10.2e}")

    # Analytic proof: for C=1, mu = x/(x+C) = x/(x+1) identically
    print(f"\n  Analytic: mu = x/(x+C) = x/(x+1) for C=1. QED.")
    print(f"  Numerical residual is machine-precision: {max_err:.2e}")
    print(sep)
    return xi, g, mu, x


# =====================================================================
# Step 2.2: Coefficient scan
# =====================================================================

def step_2_2():
    """Scan C = C1*C2 and find optimal value."""
    sep = "=" * 65
    print(sep)
    print("  Step 2.2 -- Coefficient Scan")
    print(sep)

    xi = np.geomspace(xi_min, xi_max, 2000)

    # --- (a) Scan C = C1*C2 ---
    C_vals = np.geomspace(0.01, 100, 500)
    rms_vals = np.zeros_like(C_vals)

    for i, C in enumerate(C_vals):
        _, mu, x = solve_two_channel(xi, C)
        # Only evaluate RMS where x in [0.01, 100]
        mask = (x >= 0.01) & (x <= 100)
        if mask.sum() > 0:
            residual = mu[mask] - mu_target(x[mask])
            rms_vals[i] = np.sqrt(np.mean(residual**2))
        else:
            rms_vals[i] = np.nan

    idx_opt = np.nanargmin(rms_vals)
    C_opt = C_vals[idx_opt]
    rms_opt = rms_vals[idx_opt]

    print(f"\n  (a) Scan C = C1*C2 in [0.01, 100] (500 points, log-spaced)")
    print(f"  Optimal C = {C_opt:.6f}")
    print(f"  RMS at optimal: {rms_opt:.2e}")
    print(f"  RMS at C=1.0:   {rms_vals[np.argmin(np.abs(C_vals - 1.0))]:.2e}")

    # --- (b) Independent (C1, C2) scan ---
    print(f"\n  (b) Independent (C1, C2) scan:")
    C1_vals = np.geomspace(0.1, 10, 50)
    C2_vals = np.geomspace(0.1, 10, 50)
    rms_2d = np.zeros((len(C1_vals), len(C2_vals)))

    for i, C1 in enumerate(C1_vals):
        for j, C2 in enumerate(C2_vals):
            _, mu, x = solve_two_channel(xi, C1 * C2)
            mask = (x >= 0.01) & (x <= 100)
            if mask.sum() > 0:
                residual = mu[mask] - mu_target(x[mask])
                rms_2d[i, j] = np.sqrt(np.mean(residual**2))

    # Find minimum in 2D
    imin, jmin = np.unravel_index(np.argmin(rms_2d), rms_2d.shape)
    print(f"  Optimal (C1, C2) = ({C1_vals[imin]:.4f}, {C2_vals[jmin]:.4f})")
    print(f"  Product C1*C2 = {C1_vals[imin]*C2_vals[jmin]:.4f}")
    print(f"  RMS at optimum: {rms_2d[imin, jmin]:.2e}")

    # Check degeneracy: all (C1,C2) with C1*C2 = 1 should give same RMS
    print(f"\n  Degeneracy check (C1*C2 = 1 locus):")
    print(f"  {'C1':>8s}  {'C2':>8s}  {'C1*C2':>8s}  {'RMS':>12s}")
    for C1_test in [0.2, 0.5, 1.0, 2.0, 5.0]:
        C2_test = 1.0 / C1_test
        _, mu, x = solve_two_channel(xi, C1_test * C2_test)
        mask = (x >= 0.01) & (x <= 100)
        res = mu[mask] - mu_target(x[mask])
        rms_test = np.sqrt(np.mean(res**2))
        print(f"  {C1_test:8.2f}  {C2_test:8.2f}  {C1_test*C2_test:8.4f}  {rms_test:12.2e}")

    print(sep)
    return C_vals, rms_vals, C1_vals, C2_vals, rms_2d


# =====================================================================
# Step 2.3: Verdict
# =====================================================================

def step_2_3(C_vals, rms_vals):
    """Report verdict on the C1*C2 = 1 closure condition."""
    sep = "=" * 65
    print(sep)
    print("  Step 2.3 -- VERDICT")
    print(sep)

    # The 1D scan has finite grid resolution; evaluate exactly at C = 1.0
    xi = np.geomspace(xi_min, xi_max, 2000)
    _, mu_exact_test, x_exact_test = solve_two_channel(xi, C=1.0)
    mask = (x_exact_test >= 0.01) & (x_exact_test <= 100)
    res_exact = mu_exact_test[mask] - mu_target(x_exact_test[mask])
    rms_at_1 = np.sqrt(np.mean(res_exact**2))

    # Also find grid optimum for comparison
    idx_opt = np.nanargmin(rms_vals)
    C_opt_grid = C_vals[idx_opt]
    rms_opt_grid = rms_vals[idx_opt]

    print(f"\n  Grid-scan optimal C = {C_opt_grid:.6f}  (grid resolution artifact)")
    print(f"  RMS at grid optimum:  {rms_opt_grid:.2e}")
    print(f"  RMS at exact C = 1.0: {rms_at_1:.2e}")

    is_consistent = rms_at_1 < 1e-10
    if is_consistent:
        print(f"\n  VERDICT: C1*C2 = 1 is ALGEBRAICALLY CONSISTENT.")
        print(f"  The two-channel equation g = g_N(1 + a0/g) yields")
        print(f"  mu(x) = x/(1+x) EXACTLY (to machine precision).")
        print(f"\n  This is a MATHEMATICAL IDENTITY, not a numerical result:")
        print(f"  mu = g_N/g = g/(g + C*a0) = x/(x+C).")
        print(f"  For C = 1:  mu = x/(x+1).  QED.")
        print(f"\n  The degeneracy in (C1, C2) is EXACT: mu depends only on C1*C2.")
        print(f"\n  IMPLICATION FOR PHASE 4:")
        print(f"  The closure is self-consistent. Phase 4 must verify that")
        print(f"  the PDE dynamically selects tau_rel = c/g and Omega_tr = a0/c")
        print(f"  (i.e., C1 = C2 = 1) from the bound-structure rotational transport balance.")
    else:
        print(f"\n  VERDICT: UNEXPECTED. RMS at C=1 = {rms_at_1:.2e} > 1e-10.")

    print(f"\n{sep}")
    return is_consistent


# =====================================================================
# Plots
# =====================================================================

def make_plots(xi, mu, x, C_vals, rms_vals,
               C1_vals, C2_vals, rms_2d, outdir=None):
    """Generate all Phase 2 diagnostic plots."""
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) mu(x) for C=1
    ax = axes[0, 0]
    ax.semilogx(x, mu, 'ro', ms=1.5, label=r'$C_1 C_2 = 1$ (numerical)')
    x_fine = np.geomspace(0.01, 100, 500)
    ax.semilogx(x_fine, mu_target(x_fine), 'b-', lw=2, label=r'$\mu = x/(1+x)$')
    ax.set_xlabel(r'$x = g/a_0$', fontsize=12)
    ax.set_ylabel(r'$\mu(x)$', fontsize=12)
    ax.set_title(r'(a) $\mu(x)$ for $C_1 C_2 = 1$', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    # (b) RMS vs C
    ax = axes[0, 1]
    ax.loglog(C_vals, rms_vals, 'r-', lw=2)
    ax.axvline(1.0, color='b', ls='--', lw=1.5, label='$C = 1$ (closure)')
    ax.set_xlabel(r'$C = C_1 C_2$', fontsize=12)
    ax.set_ylabel(r'RMS$[\mu - x/(1+x)]$', fontsize=12)
    ax.set_title(r'(b) RMS error vs $C$', fontsize=12)
    ax.legend(fontsize=10)

    # (c) 2D scan
    ax = axes[1, 0]
    C1_grid, C2_grid = np.meshgrid(C1_vals, C2_vals, indexing='ij')
    pcm = ax.pcolormesh(np.log10(C1_vals), np.log10(C2_vals),
                        np.log10(rms_2d + 1e-16).T,
                        cmap='viridis', shading='auto')
    # C1*C2 = 1 line
    log_c1 = np.linspace(np.log10(C1_vals[0]), np.log10(C1_vals[-1]), 100)
    ax.plot(log_c1, -log_c1, 'r--', lw=2, label=r'$C_1 C_2 = 1$')
    ax.set_xlabel(r'$\log_{10} C_1$', fontsize=12)
    ax.set_ylabel(r'$\log_{10} C_2$', fontsize=12)
    ax.set_title(r'(c) $\log_{10}$ RMS in $(C_1, C_2)$ plane', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    fig.colorbar(pcm, ax=ax, label=r'$\log_{10}$ RMS')

    # (d) mu(x) for several C values
    ax = axes[1, 1]
    xi_plot = np.geomspace(xi_min, xi_max, 2000)
    for C_test, color, ls in [(0.3, 'green', '--'), (0.5, 'orange', '--'),
                               (1.0, 'red', '-'), (2.0, 'purple', '--'),
                               (5.0, 'brown', '--')]:
        _, mu_c, x_c = solve_two_channel(xi_plot, C_test)
        ax.semilogx(x_c, mu_c, color=color, ls=ls, lw=1.5,
                     label=f'$C = {C_test}$')
    ax.semilogx(x_fine, mu_target(x_fine), 'b:', lw=2, label=r'$x/(1+x)$')
    ax.set_xlabel(r'$x = g/a_0$', fontsize=12)
    ax.set_ylabel(r'$\mu(x)$', fontsize=12)
    ax.set_title(r'(d) $\mu(x)$ for various $C = C_1 C_2$', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 1.1)

    fig.suptitle('Phase 2: Transport Scan (Strategy B)', fontsize=14, y=1.01)
    fig.tight_layout()
    fname = outdir / 'step2_transport_scan.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")
    return fname


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    # Step 2.1
    xi, g, mu, x = step_2_1()

    # Step 2.2
    C_vals, rms_vals, C1_vals, C2_vals, rms_2d = step_2_2()

    # Step 2.3
    step_2_3(C_vals, rms_vals)

    # Plots
    make_plots(xi, mu, x, C_vals, rms_vals, C1_vals, C2_vals, rms_2d)
