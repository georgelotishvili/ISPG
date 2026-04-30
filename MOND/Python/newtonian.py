"""
Newtonian baseline: solve the Poisson equation on the equatorial plane
using Chebyshev spectral collocation in s = ln(xi).

Equation:   -hat{nabla}^2 u_N = f(xi)
            => d^2 u_N / ds^2 = -xi^2 f(xi)

BCs:  u_N(xi_max) = 0   (Dirichlet, reference)
      du_N/ds(xi_min) = -m_enc(xi_min)   (Neumann, regularity)

Analytic check:  du_N/ds = -m_enc(xi)  everywhere
                 g_N = -(a0/xi^2) du_N/ds = a0 m_enc/xi^2

This is the mu = 1 (purely Newtonian) baseline.

Reference: ISPG_MOND.tex Sec. 11; plan Step 1.4.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from constants import a0, r_M, kpc, xi_min, xi_max, N_cheb
from chebyshev import cheb_matrices, xi_from_s
from source import f_source, m_enc, g_newton_dimless, eta


def solve_newtonian(N=None):
    """Solve the equatorial-plane Poisson equation for u_N.

    Returns
    -------
    s    : (N+1,) collocation points
    xi   : (N+1,) dimensionless radii
    u_N  : (N+1,) dimensionless Newtonian potential
    D1   : (N+1,N+1) first-derivative matrix d/ds
    """
    if N is None:
        N = N_cheb

    s, D1, D2 = cheb_matrices(N)
    xi = xi_from_s(s)

    # RHS of d^2 u / ds^2 = -xi^2 f(xi)
    rhs = -xi**2 * f_source(xi)

    # Assemble system: A @ u = b
    A = D2.copy()
    b = rhs.copy()

    # BC at outer boundary s[0] = s_max:  u = 0  (Dirichlet)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    # BC at inner boundary s[N] = s_min:  du/ds = -m_enc(xi_min)  (Neumann)
    A[-1, :] = D1[-1, :]
    b[-1] = -m_enc(xi[-1])

    # Solve
    u_N = np.linalg.solve(A, b)

    return s, xi, u_N, D1


def extract_g_N(s, xi, u_N, D1):
    """Extract Newtonian acceleration from the numerical potential.

    g_N_num = -(a0/xi^2) du_N/ds
    """
    du_ds = D1 @ u_N
    g_N_num_dimless = -du_ds / xi**2 * xi**2 / xi**2  # = -du_ds / xi^2 ... wait

    # g_N = -(a0 / xi) du/d(xi) = -(a0 / xi^2) du/ds
    # In dimensionless form: g_N / a0 = -du_ds / xi^2
    # But du_ds = -m_enc(xi), so g_N/a0 = m_enc/xi^2.
    # Numerically: g_N_dimless = -(D1 @ u_N) / xi^2
    # But this uses the whole formula. Let me be careful:
    # du/ds is computed at collocation points.
    # g_N / a0 = -(du/ds) / xi^2
    # Wait no. The observable from eq:observables is:
    # g_eff = -(a0/xi) du/d(xi) = -(a0/xi)(1/xi)(du/ds) = -(a0/xi^2)(du/ds)
    # So g_eff/a0 = -(du/ds)/xi^2.

    du_ds = D1 @ u_N
    g_N_num_dimless = -du_ds / xi**2
    return g_N_num_dimless


def run_diagnostics(N=None):
    """Solve Newtonian Poisson and compare with analytic result."""
    if N is None:
        N = N_cheb

    s, xi, u_N, D1 = solve_newtonian(N)

    sep = "=" * 65
    print(sep)
    print("  Step 1.4 -- Newtonian Baseline")
    print(sep)

    # --- Numerical g_N vs analytic ---
    g_num = extract_g_N(s, xi, u_N, D1)
    g_ana = g_newton_dimless(xi)

    # Interior points (exclude boundary nodes where BCs are imposed)
    interior = slice(2, -2)

    rel_err = np.abs(g_num[interior] - g_ana[interior]) / g_ana[interior]
    max_err = np.max(rel_err)
    rms_err = np.sqrt(np.mean(rel_err**2))
    median_err = np.median(rel_err)

    print(f"\n  N = {N} Chebyshev points")
    print(f"  Domain: xi in [{xi[-1]:.0e}, {xi[0]:.0e}]")

    print(f"\n  --- g_N comparison (interior points) ---")
    print(f"  Max relative error:    {max_err:.2e}")
    print(f"  RMS relative error:    {rms_err:.2e}")
    print(f"  Median relative error: {median_err:.2e}")

    # du/ds comparison
    du_ds_num = D1 @ u_N
    du_ds_ana = -m_enc(xi)
    du_err = np.abs(du_ds_num[interior] - du_ds_ana[interior])
    du_rel = du_err / np.abs(du_ds_ana[interior])
    print(f"\n  --- du/ds comparison ---")
    print(f"  Max |du/ds_num - (-m_enc)| / |m_enc|: {np.max(du_rel):.2e}")

    # Table at selected points
    print(f"\n  {'xi':>8s}  {'g_N/a0 (num)':>14s}  {'g_N/a0 (ana)':>14s}  {'rel err':>10s}")
    print("  " + "-" * 52)
    sel = np.geomspace(xi_min * 2, xi_max / 2, 12)
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        print(f"  {xi[idx]:8.4f}  {g_num[idx]:14.6e}  {g_ana[idx]:14.6e}  "
              f"{abs(g_num[idx]-g_ana[idx])/g_ana[idx]:10.2e}")

    # Convergence study
    print(f"\n  --- Convergence study ---")
    print(f"  {'N':>6s}  {'max rel err':>12s}  {'RMS rel err':>12s}")
    for Nt in [40, 80, 120, 160, 200]:
        st, xit, ut, D1t = solve_newtonian(Nt)
        gt_num = extract_g_N(st, xit, ut, D1t)
        gt_ana = g_newton_dimless(xit)
        intt = slice(2, -2)
        re = np.abs(gt_num[intt] - gt_ana[intt]) / gt_ana[intt]
        print(f"  {Nt:6d}  {np.max(re):12.2e}  {np.sqrt(np.mean(re**2)):12.2e}")

    # mu = 1 check (pure Newtonian → mu should be 1 everywhere)
    x_param = g_num   # x = g/a0 in dimensionless terms (here g = g_N)
    mu_baseline = g_ana / g_num  # mu = g_N / g_eff; for Newtonian, g_eff = g_N → mu = 1
    mu_err = np.max(np.abs(mu_baseline[interior] - 1.0))
    print(f"\n  mu = g_N/g_eff for Newtonian baseline:")
    print(f"  Max |mu - 1|: {mu_err:.2e}  (should be ~0)")

    passed = max_err < 1e-4
    print(f"\n  --- RESULT: {'PASSED' if passed else 'FAILED'} ---")
    print(f"  (threshold: max relative error < 1e-4)")
    print(sep)

    return s, xi, u_N, D1, g_num, g_ana


def make_plots(s, xi, u_N, D1, g_num, g_ana, outdir=None):
    """Generate diagnostic plots for the Newtonian baseline."""
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Potential u_N(xi)
    ax = axes[0, 0]
    ax.semilogx(xi, u_N, 'b-', lw=2)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$u_N(\xi)$', fontsize=12)
    ax.set_title(r'(a) Newtonian potential $u_N$', fontsize=12)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)

    # (b) g_N comparison
    ax = axes[0, 1]
    ax.loglog(xi, g_num, 'ro', ms=2, label='Numerical (Chebyshev)')
    ax.loglog(xi, g_ana, 'b-', lw=1.5, label='Analytic')
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$g_N / a_0$', fontsize=12)
    ax.set_title(r'(b) Newtonian acceleration', fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)

    # (c) Relative error
    ax = axes[1, 0]
    interior = slice(2, -2)
    rel_err = np.abs(g_num - g_ana) / g_ana
    ax.semilogy(xi[interior], rel_err[interior], 'r-', lw=1.5)
    ax.axhline(1e-4, color='k', ls='--', lw=1, label=r'$10^{-4}$ target')
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$|g_{N,\rm num} - g_{N,\rm ana}| / g_{N,\rm ana}$', fontsize=12)
    ax.set_title(r'(c) Relative error in $g_N$', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)

    # (d) du/ds comparison
    ax = axes[1, 1]
    du_ds_num = D1 @ u_N
    du_ds_ana = -m_enc(xi)
    ax.semilogx(xi, du_ds_num, 'ro', ms=2, label=r'$D_1 u_N$ (numerical)')
    ax.semilogx(xi, du_ds_ana, 'b-', lw=1.5, label=r'$-m_{\rm enc}(\xi)$ (analytic)')
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$du_N/ds$', fontsize=12)
    ax.set_title(r'(d) Derivative $du_N/ds$ vs $-m_{\rm enc}$', fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)

    fig.suptitle('Step 1.4: Newtonian Baseline (Poisson BVP)', fontsize=14, y=1.01)
    fig.tight_layout()
    fname = outdir / 'step1_4_newtonian_baseline.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")
    return fname


if __name__ == "__main__":
    results = run_diagnostics()
    make_plots(*results)
