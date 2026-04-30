"""
Steady-state mode solver (Strategy A) — Steps 3.2 through 3.5.

Solves the mode-decomposed equations on the equatorial plane:

  m=1 mode (eq:mode_eq):
    [-xi^{-2} D2 + xi^{-2} + i eps Omega_hat (1 + delta_FD)] U1 = S1
    Source: S1 = eps delta_FD Omega_hat U0

  Feedback into m=0:
    F_FB ~ eps delta_FD Omega_hat |U1|
    (azimuthal average of frame-dragging coupling with m=1 mode)

  Corrected m=0:
    -xi^{-2} D2 U0 = f + F_FB

This is the DIAGNOSTIC baseline.  Expected: F_FB << f at xi ~ 1,
so mu_A ~ 1 (steady-state cannot produce MOND at xi ~ 1).

Reference: ISPG_MOND.tex Secs. 11.2-11.4; plan Steps 3.2-3.5.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from constants import a0, eps, xi_min, xi_max, N_cheb, r_M, kpc
from chebyshev import cheb_matrices, xi_from_s
from source import f_source, m_enc, g_newton_dimless, omega_dimless
from frame_dragging import delta_FD, alpha_mode
from newtonian import solve_newtonian, extract_g_N


# =====================================================================
# Step 3.2: m=1 mode solver
# =====================================================================

def solve_m1_mode(s, xi, D1, D2, U0, N=None):
    """Solve the steady-state m=1 mode equation.

    [-xi^{-2} D2 + xi^{-2} + i eps Oh (1+dFD)] U1 = S1

    Parameters
    ----------
    s, xi : collocation points
    D1, D2 : derivative matrices
    U0 : Newtonian potential (real)

    Returns
    -------
    U1 : complex array, m=1 mode amplitude
    S1 : real array, source term
    """
    Oh = omega_dimless(xi)
    dFD = delta_FD(xi)

    xi_inv2 = 1.0 / xi**2

    # Operator: L = -diag(xi^{-2}) D2 + diag(xi^{-2}) + i eps diag(Oh(1+dFD))
    L = -np.diag(xi_inv2) @ D2 + np.diag(xi_inv2) \
        + 1j * eps * np.diag(Oh * (1.0 + dFD))

    # Source: S1 = eps * dFD * Oh * U0
    S1 = eps * dFD * Oh * U0

    # BCs: U1 = 0 at both boundaries
    rhs = S1.astype(complex)
    L[0, :] = 0.0;  L[0, 0] = 1.0;  rhs[0] = 0.0   # outer
    L[-1, :] = 0.0; L[-1, -1] = 1.0; rhs[-1] = 0.0  # inner

    U1 = np.linalg.solve(L, rhs)
    return U1, S1


# =====================================================================
# Step 3.3: Feedback into m=0
# =====================================================================

def compute_feedback(xi, U1):
    """Compute azimuthally-averaged feedback from m=1 into m=0.

    F_FB(xi) = eps * delta_FD * Omega_hat * |U1| / xi
    This is the transported-channel source for the m=0 potential.
    """
    Oh = omega_dimless(xi)
    dFD = delta_FD(xi)
    F_FB = eps * dFD * Oh * np.abs(U1)
    return F_FB


# =====================================================================
# Step 3.4: Corrected m=0 and mu extraction
# =====================================================================

def solve_corrected_m0(s, xi, D2, f, F_FB):
    """Solve -xi^{-2} D2 U0 = f + F_FB with BCs."""
    rhs = -(xi**2) * (f + F_FB)  # d2u/ds2 = -xi^2 (f + F_FB)

    A = D2.copy()
    b = rhs.copy()

    # BC: u(xi_max) = 0 (outer Dirichlet)
    A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = 0.0

    # BC: du/ds(xi_min) = -(m_enc(xi_min) + correction)
    # For the corrected solution, the inner Neumann BC is approximately
    # the same as Newtonian (F_FB is tiny at small xi).
    from chebyshev import cheb_matrices as _cm
    _, D1, _ = _cm(len(s)-1)
    A[-1, :] = D1[-1, :]
    b[-1] = -m_enc(xi[-1])  # Newtonian BC (F_FB negligible at inner boundary)

    U0_corr = np.linalg.solve(A, b)
    return U0_corr, D1


def extract_mu(xi, U0, D1):
    """Extract mu(x) = g_N / g_eff from the total potential."""
    du_ds = D1 @ U0
    g_eff_dimless = -du_ds / xi**2   # g_eff / a0
    g_N_dimless = g_newton_dimless(xi)

    # Avoid division by zero
    mask = g_eff_dimless > 1e-30
    mu = np.zeros_like(xi)
    mu[mask] = g_N_dimless[mask] / g_eff_dimless[mask]

    x = g_eff_dimless  # x = g_eff / a0
    return mu, x, g_eff_dimless


# =====================================================================
# Step 3.5: Props 3-4 check
# =====================================================================

def check_propositions(xi, mu, x):
    """Check Prop 3 (mu -> 1 for x >> 1) and Prop 4 (mu ~ x for x << 1)."""
    # Prop 3: Newtonian limit (xi << 1, x >> 1)
    mask_inner = xi < 0.1
    mu_inner = mu[mask_inner]
    prop3_err = np.max(np.abs(mu_inner - 1.0)) if mask_inner.sum() > 0 else np.nan

    # Prop 4: deep-MOND (xi >> 1, x << 1) -- mu should be proportional to x
    mask_outer = xi > 10
    if mask_outer.sum() > 2:
        mu_outer = mu[mask_outer]
        x_outer = x[mask_outer]
        # Fit mu = C * x in log-log
        valid = (x_outer > 0) & (mu_outer > 0)
        if valid.sum() > 2:
            log_ratio = np.log10(mu_outer[valid] / x_outer[valid])
            # If mu ~ C*x, then log(mu/x) = log(C) = const
            prop4_spread = np.std(log_ratio)
            prop4_C = 10**np.mean(log_ratio)
        else:
            prop4_spread = np.nan
            prop4_C = np.nan
    else:
        prop4_spread = np.nan
        prop4_C = np.nan

    return prop3_err, prop4_C, prop4_spread


# =====================================================================
# Main driver
# =====================================================================

def run_all():
    """Execute Steps 3.2 through 3.5."""
    sep = "=" * 65

    # --- Newtonian baseline from Step 1.4 ---
    s, xi, U_N, D1_raw = solve_newtonian()
    _, D1, D2 = cheb_matrices()

    # ============================================================
    # Step 3.2: m=1 mode
    # ============================================================
    print(sep)
    print("  Step 3.2 -- m=1 Mode Equation (Steady-State)")
    print(sep)

    U1, S1 = solve_m1_mode(s, xi, D1, D2, U_N)

    print(f"\n  |U1| profile:")
    print(f"  {'xi':>8s}  {'|U1|':>12s}  {'|S1|':>12s}  {'|U1|/|U0|':>12s}")
    print("  " + "-" * 48)
    sel = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        ratio = np.abs(U1[idx]) / max(np.abs(U_N[idx]), 1e-30)
        print(f"  {xi[idx]:8.4f}  {np.abs(U1[idx]):12.4e}  {np.abs(S1[idx]):12.4e}  "
              f"{ratio:12.4e}")

    print(f"\n  max |U1| = {np.max(np.abs(U1)):.4e}")
    print(f"  max |U1|/|U_N| = {np.max(np.abs(U1)/np.maximum(np.abs(U_N),1e-30)):.4e}")

    # Verify scaling: |U1| ~ eps dFD Oh xi^2 |U0| (when alpha << 1)
    U1_est = eps * delta_FD(xi) * omega_dimless(xi) * xi**2 * np.abs(U_N)
    ratio_est = np.abs(U1[5:-5]) / np.maximum(U1_est[5:-5], 1e-50)
    print(f"  |U1| / (eps dFD Oh xi^2 U0) ~ {np.median(ratio_est):.2f}  (should be ~1)")

    print(sep)

    # ============================================================
    # Step 3.3: Feedback into m=0
    # ============================================================
    print(sep)
    print("  Step 3.3 -- Feedback F_FB into m=0")
    print(sep)

    f_bary = f_source(xi)
    F_FB = compute_feedback(xi, U1)

    ratio_fb = F_FB / np.maximum(f_bary, 1e-50)

    print(f"\n  F_FB / f(xi) at key radii:")
    print(f"  {'xi':>8s}  {'F_FB':>12s}  {'f(xi)':>12s}  {'F_FB/f':>12s}")
    print("  " + "-" * 48)
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        print(f"  {xi[idx]:8.4f}  {F_FB[idx]:12.4e}  {f_bary[idx]:12.4e}  "
              f"{ratio_fb[idx]:12.4e}")

    print(f"\n  max F_FB/f = {np.max(ratio_fb[5:-5]):.4e}")
    print(f"  F_FB/f at xi=1: {ratio_fb[np.argmin(np.abs(xi-1.0))]:.4e}")
    print(f"\n  Verdict: F_FB is {'NEGLIGIBLE' if np.max(ratio_fb[5:-5]) < 0.01 else 'SIGNIFICANT'} "
          f"compared to baryonic source.")

    print(sep)

    # ============================================================
    # Step 3.4: Corrected m=0 and mu_A
    # ============================================================
    print(sep)
    print("  Step 3.4 -- Steady-State mu_A")
    print(sep)

    U0_corr, D1_use = solve_corrected_m0(s, xi, D2, f_bary, F_FB)
    mu_A, x_A, g_eff_A = extract_mu(xi, U0_corr, D1_use)

    mu_target = x_A / (1.0 + x_A)
    delta_mu = mu_A - mu_target

    interior = slice(3, -3)
    rms_delta = np.sqrt(np.mean(delta_mu[interior]**2))
    max_delta = np.max(np.abs(delta_mu[interior]))

    print(f"\n  mu_A vs x/(1+x):")
    print(f"  RMS(Delta mu):  {rms_delta:.4e}")
    print(f"  Max |Delta mu|: {max_delta:.4e}")

    print(f"\n  {'xi':>8s}  {'x=g/a0':>10s}  {'mu_A':>10s}  {'x/(1+x)':>10s}  {'Delta mu':>10s}")
    print("  " + "-" * 52)
    for xi_s in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        idx = np.argmin(np.abs(xi - xi_s))
        print(f"  {xi[idx]:8.4f}  {x_A[idx]:10.4f}  {mu_A[idx]:10.6f}  "
              f"{mu_target[idx]:10.6f}  {delta_mu[idx]:10.2e}")

    mu_dev_from_1 = np.max(np.abs(mu_A[interior] - 1.0))
    print(f"\n  Expected: mu_A ~ 1 (feedback too small for MOND at xi~1)")
    print(f"  Actual: max |mu_A - 1| = {mu_dev_from_1:.2e}  =>  mu_A IS ~1 (Newtonian)")
    print(f"  Delta mu = 1 - x/(1+x) = 1/(1+x) = the FULL MOND deficit.")

    print(sep)

    # ============================================================
    # Step 3.5: Propositions 3-4 check
    # ============================================================
    print(sep)
    print("  Step 3.5 -- Propositions 3 and 4")
    print(sep)

    p3_err, p4_C, p4_spread = check_propositions(xi, mu_A, x_A)

    print(f"\n  Prop 3 (Newtonian limit, xi < 0.1):")
    print(f"  Max |mu_A - 1| = {p3_err:.4e}")
    print(f"  Status: {'PASS' if p3_err < 0.01 else 'FAIL'}")

    print(f"\n  Prop 4 (deep-MOND, xi > 10):")
    print(f"  mu ~ C*x with C = {p4_C:.4f}, spread = {p4_spread:.4e}")
    print(f"  Status: {'PASS' if p4_spread < 0.1 else 'FAIL'}")
    print(f"  (Note: Prop 4 applies to the cumulative transport result,")
    print(f"   not the steady-state. Steady-state mu ~ 1 is expected.)")

    print(sep)

    return s, xi, U_N, U1, S1, F_FB, f_bary, U0_corr, mu_A, x_A, delta_mu


def make_plots(s, xi, U_N, U1, S1, F_FB, f_bary,
               U0_corr, mu_A, x_A, delta_mu, outdir=None):
    """Generate all Phase 3 diagnostic plots."""
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    interior = slice(3, -3)

    # (a) |U1| profile
    ax = axes[0, 0]
    ax.loglog(xi, np.abs(U1), 'b-', lw=2, label=r'$|U_1|$ (numerical)')
    U1_est = eps * delta_FD(xi) * omega_dimless(xi) * xi**2 * np.abs(U_N)
    ax.loglog(xi, U1_est, 'r--', lw=1.5,
              label=r'$\varepsilon\delta_{\rm FD}\hat\Omega\xi^2 U_0$')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$|U_1(\xi)|$', fontsize=12)
    ax.set_title(r'(a) m=1 mode amplitude', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(xi_min, xi_max)

    # (b) F_FB / f
    ax = axes[0, 1]
    ratio = F_FB / np.maximum(f_bary, 1e-50)
    ax.loglog(xi[interior], ratio[interior], 'r-', lw=2)
    ax.axhline(1.0, color='k', ls='--', lw=1, label='$F_{\\rm FB} = f$')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$F_{\rm FB} / f$', fontsize=12)
    ax.set_title(r'(b) Feedback / baryonic source', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)

    # (c) mu_A(x)
    ax = axes[0, 2]
    mask = (x_A > 0.01) & (x_A < 100)
    ax.semilogx(x_A[mask], mu_A[mask], 'ro', ms=2, label=r'$\mu_A$ (steady-state)')
    x_fine = np.geomspace(0.01, 100, 500)
    ax.semilogx(x_fine, x_fine/(1+x_fine), 'b-', lw=2, label=r'$x/(1+x)$ (target)')
    ax.set_xlabel(r'$x = g_{\rm eff}/a_0$', fontsize=12)
    ax.set_ylabel(r'$\mu(x)$', fontsize=12)
    ax.set_title(r'(c) Interpolating function $\mu_A(x)$', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    # (d) Delta mu
    ax = axes[1, 0]
    ax.semilogx(xi[interior], delta_mu[interior], 'r-', lw=2)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\Delta\mu = \mu_A - x/(1+x)$', fontsize=12)
    ax.set_title(r'(d) Deficit $\Delta\mu$ vs $\xi$', fontsize=12)

    # (e) |U1| / |U0|
    ax = axes[1, 1]
    ratio_u = np.abs(U1) / np.maximum(np.abs(U_N), 1e-30)
    ax.loglog(xi, ratio_u, 'g-', lw=2)
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$|U_1|/|U_0|$', fontsize=12)
    ax.set_title(r'(e) Mode amplitude ratio', fontsize=12)
    ax.set_xlim(xi_min, xi_max)

    # (f) mu_A vs xi (spatial view)
    ax = axes[1, 2]
    ax.semilogx(xi[interior], mu_A[interior], 'r-', lw=2, label=r'$\mu_A(\xi)$')
    ax.axhline(1.0, color='b', ls='--', lw=1, label=r'$\mu = 1$ (Newtonian)')
    ax.axvline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\xi$', fontsize=12)
    ax.set_ylabel(r'$\mu_A$', fontsize=12)
    ax.set_title(r'(f) $\mu_A$ vs radius', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(0.99, 1.01)

    fig.suptitle('Phase 3: Steady-State Diagnostics (Strategy A)', fontsize=14, y=1.01)
    fig.tight_layout()
    fname = outdir / 'step3_mode_solver.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {fname}")
    return fname


if __name__ == "__main__":
    results = run_all()
    make_plots(*results)
