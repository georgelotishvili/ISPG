"""
Phase AJ: occupied-branch radial BVP
===================================

Goal:
  Build a genuinely radial 1D Poisson solve for the transported potential
  on the mature occupied branch, without feeding the quadratic MOND root
  directly point-by-point.

Model used:
  - Newtonian baseline:         -hat{nabla}^2 u_N = f
  - Transported source law:     -hat{nabla}^2 u_h = chi(xi) f
  - Occupied-branch coefficient chi = A_vort / x_total
    with  A_vort = omega / (omega + Omega_tr),  x_total = g_total / a0

This is the 1D Bessel-cell / slowly-varying-chi implementation of the
source-side closure written in ISPG_MOND.tex after the occupied-branch
beat-source reduction.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, r_M, kpc, N_cheb, Omega_tr_conj
from chebyshev import cheb_matrices
from newtonian import solve_newtonian
from source import f_source, g_newton_dimless, g_newton, m_enc
from phase_ag_loaded_branch_source_closure import solve_loaded_branch_self_consistently


SEP = "=" * 78


def omega_orb_from_g(g_dimless, xi):
    """Orbital ordering rate from the TOTAL local field."""
    g_si = a0 * np.maximum(np.asarray(g_dimless, dtype=float), 0.0)
    r_si = np.maximum(np.asarray(xi, dtype=float), 1e-30) * r_M
    return np.sqrt(np.maximum(g_si / r_si, 0.0))


def a_vort(omega):
    omega = np.asarray(omega, dtype=float)
    return omega / (omega + Omega_tr_conj)


def solve_transport_bvp(xi, D1, D2, chi):
    """Solve d^2 u_h / ds^2 = -xi^2 chi(xi) f(xi)."""
    rhs = -(xi ** 2) * chi * f_source(xi)

    A = D2.copy()
    b = rhs.copy()

    # Outer boundary: u_h(xi_max) = 0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    # Inner boundary: local source-ratio Neumann closure
    A[-1, :] = D1[-1, :]
    b[-1] = -chi[-1] * m_enc(xi[-1])

    return np.linalg.solve(A, b)


def extract_g_dimless(u, D1, xi):
    du_ds = D1 @ u
    return -du_ds / np.maximum(xi ** 2, 1e-30)


def iterate_occupied_branch(N=None, max_iter=200, tol=1e-10, relax=0.65):
    if N is None:
        N = N_cheb

    s, xi, u_N, D1 = solve_newtonian(N)
    _, D1_full, D2 = cheb_matrices(N)
    g_N_dimless = g_newton_dimless(xi)

    # Start from the Newtonian branch; no MOND root is fed.
    g_tot_dimless = np.maximum(g_N_dimless.copy(), 1e-12)
    u_h = np.zeros_like(u_N)
    rel_change = np.inf
    it_used = 0

    for it in range(max_iter):
        omega = omega_orb_from_g(g_tot_dimless, xi)
        A_branch = a_vort(omega)
        chi = A_branch / np.maximum(g_tot_dimless, 1e-30)

        u_h_new = solve_transport_bvp(xi, D1_full, D2, chi)
        g_h_new = np.maximum(extract_g_dimless(u_h_new, D1_full, xi), 0.0)
        g_new_dimless = g_N_dimless + g_h_new

        rel_change = np.max(
            np.abs(g_new_dimless - g_tot_dimless) / np.maximum(g_new_dimless, 1e-30)
        )

        g_tot_dimless = relax * g_new_dimless + (1.0 - relax) * g_tot_dimless
        u_h = relax * u_h_new + (1.0 - relax) * u_h
        it_used = it + 1

        if rel_change < tol:
            break

    omega = omega_orb_from_g(g_tot_dimless, xi)
    A_branch = a_vort(omega)
    chi = A_branch / np.maximum(g_tot_dimless, 1e-30)
    g_h_dimless = np.maximum(extract_g_dimless(u_h, D1_full, xi), 0.0)

    return {
        "s": s,
        "xi": xi,
        "u_N": u_N,
        "u_h": u_h,
        "g_N_dimless": g_N_dimless,
        "g_h_dimless": g_h_dimless,
        "g_tot_dimless": g_tot_dimless,
        "A_vort": A_branch,
        "chi": chi,
        "iterations": it_used,
        "fixed_point_residual": rel_change,
    }


def compare_to_targets(sol):
    xi = sol["xi"]
    g_N_dimless = sol["g_N_dimless"]
    g_tot_dimless = sol["g_tot_dimless"]

    mu_bvp = g_N_dimless / np.maximum(g_tot_dimless, 1e-30)
    mu_mond = g_tot_dimless / (1.0 + g_tot_dimless)

    g_loaded_si, A_loaded, fp_diag = solve_loaded_branch_self_consistently(
        g_newton(xi), xi * r_M
    )
    g_loaded_dimless = g_loaded_si / a0
    mu_loaded = g_N_dimless / np.maximum(g_loaded_dimless, 1e-30)

    interior = (xi >= 0.3) & (xi <= 10.0)
    rms_loaded = np.sqrt(np.mean((mu_bvp[interior] - mu_loaded[interior]) ** 2))
    rms_mond = np.sqrt(np.mean((mu_bvp[interior] - mu_mond[interior]) ** 2))

    return {
        "mu_bvp": mu_bvp,
        "mu_mond": mu_mond,
        "mu_loaded": mu_loaded,
        "g_loaded_dimless": g_loaded_dimless,
        "A_loaded": A_loaded,
        "loaded_diag": fp_diag,
        "rms_loaded": rms_loaded,
        "rms_mond": rms_mond,
    }


def make_plot(sol, cmp_result, outdir=None):
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    xi = sol["xi"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.loglog(xi, sol["g_N_dimless"], lw=2, label=r"$g_N/a_0$")
    ax.loglog(xi, sol["g_h_dimless"], lw=2, label=r"$g_h/a_0$")
    ax.loglog(xi, sol["g_tot_dimless"], lw=2, label=r"$g/a_0$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("dimensionless acceleration")
    ax.set_title("Radial occupied-branch BVP")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.semilogx(xi, sol["A_vort"], lw=2, label=r"$A_{\rm vort}$")
    ax.semilogx(xi, sol["chi"], lw=2, label=r"$\chi=A_{\rm vort}/x$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("branch coefficients")
    ax.set_title("Occupied-branch coefficients")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.semilogx(xi, cmp_result["mu_bvp"], lw=2, label=r"$\mu_{\rm BVP}$")
    ax.semilogx(xi, cmp_result["mu_loaded"], "--", lw=2, label=r"$\mu_{\rm loaded}$")
    ax.semilogx(xi, cmp_result["mu_mond"], ":", lw=2, label=r"$x/(1+x)$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Interpolating function comparison")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.semilogx(
        xi,
        cmp_result["mu_bvp"] - cmp_result["mu_loaded"],
        lw=2,
        label=r"$\mu_{\rm BVP}-\mu_{\rm loaded}$",
    )
    ax.semilogx(
        xi,
        cmp_result["mu_bvp"] - cmp_result["mu_mond"],
        lw=2,
        label=r"$\mu_{\rm BVP}-x/(1+x)$",
    )
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("difference")
    ax.set_title("BVP mismatch diagnostics")
    ax.legend(fontsize=9)

    fig.tight_layout()
    outfile = outdir / "phase_aj_occupied_branch_radial_bvp.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def run_all():
    print(SEP)
    print("  PHASE AJ: Occupied-Branch Radial BVP")
    print(SEP)

    sol = iterate_occupied_branch()
    cmp_result = compare_to_targets(sol)
    plotfile = make_plot(sol, cmp_result)

    xi = sol["xi"]
    sel = [0.3, 1.0, 3.0, 10.0]

    print(f"\n  Fixed-point iterations: {sol['iterations']}")
    print(f"  Fixed-point residual:   {sol['fixed_point_residual']:.3e}")
    print(f"  RMS(mu_BVP - mu_loaded) over 0.3<=xi<=10: {cmp_result['rms_loaded']:.3e}")
    print(f"  RMS(mu_BVP - x/(1+x)) over 0.3<=xi<=10:   {cmp_result['rms_mond']:.3e}")
    print(f"  Plot saved to: {plotfile}")

    print(
        "\n  "
        + f"{'xi':>6s}  {'g/a0':>10s}  {'A_vort':>10s}  {'chi':>10s}  "
        + f"{'mu_BVP':>10s}  {'mu_loaded':>10s}  {'x/(1+x)':>10s}"
    )
    print("  " + "-" * 78)
    for xi_s in sel:
        idx = np.argmin(np.abs(xi - xi_s))
        print(
            f"  {xi[idx]:6.2f}  "
            f"{sol['g_tot_dimless'][idx]:10.4f}  "
            f"{sol['A_vort'][idx]:10.6f}  "
            f"{sol['chi'][idx]:10.6f}  "
            f"{cmp_result['mu_bvp'][idx]:10.6f}  "
            f"{cmp_result['mu_loaded'][idx]:10.6f}  "
            f"{cmp_result['mu_mond'][idx]:10.6f}"
        )

    print(
        """

  Reading:
  - this solve does NOT feed the quadratic MOND root point-by-point;
  - it iterates a radial Poisson BVP for u_h with the occupied-branch source law
        chi = A_vort / x_total ;
  - comparison to the loaded-branch algebraic fixed point tests whether the
    radial BVP realizes the same mature-branch closure at the profile level.
  - profile-realization residual RMS is 0.139 vs analytic x/(1+x);
    fixed-point residual is 4.2e-08.  This remains an open profile residual.
        """
    )
    return sol, cmp_result


if __name__ == "__main__":
    run_all()
