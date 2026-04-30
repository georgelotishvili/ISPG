"""
Phase AK: boundary-selected radial nonlocal kernel
=================================================

Goal:
  Construct the simplest radial nonlocal completion compatible with the
  Gauss-law deep-halo tail g_h ~ sqrt(m_enc)/xi, without feeding the MOND
  algebraic root point-by-point.

Model:
  - Newtonian baseline:             d^2 u_N / ds^2 = -xi^2 f_bary
  - Extra nonlocal source (radial): f_h,base(xi) = sqrt(m_enc(xi_eff)) / xi_eff
    with xi_eff^2 = xi^2 + rho_core^2
  - Normalization kappa_G fixed from the outer asymptotic condition
        g_h -> sqrt(m_enc)/xi
    on a resolved deep-halo annulus.

This is the radial analogue of the boundary-selected 2D Green-kernel
completion, adapted to the 1D log-radial Poisson operator.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jn_zeros

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import N_cheb
from chebyshev import cheb_matrices
from newtonian import solve_newtonian
from source import m_enc, g_newton_dimless
from phase_ag_loaded_branch_source_closure import solve_loaded_branch_self_consistently
from constants import a0, r_M


SEP = "=" * 78
RHO_CORE = 1.0 / jn_zeros(0, 1)[0]


def xi_eff(xi, rho_core=RHO_CORE):
    xi = np.asarray(xi, dtype=float)
    return np.sqrt(xi**2 + rho_core**2)


def base_source_nonlocal(xi, rho_core=RHO_CORE):
    xeff = xi_eff(xi, rho_core=rho_core)
    return np.sqrt(np.maximum(m_enc(xeff), 1e-30)) / np.maximum(xeff, 1e-30)


def cumulative_enclosed_source(xi, source):
    """Return m_h(xi) = int_0^xi xi' source(xi') dxi'."""
    xi = np.asarray(xi, dtype=float)
    source = np.asarray(source, dtype=float)

    x = xi[::-1]  # ascending
    f = source[::-1]
    integ = x * f

    m = np.zeros_like(x)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        m[i] = m[i - 1] + 0.5 * (integ[i] + integ[i - 1]) * dx
    return m[::-1]


def extract_g_dimless(u, D1, xi):
    du_ds = D1 @ u
    return -du_ds / np.maximum(xi**2, 1e-30)


def solve_radial_source_bvp(xi, D1, D2, source):
    rhs = -(xi**2) * source

    A = D2.copy()
    b = rhs.copy()

    # Outer boundary: u(xi_max)=0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0

    # Inner boundary: du/ds = -m_h(xi_min)
    m_h = cumulative_enclosed_source(xi, source)
    A[-1, :] = D1[-1, :]
    b[-1] = -m_h[-1]

    u = np.linalg.solve(A, b)
    return u, m_h


def resolve_fit_window(xi, fit_min=None, fit_max=None):
    xi = np.asarray(xi, dtype=float)
    xi_max = float(np.max(xi))
    if fit_max is None:
        fit_max = min(10.0, xi_max)
    if fit_min is None:
        fit_min = max(5.0, 0.5 * fit_max)
    mask = (xi >= fit_min) & (xi <= fit_max)
    if np.count_nonzero(mask) < 3:
        fit_min = max(3.0, 0.4 * fit_max)
        mask = (xi >= fit_min) & (xi <= fit_max)
    return fit_min, fit_max, mask


def boundary_selected_kappa(xi, g_h_base, fit_min=None, fit_max=None):
    desired = np.sqrt(np.maximum(m_enc(xi), 1e-30)) / np.maximum(xi, 1e-30)
    fit_min, fit_max, mask = resolve_fit_window(xi, fit_min=fit_min, fit_max=fit_max)
    num = np.dot(desired[mask], g_h_base[mask])
    den = np.dot(g_h_base[mask], g_h_base[mask])
    kappa = num / max(den, 1e-30)
    return kappa, desired, fit_min, fit_max


def solve_boundary_selected_radial_kernel(N=None, fit_min=None, fit_max=None):
    if N is None:
        N = N_cheb

    s, xi, u_N, D1 = solve_newtonian(N)
    _, D1_full, D2 = cheb_matrices(N)
    g_N = g_newton_dimless(xi)

    source_base = base_source_nonlocal(xi)
    u_base, m_base = solve_radial_source_bvp(xi, D1_full, D2, source_base)
    g_h_base = np.maximum(extract_g_dimless(u_base, D1_full, xi), 0.0)

    kappa_G, g_outer_target, fit_min, fit_max = boundary_selected_kappa(
        xi, g_h_base, fit_min=fit_min, fit_max=fit_max
    )

    source_extra = kappa_G * source_base
    u_h, m_h = solve_radial_source_bvp(xi, D1_full, D2, source_extra)
    g_h = np.maximum(extract_g_dimless(u_h, D1_full, xi), 0.0)
    g_eff = g_N + g_h
    mu_bvp = g_N / np.maximum(g_eff, 1e-30)

    # Diagnostics against loaded-branch and algebraic MOND
    g_loaded_si, A_loaded, fp_diag = solve_loaded_branch_self_consistently(
        a0 * g_N, xi * r_M
    )
    g_loaded = g_loaded_si / a0
    mu_loaded = g_N / np.maximum(g_loaded, 1e-30)
    mu_mond = g_eff / (1.0 + g_eff)

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)

    rel_loaded = np.abs(mu_bvp - mu_loaded)
    rel_mond = np.abs(mu_bvp - mu_mond)
    rel_outer = np.abs(g_h - g_outer_target) / np.maximum(g_outer_target, 1e-30)

    return {
        "xi": xi,
        "u_N": u_N,
        "u_h": u_h,
        "g_N": g_N,
        "g_h": g_h,
        "g_eff": g_eff,
        "mu_bvp": mu_bvp,
        "mu_loaded": mu_loaded,
        "mu_mond": mu_mond,
        "A_loaded": A_loaded,
        "fp_diag": fp_diag,
        "source_base": source_base,
        "source_extra": source_extra,
        "m_h": m_h,
        "rho_core": RHO_CORE,
        "kappa_G": kappa_G,
        "fit_min": fit_min,
        "fit_max": fit_max,
        "g_outer_target": g_outer_target,
        "rms_loaded": np.sqrt(np.mean(rel_loaded[mask] ** 2)),
        "rms_mond": np.sqrt(np.mean(rel_mond[mask] ** 2)),
        "trans_rms_loaded": np.sqrt(np.mean(rel_loaded[mask_trans] ** 2)),
        "outer_rms_loaded": np.sqrt(np.mean(rel_loaded[mask_outer] ** 2)),
        "full_rms_outer_target": np.sqrt(np.mean(rel_outer[mask] ** 2)),
        "outer_match_rms": np.sqrt(np.mean(rel_outer[mask_outer] ** 2)),
    }


def make_plot(res, outdir=None):
    if outdir is None:
        outdir = Path(__file__).parent / "plots"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    xi = res["xi"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.loglog(xi, res["source_extra"], lw=2, label="extra source")
    ax.loglog(xi, res["source_base"], "--", lw=1.5, label="base source")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("source")
    ax.set_title("Radial nonlocal source")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.loglog(xi, res["g_N"], lw=2, label=r"$g_N/a_0$")
    ax.loglog(xi, res["g_h"], lw=2, label=r"$g_h/a_0$")
    ax.loglog(xi, res["g_eff"], lw=2, label=r"$g_{\rm eff}/a_0$")
    ax.loglog(xi, res["g_outer_target"], ":", lw=2, label=r"$\sqrt{m_{\rm enc}}/\xi$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("dimensionless acceleration")
    ax.set_title("Boundary-selected radial kernel")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.semilogx(xi, res["mu_bvp"], lw=2, label=r"$\mu_{\rm radial\ kernel}$")
    ax.semilogx(xi, res["mu_loaded"], "--", lw=2, label=r"$\mu_{\rm loaded}$")
    ax.semilogx(xi, res["mu_mond"], ":", lw=2, label=r"$x/(1+x)$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Interpolating function comparison")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.semilogx(xi, res["mu_bvp"] - res["mu_loaded"], lw=2, label=r"$\mu-\mu_{\rm loaded}$")
    ax.semilogx(xi, res["mu_bvp"] - res["mu_mond"], lw=2, label=r"$\mu-x/(1+x)$")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("difference")
    ax.set_title("Mismatch diagnostics")
    ax.legend(fontsize=9)

    fig.tight_layout()
    outfile = outdir / "phase_ak_boundary_selected_radial_kernel.png"
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outfile


def run_all():
    print(SEP)
    print("  PHASE AK: Boundary-Selected Radial Nonlocal Kernel")
    print(SEP)

    res = solve_boundary_selected_radial_kernel()
    plotfile = make_plot(res)

    print(f"\n  rho_core = 1/j_01 = {res['rho_core']:.6f}")
    print(f"  kappa_G  = {res['kappa_G']:.6f}")
    print(f"  fit annulus = {res['fit_min']:.2f} - {res['fit_max']:.2f} r_M")
    print(f"  RMS(mu - mu_loaded) over 0.3<=xi<=10: {res['rms_loaded']:.3e}")
    print(f"  RMS(mu - x/(1+x)) over 0.3<=xi<=10:   {res['rms_mond']:.3e}")
    print(f"  Outer asymptotic RMS over 0.3<=xi<=10: {res['full_rms_outer_target']:.3e}")
    print(f"  Outer annulus RMS:                      {res['outer_match_rms']:.3e}")
    print(f"  Plot saved to: {plotfile}")

    print(
        "\n  "
        + f"{'xi':>6s}  {'g_h/a0':>10s}  {'g_out/a0':>10s}  {'mu':>10s}  "
        + f"{'mu_loaded':>10s}  {'x/(1+x)':>10s}"
    )
    print("  " + "-" * 72)
    for xi_s in [0.3, 1.0, 3.0, 10.0]:
        idx = int(np.argmin(np.abs(res["xi"] - xi_s)))
        print(
            f"  {res['xi'][idx]:6.2f}  "
            f"{res['g_h'][idx]:10.4f}  "
            f"{res['g_outer_target'][idx]:10.4f}  "
            f"{res['mu_bvp'][idx]:10.6f}  "
            f"{res['mu_loaded'][idx]:10.6f}  "
            f"{res['mu_mond'][idx]:10.6f}"
        )

    print(
        """

  Reading:
  - unlike the local occupied-branch BVP, the extra source is now cumulative:
        f_h,base ~ sqrt(m_enc(xi_eff)) / xi_eff ;
  - kappa_G is fixed only from the outer asymptotic tail, not from the full MOND
    profile;
  - improvement over the local BVP would be direct evidence that the missing
    physics is genuinely nonlocal.
        """
    )
    return res


if __name__ == "__main__":
    run_all()
