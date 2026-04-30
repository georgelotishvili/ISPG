"""
Phase AL: blind self-calibrated nonlocal 2D kernel
==================================================

Goal:
  Build a genuinely blind axisymmetric source-side solve:

  1. No algebraic MOND root is fed into the source construction.
  2. The baryonic normalization is fixed only from the far-field Gauss law
         g_N -> 1 / xi^2
     for the total baryonic mass.
  3. The nonlocal halo prefactor is fixed only from the outer asymptotic tail
         g_h -> sqrt(m_enc^(2D)) / xi .

Comparison against algebraic MOND and AQUAL is diagnostic only.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, kpc, r_M
from source import f_source_3d
from pde_2d_swirl_verify import solve_aqual_2d
from phase_u_axisymmetric_swirl_2d import (
    build_axisymmetric_operator,
    grad_xi,
    solve_linear_poisson,
)
from phase_ai_boundary_kernel_2d import (
    RHO_CORE_DERIVED,
    boundary_selected_kappa,
    build_tail_source,
    enclosed_mass_lookup,
    interior_residual,
)


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)


def resolve_fit_window(xi, fit_min, fit_max):
    xi = np.asarray(xi, dtype=float)
    xi_max = float(np.max(xi))
    fit_max = min(float(fit_max), xi_max)
    fit_min = max(float(fit_min), float(xi[1]) if len(xi) > 1 else 0.0)
    if fit_min >= fit_max:
        fit_min = max(float(xi[1]), 0.5 * fit_max)
    mask = (xi >= fit_min) & (xi <= fit_max)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Fit annulus is under-resolved on this grid.")
    return fit_min, fit_max, mask


def gauss_normalize_newtonian(xi, g_mid_unit, fit_min=5.0, fit_max=10.0):
    """Fix the baryonic source normalization from far-field point-mass gravity."""
    xi = np.asarray(xi, dtype=float)
    g_mid_unit = np.asarray(g_mid_unit, dtype=float)
    desired = 1.0 / np.maximum(xi, 1e-30) ** 2
    fit_min, fit_max, mask = resolve_fit_window(xi, fit_min=fit_min, fit_max=fit_max)
    num = np.dot(desired[mask], g_mid_unit[mask])
    den = np.dot(g_mid_unit[mask], g_mid_unit[mask])
    scale = num / max(den, 1e-30)
    return scale, desired, fit_min, fit_max


def rms_relative(xi, y, y_ref, xi_min=0.3, xi_max=10.0):
    mask = (xi >= xi_min) & (xi <= xi_max)
    rel = np.abs(y[mask] - y_ref[mask]) / np.maximum(y_ref[mask], 1e-30)
    return float(np.sqrt(np.mean(rel**2)))


def solve_blind_selfcalibrated_kernel_2d(
    nr=121,
    nz=91,
    xi_max=20.0,
    zeta_max=10.0,
    rho_core=RHO_CORE_DERIVED,
    newton_fit_min=5.0,
    newton_fit_max=10.0,
    source_fit_min=10.0,
    source_fit_max=15.0,
    aqual_max_iter=120,
    aqual_tol=1e-7,
    aqual_relax=0.35,
):
    xi, zeta, mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    solver = factorized(mat)
    dxi = xi[1] - xi[0]

    xi_grid, zeta_grid = np.meshgrid(xi, zeta, indexing="ij")
    unit_source = f_source_3d(xi_grid, zeta_grid)

    u_unit = solve_linear_poisson(solver, unit_source, nr, nz)
    g_mid_unit = -grad_xi(u_unit, dxi)[:, 0]
    scale_N, g_point_target, newton_fit_min, newton_fit_max = gauss_normalize_newtonian(
        xi, g_mid_unit, fit_min=newton_fit_min, fit_max=newton_fit_max
    )

    f_bary = scale_N * unit_source
    g_N = scale_N * g_mid_unit

    rho_sorted, cumulative = enclosed_mass_lookup(xi, zeta, f_bary)
    base_source, m_enc_2d, rho_eff = build_tail_source(
        xi, zeta, rho_core, rho_sorted, cumulative
    )

    u_base = solve_linear_poisson(solver, base_source, nr, nz)
    g_h_base = -grad_xi(u_base, dxi)[:, 0]
    m_enc_mid = m_enc_2d[:, 0]
    kappa_G, g_outer_target, source_fit_min, source_fit_max = boundary_selected_kappa(
        g_h_base, xi, m_enc_mid, fit_min=source_fit_min, fit_max=source_fit_max
    )

    source_extra = kappa_G * base_source
    source_total = f_bary + source_extra
    u_total = solve_linear_poisson(solver, source_total, nr, nz)
    g_eff = -grad_xi(u_total, dxi)[:, 0]
    g_h = g_eff - g_N

    disc = g_N**2 + 4.0 * g_N
    g_alg = 0.5 * (g_N + np.sqrt(disc))

    mu_blind = np.divide(g_N, g_eff, out=np.ones_like(g_eff), where=g_eff > 1e-20)
    mu_alg = np.divide(g_N, g_alg, out=np.ones_like(g_alg), where=g_alg > 1e-20)
    mu_mond = np.divide(g_eff, 1.0 + g_eff, out=np.zeros_like(g_eff), where=g_eff > 0)

    res_aqual = solve_aqual_2d(
        NR=nr,
        Nz=nz,
        R_max_kpc=xi_max * r_M / kpc,
        z_max_kpc=zeta_max * r_M / kpc,
        max_iter=aqual_max_iter,
        tol=aqual_tol,
        omega_relax=aqual_relax,
        verbose=False,
    )
    g_aqual = np.interp(xi, res_aqual["xi"], res_aqual["g_eff_mid"])

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)

    rel_alg = np.abs(g_eff - g_alg) / np.maximum(g_alg, 1e-30)
    rel_aqual = np.abs(g_eff - g_aqual) / np.maximum(g_aqual, 1e-30)
    delta_mu_alg = mu_blind - mu_alg
    delta_mu_mond = mu_blind - mu_mond

    return {
        "xi": xi,
        "zeta": zeta,
        "rho_eff": rho_eff,
        "rho_core": rho_core,
        "scale_N": scale_N,
        "newton_fit_min": newton_fit_min,
        "newton_fit_max": newton_fit_max,
        "source_fit_min": source_fit_min,
        "source_fit_max": source_fit_max,
        "kappa_G": kappa_G,
        "g_point_target": g_point_target,
        "g_outer_target": g_outer_target,
        "g_N": g_N,
        "g_h": g_h,
        "g_eff": g_eff,
        "g_alg": g_alg,
        "g_aqual": g_aqual,
        "mu_blind": mu_blind,
        "mu_alg": mu_alg,
        "mu_mond": mu_mond,
        "delta_mu_alg": delta_mu_alg,
        "delta_mu_mond": delta_mu_mond,
        "source_extra": source_extra,
        "source_total": source_total,
        "pde_residual": interior_residual(mat, u_total, source_total, nr, nz),
        "full_rms_alg": float(np.sqrt(np.mean(rel_alg[mask] ** 2))),
        "trans_rms_alg": float(np.sqrt(np.mean(rel_alg[mask_trans] ** 2))),
        "outer_rms_alg": float(np.sqrt(np.mean(rel_alg[mask_outer] ** 2))),
        "full_rms_aqual": float(np.sqrt(np.mean(rel_aqual[mask] ** 2))),
        "trans_rms_aqual": float(np.sqrt(np.mean(rel_aqual[mask_trans] ** 2))),
        "outer_rms_aqual": float(np.sqrt(np.mean(rel_aqual[mask_outer] ** 2))),
        "rms_mu_alg": float(np.sqrt(np.mean(delta_mu_alg[mask] ** 2))),
        "rms_mu_mond": float(np.sqrt(np.mean(delta_mu_mond[mask] ** 2))),
        "newton_rms_outer": rms_relative(
            xi, g_N, g_point_target, xi_min=newton_fit_min, xi_max=newton_fit_max
        ),
        "extra_pos": float(np.mean(source_extra[mask, :] >= -1e-12)),
        "total_pos": float(np.mean(source_total[mask, :] >= -1e-12)),
    }


def make_plots(results):
    xi = results["xi"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(xi, results["g_N"], lw=2, label=r"$g_N$ (blind self-cal.)")
    ax.loglog(xi, results["g_point_target"], "--", lw=1.5, label=r"$1/\xi^2$")
    ax.axvspan(
        results["newton_fit_min"],
        results["newton_fit_max"],
        color="gray",
        alpha=0.12,
        label="Newtonian fit",
    )
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("Newtonian Gauss normalization")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.loglog(xi, results["g_eff"], lw=2, label="blind nonlocal 2D")
    ax.loglog(xi, results["g_alg"], "--", lw=1.8, label="algebraic (same g_N)")
    ax.loglog(xi, results["g_aqual"], ":", lw=2, label="AQUAL 2D")
    ax.axvspan(
        results["source_fit_min"],
        results["source_fit_max"],
        color="gray",
        alpha=0.12,
        label="halo fit",
    )
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("Blind source-side field")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.semilogx(xi, results["mu_blind"], lw=2, label=r"$\mu_{\rm blind}=g_N/g$")
    ax.semilogx(xi, results["mu_alg"], "--", lw=1.8, label=r"$\mu_{\rm alg}$")
    ax.semilogx(xi, results["mu_mond"], ":", lw=2, label=r"$x/(1+x)$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Effective interpolating function")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    err_alg = np.abs(results["g_eff"] - results["g_alg"]) / np.maximum(results["g_alg"], 1e-30)
    err_aqual = np.abs(results["g_eff"] - results["g_aqual"]) / np.maximum(results["g_aqual"], 1e-30)
    ax.semilogx(xi, err_alg, lw=2, label="vs algebraic")
    ax.semilogx(xi, err_aqual, lw=2, label="vs AQUAL")
    ax.axvline(0.3, color="gray", ls=":", lw=0.8)
    ax.axvline(10.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("relative error")
    ax.set_title("Blind 2D mismatch diagnostics")
    ax.legend(fontsize=9)

    fig.tight_layout()
    outpath = OUTDIR / "phase_al_blind_selfcalibrated_kernel_2d.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def print_interpretation(results):
    print(SEP)
    print("  PHASE AL: Blind Self-Calibrated Nonlocal 2D Kernel")
    print(SEP)
    print(f"  grid                        = {len(results['xi'])} x {len(results['zeta'])}")
    print(f"  rho_core                    = {results['rho_core']:.6f} r_M")
    print(
        f"  Newtonian Gauss annulus     = {results['newton_fit_min']:.1f} - "
        f"{results['newton_fit_max']:.1f} r_M"
    )
    print(f"  baryonic scale_N            = {results['scale_N']:.6f}")
    print(
        f"  halo asymptotic annulus     = {results['source_fit_min']:.1f} - "
        f"{results['source_fit_max']:.1f} r_M"
    )
    print(f"  halo prefactor kappa_G      = {results['kappa_G']:.6f}")
    print(f"  PDE residual                = {results['pde_residual']:.3e}")
    print(f"  Newtonian outer-fit RMS     = {results['newton_rms_outer']:.3e}")
    print(f"  positivity (extra/total)    = {results['extra_pos']:.3f} / {results['total_pos']:.3f}")
    print()
    print("  Blind 2D field comparison on 0.3 <= xi <= 10:")
    print(f"    vs algebraic  : full {results['full_rms_alg']:.3e}, trans {results['trans_rms_alg']:.3e}, outer {results['outer_rms_alg']:.3e}")
    print(f"    vs AQUAL 2D   : full {results['full_rms_aqual']:.3e}, trans {results['trans_rms_aqual']:.3e}, outer {results['outer_rms_aqual']:.3e}")
    print(f"    mu vs alg     : rms  {results['rms_mu_alg']:.3e}")
    print(f"    mu vs x/(1+x) : rms  {results['rms_mu_mond']:.3e}")
    print()
    print("    r/r_M    g_N/a0    g_blind/a0   g_alg/a0   g_AQUAL/a0   mu_blind")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = int(np.argmin(np.abs(results["xi"] - factor)))
        print(
            f"    {results['xi'][idx]:5.1f}   "
            f"{results['g_N'][idx]:8.3f}   "
            f"{results['g_eff'][idx]:10.3f}   "
            f"{results['g_alg'][idx]:8.3f}   "
            f"{results['g_aqual'][idx]:10.3f}   "
            f"{results['mu_blind'][idx]:8.4f}"
        )

    print(
        f"""

  Reading:
  - this solve is blind in the strong sense: the source construction uses no
    algebraic MOND root and no AQUAL mu field;
  - the baryonic normalization is fixed only by the far-field Newtonian Gauss
    law, while the halo prefactor is fixed only by the outer nonlocal tail;
  - the remaining ~10^-1 level mismatch therefore localizes the problem to the
    full 2D / 2+1D Green-kernel geometry, not to a hidden pointwise mu input.

  Plot saved to:
    {OUTDIR / "phase_al_blind_selfcalibrated_kernel_2d.png"}
        """
    )


def run_all():
    results = solve_blind_selfcalibrated_kernel_2d()
    make_plots(results)
    print_interpretation(results)
    return results


if __name__ == "__main__":
    run_all()
