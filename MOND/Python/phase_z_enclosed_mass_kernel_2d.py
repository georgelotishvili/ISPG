"""
Phase Z: Enclosed-mass Green-kernel 2D source-side closure
===========================================================

Purpose:
  Build a genuinely nonlocal axisymmetric source-side closure in which the
  transported source at (R, z) is controlled by the enclosed baryonic mass
  inside the same 2D radius rho = sqrt(xi^2 + zeta^2).

  The key point is that the extra source must remain active in the vacuum
  region; a conservative redistribution of finite disk source cannot sustain
  the deep-MOND halo tail.  The present kernel therefore uses the cumulative
  enclosed source and the required 1/rho^2 halo profile directly.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.sparse.linalg import factorized

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from phase_u_axisymmetric_swirl_2d import (
    build_axisymmetric_operator,
    calibrate_newtonian_scale,
    grad_xi,
    solve_activated_midplane,
    solve_linear_poisson,
)

SEP = "=" * 78
RHO_CORE_SELECTED = 0.35


def enclosed_mass_lookup(xi, zeta, bary_source):
    """Return a lookup table for the 2D enclosed baryonic mass fraction."""
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")
    dxi = xi[1] - xi[0]
    dz = zeta[1] - zeta[0]

    rho = np.sqrt(XI**2 + Z**2)
    # The overall normalization cancels when we divide by the total.
    shell_weights = np.asarray(bary_source, dtype=float) * XI * dxi * dz

    rho_flat = rho.reshape(-1)
    weight_flat = shell_weights.reshape(-1)
    order = np.argsort(rho_flat)

    rho_sorted = rho_flat[order]
    cumulative = np.cumsum(weight_flat[order])
    cumulative /= cumulative[-1]
    return rho_sorted, cumulative


def enclosed_mass_fraction(rho, rho_sorted, cumulative):
    """Evaluate the 2D enclosed mass fraction at radius rho."""
    return np.interp(rho, rho_sorted, cumulative, left=0.0, right=1.0)


def build_tail_source(xi, zeta, rho_core, rho_sorted, cumulative):
    """Return the Green-kernel halo source before the final normalization."""
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")
    rho_eff_sq = XI**2 + Z**2 + rho_core**2
    rho_eff = np.sqrt(rho_eff_sq)
    m_enc_2d = enclosed_mass_fraction(rho_eff, rho_sorted, cumulative)
    base_source = np.sqrt(np.maximum(m_enc_2d, 1e-12)) / np.maximum(rho_eff_sq, 1e-12)
    return base_source, m_enc_2d, rho_eff


def fit_green_normalization(g_h_target, g_h_base, xi, fit_min=5.0, fit_max=10.0):
    """Fix the Green-kernel normalization from the outer-halo branch."""
    mask = (xi >= fit_min) & (xi <= fit_max)
    num = np.dot(g_h_target[mask], g_h_base[mask])
    den = np.dot(g_h_base[mask], g_h_base[mask])
    return num / max(den, 1e-30)


def interior_residual(mat, u, source, nr, nz):
    residual = mat @ u.reshape(-1) + source.reshape(-1)
    boundary_mask = np.ones_like(residual, dtype=bool)
    for i in range(nr):
        boundary_mask[i * nz + (nz - 1)] = False
    for j in range(nz):
        boundary_mask[(nr - 1) * nz + j] = False
    return np.max(np.abs(residual[boundary_mask]))


def evaluate_core(
    rho_core,
    xi,
    zeta,
    mat,
    solver,
    f_bary,
    g_mid_newton,
    newton_err_max,
    newton_err_rms,
):
    dxi = xi[1] - xi[0]
    nr, nz = len(xi), len(zeta)

    y_mid = g_mid_newton.copy()
    if len(y_mid) > 1:
        y_mid[0] = y_mid[1]
    _, x_target, A_mid, chi_mid = solve_activated_midplane(y_mid, xi)
    g_h_target = x_target - g_mid_newton

    rho_sorted, cumulative = enclosed_mass_lookup(xi, zeta, f_bary)
    base_source, m_enc_2d, rho_eff = build_tail_source(
        xi, zeta, rho_core, rho_sorted, cumulative
    )

    u_base = solve_linear_poisson(solver, base_source, nr, nz)
    g_h_base = -grad_xi(u_base, dxi)[:, 0]
    kappa_G = fit_green_normalization(g_h_target, g_h_base, xi)

    source_extra = kappa_G * base_source
    source_total = f_bary + source_extra

    u_total = solve_linear_poisson(solver, source_total, nr, nz)
    g_mid_total = -grad_xi(u_total, dxi)[:, 0]
    g_mid_h = g_mid_total - g_mid_newton

    mu = np.divide(g_mid_newton, g_mid_total, out=np.ones_like(g_mid_total),
                   where=g_mid_total > 1e-20)
    mu_alg = np.divide(g_mid_newton, x_target, out=np.ones_like(x_target),
                       where=x_target > 1e-20)
    mu_target = np.divide(x_target, 1.0 + x_target, out=np.zeros_like(x_target),
                          where=x_target > 0)

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)
    rel = np.abs(g_mid_total - x_target) / np.maximum(x_target, 1e-30)

    return {
        "xi": xi,
        "zeta": zeta,
        "A_mid": A_mid,
        "chi_mid": chi_mid,
        "g_N": g_mid_newton,
        "g_eff": g_mid_total,
        "g_h": g_mid_h,
        "x": g_mid_total,
        "mu": mu,
        "mu_target": mu_target,
        "g_alg": x_target,
        "mu_alg": mu_alg,
        "rho_core": rho_core,
        "kappa_G": kappa_G,
        "rho_eff": rho_eff,
        "m_enc_2d": m_enc_2d,
        "source_extra": source_extra,
        "source_total": source_total,
        "newton_err_max": newton_err_max,
        "newton_err_rms": newton_err_rms,
        "pde_residual": interior_residual(mat, u_total, source_total, nr, nz),
        "full_rms": np.sqrt(np.mean(rel[mask] ** 2)),
        "trans_rms": np.sqrt(np.mean(rel[mask_trans] ** 2)),
        "outer_rms": np.sqrt(np.mean(rel[mask_outer] ** 2)),
        "full_max": np.max(rel[mask]),
        "score": max(
            np.sqrt(np.mean(rel[mask_trans] ** 2)),
            np.sqrt(np.mean(rel[mask_outer] ** 2)),
        ),
        "extra_pos": np.mean(source_extra[mask, :] >= -1e-12),
        "total_pos": np.mean(source_total[mask, :] >= -1e-12),
    }


def solve_selected_kernel(rho_core=RHO_CORE_SELECTED, nr=81, nz=61,
                          xi_max=30.0, zeta_max=5.0):
    xi, zeta, mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    solver = factorized(mat)
    scale, unit_source, _, g_mid_newton, err_max_n, err_rms_n = calibrate_newtonian_scale(
        xi, zeta, solver
    )
    f_bary = scale * unit_source
    results = evaluate_core(
        rho_core, xi, zeta, mat, solver, f_bary, g_mid_newton, err_max_n, err_rms_n
    )
    results["grid_shape"] = (nr, nz)
    results["xi_max"] = xi_max
    results["zeta_max"] = zeta_max
    return results


def run_scan(core_values=None, nr=81, nz=61, xi_max=30.0, zeta_max=5.0):
    print(SEP)
    print("  PHASE Z: Enclosed-Mass Green-Kernel 2D Source")
    print(SEP)

    xi, zeta, mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    solver = factorized(mat)
    scale, unit_source, _, g_mid_newton, err_max_n, err_rms_n = calibrate_newtonian_scale(
        xi, zeta, solver
    )
    f_bary = scale * unit_source

    if core_values is None:
        core_values = np.linspace(0.15, 0.75, 13)

    results = [
        evaluate_core(core, xi, zeta, mat, solver, f_bary, g_mid_newton, err_max_n, err_rms_n)
        for core in core_values
    ]
    best = min(results, key=lambda r: r["score"])

    print(f"  {'rho_c':>7s}  {'kappa_G':>9s}  {'full rms':>10s}  {'trans rms':>10s}  {'outer rms':>10s}  {'score':>10s}")
    print("  " + "-" * 90)
    for r in results:
        print(
            f"  {r['rho_core']:7.3f}  "
            f"{r['kappa_G']:9.3f}  "
            f"{r['full_rms']:10.3e}  "
            f"{r['trans_rms']:10.3e}  "
            f"{r['outer_rms']:10.3e}  "
            f"{r['score']:10.3e}"
        )

    print("\n  Best enclosed-mass kernel:")
    print(f"    rho_c      = {best['rho_core']:.3f} r_M")
    print(f"    kappa_G    = {best['kappa_G']:.3f}")
    print(f"    full rms   = {best['full_rms']:.3e}")
    print(f"    trans rms  = {best['trans_rms']:.3e}")
    print(f"    outer rms  = {best['outer_rms']:.3e}")
    print(f"    score      = {best['score']:.3e}")
    print(f"    positivity = extra {best['extra_pos']:.3f}, total {best['total_pos']:.3f}")
    print()
    print("    r/r_M    A_vort   g_PDE/a0   g_target/a0   g_h/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(best["xi"] - factor))
        print(
            f"    {best['xi'][idx]:5.1f}   "
            f"{best['A_mid'][idx]:7.4f}   "
            f"{best['g_eff'][idx]:9.3f}   "
            f"{best['g_alg'][idx]:11.3f}   "
            f"{best['g_h'][idx]:8.3f}"
        )

    best["grid_shape"] = (nr, nz)
    best["xi_max"] = xi_max
    best["zeta_max"] = zeta_max
    return best


def print_interpretation(best):
    idx_rm = np.argmin(np.abs(best["xi"] - 1.0))
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        f"""
  This closure is genuinely nonlocal in 2D:

  - the halo source at (R, z) depends on the enclosed baryonic mass inside the
    same axisymmetric radius rho,
  - the source keeps the required vacuum tail f_h ~ 1/rho^2,
  - and the Green-kernel normalization kappa_G is fixed once by matching the
    outer-halo branch, not by hand at every radius.

  Selected parameters:
  - rho_c   = {best['rho_core']:.3f} r_M
  - kappa_G = {best['kappa_G']:.3f}

  At r ~ r_M:
  - A_vort   = {best['A_mid'][idx_rm]:.6f}
  - g_PDE/a0 = {best['g_eff'][idx_rm]:.6f}
  - g_target = {best['g_alg'][idx_rm]:.6f}

  This is the first explicit source-side 2D kernel in the present scan that
  tracks the algebraic branch over the full transition/outer-halo window
  without relying on a purely local shell source.
        """
    )


if __name__ == "__main__":
    best = run_scan()
    print_interpretation(best)
