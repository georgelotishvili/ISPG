"""
Phase AI: Boundary-selected nonlocal 2D kernel
==============================================

Goal:
  Replace the target-fitted kernel parameters of Phase Z by parameters fixed
  from regularity and outer-boundary asymptotics alone.

Parameter selection:

  1. rho_core is fixed by local-cell regularity:

         rho_core = 1 / j_{0,1}

     i.e. the dimensionless radius of one coherent Bessel cell at xi = 1.

  2. kappa_G is fixed by the deep-halo Gauss condition:

         g_h(outer) -> sqrt(m_enc^(2D)) / rho

     rather than by fitting to the full algebraic MOND target.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.sparse.linalg import factorized
from scipy.special import jn_zeros

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
BESSEL_ZERO = jn_zeros(0, 1)[0]
RHO_CORE_DERIVED = 1.0 / BESSEL_ZERO


def enclosed_mass_lookup(xi, zeta, bary_source):
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")
    dxi = xi[1] - xi[0]
    dz = zeta[1] - zeta[0]

    rho = np.sqrt(XI**2 + Z**2)
    shell_weights = np.asarray(bary_source, dtype=float) * XI * dxi * dz

    rho_flat = rho.reshape(-1)
    weight_flat = shell_weights.reshape(-1)
    order = np.argsort(rho_flat)

    rho_sorted = rho_flat[order]
    cumulative = np.cumsum(weight_flat[order])
    cumulative /= cumulative[-1]
    return rho_sorted, cumulative


def enclosed_mass_fraction(rho, rho_sorted, cumulative):
    return np.interp(rho, rho_sorted, cumulative, left=0.0, right=1.0)


def build_tail_source(xi, zeta, rho_core, rho_sorted, cumulative):
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")
    rho_eff_sq = XI**2 + Z**2 + rho_core**2
    rho_eff = np.sqrt(rho_eff_sq)
    m_enc_2d = enclosed_mass_fraction(rho_eff, rho_sorted, cumulative)
    base_source = np.sqrt(np.maximum(m_enc_2d, 1e-12)) / np.maximum(rho_eff_sq, 1e-12)
    return base_source, m_enc_2d, rho_eff


def interior_residual(mat, u, source, nr, nz):
    residual = mat @ u.reshape(-1) + source.reshape(-1)
    boundary_mask = np.ones_like(residual, dtype=bool)
    for i in range(nr):
        boundary_mask[i * nz + (nz - 1)] = False
    for j in range(nz):
        boundary_mask[(nr - 1) * nz + j] = False
    return np.max(np.abs(residual[boundary_mask]))


def resolve_asymptotic_fit_window(xi, fit_min=None, fit_max=None):
    """Choose a resolved deep-halo annulus for the outer-asymptotic fit.

    For the compact benchmark box (xi_max = 10) this reproduces the original
    5-10 r_M window.  On a widened box (for example xi_max = 20) it moves the
    normalization to a disjoint annulus in the outer half of the domain, so the
    source-side score on 0.3-10 r_M is no longer calibrated inside the same
    interval that is being reported.
    """
    xi = np.asarray(xi, dtype=float)
    xi_max = float(np.nanmax(xi))

    if fit_max is None:
        fit_max = xi_max
    fit_max = min(float(fit_max), xi_max)

    if fit_min is None:
        fit_min = max(5.0, 0.5 * fit_max)
    fit_min = max(float(fit_min), float(xi[1]) if len(xi) > 1 else 0.0)
    fit_min = min(fit_min, fit_max)

    mask = (xi >= fit_min) & (xi <= fit_max)
    if np.count_nonzero(mask) < 3:
        fit_min = float(xi[max(len(xi) // 2, 1)])
        fit_max = xi_max
        mask = (xi >= fit_min) & (xi <= fit_max)

    if np.count_nonzero(mask) < 3:
        raise ValueError("Asymptotic fit annulus is under-resolved on this grid.")

    return fit_min, fit_max, mask


def boundary_selected_kappa(g_h_base, xi, m_enc_mid, fit_min=None, fit_max=None):
    """
    Fix kappa_G from the deep-halo asymptotic condition

        g_h -> sqrt(m_enc) / xi

    rather than by fitting to the full algebraic branch.
    """
    xi = np.asarray(xi, dtype=float)
    g_h_base = np.asarray(g_h_base, dtype=float)
    m_enc_mid = np.asarray(m_enc_mid, dtype=float)

    desired = np.sqrt(np.maximum(m_enc_mid, 1e-30)) / np.maximum(xi, 1e-30)
    fit_min, fit_max, mask = resolve_asymptotic_fit_window(xi, fit_min, fit_max)
    num = np.dot(desired[mask], g_h_base[mask])
    den = np.dot(g_h_base[mask], g_h_base[mask])
    return num / max(den, 1e-30), desired, fit_min, fit_max


def solve_boundary_selected_kernel(
    rho_core=RHO_CORE_DERIVED,
    nr=81,
    nz=61,
    xi_max=10.0,
    zeta_max=5.0,
    fit_min=None,
    fit_max=None,
):
    xi, zeta, mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    solver = factorized(mat)
    scale, unit_source, _, g_mid_newton, err_max_n, err_rms_n = calibrate_newtonian_scale(
        xi, zeta, solver
    )
    f_bary = scale * unit_source
    dxi = xi[1] - xi[0]

    rho_sorted, cumulative = enclosed_mass_lookup(xi, zeta, f_bary)
    base_source, m_enc_2d, rho_eff = build_tail_source(xi, zeta, rho_core, rho_sorted, cumulative)

    u_base = solve_linear_poisson(solver, base_source, nr, nz)
    g_h_base = -grad_xi(u_base, dxi)[:, 0]
    m_enc_mid = m_enc_2d[:, 0]
    kappa_G, g_outer_target, fit_min, fit_max = boundary_selected_kappa(
        g_h_base, xi, m_enc_mid, fit_min=fit_min, fit_max=fit_max
    )

    source_extra = kappa_G * base_source
    source_total = f_bary + source_extra
    u_total = solve_linear_poisson(solver, source_total, nr, nz)
    g_mid_total = -grad_xi(u_total, dxi)[:, 0]
    g_mid_h = g_mid_total - g_mid_newton

    y_mid = g_mid_newton.copy()
    if len(y_mid) > 1:
        y_mid[0] = y_mid[1]
    _, x_target, A_mid, _ = solve_activated_midplane(y_mid, xi)

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)
    rel_alg = np.abs(g_mid_total - x_target) / np.maximum(x_target, 1e-30)
    rel_outer = np.abs(g_mid_h - g_outer_target) / np.maximum(g_outer_target, 1e-30)

    return {
        "xi": xi,
        "zeta": zeta,
        "rho_core": rho_core,
        "kappa_G": kappa_G,
        "fit_min": fit_min,
        "fit_max": fit_max,
        "g_N": g_mid_newton,
        "g_eff": g_mid_total,
        "g_h": g_mid_h,
        "g_alg": x_target,
        "g_outer_target": g_outer_target,
        "m_enc_mid": m_enc_mid,
        "m_enc_2d": m_enc_2d,
        "rho_eff": rho_eff,
        "source_extra": source_extra,
        "source_total": source_total,
        "A_mid": A_mid,
        "newton_err_max": err_max_n,
        "newton_err_rms": err_rms_n,
        "pde_residual": interior_residual(mat, u_total, source_total, nr, nz),
        "full_rms_alg": np.sqrt(np.mean(rel_alg[mask] ** 2)),
        "trans_rms_alg": np.sqrt(np.mean(rel_alg[mask_trans] ** 2)),
        "outer_rms_alg": np.sqrt(np.mean(rel_alg[mask_outer] ** 2)),
        "full_rms_outer": np.sqrt(np.mean(rel_outer[mask] ** 2)),
        "outer_match_rms": np.sqrt(np.mean(rel_outer[mask_outer] ** 2)),
    }


def print_interpretation(results):
    idx_rm = int(np.argmin(np.abs(results["xi"] - 1.0)))
    print(SEP)
    print("  PHASE AI: Boundary-Selected Nonlocal Kernel")
    print(SEP)
    print(f"  rho_core (regularity) = 1/j_01 = {results['rho_core']:.6f} r_M")
    print(f"  kappa_G  (Gauss asym.) = {results['kappa_G']:.6f}")
    print(
        f"  asymptotic fit annulus = {results['fit_min']:.3f} - "
        f"{results['fit_max']:.3f} r_M"
    )
    print(f"  Newtonian calibration  = max {results['newton_err_max']:.3e}, rms {results['newton_err_rms']:.3e}")
    print(f"  PDE residual           = {results['pde_residual']:.3e}")
    print()
    print("  Diagnostics against outer asymptotic target sqrt(m_enc)/xi:")
    print(f"    full rms  = {results['full_rms_outer']:.3e}")
    print(f"    outer rms = {results['outer_match_rms']:.3e}")
    print()
    print("  Diagnostic-only comparison against the old algebraic branch:")
    print(f"    full rms  = {results['full_rms_alg']:.3e}")
    print(f"    trans rms = {results['trans_rms_alg']:.3e}")
    print(f"    outer rms = {results['outer_rms_alg']:.3e}")
    print()
    print("    r/r_M    g_h/a0    g_asym/a0   g_total/a0   g_alg/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(results["xi"] - factor))
        print(
            f"    {results['xi'][idx]:5.1f}   "
            f"{results['g_h'][idx]:8.3f}   "
            f"{results['g_outer_target'][idx]:10.3f}   "
            f"{results['g_eff'][idx]:10.3f}   "
            f"{results['g_alg'][idx]:8.3f}"
        )

    print(
        f"""

  Reading:
  - rho_core is no longer scan-selected; it is fixed by the coherent Bessel
    cell radius at xi=1.
  - kappa_G is no longer target-fitted to the full MOND profile; it is fixed
    by the deep-halo asymptotic boundary condition.
  - Any agreement with the old algebraic branch is now diagnostic only.

  At r ~ r_M:
  - A_vort(mid) = {results['A_mid'][idx_rm]:.6f}
  - g_total/a0  = {results['g_eff'][idx_rm]:.6f}
  - g_alg/a0    = {results['g_alg'][idx_rm]:.6f}
        """
    )


if __name__ == "__main__":
    res = solve_boundary_selected_kernel()
    print_interpretation(res)
