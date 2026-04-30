"""
Phase U: Axisymmetric R-z embedding of the swirl-source closure
===============================================================

Purpose:
  Embed the activated source-side swirl closure into a genuine 2D axisymmetric
  Poisson solver on a cylindrical (R,z) grid for a thick exponential disk.

Important reading:
  This is a 2D axisymmetric EMBEDDING of the already-derived radial closure.
  The constitutive source factor chi(xi) is taken from the mature-vortex
  midplane solution and then extended through the disk column at fixed xi.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import factorized

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from constants import Omega_tr_conj, a0, r_M
from source import f_source_3d, g_newton_dimless

SEP = "=" * 78


def coherent_root(y):
    y = np.asarray(y, dtype=float)
    return 0.5 * (y + np.sqrt(y**2 + 4.0 * y))


def activation_from_x(x, xi):
    xi_safe = np.maximum(np.asarray(xi, dtype=float), 1e-4)
    x = np.asarray(x, dtype=float)
    omega = np.sqrt(np.maximum(a0 * x / (xi_safe * r_M), 0.0))
    return omega / (omega + Omega_tr_conj)


def solve_activated_midplane(y, xi, tol=1e-13, max_iter=400):
    y = np.asarray(y, dtype=float)
    x = coherent_root(y)
    xi_safe = np.maximum(xi, xi[1] if len(xi) > 1 else 1e-3)

    for _ in range(max_iter):
        A = activation_from_x(x, xi_safe)
        x_new = 0.5 * (y + np.sqrt(y**2 + 4.0 * A * y))
        if np.max(np.abs(x_new - x) / np.maximum(x_new, 1e-300)) < tol:
            x = x_new
            break
        x = x_new

    A = activation_from_x(x, xi_safe)
    chi = A / np.maximum(x, 1e-12)
    if len(chi) > 1:
        chi[0] = chi[1]
        A[0] = A[1]
    return y, x, A, chi


def idx(i, j, nz):
    return i * nz + j


def build_axisymmetric_operator(nr=81, nz=61, xi_max=30.0, zeta_max=5.0):
    xi = np.linspace(0.0, xi_max, nr)
    zeta = np.linspace(0.0, zeta_max, nz)
    dxi = xi[1] - xi[0]
    dz = zeta[1] - zeta[0]

    mat = lil_matrix((nr * nz, nr * nz))

    for i in range(nr):
        for j in range(nz):
            k = idx(i, j, nz)

            if i == nr - 1 or j == nz - 1:
                mat[k, k] = 1.0
                continue

            if i == 0 and j == 0:
                mat[k, k] = -4.0 / dxi**2 - 2.0 / dz**2
                mat[k, idx(1, 0, nz)] = 4.0 / dxi**2
                mat[k, idx(0, 1, nz)] = 2.0 / dz**2
                continue

            if i == 0:
                mat[k, k] = -4.0 / dxi**2 - 2.0 / dz**2
                mat[k, idx(1, j, nz)] = 4.0 / dxi**2
                mat[k, idx(0, j - 1, nz)] = 1.0 / dz**2
                mat[k, idx(0, j + 1, nz)] = 1.0 / dz**2
                continue

            xp = 1.0 / dxi**2 + 1.0 / (2.0 * xi[i] * dxi)
            xm = 1.0 / dxi**2 - 1.0 / (2.0 * xi[i] * dxi)

            if j == 0:
                mat[k, k] = -2.0 / dxi**2 - 2.0 / dz**2
                mat[k, idx(i + 1, j, nz)] = xp
                mat[k, idx(i - 1, j, nz)] = xm
                mat[k, idx(i, 1, nz)] = 2.0 / dz**2
                continue

            mat[k, k] = -2.0 / dxi**2 - 2.0 / dz**2
            mat[k, idx(i + 1, j, nz)] = xp
            mat[k, idx(i - 1, j, nz)] = xm
            mat[k, idx(i, j - 1, nz)] = 1.0 / dz**2
            mat[k, idx(i, j + 1, nz)] = 1.0 / dz**2

    return xi, zeta, csc_matrix(mat)


def solve_linear_poisson(solver, source, nr, nz):
    rhs = -np.asarray(source, dtype=float).reshape(-1).copy()

    # Dirichlet boundaries at outer radius/top boundary.
    for i in range(nr):
        rhs[idx(i, nz - 1, nz)] = 0.0
    for j in range(nz):
        rhs[idx(nr - 1, j, nz)] = 0.0

    u = solver(rhs).reshape((nr, nz))
    u[nr - 1, :] = 0.0
    u[:, nz - 1] = 0.0
    return u


def grad_xi(u, dxi):
    out = np.zeros_like(u)
    out[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dxi)
    out[0, :] = 0.0
    out[-1, :] = (u[-1, :] - u[-2, :]) / dxi
    return out


def calibrate_newtonian_scale(xi, zeta, solver):
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")
    unit_source = f_source_3d(XI, Z)
    u_unit = solve_linear_poisson(solver, unit_source, len(xi), len(zeta))

    dxi = xi[1] - xi[0]
    g_mid_unit = -grad_xi(u_unit, dxi)[:, 0]
    g_target = g_newton_dimless(xi)

    mask = (xi >= 0.2) & (xi <= 10.0)
    scale = np.dot(g_target[mask], g_mid_unit[mask]) / np.dot(g_mid_unit[mask], g_mid_unit[mask])

    u_newton = scale * u_unit
    g_mid = scale * g_mid_unit
    rel = np.abs(g_mid[mask] - g_target[mask]) / np.maximum(g_target[mask], 1e-30)

    return scale, unit_source, u_newton, g_mid, np.max(rel), np.sqrt(np.mean(rel**2))


def run_axisymmetric_embedding():
    print(SEP)
    print("  PHASE U: Axisymmetric Swirl-Source Embedding")
    print(SEP)

    xi, zeta, mat = build_axisymmetric_operator()
    solver = factorized(mat)
    dxi = xi[1] - xi[0]
    XI, Z = np.meshgrid(xi, zeta, indexing="ij")

    scale, unit_source, u_newton, g_mid_newton, err_max_n, err_rms_n = calibrate_newtonian_scale(
        xi, zeta, solver
    )
    f_bary = scale * unit_source

    y_mid = g_mid_newton.copy()
    if len(y_mid) > 1:
        y_mid[0] = y_mid[1]
    _, x_target, A_mid, chi_mid = solve_activated_midplane(y_mid, xi)
    source_total = f_bary * (1.0 + chi_mid[:, None])
    source_extra = f_bary * chi_mid[:, None]

    u_total = solve_linear_poisson(solver, source_total, len(xi), len(zeta))
    u_h = u_total - u_newton

    g_mid_total = -grad_xi(u_total, dxi)[:, 0]
    g_mid_h = -grad_xi(u_h, dxi)[:, 0]

    mask = (xi >= 0.2) & (xi <= 10.0)
    rel_mid = np.abs(g_mid_total[mask] - x_target[mask]) / np.maximum(x_target[mask], 1e-30)
    rel_mid_coh = np.abs(g_mid_total[mask] - coherent_root(y_mid[mask])) / np.maximum(
        coherent_root(y_mid[mask]), 1e-30
    )
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)

    residual = mat @ u_total.reshape(-1) + source_total.reshape(-1)
    boundary_mask = np.ones_like(residual, dtype=bool)
    nr, nz = len(xi), len(zeta)
    for i in range(nr):
        boundary_mask[idx(i, nz - 1, nz)] = False
    for j in range(nz):
        boundary_mask[idx(nr - 1, j, nz)] = False

    resid_int = np.max(np.abs(residual[boundary_mask]))

    print(f"  grid                    = {len(xi)} x {len(zeta)}")
    print(f"  Newtonian calibration   = {scale:.6f}")
    print(f"  Newtonian midplane err  = max {err_max_n:.3e}, rms {err_rms_n:.3e}")
    print(f"  total PDE residual      = {resid_int:.3e}")
    print(f"  activated midplane err  = max {np.max(rel_mid):.3e}, rms {np.sqrt(np.mean(rel_mid**2)):.3e}")
    print(
        f"  transition-zone err     = max {np.max(np.abs(g_mid_total[mask_trans]-x_target[mask_trans])/np.maximum(x_target[mask_trans],1e-30)):.3e}, "
        f"rms {np.sqrt(np.mean((np.abs(g_mid_total[mask_trans]-x_target[mask_trans])/np.maximum(x_target[mask_trans],1e-30))**2)):.3e}"
    )
    print(
        f"  outer-halo err          = max {np.max(np.abs(g_mid_total[mask_outer]-x_target[mask_outer])/np.maximum(x_target[mask_outer],1e-30)):.3e}, "
        f"rms {np.sqrt(np.mean((np.abs(g_mid_total[mask_outer]-x_target[mask_outer])/np.maximum(x_target[mask_outer],1e-30))**2)):.3e}"
    )
    print(f"  vs coherent limit       = max {np.max(rel_mid_coh):.3e}")
    print(f"  extra-source positivity = {np.mean(source_extra[mask, :] >= 0):.3f}")
    print()
    print("    r/r_M    A_vort    chi      g_PDE/a0   g_target/a0   g_h/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx_r = np.argmin(np.abs(xi - factor))
        print(
            f"    {xi[idx_r]:5.1f}   "
            f"{A_mid[idx_r]:8.4f}   "
            f"{chi_mid[idx_r]:8.3f}   "
            f"{g_mid_total[idx_r]:9.3f}   "
            f"{x_target[idx_r]:11.3f}   "
            f"{g_mid_h[idx_r]:8.3f}"
        )

    mid_idx = np.argmin(np.abs(xi - 1.0))
    z_samples = [0.0, 0.1, 0.3, 0.5, 1.0]
    print()
    print("  Vertical profile of total potential at r ~ r_M:")
    print("    z/r_M      U_total")
    for zv in z_samples:
        j = np.argmin(np.abs(zeta - zv))
        print(f"    {zeta[j]:5.2f}   {u_total[mid_idx, j]:10.4e}")

    return {
        "xi": xi,
        "zeta": zeta,
        "A_mid": A_mid,
        "chi_mid": chi_mid,
        "g_target": x_target,
        "g_mid_total": g_mid_total,
        "g_mid_h": g_mid_h,
        "u_total": u_total,
        "u_newton": u_newton,
        "source_total": source_total,
        "source_extra": source_extra,
        "newton_err_max": err_max_n,
        "activated_err_max": np.max(rel_mid),
        "activated_err_rms": np.sqrt(np.mean(rel_mid**2)),
        "residual": resid_int,
    }


def print_interpretation(results):
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    idx_rm = np.argmin(np.abs(results["xi"] - 1.0))
    print(
        f"""
  This 2D axisymmetric embedding gives a genuine R-z potential and keeps the
  extra source positive, but it is only a PARTIAL success as a profile match.

  At r ~ r_M:
  - A_vort     = {results["A_mid"][idx_rm]:.6f}
  - chi_mid    = {results["chi_mid"][idx_rm]:.6f}
  - g_PDE/a0   = {results["g_mid_total"][idx_rm]:.6f}
  - g_target   = {results["g_target"][idx_rm]:.6f}

  Program role:
  this is a 2D embedding of the radial constitutive coefficient chi(xi),
  not yet the fully local nonlinear 2D closure chi(xi,zeta,|grad U|).
  It works reasonably near the MOND transition radius, but the mismatch grows
  in the outer halo. That means a simple column-wise extension chi(xi) is not
  yet enough for a fully satisfactory 2D realization; the transported source
  likely needs a genuinely nonlocal radial/vertical spread in axisymmetry.
        """
    )


if __name__ == "__main__":
    results = run_axisymmetric_embedding()
    print_interpretation(results)
