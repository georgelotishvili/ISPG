"""
Phase V: Nonlocal 2D swirl embedding via transported-potential ratio
====================================================================

Purpose:
  Improve the axisymmetric R-z realization by replacing the local column-wise
  source ansatz with a nonlocal derivative source obtained from

      U_h(R,z) = R_prof(R) U_N(R,z),

  where R_prof is fixed by the target transported potential on the midplane.
"""

import io
import sys

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import factorized

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from constants import Omega_tr_conj, a0, r_M
from source import f_source_3d, g_newton_dimless

SEP = "=" * 78


def coherent_root(y):
    y = np.asarray(y, dtype=float)
    return 0.5 * (y + np.sqrt(y**2 + 4.0 * y))


def activation_from_x(x, xi):
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    omega = np.sqrt(np.maximum(a0 * x / (np.maximum(xi, 1e-8) * r_M), 0.0))
    return omega / (omega + Omega_tr_conj)


def solve_activated_profile(y, xi, tol=1e-13, max_iter=400):
    x = coherent_root(y)
    for _ in range(max_iter):
        A = activation_from_x(x, xi)
        x_new = 0.5 * (y + np.sqrt(y**2 + 4.0 * A * y))
        rel = np.max(np.abs(x_new - x) / np.maximum(x_new, 1e-300))
        x = x_new
        if rel < tol:
            break
    A = activation_from_x(x, xi)
    return x, A


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


def integrate_potential_midplane(xi, g_h):
    xi = np.asarray(xi, dtype=float)
    g_h = np.asarray(g_h, dtype=float)
    U_h = np.zeros_like(xi)
    for i in range(len(xi) - 2, -1, -1):
        dxi = xi[i + 1] - xi[i]
        U_h[i] = U_h[i + 1] + 0.5 * (
            xi[i] * g_h[i] + xi[i + 1] * g_h[i + 1]
        ) * dxi
    return U_h


def run_nonlocal_embedding():
    print(SEP)
    print("  PHASE V: Nonlocal Potential-Ratio 2D Embedding")
    print(SEP)

    xi, zeta, mat = build_axisymmetric_operator()
    solver = factorized(mat)
    dxi = xi[1] - xi[0]

    scale, unit_source, u_newton, g_mid_newton, err_max_n, err_rms_n = calibrate_newtonian_scale(
        xi, zeta, solver
    )
    f_bary = scale * unit_source

    y_mid = g_mid_newton.copy()
    if len(y_mid) > 1:
        y_mid[0] = y_mid[1]
    x_target, A_mid = solve_activated_profile(y_mid, xi + 1e-8)
    g_h_target = x_target - y_mid

    U_h_mid = integrate_potential_midplane(xi, g_h_target)
    R_prof = np.zeros_like(xi)
    mask_R = np.abs(u_newton[:, 0]) > 1e-12
    R_prof[mask_R] = U_h_mid[mask_R] / u_newton[mask_R, 0]
    if len(R_prof) > 1:
        R_prof[0] = R_prof[1]
    R_prof[-1] = 0.0

    u_h_ansatz = R_prof[:, None] * u_newton
    source_extra = -(mat @ u_h_ansatz.reshape(-1)).reshape(u_h_ansatz.shape)
    source_total = f_bary + source_extra

    u_total = solve_linear_poisson(solver, source_total, len(xi), len(zeta))
    g_mid_total = -grad_xi(u_total, dxi)[:, 0]
    g_mid_h = g_mid_total - g_mid_newton

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)
    rel_mid = np.abs(g_mid_total[mask] - x_target[mask]) / np.maximum(x_target[mask], 1e-30)
    rel_trans = np.abs(g_mid_total[mask_trans] - x_target[mask_trans]) / np.maximum(
        x_target[mask_trans], 1e-30
    )
    rel_outer = np.abs(g_mid_total[mask_outer] - x_target[mask_outer]) / np.maximum(
        x_target[mask_outer], 1e-30
    )

    reconstruct = np.max(np.abs(u_total - (u_newton + u_h_ansatz)))
    residual = mat @ u_total.reshape(-1) + source_total.reshape(-1)
    nr, nz = len(xi), len(zeta)
    boundary_mask = np.ones_like(residual, dtype=bool)
    for i in range(nr):
        boundary_mask[i * nz + (nz - 1)] = False
    for j in range(nz):
        boundary_mask[(nr - 1) * nz + j] = False
    resid_int = np.max(np.abs(residual[boundary_mask]))

    print(f"  grid                      = {len(xi)} x {len(zeta)}")
    print(f"  Newtonian calibration     = {scale:.6f}")
    print(f"  reconstruction mismatch   = {reconstruct:.3e}")
    print(f"  total PDE residual        = {resid_int:.3e}")
    print(f"  full midplane err         = max {np.max(rel_mid):.3e}, rms {np.sqrt(np.mean(rel_mid**2)):.3e}")
    print(f"  transition-zone err       = max {np.max(rel_trans):.3e}, rms {np.sqrt(np.mean(rel_trans**2)):.3e}")
    print(f"  outer-halo err            = max {np.max(rel_outer):.3e}, rms {np.sqrt(np.mean(rel_outer**2)):.3e}")
    print(f"  extra-source positivity   = {np.mean(source_extra[mask, :] >= -1e-10):.3f}")
    print(f"  total-source positivity   = {np.mean(source_total[mask, :] >= -1e-10):.3f}")
    print()
    print("    r/r_M    A_vort    R_prof    g_PDE/a0   g_target/a0   g_h/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {xi[idx]:5.1f}   "
            f"{A_mid[idx]:8.4f}   "
            f"{R_prof[idx]:8.3f}   "
            f"{g_mid_total[idx]:9.3f}   "
            f"{x_target[idx]:11.3f}   "
            f"{g_mid_h[idx]:8.3f}"
        )

    return {
        "xi": xi,
        "zeta": zeta,
        "A_mid": A_mid,
        "R_prof": R_prof,
        "g_target": x_target,
        "g_mid_total": g_mid_total,
        "g_mid_newton": g_mid_newton,
        "g_mid_h": g_mid_h,
        "source_extra": source_extra,
        "source_total": source_total,
        "reconstruct": reconstruct,
        "residual": resid_int,
        "rel_full_max": np.max(rel_mid),
        "rel_full_rms": np.sqrt(np.mean(rel_mid**2)),
        "rel_trans_max": np.max(rel_trans),
        "rel_trans_rms": np.sqrt(np.mean(rel_trans**2)),
        "rel_outer_max": np.max(rel_outer),
        "rel_outer_rms": np.sqrt(np.mean(rel_outer**2)),
        "extra_pos": np.mean(source_extra[mask, :] >= -1e-10),
        "total_pos": np.mean(source_total[mask, :] >= -1e-10),
    }


def print_interpretation(results):
    idx_rm = np.argmin(np.abs(results["xi"] - 1.0))
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        f"""
  This nonlocal derivative-source embedding is closer to the transport picture:

  - first determine the target transported potential on the midplane,
  - then extend the ratio U_h/U_N through the full R-z potential,
  - then let the 2D Poisson operator determine the implied source.

  At r ~ r_M:
  - A_vort   = {results["A_mid"][idx_rm]:.6f}
  - R_prof   = {results["R_prof"][idx_rm]:.6f}
  - g_PDE/a0 = {results["g_mid_total"][idx_rm]:.6f}
  - g_target = {results["g_target"][idx_rm]:.6f}

  In practice this naive derivative-source extension is too aggressive:
  it overshoots strongly and generates sign-changing source regions.
  So it does NOT yet improve the axisymmetric realization beyond Phase U.
        """
    )


if __name__ == "__main__":
    results = run_nonlocal_embedding()
    print_interpretation(results)
