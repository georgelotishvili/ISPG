"""
Phase W: Positive nonlocal 2D source via radial transport smoothing
===================================================================

Purpose:
  Improve the axisymmetric embedding while preserving source positivity by
  smoothing the local extra source radially with a transport kernel.
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
    return scale, unit_source, scale * u_unit, scale * g_mid_unit


def smooth_radially(source, xi, sigma):
    if sigma <= 1e-12:
        return source.copy()
    dx = xi[:, None] - xi[None, :]
    W = np.exp(-0.5 * (dx / sigma) ** 2)
    W /= np.sum(W, axis=1, keepdims=True)
    return W @ source


def evaluate_sigma(sigma, xi, zeta, solver, f_bary, u_newton, g_mid_newton):
    y_mid = g_mid_newton.copy()
    y_mid[0] = y_mid[1]
    x_target, A_mid = solve_activated_profile(y_mid, xi + 1e-8)
    chi_mid = A_mid / np.maximum(x_target, 1e-12)

    raw_extra = f_bary * chi_mid[:, None]
    source_extra = smooth_radially(raw_extra, xi, sigma)
    source_total = f_bary + source_extra

    u_total = solve_linear_poisson(solver, source_total, len(xi), len(zeta))
    g_mid_total = -grad_xi(u_total, xi[1] - xi[0])[:, 0]

    mask = (xi >= 0.3) & (xi <= 10.0)
    mask_trans = (xi >= 0.3) & (xi <= 3.0)
    mask_outer = (xi > 3.0) & (xi <= 10.0)

    rel = np.abs(g_mid_total - x_target) / np.maximum(x_target, 1e-30)
    return {
        "sigma": sigma,
        "xi": xi,
        "A_mid": A_mid,
        "chi_mid": chi_mid,
        "g_target": x_target,
        "g_mid_total": g_mid_total,
        "g_mid_h": g_mid_total - g_mid_newton,
        "source_extra": source_extra,
        "source_total": source_total,
        "full_max": np.max(rel[mask]),
        "full_rms": np.sqrt(np.mean(rel[mask] ** 2)),
        "trans_max": np.max(rel[mask_trans]),
        "trans_rms": np.sqrt(np.mean(rel[mask_trans] ** 2)),
        "outer_max": np.max(rel[mask_outer]),
        "outer_rms": np.sqrt(np.mean(rel[mask_outer] ** 2)),
        "extra_pos": np.mean(source_extra[mask, :] >= -1e-10),
        "total_pos": np.mean(source_total[mask, :] >= -1e-10),
    }


def run_scan():
    print(SEP)
    print("  PHASE W: Smoothed Positive 2D Source")
    print(SEP)

    xi, zeta, mat = build_axisymmetric_operator()
    solver = factorized(mat)
    scale, unit_source, u_newton, g_mid_newton = calibrate_newtonian_scale(xi, zeta, solver)
    f_bary = scale * unit_source

    sigmas = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = [evaluate_sigma(s, xi, zeta, solver, f_bary, u_newton, g_mid_newton) for s in sigmas]
    best = min(results, key=lambda r: r["full_rms"])

    print(f"  {'sigma':>6s}  {'full rms':>10s}  {'trans rms':>10s}  {'outer rms':>10s}  {'pos':>8s}")
    print("  " + "-" * 60)
    for r in results:
        print(
            f"  {r['sigma']:6.2f}  "
            f"{r['full_rms']:10.3e}  "
            f"{r['trans_rms']:10.3e}  "
            f"{r['outer_rms']:10.3e}  "
            f"{r['extra_pos']:8.3f}"
        )

    print("\n  Best sigma by full-range RMS:")
    print(f"    sigma      = {best['sigma']:.2f} r_M")
    print(f"    full rms   = {best['full_rms']:.3e}")
    print(f"    trans rms  = {best['trans_rms']:.3e}")
    print(f"    outer rms  = {best['outer_rms']:.3e}")
    print(f"    positivity = {best['extra_pos']:.3f}")
    print()
    print("    r/r_M    A_vort    chi      g_PDE/a0   g_target/a0   g_h/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(best["xi"] - factor))
        print(
            f"    {best['xi'][idx]:5.1f}   "
            f"{best['A_mid'][idx]:8.4f}   "
            f"{best['chi_mid'][idx]:8.3f}   "
            f"{best['g_mid_total'][idx]:9.3f}   "
            f"{best['g_target'][idx]:11.3f}   "
            f"{best['g_mid_h'][idx]:8.3f}"
        )

    return best


def print_interpretation(best):
    idx_rm = np.argmin(np.abs(best["xi"] - 1.0))
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        f"""
  This is the first positive nonlocal 2D transport-kernel test.

  The extra swirl source is kept positive and is spread radially over a
  transport width sigma. The best-fit width in this scan is

      sigma = {best['sigma']:.2f} r_M .

  At r ~ r_M:
  - A_vort   = {best['A_mid'][idx_rm]:.6f}
  - g_PDE/a0 = {best['g_mid_total'][idx_rm]:.6f}
  - g_target = {best['g_target'][idx_rm]:.6f}

  The scan shows a clear tradeoff:
  stronger smoothing helps the outer halo but hurts the transition zone.
  So simple positive radial spreading alone is NOT enough for the final
  2D realization of the source-side MOND closure.
        """
    )


if __name__ == "__main__":
    best = run_scan()
    print_interpretation(best)
