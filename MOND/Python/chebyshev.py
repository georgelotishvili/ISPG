"""
Chebyshev spectral differentiation in the log-radial coordinate s = ln(xi).

Maps the domain [xi_min, xi_max] -> [s_min, s_max] with uniform resolution
across decades of radius.  Chebyshev-Gauss-Lobatto collocation.

Key relations (equatorial plane, azimuthally symmetric):
  xi = e^s
  du/d(xi) = xi^{-1} du/ds
  hat{nabla}^2 u = xi^{-2} d^2u/ds^2      (cylindrical Laplacian)

Exports:
  cheb_matrices(N, s_min, s_max) -> s, D1, D2
  laplacian_1d(D2, s)            -> L  (Laplacian operator matrix)

Reference: Trefethen, Spectral Methods in MATLAB (2000), Ch. 6.
"""

import numpy as np
from constants import xi_min, xi_max, s_min, s_max, N_cheb


# =====================================================================
# Chebyshev differentiation matrix on [-1, 1]
# =====================================================================

def cheb_d_matrix(N):
    """First-derivative Chebyshev differentiation matrix on [-1, 1].

    Returns (t, D) where t are the N+1 Chebyshev-Gauss-Lobatto nodes
    t_j = cos(pi j / N), j = 0..N   (t_0 = +1, t_N = -1).
    """
    if N == 0:
        return np.array([1.0]), np.array([[0.0]])

    t = np.cos(np.pi * np.arange(N + 1) / N)
    # Barycentric weights
    cc = np.ones(N + 1)
    cc[0] = 2.0
    cc[N] = 2.0
    cc *= (-1.0) ** np.arange(N + 1)

    X = t.reshape(-1, 1) * np.ones((1, N + 1))   # X[i,j] = t[i]
    dX = X - X.T                                   # dX[i,j] = t[i] - t[j]
    D = np.outer(cc, 1.0 / cc) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return t, D


# =====================================================================
# Mapped Chebyshev matrices on [s_min, s_max]
# =====================================================================

def cheb_matrices(N=None, smin=None, smax=None):
    """Chebyshev collocation in the s = ln(xi) coordinate.

    Parameters
    ----------
    N    : int, number of intervals (N+1 collocation points). Default: N_cheb.
    smin : float, left boundary.  Default: ln(xi_min).
    smax : float, right boundary. Default: ln(xi_max).

    Returns
    -------
    s  : (N+1,) array of collocation points in [smin, smax]
         ordered from smax (outer) to smin (inner).
    D1 : (N+1, N+1) first-derivative matrix  d/ds
    D2 : (N+1, N+1) second-derivative matrix d^2/ds^2
    """
    if N is None:
        N = N_cheb
    if smin is None:
        smin = s_min
    if smax is None:
        smax = s_max

    t, Dt = cheb_d_matrix(N)

    # Affine map: s = (smax+smin)/2 + (smax-smin)/2 * t
    L = smax - smin
    s = 0.5 * (smax + smin) + 0.5 * L * t  # s[0] = smax, s[N] = smin

    # Scale derivatives: ds/dt = L/2, so d/ds = (2/L) d/dt
    D1 = (2.0 / L) * Dt
    D2 = D1 @ D1

    return s, D1, D2


def xi_from_s(s):
    """Convert log-coordinate s to dimensionless radius xi = e^s."""
    return np.exp(s)


def laplacian_1d(D2, s):
    """Laplacian operator matrix for the equatorial plane.

    hat{nabla}^2 u = xi^{-2} d^2u/ds^2.
    Returns L such that L @ u = hat{nabla}^2 u.
    """
    xi2_inv = np.exp(-2 * s)
    return np.diag(xi2_inv) @ D2


# =====================================================================
# Validation
# =====================================================================

def validate(N=None):
    """Validate D1 and D2 against known analytic derivatives.

    Test functions:
      (a) u(s) = e^s          ->  u' = e^s,   u'' = e^s
      (b) u(s) = s^3          ->  u' = 3s^2,  u'' = 6s
      (c) u(s) = sin(pi*s/L)  ->  u' = (pi/L)cos(...), u'' = -(pi/L)^2 sin(...)
    """
    if N is None:
        N = N_cheb

    s, D1, D2 = cheb_matrices(N)
    xi = xi_from_s(s)

    sep = "=" * 65
    print(sep)
    print("  Step 1.3 -- Chebyshev Spectral Validation")
    print(sep)
    print(f"\n  N = {N} collocation points, s in [{s[-1]:.2f}, {s[0]:.2f}]")

    # Interior points (exclude boundaries for fair error measure)
    interior = slice(1, -1)

    # --- Test (a): u = e^s ---
    u_a = np.exp(s)
    du_a_exact = np.exp(s)
    d2u_a_exact = np.exp(s)

    du_a_num = D1 @ u_a
    d2u_a_num = D2 @ u_a

    err_d1_a = np.max(np.abs(du_a_num[interior] - du_a_exact[interior])
                      / np.abs(du_a_exact[interior]))
    err_d2_a = np.max(np.abs(d2u_a_num[interior] - d2u_a_exact[interior])
                      / np.abs(d2u_a_exact[interior]))

    print(f"\n  (a) u = e^s:")
    print(f"      D1 relative error (interior): {err_d1_a:.2e}")
    print(f"      D2 relative error (interior): {err_d2_a:.2e}")

    # --- Test (b): u = s^3 (polynomial -- should be exact for N >= 3) ---
    u_b = s**3
    du_b_exact = 3 * s**2
    d2u_b_exact = 6 * s

    du_b_num = D1 @ u_b
    d2u_b_num = D2 @ u_b

    err_d1_b = np.max(np.abs(du_b_num - du_b_exact))
    err_d2_b = np.max(np.abs(d2u_b_num - d2u_b_exact))

    print(f"\n  (b) u = s^3 (polynomial):")
    print(f"      D1 absolute error: {err_d1_b:.2e}")
    print(f"      D2 absolute error: {err_d2_b:.2e}")

    # --- Test (c): u = sin(k*s) ---
    L = s[0] - s[-1]  # domain length
    k = np.pi / L
    u_c = np.sin(k * s)
    du_c_exact = k * np.cos(k * s)
    d2u_c_exact = -k**2 * np.sin(k * s)

    du_c_num = D1 @ u_c
    d2u_c_num = D2 @ u_c

    mask_c = np.abs(du_c_exact) > 1e-12
    err_d1_c = np.max(np.abs((du_c_num[mask_c] - du_c_exact[mask_c])
                              / du_c_exact[mask_c]))
    mask_c2 = np.abs(d2u_c_exact) > 1e-12
    err_d2_c = np.max(np.abs((d2u_c_num[mask_c2] - d2u_c_exact[mask_c2])
                              / d2u_c_exact[mask_c2]))

    print(f"\n  (c) u = sin(pi s / L):")
    print(f"      D1 relative error: {err_d1_c:.2e}")
    print(f"      D2 relative error: {err_d2_c:.2e}")

    # --- Test (d): Laplacian of known Poisson solution ---
    # For u_N such that -(1/xi) d/dxi (xi du/dxi) = f,
    # in s-coordinates: -xi^{-2} d^2u/ds^2 = f.
    # Use u = -m_enc(xi)/xi where m_enc = 1-(1+eta*xi)exp(-eta*xi)
    # Then du/dxi = ... just check Laplacian operator construction.
    L_op = laplacian_1d(D2, s)
    # L_op @ u = xi^{-2} d^2u/ds^2. For u = e^s = xi:
    # d^2(e^s)/ds^2 = e^s, so L @ (e^s) = e^{-2s} * e^s = e^{-s} = 1/xi
    Lu_a = L_op @ u_a
    Lu_a_exact = 1.0 / xi
    err_L_a = np.max(np.abs(Lu_a[interior] - Lu_a_exact[interior])
                     / np.abs(Lu_a_exact[interior]))
    print(f"\n  (d) Laplacian test: L @ (e^s) = e^{{-s}} = 1/xi:")
    print(f"      Relative error (interior): {err_L_a:.2e}")

    # Summary
    # D2 = D1@D1 accumulates roundoff at large N; 1e-6 is realistic for N~200
    all_ok = (err_d1_a < 1e-8 and err_d2_a < 1e-6
              and err_d1_b < 1e-8 and err_d2_b < 1e-5
              and err_d1_c < 1e-6 and err_d2_c < 1e-6)
    print(f"\n  --- RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ---")

    # Convergence study: error vs N
    print(f"\n  --- Convergence study (test a: u = e^s) ---")
    print(f"  {'N':>6s}  {'D1 err':>12s}  {'D2 err':>12s}")
    for Ntest in [20, 40, 80, 120, 160, 200]:
        st, D1t, D2t = cheb_matrices(Ntest)
        ut = np.exp(st)
        e1 = np.max(np.abs((D1t @ ut)[1:-1] - ut[1:-1]) / ut[1:-1])
        e2 = np.max(np.abs((D2t @ ut)[1:-1] - ut[1:-1]) / ut[1:-1])
        print(f"  {Ntest:6d}  {e1:12.2e}  {e2:12.2e}")

    print(f"\n{sep}")
    return all_ok


if __name__ == "__main__":
    validate()
