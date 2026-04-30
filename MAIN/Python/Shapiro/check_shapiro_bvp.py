"""BVP-based computation of the full 2PN Shapiro coefficient in ISPG vs GR.

Physical setup (symmetric, Cartesian 2D):
    emitter at (b, -L), receiver at (b, +L), lens at origin.
    Ray travels from emitter to receiver, bending around the lens.
    By symmetry the ray dips toward the lens, reaching geometric closest
    approach r_min < b at z=0, then returns to x=b at z=+L.

Shooting method:
    Find initial slope xp(-L) such that integrating the Fermat ODE
    forward to z=+L yields x(+L) = b.  Both endpoints at same x removes
    the asymmetric-asymptote linear IR divergence that plagues the IVP
    formulation in check_shapiro_2pn.py.

Observable Shapiro delay:
    Delta t(rg, b, L) = T_bent - L_flat
    where T_bent = integral of n(r) * sqrt(1 + xp^2) dz along bent ray,
    L_flat = sqrt((2L)^2 + (x_R - x_E)^2) = 2L (since x_E = x_R = b).

Refractive indices:
    ISPG (bi-conformal): n(r) = exp(2 rg / r)
    GR (isotropic Schwarzschild): n(r) = (1 + rg/(2r))^3 / (1 - rg/(2r))

Both reduce to 1 + 2 rg/r + O(rg^2).  At 2PN:
    ISPG coefficient of (rg/r)^2 in n:  2   (from exp expansion)
    GR   coefficient of (rg/r)^2 in n:  7/4

Expected behavior:
    1PN: Delta t = 4 rg ln(2L/b) + O(rg^2) -- SAME for ISPG and GR.
    2PN: B_ispg != B_gr; the difference is the ISPG distinctive prediction.

The refractive-only straight-line 2PN piece is 2*pi for ISPG (proven
analytically, see tools/check_shapiro_born.py) and 7*pi/4 for GR.
The full 2PN coefficient includes bent-ray and path-length contributions
beyond the straight-line Born piece.

Run:  python check_shapiro_bvp.py
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------- Refractive indices and log-gradients ----------

def n_ispg(r, rg):
    return math.exp(2.0 * rg / r)


def dlnn_grad_ispg(x, z, r, rg):
    # ln n = 2 rg / r  ->  d(ln n)/dr = -2 rg / r^2
    # d/dx = d/dr * x/r;  d/dz = d/dr * z/r
    fac = -2.0 * rg / (r * r * r)
    return fac * x, fac * z


def n_gr(r, rg):
    a = rg / (2.0 * r)
    return (1.0 + a) ** 3 / (1.0 - a)


def dlnn_grad_gr(x, z, r, rg):
    # ln n = 3 ln(1+a) - ln(1-a), a = rg/(2r)
    # d(ln n)/da = 3/(1+a) + 1/(1-a)
    # da/dr = -rg/(2 r^2)
    a = rg / (2.0 * r)
    dlnn_da = 3.0 / (1.0 + a) + 1.0 / (1.0 - a)
    da_dr = -rg / (2.0 * r * r)
    dlnn_dr = dlnn_da * da_dr
    fac = dlnn_dr / r
    return fac * x, fac * z


# ---------- Fermat ODE integrator with embedded path-time accumulation ----------

def integrate_ray(rg, b, L, slope0, n_func, dlnn_func, rtol=1e-13, atol=1e-15):
    """Integrate the Fermat ODE from z=-L with (x, xp)=(b, slope0) to z=+L.

    Returns (x_final, xp_final, T_bent).  T_bent accumulates
    integral of n(r) * sqrt(1 + xp^2) dz along the bent path.
    """
    def rhs(z, y):
        x, xp, _T = y
        r = math.sqrt(x * x + z * z)
        if r < 1e-10:
            r = 1e-10
        dlnn_dx, dlnn_dz = dlnn_func(x, z, r, rg)
        xpp = (1.0 + xp * xp) * (dlnn_dx - xp * dlnn_dz)
        dTdz = n_func(r, rg) * math.sqrt(1.0 + xp * xp)
        return [xp, xpp, dTdz]

    sol = solve_ivp(rhs, (-L, L), [b, slope0, 0.0],
                    method='DOP853',
                    rtol=rtol, atol=atol,
                    max_step=L / 4000.0)
    if not sol.success:
        return None, None, None
    return sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]


def straight_line_integral(rg, b, L, n_func):
    """Pure Born approximation: integrate n(r_0) * dz along the straight line
    x = b, z in [-L, +L].  Minus flat reference 2L gives the straight-line
    Shapiro estimate.  For ISPG this is the "refractive" piece of Eq. refr2pi
    in MAIN/6. ISPG_Shapiro.tex.
    """
    from scipy.integrate import quad
    def integrand(z):
        r = math.sqrt(b * b + z * z)
        return n_func(r, rg) - 1.0
    val, _ = quad(integrand, -L, L, limit=400, epsabs=1e-15, epsrel=1e-13)
    return val


# ---------- Shooting for symmetric BVP: x(-L)=b, x(+L)=b ----------

def shapiro_delay_bvp(rg, b, L, n_func, dlnn_func, verbose=False):
    """Shoot for slope0 such that x(+L)=b.  Return Delta t = T_bent - 2L."""
    def residual(slope0):
        x_f, _, _ = integrate_ray(rg, b, L, slope0, n_func, dlnn_func)
        if x_f is None:
            return 1e10
        return x_f - b

    # Bracket: small-deflection estimate places slope0 ~ -O(rg/b).
    # Start with generous bracket and expand if needed.
    scale = max(10.0 * rg / b, 1e-7)
    slope_lo, slope_hi = -scale, scale
    r_lo, r_hi = residual(slope_lo), residual(slope_hi)
    attempts = 0
    while r_lo * r_hi > 0.0 and attempts < 12:
        slope_lo *= 2.0
        slope_hi *= 2.0
        r_lo, r_hi = residual(slope_lo), residual(slope_hi)
        attempts += 1
    if r_lo * r_hi > 0.0:
        raise RuntimeError(
            f"Cannot bracket shooting slope: rg={rg}, b={b}, L={L}"
        )

    slope_opt = brentq(residual, slope_lo, slope_hi, xtol=1e-14, rtol=1e-14)
    _, _, T_bent = integrate_ray(rg, b, L, slope_opt, n_func, dlnn_func)
    L_flat = 2.0 * L  # endpoints are (b, ±L), straight-line length is exactly 2L
    dt = T_bent - L_flat
    if verbose:
        print(f"     rg={rg:.3e}  slope0={slope_opt:.3e}  dt={dt:.6e}")
    return dt


# ---------- Polynomial fit in rg: extract A (1PN), B (2PN), C (3PN) ----------

def extract_coeffs(rg_values, b, L, n_func, dlnn_func):
    dts = []
    for rg in rg_values:
        dt = shapiro_delay_bvp(rg, b, L, n_func, dlnn_func)
        dts.append(dt)
    rg_arr = np.array(rg_values, dtype=float)
    dt_arr = np.array(dts, dtype=float)
    # Fit dt = A*rg + B*rg^2 + C*rg^3 + D*rg^4
    M = np.column_stack([rg_arr, rg_arr ** 2, rg_arr ** 3, rg_arr ** 4])
    sol, *_ = np.linalg.lstsq(M, dt_arr, rcond=None)
    return sol, dt_arr


# ---------- Convergence test with log L ----------

def scan(L_values, rg_values, b):
    print("=" * 72)
    print("ISPG vs GR 2PN Shapiro delay via BVP shooting (fixed endpoints)")
    print(f"b = {b}, rg values = {rg_values}")
    print("Fit: Delta t(rg) = A*rg + B*rg^2 + C*rg^3 + D*rg^4")
    print("=" * 72)
    rows_ispg = []
    rows_gr = []
    for L in L_values:
        expected_A = 4.0 * math.log(2.0 * L / b)
        sol_i, _ = extract_coeffs(rg_values, b, L, n_ispg, dlnn_grad_ispg)
        sol_g, _ = extract_coeffs(rg_values, b, L, n_gr, dlnn_grad_gr)
        A_i, B_i, C_i, _ = sol_i
        A_g, B_g, C_g, _ = sol_g
        rows_ispg.append((L, A_i, B_i, C_i))
        rows_gr.append((L, A_g, B_g, C_g))
        print(f"\nL = {L:>6}  (expected 1PN A = 4 ln(2L/b) = {expected_A:.5f})")
        print(f"  ISPG   A = {A_i:>10.5f}   B_2PN = {B_i:>10.5f}   C_3PN = {C_i:>10.5f}")
        print(f"  GR     A = {A_g:>10.5f}   B_2PN = {B_g:>10.5f}   C_3PN = {C_g:>10.5f}")
        print(f"  diff   dA = {A_i-A_g:>10.5f}   dB_2PN = {B_i-B_g:>10.5f}   dC_3PN = {C_i-C_g:>10.5f}")
    return rows_ispg, rows_gr


def verify_straight_line_refractive(b, L):
    """Independent check: straight-line Born integral should give exactly
    the analytic 2PN refractive coefficient for each theory.
    ISPG: 2*pi,  GR: 7*pi/4.
    """
    rg_values = [1e-4, 3e-4, 1e-3, 3e-3]
    rg_arr = np.array(rg_values, dtype=float)

    def extract(n_func):
        vals = np.array([straight_line_integral(rg, b, L, n_func) for rg in rg_values])
        M = np.column_stack([rg_arr, rg_arr ** 2, rg_arr ** 3])
        sol, *_ = np.linalg.lstsq(M, vals, rcond=None)
        return sol

    sol_i = extract(n_ispg)
    sol_g = extract(n_gr)
    print("\n" + "=" * 72)
    print(f"Straight-line Born verification at L={L}, b={b}")
    print("=" * 72)
    A_i, B_i, _ = sol_i
    A_g, B_g, _ = sol_g
    print(f"  ISPG:  A={A_i:.5f}  B_2PN={B_i:.5f}  (expect B -> 2*pi = {2*math.pi:.5f})")
    print(f"  GR:    A={A_g:.5f}  B_2PN={B_g:.5f}  (expect B -> 7*pi/4 = {7*math.pi/4:.5f})")
    print(f"  dB (ISPG-GR, straight) = {B_i - B_g:.5f}  (expect pi/4 = {math.pi/4:.5f})")


def main():
    b = 1.0
    rg_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    L_values = [30.0, 50.0, 100.0, 200.0]
    rows_i, rows_g = scan(L_values, rg_values, b)

    # Straight-line sanity check
    verify_straight_line_refractive(b, 500.0)

    # Summary for log-L dependence of the 2PN coefficient B
    print("\n" + "=" * 72)
    print("BVP 2PN summary and ISPG-GR discriminator")
    print("=" * 72)
    print(f"  straight-line refractive 2PN  (analytic):")
    print(f"    ISPG = 2π      = {2.0*math.pi:.6f}")
    print(f"    GR   = 7π/4    = {7.0*math.pi/4.0:.6f}")
    print(f"    ΔB   = π/4     = {math.pi/4:.6f}   <-- ISPG 2PN distinctive signature")
    print()
    print(f"{'L':>8}   {'B_ISPG':>14}  {'B_GR':>14}  {'ΔB (ISPG-GR)':>14}")
    for (L, _, Bi, _), (_, _, Bg, _) in zip(rows_i, rows_g):
        print(f"{L:>8.1f}   {Bi:>14.5f}  {Bg:>14.5f}  {Bi-Bg:>14.6f}")

    print("\nEach individual B_2PN grows approximately linearly with L — this")
    print("is the UNIVERSAL path-length contribution (1/2) * integral(x'^2) dz,")
    print("with asymptotic x' → ±2rg/b, which gives a + 4 rg^2 L / b^2 term")
    print("modulo trajectory details.  This piece is IDENTICAL in ISPG and GR")
    print("because the 1PN trajectory is the same (both have n ≈ 1 + 2rg/r).")
    print("The distinctive ISPG 2PN signature is ΔB = B_ISPG - B_GR = π/4,")
    print("L-independent, and equal to the straight-line refractive difference.")


if __name__ == "__main__":
    main()
