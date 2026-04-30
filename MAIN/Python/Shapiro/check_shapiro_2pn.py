"""Research framework for the full ISPG 2PN Shapiro-delay coefficient.

STATUS (2026-04-18): The full 2PN coefficient in the bi-conformal ISPG
metric is OPEN.  See MAIN/6. ISPG_Shapiro.tex Sec. "Bent-ray refractive
and path-length contributions (in progress)".

What IS proven: the refractive (straight-line Born) sub-piece is exactly
  Delta t_refr = (r_g^2 / (c*b)) * (2 pi)
and is verified numerically in tools/check_shapiro_born.py.

What remains OPEN: the bent-ray correction to the refractive integral
and the geometric path-length excess.  Both require a finite-endpoint
boundary-value formulation (source at r_1, observer at r_2) with a
consistent impact-parameter convention (asymptotic Hamilton-Jacobi b
versus geometric closest approach), since individual 2PN pieces are
infra-red divergent and only their properly regularized sum is finite.

This script performs initial-value integration of the exact Fermat ODE
for the null ray, with symmetric BC at z=0 (closest approach x=b, xp=0).
Running it naively yields a 2PN coefficient B that is LINEARLY DIVERGENT
in the integration cutoff z_max, which is consistent with the known
infra-red issue.  The script is kept as a starting point for a future
finite-endpoint BVP (shooting method) implementation; it is NOT a
verification of any total 2PN coefficient.

Run:  python check_shapiro_2pn.py  (diagnostic only).
"""
import math
import numpy as np
from scipy.integrate import solve_ivp, quad


def phi(r, rg):
    """ISPG scalar: phi = -2 r_g / r."""
    return -2.0 * rg / r


def metric_null_dt_per_dl(r, rg):
    """dt/dl along null ray in bi-conformal metric:
    0 = -e^phi dt^2 + e^(-phi) dl^2  =>  dt = e^(-phi) dl
    (We set c = 1.)
    """
    return math.exp(-phi(r, rg))


def integrate_straight_delay(rg, b, zmax):
    """Delta t along the STRAIGHT line x = b, parameterized by z in [-zmax, zmax].
    This is the 'flat space at impact parameter b' baseline: what an
    observer would see if no gravity (just geometric path).
    """
    # Straight line: dl = dz, r(z) = sqrt(b^2 + z^2).
    def integrand(z):
        r = math.sqrt(b * b + z * z)
        return metric_null_dt_per_dl(r, rg)

    val, _ = quad(integrand, -zmax, zmax, limit=400)
    return val  # this is total coordinate time for traversal; baseline straight_no_gravity = 2*zmax


def integrate_bent_ray(rg, b, zmax, rtol=1e-10, atol=1e-12):
    """Integrate the null ray equation in the Cartesian (x, z) plane with x
    transverse, z along the asymptotic direction.  Ray is planar, moves in +z,
    and is minimally deflected by the scalar-bi-conformal metric.

    We use Hamilton's equations derived from the null condition.  Since the
    bi-conformal metric is spatially conformally flat, null geodesics follow
    Fermat's principle with n(r) = e^{-phi(r)} = e^{+2*rg/r} (index of
    refraction), and the ray equation is:
        d^2 x/d z^2 = (1/n) * dn/dx  (with appropriate param)
    but the exact Fermat ODE from ds^2 = 0 is:
        d/dz [n (dx/dz)/sqrt(1 + (dx/dz)^2)] = dn/dx * sqrt(1+(dx/dz)^2).
    We integrate this exactly (no expansion), then compute Delta t = integral
    of (e^{-phi} - 1) * dl along the bent ray plus path excess.
    """
    # State: (x, xp) where xp = dx/dz. Since the ray travels in +z direction.
    def rhs(z, y):
        x, xp = y
        r = math.sqrt(x * x + z * z)
        if r < 1e-12:
            return [xp, 0.0]
        # n = exp(-phi) = exp(+2*rg/r)
        # ln n = -phi = 2*rg/r
        # d(ln n)/dx = -phi_x = (2*rg) * (-x/r^3) ... wait phi = -2rg/r so dphi/dx = +2rg*x/r^3
        # => d(ln n)/dx = -(dphi/dx) = -2*rg*x/r^3
        # Fermat ODE in form (d/dz)[n xp / sqrt(1+xp^2)] = dn/dx * sqrt(1+xp^2):
        # This simplifies under expansion to: xp' = [(1+xp^2)/n] * (dn/dx - xp*dn/dz)
        # where dn/dx = n * d(ln n)/dx, dn/dz = n * d(ln n)/dz
        phi_x = 2.0 * rg * x / r**3  # dphi/dx
        phi_z = 2.0 * rg * z / r**3  # dphi/dz
        dlnn_dx = -phi_x
        dlnn_dz = -phi_z
        # Fermat: for a general n(x,z), parameterizing by z:
        # d^2x/dz^2 = (1 + (dx/dz)^2) * [dlnn/dx - (dx/dz) * dlnn/dz]
        xpp = (1.0 + xp * xp) * (dlnn_dx - xp * dlnn_dz)
        return [xp, xpp]

    # SYMMETRIC boundary conditions: closest approach at z=0, x=b, xp=0.
    # Integrate forward to +zmax and backward to -zmax, then splice.  This
    # enforces symmetric asymptotes (xp -> +alpha/2 at +zmax, -alpha/2 at -zmax),
    # so the bent-ray endpoints x(-zmax) and x(+zmax) are equal by symmetry, and
    # the spurious "asymmetric-asymptote" O(r_g^2 zmax/b^2) path contribution
    # vanishes.
    y0 = [b, 0.0]
    sol_fwd = solve_ivp(rhs, (0.0, zmax), y0, rtol=rtol, atol=atol,
                        dense_output=True, max_step=zmax / 200.0)
    sol_bwd = solve_ivp(rhs, (0.0, -zmax), y0, rtol=rtol, atol=atol,
                        dense_output=True, max_step=zmax / 200.0)
    if not (sol_fwd.success and sol_bwd.success):
        raise RuntimeError("ODE failed")
    return sol_fwd, sol_bwd


def delay_along_bent_ray(sols, rg, zmax, b, n_samp=4000):
    """Compute Delta t = T_bent - |AB|_flat with SYMMETRIC endpoints.
    sols = (sol_fwd on [0, zmax], sol_bwd on [0, -zmax]).
    Bent ray is spliced at z=0.  By symmetry x(-zmax) = x(+zmax), so the
    spurious O(r_g^2 zmax/b^2) asymptote-mismatch contribution cancels.
    """
    sol_fwd, sol_bwd = sols
    zs_fwd = np.linspace(0.0, zmax, n_samp)
    zs_bwd = np.linspace(0.0, -zmax, n_samp)
    y_fwd = sol_fwd.sol(zs_fwd)
    y_bwd = sol_bwd.sol(zs_bwd)

    def accumulate(zs, ys):
        xs = ys[0]
        xps = ys[1]
        rs = np.sqrt(xs * xs + zs * zs)
        dl_per_dz = np.sqrt(1.0 + xps * xps)
        integrand = np.exp(2.0 * rg / rs) * dl_per_dz
        return np.trapz(integrand, zs), xs[-1]

    t_fwd, x_end_fwd = accumulate(zs_fwd, y_fwd)
    t_bwd, x_end_bwd = accumulate(zs_bwd, y_bwd)  # zs_bwd goes 0 -> -zmax; trapz gives negative value
    total_coord_time = t_fwd + (-t_bwd)  # flip sign for the backward branch
    # Endpoints: (-zmax, x_end_bwd) and (+zmax, x_end_fwd); by symmetry, x_end_bwd = x_end_fwd
    flat_distance = math.sqrt((2.0 * zmax) ** 2 + (x_end_fwd - x_end_bwd) ** 2)
    return total_coord_time - flat_distance


def shapiro_delay_ispg(rg, b, zmax, **kw):
    """Full ISPG Shapiro excess delay over flat vacuum, for the given parameters."""
    sols = integrate_bent_ray(rg, b, zmax, **kw)
    return delay_along_bent_ray(sols, rg, zmax, b)


def extract_2pn_coeff(rg_values, b, zmax):
    """For several rg/b, compute Delta t = A (rg) + B (rg)^2 + ...,
    and extract B by finite differences.  Actually we compute
      Delta t / rg  vs  rg  ->  intercept is A-coefficient, slope is B.
    But the 1PN A coefficient depends logarithmically on zmax, so we compare
    at FIXED zmax.  Then:
      Delta t (rg) / rg - Delta t (rg_ref) / rg_ref  ->  B * (rg - rg_ref).
    Equivalently, fit Delta t (rg) = A_log * rg + B * rg^2 + ... with 2 data points.
    """
    dts = []
    for rg in rg_values:
        dt = shapiro_delay_ispg(rg, b, zmax)
        dts.append(dt)
    # fit Delta t = A rg + B rg^2 + O(rg^3)
    # with 2 unknowns (A, B) and len(rg_values) >= 2 equations.
    # Use 3 smallest values and lstsq.
    M = np.array([[rg, rg * rg] for rg in rg_values])
    sol, *_ = np.linalg.lstsq(M, np.array(dts), rcond=None)
    A, B = sol
    # The 2PN coefficient B (dimensionless since c = b = 1) is NOT a
    # verification target here — see module docstring.  In a naive IVP
    # integration with a z_max cutoff, B is linearly divergent in z_max
    # (infra-red issue), reflecting the need for a finite-endpoint BVP.
    B_coeff = B
    return A, B_coeff, dts


def main():
    b = 1.0
    # use small rg values to stay well in weak-field (rg/b << 1)
    rg_values = [1e-3, 3e-3, 1e-2, 3e-2]
    zmax_values = [50.0, 100.0, 200.0]
    print("ISPG 2PN Shapiro — diagnostic for the naive absolute coefficient.")
    print("See tools/check_shapiro_born.py for the proven refractive")
    print("sub-piece B_refr = 2*pi =", 2.0 * math.pi)
    print("The ISPG-GR differential coefficient is closed in Appendix 6: Delta B = pi/4.")
    print()
    for zmax in zmax_values:
        A, B, dts = extract_2pn_coeff(rg_values, b, zmax)
        print(f"zmax = {zmax}:  1PN log-A = {A:.4f}   2PN B (cutoff-dep.) = {B:.6f}")
        print(f"             Delta t samples = {[f'{d:.6e}' for d in dts]}")
    # B grows linearly in zmax here: this IS the infra-red signature, not
    # a bug.  A standalone absolute coefficient requires a finite-endpoint
    # convention.  The ISPG-GR difference is closed analytically in
    # MAIN/6. ISPG_Shapiro.tex (sec:diff).


if __name__ == "__main__":
    main()
