"""Born (straight-line) verification of the ISPG 2PN refractive piece.

For the bi-conformal ISPG metric with n = e^{+2 r_g / r} (refractive index),
the STRAIGHT-LINE refractive integral is

    Delta t_straight(Z, r_g, b) = int_{-Z}^{+Z} [n(r_0) - 1] dz,    r_0 = sqrt(b^2 + z^2).

Expand:   n - 1 = 2 r_g/r + 2 (r_g/r)^2 + (4/3)(r_g/r)^3 + ...

  O(r_g^1):  2 r_g * [2 arcsinh(Z/b)]        -> 4 r_g ln(2 Z/b)  as Z -> infty   (1PN log)
  O(r_g^2):  2 r_g^2 * [2 arctan(Z/b)/b]     -> 2 pi r_g^2 / b   as Z -> infty   (refractive 2PN)

So on the straight line, the 2PN refractive coefficient is EXACTLY 2*pi
(when c = b = 1).  This is the "refractive" sub-piece of the full 2PN
Shapiro delay in the bi-conformal ISPG metric (MAIN/6. ISPG_Shapiro.tex
Sec. "Refractive-source term", eq:refr2pi).

Numerical result (Z=2000 b): B -> 6.2811 ~= 2 pi. CONFIRMED.

The remaining bent-ray and path-length 2PN contributions are flagged as
open in Appendix 6 (Sec. "Bent-ray refractive and path-length
contributions") and are not tested here.  The full 2PN coefficient is
pending future work.

Run:  python check_shapiro_born.py
"""
import math
import numpy as np
from scipy.integrate import quad


def straight_born(rg, b, Z):
    def integrand(z):
        r = math.sqrt(b * b + z * z)
        return math.exp(2.0 * rg / r) - 1.0
    val, _ = quad(integrand, -Z, Z, limit=400)
    return val


def extract_coeffs(rg_values, b, Z):
    """Fit Delta t = A*rg + B*rg^2 + C*rg^3 at fixed Z."""
    dts = [straight_born(rg, b, Z) for rg in rg_values]
    M = np.array([[rg, rg ** 2, rg ** 3] for rg in rg_values])
    sol, *_ = np.linalg.lstsq(M, np.array(dts), rcond=None)
    return sol, dts


def main():
    b = 1.0
    rg_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    Zs = [50.0, 100.0, 500.0, 2000.0]
    print("Straight-line Born 2PN metric coefficient — must -> 2*pi =", 2 * math.pi)
    print()
    for Z in Zs:
        (A, B, C), _ = extract_coeffs(rg_values, b, Z)
        print(f"Z = {Z:7.0f}:  A (1PN) = {A:.4f}   B (2PN straight) = {B:.6f}")
        print(f"             expected A = 4*ln(2Z/b) = {4 * math.log(2 * Z / b):.4f}")
        print(f"             expected B = 2*pi = {2 * math.pi:.6f}")


if __name__ == "__main__":
    main()
