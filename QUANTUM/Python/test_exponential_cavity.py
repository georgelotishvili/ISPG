"""Full ISPG exponential cavity: e^{-2*Phi} coupling.

The ISPG bi-conformal metric creates exponential coupling:
  e^{-2*Phi} * Phi = Phi - 2*Phi^2 + 2*Phi^3 - (4/3)*Phi^4 + ...

Background oscillon: Phi'' + (2/r)Phi' + (Om^2 - 1)Phi + F_NL(Phi) = 0
where F_NL(Phi) = Om^2 * [Phi*exp(-2*Phi) - Phi] / (-2)
     = Om^2 * sum_{n>=2} (-2)^{n-1}/n! * Phi^n

After time-averaging of cos^n(Om*t), the effective coefficients change.
For the secular (cos(Om*t)) term:
  cos^2 -> 1/2, cos^3 -> 3/4, cos^4 -> 3/8, cos^5 -> 5/16, ...

We test multiple truncation levels of the exponential.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from itertools import combinations


def F_exp_full(Phi, Om2):
    """Full exponential nonlinearity: Om^2 * Phi * (e^{-2*Phi} - 1) / (-2)
    = Om^2 * [Phi - Phi*e^{-2*Phi}] / 2
    For the oscillon equation with the form:
    Phi'' + (2/r)Phi' + (Om^2 - 1)*Phi + F_NL(Phi, Om) = 0

    The nonlinear part from the exponential coupling (time-averaged):
    F_NL = Phi * [1 - exp(-Phi)] (simplified effective form)
    """
    return Phi * (1.0 - np.exp(-Phi))


def dF_exp_full(Phi):
    """Derivative of F_NL for linearization: d/dPhi [Phi*(1 - exp(-Phi))]"""
    return (1.0 - np.exp(-Phi)) + Phi * np.exp(-Phi)


def solve_bg_exp(Phi0, r_max=50.0, Om_guess=None, nl_func=None,
                 r_prev=None, y_prev=None, p_prev=None):
    """Solve background with arbitrary nonlinearity."""
    if Om_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Om_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = nl_func(Phi)
        d2 = -(2.0/r_safe)*dPhi - (Om**2 - 1)*Phi - NL
        d2_0 = -(Om**2 - 1)*Phi/3.0 - NL/3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None:
        N_pts = max(400, len(r_prev))
        r = np.linspace(1e-6, r_max, N_pts)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r)*sc, f1(r)*sc])
        Om_guess = p_prev[0] if p_prev is not None else Om_guess
    else:
        r = np.linspace(1e-6, r_max, 400)
        kappa_g = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kappa_g)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None


def cavity_eigs_gen(r_bg, Phi_bg, l_val, dF_func, N=2000):
    """Eigenvalues with general linearized potential.

    V_eff = 1 - dF(Phi0(r)) + l(l+1)/r^2
    """
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi = f(r)

    c_lin = dF_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2

    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')

    n_eig = min(20, N-2)
    evals, _ = eigsh(H, k=n_eig, which='SM')
    bound = evals[evals < 1.0]
    return np.sqrt(np.maximum(np.sort(bound), 0))


def E_1d(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*Om**2 + 1)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86

print("=" * 70)
print("  Full Exponential ISPG Cavity")
print("  F_NL(Phi) = Phi*(1 - exp(-alpha*Phi))")
print("  alpha controls the 'exponential depth'")
print("=" * 70)
print(f"  Target Q = {koide(m_e, m_mu, m_tau):.8f}")
print(f"  Target ratios = 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}\n")

for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
    print(f"\n{'='*70}")
    print(f"  alpha = {alpha}")
    print(f"{'='*70}")

    nl_func = lambda Phi, a=alpha: Phi * (1.0 - np.exp(-a * Phi))
    dF_func = lambda Phi, a=alpha: (1.0 - np.exp(-a*Phi)) + a*Phi*np.exp(-a*Phi)

    prev = None
    best_Q_diff = 999
    best_info = None

    for Phi0 in np.arange(0.05, 5.01, 0.05):
        if prev:
            Om, sol = solve_bg_exp(Phi0, r_max=50.0, nl_func=nl_func,
                                    r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
        else:
            Om, sol = solve_bg_exp(Phi0, r_max=50.0, nl_func=nl_func)

        if Om is None:
            if prev is None:
                continue
            else:
                break
        prev = sol

        all_oms = []
        all_labs = []
        for l_val in range(4):
            eigs = cavity_eigs_gen(sol.x, sol.y[0], l_val, dF_func, N=1500)
            for n, e in enumerate(eigs):
                if e > 0.001:
                    all_oms.append(e)
                    all_labs.append(f"(n={n},l={l_val})")

        n_modes = len(all_oms)
        if n_modes >= 3:
            for combo in combinations(range(len(all_oms)), 3):
                oms = sorted([all_oms[i] for i in combo])
                E_vals = sorted([E_1d(o) for o in oms], reverse=True)
                if E_vals[-1] > 1e-15:
                    Q_E = koide(*E_vals)
                    if abs(Q_E - 2/3) < best_Q_diff:
                        best_Q_diff = abs(Q_E - 2/3)
                        labs = [all_labs[i] for i in combo]
                        best_info = (Phi0, Om, Q_E, oms, labs, E_vals, n_modes)

        if abs(Phi0 - round(Phi0*2)/2) < 0.03 and n_modes >= 1:
            om_str = " ".join(f"{o:.4f}" for o in all_oms[:5])
            print(f"  Phi0={Phi0:5.2f} Om={Om:.4f} | {n_modes} modes: {om_str}")

    if best_info:
        Phi0, Om, Q, oms, labs, E_vals, nm = best_info
        r0 = min(E_vals)
        print(f"\n  BEST: Phi0={Phi0:.2f}, Om={Om:.4f}")
        print(f"  Q(E_1d) = {Q:.8f}  |Q-2/3| = {abs(Q-2/3):.2e}")
        print(f"  Modes: {labs}")
        print(f"  Omegas: [{oms[0]:.6f}, {oms[1]:.6f}, {oms[2]:.6f}]")
        print(f"  E_1d: [{E_vals[0]:.6e}, {E_vals[1]:.6e}, {E_vals[2]:.6e}]")
        print(f"  Mass ratios: 1 : {E_vals[1]/r0:.1f} : {E_vals[0]/r0:.1f}")
        print(f"  N_modes = {nm}")
    else:
        print(f"\n  No valid triple found!")

print(f"\n\n{'='*70}")
print("  SUMMARY: Q(E_1d) vs alpha")
print("=" * 70)
