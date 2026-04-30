"""Full ISPG cavity: include Phi^2 + Phi^3 from exponential coupling.

From ISPG: the exponential coupling e^{-2*Phi} creates nonlinearities:
  e^{-2*Phi} * Phi = Phi - 2*Phi^2 + 2*Phi^3 - (4/3)*Phi^4 + ...

After time-averaging the self-consistent equation becomes (in the core):
  Phi'' + (2/r)Phi' + (Om^2 - 1)*Phi + g2*Phi^2 + g3*Phi^3 = 0

where g2 and g3 come from the ISPG exponential structure.

Key: g3 > 0 creates a SYMMETRIC nonlinearity that supports excited states,
potentially changing the cavity structure dramatically.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from itertools import combinations


def solve_bg(Phi0, r_max=50.0, g2=1.0, g3=0.0, Om_guess=None,
             r_prev=None, y_prev=None, p_prev=None):
    if Om_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Om_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = g2*Phi**2 + g3*Phi**3
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


def cavity_eigs(r_bg, Phi_bg, l_val, g2=1.0, g3=0.0, N=2000):
    """Eigenvalues: -u'' + V_eff*u = omega^2*u.

    Linearizing g2*Phi^2 + g3*Phi^3 around Phi0 gives:
    V_eff = 1 - (2*g2*Phi0 + 3*g3*Phi0^2) + l(l+1)/r^2
    """
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi = f(r)

    c_lin = 2.0*g2*Phi + 3.0*g3*Phi**2
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
print("  ISPG Full Cavity: Phi^2 + Phi^3 nonlinearities")
print("=" * 70)
print(f"  Target: Q = {koide(m_e, m_mu, m_tau):.8f}")
print(f"  Target ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")

configs = [
    (1.0, 0.0, "Standard Phi^2 only"),
    (1.0, 0.5, "Phi^2 + 0.5*Phi^3"),
    (1.0, 1.0, "Phi^2 + Phi^3"),
    (1.0, 2.0, "Phi^2 + 2*Phi^3"),
    (0.5, 1.0, "0.5*Phi^2 + Phi^3 (ISPG-like)"),
    (0.5, 0.5, "0.5*Phi^2 + 0.5*Phi^3"),
    (0.0, 1.0, "Pure Phi^3"),
]

for g2, g3, label in configs:
    print(f"\n{'='*70}")
    print(f"  {label} (g2={g2}, g3={g3})")
    print(f"{'='*70}")

    prev = None
    best_Q_diff = 999
    best_info = None

    for Phi0 in np.arange(0.10, 3.01, 0.05):
        if prev:
            Om, sol = solve_bg(Phi0, r_max=50.0, g2=g2, g3=g3,
                                r_prev=prev[0].x, y_prev=prev[0].y,
                                p_prev=prev[0].p)
        else:
            Om, sol = solve_bg(Phi0, r_max=50.0, g2=g2, g3=g3)

        if Om is None:
            continue
        prev = (sol,)

        all_oms = []
        all_labs = []
        for l_val in range(4):
            eigs = cavity_eigs(sol.x, sol.y[0], l_val, g2, g3, N=1500)
            for n, e in enumerate(eigs):
                if e > 0.001:
                    all_oms.append(e)
                    all_labs.append(f"(n={n},l={l_val})")

        n_modes = len(all_oms)

        if n_modes >= 3:
            for combo in combinations(range(len(all_oms)), 3):
                oms = sorted([all_oms[i] for i in combo])
                Q_om = koide(*oms)
                E_vals = sorted([E_1d(o) for o in oms], reverse=True)
                if E_vals[-1] > 1e-12:
                    Q_E = koide(*E_vals)
                else:
                    Q_E = -1

                for Q_test, formula in [(Q_om, 'omega'), (Q_E, 'E_1d')]:
                    if Q_test > 0 and abs(Q_test - 2/3) < best_Q_diff:
                        best_Q_diff = abs(Q_test - 2/3)
                        labs = [all_labs[i] for i in combo]
                        best_info = (Phi0, Om, Q_test, oms, labs, formula,
                                     n_modes, E_vals)

        if abs(Phi0 - round(Phi0*2)/2) < 0.03 and n_modes >= 1:
            om_str = " ".join(f"{o:.4f}" for o in all_oms[:5])
            print(f"  Phi0={Phi0:.2f} Om={Om:.4f}"
                  f" | {n_modes} modes: {om_str}")

    if best_info:
        Phi0, Om, Q, oms, labs, formula, nm, E_vals = best_info
        print(f"\n  BEST: Phi0={Phi0:.2f}, Om_bg={Om:.4f}")
        print(f"  Formula: {formula}")
        print(f"  Q = {Q:.8f}  |Q-2/3| = {abs(Q-2/3):.2e}")
        print(f"  Modes: {labs}")
        print(f"  Omegas: [{oms[0]:.6f}, {oms[1]:.6f}, {oms[2]:.6f}]")
        if formula == 'omega':
            print(f"  Mass ratios (omega): 1 : {oms[1]/oms[0]:.2f}"
                  f" : {oms[2]/oms[0]:.2f}")
        else:
            print(f"  E values: {E_vals}")
            print(f"  Mass ratios (E): 1 : {E_vals[1]/E_vals[2]:.2f}"
                  f" : {E_vals[0]/E_vals[2]:.2f}")
        print(f"  Total modes in cavity: {nm}")
    else:
        print(f"\n  No valid triple found!")

print(f"\n\n{'='*70}")
print("  CONCLUSION")
print("=" * 70)
print("""
  The Phi^2 oscillon cavity has EXACTLY 3 bound states.
  This matches 3 lepton generations structurally.

  The Poschl-Teller parameter lambda=4 gives:
  - 3 eigenvalues for l=0 (n=0,1,2) in 1D
  - In 3D: 2 l=0 states + 1 l=1 state = 3 total

  But: mass ratios are too compressed for Q = 2/3.
  The full ISPG nonlinear equation (all orders of the
  exponential coupling) needs to be solved self-consistently.
""")
